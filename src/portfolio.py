"""多对组合（P0）：滚动协整门控（P1）+ 逆波动率风险平价。

设计：
- 宇宙由经济先验固定（7 个同行业/同驱动组），不靠全样本 p 值选队，
  避免 selection 层面的 data snooping；
- 每对每天的"是否可交易"由滚动协整门控决定：过去 gate_window 天内
  残差 ADF p < gate_p 才允许持有仓位（只用历史数据，无前视）；
- 时间止损用滚动半衰期（开仓日的估计值）；
- 组合层面对各对策略收益做逆波动率加权（trailing vol_window 天），
  归一到总敞口 1。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.beta import rolling_ols_beta
from src.pair_selection import rolling_cointegration
from src.spread import compute_spread, compute_zscore, rolling_half_life
from src.signal import generate_signals, apply_time_stop
from src.backtest import run_backtest

logger = logging.getLogger(__name__)


def prepare_pair(px: pd.DataFrame, beta_window: int, z_window: int,
                 gate_window: int, entry: float, exit_: float, stop: float,
                 time_stop_halflives: float = 2.0) -> dict:
    """计算单对的全部交易组件（与门控阈值无关的部分，可复用）。

    Returns dict: beta, coint_p（滚动协整 p 值）, roll_half_life,
                  raw_sig（未门控、已时间止损的信号）
    """
    y, x = px.iloc[:, 0], px.iloc[:, 1]
    beta = rolling_ols_beta(y, x, beta_window)
    spread = compute_spread(y, x, beta)
    z = compute_zscore(spread, z_window)
    coint = rolling_cointegration(y, x, gate_window)
    roll_hl = rolling_half_life(spread, gate_window)
    sig = generate_signals(z, entry=entry, exit=exit_, stop=stop)
    sig = apply_time_stop(sig, roll_hl, max_halflives=time_stop_halflives)
    return {"beta": beta, "coint_p": coint["resid_adf_pvalue"],
            "roll_half_life": roll_hl, "raw_sig": sig}


def run_pair_gated(px: pd.DataFrame, entry: float, exit_: float, stop: float,
                   beta_window: int, z_window: int, gate_window: int,
                   gate_p: float, cost_bps: float, slippage_bps: float,
                   borrow_bps: float, time_stop_halflives: float = 2.0
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """单对带滚动协整门控的回测。

    Returns (backtest_df, diagnostics_df, signals_df)：
    diagnostics 含 rolling beta / 滚动协整 p 值 / 滚动半衰期 / gate；
    signals 为执行前信号（供成本敏感性重复使用）。
    """
    comp = prepare_pair(px, beta_window, z_window, gate_window,
                        entry, exit_, stop, time_stop_halflives)
    gate = (comp["coint_p"] < gate_p).astype(float)
    sig = comp["raw_sig"].copy()
    sig["position"] = sig["position"] * gate.reindex(sig.index).fillna(0)

    bt = run_backtest(px, sig, cost_bps=cost_bps, borrow_bps=borrow_bps,
                      slippage_bps=slippage_bps, beta=comp["beta"])
    diag = pd.DataFrame({"beta": comp["beta"], "roll_coint_p": comp["coint_p"],
                         "roll_half_life": comp["roll_half_life"], "gate": gate})
    logger.info("run_pair_gated %s/%s: 门控开放 %.0f%% 天数",
                px.columns[0], px.columns[1], gate.mean() * 100)
    return bt, diag, sig


def risk_parity_portfolio(pair_returns: pd.DataFrame,
                          vol_window: int = 60) -> pd.DataFrame:
    """逆波动率风险平价组合。

    权重 w_i,t ∝ 1/vol_i,t-1（trailing vol_window 天收益波动），
    仅对最近有实际持仓的对分配权重（避免把资金分给长期空仓的对），
    归一使 sum(w) = 1。收益滞后一天实现，无前视。

    Returns DataFrame[weight_{pair}, port_return, equity, drawdown]
    """
    rets = pair_returns.fillna(0.0)
    vol = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    active = (pair_returns.abs().rolling(vol_window, min_periods=1).sum() > 0)
    inv_vol = (1.0 / vol.replace(0, np.nan)) * active
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    w = w.shift(1).fillna(0.0)          # 权重用昨日信息，今日生效

    port_ret = (w * rets).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    dd = equity / equity.cummax() - 1
    out = w.add_prefix("w_")
    out["port_return"] = port_ret
    out["equity"] = equity
    out["drawdown"] = dd
    logger.info("组合: %d 对, 平均活跃 %.1f 对", rets.shape[1],
                float((w > 0).sum(axis=1).mean()))
    return out


def pair_correlation(pair_returns: pd.DataFrame) -> pd.DataFrame:
    """各对策略净收益的相关性矩阵（组合分散度诊断）。"""
    return pair_returns.corr()
