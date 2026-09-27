"""回测模块：次日执行、交易成本（佣金/滑点/借券费）、流动性过滤。

执行假设：信号在 T 日收盘确认，T+1 日执行（position.shift(1)），
用 T+1 收盘价近似次日开盘价成交。开仓后两腿权重在该笔交易内固定
（静态对冲，不因 beta 漂移每日再平衡，避免高估换手）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.metrics import compute_metrics  # noqa: F401  (re-export, 规范要求的接口)

logger = logging.getLogger(__name__)


def liquidity_report(prices: pd.DataFrame, volumes: pd.DataFrame,
                     window: int = 20) -> pd.DataFrame:
    """日均成交额报告：rolling window 平均 price * volume（美元）。"""
    dollar_vol = (prices * volumes).rolling(window, min_periods=window).mean()
    return dollar_vol


def liquidity_ok(prices: pd.DataFrame, volumes: pd.DataFrame,
                 min_dollar_volume: float = 1e7, window: int = 20) -> bool:
    """检查两腿在样本期内的流动性中位数是否均超过阈值。"""
    dv = liquidity_report(prices, volumes, window)
    med = dv.median()
    ok = bool((med > min_dollar_volume).all())
    for t in med.index:
        logger.info("流动性: %s 20日均成交额中位数 $%.1fM (阈值 $%.1fM)",
                    t, med[t] / 1e6, min_dollar_volume / 1e6)
    if not ok:
        logger.warning("流动性过滤未通过，该对不交易")
    return ok


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame,
                 cost_bps: float = 5, borrow_bps: float = 100,
                 slippage_bps: float = 5, beta: pd.Series | None = None
                 ) -> pd.DataFrame:
    """运行回测。

    Parameters
    ----------
    prices   : 两列 DataFrame [y, x]（调整后价格）
    signals  : 含 position 列（T 日收盘确认，T+1 执行）
    cost_bps : 单边佣金 bps
    borrow_bps : 借券费（年化 bps，按日对空仓腿计提）
    slippage_bps : 单边滑点 bps
    beta     : 对冲比序列（T 日值用于 T+1 执行时的权重，内部 shift(1)）

    Returns
    -------
    pd.DataFrame: position, w_y, w_x, gross_return, cost, borrow_fee,
                  net_return, equity, drawdown, turnover
    """
    y_col, x_col = prices.columns[0], prices.columns[1]
    px = prices[[y_col, x_col]].astype(float)
    rets = px.pct_change().fillna(0.0)

    sig = signals.reindex(px.index)
    pos_exec = sig["position"].shift(1).fillna(0.0)   # 次日执行
    beta_exec = None
    if beta is not None:
        beta_exec = pd.Series(beta).reindex(px.index).shift(1)  # 用信号日的 beta

    n = len(px)
    w_y = np.zeros(n)
    w_x = np.zeros(n)
    turnover = np.zeros(n)

    cur = 0.0
    wy = wx = 0.0
    last_beta = 1.0
    for i in range(n):
        p = pos_exec.iloc[i]
        if p != cur:
            if p == 0.0:
                new_wy, new_wx = 0.0, 0.0
            else:
                b = beta_exec.iloc[i] if beta_exec is not None else np.nan
                if np.isnan(b):
                    b = last_beta
                b = float(np.clip(b, 0.05, 20.0))
                last_beta = b
                # 美元中性：总敞口 = 1，w_y = p/(1+b), w_x = -p*b/(1+b)
                new_wy = p / (1.0 + b)
                new_wx = -p * b / (1.0 + b)
            turnover[i] = abs(new_wy - wy) + abs(new_wx - wx)
            wy, wx = new_wy, new_wx
            cur = p
        w_y[i], w_x[i] = wy, wx

    w_y_s = pd.Series(w_y, index=px.index)
    w_x_s = pd.Series(w_x, index=px.index)
    turnover_s = pd.Series(turnover, index=px.index)

    gross = w_y_s * rets[y_col] + w_x_s * rets[x_col]

    trade_cost = turnover_s * (cost_bps + slippage_bps) / 1e4
    short_notional = w_y_s.clip(upper=0).abs() + w_x_s.clip(upper=0).abs()
    borrow_fee = short_notional * (borrow_bps / 1e4) / 252.0

    net = gross - trade_cost - borrow_fee
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    out = pd.DataFrame({
        "position": pos_exec,
        "w_y": w_y_s,
        "w_x": w_x_s,
        "gross_return": gross,
        "cost": trade_cost,
        "borrow_fee": borrow_fee,
        "net_return": net,
        "equity": equity,
        "drawdown": drawdown,
        "turnover": turnover_s,
    })
    n_trades = int((turnover_s > 0).sum() // 2)
    logger.info("run_backtest: %d 天, 约 %d 笔交易, 总换手 %.1f, 成本合计 %.4f",
                n, n_trades, turnover_s.sum(), (trade_cost + borrow_fee).sum())
    return out
