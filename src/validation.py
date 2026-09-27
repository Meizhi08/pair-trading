"""验证模块：时间序列样本内/外切分、Walk-Forward、参数敏感性。

铁律：任何参数选择、选队、Beta 估计只允许使用决策时点之前的数据。
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from src.beta import rolling_ols_beta
from src.spread import compute_spread, compute_zscore, half_life
from src.signal import generate_signals, apply_time_stop
from src.backtest import run_backtest
from src.metrics import compute_metrics

logger = logging.getLogger(__name__)


def block_bootstrap_ci(returns: pd.Series, n_boot: int = 2000, block: int = 20,
                       seed: int = 42) -> dict:
    """循环块 bootstrap（circular block bootstrap）估计绩效置信区间。

    日收益有自相关，不能逐日 iid bootstrap；取长度 block 的连续块重采样。
    Returns dict: sharpe_ci (5%, 95%), sharpe_p_gt0 (自助 Sharpe>0 占比),
                  maxdd_ci, ann_return_ci, n_boot, block
    """
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < block * 5:
        raise ValueError(f"序列太短 ({n})，无法块 bootstrap")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes, maxdds, annrets = [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sample = r[idx][:n]
        m = compute_metrics(pd.Series(sample))
        sharpes.append(m["sharpe"])
        maxdds.append(m["max_drawdown"])
        annrets.append(m["ann_return"])
    sharpes, maxdds, annrets = map(np.array, (sharpes, maxdds, annrets))
    out = {
        "sharpe_ci": (float(np.percentile(sharpes, 5)), float(np.percentile(sharpes, 95))),
        "sharpe_p_gt0": float((sharpes > 0).mean()),
        "maxdd_ci": (float(np.percentile(maxdds, 5)), float(np.percentile(maxdds, 95))),
        "ann_return_ci": (float(np.percentile(annrets, 5)), float(np.percentile(annrets, 95))),
        "n_boot": n_boot,
        "block": block,
    }
    logger.info("bootstrap: Sharpe 90%% CI [%.2f, %.2f], P(Sharpe>0)=%.2f",
                out["sharpe_ci"][0], out["sharpe_ci"][1], out["sharpe_p_gt0"])
    return out


def cusum_test(spread: pd.Series) -> dict:
    """CUSUM 结构断点检测（P1）：检验价差均值回归关系是否稳定。

    对 spread_t = c + eps_t 做 OLS，递归残差 CUSUM 超出 5% 边界即拒绝稳定。
    Returns dict: stat, pvalue, break_detected
    """
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    s = pd.Series(spread).dropna().astype(float)
    resid = s - s.mean()          # 对常数回归的残差
    stat, p, _ = breaks_cusumolsresid(resid.values, ddof=1)
    logger.info("CUSUM: stat=%.3f, p=%.4f, 结构断点=%s", stat, p, p < 0.05)
    return {"stat": float(stat), "pvalue": float(p), "break_detected": bool(p < 0.05)}


def train_test_split_time(df: pd.DataFrame, split_date):
    """按日期切分：train = index < split_date, test = index >= split_date。"""
    split = pd.Timestamp(split_date)
    train = df.loc[df.index < split]
    test = df.loc[df.index >= split]
    logger.info("train/test 切分 @ %s: train %d 天 (%s..%s), test %d 天 (%s..%s)",
                split.date(), len(train),
                train.index.min().date(), train.index.max().date(),
                len(test), test.index.min().date(), test.index.max().date())
    return train, test


def walk_forward(prices: pd.DataFrame, start, end, window: int,
                 step_months: int = 6):
    """Walk-forward 折叠生成器。

    Parameters
    ----------
    prices : 全样本价格
    start  : 第一个样本外窗口的起点（之前的数据作为首个训练窗）
    end    : 样本外终点
    window : 训练窗口长度（交易日）
    step_months : 每次向前滚动的月数（默认 6，即每半年重新选队/调参）

    Yields (train_df, test_df)：train 为 test 起点之前最近 window 个交易日。
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    fold_start = start
    while fold_start < end:
        fold_end = fold_start + pd.DateOffset(months=step_months)
        test = prices.loc[(prices.index >= fold_start) & (prices.index < fold_end)]
        train = prices.loc[prices.index < fold_start].tail(window)
        if len(test) > 0 and len(train) >= 252:
            yield train, test
        fold_start = fold_end


def build_and_backtest(px: pd.DataFrame, entry: float, exit_: float, stop: float,
                       window: int, cost_bps: float, slippage_bps: float,
                       borrow_bps: float, time_stop_halflives: float = 2.0,
                       ref_spread: pd.Series | None = None) -> pd.DataFrame:
    """给定参数，构建 beta -> spread -> zscore -> signals -> backtest 全链条。

    ref_spread: 用于估计半衰期（时间止损）的参考价差，应为训练期价差，
                避免用未来数据决定持仓周期；为 None 时用本段价差估计。
    """
    y, x = px.iloc[:, 0], px.iloc[:, 1]
    beta = rolling_ols_beta(y, x, window)
    spread = compute_spread(y, x, beta)
    z = compute_zscore(spread, window)
    hl = half_life(ref_spread if ref_spread is not None else spread)
    sig = generate_signals(z, entry=entry, exit=exit_, stop=stop)
    sig = apply_time_stop(sig, hl, max_halflives=time_stop_halflives)
    bt = run_backtest(px, sig, cost_bps=cost_bps, borrow_bps=borrow_bps,
                      slippage_bps=slippage_bps, beta=beta)
    return bt


def parameter_sensitivity(prices: pd.DataFrame, param_grid: dict,
                          cost_bps: float = 5, slippage_bps: float = 5,
                          borrow_bps: float = 100, stop: float = 3.5
                          ) -> pd.DataFrame:
    """参数敏感性分析（调用方必须只传入样本内数据）。

    param_grid: {"entry": [...], "exit": [...], "window": [...]}
    对网格中每个组合完整跑 beta->spread->zscore->signal->backtest，
    返回按净 Sharpe 降序的 DataFrame。
    """
    keys = ["entry", "exit", "window"]
    grid = [dict(zip(keys, v)) for v in itertools.product(
        param_grid["entry"], param_grid["exit"], param_grid["window"])]

    rows = []
    for p in grid:
        try:
            bt = build_and_backtest(
                prices, entry=p["entry"], exit_=p["exit"], stop=stop,
                window=int(p["window"]), cost_bps=cost_bps,
                slippage_bps=slippage_bps, borrow_bps=borrow_bps)
            m = compute_metrics(bt["net_return"], bt["position"], bt["turnover"])
            rows.append({**p, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                         "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"],
                         "win_rate": m["win_rate"]})
        except Exception as exc:
            logger.warning("参数组合 %s 回测失败: %s", p, exc)
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    logger.info("参数敏感性: %d 个组合, Sharpe 区间 [%.2f, %.2f]",
                len(out), out["sharpe"].min(), out["sharpe"].max())
    return out
