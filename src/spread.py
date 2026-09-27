"""价差与 Z-Score：对数价差、滚动标准化、OU 半衰期。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_spread(y: pd.Series, x: pd.Series, beta) -> pd.Series:
    """对数价差: spread_t = log(y_t) - beta_t * log(x_t)。

    beta 可以是标量（静态，仅诊断用）或与 y 对齐的 pd.Series（滚动/Kalman）。
    """
    ly = np.log(y.astype(float))
    lx = np.log(x.astype(float))
    if np.isscalar(beta):
        spread = ly - beta * lx
    else:
        b = pd.Series(beta).reindex(ly.index)
        spread = ly - b * lx
    spread.name = "spread"
    return spread


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """滚动 Z-Score: (s_t - mean_{t-window+1..t}) / std_{t-window+1..t}。

    只用历史窗口（含当日），信号次日执行，无前视。
    """
    m = spread.rolling(window, min_periods=window).mean()
    s = spread.rolling(window, min_periods=window).std(ddof=0)
    z = (spread - m) / s.replace(0.0, np.nan)
    z.name = "zscore"
    return z


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck 半衰期（交易日）。

    对 AR(1) 离散形式  ds_t = c + lambda * s_{t-1} + e_t  做 OLS，
    半衰期 = -ln(2) / lambda。lambda >= 0（无均值回归）或样本不足时返回 inf。
    """
    s = pd.Series(spread).dropna().astype(float)
    if len(s) < 60:
        return np.inf
    lag = s.shift(1)
    ds = s.diff()
    df = pd.concat([ds, lag], axis=1).dropna()
    lam = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
    if lam >= 0:
        return np.inf
    return float(-np.log(2.0) / lam)


def rolling_half_life(spread: pd.Series, window: int = 250) -> pd.Series:
    """滚动 OU 半衰期（P1）：每个时点只用过去 window 天价差。

    用于动态时间止损与 regime 监测。窗口内 lambda >= 0 时记为 NaN
    （该窗口无均值回归证据，区别于 inf 的"永不回归"语义）。
    """
    s = pd.Series(spread).astype(float)
    ds = s.diff()
    out = np.full(len(s), np.nan)
    sv, dv = s.values, ds.values
    for t in range(window, len(s)):
        lag_w = sv[t - window:t - 1]
        ds_w = dv[t - window + 1:t]
        mask = ~(np.isnan(lag_w) | np.isnan(ds_w))
        if mask.sum() < 60:
            continue
        lam = np.polyfit(lag_w[mask], ds_w[mask], 1)[0]
        if lam < 0:
            out[t] = -np.log(2.0) / lam
    return pd.Series(out, index=s.index, name="rolling_half_life")
