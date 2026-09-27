"""动态对冲比 Beta：滚动 OLS 与 Kalman Filter。

严禁使用全样本 Beta。所有估计均为时点 t 处仅用 [t-window+1, t]（滚动）
或 [0, t]（Kalman 递推）的历史数据，信号次日执行，无前视偏差。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """滚动窗口 OLS 对冲比（对数价格）。

    beta_t = Cov(log x, log y) / Var(log x)，窗口 [t-window+1, t]，含当日。
    前 window-1 个点为 NaN（数据不足，不交易）。
    """
    ly = np.log(y.astype(float))
    lx = np.log(x.astype(float))
    cov = ly.rolling(window, min_periods=window).cov(lx)
    var = lx.rolling(window, min_periods=window).var()
    beta = cov / var
    beta.name = f"beta_{window}"
    return beta


def kalman_beta(y: pd.Series, x: pd.Series, delta: float = 1e-4,
                n_burn: int = 60) -> pd.Series:
    """Kalman Filter 时变对冲比（加分项）。

    状态空间模型（对数价格）：
        观测:  y_t = alpha_t + beta_t * x_t + v_t,  v_t ~ N(0, R)
        状态:  theta_t = theta_{t-1} + w_t,         w_t ~ N(0, Q), theta=(alpha, beta)

    - Q = delta * I（状态游走噪声，delta 越大 beta 变化越快）
    - R 用前 n_burn 个样本的 OLS 残差方差估计（只用历史前缀，无前视）
    返回 beta_t 序列（前几个点尚未收敛，配合 zscore 窗口自然剔除）。
    """
    df = pd.concat([y, x], axis=1).dropna().astype(float)
    ly = np.log(df.iloc[:, 0]).values
    lx = np.log(df.iloc[:, 1]).values
    n = len(df)
    if n < n_burn + 10:
        raise ValueError(f"样本太短 ({n})，无法估计 Kalman beta")

    burn = min(n_burn, n // 3)
    A = np.column_stack([np.ones(burn), lx[:burn]])
    coef, *_ = np.linalg.lstsq(A, ly[:burn], rcond=None)
    resid = ly[:burn] - A @ coef
    R = max(float(resid.var()), 1e-8)

    Q = np.eye(2) * delta
    theta = np.array([coef[0], coef[1]], dtype=float)
    P = np.eye(2)
    I = np.eye(2)
    betas = np.full(n, np.nan)

    for t in range(n):
        P = P + Q                                   # 状态预测
        H = np.array([1.0, lx[t]])
        e = ly[t] - H @ theta                       #  innovation
        S = H @ P @ H + R
        K = P @ H / S                               # Kalman 增益
        theta = theta + K * e
        P = (I - np.outer(K, H)) @ P
        betas[t] = theta[1]

    logger.info("kalman_beta: delta=%g, R=%.6f, 最终 beta=%.3f", delta, R, betas[-1])
    return pd.Series(betas, index=df.index, name="beta_kalman")
