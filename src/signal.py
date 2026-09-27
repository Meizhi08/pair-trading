"""信号模块：Z-Score 阈值状态机 + 时间止损。

约定 position:
    +1 = 做多价差 = 做多 y / 做空 x（z < -entry 时开仓）
    -1 = 做空价差 = 做空 y / 做多 x（z > +entry 时开仓）
     0 = 空仓

信号在收盘确认，次日执行（backtest 中 position.shift(1)）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_signals(zscore: pd.Series, entry: float = 2.0, exit: float = 0.5,
                     stop: float = 3.5) -> pd.DataFrame:
    """由 Z-Score 生成持仓信号（状态机）。

    规则：
      z > +entry      -> 做空价差 (position = -1)
      z < -entry      -> 做多价差 (position = +1)
      |z| < exit      -> 平仓
      |z| > stop      -> 止损平仓，并在 |z| 回落到 exit 以内之前禁止再开仓（冷却）
      z 为 NaN        -> 维持现状（滚动窗口热身期不交易）

    Returns DataFrame[position, days_in_trade, entry_flag, exit_flag]
    （均为收盘确认值，执行由回测模块后移一日）。
    """
    z = zscore.values
    n = len(z)
    pos = np.zeros(n)
    days = np.zeros(n)
    entry_flag = np.zeros(n, dtype=bool)
    exit_flag = np.zeros(n, dtype=bool)

    state = 0
    entry_day = -1
    blocked = False  # 止损冷却

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = state
            days[i] = (i - entry_day) if state != 0 else 0
            continue

        if state == 0:
            if abs(zi) < exit:
                blocked = False
            if not blocked:
                if zi > entry:
                    state, entry_day = -1, i
                    entry_flag[i] = True
                elif zi < -entry:
                    state, entry_day = 1, i
                    entry_flag[i] = True
        else:
            if abs(zi) < exit:
                state = 0
                exit_flag[i] = True
            elif abs(zi) > stop:
                state = 0
                blocked = True
                exit_flag[i] = True

        pos[i] = state
        days[i] = (i - entry_day) if state != 0 else 0

    out = pd.DataFrame({
        "position": pos,
        "days_in_trade": days,
        "entry": entry_flag,
        "exit": exit_flag,
    }, index=zscore.index)
    logger.info("generate_signals: entry=%.1f exit=%.1f stop=%.1f, 开仓 %d 次",
                entry, exit, stop, int(entry_flag.sum()))
    return out


def apply_time_stop(signals: pd.DataFrame, half_life_days,
                    max_halflives: float = 2.0) -> pd.DataFrame:
    """时间止损：持仓超过 max_halflives * half_life 个交易日未回归则强制平仓。

    half_life_days 可以是标量（样本内冻结值）或与 signals 对齐的 pd.Series
    （滚动半衰期，P1）；为序列时取开仓日的半衰期作为该笔交易的上限。
    平仓后直到状态机自然回到空仓并再次开仓前保持空仓（防止同一笔交易
    被时间止损后立即重新进入）。
    """
    out = signals.copy()
    hl_series = None
    if isinstance(half_life_days, pd.Series):
        hl_series = half_life_days.reindex(out.index)
        hl_scalar = np.nan
    else:
        hl_scalar = float(half_life_days)
    if hl_series is None and (not np.isfinite(hl_scalar) or hl_scalar <= 0):
        logger.warning("半衰期无效 (%s)，跳过时间止损", half_life_days)
        out["time_stopped"] = False
        return out

    pos = out["position"].values.copy()
    days = out["days_in_trade"].values
    time_stopped = np.zeros(len(out), dtype=bool)

    forced_flat = False
    max_days = np.inf
    for i in range(len(out)):
        if forced_flat:
            if pos[i] == 0:
                forced_flat = False
            else:
                pos[i] = 0
                time_stopped[i] = True
                continue
        if pos[i] != 0 and days[i] == 0:
            # 新开仓：确定该笔交易的时间止损上限
            hl_i = hl_series.iloc[i] if hl_series is not None else hl_scalar
            max_days = (int(np.ceil(max_halflives * hl_i))
                        if np.isfinite(hl_i) and hl_i > 0 else np.inf)
        if pos[i] != 0 and days[i] > max_days:
            pos[i] = 0
            time_stopped[i] = True
            forced_flat = True

    out["position"] = pos
    out["time_stopped"] = time_stopped
    n_ts = int(np.sum(np.diff(time_stopped.astype(int)) == 1)) or int(time_stopped.any())
    if hl_series is None:
        logger.info("apply_time_stop: 半衰期=%.1f 天, 上限=%s 天, 强平段 %d 次",
                    hl_scalar,
                    int(np.ceil(max_halflives * hl_scalar))
                    if np.isfinite(hl_scalar) else "inf", n_ts)
    else:
        logger.info("apply_time_stop: 滚动半衰期(中位 %.1f 天), 强平段 %d 次",
                    float(np.nanmedian(hl_series)), n_ts)
    return out
