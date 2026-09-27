"""绩效指标：年化收益、波动、Sharpe、Sortino、最大回撤、Calmar、
胜率、盈亏比、平均持仓天数、交易次数、换手率。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarize_trades(position: pd.Series, net_returns: pd.Series) -> dict:
    """从持仓序列提取逐笔交易统计（一段连续非零持仓算一笔）。

    Returns dict: n_trades, win_rate, profit_factor, avg_holding_days,
                  avg_trade_pnl
    """
    pos = position.fillna(0).values
    ret = net_returns.fillna(0).values
    pnls, holds = [], []
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1
            continue
        j = i
        wealth = 1.0
        while j < n and pos[j] != 0:
            wealth *= 1.0 + ret[j]
            j += 1
        pnls.append(wealth - 1.0)
        holds.append(j - i)
        i = j

    pnls = np.array(pnls)
    if len(pnls) == 0:
        return {"n_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                "avg_holding_days": np.nan, "avg_trade_pnl": np.nan}
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "n_trades": int(len(pnls)),
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": float(pf),
        "avg_holding_days": float(np.mean(holds)),
        "avg_trade_pnl": float(pnls.mean()),
    }


def compute_metrics(returns: pd.Series, position: pd.Series | None = None,
                    turnover: pd.Series | None = None) -> dict:
    """计算绩效指标。

    Parameters
    ----------
    returns  : 日收益序列（净或毛）
    position : 持仓序列（可选，提供时输出逐笔交易统计）
    turnover : 换手序列（可选，提供时输出年化换手）

    Returns
    -------
    dict: ann_return, ann_vol, sharpe, sortino, max_drawdown, calmar,
          win_rate, profit_factor, avg_holding_days, n_trades,
          annual_turnover, total_return
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        raise ValueError("收益序列太短，无法计算指标")

    total = float((1 + r).prod() - 1)
    ann_ret = float((1 + total) ** (TRADING_DAYS / len(r)) - 1)
    ann_vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0]
    dd_vol = float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else np.nan
    sortino = ann_ret / dd_vol if dd_vol and dd_vol > 0 else np.nan

    eq = (1 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1).min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    m = {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
    }

    if position is not None:
        m.update(summarize_trades(position.reindex(r.index), r))
    else:
        m.update({"n_trades": np.nan, "win_rate": float((r > 0).mean()),
                  "profit_factor": np.nan, "avg_holding_days": np.nan,
                  "avg_trade_pnl": np.nan})

    if turnover is not None:
        t = pd.Series(turnover).reindex(r.index).fillna(0)
        m["annual_turnover"] = float(t.sum() * TRADING_DAYS / len(r))
    else:
        m["annual_turnover"] = np.nan

    return m
