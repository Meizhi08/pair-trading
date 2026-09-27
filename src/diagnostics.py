"""深度诊断模块（复评第 2 轮）：regime 分析、门控敏感性、留一法、
随机权重/随机信号基线、因子归因、成本分解、CUSUM 断点日期、
Kalman 诊断、统计功效分析。

全部为"策略是否有真实 edge"提供可证伪的量化证据。
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.metrics import compute_metrics
from src.portfolio import prepare_pair, risk_parity_portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 问题1: regime

def regime_analysis(returns: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """按 VIX 水平（三分位）与宏观阶段分样本计算绩效。

    回答"IS/OOS 倒挂是否 regime 效应"：如果策略只在高 VIX 赚钱，
    则 2019–2024 的正收益是疫情+加息的时代特征，不可外推。
    """
    df = pd.concat([returns.rename("ret"), vix.rename("vix")], axis=1).dropna()
    q1, q2 = df["vix"].quantile([1 / 3, 2 / 3])
    rows = {}
    for lab, mask in [("low VIX", df["vix"] <= q1),
                      ("mid VIX", (df["vix"] > q1) & (df["vix"] <= q2)),
                      ("high VIX", df["vix"] > q2)]:
        m = compute_metrics(df.loc[mask, "ret"])
        rows[lab] = {"sharpe": m["sharpe"], "ann_return": m["ann_return"],
                     "n_days": int(mask.sum())}
    for lab, (a, b) in {"QE/low-vol 2012–2019": ("2012-01-01", "2019-12-31"),
                        "COVID 2020": ("2020-01-01", "2020-12-31"),
                        "hiking 2022–2023": ("2022-01-01", "2023-12-31"),
                        "post-hike 2024": ("2024-01-01", "2024-12-31")}.items():
        sub = df.loc[a:b, "ret"]
        if len(sub) > 20:
            m = compute_metrics(sub)
            rows[lab] = {"sharpe": m["sharpe"], "ann_return": m["ann_return"],
                         "n_days": len(sub)}
    out = pd.DataFrame(rows).T
    logger.info("regime 分析:\n%s", out.round(3).to_string())
    return out


# ---------------------------------------------------------------- 问题2: 门控敏感性

def gate_sensitivity(groups_data: dict, windows, pvalues, cfg,
                     split_date) -> pd.DataFrame:
    """门控窗口 × 门控阈值的网格：篮子 OOS Sharpe + 平均门控开放率。

    prepare_pair 的结果按 (pair, window) 缓存，阈值变化只改掩码，
    避免重复滚动协整计算。
    """
    bt_cfg = cfg["backtest"]
    sig_cfg = cfg["signal"]
    port_cfg = cfg["validation"]["portfolio"]
    cache = {}
    for w in windows:
        for name, px in groups_data.items():
            cache[(w, name)] = prepare_pair(
                px, beta_window=cfg["beta"]["window"],
                z_window=cfg["zscore"]["window"], gate_window=w,
                entry=sig_cfg["entry"], exit_=sig_cfg["exit"], stop=sig_cfg["stop"],
                time_stop_halflives=sig_cfg["time_stop_halflives"])

    rows = []
    for w, p in itertools.product(windows, pvalues):
        pair_net, opens = {}, []
        for name, px in groups_data.items():
            comp = cache[(w, name)]
            gate = (comp["coint_p"] < p).astype(float)
            opens.append(float(gate.mean()))
            sig = comp["raw_sig"].copy()
            sig["position"] = sig["position"] * gate.reindex(sig.index).fillna(0)
            bt = run_backtest(px, sig, cost_bps=bt_cfg["cost_bps"],
                              slippage_bps=bt_cfg["slippage_bps"],
                              borrow_bps=bt_cfg["borrow_bps"], beta=comp["beta"])
            pair_net[name] = bt["net_return"]
        port = risk_parity_portfolio(pd.DataFrame(pair_net),
                                     vol_window=port_cfg["vol_window"])
        for seg, mask in [("IS", port.index < pd.Timestamp(split_date)),
                          ("OOS", port.index >= pd.Timestamp(split_date))]:
            m = compute_metrics(port.loc[mask, "port_return"])
            rows.append({"gate_window": w, "gate_p": p, "segment": seg,
                         "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                         "gate_open": float(np.mean(opens))})
    out = pd.DataFrame(rows)
    logger.info("门控敏感性:\n%s",
                out.round(3).to_string())
    return out


# ---------------------------------------------------------------- 问题3: 组合构成

def leave_one_out(pair_net: pd.DataFrame, vol_window: int,
                  split_date) -> pd.Series:
    """留一法：逐个剔除一对，看篮子 OOS Sharpe 变化。

    若剔除某对后 Sharpe 大幅上升，说明该对在拖累组合；
    若大幅下降，说明组合结果依赖单一成员。
    """
    split = pd.Timestamp(split_date)

    def oos_sharpe(rets):
        port = risk_parity_portfolio(rets, vol_window)
        return compute_metrics(port.loc[port.index >= split, "port_return"])["sharpe"]

    out = {"full": oos_sharpe(pair_net)}
    for c in pair_net.columns:
        out[f"-{c}"] = oos_sharpe(pair_net.drop(columns=c))
    s = pd.Series(out)
    logger.info("留一法 OOS Sharpe:\n%s", s.round(3).to_string())
    return s


def random_weights_sharpe(pair_net: pd.DataFrame, split_date,
                          actual_sharpe: float, n_sims: int = 1000,
                          seed: int = 42) -> dict:
    """随机静态权重基线：Dirichlet 随机权重 n_sims 次（OOS 段）。

    检验风险平价权重是否比"随便配"更好；若 actual 处于随机分布中部，
    说明组合表现与权重选择无关（纯分散数学效应）。
    """
    R = pair_net.loc[pair_net.index >= pd.Timestamp(split_date)].fillna(0.0)
    rng = np.random.default_rng(seed)
    Rv = R.values
    sharpes = np.empty(n_sims)
    for i in range(n_sims):
        w = rng.dirichlet(np.ones(Rv.shape[1]))
        sharpes[i] = compute_metrics(pd.Series(Rv @ w, index=R.index))["sharpe"]
    return {"random_sharpe_q05": float(np.percentile(sharpes, 5)),
            "random_sharpe_median": float(np.median(sharpes)),
            "random_sharpe_q95": float(np.percentile(sharpes, 95)),
            "actual_sharpe": actual_sharpe,
            "actual_percentile": float((sharpes < actual_sharpe).mean()),
            "_dist": sharpes}


def random_signal_baseline(px: pd.DataFrame, exposure: float,
                           avg_holding_days: float, cost_bps: float,
                           slippage_bps: float, borrow_bps: float,
                           n_sims: int = 1000, seed: int = 42) -> np.ndarray:
    """随机信号基线：与真实策略相同的暴露比例与平均持仓长度，
    但开平仓时机与方向完全随机。返回 n_sims 个全样本净 Sharpe。
    """
    r = px.pct_change().fillna(0.0).values
    n = len(r)
    p_close = 1.0 / max(avg_holding_days, 1.0)
    p_open = p_close * exposure / max(1e-9, 1.0 - exposure)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_sims)
    for s in range(n_sims):
        u_open = rng.random(n)
        u_close = rng.random(n)
        direction = rng.choice([-1.0, 1.0], n)
        pos = np.zeros(n)
        state = 0.0
        for i in range(1, n):
            if state == 0.0:
                if u_open[i] < p_open:
                    state = direction[i]
            elif u_close[i] < p_close:
                state = 0.0
            pos[i] = state
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        gross = pos * (r[:, 0] - r[:, 1]) / 2.0
        net = gross - turn * (cost_bps + slippage_bps) / 1e4 \
            - np.abs(pos) * 0.5 * borrow_bps / 1e4 / 252.0
        sharpes[s] = compute_metrics(pd.Series(net, index=px.index))["sharpe"]
    return sharpes


# ---------------------------------------------------------------- 问题9: 归因

def factor_attribution(returns: pd.Series, bench_ret: pd.Series) -> dict:
    """对市场基准（SPY）的 OLS 归因：alpha（年化）、beta、R²、相关系数。

    市场中性策略应满足 beta ≈ 0、R² ≈ 0；若不满足，说明收益里有市场暴露。
    """
    df = pd.concat([returns.rename("s"), bench_ret.rename("m")], axis=1).dropna()
    b, a = np.polyfit(df["m"], df["s"], 1)
    pred = a + b * df["m"]
    r2 = 1 - ((df["s"] - pred) ** 2).sum() / ((df["s"] - df["s"].mean()) ** 2).sum()
    out = {"alpha_ann": float(a * 252), "beta": float(b), "r2": float(r2),
           "corr": float(df["s"].corr(df["m"]))}
    logger.info("因子归因: alpha=%.2f%%, beta=%.3f, R2=%.4f, corr=%.3f",
                out["alpha_ann"] * 100, out["beta"], out["r2"], out["corr"])
    return out


def cost_decomposition(bt: pd.DataFrame, commission_bps: float,
                       slippage_bps: float) -> dict:
    """成本分解：佣金、滑点、借券费各占多少（对毛 PnL 的占比）。

    bt["cost"] 是佣金+滑点的合计，按费率比例拆分。
    """
    trade_cost = float(bt["cost"].sum())
    borrow = float(bt["borrow_fee"].sum())
    gross = float(bt["gross_return"].sum())
    net = float(bt["net_return"].sum())
    total_rate = commission_bps + slippage_bps
    out = {
        "gross_pnl": gross,
        "commission": trade_cost * commission_bps / total_rate,
        "slippage": trade_cost * slippage_bps / total_rate,
        "borrow_fee": borrow,
        "total_cost": trade_cost + borrow,
        "net_pnl": net,
        "cost_share_of_gross": (trade_cost + borrow) / gross if gross > 0 else np.nan,
    }
    logger.info("成本分解: 毛 %.4f, 佣金 %.4f, 滑点 %.4f, 借券 %.4f, 净 %.4f",
                gross, out["commission"], out["slippage"], borrow, net)
    return out


# ---------------------------------------------------------------- 问题7: 断点日期

def cusum_break_date(spread: pd.Series) -> dict:
    """CUSUM 递归残差首次越界日期（结构断点的时间定位）。

    手工实现 Brown-Durbin-Evans (1975) CUSUM（对常数回归）：
      递归残差 w_t = (y_t - mean(y_{1..t-1})) / sqrt(1 + 1/(t-1))
      W_t = cumsum(w) / s_hat
      5% 边界：连接 (0, ±a√T) 与 (T, ±3a√T)，a = 0.948

    Returns dict: break_date (str|None), rcusum/lower/upper/index（供画图）
    """
    s = pd.Series(spread).dropna().astype(float)
    y = s.values
    n = len(y)
    if n < 60:
        return {"break_date": None, "rcusum": np.array([]), "lower": np.array([]),
                "upper": np.array([]), "index": s.index}

    csum = np.cumsum(y)
    mean_prev = np.concatenate([[y[0]], csum[:-1] / np.arange(1, n)])  # mean(y_1..t-1)
    t_arr = np.arange(1, n + 1)
    w = (y - mean_prev) / np.sqrt(1.0 + 1.0 / np.maximum(t_arr - 1, 1))
    s_hat = w[1:].std(ddof=1)
    W = np.cumsum(w) / s_hat

    a = 0.948
    bound = a * np.sqrt(n) * (1.0 + 2.0 * t_arr / n)
    idx = s.index
    cross = np.where(np.abs(W) > bound)[0]
    break_date = idx[cross[0]] if len(cross) else None
    logger.info("CUSUM 断点日期: %s", break_date)
    return {"break_date": str(break_date.date()) if break_date is not None else None,
            "rcusum": W, "lower": -bound, "upper": bound, "index": idx}


# ---------------------------------------------------------------- 问题4: Kalman 诊断

def kalman_delta_grid(px: pd.DataFrame, deltas, cfg, split_date) -> pd.DataFrame:
    """Kalman 过程噪声 delta 网格：OOS 净 Sharpe 对 delta 的敏感性。"""
    from src.beta import kalman_beta
    from src.spread import compute_spread, compute_zscore, half_life
    from src.signal import generate_signals, apply_time_stop

    y, x = px.iloc[:, 0], px.iloc[:, 1]
    sig_cfg, bt_cfg = cfg["signal"], cfg["backtest"]
    split = pd.Timestamp(split_date)
    rows = []
    for d in deltas:
        beta = kalman_beta(y, x, delta=d, n_burn=cfg["beta"]["kalman_burn_in"])
        spread = compute_spread(y, x, beta)
        z = compute_zscore(spread, cfg["zscore"]["window"])
        hl = half_life(spread.loc[spread.index < split])
        sig = generate_signals(z, entry=sig_cfg["entry"], exit=sig_cfg["exit"],
                               stop=sig_cfg["stop"])
        sig = apply_time_stop(sig, hl, max_halflives=sig_cfg["time_stop_halflives"])
        bt = run_backtest(px, sig, cost_bps=bt_cfg["cost_bps"],
                          slippage_bps=bt_cfg["slippage_bps"],
                          borrow_bps=bt_cfg["borrow_bps"], beta=beta)
        oos = bt.loc[bt.index >= split]
        m = compute_metrics(oos["net_return"])
        rows.append({"delta": d, "oos_sharpe": m["sharpe"],
                     "oos_ann_return": m["ann_return"], "is_half_life": hl,
                     "beta_final": float(beta.iloc[-1])})
    out = pd.DataFrame(rows)
    logger.info("Kalman delta 网格:\n%s", out.round(4).to_string())
    return out


def kalman_residual_whiteness(y: pd.Series, x: pd.Series, delta: float,
                              n_burn: int) -> dict:
    """Kalman 价差的 Ljung-Box 白噪声检验（滞后 10/20）。

    若残差显著自相关，说明状态空间设定漏掉了结构（delta 太小）。
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from src.beta import kalman_beta

    b = kalman_beta(y, x, delta=delta, n_burn=n_burn)
    resid = (np.log(y.astype(float)) - b * np.log(x.astype(float))).dropna()
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    return {"lb_pvalue_lag10": float(lb["lb_pvalue"].iloc[0]),
            "lb_pvalue_lag20": float(lb["lb_pvalue"].iloc[1])}


# ---------------------------------------------------------------- 问题6: 功效分析

def power_analysis(target_sharpes=(0.3, 0.5, 1.0), alpha: float = 0.05,
                   power: float = 0.8) -> dict:
    """检测给定年化 Sharpe 所需的样本年数（近似，Lo 2002 修正）。

    SE(SR_ann) ≈ sqrt((1 + SR²/2) / T)，拒绝 H0: SR=0 需要
    T = (z_{1-α/2} + z_{power})² (1 + SR²/2) / SR²  年。
    """
    from scipy.stats import norm

    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return {f"SR={sr}": float((za + zb) ** 2 * (1 + sr ** 2 / 2) / sr ** 2)
            for sr in target_sharpes}
