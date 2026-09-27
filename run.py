#!/usr/bin/env python3
"""Pairs Trading 统计套利策略：一条命令跑完整流程。

用法:
    python run.py --config config.yaml

流程:
    1. 下载/清洗 14 只标的的调整后价格（2010–2024，缓存于 data/）
    2. 样本内 (2010–2018) Engle-Granger 协整筛选候选对
    3. 对最终选出的对: 滚动 Beta -> 对数价差 -> 滚动 Z-Score -> 信号 -> 回测
    4. 样本内 + 样本外 (2019–2024) 绩效（含佣金/滑点/借券费）
    5. 参数敏感性、Kalman Beta 对比、Walk-Forward 样本外验证
    6. 图表与指标写入 reports/
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data import load_prices, load_volumes, clean_prices, align_dates
from src.pair_selection import (find_candidate_pairs, cointegration_test,
                                fdr_bh, rolling_cointegration)
from src.beta import rolling_ols_beta, kalman_beta
from src.spread import compute_spread, compute_zscore, half_life, rolling_half_life
from src.signal import generate_signals, apply_time_stop
from src.backtest import run_backtest, liquidity_ok
from src.metrics import compute_metrics
from src.portfolio import run_pair_gated, risk_parity_portfolio, pair_correlation
from src.validation import (train_test_split_time, walk_forward,
                            parameter_sensitivity, build_and_backtest,
                            block_bootstrap_ci, cusum_test)
from src import diagnostics as diag

logger = logging.getLogger("run")


# ---------------------------------------------------------------- config / io

def parse_args():
    ap = argparse.ArgumentParser(description="Pairs trading research pipeline")
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def setup_logging(report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(report_dir / "run.log", mode="w")])


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("图表已保存: %s", path)


# ---------------------------------------------------------------- figures

def fig_prices(px, name, split_date, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    norm = px / px.iloc[0]
    ax.plot(norm.index, norm.iloc[:, 0], label=px.columns[0])
    ax.plot(norm.index, norm.iloc[:, 1], label=px.columns[1])
    ax.axvline(pd.Timestamp(split_date), color="k", ls="--", lw=1, label="IS/OOS split")
    ax.set_title(f"{name} — normalized adjusted prices")
    ax.legend()
    save_fig(fig, path)


def fig_beta(betas: dict, name, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, b in betas.items():
        ax.plot(b.index, b.values, label=label, lw=1)
    ax.set_title(f"{name} — dynamic hedge ratio (rolling OLS vs Kalman)")
    ax.legend()
    save_fig(fig, path)


def fig_spread(spread, name, split_date, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(spread.index, spread.values, lw=0.8)
    m = spread.rolling(60, min_periods=60).mean()
    ax.plot(m.index, m.values, color="orange", lw=1.2, label="60d mean")
    ax.axvline(pd.Timestamp(split_date), color="k", ls="--", lw=1)
    ax.set_title(f"{name} — log spread (rolling-beta hedged)")
    ax.legend()
    save_fig(fig, path)


def fig_zscore(z, signals, cfg_sig, name, split_date, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z.index, z.values, lw=0.8, label="z-score")
    for lvl, c, lab in [(cfg_sig["entry"], "r", "entry"), (-cfg_sig["entry"], "r", None),
                        (cfg_sig["exit"], "g", "exit"), (-cfg_sig["exit"], "g", None),
                        (cfg_sig["stop"], "m", "stop"), (-cfg_sig["stop"], "m", None)]:
        ax.axhline(lvl, color=c, ls="--", lw=0.8, label=lab)
    pos = signals["position"]
    in_trade = pos != 0
    ax.fill_between(z.index, z.min(), z.max(), where=in_trade.values,
                    alpha=0.15, color="steelblue", label="in position")
    ax.axvline(pd.Timestamp(split_date), color="k", ls="--", lw=1)
    ax.set_ylim(z.min() - 0.5, z.max() + 0.5)
    ax.set_title(f"{name} — rolling z-score and positions")
    ax.legend(ncol=4, fontsize=8)
    save_fig(fig, path)


def fig_equity(results: dict, name, split_date, path):
    """results: {'IS gross': bt_df, 'IS net': ..., 'OOS gross': ..., 'OOS net': ...}"""
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, bt in results.items():
        eq = bt["equity"] if "net" in label else (1 + bt["gross_return"]).cumprod()
        eq = eq / eq.iloc[0]  # 每段归一到 1，IS/OOS 可比较
        ls = "-" if "net" in label else "--"
        ax.plot(bt.index, eq.values, ls, label=label, lw=1.2)
    ax.axvline(pd.Timestamp(split_date), color="k", ls=":", lw=1)
    ax.set_title(f"{name} — equity curves (gross vs net, IS vs OOS)")
    ax.legend()
    save_fig(fig, path)


def fig_drawdown(results: dict, name, split_date, path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for label, bt in results.items():
        if "net" not in label:
            continue
        eq = bt["equity"] / bt["equity"].iloc[0]   # 段内归一后再算回撤
        dd = eq / eq.cummax() - 1.0
        ax.fill_between(bt.index, dd.values, 0, alpha=0.4, label=label)
    ax.axvline(pd.Timestamp(split_date), color="k", ls=":", lw=1)
    ax.set_title(f"{name} — drawdown (net)")
    ax.legend()
    save_fig(fig, path)


def fig_sensitivity(sens: pd.DataFrame, name, path):
    """entry × window 的净 Sharpe 热力图（exit 取中位档）。"""
    exit_vals = sorted(sens["exit"].unique())
    exit_mid = exit_vals[len(exit_vals) // 2]
    sub = sens[sens["exit"] == exit_mid]
    piv = sub.pivot_table(index="window", columns="entry", values="sharpe")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:.1f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(int(i)) for i in piv.index])
    ax.set_xlabel("entry threshold")
    ax.set_ylabel("rolling window (days)")
    ax.set_title(f"{name} — net Sharpe sensitivity (exit={exit_mid})")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    save_fig(fig, path)


def fig_walkforward(oos_ret: pd.Series, path):
    eq = (1 + oos_ret).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values, lw=1.2)
    ax.set_title("Walk-forward out-of-sample equity (net, re-selected & re-tuned every 6M)")
    save_fig(fig, path)


def fig_fdr_screening(scr: pd.DataFrame, path):
    """P0: 原始 p 值 vs FDR 校正后 p 值（排序）。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    s = scr.sort_values("eg_pvalue").reset_index(drop=True)
    ax.plot(-np.log10(s["eg_pvalue"]), label="raw p", lw=1)
    if "p_adj" in s:
        ax.plot(-np.log10(s["p_adj"]), label="FDR-adjusted p", lw=1)
    ax.axhline(-np.log10(0.05), color="r", ls="--", lw=1, label="p=0.05")
    ax.set_xlabel("pairs ranked by p-value")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Multiple-testing correction: raw vs FDR-adjusted p-values (selection window)")
    ax.legend()
    save_fig(fig, path)


def fig_rolling_gate(diags: dict, path):
    """P1: 7 对滚动协整 p 值 + 滚动半衰期网格图。"""
    n = len(diags)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.2 * n), sharex=True)
    for i, (name, d) in enumerate(diags.items()):
        axp, axh = axes[i]
        axp.plot(d.index, d["roll_coint_p"], lw=0.8)
        axp.axhline(0.10, color="r", ls="--", lw=0.8)
        axp.set_ylabel("ADF p", fontsize=8)
        axp.set_title(f"{name} — rolling cointegration (250d)", fontsize=9)
        axp.set_ylim(0, 1)
        axh.plot(d.index, d["roll_half_life"], lw=0.8, color="darkgreen")
        axh.set_ylabel("HL (days)", fontsize=8)
        axh.set_title(f"{name} — rolling half-life (250d)", fontsize=9)
        axh.set_ylim(0, 400)
    fig.tight_layout()
    save_fig(fig, path)


def fig_portfolio_equity(port: pd.DataFrame, singles: dict, split_date, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for seg, sub in [("IS", port.loc[port.index < pd.Timestamp(split_date)]),
                     ("OOS", port.loc[port.index >= pd.Timestamp(split_date)])]:
        eq = sub["equity"] / sub["equity"].iloc[0]
        ax.plot(eq.index, eq.values, lw=1.4, label=f"basket {seg} (net)")
    for name, bt in singles.items():
        sub = bt.loc[bt.index >= pd.Timestamp(split_date)]
        eq = sub["equity"] / sub["equity"].iloc[0]
        ax.plot(eq.index, eq.values, lw=1, ls="--", label=f"{name} OOS single (net)")
    ax.axvline(pd.Timestamp(split_date), color="k", ls=":", lw=1)
    ax.set_title("Gated multi-pair basket (risk parity) vs single pair")
    ax.legend(fontsize=8)
    save_fig(fig, path)


def fig_corr(corr: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Pair strategy return correlation (net, full sample)")
    fig.colorbar(im, ax=ax)
    save_fig(fig, path)


def fig_cost_sensitivity(df: pd.DataFrame, name, path):
    """净 Sharpe 随成本水平变化（borrow=100bps 切片 + 全网格散布）。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    df = df.copy()
    df["roundtrip_bps"] = 2 * (df["commission_bps"] + df["slippage_bps"])
    sub = df[df["borrow_bps"] == 100]
    piv = sub.pivot_table(index="commission_bps", columns="slippage_bps", values="sharpe")
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index])
    ax.set_xlabel("slippage (bps, per side)")
    ax.set_ylabel("commission (bps, per side)")
    ax.set_title(f"{name} — net Sharpe vs costs (borrow=100bps)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    save_fig(fig, path)


def fig_regime(reg: pd.DataFrame, title, path):
    """问题1：分 regime 的 Sharpe 条形图。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["steelblue"] * 3 + ["darkorange"] * (len(reg) - 3)
    ax.bar(reg.index, reg["sharpe"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Sharpe")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    save_fig(fig, path)


def fig_gate_sensitivity(gs: pd.DataFrame, path):
    """问题2：门控窗口 × 阈值 → OOS Sharpe 热力图。"""
    sub = gs[gs["segment"] == "OOS"]
    piv = sub.pivot_table(index="gate_window", columns="gate_p", values="sharpe")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("gate ADF p-value threshold")
    ax.set_ylabel("gate window (days)")
    ax.set_title("Gate sensitivity — basket OOS Sharpe")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    save_fig(fig, path)


def fig_loo(loo: pd.Series, path):
    """问题3：留一法 OOS Sharpe。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["black"] + ["steelblue"] * (len(loo) - 1)
    ax.bar(loo.index, loo.values, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(loo["full"], color="red", ls="--", lw=1,
               label=f"full = {loo['full']:.2f}")
    ax.set_ylabel("OOS Sharpe")
    ax.set_title("Leave-one-out: basket OOS Sharpe when dropping each pair")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    save_fig(fig, path)


def fig_random_baselines(rw: dict, rs: np.ndarray, actual_sig_sharpe: float, path):
    """问题3/8：随机权重与随机信号基线分布 vs 实际值。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(rw["_dist"], bins=40, color="steelblue", alpha=0.8)
    ax1.axvline(rw["actual_sharpe"], color="r", lw=1.5,
                label=f"risk parity = {rw['actual_sharpe']:.2f}\n"
                      f"(pct {rw['actual_percentile']:.2f})")
    ax1.set_title("Random Dirichlet weights (basket, OOS)")
    ax1.set_xlabel("Sharpe")
    ax1.legend(fontsize=8)
    ax2.hist(rs, bins=40, color="darkorange", alpha=0.8)
    ax2.axvline(actual_sig_sharpe, color="r", lw=1.5,
                label=f"actual strategy = {actual_sig_sharpe:.2f}\n"
                      f"(pct {(rs < actual_sig_sharpe).mean():.2f})")
    ax2.set_title("Random entry/exit timing (EWJ/EWG, full)")
    ax2.set_xlabel("Sharpe")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, path)


def fig_cusum(cb: dict, name, path):
    """问题7：CUSUM 递归残差与 5% 边界、断点日期。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cb["index"], cb["rcusum"], lw=1.2, label="CUSUM")
    ax.plot(cb["index"], cb["upper"], color="r", ls="--", lw=1, label="5% bounds")
    ax.plot(cb["index"], cb["lower"], color="r", ls="--", lw=1)
    if cb["break_date"]:
        ax.axvline(pd.Timestamp(cb["break_date"]), color="k", ls=":", lw=1,
                   label=f"break: {cb['break_date']}")
    ax.set_title(f"{name} — CUSUM structural stability test")
    ax.legend()
    save_fig(fig, path)


# ---------------------------------------------------------------- new phases

def run_two_stage_screen(prices, groups, cfg, figdir, report_dir):
    """P0: FDR 校正 + 两阶段筛选（选队窗 2010–2014 / 确认窗 2015–2018，OOS 不动）。

    第一层：选队窗内全部两两组合做 EG 检验，Benjamini-Hochberg FDR 校正；
    第二层：FDR 幸存者在确认窗重新检验（raw p < confirm_pvalue）。
    预期大概率无幸存者——这本身就是结论：严格统计下无可交易对。
    """
    ts = cfg["validation"]["two_stage"]
    select_end = pd.Timestamp(ts["select_end"])
    split = pd.Timestamp(cfg["validation"]["split_date"])
    px_sel = prices.loc[prices.index < select_end]
    px_cfm = prices.loc[(prices.index >= select_end) & (prices.index < split)]
    tickers = sorted({t for g in groups for t in g["tickers"]})
    econ = {frozenset(g["tickers"]): g["name"] for g in groups}

    logger.info("=== P0 两阶段筛选: 选队窗 ..%s, 确认窗 %s..%s ===",
                select_end.date(), select_end.date(), split.date())
    sel = pd.DataFrame(find_candidate_pairs(tickers, px_sel))
    fdr = fdr_bh(sel["eg_pvalue"].values, alpha=ts["fdr_alpha"])
    sel["p_adj"] = fdr["p_adj"].values
    sel["reject_fdr"] = fdr["reject_fdr"].values
    sel["econ_group"] = [econ.get(frozenset([r.ticker_y, r.ticker_x]), "")
                         for r in sel.itertuples()]
    sel.to_csv(report_dir / "screening_twostage_select.csv", index=False)
    fig_fdr_screening(sel, figdir / "fdr_screening.png")

    survivors = []
    for r in sel[sel["reject_fdr"]].itertuples():
        res = cointegration_test(px_cfm[r.ticker_y], px_cfm[r.ticker_x])
        ok = res["eg_pvalue"] < ts["confirm_pvalue"]
        survivors.append({"ticker_y": r.ticker_y, "ticker_x": r.ticker_x,
                          "select_p": r.eg_pvalue, "select_p_adj": r.p_adj,
                          "confirm_p": res["eg_pvalue"], "confirmed": ok})
    cfm_df = pd.DataFrame(survivors)
    if len(cfm_df):
        cfm_df.to_csv(report_dir / "screening_twostage_confirm.csv", index=False)
    n_confirmed = int(cfm_df["confirmed"].sum()) if len(cfm_df) else 0
    logger.info("FDR 结果: %d 组合, 最小 raw p=%.4f, 最小校正 p=%.4f, "
                "FDR 幸存 %d, 确认窗再确认 %d",
                len(sel), sel["eg_pvalue"].min(), sel["p_adj"].min(),
                int(sel["reject_fdr"].sum()), n_confirmed)
    return {"n_tests": len(sel), "min_p": float(sel["eg_pvalue"].min()),
            "min_p_adj": float(sel["p_adj"].min()),
            "n_fdr_pass": int(sel["reject_fdr"].sum()),
            "n_confirmed": n_confirmed}


def run_basket(prices, groups, cfg, split_date, figdir, report_dir):
    """P0 多对组合 + P1 滚动协整门控/滚动半衰期 + 风险平价。"""
    gate_cfg = cfg["validation"]["gate"]
    port_cfg = cfg["validation"]["portfolio"]
    bt_cfg = cfg["backtest"]
    sig_cfg = cfg["signal"]

    pair_net, singles, diags, cache = {}, {}, {}, {}
    for g in groups:
        ty, tx = g["tickers"]
        px = align_dates(prices[[ty]], prices[[tx]])
        bt, diag, sig = run_pair_gated(
            px, entry=sig_cfg["entry"], exit_=sig_cfg["exit"], stop=sig_cfg["stop"],
            beta_window=cfg["beta"]["window"], z_window=cfg["zscore"]["window"],
            gate_window=gate_cfg["window"], gate_p=gate_cfg["pvalue"],
            cost_bps=bt_cfg["cost_bps"], slippage_bps=bt_cfg["slippage_bps"],
            borrow_bps=bt_cfg["borrow_bps"],
            time_stop_halflives=sig_cfg["time_stop_halflives"])
        pair_net[g["name"]] = bt["net_return"]
        singles[g["name"]] = bt
        diags[g["name"]] = diag
        cache[g["name"]] = {"px": px, "sig": sig, "beta": diag["beta"]}

    rets = pd.DataFrame(pair_net)
    rets.to_csv(report_dir / "basket_pair_returns.csv")
    port = risk_parity_portfolio(rets, vol_window=port_cfg["vol_window"])
    port.to_csv(report_dir / "basket_portfolio.csv")

    corr = pair_correlation(rets)
    corr.to_csv(report_dir / "basket_correlation.csv")
    fig_corr(corr, figdir / "basket_correlation.png")
    fig_rolling_gate(diags, figdir / "rolling_gate.png")
    fig_portfolio_equity(port, {k: v for k, v in singles.items() if k == "EWJ/EWG"},
                         split_date, figdir / "basket_equity.png")

    rows = {}
    for seg, sl in [("IS", port.index < pd.Timestamp(split_date)),
                    ("OOS", port.index >= pd.Timestamp(split_date))]:
        rows[f"basket {seg}"] = compute_metrics(port.loc[sl, "port_return"])
    for name, bt in singles.items():
        oos = bt.loc[bt.index >= pd.Timestamp(split_date)]
        rows[f"{name} OOS"] = compute_metrics(oos["net_return"], oos["position"],
                                              oos["turnover"])
    table = pd.DataFrame(rows).T
    table.to_csv(report_dir / "basket_metrics.csv")
    logger.info("篮子绩效:\n%s",
                table[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].round(3).to_string())
    return {"metrics": table, "port": port, "singles": singles, "cache": cache,
            "gate_open_pct": {k: float(d["gate"].mean()) for k, d in diags.items()}}


def run_cost_sensitivity(basket, chosen_name, cfg, report_dir, figdir):
    """P0: 成本敏感性——佣金×滑点×借券费全网格，单对 + 篮子两个口径。

    信号只算一次（cache 复用），每个成本组合仅重跑 run_backtest 与组合加权。
    """
    cs = cfg["validation"]["cost_sensitivity"]
    port_cfg = cfg["validation"]["portfolio"]
    grid = list(itertools.product(cs["commission_bps"], cs["slippage_bps"],
                                  cs["borrow_bps"]))
    rows = []
    for cm, sl, bw in grid:
        # 单对（沿用 chosen 的信号与 beta）
        c0 = basket["cache"][chosen_name]
        bt1 = run_backtest(c0["px"], c0["sig"], cost_bps=cm, slippage_bps=sl,
                           borrow_bps=bw, beta=c0["beta"])
        m1 = compute_metrics(bt1["net_return"], bt1["position"], bt1["turnover"])
        # 篮子
        pair_net = {}
        for name, c in basket["cache"].items():
            bt_i = run_backtest(c["px"], c["sig"], cost_bps=cm, slippage_bps=sl,
                                borrow_bps=bw, beta=c["beta"])
            pair_net[name] = bt_i["net_return"]
        port = risk_parity_portfolio(pd.DataFrame(pair_net),
                                     vol_window=port_cfg["vol_window"])
        m2 = compute_metrics(port["port_return"])
        rows.append({"commission_bps": cm, "slippage_bps": sl, "borrow_bps": bw,
                     "single_sharpe": m1["sharpe"], "single_ann_return": m1["ann_return"],
                     "basket_sharpe": m2["sharpe"], "basket_ann_return": m2["ann_return"]})
    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "cost_sensitivity.csv", index=False)
    fig_cost_sensitivity(df.rename(columns={"single_sharpe": "sharpe"}),
                         f"single {chosen_name}", figdir / "cost_sensitivity_single.png")
    fig_cost_sensitivity(df.rename(columns={"basket_sharpe": "sharpe"}),
                         "basket", figdir / "cost_sensitivity_basket.png")
    base = df[(df["commission_bps"] == 5) & (df["slippage_bps"] == 5)
              & (df["borrow_bps"] == 100)]
    zero = df[(df["commission_bps"] == 0) & (df["slippage_bps"] == 0)
              & (df["borrow_bps"] == 0)]
    logger.info("成本敏感性: 单对 Sharpe 零成本 %.2f -> 基准成本 %.2f; "
                "篮子 %.2f -> %.2f",
                zero["single_sharpe"].iloc[0], base["single_sharpe"].iloc[0],
                zero["basket_sharpe"].iloc[0], base["basket_sharpe"].iloc[0])
    return {"zero_cost": zero.iloc[0].to_dict(), "base_cost": base.iloc[0].to_dict(),
            "basket_sharpe_min": float(df["basket_sharpe"].min()),
            "basket_sharpe_max": float(df["basket_sharpe"].max())}


def run_kalman_vs_ols(name, px, cfg, split_date, report_dir):
    """P1: Kalman beta vs 滚动 OLS beta 的同信号回测对比。"""
    y, x = px.iloc[:, 0], px.iloc[:, 1]
    sig_cfg, bt_cfg = cfg["signal"], cfg["backtest"]
    split = pd.Timestamp(split_date)
    rows = {}
    for method in ["ols_60d", "kalman"]:
        beta = (rolling_ols_beta(y, x, cfg["beta"]["window"]) if method == "ols_60d"
                else kalman_beta(y, x, delta=cfg["beta"]["kalman_delta"],
                                 n_burn=cfg["beta"]["kalman_burn_in"]))
        spread = compute_spread(y, x, beta)
        z = compute_zscore(spread, cfg["zscore"]["window"])
        hl = half_life(spread.loc[spread.index < split])
        sig = generate_signals(z, entry=sig_cfg["entry"], exit=sig_cfg["exit"],
                               stop=sig_cfg["stop"])
        sig = apply_time_stop(sig, hl, max_halflives=sig_cfg["time_stop_halflives"])
        bt = run_backtest(px, sig, cost_bps=bt_cfg["cost_bps"],
                          borrow_bps=bt_cfg["borrow_bps"],
                          slippage_bps=bt_cfg["slippage_bps"], beta=beta)
        for seg, sub in [("IS", bt.loc[bt.index < split]),
                         ("OOS", bt.loc[bt.index >= split])]:
            rows[f"{method} {seg}"] = compute_metrics(sub["net_return"],
                                                      sub["position"],
                                                      sub["turnover"])
    table = pd.DataFrame(rows).T
    table.to_csv(report_dir / f"kalman_vs_ols_{name.replace('/', '_')}.csv")
    logger.info("Kalman vs OLS (%s):\n%s", name,
                table[["ann_return", "sharpe", "max_drawdown", "n_trades"]]
                .round(3).to_string())
    return table


def run_diagnostics(prices, groups, cfg, split_date, basket, analyses,
                    chosen_name, figdir, report_dir):
    """复评第 2 轮的 8 项深度诊断：regime / 门控敏感性 / 留一法 / 随机基线 /
    因子归因 / 成本分解 / CUSUM 断点日期 / Kalman 诊断 / 功效分析 / SPY 基准。"""
    d_cfg = cfg["validation"]["diagnostics"]
    bt_cfg = cfg["backtest"]
    split = pd.Timestamp(split_date)
    out = {}

    # SPY / VIX（独立缓存，不进筛选宇宙）
    extra = clean_prices(load_prices(["SPY", "^VIX"], cfg["data"]["start"],
                                     cfg["data"]["end"], cfg["data"]["cache_dir"]),
                         ffill_limit=cfg["data"]["ffill_limit"])
    spy_ret = extra["SPY"].pct_change().fillna(0.0)
    vix = extra["^VIX"]

    groups_data = {name: c["px"] for name, c in basket["cache"].items()}
    pair_net = pd.DataFrame({name: bt["net_return"]
                             for name, bt in basket["singles"].items()})
    port_ret = basket["port"]["port_return"]

    # 问题1: regime 分样本（篮子全样本 + OOS）
    reg_full = diag.regime_analysis(port_ret, vix)
    reg_full.to_csv(report_dir / "diag_regime_basket_full.csv")
    fig_regime(reg_full, "Basket net Sharpe by regime (full sample)",
               figdir / "diag_regime_basket.png")
    out["regime_basket_full"] = reg_full.round(4).to_dict()

    # 问题2: 门控敏感性（窗口 × 阈值）
    gs = diag.gate_sensitivity(groups_data, d_cfg["gate_windows"],
                               d_cfg["gate_pvalues"], cfg, split_date)
    gs.to_csv(report_dir / "diag_gate_sensitivity.csv", index=False)
    fig_gate_sensitivity(gs, figdir / "diag_gate_sensitivity.png")
    out["gate_sensitivity"] = gs.round(4).to_dict()

    # 问题3: 留一法 + 随机权重
    loo = diag.leave_one_out(pair_net, cfg["validation"]["portfolio"]["vol_window"],
                             split_date)
    loo.to_csv(report_dir / "diag_leave_one_out.csv")
    fig_loo(loo, figdir / "diag_leave_one_out.png")
    out["leave_one_out"] = {k: round(v, 4) for k, v in loo.items()}

    basket_oos_sharpe = basket["metrics"].loc["basket OOS", "sharpe"]
    rw = diag.random_weights_sharpe(pair_net, split, basket_oos_sharpe,
                                    n_sims=d_cfg["random_sims"], seed=cfg["seed"])
    out["random_weights"] = {k: round(v, 4) for k, v in rw.items() if k != "_dist"}

    # 问题8: 随机信号基线（与 EWJ/EWG 相同的暴露与持仓长度）+ SPY 基准
    bt0 = analyses[chosen_name]["backtest"]
    exposure = float((bt0["position"] != 0).mean())
    avg_hold = float(compute_metrics(bt0["net_return"], bt0["position"])
                     ["avg_holding_days"])
    rs = diag.random_signal_baseline(basket["cache"][chosen_name]["px"],
                                     exposure, avg_hold, bt_cfg["cost_bps"],
                                     bt_cfg["slippage_bps"], bt_cfg["borrow_bps"],
                                     n_sims=d_cfg["random_sims"], seed=cfg["seed"])
    actual_full_sharpe = compute_metrics(bt0["net_return"])["sharpe"]
    fig_random_baselines(rw, rs, actual_full_sharpe,
                         figdir / "diag_random_baselines.png")
    out["random_signal"] = {
        "actual_full_sharpe": round(actual_full_sharpe, 4),
        "random_q05": round(float(np.percentile(rs, 5)), 4),
        "random_median": round(float(np.median(rs)), 4),
        "random_q95": round(float(np.percentile(rs, 95)), 4),
        "actual_percentile": round(float((rs < actual_full_sharpe).mean()), 4),
    }
    spy_oos = spy_ret.loc[spy_ret.index >= split]
    out["benchmark_spy_oos"] = {k: round(v, 4) for k, v in
                                compute_metrics(spy_oos).items()
                                if isinstance(v, (int, float))}

    # 问题9: 因子归因（篮子 OOS vs SPY）+ 成本分解（EWJ/EWG 全样本）
    out["factor_attribution_basket_oos"] = diag.factor_attribution(
        port_ret.loc[port_ret.index >= split], spy_oos)
    out["cost_decomposition"] = diag.cost_decomposition(
        bt0, bt_cfg["cost_bps"], bt_cfg["slippage_bps"])

    # 问题7: CUSUM 断点日期（每对）+ EWJ/EWG 示例图
    breaks = {}
    for g in groups:
        ty, tx = g["tickers"]
        res = cointegration_test(prices[ty], prices[tx])
        resid = np.log(prices[ty].astype(float)) - res["ols_beta"] * np.log(
            prices[tx].astype(float))
        cb = diag.cusum_break_date(resid)
        breaks[g["name"]] = cb["break_date"]
        if g["name"] == chosen_name:
            fig_cusum(cb, g["name"], figdir / "diag_cusum_break.png")
    out["cusum_break_dates"] = breaks
    logger.info("CUSUM 断点日期: %s", breaks)

    # 问题4: Kalman delta 网格 + 残差白噪声
    px0 = basket["cache"][chosen_name]["px"]
    kg = diag.kalman_delta_grid(px0, d_cfg["kalman_deltas"], cfg, split_date)
    kg.to_csv(report_dir / "diag_kalman_delta.csv", index=False)
    out["kalman_delta_grid"] = kg.round(4).to_dict()
    out["kalman_whiteness"] = diag.kalman_residual_whiteness(
        px0.iloc[:, 0], px0.iloc[:, 1], cfg["beta"]["kalman_delta"],
        cfg["beta"]["kalman_burn_in"])

    # 问题6: 功效分析
    out["power_analysis"] = {k: round(v, 1)
                             for k, v in diag.power_analysis().items()}
    logger.info("功效分析（检测 Sharpe 所需年数）: %s", out["power_analysis"])
    return out


# ---------------------------------------------------------------- pipeline

def analyze_pair(name, px_full, vol_full, cfg, split_date, figdir):
    """单对完整分析：beta/spread/zscore/signal/backtest + 图表 + IS/OOS 指标。"""
    y_col, x_col = px_full.columns
    window = cfg["beta"]["window"]
    zwin = cfg["zscore"]["window"]
    sig_cfg = cfg["signal"]
    bt_cfg = cfg["backtest"]

    # 流动性过滤（全样本 20 日均成交额中位数）
    liquidity_ok(px_full, vol_full, min_dollar_volume=bt_cfg["min_avg_dollar_volume"])

    # 动态 beta：滚动 OLS 30/60/120 + Kalman（全部仅用历史窗口）
    betas = {f"OLS {w}d": rolling_ols_beta(px_full[y_col], px_full[x_col], w)
             for w in cfg["beta"]["windows_to_test"]}
    betas["Kalman"] = kalman_beta(px_full[y_col], px_full[x_col],
                                  delta=cfg["beta"]["kalman_delta"],
                                  n_burn=cfg["beta"]["kalman_burn_in"])
    fig_beta(betas, name, figdir / f"{name.replace('/', '_')}_beta.png")

    beta = betas[f"OLS {window}d"]
    spread = compute_spread(px_full[y_col], px_full[x_col], beta)
    z = compute_zscore(spread, zwin)

    # 半衰期只用样本内价差估计（冻结参数，供 IS 与 OOS 共同使用）
    spread_is = spread.loc[spread.index < pd.Timestamp(split_date)]
    hl = half_life(spread_is)
    logger.info("%s 样本内半衰期: %.1f 天", name, hl)

    sig = generate_signals(z, entry=sig_cfg["entry"], exit=sig_cfg["exit"],
                           stop=sig_cfg["stop"])
    sig = apply_time_stop(sig, hl, max_halflives=sig_cfg["time_stop_halflives"])

    fig_prices(px_full, name, split_date, figdir / f"{name.replace('/', '_')}_prices.png")
    fig_spread(spread, name, split_date, figdir / f"{name.replace('/', '_')}_spread.png")
    fig_zscore(z, sig, sig_cfg, name, split_date, figdir / f"{name.replace('/', '_')}_zscore.png")

    bt_full = run_backtest(px_full, sig, cost_bps=bt_cfg["cost_bps"],
                           borrow_bps=bt_cfg["borrow_bps"],
                           slippage_bps=bt_cfg["slippage_bps"], beta=beta)
    bt_is, bt_oos = train_test_split_time(bt_full, split_date)

    results = {"IS": bt_is, "OOS": bt_oos}
    metrics_rows = {}
    for seg, bt in results.items():
        for kind, col in [("gross", "gross_return"), ("net", "net_return")]:
            metrics_rows[f"{seg} {kind}"] = compute_metrics(
                bt[col], bt["position"], bt["turnover"])

    fig_equity({"IS gross": bt_is, "IS net": bt_is, "OOS gross": bt_oos, "OOS net": bt_oos},
               name, split_date, figdir / f"{name.replace('/', '_')}_equity.png")
    fig_drawdown({"IS net": bt_is, "OOS net": bt_oos}, name, split_date,
                 figdir / f"{name.replace('/', '_')}_drawdown.png")

    table = pd.DataFrame(metrics_rows).T
    return {"beta": beta, "spread": spread, "zscore": z, "signals": sig,
            "half_life": hl, "backtest": bt_full, "metrics": table}


def run_walkforward(px_all, groups, cfg, split_date, figdir, report_dir):
    """每 6 个月：用过去 train_days 数据重新选对 + 网格调参，交易下一个半年。"""
    wf_cfg = cfg["validation"]["walkforward"]
    bt_cfg = cfg["backtest"]
    sig_cfg = cfg["signal"]
    window = wf_cfg["beta_window"]

    fold_log, oos_parts = [], []
    start = pd.Timestamp(split_date) + pd.DateOffset(months=0)
    # walk-forward 从 2013 年开始（首个训练窗 2010–2012），覆盖尽可能长的样本外
    wf_start = px_all.index.min() + pd.DateOffset(days=wf_cfg["train_days"] * 1.5)
    for train, test in walk_forward(px_all, wf_start, px_all.index.max(),
                                    window=wf_cfg["train_days"],
                                    step_months=wf_cfg["test_months"]):
        # 1) 训练窗内对 7 个经济组重新做协整筛选
        best = None
        for g in groups:
            ty, tx = g["tickers"]
            try:
                res = cointegration_test(train[ty], train[tx])
            except Exception as exc:
                logger.warning("WF 协整失败 %s: %s", g["name"], exc)
                continue
            if (res["eg_pvalue"] < 0.10 and np.isfinite(res["half_life"])
                    and res["half_life"] < cfg["pair_selection"]["max_half_life_days"]):
                if best is None or res["eg_pvalue"] < best[1]["eg_pvalue"]:
                    best = (g, res)
        if best is None:
            logger.warning("WF %s: 无合格对，该期空仓", test.index[0].date())
            continue
        g, res = best

        # 2) 训练窗内网格调参（entry），其余参数冻结
        px_train = train[g["tickers"]]
        best_entry, best_sharpe = sig_cfg["entry"], -np.inf
        for e in wf_cfg["entry_grid"]:
            bt_tr = build_and_backtest(px_train, entry=e, exit_=sig_cfg["exit"],
                                       stop=sig_cfg["stop"], window=window,
                                       cost_bps=bt_cfg["cost_bps"],
                                       slippage_bps=bt_cfg["slippage_bps"],
                                       borrow_bps=bt_cfg["borrow_bps"])
            m = compute_metrics(bt_tr["net_return"], bt_tr["position"])
            if (m["n_trades"] or 0) >= 3 and m["sharpe"] > best_sharpe:
                best_sharpe, best_entry = m["sharpe"], e

        # 3) 样本外半年：滚动统计在 train 尾部+test 拼接段上计算（只用过去数据）
        warm = train.tail(window * 2)
        px_seg = pd.concat([warm, test])
        beta_seg = rolling_ols_beta(px_seg[g["tickers"][0]], px_seg[g["tickers"][1]], window)
        spread_seg = compute_spread(px_seg[g["tickers"][0]], px_seg[g["tickers"][1]], beta_seg)
        z_seg = compute_zscore(spread_seg, window)
        sig_seg = generate_signals(z_seg, entry=best_entry, exit=sig_cfg["exit"],
                                   stop=sig_cfg["stop"])
        sig_seg = apply_time_stop(sig_seg, res["half_life"],
                                  max_halflives=sig_cfg["time_stop_halflives"])
        bt_seg = run_backtest(px_seg, sig_seg, cost_bps=bt_cfg["cost_bps"],
                              borrow_bps=bt_cfg["borrow_bps"],
                              slippage_bps=bt_cfg["slippage_bps"], beta=beta_seg)
        bt_test = bt_seg.loc[test.index]
        # 跨 fold 强制空仓：期初若带仓（不可能，状态机从 warmup 起步），截断即可
        oos_parts.append(bt_test["net_return"])
        fold_log.append({
            "fold_start": str(test.index[0].date()),
            "fold_end": str(test.index[-1].date()),
            "pair": g["name"], "eg_pvalue": res["eg_pvalue"],
            "half_life": res["half_life"], "entry": best_entry,
            "train_sharpe": best_sharpe,
            "test_return": float((1 + bt_test["net_return"]).prod() - 1),
            "test_sharpe": compute_metrics(bt_test["net_return"])["sharpe"],
        })
        logger.info("WF fold %s: pair=%s entry=%.1f trainSR=%.2f testSR=%.2f",
                    test.index[0].date(), g["name"], best_entry, best_sharpe,
                    fold_log[-1]["test_sharpe"])

    fold_df = pd.DataFrame(fold_log)
    fold_df.to_csv(report_dir / "walkforward_folds.csv", index=False)
    oos_ret = pd.concat(oos_parts).sort_index()
    oos_ret = oos_ret[~oos_ret.index.duplicated()]
    fig_walkforward(oos_ret, figdir / "walkforward_equity.png")
    m = compute_metrics(oos_ret)
    logger.info("Walk-forward 样本外合计: ann_ret=%.2f%%, Sharpe=%.2f, maxDD=%.2f%%",
                m["ann_return"] * 100, m["sharpe"], m["max_drawdown"] * 100)
    return fold_df, m


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    report_dir = Path(cfg["reports"]["dir"])
    figdir = report_dir / "figures"
    setup_logging(report_dir)
    np.random.seed(cfg["seed"])
    logger.info("配置: %s", args.config)

    # ---- 1. 数据 ----------------------------------------------------------
    groups = cfg["pair_groups"]
    tickers = sorted({t for g in groups for t in g["tickers"]})
    prices_raw = load_prices(tickers, cfg["data"]["start"], cfg["data"]["end"],
                             cfg["data"]["cache_dir"])
    volumes_raw = load_volumes(tickers, cfg["data"]["start"], cfg["data"]["end"],
                               cfg["data"]["cache_dir"])
    prices = clean_prices(prices_raw, ffill_limit=cfg["data"]["ffill_limit"])
    volumes = volumes_raw.reindex(prices.index).ffill(limit=cfg["data"]["ffill_limit"])

    split_date = cfg["validation"]["split_date"]
    px_is, px_oos = train_test_split_time(prices, split_date)

    # ---- 2. 样本内候选对筛选 ----------------------------------------------
    logger.info("=== 样本内协整筛选 (%s..%s) ===", px_is.index[0].date(), px_is.index[-1].date())
    screening = find_candidate_pairs(tickers, px_is)
    scr_df = pd.DataFrame(screening)
    scr_df.to_csv(report_dir / "pair_screening_insample.csv", index=False)
    logger.info("样本内筛选 Top10:\n%s", scr_df.head(10).to_string(index=False))

    # 经济逻辑分组内筛选 + p 值/半衰期门槛，取前 n_final_pairs 对
    sel_cfg = cfg["pair_selection"]
    chosen = []
    for g in groups:
        ty, tx = g["tickers"]
        row = next((r for r in screening
                    if {r["ticker_y"], r["ticker_x"]} == {ty, tx}), None)
        if row is None:
            logger.warning("组 %s 无筛选结果", g["name"])
            continue
        row = dict(row, name=g["name"], rationale=g["rationale"])
        if (row["eg_pvalue"] < sel_cfg["pvalue_threshold"]
                and np.isfinite(row["half_life"])
                and row["half_life"] < sel_cfg["max_half_life_days"]):
            chosen.append(row)
        else:
            logger.info("组 %s 未通过门槛: p=%.4f, hl=%.1f", g["name"],
                        row["eg_pvalue"], row["half_life"])
    chosen = sorted(chosen, key=lambda r: r["eg_pvalue"])[:sel_cfg["n_final_pairs"]]
    if not chosen:
        raise RuntimeError("没有任何候选对通过样本内协整门槛")
    logger.info("最终回测对: %s", [c["name"] for c in chosen])

    # 全样本诊断（警示：不可用于选队）
    logger.info("（诊断）全样本协整 p 值，仅供对比，未用于选队:")
    for c in chosen:
        full_res = cointegration_test(prices[c["ticker_y"]], prices[c["ticker_x"]])
        logger.info("  %s: 全样本 EG p=%.4f (样本内 p=%.4f)",
                    c["name"], full_res["eg_pvalue"], c["eg_pvalue"])

    # ---- 3-4. 最终对回测（IS + OOS, 含成本） -------------------------------
    analyses = {}
    for c in chosen:
        name = c["name"]
        ty, tx = c["ticker_y"], c["ticker_x"]
        px_pair = align_dates(prices[[ty]], prices[[tx]])
        vol_pair = volumes[[ty, tx]].reindex(px_pair.index)
        logger.info("=== 分析 %s (%s/%s) ===", name, ty, tx)
        analyses[name] = analyze_pair(name, px_pair, vol_pair, cfg, split_date, figdir)
        table = analyses[name]["metrics"]
        table.to_csv(report_dir / f"metrics_{name.replace('/', '_')}.csv")
        logger.info("%s 绩效:\n%s", name,
                    table[["ann_return", "ann_vol", "sharpe", "sortino",
                           "max_drawdown", "calmar", "n_trades", "win_rate",
                           "profit_factor", "avg_holding_days",
                           "annual_turnover"]].round(3).to_string())

    # ---- 5. 参数敏感性（仅样本内） -----------------------------------------
    sens_cfg = cfg["validation"]["sensitivity"]
    sens_results = {}
    for c in chosen:
        name = c["name"]
        px_pair_is = align_dates(px_is[[c["ticker_y"]]], px_is[[c["ticker_x"]]])
        sens = parameter_sensitivity(
            px_pair_is, sens_cfg, cost_bps=cfg["backtest"]["cost_bps"],
            slippage_bps=cfg["backtest"]["slippage_bps"],
            borrow_bps=cfg["backtest"]["borrow_bps"], stop=cfg["signal"]["stop"])
        sens.to_csv(report_dir / f"sensitivity_{name.replace('/', '_')}.csv", index=False)
        fig_sensitivity(sens, name, figdir / f"{name.replace('/', '_')}_sensitivity.png")
        sens_results[name] = {"sharpe_min": float(sens["sharpe"].min()),
                              "sharpe_max": float(sens["sharpe"].max()),
                              "sharpe_median": float(sens["sharpe"].median())}
        logger.info("%s 敏感性: Sharpe min/median/max = %.2f / %.2f / %.2f",
                    name, sens["sharpe"].min(), sens["sharpe"].median(), sens["sharpe"].max())

    # ---- 6. Walk-Forward 样本外验证 ---------------------------------------
    logger.info("=== Walk-Forward（每 6 个月重新选对 + 调参） ===")
    wf_folds, wf_metrics = run_walkforward(prices, groups, cfg, split_date,
                                           figdir, report_dir)

    # ---- 7. P0: FDR 校正 + 两阶段筛选 --------------------------------------
    twostage = run_two_stage_screen(prices, groups, cfg, figdir, report_dir)

    # ---- 8. P0+P1: 滚动协整门控的多对篮子（风险平价） -----------------------
    logger.info("=== 多对篮子（滚动协整门控 + 逆波动率平价） ===")
    basket = run_basket(prices, groups, cfg, split_date, figdir, report_dir)

    # ---- 9. P0: 成本敏感性（单对 + 篮子） ----------------------------------
    cost_sens = run_cost_sensitivity(basket, chosen[0]["name"], cfg,
                                     report_dir, figdir)

    # ---- 10. P1: Kalman vs OLS --------------------------------------------
    c0 = chosen[0]
    px_c0 = align_dates(prices[[c0["ticker_y"]]], prices[[c0["ticker_x"]]])
    kalman_tbl = run_kalman_vs_ols(c0["name"], px_c0, cfg, split_date, report_dir)

    # ---- 11. P1: Bootstrap 置信区间 + CUSUM 结构断点 ------------------------
    boot_cfg = cfg["validation"]["bootstrap"]
    oos_single = analyses[c0["name"]]["backtest"]
    oos_single = oos_single.loc[oos_single.index >= pd.Timestamp(split_date)]
    oos_basket = basket["port"].loc[basket["port"].index >= pd.Timestamp(split_date)]
    boot = {
        f"{c0['name']} OOS": block_bootstrap_ci(
            oos_single["net_return"], boot_cfg["n_boot"], boot_cfg["block"],
            seed=cfg["seed"]),
        "basket OOS": block_bootstrap_ci(
            oos_basket["port_return"], boot_cfg["n_boot"], boot_cfg["block"],
            seed=cfg["seed"]),
    }
    logger.info("Bootstrap: %s", {k: v["sharpe_ci"] for k, v in boot.items()})

    cusum = {}
    for g in groups:
        ty, tx = g["tickers"]
        res = cointegration_test(prices[ty], prices[tx])
        resid = np.log(prices[ty].astype(float)) - res["ols_beta"] * np.log(
            prices[tx].astype(float))
        cusum[g["name"]] = cusum_test(resid)
    cusum_df = pd.DataFrame(cusum).T
    cusum_df.to_csv(report_dir / "cusum_breaks.csv")
    logger.info("CUSUM 结构断点:\n%s", cusum_df.round(4).to_string())

    # ---- 11.5 深度诊断（复评第 2 轮：问题 1/2/3/4/6/7/8/9） -----------------
    logger.info("=== 深度诊断 ===")
    diag_out = run_diagnostics(prices, groups, cfg, split_date, basket,
                               analyses, c0["name"], figdir, report_dir)

    # ---- 12. 汇总 -----------------------------------------------------------
    summary = {
        "chosen_pairs": [c["name"] for c in chosen],
        "screening_top5": [{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()} for r in screening[:5]],
        "metrics": {name: a["metrics"].round(4).to_dict() for name, a in analyses.items()},
        "half_lives": {name: a["half_life"] for name, a in analyses.items()},
        "sensitivity": sens_results,
        "walkforward": {"metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in wf_metrics.items()},
                        "n_folds": len(wf_folds)},
        "two_stage_fdr": twostage,
        "basket": {"metrics": basket["metrics"].round(4).to_dict(),
                   "gate_open_pct": basket["gate_open_pct"]},
        "cost_sensitivity": cost_sens,
        "kalman_vs_ols": kalman_tbl.round(4).to_dict(),
        "bootstrap": boot,
        "cusum": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                      for kk, vv in v.items()} for k, v in cusum.items()},
        "diagnostics": diag_out,
    }
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("全部完成。报告目录: %s", report_dir)


if __name__ == "__main__":
    main()
