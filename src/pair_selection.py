"""候选对筛选：ADF 单位根检验 + Engle-Granger 协整检验。

注意：筛选必须只在样本内数据上进行（run.py 传入的 prices 即样本内）。
全样本 p 值仅作诊断参考，不能用于选队结论。
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from src.spread import half_life

logger = logging.getLogger(__name__)


def fdr_bh(pvalues, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg FDR 校正。

    Returns pd.Series(index=原索引): p_adj, reject (校正后是否通过)。
    91 个组合只报最小 p 值是典型的 data snooping，必须看校正后 p 值。
    """
    p = np.asarray(pvalues, dtype=float)
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return pd.DataFrame({"p_adj": p_adj, "reject_fdr": reject})


def rolling_cointegration(y: pd.Series, x: pd.Series, window: int = 250) -> pd.DataFrame:
    """滚动协整检验（P1）：每个时点只用过去 window 天数据。

    实现：滚动 OLS beta（窗口内）-> 残差 -> 窗口内 ADF(maxlag=1)。
    这是对 Engle-Granger 的近似（ADF 临界值与 EG 略有差异），
    但作为"现在是否仍处于协整状态"的门控信号足够。

    Returns DataFrame[beta, resid_adf_pvalue]，前 window 行为 NaN。
    """
    df = pd.concat([y, x], axis=1).dropna().astype(float)
    ly = np.log(df.iloc[:, 0])
    lx = np.log(df.iloc[:, 1])
    beta = ly.rolling(window, min_periods=window).cov(lx) / lx.rolling(
        window, min_periods=window).var()
    resid = ly - beta * lx

    n = len(df)
    pvals = np.full(n, np.nan)
    rv = resid.values
    for t in range(window, n):
        w = rv[t - window:t]
        if np.isnan(w).any():
            continue
        try:
            pvals[t] = adfuller(w, maxlag=1, autolag=None, regression="n",
                                result_object=False)[1]
        except Exception:
            continue
    out = pd.DataFrame({"beta": beta, "resid_adf_pvalue": pvals}, index=df.index)
    return out


def adf_test(series: pd.Series, regression: str = "c") -> dict:
    """Augmented Dickey-Fuller 单位根检验。

    Returns dict: adf_stat, pvalue, lags, nobs, crit_1%/5%/10%, stationary_5pct
    """
    s = pd.Series(series).dropna().astype(float)
    stat, p, lags, nobs, crit, _ = adfuller(s, autolag="AIC", regression=regression,
                                            result_object=False)
    return {
        "adf_stat": stat,
        "pvalue": p,
        "lags": lags,
        "nobs": nobs,
        "crit_1%": crit["1%"],
        "crit_5%": crit["5%"],
        "crit_10%": crit["10%"],
        "stationary_5pct": bool(p < 0.05),
    }


def cointegration_test(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger 两步协整检验（对数价格）。

    步骤1: OLS 回归 log(y) = a + b*log(x)，得静态对冲比 b（仅用于筛选诊断，
           交易中禁止使用全样本 beta）。
    步骤2: 对残差做 ADF 检验（Engle-Granger 临界值由 statsmodels coint 给出）。

    Returns dict: eg_stat, eg_pvalue, eg_crit_1%/5%/10%, ols_beta,
                  resid_adf (dict), half_life（残差 OU 半衰期，交易日）
    """
    df = pd.concat([y, x], axis=1).dropna()
    ly = np.log(df.iloc[:, 0].astype(float))
    lx = np.log(df.iloc[:, 1].astype(float))

    stat, p, crit = coint(ly, lx)

    beta = np.polyfit(lx, ly, 1)[0]
    resid = ly - beta * lx
    adf = adf_test(resid)
    hl = half_life(resid)

    return {
        "eg_stat": stat,
        "eg_pvalue": p,
        "eg_crit_1%": crit[0],
        "eg_crit_5%": crit[1],
        "eg_crit_10%": crit[2],
        "ols_beta": beta,
        "resid_adf": adf,
        "half_life": hl,
    }


def find_candidate_pairs(tickers, prices: pd.DataFrame) -> list[dict]:
    """对 tickers 的所有两两组合做协整检验，按 EG p 值升序返回。

    返回 list[dict]，每项包含 ticker_y, ticker_x, nobs, eg_stat, eg_pvalue,
    eg_crit_5%, resid_adf_stat, resid_adf_pvalue, ols_beta, half_life。
    经济逻辑筛选（同行业 / 同驱动因素）在 run.py 按分组完成。
    """
    results = []
    for a, b in combinations(list(tickers), 2):
        pair = prices[[a, b]].dropna()
        if len(pair) < 252:
            logger.warning("%s/%s 样本不足 (%d)，跳过", a, b, len(pair))
            continue
        try:
            res = cointegration_test(pair[a], pair[b])
        except Exception as exc:  # 不静默失败：记录并继续
            logger.warning("协整检验失败 %s/%s: %s", a, b, exc)
            continue
        results.append({
            "ticker_y": a,
            "ticker_x": b,
            "nobs": len(pair),
            "eg_stat": res["eg_stat"],
            "eg_pvalue": res["eg_pvalue"],
            "eg_crit_5%": res["eg_crit_5%"],
            "resid_adf_stat": res["resid_adf"]["adf_stat"],
            "resid_adf_pvalue": res["resid_adf"]["pvalue"],
            "ols_beta": res["ols_beta"],
            "half_life": res["half_life"],
        })
    results.sort(key=lambda r: r["eg_pvalue"])
    logger.info("候选对筛选完成: %d 组合", len(results))
    return results
