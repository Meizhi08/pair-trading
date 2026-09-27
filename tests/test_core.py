"""核心模块单元测试：用合成数据验证统计与回测逻辑的正确性。

合成对：x 为随机游走，y = beta * x + OU(半衰期=20天) 残差，
协整关系已知，可检验各模块能否恢复真实参数。
"""
import numpy as np
import pandas as pd
import pytest

from src.pair_selection import cointegration_test, adf_test, fdr_bh
from src.spread import compute_spread, compute_zscore, half_life, rolling_half_life
from src.signal import generate_signals, apply_time_stop
from src.beta import rolling_ols_beta, kalman_beta
from src.backtest import run_backtest
from src.metrics import compute_metrics
from src.validation import train_test_split_time, block_bootstrap_ci


def make_pair(n=1500, beta=1.2, hl=20, seed=0):
    """构造已知协整关系 y = beta*x + OU 的合成对。"""
    rng = np.random.default_rng(seed)
    x = np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    lam = -np.log(2.0) / hl
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = eps[t - 1] * (1 + lam) + rng.normal(0, 0.01)
    y = np.exp(beta * np.log(x) + eps)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, idx, name="Y"), pd.Series(x, idx, name="X")


def make_random_walks(n=1500, seed=1):
    """两个独立随机游走（不协整）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    y = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))), idx, name="Y")
    x = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))), idx, name="X")
    return y, x


# ---------------------------------------------------------------- 统计模块

def test_cointegration_detected_on_synthetic_pair():
    y, x = make_pair()
    res = cointegration_test(y, x)
    assert res["eg_pvalue"] < 0.01
    assert res["resid_adf"]["stationary_5pct"]
    assert 0.8 < res["ols_beta"] < 1.6          # 真值 1.2


def test_random_walks_not_cointegrated():
    y, x = make_random_walks()
    res = cointegration_test(y, x)
    assert res["eg_pvalue"] > 0.01


def test_adf_on_stationary_series():
    rng = np.random.default_rng(0)
    assert adf_test(pd.Series(rng.normal(size=500)))["stationary_5pct"]


def test_half_life_recovers_true_value():
    y, x = make_pair(hl=20)
    spread = compute_spread(y, x, beta=1.2)
    hl = half_life(spread)
    assert 8 < hl < 60                            # 真值 20，容许估计噪声


def test_rolling_half_life_uses_only_past():
    y, x = make_pair(hl=20)
    spread = compute_spread(y, x, beta=1.2)
    r = rolling_half_life(spread, window=250)
    assert r.iloc[:250].isna().all()             # 热身期全 NaN
    k = 1000
    r_trunc = rolling_half_life(spread.iloc[:k], window=250)
    pd.testing.assert_series_equal(r.iloc[:k], r_trunc)   # 无前视


def test_fdr_bh_correction():
    pvals = list(np.linspace(0.05, 0.9, 50)) + [0.0001]   # 一个真信号 + 均匀噪声
    out = fdr_bh(pvals, alpha=0.05)
    assert out["reject_fdr"].iloc[-1]             # 真信号通过
    assert not out["reject_fdr"].iloc[:-1].any()  # 噪声全部被校正掉
    assert out["p_adj"].iloc[-1] < 0.05
    assert (out["p_adj"] >= np.array(pvals)).all()  # 校正后不小于原始值


# ---------------------------------------------------------------- 信号模块

def test_generate_signals_state_machine():
    z = pd.Series([0, 2.5, 2.6, 0.2, -2.5, -0.1, 0.0, 4.0, 3.9, 0.1])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5)
    p = sig["position"].values
    assert p[1] == -1      # z=2.5 > entry: 做空价差
    assert p[3] == 0       # |z|=0.2 < exit: 平仓
    assert p[4] == 1       # z=-2.5 < -entry: 做多价差
    assert p[5] == 0       # |z|=0.1 平仓
    assert p[7] == -1      # z=4.0 > entry: 开仓
    assert p[8] == 0       # z=3.9 > stop: 止损
    assert p[9] == 0       # 冷却期不立即重开


def test_time_stop_forces_exit():
    z = pd.Series([-2.5] + [1.0] * 19 + [0.0])   # 开仓后 19 天不回归
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=5.0)
    stopped = apply_time_stop(sig, half_life_days=3, max_halflives=2.0)  # 上限 6 天
    assert stopped["position"].iloc[1] == 1
    assert stopped["position"].iloc[8] == 0       # 第 8 天 > 6 天上限，强平
    assert stopped["time_stopped"].any()


# ---------------------------------------------------------------- 回测模块

def test_backtest_no_lookahead():
    """T 日信号必须 T+1 日才进入持仓。"""
    y, x = make_pair(n=400)
    px = pd.concat([y, x], axis=1)
    beta = rolling_ols_beta(y, x, 60)
    z = compute_zscore(compute_spread(y, x, beta), 60)
    sig = generate_signals(z)
    sig = apply_time_stop(sig, 20)
    bt = run_backtest(px, sig, beta=beta)
    # 持仓 = 信号后移一日
    pd.testing.assert_series_equal(
        bt["position"], sig["position"].shift(1).fillna(0), check_names=False)


def test_backtest_costs_reduce_returns():
    y, x = make_pair(n=400)
    px = pd.concat([y, x], axis=1)
    beta = rolling_ols_beta(y, x, 60)
    z = compute_zscore(compute_spread(y, x, beta), 60)
    sig = apply_time_stop(generate_signals(z), 20)
    bt_free = run_backtest(px, sig, cost_bps=0, slippage_bps=0, borrow_bps=0, beta=beta)
    bt_cost = run_backtest(px, sig, cost_bps=50, slippage_bps=50, borrow_bps=500, beta=beta)
    assert bt_cost["net_return"].sum() < bt_free["net_return"].sum()
    assert (bt_cost["cost"] >= 0).all() and (bt_cost["borrow_fee"] >= 0).all()


def test_metrics_keys_and_values():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    m = compute_metrics(r)
    for k in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
              "calmar", "win_rate"]:
        assert k in m
    assert m["ann_vol"] == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_bootstrap_ci_shape():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    out = block_bootstrap_ci(r, n_boot=100, block=20, seed=0)
    lo, hi = out["sharpe_ci"]
    assert lo < hi
    assert 0.0 <= out["sharpe_p_gt0"] <= 1.0


# ---------------------------------------------------------------- 其他

def test_kalman_beta_converges():
    y, x = make_pair(beta=1.2, n=800)
    b = kalman_beta(y, x, delta=1e-3, n_burn=60)
    assert b.iloc[-100:].mean() == pytest.approx(1.2, abs=0.3)


def test_train_test_split_time_boundary():
    idx = pd.bdate_range("2018-01-01", periods=500)
    df = pd.DataFrame({"a": range(500)}, index=idx)
    tr, te = train_test_split_time(df, "2019-01-01")
    assert tr.index.max() < pd.Timestamp("2019-01-01")
    assert te.index.min() >= pd.Timestamp("2019-01-01")
    assert len(tr) + len(te) == len(df)
