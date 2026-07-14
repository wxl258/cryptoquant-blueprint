"""P1-16 / P1-14-quant 回归：受控 A/B（Welch 自相关修正、DSR/PSR 标签）。"""
import numpy as np
import pytest

from cryptoquant_auto.sim.ab_harness import (
    controlled_ab, make_synthetic_returns, _welch_p, _lag1_acf, _eff_n,
)


def test_welch_iid_equal_means_high_p():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, 500)
    b = rng.normal(0.0, 0.05, 500)
    p = _welch_p(a, b)
    assert 0.0 <= p <= 1.0
    # 同分布 → 不应显著（p 偏大）
    assert p > 0.05


def test_welch_detects_mean_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.05, 500)
    b = rng.normal(0.05, 0.05, 500)
    p = _welch_p(a, b)
    assert p < 0.05  # 均值显著不同


def test_lag1_acf_near_zero_for_iid():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 500)
    assert abs(_lag1_acf(x)) < 0.2


def test_lag1_acf_high_for_random_walk():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 1, 500))
    assert _lag1_acf(x) > 0.9


def test_eff_n_shrinks_under_autocorrelation():
    # 强正自相关 → 有效样本量应小于名义 n
    assert _eff_n(500, 0.9) < 500
    assert _eff_n(500, 0.0) == 500.0


def test_controlled_ab_labels_dsr_and_psr():
    a = make_synthetic_returns(300, sr_target=0.15, seed=11)
    b = make_synthetic_returns(300, sr_target=0.45, seed=12)
    res = controlled_ab(a, b, n_trials=20, sr0=0.0, periods_per_year=1)
    assert 0.0 <= res.dsr_rule <= 1.0
    assert 0.0 <= res.psr_rule <= 1.0
    assert res.winner in ("llm", "rule", "tie")
    assert isinstance(res.significant, bool)


def test_controlled_ab_does_not_crash_on_small_input():
    a = make_synthetic_returns(10, 0.2, seed=5)
    b = make_synthetic_returns(10, 0.2, seed=6)
    res = controlled_ab(a, b, n_trials=2)
    assert 0.0 <= res.p_value <= 1.0
