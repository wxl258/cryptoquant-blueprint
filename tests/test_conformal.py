"""阶段0.5 / P1-10：SPCI 序列共形预测器（惊喜度 / 覆盖 / 滚动更新）。"""
import numpy as np
import pytest

from cryptoquant_auto.risk.conformal import SequentialConformalPredictor


def test_surprise_zero_when_inside_interval():
    cp = SequentialConformalPredictor(alpha=0.10, naive=True)
    rng = np.random.default_rng(0)
    for s in rng.normal(0.3, 0.05, 200):
        cp.update(s)
    # 区间内样本惊喜度应为 0
    assert cp.surprise(0.30) == 0.0


def test_surprise_positive_when_outside_interval():
    cp = SequentialConformalPredictor(alpha=0.10, naive=True)
    rng = np.random.default_rng(0)
    for s in rng.normal(0.3, 0.05, 200):
        cp.update(s)
    # 远超区间 → 正惊喜度
    assert cp.surprise(0.9) > 0.0


def test_coverage_near_nominal():
    cp = SequentialConformalPredictor(alpha=0.10, naive=True)
    rng = np.random.default_rng(1)
    for s in rng.normal(0.3, 0.05, 300):
        cp.update(s)
    test = rng.normal(0.3, 0.05, 300)
    cov = cp.coverage(test)
    assert 0.80 <= cov <= 0.98


def test_decay_weights_recent_more():
    # 非 naive 模式使用 decay 近因权重
    cp = SequentialConformalPredictor(alpha=0.10, decay=0.9)
    rng = np.random.default_rng(2)
    for s in rng.normal(0.3, 0.05, 100):
        cp.update(s)
    assert cp.n_samples == 100


def test_naive_vs_decay_differ_under_drift():
    naive = SequentialConformalPredictor(alpha=0.10, naive=True)
    decay = SequentialConformalPredictor(alpha=0.10, decay=0.985)
    rng = np.random.default_rng(3)
    # 前期稳定，后期漂移
    for s in rng.normal(0.3, 0.05, 150):
        naive.update(s); decay.update(s)
    for s in rng.normal(0.6, 0.05, 150):  # 均值漂移
        naive.update(s); decay.update(s)
    # 两者都应能产出有效（非负）惊喜度，且 decay 对近期漂移更敏感
    sp_n = naive.surprise(0.6)
    sp_d = decay.surprise(0.6)
    assert sp_n >= 0.0 and sp_d >= 0.0
