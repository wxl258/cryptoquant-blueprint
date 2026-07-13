"""阶段0.5 / P1-14-quant 回归：DSR / PSR / BH-FDR 闸门。"""
import numpy as np
import pytest

from cryptoquant_auto.sim.metrics import (
    deflated_sharpe, probabilistic_sharpe, bh_fdr,
)


def test_deflated_sharpe_monotonic_in_n_trials():
    # DSR(N) 随 N 增大而单调不增（多重检验更严）
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 500)
    dsr_1 = deflated_sharpe(r, n_trials=1, sr0=0.0, periods_per_year=1)
    dsr_50 = deflated_sharpe(r, n_trials=50, sr0=0.0, periods_per_year=1)
    assert dsr_50 <= dsr_1 + 1e-9


def test_deflated_sharpe_range():
    rng = np.random.default_rng(2)
    r = rng.normal(0.002, 0.01, 400)
    assert 0.0 <= deflated_sharpe(r, n_trials=1, periods_per_year=1) <= 1.0


def test_probabilistic_sharpe_range():
    rng = np.random.default_rng(3)
    r = rng.normal(0.002, 0.01, 400)
    assert 0.0 <= probabilistic_sharpe(r, periods_per_year=1) <= 1.0


def test_bh_fdr_rejects_strong_signal():
    # 1 个极显著 + 多个不显著 → 仅最强者被拒
    pvals = [0.0001, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3]
    reject, adj = bh_fdr(pvals, alpha=0.10)
    assert reject[0] is True
    assert sum(reject) == 1
    # 调整后 p 在「按原始 p 升序」下单调不降（BH 性质）
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(len(order) - 1))


def test_bh_fdr_empty():
    reject, adj = bh_fdr([], alpha=0.10)
    assert reject == [] and adj == []
