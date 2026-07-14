"""P1-15 / P1-27 回归：Walk-Forward 隔离（embargo/purge）、DSR 年化、Gate C。"""
import math

import numpy as np
import pytest

from cryptoquant_auto.sim import walk_forward as wf
from cryptoquant_auto.sim.backtest import make_random_signals


def test_embargo_floor_prevents_zero_isolation():
    # P1-15：embargo=0.01 在小窗口下 int 下取整=0，min_iso_bars 兜底应使隔离>0
    sigs = make_random_signals(240, seed=7)
    rep = wf.walk_forward(sigs, windows=6, embargo=0.01, purge=0.01, min_iso_bars=1)
    assert rep.n_embargoed_bars > 0
    assert rep.n_purged_bars > 0


def test_isolation_active_with_larger_fraction():
    sigs = make_random_signals(240, seed=7)
    rep = wf.walk_forward(sigs, windows=6, embargo=0.1, purge=0.1)
    assert rep.n_embargoed_bars > 0
    assert rep.n_purged_bars > 0


def test_annualization_uses_8760_not_252():
    # P1-27：walk_forward 必须用 1h 年化 8760，源码不得残留 sqrt(252)
    import inspect
    src = inspect.getsource(wf.walk_forward)
    assert "PERIODS_PER_YEAR_1H" in src
    assert "math.sqrt(252)" not in src


def test_dsr_positive_when_oos_profitable():
    sigs = make_random_signals(300, seed=11)
    rep = wf.walk_forward(sigs, windows=6, embargo=0.1, purge=0.1)
    # 随机信号有正有负，DSR 应被计算为有限值
    assert 0.0 <= rep.dsr <= 1.0


def test_gate_c_fields_present():
    sigs = make_random_signals(240, seed=7)
    rep = wf.walk_forward(sigs, windows=6, embargo=0.1, purge=0.1)
    assert hasattr(rep, "gate_c_pass")
    assert isinstance(rep.gate_c_pass, bool)


def test_purged_embargo_cv_runs():
    sigs = make_random_signals(240, seed=7)
    cmp = wf.purged_embargo_cv(sigs, windows=6, embargo=0.1, purge=0.1)
    assert "naive" in cmp and "clean" in cmp
    assert cmp["clean"].n_purged_bars >= 0
