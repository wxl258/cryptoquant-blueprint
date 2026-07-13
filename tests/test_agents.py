"""P0-3 / P1-10 回归：四角色议会决策、禁易 regime 双命名空间、SPCI 惊喜度软降级。"""
import numpy as np
import pytest

from cryptoquant_auto.meta.agents import FourRoleCouncil, LONG, SHORT, HOLD
from cryptoquant_auto.meta.memory import FinMemMemory
from cryptoquant_auto.risk.conformal import SequentialConformalPredictor


def _feat():
    # 9 维特征占位（adx, rsi, atr_pct, vol_regime, fr, fr_delta, oi_pct, fng, momentum）
    return np.array([0.3, 0.5, 0.02, 1.0, 0.0, 0.0, 0.0, 0.5, 0.01], dtype=float)


def _council(tmp_path=None):
    mem = FinMemMemory(base_dir=str(tmp_path)) if tmp_path else FinMemMemory()
    return FourRoleCouncil(mem)


def test_decide_returns_verdict_with_action(tmp_path):
    c = _council(tmp_path)
    v = c.decide("BTC", _feat(), "TREND", record=True)
    assert v.action in (LONG, SHORT, HOLD)
    assert 0.0 <= v.confidence <= 1.0
    assert v.decision_id  # P0-4 回填 ID


def test_forbidden_regime_blocks_via_profile(tmp_path):
    c = _council(tmp_path)
    c.profile.forbidden_regimes.append("TREND")
    v = c.decide("BTC", _feat(), "TREND", record=False)
    assert v.action == HOLD
    assert any("forbidden_regime" in vt for vt in v.vetoes)


def test_conformal_surprise_soft_holds_low_conviction(tmp_path):
    # P1-10：conformal 跨 tick 学习后，低置信决策被高惊喜度软降级为 HOLD
    c = _council(tmp_path)
    conf = SequentialConformalPredictor(alpha=0.10)
    # 预热：若干轮建立置信分布
    for _ in range(8):
        c.decide("BTC", _feat(), "TREND", record=False, conformal=conf)
    # 制造一个异常低置信情景：极端恐惧特征
    bad = np.array([0.05, 0.1, 0.05, -1.0, 0.0, 0.0, 0.0, 0.02, -0.02], dtype=float)
    v = c.decide("BTC", bad, "CRASH", record=False, conformal=conf)
    assert conf.n_samples >= 5
    # 异常低置信情景应被高惊喜度软降级（或本身低置信被钳制为 HOLD）
    assert v.action == HOLD


def test_conformal_does_not_break_normal_flow(tmp_path):
    c = _council(tmp_path)
    conf = SequentialConformalPredictor(alpha=0.10)
    v = c.decide("BTC", _feat(), "TREND", record=True, conformal=conf)
    assert v.action in (LONG, SHORT, HOLD)
