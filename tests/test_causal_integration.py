"""因果发现 → 特征管线集成测试（蓝图 v0.3 · 下一步第1周）。

验证：
  1) get_causal_features 返回 FEATURE_NAMES 的子集且非空（真实跑通 Granger 路径）。
  2) 议会 decide() 接受 reduced feature_names 不抛 KeyError。
  3) 被因果剔除的特征经 _fget 中性化：剔除 adx 后 Analyst.conviction 下降。
  4) 不传 feature_names 时行为与改动前完全一致（向后兼容）。
"""
import numpy as np
from types import SimpleNamespace

from cryptoquant_auto.meta.agents import Analyst, FourRoleCouncil, FEATURE_NAMES
from cryptoquant_auto.causal_discovery import get_causal_features


def _feat_vector(**kw):
    """按 FEATURE_NAMES 顺序构造 9 维向量，未指定项填 0。"""
    v = np.zeros(len(FEATURE_NAMES), dtype=float)
    for k, val in kw.items():
        v[FEATURE_NAMES.index(k)] = val
    return v


def test_get_causal_features_returns_subset():
    feats = get_causal_features("BTC")
    assert isinstance(feats, list) and feats, "应返回非空特征列表"
    assert set(feats).issubset(set(FEATURE_NAMES)), "应是 FEATURE_NAMES 的子集"


def test_decide_with_reduced_feature_names_no_error():
    from cryptoquant_auto.meta.memory import FinMemMemory
    from cryptoquant_auto.adapters.mock_llm import MockLLM
    mem = FinMemMemory()
    council = FourRoleCouncil(mem, llm=MockLLM())
    feat = _feat_vector(momentum=0.05, vol_regime=1.0)
    v = council.decide("BTC", feat, "TREND", record=False,
                       feature_names=["momentum", "vol_regime"])
    assert v.action in ("LONG", "SHORT", "HOLD")


def test_dropped_feature_neutralized():
    profile = SimpleNamespace(risk_appetite=0.5)
    feat = _feat_vector(adx=1.0, momentum=0.05, fng=0.5, vol_regime=0.0)
    a_full = Analyst().assess("BTC", feat, "TREND", profile,
                              active=set(FEATURE_NAMES))
    a_no_adx = Analyst().assess("BTC", feat, "TREND", profile,
                                active=set(FEATURE_NAMES) - {"adx"})
    assert a_no_adx["conviction"] < a_full["conviction"], \
        "剔除 adx 后 conviction 应下降（被中性化），而非保持或升高"
    # 被剔除 adx 的读数应为中性默认值 0.0
    assert "adx=0.00" in " ".join(a_no_adx["drivers"])


def test_full_feature_names_equals_backward_compat():
    """不传 feature_names 时行为与改动前完全一致（向后兼容校验）。"""
    from cryptoquant_auto.meta.memory import FinMemMemory
    from cryptoquant_auto.adapters.mock_llm import MockLLM
    mem = FinMemMemory()
    council = FourRoleCouncil(mem, llm=MockLLM())
    feat = _feat_vector(momentum=0.05, vol_regime=1.0, adx=0.6)
    v1 = council.decide("BTC", feat, "TREND", record=False)
    v2 = council.decide("BTC", feat, "TREND", record=False,
                        feature_names=list(FEATURE_NAMES))
    assert v1.action == v2.action
    assert v1.confidence == v2.confidence
