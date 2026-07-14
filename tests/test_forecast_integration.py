"""TSFM 预报 → 决策管线集成测试（蓝图 v0.3 · 路线图 A.TSFM）。

验证：
  1) 强正预报且动量≈0 → 议会翻多（LONG）；无预报 → HOLD（向后兼容）。
  2) 预报与动量背离 → 置信下降（不确定性软降级）。
  3) forecast=None 与不改签名行为完全一致（向后兼容）。
  4) make_tsfm('onnx', 权重缺失) 优雅降级到 DistilledTSFM（numpy）。
  5) ONNX Runtime 后端真实加载并推理（需 onnx + onnxruntime，否则跳过）。
"""
import numpy as np
import pytest
from types import SimpleNamespace

from cryptoquant_auto.meta.agents import Analyst, FourRoleCouncil, FEATURE_NAMES
from cryptoquant_auto.signals.tsfm import (
    make_tsfm, DistilledTSFM, OnnxTimeMoeForecaster,
)


def _feat_vector(**kw):
    v = np.zeros(len(FEATURE_NAMES), dtype=float)
    for k, val in kw.items():
        v[FEATURE_NAMES.index(k)] = val
    return v


def _council():
    from cryptoquant_auto.meta.memory import FinMemMemory
    from cryptoquant_auto.adapters.mock_llm import MockLLM
    return FourRoleCouncil(FinMemMemory(), llm=MockLLM())


def test_forecast_blend_flips_to_long():
    council = _council()
    # 动量≈0、adx 中性偏高、fng 中性（基准 conviction≈0.525 越过 min_conviction）
    feat = _feat_vector(adx=0.5, fng=0.5)
    v_none = council.decide("BTC", feat, "TREND", record=False, forecast=None)
    v_pos = council.decide("BTC", feat, "TREND", record=False, forecast=0.02)
    assert v_none.action == "HOLD", "无预报且动量≈0 → HOLD"
    assert v_pos.action == "LONG", "强正预报应翻多"


def test_forecast_divergence_lowers_confidence():
    council = _council()
    feat = _feat_vector(momentum=0.01)   # 动量正 → 基准偏多
    v_none = council.decide("BTC", feat, "TREND", record=False, forecast=None)
    v_div = council.decide("BTC", feat, "TREND", record=False, forecast=-0.02)
    assert v_div.confidence < v_none.confidence, \
        "预报与动量背离应降低置信（不确定性软降级）"


def test_forecast_none_is_backward_compat():
    council = _council()
    feat = _feat_vector(momentum=0.01, adx=0.6)
    v1 = council.decide("BTC", feat, "TREND", record=False)          # 默认 forecast=None
    v2 = council.decide("BTC", feat, "TREND", record=False, forecast=None)
    assert v1.action == v2.action
    assert v1.confidence == v2.confidence


def test_make_tsfm_onnx_degrades_without_model():
    fc = make_tsfm("onnx", model_path="/nonexistent/time_moe_small.onnx")
    assert isinstance(fc, DistilledTSFM), "权重缺失 → 降级 DistilledTSFM"
    assert fc.name == "distilled_numpy"


def test_onnx_forecaster_plumbing():
    """真实构造一个线性 ONNX 模型，验证 OnnxTimeMoeForecaster 加载+推理路径。"""
    onnx_mod = pytest.importorskip("onnx")
    import os
    import tempfile
    from onnx import helper, TensorProto

    lookback = 4
    rng = np.random.RandomState(0)
    W = rng.randn(1, lookback).astype(np.float32)
    b = rng.randn(1).astype(np.float32)
    node_mm = helper.make_node("MatMul", ["input", "W"], ["mm"])
    node_add = helper.make_node("Add", ["mm", "b"], ["output"])
    graph = helper.make_graph(
        [node_mm, node_add], "lin",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, lookback])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor(W, "W", [1, lookback]),
         helper.make_tensor(b, "b", [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx_mod.save(model, path)
    try:
        fc = OnnxTimeMoeForecaster(path, lookback=lookback)
        recent = rng.randn(lookback).astype(float)
        pt, _, _ = fc.forecast(recent, horizon=1)
        expected = float(W @ recent + b)
        assert abs(pt[0] - expected) < 1e-4, "ONNX 推理应等于 W·recent+b"
    finally:
        os.remove(path)
