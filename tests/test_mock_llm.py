"""P1-14 / P0 回归：LLMDecision 严格 schema + 强制校验 + MockLLM 接地。"""
import numpy as np
import pytest

from cryptoquant_auto.adapters.mock_llm import (
    LLMDecision, CouncilContext, MockLLM, SchemaValidationError, ACTIONS, MARKET_STATES,
)


def _ctx(**kw):
    base = dict(symbol="BTC", regime="TREND", market_state="BULL",
                fused_action="LONG", base_confidence=0.6, support=["a"],
                contra=[], retrieved_insights=[])
    base.update(kw)
    return CouncilContext(**base)


def test_valid_decision_normalizes_confidence():
    d = LLMDecision(market_state="BULL", confidence=0.77777,
                    rationale=["x"], proposed_action="LONG")
    assert abs(d.confidence - 0.7778) < 1e-9


def test_invalid_market_state_raises_at_construction():
    # P1-14：__post_init__ 强制校验，任何构造必经
    with pytest.raises(SchemaValidationError):
        LLMDecision(market_state="BOGUS", confidence=0.5,
                    rationale=["x"], proposed_action="LONG")


def test_invalid_action_raises():
    with pytest.raises(SchemaValidationError):
        LLMDecision(market_state="BULL", confidence=0.5,
                    rationale=["x"], proposed_action="HODL")


def test_empty_rationale_raises():
    with pytest.raises(SchemaValidationError):
        LLMDecision(market_state="BULL", confidence=0.5,
                    rationale=[], proposed_action="LONG")


def test_confidence_out_of_range_raises():
    with pytest.raises(SchemaValidationError):
        LLMDecision(market_state="BULL", confidence=1.5,
                    rationale=["x"], proposed_action="LONG")


def test_mock_llm_produce_is_deterministic_and_valid():
    llm = MockLLM()
    a = llm.produce(_ctx())
    b = llm.produce(_ctx())
    assert a.proposed_action == b.proposed_action
    assert a.confidence == b.confidence
    assert a.market_state in MARKET_STATES
    assert a.proposed_action in ACTIONS
    assert 0.0 <= a.confidence <= 1.0


def test_mock_llm_low_spi_boosts_confidence():
    llm = MockLLM()
    hi = llm.produce(_ctx(base_confidence=0.6, spi_surprise=0.0))
    lo = llm.produce(_ctx(base_confidence=0.6, spi_surprise=0.9))
    assert lo.confidence < hi.confidence  # 高不确定 → 降置信


def test_complete_alias():
    llm = MockLLM()
    assert llm.complete(_ctx()).proposed_action in ACTIONS
