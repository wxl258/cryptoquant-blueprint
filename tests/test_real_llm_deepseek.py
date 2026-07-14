"""RealLLM 供应商切换离线验证（沙箱无外网，monkeypatch openai 模块）。

覆盖：
  - 无 CRYPTOQUANT_LLM_KEY → 返回 MockLLM（降级，零改代码）
  - 设 DeepSeek 环境变量 → 返回 RealLLM，且底层 OpenAI 客户端指向 deepseek base_url/模型
  - produce() 走 tools 锁表路径，返回合法 LLMDecision（验证 Function Calling 契约不破）
"""
from __future__ import annotations

import sys
import types

import pytest

from cryptoquant_auto.adapters.real_llm import get_llm, RealLLM, MockLLM
from cryptoquant_auto.adapters.mock_llm import CouncilContext, LLMDecision


# 伪造 openai 模块：捕获构造参数 + 返回带 tool_calls 的响应
@pytest.fixture
def fake_openai(monkeypatch):
    captured = {}

    class _TC:
        def __init__(self, args: str):
            self.function = types.SimpleNamespace(arguments=args)

    class _Msg:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    class _Choice:
        def __init__(self, tool_calls):
            self.message = _Msg(tool_calls)

    class _Resp:
        def __init__(self, tool_calls):
            self.choices = [_Choice(tool_calls)]

    class _Completions:
        def create(self, **kw):
            captured["create_kwargs"] = kw
            args = '{"market_state":"RANGE","proposed_action":"HOLD","confidence":0.5,"rationale":["x"]}'
            return _Resp([_TC(args)])

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self, api_key, base_url=None, timeout=0.0, max_retries=0):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = _Chat()

    mod = types.ModuleType("openai")
    mod.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", mod)
    return captured


def _ctx() -> CouncilContext:
    return CouncilContext(
        symbol="BTC", regime="RANGE", market_state="RANGE",
        fused_action="HOLD", base_confidence=0.5, support=[], contra=[],
        retrieved_insights=[], spi_surprise=0.0,
    )


def test_no_key_returns_mock(monkeypatch):
    monkeypatch.delenv("CRYPTOQUANT_LLM_KEY", raising=False)
    monkeypatch.delenv("CRYPTOQUANT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CRYPTOQUANT_LLM_MODEL", raising=False)
    assert isinstance(get_llm(), MockLLM)


def test_deepseek_env_builds_real_llm(fake_openai, monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_LLM_KEY", "sk-deepseek-test")
    monkeypatch.setenv("CRYPTOQUANT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRYPTOQUANT_LLM_MODEL", "deepseek-chat")
    llm = get_llm()
    assert isinstance(llm, RealLLM)
    assert llm.model == "deepseek-chat"
    assert fake_openai["base_url"] == "https://api.deepseek.com"
    assert fake_openai["api_key"] == "sk-deepseek-test"


def test_deepseek_produce_uses_tools_locktable(fake_openai, monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_LLM_KEY", "sk-deepseek-test")
    monkeypatch.setenv("CRYPTOQUANT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRYPTOQUANT_LLM_MODEL", "deepseek-chat")
    llm = get_llm()
    d: LLMDecision = llm.produce(_ctx())
    assert d.proposed_action == "HOLD"
    assert d.market_state == "RANGE"
    # 确认走了 Function Calling 锁表（tools 非空 + tool_choice 锁定 emit_trade_decision）
    assert fake_openai["create_kwargs"]["tools"]
    assert fake_openai["create_kwargs"]["tool_choice"]["function"]["name"] == "emit_trade_decision"
