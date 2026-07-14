"""RealLLM 供应商切换离线验证（沙箱无外网，monkeypatch openai 模块）。

覆盖：
  - 无 CRYPTOQUANT_LLM_KEY → 返回 MockLLM（降级，零改代码）
  - 设 DeepSeek V4 环境变量 → 返回 RealLLM，底层客户端指向 deepseek base_url/模型
  - produce() 走 tools 锁表路径，返回合法 LLMDecision
  - 健壮性：模型未在 tool_calls 而在 content 返回 JSON 时，仍解析并走严格 schema 校验
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from cryptoquant_auto.adapters.real_llm import get_llm, RealLLM, MockLLM, _extract_json
from cryptoquant_auto.adapters.mock_llm import CouncilContext, LLMDecision

_ARGS = '{"market_state":"RANGE","proposed_action":"HOLD","confidence":0.5,"rationale":["x"]}'


# 伪造 openai 模块：捕获构造参数 + 返回带 tool_calls 的响应
@pytest.fixture
def fake_openai(monkeypatch):
    captured = {}

    class _TC:
        def __init__(self, args: str):
            self.function = types.SimpleNamespace(arguments=args)

    class _Msg:
        def __init__(self, tool_calls, content=None):
            self.tool_calls = tool_calls
            self.content = content

    class _Choice:
        def __init__(self, tool_calls, content=None):
            self.message = _Msg(tool_calls, content)

    class _Resp:
        def __init__(self, tool_calls=None, content=None):
            self.choices = [_Choice(tool_calls, content)]

    class _Completions:
        def create(self, **kw):
            captured["create_kwargs"] = kw
            return _Resp([_TC(_ARGS)])

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


# 兜底模式：content 返回 JSON，无 tool_calls
@pytest.fixture
def fake_openai_content(monkeypatch):
    captured = {}

    class _Msg:
        def __init__(self, content):
            self.tool_calls = None
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, **kw):
            captured["create_kwargs"] = kw
            return _Resp(_ARGS)

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


def test_deepseek_v4_flash_env(fake_openai, monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_LLM_KEY", "sk-deepseek-test")
    monkeypatch.setenv("CRYPTOQUANT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRYPTOQUANT_LLM_MODEL", "deepseek-v4-flash")
    llm = get_llm()
    assert isinstance(llm, RealLLM)
    assert llm.model == "deepseek-v4-flash"
    assert fake_openai["base_url"] == "https://api.deepseek.com"
    assert fake_openai["api_key"] == "sk-deepseek-test"


def test_deepseek_v4_pro_model_name(fake_openai, monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_LLM_KEY", "sk-deepseek-test")
    monkeypatch.setenv("CRYPTOQUANT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRYPTOQUANT_LLM_MODEL", "deepseek-v4-pro")
    assert get_llm().model == "deepseek-v4-pro"


def test_deepseek_produce_uses_tools_locktable(fake_openai, monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_LLM_KEY", "sk-deepseek-test")
    monkeypatch.setenv("CRYPTOQUANT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRYPTOQUANT_LLM_MODEL", "deepseek-v4-flash")
    llm = get_llm()
    d: LLMDecision = llm.produce(_ctx())
    assert d.proposed_action == "HOLD"
    assert d.market_state == "RANGE"
    assert fake_openai["create_kwargs"]["tools"]
    assert fake_openai["create_kwargs"]["tool_choice"]["function"]["name"] == "emit_trade_decision"


def test_content_json_fallback_still_validates(fake_openai_content, monkeypatch):
    monkeypatch.setenv("CRYPTOQUANT_LLM_KEY", "sk-deepseek-test")
    monkeypatch.setenv("CRYPTOQUANT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CRYPTOQUANT_LLM_MODEL", "deepseek-v4-pro")
    llm = get_llm()
    d: LLMDecision = llm.produce(_ctx())
    assert d.proposed_action == "HOLD"   # 无 tool_calls，content JSON 兜底解析成功


def test_extract_json_handles_fence_and_surrounding_text():
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert _extract_json('思考：{"market_state":"RANGE"} 结束') == {"market_state": "RANGE"}
    with pytest.raises(json.JSONDecodeError):
        _extract_json("完全不是 json")
