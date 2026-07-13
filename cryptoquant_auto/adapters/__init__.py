"""交易所适配层 + 接地 LLM 适配器。

- mock        : 零网络 Mock 交易所（Paper/Shadow 调试、故障注入）
- mock_llm    : 阶段3 接地 LLM（严格 schema + Function Calling，零依赖 mock 填表）
"""
from .mock import MockAdapter
from .mock_llm import (MockLLM, LLMDecision, CouncilContext, tool_spec,
                       SchemaValidationError)
from .real_llm import RealLLM, get_llm

__all__ = ["MockAdapter", "MockLLM", "LLMDecision", "CouncilContext",
           "tool_spec", "SchemaValidationError", "RealLLM", "get_llm"]
