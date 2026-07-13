"""真实 LLM 适配器（生产部署用 · 沙箱禁用）。

严格复用阶段3 的锁表与校验，绝不绕过关口：
  - 复用 mock_llm.tool_spec() 作为 Function Calling 约束（LLM 只能填 emit_trade_decision 表）
  - 复用 mock_llm.LLMDecision.validate() 做严格 schema 校验（任何越界直接拒）
  - 复用 mock_llm.CouncilContext 作为统一上下文入参

部署前提（仅在 8.217.35.251 生产服务器，有外网）：
  - pip install openai
  - 设环境变量 CRYPTOQUANT_LLM_KEY（密钥走 env/Secret Manager，禁止硬编码）
  - 沙箱无外网 + 真实 LLM 被禁：RealLLM 无法在沙箱实例化/调用

降级纪律：调用失败（超时/拒答/越界）默认降级为 MockLLM 填表（fail-closed，不阻塞管线）；
设 degrade_on_error=False 可改为硬失败（上线初期暴露问题用）。
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .mock_llm import (
    LLMDecision, CouncilContext, tool_spec, MockLLM, SchemaValidationError,
)


class RealLLM:
    """OpenAI 兼容 chat-completions + Function Calling 锁表适配器。"""

    def __init__(self, model: str = "gpt-4o-mini", *,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = 20.0, max_retries: int = 3,
                 degrade_on_error: bool = True):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "生产环境需 pip install openai；沙箱禁用真实 LLM") from e
        key = api_key or os.getenv("CRYPTOQUANT_LLM_KEY")
        if not key:
            raise RuntimeError(
                "缺少 CRYPTOQUANT_LLM_KEY（生产服务器 env/Secret Manager 设置）")
        self.model = model
        self._degrade = degrade_on_error
        self._client = OpenAI(api_key=key, base_url=base_url,
                              timeout=timeout, max_retries=max_retries)

    # ---- 对外统一接口（与 MockLLM 一致）----
    def produce(self, ctx: CouncilContext) -> LLMDecision:
        try:
            return self._call(ctx)
        except Exception:
            if not self._degrade:
                raise
            # fail-closed：降级为确定性接地 mock，保证管线不中断
            return MockLLM().produce(ctx)

    def complete(self, ctx: CouncilContext) -> LLMDecision:
        return self.produce(ctx)

    def _call(self, ctx: CouncilContext) -> LLMDecision:
        sys_prompt = (
            "你是加密货币合约信号决策引擎。必须调用 emit_trade_decision 工具输出，"
            "严禁自由文本。market_state 限 BULL/BEAR/RANGE/CRASH；"
            "proposed_action 限 LONG/SHORT/HOLD；confidence ∈ [0,1]。"
        )
        user_payload = {
            "symbol": ctx.symbol,
            "regime": ctx.regime,
            "analyst_market_state": ctx.market_state,
            "fused_action": ctx.fused_action,
            "base_confidence": round(float(ctx.base_confidence), 3),
            "support": ctx.support,
            "contra": ctx.contra,
            "retrieved_insights": ctx.retrieved_insights,
            "spi_surprise": round(float(ctx.spi_surprise), 3),
        }
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            tools=[tool_spec()],                       # 复用阶段3 锁表
            tool_choice={"type": "function",
                         "function": {"name": "emit_trade_decision"}},
            temperature=0.0,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            raise SchemaValidationError("LLM 未返回工具调用（未锁表）")
        args = json.loads(msg.tool_calls[0].function.arguments)
        # 复用严格 schema 校验：越界直接拒（等价于 pydantic 失败）
        return LLMDecision(**args).validate()


def get_llm(degrade_on_error: bool = True):
    """零配置工厂（小白友好）：

    - 设了 CRYPTOQUANT_LLM_KEY 且 openai 可用 → 返回 RealLLM（真 LLM）
    - 没设密钥 / 缺 openai / 任何异常 → 返回 MockLLM（确定性接地 mock）

    调用方（四角色决策）无需判断，直接 get_llm().produce(ctx)。
    生产服务器设了密钥就自动用真 LLM，沙箱/没密钥自动降级，零改代码。
    """
    try:
        if not os.getenv("CRYPTOQUANT_LLM_KEY"):
            return MockLLM()
        return RealLLM(degrade_on_error=degrade_on_error)
    except Exception:
        return MockLLM()
