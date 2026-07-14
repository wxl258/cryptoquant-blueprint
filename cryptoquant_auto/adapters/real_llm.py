"""真实 LLM 适配器（生产部署用 · 沙箱禁用）。

严格复用阶段3 的锁表与校验，绝不绕过关口：
  - 复用 mock_llm.tool_spec() 作为 Function Calling 约束（LLM 只能填 emit_trade_decision 表）
  - 复用 mock_llm.LLMDecision.validate() 做严格 schema 校验（任何越界直接拒）
  - 复用 mock_llm.CouncilContext 作为统一上下文入参

部署前提（仅在 8.217.35.251 生产服务器，有外网）：
  - pip install openai
  - 设环境变量 CRYPTOQUANT_LLM_KEY（密钥走 env/Secret Manager，禁止硬编码）
  - 沙箱无外网 + 真实 LLM 被禁：RealLLM 无法在沙箱实例化/调用

供应商切换（均 OpenAI 兼容 chat-completions + Function Calling 锁表）：
  - 默认（不设）指向 OpenAI 官方，模型 gpt-4o-mini
  - DeepSeek（最新一代 V4）：设
        CRYPTOQUANT_LLM_BASE_URL=https://api.deepseek.com
        CRYPTOQUANT_LLM_MODEL=deepseek-v4-flash   （推荐：低成本快；支持 tools）
                             或 deepseek-v4-pro    （质量优先；支持 tools）
    ⚠️ 旧 deepseek-chat / deepseek-reasoner 将于 2026-07-24 15:59 UTC 退役，勿再用。
       V4 的 reasoning 是参数(thinking)而非独立模型，默认非思考模式即吐 tool_calls，锁表正常。
  - 其他 OpenAI 兼容端点同理，仅改 base_url / model，零改代码
  - 健壮性：模型若未吐 tool_calls 但在 content 返回 JSON，也会解析并走同一严格 schema 校验（锁表不破）

降级纪律：调用失败（超时/拒答/越界）默认降级为 MockLLM 填表（fail-closed，不阻塞管线）；
设 degrade_on_error=False 可改为硬失败（上线初期暴露问题用）。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from .mock_llm import (
    LLMDecision, CouncilContext, tool_spec, MockLLM, SchemaValidationError,
)

logger = logging.getLogger("cryptoquant.real_llm")


def _extract_json(text: str):
    """从模型 content 中提取 JSON 对象（兼容 ```json 围栏与前后夹杂文本）。

    仅用于 tool_calls 缺失时的兜底解析；提取出的 dict 仍须经 LLMDecision.validate()
    严格 schema 校验，锁表保证不破。
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
        raise


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
        self.base_url = base_url   # 存起来供 _call 判断是否需要关闭思考模式
        # 【P1-13 修复】降级可观测：记录最近一次降级原因，避免「静默降级」让运维误以为真 LLM 在线。
        self.degraded = False
        self.last_error: Optional[BaseException] = None

    # ---- 对外统一接口（与 MockLLM 一致）----
    def produce(self, ctx: CouncilContext) -> LLMDecision:
        try:
            return self._call(ctx)
        except Exception as e:
            if not self._degrade:
                raise
            # fail-closed：降级为确定性接地 mock，保证管线不中断
            self.degraded = True
            self.last_error = e
            logger.warning("RealLLM 调用失败，已降级 MockLLM（fail-closed）：%s", e)
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
        # DeepSeek V4/V4-Flash 默认启用思考模式，但思考模式下不支持强制 tool_choice
        # （本系统 lock-table 依赖 tool_choice 锁定 emit_trade_decision）。
        # 显式禁用思考模式使工具锁表正常；用 extra_body 透传以避免 openai SDK 参数校验拦截。
        extra_body: Optional[dict] = None
        if self.base_url and "deepseek" in self.base_url:
            extra_body = {"thinking": {"type": "disabled"}}
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
            extra_body=extra_body,
        )
        msg = resp.choices[0].message
        args = None
        if msg.tool_calls:
            args = msg.tool_calls[0].function.arguments
        elif msg.content and msg.content.strip():
            # 健壮性：推理/思考模型可能不吐 tool_calls，而在 content 返回 JSON。
            # 仍走同一严格 schema 校验，锁表保证不破（fail-closed）。
            args = _extract_json(msg.content)
        if not args:
            raise SchemaValidationError(
                "LLM 未返回工具调用且 content 非合法 JSON（未锁表）")
        if isinstance(args, str):
            args = json.loads(args)
        # 复用严格 schema 校验：越界直接拒（等价于 pydantic 失败）
        return LLMDecision(**args).validate()


def get_llm(degrade_on_error: bool = True):
    """零配置工厂（小白友好）：

    - 设了 CRYPTOQUANT_LLM_KEY 且 openai 可用 → 返回 RealLLM（真 LLM）
    - 可选环境变量切换 OpenAI 兼容供应商（如 DeepSeek）：
        CRYPTOQUANT_LLM_BASE_URL  e.g. https://api.deepseek.com（不设→OpenAI 官方）
        CRYPTOQUANT_LLM_MODEL     e.g. deepseek-chat（不设→gpt-4o-mini）
    - 没设密钥 / 缺 openai / 任何异常 → 返回 MockLLM（确定性接地 mock）

    调用方（四角色决策）无需判断，直接 get_llm().produce(ctx)。
    生产服务器设了密钥+base_url 就自动用对应真 LLM，沙箱/没密钥自动降级，零改代码。
    """
    try:
        key = os.getenv("CRYPTOQUANT_LLM_KEY")
        if not key:
            return MockLLM()
        base_url = os.getenv("CRYPTOQUANT_LLM_BASE_URL") or None
        model = os.getenv("CRYPTOQUANT_LLM_MODEL") or "gpt-4o-mini"
        return RealLLM(model=model, base_url=base_url, degrade_on_error=degrade_on_error)
    except Exception as e:
        logger.warning("RealLLM 初始化失败，回退 MockLLM：%s", e)
        return MockLLM()
