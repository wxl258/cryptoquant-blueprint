"""接地 LLM · 严格 JSON schema + Function Calling（蓝图阶段3 · 任务17 · 专家B）。

设计要点（零依赖纪律）：
  - 不引 pydantic / 不接真实 LLM API（原型沙盒禁网 + 零依赖）。用纯 stdlib 实现
    与 Pydantic 同契约的严格 schema：LLMDecision（必需字段 + 类型 + 取值域 + 边界）。
  - 「Function Calling」= 把 LLM 锁死在一个 function/tool 的 JSON schema 里：
    LLM 只能**填表**，不能产自由文本。tool_spec() 直接返回 OpenAI 风格的 `tools`
    条目，未来接真实 LLM 时原样传给 client 即可。
  - MockLLM.complete()：由四角色上下文**确定性接地**地填表（非随机），并强制 validate()，
    任何违约都抛错（模拟 pydantic 校验失败 → 逼出 bug）。这是「LLM 只填表」的诚实实现。
  - 降级路径：call_real_llm() 在沙盒抛 NotImplementedError；接真实 LLM（阶段3-4 测试网/
    云）时只需替换 MockLLM 为真实 client，仍走同一 LLMDecision 接口与 tool_spec。

零依赖：仅 dataclasses + 手动校验。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 取值域（与 metacontroller 的 BULL/BEAR/RANGE/CRASH 与 LONG/SHORT/HOLD 对齐）
MARKET_STATES = ("BULL", "BEAR", "RANGE", "CRASH")
ACTIONS = ("LONG", "SHORT", "HOLD")


class SchemaValidationError(ValueError):
    """模拟 pydantic ValidationError：schema 违约时抛出。"""


@dataclass
class LLMDecision:
    """LLM 被允许产出的唯一结构（Function Calling 的表）。

    字段（严格契约）：
      market_state    : 市场态，取值 MARKET_STATES
      confidence      : 置信度 [0,1]
      rationale       : 依据列表，≥1 条，每条为 str
      proposed_action : 提案动作，取值 ACTIONS
    接地扩展字段（不影响 schema 校验，仅供可解释/回填）：
      source_regimes  : 检索到的相关 regime 标签
      retrieved_insights: 来自 FinMem 的长期洞察文本
    """
    market_state: str
    confidence: float
    rationale: List[str]
    proposed_action: str
    source_regimes: List[str] = field(default_factory=list)
    retrieved_insights: List[str] = field(default_factory=list)

    # 【P1-14 修复】schema 校验强制化：任何构造（含 LLM/反序列化）都必经校验，
    # 不再依赖调用方手动 produce() 才触发——杜绝绕过校验的脏数据进入风控。
    def __post_init__(self) -> "LLMDecision":
        self.validate()
        return self

    # ---- 严格 schema 校验（零依赖，等价于 pydantic model_validator）----
    def validate(self) -> "LLMDecision":
        if self.market_state not in MARKET_STATES:
            raise SchemaValidationError(
                f"market_state 越界: {self.market_state!r} ∉ {MARKET_STATES}")
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= float(self.confidence) <= 1.0):
            raise SchemaValidationError(
                f"confidence 越界: {self.confidence!r} 必须在 [0,1]")
        if not isinstance(self.rationale, list) or len(self.rationale) < 1:
            raise SchemaValidationError("rationale 至少 1 条")
        for r in self.rationale:
            if not isinstance(r, str) or not r.strip():
                raise SchemaValidationError(f"rationale 条目须为非空 str: {r!r}")
        if self.proposed_action not in ACTIONS:
            raise SchemaValidationError(
                f"proposed_action 越界: {self.proposed_action!r} ∉ {ACTIONS}")
        # 归一化
        self.market_state = str(self.market_state)
        self.proposed_action = str(self.proposed_action)
        self.confidence = float(round(self.confidence, 4))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_state": self.market_state,
            "confidence": self.confidence,
            "rationale": list(self.rationale),
            "proposed_action": self.proposed_action,
            "source_regimes": list(self.source_regimes),
            "retrieved_insights": list(self.retrieved_insights),
        }


def tool_spec() -> Dict[str, Any]:
    """OpenAI 风格 `tools` 条目：LLM 唯一可调用函数（Function Calling 锁表）。

    未来接真实 LLM：把返回值原样塞进 client.chat.completions.create(tools=[tool_spec()])。
    """
    return {
        "type": "function",
        "function": {
            "name": "emit_trade_decision",
            "description": "产出结构化交易决策。LLM 只能填此表，不产自由文本。",
            "parameters": {
                "type": "object",
                "required": ["market_state", "confidence", "rationale", "proposed_action"],
                "properties": {
                    "market_state": {
                        "type": "string",
                        "enum": list(MARKET_STATES),
                        "description": "当前市场状态",
                    },
                    "confidence": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0,
                        "description": "提案置信度 [0,1]",
                    },
                    "rationale": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "决策依据，至少 1 条",
                    },
                    "proposed_action": {
                        "type": "string",
                        "enum": list(ACTIONS),
                        "description": "提案动作 LONG/SHORT/HOLD",
                    },
                },
            },
        },
    }


@dataclass
class CouncilContext:
    """四角色喂给 LLM 的上下文（LLM 据此填表，不接触原始日志）。"""
    symbol: str
    regime: str
    market_state: str                 # Analyst 给出的市场态
    fused_action: str                 # DecisionMaker 融合方向 LONG/SHORT/HOLD
    base_confidence: float            # DecisionMaker 基础置信
    support: List[str]                # Researcher 支持论据
    contra: List[str]                 # Researcher 反对论据
    retrieved_insights: List[str]     # 来自 FinMem 的长期洞察
    spi_surprise: float = 0.0         # SPCI 惊喜度（越高越不确定）


class MockLLM:
    """确定性接地 mock：把四角色上下文映射为严格 schema 的 LLMDecision。

    行为刻意确定（非随机），让受控 A/B 可复现；且严格调用 validate() 守门。
    真实 LLM 接入时，子类/替换只用同一接口 produce(context)->LLMDecision。
    """

    def produce(self, ctx: CouncilContext) -> LLMDecision:
        # 1) 市场态来自 Analyst 投票（已在取值域内）
        market_state = ctx.market_state if ctx.market_state in MARKET_STATES else "RANGE"

        # 2) 动作来自 DecisionMaker 融合方向
        action = ctx.fused_action if ctx.fused_action in ACTIONS else "HOLD"

        # 3) 置信：基础置信 − SPCI 惊喜度（越高越不确定）→ 越不确定越降置信
        conf = float(ctx.base_confidence) - 0.5 * float(ctx.spi_surprise)
        # 反对论据越多越降置信
        conf -= 0.08 * max(0, len(ctx.contra) - len(ctx.support))
        conf = max(0.0, min(1.0, conf))

        # 4) rationale：支持/反对 + 检索洞察，截断到 4 条，确保 ≥1
        rationale: List[str] = []
        for s in ctx.support[:2]:
            rationale.append(f"支持: {s}")
        for c in ctx.contra[:1]:
            rationale.append(f"反对: {c}")
        for ins in ctx.retrieved_insights[:1]:
            rationale.append(f"记忆: {ins}")
        if not rationale:
            rationale.append(f"规则融合指向 {action}（无显著分歧）")

        dec = LLMDecision(
            market_state=market_state,
            confidence=round(conf, 4),
            rationale=rationale[:4],
            proposed_action=action,
            source_regimes=[ctx.regime],
            retrieved_insights=list(ctx.retrieved_insights[:3]),
        )
        # 严格 schema 校验：任何违约都抛（模拟 pydantic 失败 → 逼出 bug）
        return dec.validate()

    # ---- 降级路径：真实 LLM 钩子（沙盒禁用）----
    def call_real_llm(self, ctx: CouncilContext, api_key: Optional[str] = None):
        """接真实 LLM 的位置（阶段3-4 测试网/云）。沙盒内禁用。

        真实实现应：用 tool_spec() 约束 client，把 ctx 拼成 prompt，解析返回填表。
        原型期一律抛 NotImplementedError，强制走 MockLLM，避免任何外网/密钥依赖。
        """
        raise NotImplementedError(
            "沙盒禁用真实 LLM API；请使用 MockLLM（确定性接地）。"
            "接真实 LLM 须在阶段3-4 测试网/云环境替换本方法，且必须复用 tool_spec() 锁表。")

    # 便捷别名（与 Stage 0.5 其它 harness 命名一致）
    def complete(self, ctx: CouncilContext) -> LLMDecision:
        return self.produce(ctx)
