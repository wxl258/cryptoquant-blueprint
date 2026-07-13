"""四角色智能体（蓝图阶段3 · 任务18 · 专家B）。

复用 v26「三专家 + 逆向」架构，落地 FinMem 接地 LLM 四角色：
  ① Analyst（分析）     → 读特征/工作记忆，给市场态 + 驱动 + 初步信念
  ② Researcher（研究辩论）→ 从 FinMem 检索长期洞察，做支持/反对辩论并调置信
  ③ DecisionMaker（决策）→ 融合 Analyst+Researcher，用 MockLLM 填严格 schema 表
  ④ RiskController（风控）→ 宪法式硬锁：禁易 regime / 置信钳制 / 高不确定转 HOLD

FourRoleCouncil 编排整条管线，产出可解释 CouncilVerdict，并写盘 Episode 待回填。
LLM 全程只「填表」（见 adapters/mock_llm.tool_spec），不产自由文本——接地且可被 schema 守门。

零依赖纪律：仅 numpy + 包内 meta.memory / adapters.mock_llm。torch/LLM 不引入。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .memory import FinMemMemory, Episode
from ..adapters.mock_llm import MockLLM, CouncilContext, LLMDecision, ACTIONS
from ..adapters.real_llm import get_llm

# 与 stage2_features.FEATURE_NAMES 对齐（若包内变更，这里从源读取，避免硬编码漂移）
try:
    from ..stage2_features import FEATURE_NAMES
except Exception:  # 降级：写死顺序，保持可跑
    FEATURE_NAMES = ["adx", "rsi", "atr_pct", "vol_regime", "fr",
                     "fr_delta", "oi_pct", "fng", "momentum"]

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
LONG, SHORT, HOLD = "LONG", "SHORT", "HOLD"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================ ① Analyst ============================
class Analyst:
    """读特征 + 工作记忆 → 市场态 + 驱动 + 初步信念。"""

    def assess(self, symbol: str, feat: np.ndarray, regime: str,
               profile) -> Dict:
        adx = float(feat[_IDX["adx"]])
        mom = float(feat[_IDX["momentum"]])
        fng = float(feat[_IDX["fng"]])
        vreg = float(feat[_IDX["vol_regime"]])   # +1 扩张 / -1 收敛

        # 市场态：regime + 动量 + 恐慌贪婪 三角投票
        if regime == "CRASH":
            state = "CRASH"
        elif regime == "TREND":
            state = "BULL" if mom > 0.003 else ("BEAR" if mom < -0.003 else "RANGE")
        else:
            state = "RANGE"
        if state == "RANGE":
            if fng > 0.80:
                state = "BULL"
            elif fng < 0.20:
                state = "BEAR"

        # 初步信念：趋势强度 + 动量幅度，乘风险偏好缩放
        conv = 0.25 + 0.55 * adx + 0.40 * min(1.0, abs(mom) * 20.0)
        conv *= (0.5 + profile.risk_appetite)
        conv = _clamp(conv, 0.10, 0.95)

        drivers = [
            f"adx={adx:.2f}", f"momentum={mom:+.3f}", f"fng={fng:.2f}",
            f"vol_regime={vreg:+.0f}", f"regime={regime}",
        ]
        return {"market_state": state, "conviction": conv, "drivers": drivers,
                "adx": adx, "momentum": mom, "fng": fng, "vol_regime": vreg}


# ============================ ② Researcher ============================
class Researcher:
    """从 FinMem 检索长期洞察，做支持/反对辩论并调置信。"""

    def debate(self, symbol: str, feat: np.ndarray, regime: str,
               analyst_out: Dict, memory: FinMemMemory) -> Dict:
        mom = analyst_out["momentum"]
        fng = analyst_out["fng"]
        vreg = analyst_out["vol_regime"]
        fr = float(feat[_IDX["fr"]])

        support: List[str] = []
        contra: List[str] = []
        conf_adj = 0.0

        # 特征面辩论
        if vreg > 0 and abs(mom) > 0.003:
            support.append("波动率扩张配合方向，趋势延续概率高")
            conf_adj += 0.05
        if fng > 0.80:
            contra.append("恐慌贪婪极端贪婪，逆向回撤风险")
            conf_adj -= 0.10
        if fng < 0.20:
            contra.append("极端恐惧，流动性/踩踏风险")
            conf_adj -= 0.10
        if abs(fr) > 0.0005:
            contra.append(f"资金费率偏高({fr:+.4f})，持仓拥挤")
            conf_adj -= 0.05

        # 记忆面辩论：检索该 regime 的长期洞察
        insights = memory.retrieve(regime=regime, k=3)
        for ins in insights:
            if ins.weight >= 0.5 and "胜率" in ins.text:
                support.append(f"记忆: {ins.text}")
                conf_adj += 0.03 * ins.weight
            elif ins.weight < 0.5 and "胜率" in ins.text:
                contra.append(f"记忆: {ins.text}")
                conf_adj -= 0.05 * (1 - ins.weight)

        return {"support": support, "contra": contra,
                "conf_adj": conf_adj,
                "retrieved_insights": [i.text for i in insights]}


# ============================ ③ DecisionMaker ============================
class DecisionMaker:
    """融合 Analyst+Researcher，用 MockLLM 填严格 schema 表（接地 LLM）。"""

    def __init__(self, llm: MockLLM):
        self.llm = llm

    def propose(self, symbol: str, feat: np.ndarray, regime: str,
                analyst_out: Dict, researcher_out: Dict,
                memory: FinMemMemory, spi_surprise: float = 0.0) -> LLMDecision:
        # 融合方向：动量 × 趋势强度
        lean = analyst_out["momentum"]
        if analyst_out["adx"] > 0.25:
            lean *= (1.0 + analyst_out["adx"])
        thr = 0.004
        if lean > thr:
            fused = LONG
        elif lean < -thr:
            fused = SHORT
        else:
            fused = HOLD

        base_conf = _clamp(analyst_out["conviction"] + researcher_out["conf_adj"],
                           0.0, 1.0)
        ctx = CouncilContext(
            symbol=symbol, regime=regime,
            market_state=analyst_out["market_state"],
            fused_action=fused, base_confidence=base_conf,
            support=researcher_out["support"],
            contra=researcher_out["contra"],
            retrieved_insights=researcher_out["retrieved_insights"],
            spi_surprise=spi_surprise,
        )
        # LLM 只填表（确定性接地 + 严格 schema 校验）
        return self.llm.produce(ctx)


# ============================ ④ RiskController ============================
class RiskController:
    """宪法式硬锁：禁易 regime / 置信钳制 / 高不确定转 HOLD。"""

    def guard(self, dec: LLMDecision, profile, spi_surprise: float = 0.0
              ) -> (str, List[str]):
        vetoes: List[str] = []
        action = dec.proposed_action
        conf = dec.confidence

        # 禁易 regime（反思自改进写入 Profile）
        if dec.market_state in profile.forbidden_regimes:
            action = HOLD
            vetoes.append(f"forbidden_regime:{dec.market_state}")
        # 置信低于最小信念 → 软降级 HOLD
        if conf < profile.min_conviction:
            if action != HOLD:
                vetoes.append("low_conviction")
            action = HOLD
        # SPCI 高惊喜度（高不确定）→ 转 HOLD
        if spi_surprise > 0.6 and action != HOLD:
            vetoes.append("high_uncertainty")
            action = HOLD
        # 置信钳制到上限
        conf = _clamp(conf, 0.0, profile.max_confidence)
        return action, vetoes, conf


# ============================ 编排 ============================
@dataclass
class CouncilVerdict:
    symbol: str
    regime: str
    market_state: str
    action: str                      # 风控后最终 LONG/SHORT/HOLD
    confidence: float
    rationale: List[str]
    vetoes: List[str]
    llm_decision: Dict
    analyst_drivers: List[str]

    def direction_int(self) -> int:
        return 1 if self.action == LONG else (-1 if self.action == SHORT else 0)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "regime": self.regime,
            "market_state": self.market_state, "action": self.action,
            "confidence": round(self.confidence, 4),
            "rationale": list(self.rationale),
            "vetoes": list(self.vetoes),
            "llm": self.llm_decision,
        }


class FourRoleCouncil:
    """四角色编排器：分析→研究辩论→决策(LLM 填表)→风控。"""

    def __init__(self, memory: FinMemMemory, llm: Optional[MockLLM] = None,
                 profile=None):
        self.memory = memory
        self.profile = profile or memory.profile
        self.llm = llm or get_llm()
        self.analyst = Analyst()
        self.researcher = Researcher()
        self.decider = DecisionMaker(self.llm)

    def decide(self, symbol: str, feat: np.ndarray, regime: str,
               spi_surprise: float = 0.0, record: bool = True
               ) -> CouncilVerdict:
        a = self.analyst.assess(symbol, feat, regime, self.profile)
        r = self.researcher.debate(symbol, feat, regime, a, self.memory)
        dec = self.decider.propose(symbol, feat, regime, a, r, self.memory,
                                    spi_surprise=spi_surprise)
        action, vetoes, conf = RiskController().guard(
            dec, self.profile, spi_surprise=spi_surprise)

        verdict = CouncilVerdict(
            symbol=symbol, regime=regime,
            market_state=dec.market_state, action=action,
            confidence=conf, rationale=list(dec.rationale),
            vetoes=vetoes, llm_decision=dec.to_dict(),
            analyst_drivers=a["drivers"],
        )
        if record:
            # 写盘情景记忆（待平仓后 set_outcome 回填）
            self.memory.record_decision(Episode(
                ts=time.time(), symbol=symbol, regime=regime,
                decision=action, confidence=conf,
                rationale=verdict.rationale,
            ))
        return verdict
