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


# 被因果发现剔除的特征 → 中性默认值（数值上「不贡献信号」，而不是 KeyError）。
# 语义：某特征未通过 Granger 稳定性筛选，就让它对议会数值逻辑保持中性，
# 从而真正降低噪声、聚焦因果特征，而非粗暴删列导致索引越界。
_FEATURE_DEFAULTS = {
    "adx": 0.0, "rsi": 0.0, "atr_pct": 0.0, "vol_regime": 0.0,
    "fr": 0.0, "fr_delta": 0.0, "oi_pct": 0.0, "fng": 0.5, "momentum": 0.0,
}


def _fget(feat, name: str, active, default: Optional[float] = None) -> float:
    """安全读取特征：name 在被激活集合内 → 取真实值；否则返回中性默认值。

    active 为「本次议会生效的特征名集合」（因果发现筛选后的子集）。
    """
    if name in active:
        return float(feat[_IDX[name]])
    return _FEATURE_DEFAULTS.get(name, default if default is not None else 0.0)


# ============================ ① Analyst ============================
class Analyst:
    """读特征 + 工作记忆 → 市场态 + 驱动 + 初步信念。"""

    def assess(self, symbol: str, feat: np.ndarray, regime: str,
               profile, active=None, forecast=None) -> Dict:
        active = active or set(FEATURE_NAMES)
        adx = _fget(feat, "adx", active)
        mom = _fget(feat, "momentum", active)
        fng = _fget(feat, "fng", active)
        vreg = _fget(feat, "vol_regime", active)   # +1 扩张 / -1 收敛

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
                "adx": adx, "momentum": mom, "fng": fng, "vol_regime": vreg,
                "forecast": forecast}


# ============================ ② Researcher ============================
class Researcher:
    """从 FinMem 检索长期洞察，做支持/反对辩论并调置信。"""

    def debate(self, symbol: str, feat: np.ndarray, regime: str,
               analyst_out: Dict, memory: FinMemMemory, active=None,
               forecast=None) -> Dict:
        active = active or set(FEATURE_NAMES)
        mom = analyst_out["momentum"]
        fng = analyst_out["fng"]
        vreg = analyst_out["vol_regime"]
        fr = _fget(feat, "fr", active)

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
        # TSFM 预报与动量背离 → 方向信号打架，不确定性上升
        if forecast is not None and abs(forecast) > 1e-4 and mom * forecast < 0:
            contra.append(f"TSFM 预报({forecast:+.4f})与动量背离，不确定性上升")
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
                memory: FinMemMemory, spi_surprise: float = 0.0,
                forecast=None) -> LLMDecision:
        # 融合方向：动量 × 趋势强度
        lean = analyst_out["momentum"]
        if analyst_out["adx"] > 0.25:
            lean *= (1.0 + analyst_out["adx"])
        # TSFM 预报作为第 10 路方向信号，与动量加权融合（无预报则退化为原行为）
        base_extra = 0.0
        if forecast is not None:
            lean_fc = _clamp(forecast * 20.0, -1.0, 1.0)   # 对数收益同量级缩放
            lean = 0.6 * lean + 0.4 * lean_fc
            # 方向一致 → 置信微增；背离 → 微减（不确定性）
            base_extra = 0.05 if analyst_out["momentum"] * forecast >= 0 else -0.05
        thr = 0.004
        if lean > thr:
            fused = LONG
        elif lean < -thr:
            fused = SHORT
        else:
            fused = HOLD

        base_conf = _clamp(analyst_out["conviction"] + researcher_out["conf_adj"]
                           + base_extra, 0.0, 1.0)
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

    def guard(self, dec: LLMDecision, profile, spi_surprise: float = 0.0,
              regime: Optional[str] = None) -> (str, List[str], float):
        vetoes: List[str] = []
        action = dec.proposed_action
        conf = dec.confidence

        # 禁易 regime（反思自改进写入 Profile）。【P0-3 修复】统一命名空间：
        # reflect() 把禁易键存为原始 regime（如 "TREND"），而本方法旧版只比对
        # dec.market_state（BULL/BEAR/RANGE/CRASH）→ "TREND" 永不等于任何 market_state，
        # 禁易锁对非 CRASH 永不触发（仅在 CRASH 巧合生效）。现同时比对原始 regime。
        if (dec.market_state in profile.forbidden_regimes
                or (regime is not None and regime in profile.forbidden_regimes)):
            action = HOLD
            vetoes.append(f"forbidden_regime:{dec.market_state}/{regime}")
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
    decision_id: str = ""            # 【P0-4】关联短期记忆 Episode，供 set_outcome 精确回填

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
               spi_surprise: float = 0.0, record: bool = True,
               conformal=None, feature_names=None, forecast=None) -> CouncilVerdict:
        # feature_names：因果发现筛选后的生效特征子集；None → 全量（向后兼容）
        # forecast：TSFM 单步点预测（log return），None → 不引入预报信号（向后兼容）
        active = set(feature_names) if feature_names else set(FEATURE_NAMES)
        a = self.analyst.assess(symbol, feat, regime, self.profile, active=active,
                                forecast=forecast)
        r = self.researcher.debate(symbol, feat, regime, a, self.memory, active=active,
                                   forecast=forecast)
        dec = self.decider.propose(symbol, feat, regime, a, r, self.memory,
                                   spi_surprise=spi_surprise, forecast=forecast)
        # 初筛：先以 spi=0 跑一次风控门，得到本轮置信（供 conformal 评估）
        action, vetoes, conf = RiskController().guard(
            dec, self.profile, spi_surprise=0.0, regime=regime)
        # 【P1-10 修复】SPCI 在线惊喜度：用跨 tick 历史置信分布自适应评估
        # 「本轮置信是否异常」。先以 prior 窗口算惊喜度（留一，避免自污染），
        # 再喂入本轮 score 供下轮使用；最后用真实 spi 重跑风控门使软降级生效。
        spi = spi_surprise
        if conformal is not None:
            score = 1.0 - float(conf)
            if conformal.n_samples >= 5:
                spi = float(conformal.surprise(score))
            conformal.update(score)
            if spi != 0.0:
                action, vetoes, conf = RiskController().guard(
                    dec, self.profile, spi_surprise=spi, regime=regime)

        verdict = CouncilVerdict(
            symbol=symbol, regime=regime,
            market_state=dec.market_state, action=action,
            confidence=conf, rationale=list(dec.rationale),
            vetoes=vetoes, llm_decision=dec.to_dict(),
            analyst_drivers=a["drivers"],
        )
        if record:
            # 写盘情景记忆（待平仓后 set_outcome 精确回填）；返回 decision_id 供上层关联
            did = self.memory.record_decision(Episode(
                ts=time.time(), symbol=symbol, regime=regime,
                decision=action, confidence=conf,
                rationale=verdict.rationale,
            ))
            verdict.decision_id = did
        return verdict
