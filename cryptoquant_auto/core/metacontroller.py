"""概率化贝叶斯脊柱（蓝图阶段1）：多源意见 → 贝叶斯融合 → 不确定性 → 软降级。

把现有 gen_signal 的硬规则输出升级为"分布"，多个专家/模型意见在
对数空间相乘（独立假设 = 贝叶斯似然相乘）得到后验。后验的熵即不确定性，
高不确定 → 软降级为观望（而非硬禁）。原生带可解释（来源贡献）与不确定性。

原型沙盒专用，零依赖（numpy）。后续 Causal DSL / World Model 都挂在这一层之上。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..signals.engine import SignalCandidate

LONG, SHORT, HOLD = 0, 1, 2
ACTION_NAMES = {LONG: "做多", SHORT: "做空", HOLD: "观望"}


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


@dataclass
class Opinion:
    """单个专家/模型对某标的的意见：3 向分布 + 置信 + 依据。"""
    symbol: str
    probs: np.ndarray            # 3-vector over LONG/SHORT/HOLD
    source: str = "expert"
    confidence: float = 0.5
    rationale: str = ""

    def action(self) -> int:
        return int(np.argmax(self.probs))


@dataclass
class MetaDecision:
    symbol: str
    action: int
    action_name: str
    confidence: float
    uncertainty: float
    rationale: str
    sources: List[str] = field(default_factory=list)
    degraded: bool = False
    constitution_ok: bool = True
    violations: List[str] = field(default_factory=list)
    proposed_exposure: Optional[float] = None
    spi_conf: bool = False            # 不确定度是否由 SPCI 提供（否=朴素熵）


def opinion_from_candidate(c: SignalCandidate, source: str = "rule_engine") -> Opinion:
    """把现有 gen_signal 的 SignalCandidate 转成贝叶斯意见（3 向分布 + 置信）。

    映射：规则命中的方向得 score 加权；未通过(passed=False)则压低并偏向观望；
    置信由 (score - min_score_adj) 边际决定；依据取 conds 末几条。
    """
    s = np.array([1.0, 1.0, 1.0], dtype=float)
    if c.direction == "做多":
        s[LONG] += c.score
    elif c.direction == "做空":
        s[SHORT] += c.score
    else:
        s[HOLD] += c.score
    if not c.passed:
        s *= 0.5
        s[HOLD] += c.min_score_adj
    temp = 4.0
    probs = _softmax(s / temp)
    margin = c.score - c.min_score_adj
    conf = float(np.clip(0.5 + margin / 6.0, 0.05, 0.95))
    rationale = "; ".join(c.conds[-3:]) if c.conds else ""
    return Opinion(symbol=c.symbol, probs=probs, source=source,
                   confidence=conf, rationale=rationale)


class BayesianMetacontroller:
    def __init__(self, uncertainty_thresh: float = 0.55,
                 prior: Optional[np.ndarray] = None, conformal=None):
        self.uncertainty_thresh = uncertainty_thresh
        # 默认均匀先验；后续可由 regime 驱动（TREND 略偏向动作、RANGE 偏向观望）
        self.prior = prior if prior is not None else np.array([1.0, 1.0, 1.0])
        # 阶段0.5：可选 SPCI 预测器（risk/conformal.SequentialConformalPredictor）。
        # 接入后用「conformity score（posterior 熵）的时依赖区间惊喜度」作为不确定度，
        # 取代朴素熵；未接入则退回熵归一化（向后兼容）。
        self.conformal = conformal

    @staticmethod
    def entropy(p):
        p = np.asarray(p, dtype=float) + 1e-9
        return float(-(p * np.log(p)).sum())

    def set_conformal(self, predictor) -> None:
        """接入 SPCI 预测器（蓝图阶段1 第9条：熵 → SPCI）。"""
        self.conformal = predictor

    def _uncertainty(self, fused: np.ndarray, ent: float) -> float:
        """统一不确定度：有 SPCI 用惊喜度，否则用熵归一化。"""
        if self.conformal is not None and self.conformal.n_samples >= 5:
            # 用历史 posterior 熵分布预测当前熵的异常度（时依赖）
            return float(np.clip(self.conformal.surprise(ent), 0.0, 1.0))
        return float(np.clip(ent / np.log(3), 0.0, 1.0))

    def fuse(self, opinions: List[Opinion]) -> np.ndarray:
        """贝叶斯融合：log(prior) + Σ log(opinion) → softmax。"""
        logp = np.log(np.asarray(self.prior, dtype=float))
        for o in opinions:
            logp = logp + np.log(np.asarray(o.probs, dtype=float) + 1e-9)
        return _softmax(logp)

    def decide(self, opinions: List[Opinion], symbol: str = "BTC",
                proposed_exposure: Optional[float] = None) -> MetaDecision:
        fused = self.fuse(opinions)
        action = int(np.argmax(fused))
        conf = float(fused[action])
        ent = self.entropy(fused)                 # conformity score：posterior 熵
        norm_unc = self._uncertainty(fused, ent)
        # SPCI 在线学习：先以「历史分布」评估当前，再把当前喂入（留一预测，避免自污染）
        if self.conformal is not None:
            self.conformal.update(ent)
        degraded = False
        # 软降级：高不确定且非观望 → 降级观望（不硬禁，保留信息）
        if norm_unc > self.uncertainty_thresh and action != HOLD:
            action = HOLD
            degraded = True
            conf = float(fused[HOLD])
        rationale = " | ".join(f"[{o.source}] {o.rationale}"
                               for o in opinions if o.rationale)
        return MetaDecision(
            symbol=symbol, action=action, action_name=ACTION_NAMES[action],
            confidence=conf, uncertainty=norm_unc, rationale=rationale,
            sources=[o.source for o in opinions], degraded=degraded,
            proposed_exposure=proposed_exposure,
            spi_conf=self.conformal is not None,
        )
