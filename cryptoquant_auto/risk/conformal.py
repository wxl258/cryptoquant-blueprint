"""序列共形预测（SPCI，蓝图阶段0.5 不确定性层）。

用途：给「决策/ posterior」配一个**时依赖自适应**的预测区间，作为
metacontroller 软降级的不确定度来源（取代朴素熵）。核心性质：
  - 在线更新：每个决策后喂入一个 conformality score（如 posterior 熵），
    区间随近期分布自适应收缩/扩张；
  - 时依赖：近因权重更高（decay），捕捉 regime 漂移，比朴素 CP 更稳；
  - 降级路径：naive=True 退化为等权经验分位（无 decay），保证可复现。

纯 numpy，零外部依赖。接口与 metacontroller 解耦：只认 `update(score)` /
`predict_interval()` / `surprise(score)` / `coverage(scores)`。
"""
from __future__ import annotations

import math
from typing import List, Tuple


class SequentialConformalPredictor:
    def __init__(self, alpha: float = 0.10, maxlen: int = 500,
                 decay: float = 0.985, naive: bool = False):
        """
        alpha  : 目标误覆盖率（1-alpha 为置信水平，默认 90% 区间）
        maxlen : 滑窗容量（超限丢弃最旧样本）
        decay  : 近因衰减（<1；越接近 1 越重近期）；naive=True 时忽略
        naive  : True → 等权经验分位（朴素 CP 降级路径）
        """
        self.alpha = alpha
        self.maxlen = maxlen
        self.decay = decay
        self.naive = naive
        self._buf: List[float] = []   # 仅存 score，权重在分位时按「距现在位置」现算

    # ---- 在线更新 ----
    def update(self, score: float) -> None:
        self._buf.append(float(score))
        if len(self._buf) > self.maxlen:
            self._buf.pop(0)

    # ---- 加权分位（近因权重更高：位置越新权重越大）----
    def _weighted_quantile(self, q: float) -> float:
        buf = self._buf
        n = len(buf)
        if n == 0:
            return 0.0
        if self.naive:
            ordered = sorted(buf)
            idx = min(n - 1, max(0, int(q * n)))
            return ordered[idx]
        # (score, 近因权重)；最新样本(i=n-1)权重=decay^0=1，最旧=decay^(n-1)
        pairs = sorted((buf[i], self.decay ** (n - 1 - i)) for i in range(n))
        cum = [0.0]
        for _, w in pairs:
            cum.append(cum[-1] + w)
        tot = cum[-1]
        if tot <= 0:
            idx = min(n - 1, max(0, int(q * n)))
            return pairs[idx][0]
        target = q * tot
        for i in range(n):
            if cum[i + 1] >= target:
                return pairs[i][0]
        return pairs[-1][0]

    # ---- 预测区间 ----
    def predict_interval(self) -> Tuple[float, float]:
        lo = self._weighted_quantile(self.alpha / 2.0)
        hi = self._weighted_quantile(1.0 - self.alpha / 2.0)
        return lo, hi

    # ---- 惊喜度（不确定度代理）----
    def surprise(self, score: float) -> float:
        """0 = 落在区间内；>0 = 落在区间外，按半宽归一化（越大越异常/越不确定）。"""
        lo, hi = self.predict_interval()
        if hi <= lo:
            return 0.0
        if score < lo:
            return (lo - score) / (hi - lo)
        if score > hi:
            return (score - hi) / (hi - lo)
        return 0.0

    # ---- 经验覆盖率（监控用）----
    def coverage(self, scores) -> float:
        lo, hi = self.predict_interval()
        n = len(scores)
        if n == 0:
            return 1.0
        inside = sum(1 for s in scores if lo <= s <= hi)
        return inside / n

    @property
    def n_samples(self) -> int:
        return len(self._buf)
