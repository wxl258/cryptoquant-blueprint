"""信号质量前置过滤器（移植服务器 多周期共振/ADX-DI/ATR区间/FG对齐 优点）。

放在四闸门**之前**：低质信号根本不进 assert_pre_trade。
调试期无多周期/ADX/FG 数据时，缺省安全降级（只跑能用 sig.atr 算的检查）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..models import Signal, Direction


@dataclass
class TfView:
    """单周期视角（来自真实数据源时填入）。"""
    tf: str
    direction: Direction
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0


@dataclass
class MarketSnapshot:
    """单 symbol 行情上下文；调试期可合成。"""
    symbol: str
    tf_views: List[TfView] = field(default_factory=list)
    atr_pct: float = 0.0       # ATR/entry
    fg: int = 50               # 恐慌贪婪指数 0~100
    ts: float = 0.0


@dataclass
class SignalQualityConfig:
    # 多周期共振
    resonance_tfs: List[str] = field(default_factory=lambda: ["1H", "4H", "1D", "W"])
    min_agree_tfs: int = 2
    # ADX/DI 趋势强度
    adx_min: float = 30.0
    di_diff_min: float = 20.0
    # ATR 区间禁入（atr_pct=ATR占价百分比，0-100 量纲，1.0=1%，与 calc_atr/engine 一致）
    atr_pct_min: float = 0.3     # <0.3% 无趋势
    atr_pct_max: float = 9.0     # >9% 波动过高
    # FG 情绪对齐
    fg_extreme_low: int = 20     # 恐慌极值不追空
    fg_extreme_high: int = 80    # 贪婪极值不追多


@dataclass
class FilterResult:
    ok: bool = True
    reasons: List[str] = field(default_factory=list)

    def fail(self, r: str) -> None:
        self.ok = False
        self.reasons.append(r)


_RANK = {"1H": 0, "4H": 1, "1D": 2, "W": 3}


class SignalQualityGate:
    def __init__(self, cfg: SignalQualityConfig = None):
        self.cfg = cfg or SignalQualityConfig()

    def check(self, sig: Signal, snap: Optional[MarketSnapshot] = None) -> FilterResult:
        res = FilterResult()
        self._atr(sig, snap, res)
        if not res.ok:
            return res
        if snap is None:
            return res  # 无上下文：仅跑 ATR（最稳的检查）
        self._resonance(sig, snap, res)
        self._trend(sig, snap, res)
        self._fg(sig, snap, res)
        return res

    def _atr(self, sig, snap, res: FilterResult) -> None:
        atr_pct = snap.atr_pct if snap else (sig.atr / sig.entry * 100 if sig.entry else 0.0)
        if atr_pct < self.cfg.atr_pct_min:
            res.fail("atr_flat")          # 波动率过低 = 无趋势
        elif atr_pct > self.cfg.atr_pct_max:
            res.fail("atr_risky")         # 波动率过高 = 风险

    def _resonance(self, sig, snap, res: FilterResult) -> None:
        views = {v.tf: v for v in snap.tf_views if v.tf in self.cfg.resonance_tfs}
        if not views:
            return
        agrees = sum(1 for v in views.values() if v.direction == sig.direction)
        higher = [t for t in self.cfg.resonance_tfs
                  if _RANK.get(t, 0) > _RANK.get(sig.tf, 0) and t in views]
        if agrees < self.cfg.min_agree_tfs or \
           not any(views[t].direction == sig.direction for t in higher):
            res.fail("mtf_resonance")

    def _trend(self, sig, snap, res: FilterResult) -> None:
        v = next((x for x in snap.tf_views if x.tf == sig.tf), None)
        if v is None:
            return
        if v.adx < self.cfg.adx_min:
            res.fail("adx_weak")
        elif abs(v.plus_di - v.minus_di) < self.cfg.di_diff_min:
            res.fail("di_flat")

    def _fg(self, sig, snap, res: FilterResult) -> None:
        if sig.direction is Direction.SHORT and snap.fg <= self.cfg.fg_extreme_low:
            res.fail("fg_panic_no_short")
        elif sig.direction is Direction.LONG and snap.fg >= self.cfg.fg_extreme_high:
            res.fail("fg_greed_no_long")


def synth_snapshot(sig: Signal, fg: int = 50, noise: float = 0.0) -> MarketSnapshot:
    """调试期合成快照：默认各周期与信号同向（让信号通过共振/趋势检查）。"""
    import random
    rnd = random.Random(int(sig.entry) + len(sig.signal_id))
    views = []
    for tf in ["1H", "4H", "1D", "W"]:
        adx = 35.0 + rnd.uniform(-5, 15) + noise * 30
        spread = 25.0 + rnd.uniform(-5, 10)
        views.append(TfView(tf=tf, direction=sig.direction,
                            adx=adx, plus_di=spread, minus_di=spread - 10))
    atr_pct = sig.atr / sig.entry * 100 if sig.entry else 0.01
    return MarketSnapshot(symbol=sig.symbol, tf_views=views, atr_pct=atr_pct, fg=fg)
