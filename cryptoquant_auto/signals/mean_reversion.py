"""均值回归信号（吸收生产系统 signals/mean_reversion.py 优点 P0-B）。

逻辑：ADX<20 启动均值回归；资金费率绝对值 > 阈值(0.03%) 视为极端，
反向入场捕捉费率回归。与趋势引擎互补（震荡市友好）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .indicators import calc_adx, calc_rsi
from .engine import SignalCandidate, MarketContext

FR_EXTREME = 0.0003  # 资金费率极端阈值 0.03%
# P0-1 修复：与 signals/engine.py RC["ext_fr"] 对齐的 fr 门控阈值
FR_GATE = 0.001       # 0.1%：做多 fr<-0.001 / 做空 fr>+0.001 → BLOCK（极负费率做多陷阱）


@dataclass
class MeanReversionSignal:
    symbol: str
    direction: str = "观望"
    score: float = 0.0
    conds: List[str] = field(default_factory=list)
    fr: float = 0.0

    @property
    def triggered(self) -> bool:
        return self.direction != "观望" and self.score >= 6


def gen_mean_reversion(symbol: str, candles_1h: List[dict],
                       fr: float = 0.0, ctx: MarketContext = None) -> MeanReversionSignal:
    """ADX<20 且费率极端时，反向均值回归信号。"""
    ctx = ctx or MarketContext()
    sig = MeanReversionSignal(symbol=symbol, fr=fr)
    adx, _, _ = calc_adx(candles_1h)
    rsi = calc_rsi([x["c"] for x in candles_1h])

    if adx >= 20:
        sig.conds.append(f"ADX={adx}≥20 趋势市，均值回归不触发")
        return sig

    # ADX<20 震荡市：费率极端 → 反向
    if abs(fr) > FR_EXTREME:
        # 极正费率：做空拥挤 → 反向做空（费率回归）
        # 极负费率：做多拥挤 → 反向做多
        # P0-1 修复：fr 门控（与趋势引擎一致）—— 极端费率方向 = 陷阱，禁止反向
        if fr > 0:
            if fr > FR_GATE:
                sig.conds.append(f"费率={fr*100:.3f}% 极正费率做空陷阱→不触发")
                return sig
            sig.direction = "做空"
            sig.score = 6 + min(2, int(abs(fr) / FR_EXTREME))
            sig.conds.append(f"费率={fr*100:.3f}% 极正(做多拥挤) → 反向做空 score={sig.score}")
        else:
            if fr < -FR_GATE:
                sig.conds.append(f"费率={fr*100:.3f}% 极负费率做多陷阱→不触发")
                return sig
            sig.direction = "做多"
            sig.score = 6 + min(2, int(abs(fr) / FR_EXTREME))
            sig.conds.append(f"费率={fr*100:.3f}% 极负(做空拥挤) → 反向做多 score={sig.score}")
        if rsi > 70:
            sig.score += 1; sig.conds.append(f"RSI={rsi} 超买强化做空")
        elif rsi < 30:
            sig.score += 1; sig.conds.append(f"RSI={rsi} 超卖强化做多")
    else:
        sig.conds.append(f"ADX={adx}<20 但费率={fr*100:.3f}% 未达极端(>{FR_EXTREME*100:.2f}%)，不触发")
    return sig
