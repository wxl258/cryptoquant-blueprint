"""信号生成器：把信号引擎候选（趋势/均值回归）桥接为原型 Signal。

吸收生产系统"信号 → 执行"的桥接思路，但保持零资金：Signal 仅含
入场/止损/止盈派生（基于 ATR），由 ExecutionEngine 后续消费。
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from ..models import Signal, Direction
from .engine import gen_signal, SignalCandidate, MarketContext
from .mean_reversion import gen_mean_reversion


def _direction_of(s: str) -> Direction:
    return Direction.LONG if s == "做多" else Direction.SHORT


def candidate_to_signal(cand: SignalCandidate, price: float,
                        tf: str = "1H") -> Optional[Signal]:
    """趋势引擎候选 → 原型 Signal（基于 ATR 派生 SL/TP）。"""
    if not cand.passed:
        return None
    atr = cand.atr or price * 0.01
    mult = 2.0
    if cand.direction == "做多":
        sl = price - mult * atr
        tp1 = price + cand.rr * mult * atr
        tp2 = price + cand.rr * 2 * mult * atr
    else:
        sl = price + mult * atr
        tp1 = price - cand.rr * mult * atr
        tp2 = price - cand.rr * 2 * mult * atr
    conf = max(0.1, min(1.0, cand.score / 12.0))
    return Signal(
        symbol=cand.symbol, tf=tf, direction=_direction_of(cand.direction),
        entry=round(price, 2), sl=round(sl, 2), tp1=round(tp1, 2), tp2=round(tp2, 2),
        rr=cand.rr, confidence=conf,
        signal_id=f"{cand.symbol}_{tf}_{uuid.uuid4().hex[:8]}", atr=atr,
    )


def generate_signals(symbols: List[str], market_data: dict,
                     ctx: MarketContext = None, price_map: dict = None,
                     tf: str = "1H") -> List[Signal]:
    """批量生成信号（regime 感知路由，圆桌决议）。

    根据 ctx.regime 切换信号生成路径：
      - TREND: 趋势优先 → MR 回退（当前行为）
      - RANGE: MR 优先 → 趋势回退（防止趋势假信号）
      - CRASH: 趋势引擎 + 短窗口过滤（禁用 MR，规避反弹陷阱）
      - None/其他: 趋势优先（向后兼容）
    """
    ctx = ctx or MarketContext()
    regime = ctx.regime or "TREND"
    out: List[Signal] = []
    for sym in symbols:
        d = market_data.get(sym, {})
        c1h = d.get("1h", [])
        if len(c1h) < 30:
            continue
        price = (price_map or {}).get(sym, c1h[-1]["c"])
        fr = d.get("fr", 0.0)

        if regime == "RANGE":
            # RANGE: MR 优先，趋势回退
            mr = gen_mean_reversion(sym, c1h, fr, ctx)
            if mr.triggered:
                mc = SignalCandidate(
                    symbol=sym, direction=mr.direction, score=mr.score,
                    conds=mr.conds, atr=0.0, atr_pct=0.0, adx=15.0, rr=1.5,
                )
                sig = candidate_to_signal(mc, price, tf)
            else:
                cand = gen_signal(sym, c1h, d.get("4h"), d.get("1w"), ctx)
                sig = candidate_to_signal(cand, price, tf)
        elif regime == "CRASH":
            # CRASH: 趋势引擎 + 短窗口（禁用 MR）
            cand = gen_signal(sym, c1h, d.get("4h"), d.get("1w"), ctx)
            # CRASH 下提高门槛：只收评分 7+ 的信号（engine 已有 CRASH 禁做空逻辑）
            if cand.passed and cand.score >= 7:
                sig = candidate_to_signal(cand, price, tf)
            else:
                sig = None
        else:
            # TREND: 趋势优先 → MR 回退（原逻辑）
            cand = gen_signal(sym, c1h, d.get("4h"), d.get("1w"), ctx)
            sig = candidate_to_signal(cand, price, tf)
            if sig is None:
                mr = gen_mean_reversion(sym, c1h, fr, ctx)
                if mr.triggered:
                    mc = SignalCandidate(
                        symbol=sym, direction=mr.direction, score=mr.score,
                        conds=mr.conds, atr=cand.atr, atr_pct=cand.atr_pct,
                        adx=cand.adx, rr=1.5,
                    )
                    sig = candidate_to_signal(mc, price, tf)

        if sig is not None:
            out.append(sig)
    return out
