"""对账防超仓：期望持仓（引擎意图） vs 实际持仓（适配器真实）。

一切以实际持仓为准，信号只是意图。差异超阈值 -> 标记并建议动作。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..models import Position, Direction

TOL_NOTIONAL = 50.0   # 名义差异容差（USD）


@dataclass
class ReconcileItem:
    symbol: str
    expected_qty: float
    actual_qty: float
    diff_qty: float
    action: str          # OK | OVER（超仓，建议减仓）| UNDER（缺仓，建议补/重判）| DIR_FLIP（方向反了！）
    expected_dir: str = ""
    actual_dir: str = ""


@dataclass
class ReconcileReport:
    items: List[ReconcileItem] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def clean(self) -> bool:
        return all(i.action == "OK" for i in self.items)


def _signed_notional(p: Position, price: float) -> float:
    """带方向的名义：LONG 为正、SHORT 为负，使反向持仓不可相互抵消。"""
    sign = 1.0 if p.direction is Direction.LONG else -1.0
    return sign * p.qty * price


def reconcile(expected: List[Position], actual: List[Position],
              tol_notional: float = TOL_NOTIONAL) -> ReconcileReport:
    rep = ReconcileReport()
    actual_map = {p.symbol: p for p in actual}
    # expected 可能含同一标的多笔（不同 signal_id）：按「标的 + 方向」聚合带符号名义
    exp_signed: Dict[str, float] = {}
    exp_dir: Dict[str, str] = {}
    exp_price: Dict[str, float] = {}
    for p in expected:
        price = p.entry_price or 1.0
        exp_signed[p.symbol] = exp_signed.get(p.symbol, 0.0) + _signed_notional(p, price)
        exp_dir[p.symbol] = p.direction.name if hasattr(p.direction, "name") else str(p.direction)
        exp_price[p.symbol] = price
    symbols = set(exp_signed.keys()) | set(actual_map.keys())
    for sym in sorted(symbols):
        exp = exp_signed.get(sym, 0.0)
        act = actual_map.get(sym)
        aq = act.qty if act else 0.0
        act_signed = _signed_notional(act, act.entry_price) if act else 0.0
        act_dir = act.direction.name if (act and hasattr(act.direction, "name")) else ""
        price = exp_price.get(sym, act.entry_price if act else 1.0) or 1.0
        diff_notional = abs(exp - act_signed)
        exp_qty = exp_signed.get(sym, 0.0)
        # 【P1-7 修复】先判方向：期望与实际的带符号名义异号 → 反向持仓（裸卖/裸买错配），
        # 即便数量相等也须告警，绝不抵消成 OK。
        if act is not None and exp != 0.0 and (exp > 0) != (act_signed > 0):
            action = "DIR_FLIP"
        elif diff_notional <= tol_notional:
            action = "OK"
        elif exp > act_signed:
            action = "UNDER"
        else:
            action = "OVER"
        rep.items.append(ReconcileItem(sym, exp_qty, aq, exp - act_signed,
                                       action, exp_dir.get(sym, ""), act_dir))
    return rep
