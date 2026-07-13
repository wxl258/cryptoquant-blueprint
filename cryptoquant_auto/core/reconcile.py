"""对账防超仓：期望持仓（引擎意图） vs 实际持仓（适配器真实）。

一切以实际持仓为准，信号只是意图。差异超阈值 -> 标记并建议动作。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..models import Position

TOL_NOTIONAL = 50.0   # 名义差异容差（USD）


@dataclass
class ReconcileItem:
    symbol: str
    expected_qty: float
    actual_qty: float
    diff_qty: float
    action: str          # OK | OVER（超仓，建议减仓）| UNDER（缺仓，建议补/重判）


@dataclass
class ReconcileReport:
    items: List[ReconcileItem] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def clean(self) -> bool:
        return all(i.action == "OK" for i in self.items)


def reconcile(expected: List[Position], actual: List[Position],
              tol_notional: float = TOL_NOTIONAL) -> ReconcileReport:
    rep = ReconcileReport()
    actual_map = {p.symbol: p for p in actual}
    # expected 可能含同一标的多笔（不同 signal_id），按 symbol 聚合名义
    exp_map: Dict[str, float] = {}
    exp_price: Dict[str, float] = {}
    for p in expected:
        exp_map[p.symbol] = exp_map.get(p.symbol, 0.0) + p.qty
        exp_price[p.symbol] = p.entry_price
    symbols = set(exp_map.keys()) | set(actual_map.keys())
    for sym in sorted(symbols):
        eq = exp_map.get(sym, 0.0)
        act = actual_map.get(sym)
        aq = act.qty if act else 0.0
        price = exp_price.get(sym, act.entry_price if act else 1.0) or 1.0
        diff_notional = abs(eq - aq) * price
        if diff_notional <= tol_notional:
            action = "OK"
        elif eq > aq:
            action = "UNDER"
        else:
            action = "OVER"
        rep.items.append(ReconcileItem(sym, eq, aq, eq - aq, action))
    return rep
