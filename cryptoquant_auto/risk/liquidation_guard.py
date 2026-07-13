"""强平价预警（移植服务器 liquidation_guard 优点，仅告警不改仓）。

estimate_liquidation_price 已在 risk/exec_sl 就位；此处补"遍历持仓算缓冲"循环。
Fail-closed：只读巡检，只产生告警事件，绝不触发任何平仓/下单。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..models import Position, Direction
from .exec_sl import estimate_liquidation_price


def check_liquidation_guard(active_positions: List[Position], tickers: Dict[str, float],
                             warn_buffer_pct: float = 0.10) -> List[str]:
    """返回告警列表。距强平价不足缓冲即告警。

    P2-2 修复：杠杆取自持仓实际 leverage（由订单名义推导），不再硬编码 5.0；
    liq_price 为 None（≈1x 无强平）的持仓直接跳过。
    """
    alerts: List[str] = []
    for p in active_positions:
        if p.liq_price is None:
            continue
        price = tickers.get(p.symbol)
        if price is None or price <= 0:
            continue
        liq = estimate_liquidation_price(p.entry_price, p.leverage, p.direction)
        dist = abs(price - liq) / price
        if dist < warn_buffer_pct:
            alerts.append(f"{p.symbol} 距强平价仅 {dist:.1%}（强平价≈{liq:.2f}）")
    return alerts
