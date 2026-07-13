"""跨所 fallback 路由（Binance 主 -> OKX -> GateIO）。

同一标的持仓不跨所拆分；fallback 仅整体改路由该标的。
网络超时/连接错 -> 尝试下一所；参数错/余额不足 -> 断路（不重试）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..models import Order, OrderStatus
from ..adapters.base import ExchangeAdapter


class FallbackRouter(ExchangeAdapter):
    def __init__(self, adapters: Dict[str, ExchangeAdapter], priority: List[str] = None):
        self.adapters = adapters
        self.priority = priority or list(adapters.keys())

    def submit(self, order: Order) -> Optional[Order]:
        last_err = None
        for venue in self.priority:
            ad = self.adapters.get(venue)
            if ad is None:
                continue
            try:
                o = ad.submit(order)
                if o.status is OrderStatus.REJECTED:
                    last_err = f"reject@{venue}"
                    continue
                return o
            except (TimeoutError, ConnectionError, OSError) as e:
                last_err = f"transient@{venue}:{e}"
                continue
        # 全部失败：返回 REJECTED 占位（真实环境应告警断路）
        order.status = OrderStatus.REJECTED
        order.signal_id = order.signal_id  # 保留
        setattr(order, "_err", last_err)
        return order

    def cancel(self, coid: str) -> bool:
        for venue in self.priority:
            ad = self.adapters.get(venue)
            if ad and ad.cancel(coid):
                return True
        return False

    def query_open(self):
        out = []
        for venue in self.priority:
            ad = self.adapters.get(venue)
            if ad:
                out.extend(ad.query_open())
        return out

    def query_position(self, symbol: str):
        for venue in self.priority:
            ad = self.adapters.get(venue)
            if ad:
                p = ad.query_position(symbol)
                if p:
                    return p
        return None

    def query_positions(self):
        out = []
        for venue in self.priority:
            ad = self.adapters.get(venue)
            if ad:
                out.extend(ad.query_positions())
        return out

    def simulate_market(self, prices: dict) -> None:
        for venue in self.priority:
            ad = self.adapters.get(venue)
            if ad:
                ad.simulate_market(prices)
