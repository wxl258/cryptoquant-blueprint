"""Mock 交易所适配器：零网络、可模拟成交与故障注入。

用于本地 Shadow/Paper 调试，验证「下单->成交->对账->KillSwitch->执行级SL」
整条管线，以及 chaos game day 故障注入。所有成交由 simulate_market() 驱动。
"""
from __future__ import annotations

import time
from typing import List, Optional

from ..models import Order, OrderStatus, OrderType, Fill, Position, Direction
from .base import ExchangeAdapter


class MockAdapter(ExchangeAdapter):
    def __init__(self, equity: float = 100_000.0, fault_mode: str = "none"):
        """
        fault_mode:
          none        - 正常
          timeout     - 入场单提交抛 TimeoutError（验证幂等重试）
          partial     - 入场单部分成交（filled_qty < qty）
          maintenance - 所有提交被拒（验证断路）
        """
        self.equity = equity
        self.fault_mode = fault_mode
        self.open_orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.fills: List[Fill] = []
        self.last_ts = time.time()
        self._timed_out: set[str] = set()   # 已超时过一次的 coid（瞬时故障模拟）

    # ---- 幂等与提交 ----
    def submit(self, order: Order) -> Order:
        if self.fault_mode == "maintenance":
            order.status = OrderStatus.REJECTED
            return order
        if self.fault_mode == "timeout" and order.otype is OrderType.ENTRY and order.coid not in self._timed_out:
            self._timed_out.add(order.coid)
            raise TimeoutError("mock api timeout (entry)")  # 瞬时故障：仅首次抛
        # 幂等：同 coid 已存在则直接返回，绝不重复下单
        if order.coid in self.open_orders:
            return self.open_orders[order.coid]
        order.status = OrderStatus.OPEN
        self.open_orders[order.coid] = order
        return order

    def cancel(self, coid: str) -> bool:
        o = self.open_orders.pop(coid, None)
        if o:
            o.status = OrderStatus.CANCELED
            return True
        return False

    def query_open(self) -> List[Order]:
        return list(self.open_orders.values())

    def query_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def query_positions(self) -> List[Position]:
        return list(self.positions.values())

    # ---- 成交模拟 ----
    def simulate_market(self, prices: dict) -> None:
        for coid, o in list(self.open_orders.items()):
            if o.status is not OrderStatus.OPEN:
                continue
            p = prices.get(o.symbol)
            if p is None:
                continue
            if self._touched(o, p):
                self._fill(o, p)

    @staticmethod
    def _touched(o: Order, p: float) -> bool:
        if o.otype is OrderType.ENTRY:
            return (o.side == "BUY" and p <= o.price) or (o.side == "SELL" and p >= o.price)
        if o.otype is OrderType.TP:
            # 多头止盈在上方，空头止盈在下方
            return (o.side == "SELL" and p >= o.price) or (o.side == "BUY" and p <= o.price)
        if o.otype is OrderType.SL:
            # 多头止损在下方，空头止损在上方
            return (o.side == "SELL" and p <= o.price) or (o.side == "BUY" and p >= o.price)
        return False

    def _fill(self, o: Order, p: float) -> None:
        if self.fault_mode == "partial" and o.otype is OrderType.ENTRY:
            o.filled_qty = o.qty * 0.5
            o.status = OrderStatus.PARTIAL
        else:
            o.filled_qty = o.qty
            o.status = OrderStatus.FILLED
        o.filled_price = p
        self.fills.append(Fill(o.coid, o.symbol, o.side, p, o.filled_qty, time.time()))
        if o.otype is OrderType.ENTRY and o.status is OrderStatus.FILLED:
            self._open_position(o, p)
        elif o.otype in (OrderType.TP, OrderType.SL):
            self._close_or_reduce(o, p)

    def _close_or_reduce(self, o: Order, p: float) -> None:
        pos = self.positions.get(o.symbol)
        if pos is None:
            return
        # 【C1 修复】按子单 leg 比例减仓：
        #   tp1/数量 = 50% → 减 0.5 持仓；tp2/数量 = 50% → 再减 0.5；sl = 全量兜底
        # 原实现 pos.qty -= 全量 → 一次清零 → 后续 leg 永不触发（状态机崩坏）。
        frac = 1.0
        if o.leg == "tp1":
            frac = 0.5
        elif o.leg == "tp2":
            frac = 0.5
        reduce_qty = o.filled_qty  # 子单 qty 已在 build_tp_sl_orders 按比例设好
        pos.qty -= reduce_qty
        pnl = (p - pos.entry_price) * reduce_qty * pos.direction.sign
        pos.realized_pnl += pnl
        if o.otype is OrderType.TP:
            pos.state = "TP1_FILLED" if o.leg == "tp1" else "TP2_CLOSED"
        else:
            pos.state = "SL_CLOSED"
        # 仅当持仓减到近似 0（或 SL 兜底）才清仓；tp1/tp2 不会触发清零
        if pos.qty <= 1e-9 or o.otype is OrderType.SL:
            del self.positions[o.symbol]

    def _open_position(self, o: Order, p: float) -> None:
        direction = Direction.LONG if o.side == "BUY" else Direction.SHORT
        pos = Position(
            symbol=o.symbol,
            direction=direction,
            entry_price=p,
            qty=o.filled_qty,
            initial_qty=o.filled_qty,
            sl_price=0.0,     # 由引擎下 SL 子单并回填
            tp1_price=0.0,
            tp2_price=0.0,
            entry_coid=o.coid,
            signal_id=o.signal_id,
        )
        self.positions[o.symbol] = pos
