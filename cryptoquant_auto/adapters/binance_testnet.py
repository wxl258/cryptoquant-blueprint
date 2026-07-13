"""Binance USDT-M 合约测试网适配器（真实 REST，零真金白银）。

接 Key 后即可跑真实 API 假钱。签名/端点按 Binance 文档实现。
注意：simulate_market 用价格穿越检测成交（本地可调试）；生产应改为
user data stream 真实成交回写。OKX/GateIO 见 testnet_stub.py。
"""
from __future__ import annotations

import time
import hmac
import hashlib
import urllib.parse

import requests

from ..models import Order, OrderStatus, OrderType, Position, Direction, Fill
from .base import ExchangeAdapter

BASE = "https://testnet.binancefuture.com"


class BinanceTestnetAdapter(ExchangeAdapter):
    def __init__(self, api_key: str, api_secret: str, recv_window: int = 5000,
                 timeout: float = 10.0):
        self.key = api_key
        self.sec = api_secret
        self.recv = recv_window
        self.sess = requests.Session()
        self.sess.headers.update({"X-MBX-APIKEY": self.key})
        self._timeout = timeout
        self.open_orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []

    # ---- 签名 ----
    def _sign(self, params: dict) -> dict:
        params["recvWindow"] = self.recv
        params["timestamp"] = int(time.time() * 1000)
        q = urllib.parse.urlencode(params)
        params["signature"] = hmac.new(self.sec.encode(), q.encode(),
                                       hashlib.sha256).hexdigest()
        return params

    @staticmethod
    def _sym(s: str) -> str:
        return f"{s}USDT"

    # ---- 下单 ----
    def submit(self, order: Order) -> Order:
        params = {"symbol": self._sym(order.symbol), "side": order.side,
                  "quantity": round(order.qty, 6)}
        if order.otype is OrderType.ENTRY:
            params.update(type="LIMIT", price=str(order.price),
                          timeInForce="GTX" if order.post_only else "GTC")
        elif order.otype is OrderType.TP:
            params.update(type="TAKE_PROFIT", stopPrice=str(order.price),
                          price=str(order.price), timeInForce="GTC")
        elif order.otype is OrderType.SL:
            params.update(type="STOP", stopPrice=str(order.price),
                          price=str(order.price), timeInForce="GTC")
        params["newClientOrderId"] = order.coid
        try:
            r = self.sess.post(BASE + "/fapi/v1/order", params=self._sign(params),
                               timeout=self._timeout)
        except (requests.RequestException, TimeoutError) as e:
            raise TimeoutError(f"binance submit: {e}")
        if r.status_code != 200:
            order.status = OrderStatus.REJECTED
            return order
        order.status = OrderStatus.OPEN
        self.open_orders[order.coid] = order
        return order

    def cancel(self, coid: str) -> bool:
        o = self.open_orders.pop(coid, None)
        if not o:
            return False
        params = {"symbol": self._sym(o.symbol), "origClientOrderId": coid}
        try:
            r = self.sess.delete(BASE + "/fapi/v1/order", params=self._sign(params),
                                 timeout=self._timeout)
        except (requests.RequestException, TimeoutError):
            return False
        ok = r.status_code == 200
        if ok:
            o.status = OrderStatus.CANCELED
        return ok

    def query_open(self):
        return list(self.open_orders.values())

    def query_position(self, symbol: str):
        return self.positions.get(symbol)

    def query_positions(self):
        return list(self.positions.values())

    # ---- 成交（价格穿越检测；生产改 user data stream）----
    def simulate_market(self, prices: dict) -> None:
        for coid, o in list(self.open_orders.items()):
            p = prices.get(o.symbol)
            if p is None:
                continue
            if self._touched(o, p):
                o.filled_qty = o.qty
                o.status = OrderStatus.FILLED
                o.filled_price = p
                self.fills.append(Fill(coid, o.symbol, o.side, p, o.filled_qty, time.time()))
                if o.otype is OrderType.ENTRY:
                    self._open_position(o, p)
                elif o.otype in (OrderType.TP, OrderType.SL):
                    self._close_or_reduce(o, p)

    @staticmethod
    def _touched(o: Order, p: float) -> bool:
        if o.otype is OrderType.ENTRY:
            return (o.side == "BUY" and p <= o.price) or (o.side == "SELL" and p >= o.price)
        if o.otype is OrderType.TP:
            return (o.side == "SELL" and p >= o.price) or (o.side == "BUY" and p <= o.price)
        if o.otype is OrderType.SL:
            return (o.side == "SELL" and p <= o.price) or (o.side == "BUY" and p >= o.price)
        return False

    def _open_position(self, o: Order, p: float) -> None:
        direction = Direction.LONG if o.side == "BUY" else Direction.SHORT
        self.positions[o.symbol] = Position(
            symbol=o.symbol, direction=direction, entry_price=p, qty=o.filled_qty,
            initial_qty=o.filled_qty, sl_price=0.0, tp1_price=0.0, tp2_price=0.0,
            entry_coid=o.coid, signal_id=o.signal_id)

    def _close_or_reduce(self, o: Order, p: float) -> None:
        pos = self.positions.get(o.symbol)
        if pos is None:
            return
        # 【C1 修复】与 mock.py 一致：子单 qty 已按比例设好，按 filled_qty 减仓；
        # 仅 SL 兜底或持仓减到近似 0 才清仓，tp1/tp2 不再一次性清零。
        reduce_qty = o.filled_qty
        pos.qty -= reduce_qty
        pos.realized_pnl += (p - pos.entry_price) * reduce_qty * pos.direction.sign
        if o.otype is OrderType.TP:
            pos.state = "TP1_FILLED" if o.leg == "tp1" else "TP2_CLOSED"
        else:
            pos.state = "SL_CLOSED"
        if pos.qty <= 1e-9 or o.otype is OrderType.SL:
            del self.positions[o.symbol]
