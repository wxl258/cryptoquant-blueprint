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
                 timeout: float = 10.0, constitution=None):
        # 【P0-2 修复】注入宪法，使适配器级兜底实盘硬锁：即便引擎层被绕过，
        # 只要 constitution.live_capital=True 就拒绝任何真实下单。否则改 BASE 为主网
        # + 接 Key 即可直触真钱，原硬锁仅存在于 run_validation 的 assert（非执行入口）。
        # 另：若 BASE 指向主网（非 testnet）则架构级拒绝构建，杜绝一念之差触真钱。
        if constitution is None:
            from ..risk.constitution import TradingConstitution
            constitution = TradingConstitution(live_capital=False)
        if "testnet" not in BASE:
            raise RuntimeError(
                "BinanceTestnetAdapter 拒绝接入主网 BASE；原型仅允许 testnet 假钱")
        if constitution.live_capital:
            raise RuntimeError(
                "constitution.live_capital=True：架构级禁止任何实盘动作（原型仅沙盒）")
        self.constitution = constitution
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
        # Binance USDT-M 合约要求参数按字母序排序后做 HMAC 签名
        sorted_pairs = sorted(params.items(), key=lambda kv: kv[0])
        q = urllib.parse.urlencode(sorted_pairs)
        sig = hmac.new(self.sec.encode(), q.encode(),
                       hashlib.sha256).hexdigest()
        sorted_pairs.append(("signature", sig))
        return dict(sorted_pairs)

    @staticmethod
    def _sym(s: str) -> str:
        return f"{s}USDT"

    # ---- 下单 ----
    def submit(self, order: Order) -> Order:
        # 【P0-2 修复】适配器级实盘硬锁兜底：即便引擎层被绕过，live_capital=True 也拒绝下单。
        if self.constitution.live_capital:
            order.status = OrderStatus.REJECTED
            return order
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
        elif order.otype is OrderType.REDUCE:
            # 【P1-8 修复】L3 生存态减仓单：市价减仓（不挂限价，确保生存态能真正减仓）。
            # 原实现缺此分支 → REDUCE 无 type/price → 真实路径拒单/异常，减仓失效。
            params.update(type="MARKET",
                          reduceOnly=True,
                          closePosition=False)
            params.pop("price", None)
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
    def submit_market(self, symbol: str, side: str, qty: float,
                      signal_id: str = "", price: float = 0) -> dict:
        """市价入场单（测试网即刻成交，绕过 LIMIT-only submit 路径）。

        参数同 Binance /fapi/v1/order：symbol（如 BTC）, side（BUY/SELL）, qty。
        若传 price 则使用 LIMIT 激进限价(测试网 MARKET 常驻 NEW 不成交)。
        返回测试网 API 原始 JSON；同时更新内部 position/fills 跟踪。
        """
        if self.constitution.live_capital:
            return {"status": "REJECTED", "reason": "live_capital=True 禁止下单"}
        if price and price > 0:
            params = {"symbol": self._sym(symbol), "side": side,
                      "quantity": round(qty, 6), "type": "LIMIT",
                      "price": f"{price:.8f}".rstrip("0").rstrip("."),
                      "timeInForce": "GTC",
                      "newClientOrderId": f"mkt_{symbol}_{side}_{int(time.time())}"}
        else:
            params = {"symbol": self._sym(symbol), "side": side,
                      "quantity": round(qty, 6), "type": "MARKET",
                      "newClientOrderId": f"mkt_{symbol}_{side}_{int(time.time())}"}
        r = self.sess.post(BASE + "/fapi/v1/order", params=self._sign(params),
                           timeout=self._timeout)
        data = r.json()
        if data.get("orderId"):
            fq = float(data.get("executedQty", 0))
            ap = float(data.get("avgPrice", data.get("price", 0)))
            if fq > 0:
                o = Order(
                    coid=data.get("clientOrderId", ""), symbol=symbol,
                    side=side, otype=OrderType.ENTRY,
                    price=ap, qty=fq, signal_id=signal_id)
                o.filled_qty = fq
                o.filled_price = ap
                o.status = OrderStatus.FILLED
                self._open_position(o, ap)
                self.fills.append(Fill(
                    o.coid, symbol, side, ap, fq, time.time()))
                # 调试确认
                logger = __import__("logging").getLogger("cryptoquant.testnet")
                logger.info("[submit_market] %s %s FILLED qty=%s price=%s "
                           "fills=%d positions=%d",
                           symbol, side, fq, ap,
                           len(self.fills), len(self.positions))
            else:
                # 市价单已接受但尚未完全成交（测试网偶有延迟）
                logger = __import__("logging").getLogger("cryptoquant.testnet")
                logger.warning("[submit_market] %s %s PENDING orderId=%s "
                              "status=%s fq=%s", symbol, side,
                              data.get("orderId"), data.get("status", "?"), fq)
                # 等 1.5s 后查一次订单状态看是否已成交
                __import__("time").sleep(1.5)
                oid = data.get("orderId")
                try:
                    qp = self.sess.get(
                        BASE + "/fapi/v1/order",
                        params=self._sign({"symbol": self._sym(symbol),
                                           "orderId": oid}),
                        timeout=self._timeout)
                    od = qp.json()
                    if od.get("status") == "FILLED":
                        fq2 = float(od.get("executedQty", 0))
                        ap2 = float(od.get("avgPrice", od.get("price", 0)))
                        if fq2 > 0:
                            o = Order(
                                coid=od.get("clientOrderId",""), symbol=symbol,
                                side=side, otype=OrderType.ENTRY,
                                price=ap2, qty=fq2, signal_id=signal_id)
                            o.filled_qty = fq2; o.filled_price = ap2
                            o.status = OrderStatus.FILLED
                            self._open_position(o, ap2)
                            self.fills.append(Fill(
                                o.coid, symbol, side, ap2, fq2, time.time()))
                            logger.info("[submit_market] %s %s FILLED(retry) "
                                        "qty=%s price=%s", symbol, side, fq2, ap2)
                except Exception as e:
                    logger.warning("[submit_market] retry query %s: %s", oid, e)
        else:
            # 非预期响应（典型：密钥错误 / 权限不足 / 参数越界）
            logger = __import__("logging").getLogger("cryptoquant.testnet")
            logger.warning("[submit_market] %s %s http=%d code=%s msg=%s",
                          symbol, side, getattr(r, "status_code", 0),
                          data.get("code", "?"), data.get("msg", "?"))
        return data

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
