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
import logging

import requests

from ..models import Order, OrderStatus, OrderType, Position, Direction, Fill
from .base import ExchangeAdapter

BASE = "https://testnet.binancefuture.com"
logger = logging.getLogger("cryptoquant.testnet")


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
        self._leverage_set: set[str] = set()  # 已设杠杆的币对（避免每轮重复 API 调用）

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

    def _ensure_leverage(self, symbol: str, leverage: int) -> None:
        """确保该币对杠杆倍数正确（仅对首次交易的币对发起 API 调用，避免每轮浪费请求）。

        不设杠杆时，币安使用账户默认值（测试网常为 0x/未定义），导致保证金显示 0、
        实际杠杆与预期不符。本方法在首次下单前调用一次即可。
        """
        if symbol in self._leverage_set:
            return
        try:
            r = self.sess.post(BASE + "/fapi/v1/leverage",
                               params=self._sign({"symbol": self._sym(symbol),
                                                  "leverage": leverage}),
                               timeout=self._timeout)
            if r.status_code == 200:
                self._leverage_set.add(symbol)
                logger.info("[leverage] %s → %dx (已设)", symbol, leverage)
            else:
                logger.warning("[leverage] %s 设杠杆 http=%d: %s", symbol, r.status_code, r.text[:100])
        except Exception as e:
            logger.warning("[leverage] %s 设杠杆异常: %s (继续交易)", symbol, e)
            # 设杠杆失败不阻断交易（测试网默认值也能成交，只是杠杆不对）


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
            # 止盈市价单：仅设 stopPrice，不设 limit price（确保触发即成交）
            params.update(type="TAKE_PROFIT_MARKET", stopPrice=str(order.price),
                          closePosition=True)
        elif order.otype is OrderType.SL:
            # 止损市价单：仅设 stopPrice，不设 limit price（确保触发即成交）
            params.update(type="STOP_MARKET", stopPrice=str(order.price),
                          closePosition=True)
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

    def refresh_positions(self):
        """从测试网 REST 拉取真实持仓，覆盖内存 self.positions（跨 run 持久化）。

        修复：原 query_positions 只读内存，每次 cron 新建适配器都从空开始，
        导致看不到测试网既有持仓 → 重复开单、无法平仓历史单。
        失败返回 None（调用方据此跳过本轮，避免盲目重复开单）。
        """
        try:
            r = self.sess.get(BASE + "/fapi/v2/positionRisk",
                              params=self._sign({}), timeout=self._timeout)
        except (requests.RequestException, TimeoutError) as e:
            logger.error("[refresh_positions] 网络失败, 跳过本轮: %s", e)
            return None
        if r.status_code != 200:
            logger.error("[refresh_positions] http=%d %s", r.status_code, str(r.text)[:200])
            return None
        try:
            data = r.json()
        except Exception as e:
            logger.error("[refresh_positions] json %s", e)
            return None
        self.positions = {}
        for p in data:
            amt = float(p.get("positionAmt", 0) or 0)
            if abs(amt) < 1e-9:
                continue
            sym = str(p.get("symbol", "")).replace("USDT", "")
            direction = Direction.LONG if amt > 0 else Direction.SHORT
            self.positions[sym] = Position(
                symbol=sym, direction=direction,
                entry_price=float(p.get("entryPrice", 0) or 0),
                qty=abs(amt), initial_qty=abs(amt),
                sl_price=0.0, tp1_price=0.0, tp2_price=0.0,
                entry_coid="", signal_id="")
        logger.info("[refresh_positions] 拉取真实持仓 %d 个", len(self.positions))
        return list(self.positions.values())

    def refresh_open_orders(self):
        """从测试网 REST 拉取真实挂单，覆盖内存 self.open_orders（避免重复挂单）。
        失败返回 None。"""
        try:
            r = self.sess.get(BASE + "/fapi/v1/openOrders",
                              params=self._sign({}), timeout=self._timeout)
        except (requests.RequestException, TimeoutError) as e:
            logger.error("[refresh_open_orders] 网络失败, 跳过本轮: %s", e)
            return None
        if r.status_code != 200:
            logger.error("[refresh_open_orders] http=%d %s", r.status_code, str(r.text)[:200])
            return None
        try:
            data = r.json()
        except Exception as e:
            logger.error("[refresh_open_orders] json %s", e)
            return None
        self.open_orders = {}
        for o in data:
            coid = o.get("clientOrderId", "")
            side = o.get("side", "")
            sym = str(o.get("symbol", "")).replace("USDT", "")
            order = Order(
                coid=coid, symbol=sym, side=side, otype=OrderType.ENTRY,
                price=float(o.get("price", 0) or 0),
                qty=float(o.get("origQty", 0) or 0), signal_id="")
            order.status = OrderStatus.OPEN
            self.open_orders[coid] = order
        logger.info("[refresh_open_orders] 拉取真实挂单 %d 个", len(self.open_orders))
        return list(self.open_orders.values())

    # ---- 成交（价格穿越检测；生产改 user data stream）----
    def query_order(self, symbol: str, coid: str = "", order_id: int = 0):
        '''按 newClientOrderId / orderId 回查订单真实状态（回执确认用）。'''
        if not coid and not order_id:
            return None
        params = {"symbol": self._sym(symbol)}
        if order_id:
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = coid
        try:
            r = self.sess.get(BASE + "/fapi/v1/order",
                              params=self._sign(params), timeout=self._timeout)
        except (requests.RequestException, TimeoutError) as e:
            logger.warning("[query_order] %s 回查失败: %s", symbol, e)
            return {"ok": False, "status": "ERROR", "retryable": True,
                    "error": f"net:{e}", "coid": coid}
        if r.status_code == 400 and r.json().get("code") == -2013:
            return {"ok": False, "status": "NOT_FOUND", "coid": coid}
        return self._normalize_receipt(symbol, r.json().get("side", ""), 0.0, "", r, coid)

    def _normalize_receipt(self, symbol: str, side: str, qty: float,
                           signal_id: str, r, coid: str, raw=None):
        '''把 Binance 原始响应规范化为带明确 status/ok 的回执（杜绝 '?'）。

        status in {FILLED, NEW, PARTIALLY_FILLED, CANCELED, EXPIRED,
                  REJECTED, ERROR, NOT_FOUND}
        '''
        try:
            data = raw if raw is not None else r.json()
        except Exception:
            return {"ok": False, "status": "ERROR", "retryable": True,
                    "error": "响应非JSON", "coid": coid}
        if r.status_code != 200 or not data.get("orderId"):
            code = data.get("code")
            msg = data.get("msg", "")
            retryable = (r.status_code in (429, 500, 502, 503, 504)) or (code in (429, -1003, -1021))
            return {"ok": False, "status": "REJECTED", "retryable": retryable,
                    "code": code, "msg": msg, "coid": coid,
                    "error": f"http={r.status_code} code={code} {msg}"}
        fq = float(data.get("executedQty", 0))
        ap = float(data.get("avgPrice", data.get("price", 0)))
        status = data.get("status", "NEW")
        receipt = {"ok": True, "status": status, "order_id": data.get("orderId"),
                   "coid": coid, "executed_qty": fq, "avg_price": ap,
                   "code": data.get("code"), "msg": data.get("msg", "")}
        if fq > 0:
            o = Order(coid=data.get("clientOrderId", ""), symbol=symbol, side=side,
                      otype=OrderType.ENTRY, price=ap, qty=fq, signal_id=signal_id)
            o.filled_qty = fq
            o.filled_price = ap
            o.status = OrderStatus.FILLED
            self._open_position(o, ap)
            self.fills.append(Fill(o.coid, symbol, side, ap, fq, time.time()))
            logger.info("[submit_market] %s %s FILLED qty=%s price=%s fills=%d positions=%d",
                        symbol, side, fq, ap, len(self.fills), len(self.positions))
        else:
            logger.info("[submit_market] %s %s 已接受 status=%s orderId=%s (未即刻成交)",
                        symbol, side, status, data.get("orderId"))
        return receipt

    def submit_market(self, symbol: str, side: str, qty: float,
                      signal_id: str = "", price: float = 0,
                      max_retries: int = 2, retry_backoff: float = 1.0):
        '''市价/限价入场单（测试网即刻成交，绕过 LIMIT-only submit 路径）。

        回执校验（核心修复）：
          - 永远返回带明确 status/ok 的规范回执，杜绝 '?' 模糊态。
          - 网络/超时/5xx/限频 视为可重试(transient)，用同一 newClientOrderId
            幂等重试，不会重复下单。
          - Binance 业务拒绝(无 orderId / 4xx 业务码)判定 REJECTED 并放弃，不重试。
          - 已接受但未即刻成交(NEW)时，短等后按 coid 回查确认真实状态，避免悬空单。
          - 重试耗尽仍不明时按 coid 回查：找到返回真实状态；未找到(-2013)判未成交。
        '''
        if self.constitution.live_capital:
            return {"ok": False, "status": "REJECTED",
                    "reason": "live_capital=True 禁止下单", "coid": ""}
        coid = f"mkt_{symbol}_{side}_{int(time.time() * 1000)}"
        if price and price > 0:
            params = {"symbol": self._sym(symbol), "side": side,
                      "quantity": round(qty, 6), "type": "LIMIT",
                      "price": f"{price:.8f}".rstrip("0").rstrip("."),
                      "timeInForce": "GTC",
                      "newClientOrderId": coid}
        else:
            params = {"symbol": self._sym(symbol), "side": side,
                      "quantity": round(qty, 6), "type": "MARKET",
                      "newClientOrderId": coid}
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                r = self.sess.post(BASE + "/fapi/v1/order",
                                   params=self._sign(params), timeout=self._timeout)
            except (requests.RequestException, TimeoutError) as e:
                last_err = f"net:{e}"
                logger.warning("[submit_market] %s %s 网络异常(第%d次): %s",
                               symbol, side, attempt + 1, e)
                time.sleep(retry_backoff)
                continue
            receipt = self._normalize_receipt(symbol, side, qty, signal_id, r, coid)
            if receipt["ok"]:
                if (receipt.get("executed_qty", 0) <= 0
                        and receipt.get("status") in ("NEW", "PARTIALLY_FILLED", "PENDING_NEW")):
                    time.sleep(1.5)
                    conf = self.query_order(symbol, coid=coid)
                    if conf and conf.get("ok") and conf.get("executed_qty", 0) > 0:
                        return conf
                return receipt
            if not receipt.get("retryable"):
                return receipt
            last_err = receipt.get("error")
            time.sleep(retry_backoff)
        conf = self.query_order(symbol, coid=coid)
        if conf and conf.get("ok"):
            return conf
        if conf and conf.get("status") == "NOT_FOUND":
            return {"ok": False, "status": "REJECTED",
                    "error": "回查未找到该订单，判定未成交", "coid": coid}
        return {"ok": False, "status": "ERROR", "retryable": True,
                "error": last_err or "重试耗尽", "coid": coid}


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
        """按成交更新持仓；反向成交做净额（平仓/减仓正确清零），同向做加权。

        修复：原实现总是覆盖为同向持仓，导致平仓后仪表盘残留反向幽灵持仓。
        """
        direction = Direction.LONG if o.side == "BUY" else Direction.SHORT
        existing = self.positions.get(o.symbol)
        if existing and existing.direction != direction:
            remaining = existing.qty - o.filled_qty
            if remaining <= 1e-9:
                del self.positions[o.symbol]
            else:
                existing.qty = remaining
                existing.initial_qty = remaining
            return
        if existing and existing.direction == direction:
            total = existing.qty + o.filled_qty
            if total > 1e-9:
                existing.entry_price = (existing.entry_price * existing.qty
                                        + p * o.filled_qty) / total
            existing.qty = total
            existing.initial_qty = total
            return
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
