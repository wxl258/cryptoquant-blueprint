from __future__ import annotations

from types import SimpleNamespace

import requests
import pytest

from cryptoquant_auto.adapters.binance_testnet import BinanceTestnetAdapter
from cryptoquant_auto.models import Direction
from cryptoquant_auto.risk.constitution import TradingConstitution
from cryptoquant_auto.risk.kill_switch import KillSwitch
from cryptoquant_auto.risk.gate import GateConfig


def _adapter():
    return BinanceTestnetAdapter("k", "s", constitution=TradingConstitution(live_capital=False))


def _resp(code=200, data=None):
    return SimpleNamespace(status_code=code, json=lambda: data)


def test_filled_ok():
    a = _adapter()
    a.sess = SimpleNamespace(
        post=lambda *x, **k: _resp(200, {"orderId": 1, "status": "FILLED",
            "executedQty": "0.1", "avgPrice": "100", "clientOrderId": "c1", "symbol": "BTCUSDT"}),
        get=lambda *x, **k: _resp(200, {"orderId": 1, "status": "FILLED", "executedQty": "0.1",
            "avgPrice": "100", "side": "BUY", "clientOrderId": "c1"}))
    r = a.submit_market("BTC", "BUY", 0.1, price=100)
    assert r["ok"] is True and r["status"] == "FILLED"
    assert len(a.positions) == 1


def test_new_then_confirm_filled():
    a = _adapter()
    a.sess = SimpleNamespace(
        post=lambda *x, **k: _resp(200, {"orderId": 2, "status": "NEW",
            "executedQty": "0", "avgPrice": "0", "clientOrderId": "c2"}),
        get=lambda *x, **k: _resp(200, {"orderId": 2, "status": "FILLED", "executedQty": "0.2",
            "avgPrice": "50", "side": "SELL", "clientOrderId": "c2"}))
    r = a.submit_market("ETH", "SELL", 0.2, price=50)
    assert r["ok"] and r["status"] == "FILLED"


def test_business_reject_no_retry():
    calls = {"n": 0}
    a = _adapter()

    def post(*x, **k):
        calls["n"] += 1
        return _resp(400, {"code": -1111, "msg": "precision"})

    a.sess = SimpleNamespace(post=post, get=lambda *x, **k: _resp(400, {"code": -2013}))
    r = a.submit_market("SOL", "BUY", 0.1, price=10)
    assert r["ok"] is False and r["status"] == "REJECTED"
    assert calls["n"] == 1


def test_network_retry_then_success():
    seq = {"n": 0}
    a = _adapter()

    def post(*x, **k):
        seq["n"] += 1
        if seq["n"] == 1:
            raise requests.RequestException("timeout")
        return _resp(200, {"orderId": 3, "status": "FILLED", "executedQty": "1",
            "avgPrice": "1", "clientOrderId": "c3"})

    a.sess = SimpleNamespace(post=post,
        get=lambda *x, **k: _resp(200, {"orderId": 3, "status": "FILLED", "executedQty": "1",
            "avgPrice": "1", "side": "BUY", "clientOrderId": "c3"}))
    r = a.submit_market("BNB", "BUY", 1, price=1)
    assert r["ok"] and seq["n"] == 2


def test_network_exhausted_not_found():
    a = _adapter()

    def post(*x, **k):
        raise requests.RequestException("down")

    a.sess = SimpleNamespace(post=post,
        get=lambda *x, **k: _resp(400, {"code": -2013, "msg": "order not found"}))
    r = a.submit_market("XRP", "SELL", 1, price=1)
    assert r["ok"] is False and r["status"] == "REJECTED"


def test_refresh_positions_parses_real():
    a = _adapter()
    a.sess = SimpleNamespace(
        get=lambda *x, **k: _resp(200, [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "entryPrice": "30000", "leverage": "2"},
            {"symbol": "ETHUSDT", "positionAmt": "-0.2", "entryPrice": "1800", "leverage": "2"},
            {"symbol": "SOLUSDT", "positionAmt": "0", "entryPrice": "0", "leverage": "2"},
        ]),
        post=lambda *x, **k: _resp(200, {}))
    pos = a.refresh_positions()
    assert len(pos) == 2
    m = {p.symbol: p for p in pos}
    assert m["BTC"].direction == Direction.LONG and m["BTC"].qty == 0.5
    assert m["ETH"].direction == Direction.SHORT and m["ETH"].qty == 0.2
    # query_positions 现在返回真实持仓
    assert len(a.query_positions()) == 2


def test_refresh_open_orders_parses_real():
    a = _adapter()
    a.sess = SimpleNamespace(
        get=lambda *x, **k: _resp(200, [
            {"symbol": "BTCUSDT", "side": "BUY", "price": "30000",
             "origQty": "0.5", "clientOrderId": "x1"},
        ]),
        post=lambda *x, **k: _resp(200, {}))
    ops = a.refresh_open_orders()
    assert len(ops) == 1 and ops[0].symbol == "BTC" and ops[0].side == "BUY"
    assert len(a.query_open()) == 1


def test_open_position_nets_on_close():
    # 修复：平仓后不应残留反向幽灵持仓
    a = _adapter()
    a.sess = SimpleNamespace(
        post=lambda *x, **k: _resp(200, {"orderId": 9, "status": "FILLED",
            "executedQty": "0.1", "avgPrice": "100", "clientOrderId": "c9", "symbol": "BTCUSDT"}),
        get=lambda *x, **k: _resp(200, {}))
    # 先开多
    a.submit_market("BTC", "BUY", 0.1, price=100)
    assert len(a.positions) == 1 and a.positions["BTC"].direction == Direction.LONG
    # 再平多(卖同等量) → 持仓应清零
    a.submit_market("BTC", "SELL", 0.1, price=100)
    assert "BTC" not in a.positions, "平仓后残留幽灵持仓"


def test_sync_skips_when_refresh_fails():
    # 关联隐患修复：持仓/挂单刷新失败(返回 None)时, 必须跳过本轮,
    # 绝不能用空内存持仓盲目重复开仓。
    from cryptoquant_auto.testnet_runner import sync_positions
    a = _adapter()
    a.refresh_positions = lambda: None
    a.refresh_open_orders = lambda: []
    called = []
    a.submit_market = lambda *x, **k: called.append((x, k)) or {"ok": True, "status": "FILLED"}
    trades = sync_positions(a, {"XRP": "SHORT", "SOL": "LONG"}, {"XRP": 1.1, "SOL": 75.0},
                            KillSwitch(), GateConfig(equity=10_000_000, enforce_gate_b=False))
    assert trades == [], "刷新失败应跳过本轮"
    assert called == [], "刷新失败不应触发任何下单"


def test_sync_no_duplicate_existing_position():
    # 关联隐患修复：刷新拉到既有 XRP 空头, 同向下一信号不应重复开仓;
    # 新信号 SOL LONG 应正常开仓。
    from cryptoquant_auto.testnet_runner import sync_positions
    from cryptoquant_auto.models import Position
    a = _adapter()
    xrp = Position(symbol="XRP", direction=Direction.SHORT, entry_price=1.07,
                   qty=9685, initial_qty=9685, sl_price=0.0, tp1_price=0.0,
                   tp2_price=0.0, entry_coid="", signal_id="")
    a.positions = {"XRP": xrp}
    a.refresh_positions = lambda: [xrp]
    a.refresh_open_orders = lambda: []
    called = []
    a.submit_market = lambda *x, **k: called.append((x, k)) or {"ok": True, "status": "FILLED"}
    trades = sync_positions(a, {"XRP": "SHORT", "SOL": "LONG"}, {"XRP": 1.1, "SOL": 75.0},
                            KillSwitch(), GateConfig(equity=10_000_000, enforce_gate_b=False))
    opened = [t["symbol"] for t in trades]
    assert "XRP" not in opened, "既有同向持仓不应重复开仓"
    assert "SOL" in opened, "新信号应开仓"


