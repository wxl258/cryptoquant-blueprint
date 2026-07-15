from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from cryptoquant_auto.testnet_runner import _calc_pnl, FEE_RATE_TAKER
from cryptoquant_auto.models import Direction, Position


def _adapter_with_position(direction, entry, qty, symbol="BTC"):
    """构造一个只持有一笔仓的假 adapter（含 fills / query_positions）。"""
    pos = Position(symbol=symbol, direction=direction, entry_price=entry,
                   qty=qty, initial_qty=qty, sl_price=0.0, tp1_price=0.0,
                   tp2_price=0.0, entry_coid="", signal_id="")
    a = SimpleNamespace()
    a.fills = []
    a.query_positions = lambda: [pos]
    return a, pos


def _patch_fill_io(monkeypatch):
    """把跨 run 的 fill 历史 IO 替换成内存字典，避免单测写盘污染。"""
    store = {"h": []}
    monkeypatch.setattr("cryptoquant_auto.testnet_runner._load_fill_history",
                        lambda: store["h"])
    monkeypatch.setattr("cryptoquant_auto.testnet_runner._save_fill_history",
                        lambda h: store.__setitem__("h", list(h)))
    return store


def test_calc_pnl_unrealized_nonzero_long_profit(monkeypatch):
    # 【P0-6 回归】upnl 必须由实时 mark 价计算，不得再被硬编码为 0.0（P1 修复点）。
    _patch_fill_io(monkeypatch)
    a, _ = _adapter_with_position(Direction.LONG, entry=100.0, qty=0.5, symbol="BTC")
    res = _calc_pnl(a, {"BTC": 110.0})          # 多单浮盈 (110-100)*0.5 = +5.0
    assert res["unrealized_pnl"] != 0.0, "upnl 被硬编码为 0，P1 修复已回归"
    assert abs(res["unrealized_pnl"] - 5.0) < 1e-6
    assert res["positions"][0]["unrealized_pnl"] == 5.0


def test_calc_pnl_unrealized_short_profit(monkeypatch):
    _patch_fill_io(monkeypatch)
    a, _ = _adapter_with_position(Direction.SHORT, entry=100.0, qty=0.5, symbol="ETH")
    res = _calc_pnl(a, {"ETH": 90.0})           # 空单浮盈 (100-90)*0.5 = +5.0
    assert res["unrealized_pnl"] != 0.0
    assert abs(res["unrealized_pnl"] - 5.0) < 1e-6


def test_calc_pnl_no_position_zero_upnl(monkeypatch):
    _patch_fill_io(monkeypatch)
    a = SimpleNamespace()
    a.fills = []
    a.query_positions = lambda: []
    res = _calc_pnl(a, {})
    assert res["unrealized_pnl"] == 0.0
    assert res["positions"] == []


def _sample_fills():
    # 先买后卖、低买高卖 → 正确已实现应为 +20（此前符号反转时为 -20）。
    # ts 用当前时间，确保 _daily_realized_pnl 的自然日过滤能命中（生产即当日成交）。
    now = time.time()
    return [
        {"coid": "b1", "symbol": "SOL", "side": "BUY", "price": 100.0,
         "qty": 1.0, "ts": now - 10},
        {"coid": "s1", "symbol": "SOL", "side": "SELL", "price": 120.0,
         "qty": 1.0, "ts": now},
    ]


def test_calc_pnl_realized_sign_correct(monkeypatch):
    # 【P0 符号修复回归】买@100 卖@120 必须 +20，不得再是 -20。
    store = _patch_fill_io(monkeypatch)
    a = SimpleNamespace()
    a.fills = []
    a.query_positions = lambda: []
    assert _calc_pnl(a, {})["realized_pnl"] == 0.0   # 尚无成交
    for f in _sample_fills():                         # 累积写入买/卖两笔
        store["h"].append(f)
        _calc_pnl(a, {})
    r2 = _calc_pnl(a, {})
    # 【P2-2】含双边手续费：净额 + 手续费 = 毛利 20；盈利日仍为正（舍入不变式）。
    assert r2["realized_pnl"] > 0, "盈利日已实现盈亏应为正"
    assert abs((r2["realized_pnl"] + r2["total_fees"]) - 20.0) < 1e-9, "净额+手续费应等于毛利20"
    assert r2["total_fills"] == 2
    assert r2["total_fees"] > 0


def test_daily_realized_pnl_sign_correct(monkeypatch):
    # 【P0 符号修复回归】_daily_realized_pnl 是 KillSwitch 日亏熔断的输入；
    # 盈利日必须返回正值，否则会误触发 L1 暂停新开。
    monkeypatch.setattr("cryptoquant_auto.testnet_runner._load_fill_history",
                        lambda: _sample_fills())
    from cryptoquant_auto.testnet_runner import _daily_realized_pnl
    assert _daily_realized_pnl() > 0, "日亏熔断输入符号反：盈利日被判为亏损"


def test_calc_pnl_fees_deducted(monkeypatch):
    # 【P2-2】已实现盈亏必须扣除双边手续费；total_fees 暴露且 > 0。
    store = _patch_fill_io(monkeypatch)
    a = SimpleNamespace()
    a.fills = []
    a.query_positions = lambda: []
    for f in _sample_fills():
        store["h"].append(f)
    r = _calc_pnl(a, {})
    gross = 120.0 - 100.0  # 低买高卖毛利 20.0
    # 净额 + 手续费 = 毛利（round 到分后仍需成立，舍入不变式）
    assert abs((r["realized_pnl"] + r["total_fees"]) - gross) < 1e-9
    assert r["total_fees"] > 0
    # 手续费 = (100+120)*1*FEE_RATE_TAKER，约吞 0.088 USDT（对分位舍入宽容）
    assert abs(r["total_fees"] - (100.0 + 120.0) * 1.0 * FEE_RATE_TAKER) < 0.01
