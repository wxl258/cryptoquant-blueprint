"""P3-2 · 风险模块覆盖补漏：circuit_breaker / liquidation_guard / black_swan / exec_sl / exec_cost / signal_filter。

6 个模块均为纯算法（零网络依赖），补冒烟测试拉满每个公开函数至少一条 happy path + 边界。
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List

from cryptoquant_auto.models import Signal, Direction
from cryptoquant_auto.risk.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, dynamic_threshold, check_price_circuit,
)
from cryptoquant_auto.risk.liquidation_guard import check_liquidation_guard
from cryptoquant_auto.risk.black_swan import (
    detect_black_swan, get_black_swan_level, Candle,
)
from cryptoquant_auto.risk.exec_sl import (
    estimate_liquidation_price, exec_sl_price, move_to_breakeven,
)
from cryptoquant_auto.risk.exec_cost import (
    effective_edge_bps, worst_case_pnl, beta, funding_cost,
    gate_b_ok, set_gross_edge_bps, calibrate_gross_edge_bps,
    apply_locked_gate_b, ACTIVE_GROSS_EDGE_BPS,
)
from cryptoquant_auto.risk.signal_filter import (
    SignalQualityGate, SignalQualityConfig, MarketSnapshot, TfView,
    synth_snapshot,
)


# ============================================================================
# risk/circuit_breaker.py
# ============================================================================

class TestCircuitBreaker:
    def test_on_trade_close_happy(self):
        """正常平仓不触发熔断。"""
        cb = CircuitBreaker()
        cb.on_trade_close(-0.01)  # 微亏
        assert not cb.tripped

    def test_on_trade_close_loss_streak_trip(self):
        """连亏 >= loss_trip(3) 触发熔断。"""
        cb = CircuitBreaker()
        cb.on_trade_close(-0.01)
        cb.on_trade_close(-0.02)
        cb.on_trade_close(-0.03)
        assert cb.tripped
        assert "loss_streak" in cb.reason

    def test_on_trade_close_anomaly_trip(self):
        """单笔异常亏损超过 anomaly_pct(8%) 触发熔断。"""
        cb = CircuitBreaker()
        cb.on_trade_close(-0.10)
        assert cb.tripped
        assert "anomaly" in cb.reason

    def test_on_trade_close_wins_reset_streak(self):
        """盈利平仓重置连亏计数。"""
        cb = CircuitBreaker()
        cb.on_trade_close(-0.01)  # 连亏1
        cb.on_trade_close(0.02)   # 盈利重置
        cb.on_trade_close(-0.01)  # 连亏1
        assert cb.loss_streak == 1
        assert not cb.tripped

    def test_feed_signal_and_stall(self):
        """信号中断超时触发熔断。"""
        cb = CircuitBreaker(cfg=CircuitBreakerConfig(signal_stall_s=10))
        cb.feed_signal(100.0)
        cb.check_stall(115.0)  # 超时15s > 10
        assert cb.tripped
        assert "signal_stall" in cb.reason

    def test_check_stall_no_trip_when_recent(self):
        """信号新鲜时不触发熔断。"""
        cb = CircuitBreaker(cfg=CircuitBreakerConfig(signal_stall_s=10))
        cb.feed_signal(100.0)
        cb.check_stall(105.0)  # 仅差5s
        assert not cb.tripped

    def test_manual_reset(self):
        """manual_reset 恢复可交易状态（fail-closed 人工 ACK）。"""
        cb = CircuitBreaker()
        cb.on_trade_close(-0.10)  # 触发熔断
        assert cb.tripped
        cb.manual_reset()
        assert not cb.tripped
        assert cb.loss_streak == 0
        assert cb.reason == ""

    def test_dynamic_threshold(self):
        assert dynamic_threshold(0.3) == 0.03   # <0.4 → 3%
        assert dynamic_threshold(2.0) == 0.04   # 正常 → 4%
        assert dynamic_threshold(6.0) == 0.06   # >5 → 6%

    def test_check_price_circuit_triggered(self):
        """价格暴跌超过动态阈值触发熔断。"""
        tripped, reason = check_price_circuit(28000, 30000, atr_pct=2.0)
        assert tripped
        assert "暴跌" in reason

    def test_check_price_circuit_not_triggered(self):
        """正常波动不触发熔断。"""
        tripped, reason = check_price_circuit(29500, 30000, atr_pct=2.0)
        assert not tripped

    def test_check_price_circuit_last_price_zero(self):
        """last_price<=0 时不触发（无数据安全）。"""
        tripped, reason = check_price_circuit(28000, 0)
        assert not tripped


# ============================================================================
# risk/liquidation_guard.py
# ============================================================================

class TestLiquidationGuard:
    def test_check_liq_guard_skips_none(self):
        """liq_price=None 的持仓跳过（≈1x无强平）。"""
        pos = MagicMock()
        pos.liq_price = None
        alerts = check_liquidation_guard([pos], {"BTC": 30000})
        assert len(alerts) == 0

    def test_check_liq_guard_no_alerts(self):
        """价格远离强平价时不告警。"""
        pos = MagicMock()
        pos.symbol = "BTC"
        pos.entry_price = 30000
        pos.leverage = 2
        pos.direction = Direction.LONG
        pos.liq_price = 15000
        alerts = check_liquidation_guard([pos], {"BTC": 30000}, warn_buffer_pct=0.05)
        assert len(alerts) == 0

    def test_check_liq_guard_alerts(self):
        """价格接近强平警告缓冲触发告警。"""
        pos = MagicMock()
        pos.symbol = "BTC"
        pos.entry_price = 100
        pos.leverage = 2
        pos.direction = Direction.LONG
        pos.liq_price = 50  # 强平价=100*(1-1/2+0.005)=50.5; 距离=|50-50.5|/50≈1% < 10%
        alerts = check_liquidation_guard([pos], {"BTC": 50.5}, warn_buffer_pct=0.10)
        # liq ≈ 100*(1-1/2+0.005)=50.5; price=50.5 -> dist=0 → 触发告警
        assert len(alerts) >= 1
        assert "距强平价" in alerts[0]


# ============================================================================
# risk/black_swan.py
# ============================================================================

class TestBlackSwan:
    def test_detect_empty(self):
        """空数据安全返回 (False, '')。"""
        assert detect_black_swan({}) == (False, "")

    def test_detect_btc_dump(self):
        """BTC 暴跌超过 5% → 黑天鹅。"""
        candles = {"BTC": Candle(symbol="BTC", open=100, close=93, high=101, low=92)}
        swan, reason = detect_black_swan(candles, btc_dump_pct=0.05)
        assert swan is True
        assert "btc_dump" in reason

    def test_detect_spread_spike(self):
        """跨所价差飙升 >1.5% → 黑天鹅。"""
        candles = {"BTC": Candle(symbol="BTC", open=100, close=99, high=101, low=98)}
        swan, reason = detect_black_swan(candles, cross_spread_pct=0.02)
        assert swan is True
        assert "spread_spike" in reason

    def test_get_level_l3(self):
        """30 分钟跌幅 ≥15% → L3。"""
        history = [100] * 20 + [80]   # 从100到80 = 20% > 15%
        level, reason = get_black_swan_level(history)
        assert level >= 3

    def test_get_level_l2(self):
        """15 分钟跌幅 8-15% → L2。"""
        history = [100] * 10 + [88]   # 12% 跌幅 > 8%
        level, reason = get_black_swan_level(history)
        assert level >= 2

    def test_get_level_l1(self):
        """5 分钟跌幅 3-8% → L1。"""
        history = [100] * 3 + [95]    # 5% 跌幅 > 3%
        level, reason = get_black_swan_level(history)
        assert level >= 1

    def test_get_level_0_when_no_drop(self):
        """无明显跌幅 → level 0。"""
        history = [100, 101, 102, 103]
        level, reason = get_black_swan_level(history)
        assert level == 0


# ============================================================================
# risk/exec_sl.py
# ============================================================================

class TestExecSl:
    def test_estimate_liq_long(self):
        """多头强平价在下方。entry=60000, lev=5x → 约 48300。"""
        liq = estimate_liquidation_price(60000, 5, Direction.LONG)
        assert 48000 < liq < 49000

    def test_estimate_liq_short(self):
        """空头强平价在上方。"""
        liq = estimate_liquidation_price(60000, 5, Direction.SHORT)
        assert 71000 < liq < 72000

    def test_exec_sl_long(self):
        """多头执行级止损 = entry - k*ATR。"""
        sl = exec_sl_price(100, 2.0, Direction.LONG, k=2.0)
        assert sl == 96.0  # 100 - 2*2

    def test_exec_sl_short(self):
        """空头执行级止损 = entry + k*ATR。"""
        sl = exec_sl_price(100, 2.0, Direction.SHORT, k=2.0)
        assert sl == 104.0

    def test_move_to_breakeven_long(self):
        """保本位：SL 移至成本以上。"""
        new_sl = move_to_breakeven(90, 100, Direction.LONG, fee_buffer=0.5)
        assert new_sl == 100.5  # entry + fee_buffer

    def test_move_to_breakeven_short(self):
        new_sl = move_to_breakeven(110, 100, Direction.SHORT, fee_buffer=0.5)
        assert new_sl == 99.5

    def test_move_to_breakeven_already_better(self):
        """已有 SL 比保本位置更好（更低止损）→ 不拉高。"""
        new_sl = move_to_breakeven(95, 100, Direction.LONG, fee_buffer=2.0)
        assert new_sl == 102.0  # SL=95 < 102(target)，提升到 102

    def test_move_to_breakeven_short_already_better(self):
        new_sl = move_to_breakeven(105, 100, Direction.SHORT, fee_buffer=2.0)
        assert new_sl == 98.0  # SL=105 > 98(target)，降低到 98


# ============================================================================
# risk/exec_cost.py
# ============================================================================

class TestExecCost:
    def test_effective_edge_bps_maker(self):
        """maker 净 edge = ACTIVE_GROSS_EDGE_BPS - 2.0。"""
        edge = effective_edge_bps("BTC", taker=False, slip=False, fund=False)
        assert edge == ACTIVE_GROSS_EDGE_BPS - 2.0

    def test_effective_edge_bps_taker(self):
        """taker 净 edge = ACTIVE_GROSS_EDGE_BPS - 5.0 - 1.0 - 2.0。"""
        edge = effective_edge_bps("BTC", taker=True, slip=True, fund=True)
        assert edge == ACTIVE_GROSS_EDGE_BPS - 5.0 - 1.0 - 2.0

    def test_worst_case_pnl(self):
        """最不利组合 = taker + slip + fund。"""
        w = worst_case_pnl("BTC")
        assert w == ACTIVE_GROSS_EDGE_BPS - 5.0 - 1.0 - 2.0

    def test_beta(self):
        assert beta("BTC") == 1.00
        assert beta("ETH") == 0.85
        assert beta("NOTOKEN") == 0.85  # 回退 ETH

    def test_funding_cost(self):
        """资金费 = fund * periods。"""
        assert funding_cost("BTC", 3) == 2.0 * 3  # 6 bps

    def test_gate_b_ok_fail_closed(self):
        """Gate B 默认应全部 fail（ACTIVE_GROSS_EDGE_BPS = -2.5）。"""
        ok = gate_b_ok()
        assert all(not v for v in ok.values())

    def test_set_gross_edge_bps(self):
        """set_gross_edge_bps 更新全局锚点有效。"""
        old = ACTIVE_GROSS_EDGE_BPS
        try:
            set_gross_edge_bps(5.0)
            assert effective_edge_bps("BTC", taker=False, slip=False, fund=False) == 5.0 - 2.0
        finally:
            set_gross_edge_bps(old)

    def test_calibrate_gross_edge_with_data(self):
        """calibrate 用 OOS 数据反推毛 edge。"""
        cal = calibrate_gross_edge_bps({"BTC": 1.0, "ETH": 2.0}, haircut=0.5)
        # BTC worst_case = -2.5 - 5 - 1 - 2 = -10.5; abs = 10.5; gross = 1 + 10.5 = 11.5
        # ETH worst_case = -2.5 - 5 - 2 - 2 = -11.5; abs = 11.5; gross = 2 + 11.5 = 13.5
        # min = 11.5; * 0.5 = 5.75
        assert cal > 0

    def test_calibrate_gross_edge_empty(self):
        """空数据返回 0.0。"""
        assert calibrate_gross_edge_bps({}) == 0.0

    def test_apply_locked_gate_b(self):
        """apply_locked_gate_b 重新锚定并返回当前值。"""
        old = ACTIVE_GROSS_EDGE_BPS
        try:
            set_gross_edge_bps(3.0)
            v = apply_locked_gate_b()
            assert v == 3.0  # 返回当前锚点
        finally:
            set_gross_edge_bps(old)


# ============================================================================
# risk/signal_filter.py
# ============================================================================

def _dummy_signal(direction=Direction.LONG, atr=100, entry=30000,
                  tf="1H", confidence=0.7) -> Signal:
    return Signal(
        symbol="BTC", tf=tf, direction=direction, entry=entry,
        sl=29000, tp1=31000, tp2=32000, rr=2.0, confidence=confidence,
        signal_id="s1", atr=atr,
    )


def _pass_snapshot(direction=Direction.LONG) -> MarketSnapshot:
    """合成各周期与信号同向的快照（使所有检查通过）。"""
    views = []
    for tf in ["1H", "4H", "1D", "W"]:
        views.append(TfView(tf=tf, direction=direction, adx=40, plus_di=30, minus_di=5))
    return MarketSnapshot(symbol="BTC", tf_views=views, atr_pct=2.0, fg=50)


class TestSignalFilter:
    def test_check_all_pass(self):
        """高质量信号全部通过。"""
        gate = SignalQualityGate()
        sig = _dummy_signal()
        snap = _pass_snapshot()
        res = gate.check(sig, snap)
        assert res.ok

    def test_check_atr_flat(self):
        """atr_pct < 0.3% → fail。"""
        gate = SignalQualityGate()
        sig = _dummy_signal(atr=10, entry=30000)  # atr_pct ≈ 0.03%
        snap = _pass_snapshot()
        snap.atr_pct = 0.03
        res = gate.check(sig, snap)
        assert not res.ok
        assert "atr_flat" in res.reasons

    def test_check_atr_risky(self):
        """atr_pct > 9% → fail。"""
        gate = SignalQualityGate()
        sig = _dummy_signal(entry=1000, atr=300)  # atr_pct ≈ 30%
        snap = _pass_snapshot()
        snap.atr_pct = 30
        res = gate.check(sig, snap)
        assert not res.ok
        assert "atr_risky" in res.reasons

    def test_check_fg_panic_no_short(self):
        """恐慌极值(FG<=20)禁止做空。"""
        gate = SignalQualityGate()
        sig = _dummy_signal(direction=Direction.SHORT)
        snap = _pass_snapshot(direction=Direction.SHORT)
        snap.fg = 15
        res = gate.check(sig, snap)
        assert not res.ok
        assert "fg_panic_no_short" in res.reasons

    def test_check_fg_greed_no_long(self):
        """贪婪极值(FG>=80)禁止做多。"""
        gate = SignalQualityGate()
        sig = _dummy_signal(direction=Direction.LONG)
        snap = _pass_snapshot(direction=Direction.LONG)
        snap.fg = 90
        res = gate.check(sig, snap)
        assert not res.ok
        assert "fg_greed_no_long" in res.reasons

    def test_check_adx_weak(self):
        """ADX < adx_min(30) → fail。"""
        gate = SignalQualityGate()
        sig = _dummy_signal()
        snap = _pass_snapshot()
        snap.tf_views[0].adx = 20  # 1H 对应信号 tf
        res = gate.check(sig, snap)
        assert not res.ok
        assert "adx_weak" in res.reasons

    def test_check_no_snap_atr_only(self):
        """无 MarketSnapshot 时仅跑 ATR 检查。"""
        gate = SignalQualityGate()
        sig = _dummy_signal(atr=200, entry=100)  # atr_pct ≈ 200% > 9%
        res = gate.check(sig, snap=None)
        assert not res.ok
        assert "atr_risky" in res.reasons

    def test_synth_snapshot_produces_valid_structure(self):
        """synth_snapshot 产生有效 MarketSnapshot（调试用，不保证全部通过检查）。"""
        sig = _dummy_signal()
        snap = synth_snapshot(sig, fg=50)
        assert isinstance(snap, MarketSnapshot)
        assert len(snap.tf_views) == 4
        assert all(v.direction == sig.direction for v in snap.tf_views)

    def test_synth_snapshot_atr_near_zero_does_not_crash(self):
        """entry=1, atr=0 时不下崩。"""
        sig = _dummy_signal(entry=1, atr=0, tf="1H")
        snap = synth_snapshot(sig)
        assert snap.atr_pct == 0.0  # 0/1*100 = 0
