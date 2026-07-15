"""P3-3 · 核心/引擎模块覆盖补漏：order_builder / reconcile / MockAdapter / metacontroller / generator。

包括 core/order_builder (22%) + core/reconcile (41%) + adapters/mock (20%) +
core/metacontroller (45%) + signals/generator.candidate_to_signal (16%)。
均为纯算法或轻量依赖模块。
"""
import numpy as np
import pytest
from pytest import approx
from unittest.mock import MagicMock

from cryptoquant_auto.models import Signal, Order, OrderType, OrderStatus, Direction, Position
from cryptoquant_auto.core.order_builder import (
    kelly_size, build_entry_order, build_reduce_order, build_tp_sl_orders,
)
from cryptoquant_auto.core.reconcile import reconcile, ReconcileReport
from cryptoquant_auto.adapters.mock import MockAdapter
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.risk.kelly import KellyConfig
from cryptoquant_auto.core.metacontroller import (
    BayesianMetacontroller, opinion_from_candidate, Opinion, MetaDecision,
    LONG, SHORT, HOLD,
)
from cryptoquant_auto.signals.engine import SignalCandidate
from cryptoquant_auto.signals.generator import candidate_to_signal


# Helper
def _sig(entry=30000, direction=Direction.LONG, atr=100, sl=29000,
         tp1=31000, tp2=32000, confidence=0.7, symbol="BTC") -> Signal:
    return Signal(
        symbol=symbol, tf="1H", direction=direction, entry=entry,
        sl=sl, tp1=tp1, tp2=tp2, rr=2.0, confidence=confidence,
        signal_id="s1", atr=atr,
    )


# ============================================================================
# core/order_builder.py
# ============================================================================

class TestOrderBuilder:
    def test_kelly_size_basic(self):
        """kelly_size 返回正数数量（默认保守缺省）。"""
        sig = _sig()
        cfg = GateConfig(equity=100_000, single_cap_pct=0.04)
        qty = kelly_size(sig, cfg)
        assert qty > 0

    def test_kelly_size_with_calibrated(self):
        """已校准的 KellyConfig 使用注入值。"""
        sig = _sig()
        cfg = GateConfig(equity=100_000, single_cap_pct=0.04)
        kc = KellyConfig(win_rate_est=0.4, payoff_ratio_est=2.0,
                         kelly_frac=0.25, calibrated=True)
        qty = kelly_size(sig, cfg, kc=kc)
        assert qty > 0

    def test_kelly_size_with_7step(self):
        """注入 score/adx/atr_pct 时启用 7 步链二次缩放。"""
        sig = _sig()
        cfg = GateConfig(equity=100_000, single_cap_pct=0.04)
        qty = kelly_size(sig, cfg, score=5, adx=30, atr_pct=2.0)
        assert qty > 0

    def test_build_entry_order(self):
        """build_entry_order 产生入场限价单。"""
        sig = _sig()
        cfg = GateConfig(equity=100_000)
        order = build_entry_order(sig, cfg)
        assert isinstance(order, Order)
        assert order.otype == OrderType.ENTRY
        assert order.side == "BUY"
        assert order.price == 30000
        assert order.post_only is True

    def test_build_entry_order_short(self):
        """空头入场单 side=SELL。"""
        sig = _sig(direction=Direction.SHORT)
        cfg = GateConfig(equity=100_000)
        order = build_entry_order(sig, cfg)
        assert order.side == "SELL"
        assert order.price == 30000

    def test_build_reduce_order(self):
        """build_reduce_order 产生反向减仓单，数量按 reduce_to 比例。"""
        sig = _sig()
        cfg = GateConfig(equity=100_000)
        entry = build_entry_order(sig, cfg)
        reduce = build_reduce_order(sig, entry, reduce_to=0.5)
        assert reduce.otype == OrderType.REDUCE
        assert reduce.side == "SELL"  # 平多头
        assert reduce.qty == pytest.approx(entry.qty * 0.5)
        assert reduce.parent_coid == entry.coid

    def test_build_tp_sl_orders(self):
        """build_tp_sl_orders 产生 3 张子单：tp1/tp2/sl。"""
        sig = _sig()
        cfg = GateConfig(equity=100_000)
        entry = build_entry_order(sig, cfg)
        children = build_tp_sl_orders(sig, entry, atr=100)
        assert len(children) == 3
        legs = [c.leg for c in children]
        assert "tp1" in legs and "tp2" in legs and "sl" in legs
        assert children[0].qty == entry.qty * 0.5  # tp1
        assert children[2].qty == entry.qty         # sl
        assert children[0].otype == OrderType.TP
        assert children[2].otype == OrderType.SL
        assert all(c.parent_coid == entry.coid for c in children)


# ============================================================================
# core/reconcile.py
# ============================================================================

class TestReconcile:
    @staticmethod
    def _pos(symbol="BTC", direction=Direction.LONG, qty=1.0, entry=30000) -> Position:
        return Position(
            symbol=symbol, direction=direction, entry_price=entry,
            qty=qty, initial_qty=qty, sl_price=entry*0.97,
            tp1_price=entry*1.03, tp2_price=entry*1.05,
            entry_coid=f"c_{symbol}",
        )

    def test_reconcile_clean(self):
        rep = reconcile([self._pos()], [self._pos()])
        assert rep.clean

    def test_reconcile_over(self):
        rep = reconcile([self._pos(qty=1.0)], [self._pos(qty=2.0)])
        assert not rep.clean
        assert rep.items[0].action == "OVER"

    def test_reconcile_under(self):
        rep = reconcile([self._pos(qty=2.0)], [self._pos(qty=1.0)])
        assert not rep.clean
        assert rep.items[0].action == "UNDER"

    def test_reconcile_dir_flip(self):
        rep = reconcile([self._pos(direction=Direction.LONG)],
                        [self._pos(direction=Direction.SHORT)])
        assert not rep.clean
        assert rep.items[0].action == "DIR_FLIP"


# ============================================================================
# adapters/mock.py
# ============================================================================

class TestMockAdapter:
    def test_mock_submit_entry(self):
        mock = MockAdapter(equity=100_000)
        order = build_entry_order(_sig(), GateConfig(equity=100_000))
        out = mock.submit(order)
        assert out.status == OrderStatus.OPEN

    def test_mock_submit_reduce(self):
        mock = MockAdapter(equity=100_000)
        sig = _sig(); cfg = GateConfig(equity=100_000)
        reduce = build_reduce_order(sig, build_entry_order(sig, cfg))
        out = mock.submit(reduce)
        assert out.status == OrderStatus.OPEN

    def test_mock_cancel_missing(self):
        """未提交的 coid 取消返回 False。"""
        assert MockAdapter().cancel("nonexistent") is False

    def test_mock_query_open(self):
        assert MockAdapter().query_open() == []

    def test_mock_query_positions_empty(self):
        assert MockAdapter().query_positions() == []

    def test_mock_fault_maintenance(self):
        """maintenance 模式 submit 返回 REJECTED。"""
        mock = MockAdapter(fault_mode="maintenance")
        out = mock.submit(build_entry_order(_sig(), GateConfig(equity=100_000)))
        assert out.status == OrderStatus.REJECTED

    def test_mock_simulate_market(self):
        MockAdapter().simulate_market({"BTC": 31000.0})  # 不抛


# ============================================================================
# core/metacontroller.py
# ============================================================================

class TestMetacontroller:
    def test_opinion_from_candidate_long(self):
        c = SignalCandidate(symbol="BTC", direction="做多", score=7,
                          min_score_adj=5.0, conds=["adx_strong"])
        o = opinion_from_candidate(c)
        assert o.action() == LONG
        assert o.confidence > 0.5

    def test_opinion_not_passed(self):
        """未通过(score<min_score_adj)的候选被压低并偏向观望。"""
        c = SignalCandidate(symbol="BTC", direction="做多", score=4,
                          min_score_adj=5.0, conds=["weak"])
        assert not c.passed  # score < min_score_adj → passed=False
        o = opinion_from_candidate(c)
        assert "weak" in o.rationale

    def test_entropy_uniform(self):
        assert BayesianMetacontroller().entropy(np.array([1/3, 1/3, 1/3])) == approx(np.log(3), rel=0.01)

    def test_entropy_certain(self):
        assert BayesianMetacontroller().entropy(np.array([1.0, 0.0, 0.0])) < 1e-6

    def test_fuse(self):
        mc = BayesianMetacontroller()
        o1 = Opinion(symbol="BTC", probs=np.array([0.7, 0.1, 0.2]), source="e1")
        o2 = Opinion(symbol="BTC", probs=np.array([0.6, 0.2, 0.2]), source="e2")
        assert np.argmax(mc.fuse([o1, o2])) == LONG

    def test_decide_normal(self):
        """低不确定度 → 正常执行（非降级观望）。"""
        d = BayesianMetacontroller(uncertainty_thresh=0.55).decide(
            [Opinion(symbol="BTC", probs=np.array([0.95, 0.025, 0.025]), source="ta")])
        assert d.action == LONG
        assert not d.degraded

    def test_decide_high_uncertainty_degraded(self):
        d = BayesianMetacontroller(uncertainty_thresh=0.3).decide(
            [Opinion(symbol="BTC", probs=np.array([0.8, 0.1, 0.1]), source="noisy")])
        assert d.action == HOLD
        assert d.degraded

    def test_decide_rationale_included(self):
        d = BayesianMetacontroller().decide(
            [Opinion(symbol="BTC", probs=np.array([0.7, 0.1, 0.2]),
                     source="ta", rationale="adx>30")])
        assert "adx>30" in d.rationale


# ============================================================================
# signals/generator.py — candidate_to_signal
# ============================================================================

class TestGenerator:
    def test_candidate_to_signal_long(self):
        cand = SignalCandidate(symbol="BTC", direction="做多", score=7,
                              min_score_adj=5.0, conds=["adx_strong"],
                              atr=100, atr_pct=2.0, adx=35, rr=2.0)
        sig = candidate_to_signal(cand, price=30000, tf="1H")
        assert sig.direction == Direction.LONG
        assert sig.entry == 30000
        assert sig.atr == 100

    def test_candidate_to_signal_short(self):
        cand = SignalCandidate(symbol="ETH", direction="做空", score=6,
                              min_score_adj=5.0, conds=["mtf"])
        sig = candidate_to_signal(cand, price=2000, tf="1H")
        assert sig.direction == Direction.SHORT
        assert sig.symbol == "ETH"
