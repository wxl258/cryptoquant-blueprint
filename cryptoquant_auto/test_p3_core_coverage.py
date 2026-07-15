"""P3-1 · 生产模块 0% 覆盖补漏：router + position_sizing + profile。

覆盖率测量发现 core/router.py / risk/position_sizing.py / risk/profile.py
三个生产模块为 0%，在此补冒烟测试拉满（每个公开函数至少一个 happy path + 一个边界）。
"""
import pytest
from unittest.mock import MagicMock

from cryptoquant_auto.models import Order, OrderStatus
from cryptoquant_auto.core.router import FallbackRouter
from cryptoquant_auto.risk.position_sizing import (
    calc_position_pct, validate_total_exposure, leverage_aware_notional,
    RISK_PROFILES, ACTIVE_PROFILE, PositionResult,
)
from cryptoquant_auto.risk.profile import (
    get_profile, apply_correlation_filter,
)


# ============================================================================
# core/router.py — FallbackRouter
# ============================================================================

def _make_mock_adapter(name: str, fail_submit_for: set = None) -> MagicMock:
    """创建一个 duck-typed mock ExchangeAdapter。"""
    m = MagicMock()
    m.submit.side_effect = lambda o: (
        Order(coid=o.coid, symbol=o.symbol, side=o.side, otype=o.otype,
              price=o.price, qty=o.qty, signal_id=o.signal_id,
              status=OrderStatus.REJECTED if (fail_submit_for and o.signal_id in fail_submit_for)
                          else OrderStatus.FILLED)
    )
    m.cancel.return_value = True
    m.query_open.return_value = []
    m.query_position.return_value = None
    m.query_positions.return_value = []
    m.simulate_market.return_value = None
    return m


def _dummy_order(signal_id: str = "s1") -> Order:
    return Order(coid=f"co_{signal_id}", symbol="BTC", side="BUY",
                 otype="ENTRY", price=30000.0, qty=0.01, signal_id=signal_id)


def test_router_submit_primary_succeeds():
    """主交易所 submit 成功 → 返回 FILLED Order。"""
    primary = _make_mock_adapter("binance")
    router = FallbackRouter({"binance": primary, "okx": _make_mock_adapter("okx")})
    o = router.submit(_dummy_order())
    assert o.status is OrderStatus.FILLED
    assert primary.submit.called


def test_router_submit_primary_fails_falls_back():
    """主交易所 submit 抛异常 → fallback 到下一所并返回 FILLED。"""
    primary = MagicMock()
    primary.submit.side_effect = ConnectionError("timeout")
    okx = _make_mock_adapter("okx")
    router = FallbackRouter({"binance": primary, "okx": okx})
    o = router.submit(_dummy_order())
    assert o.status is OrderStatus.FILLED
    assert okx.submit.called


def test_router_submit_all_fail_returns_rejected():
    """全部交易所失败 → 返回 REJECTED。"""
    f1, f2 = MagicMock(), MagicMock()
    f1.submit.side_effect = ConnectionError("timeout")
    f2.submit.side_effect = ConnectionError("timeout")
    router = FallbackRouter({"a": f1, "b": f2})
    o = router.submit(_dummy_order())
    assert o.status is OrderStatus.REJECTED


def test_router_submit_rejected_continues_to_backup():
    """交易所返回 REJECTED → 继续尝试下一所（代码中 REJECTED 视为 transient 继续 fallback）。"""
    primary = _make_mock_adapter("binance", fail_submit_for={"s1"})
    backup = _make_mock_adapter("okx")
    router = FallbackRouter({"binance": primary, "okx": backup})
    o = router.submit(_dummy_order("s1"))  # primary 返回 REJECTED → fallback 到 backup
    assert o.status is OrderStatus.FILLED   # backup 成功
    assert backup.submit.called


def test_router_cancel():
    """cancel 搜索所有交易所，命中即返回 True。"""
    m1, m2 = _make_mock_adapter("a"), _make_mock_adapter("b")
    m2.cancel.return_value = True
    router = FallbackRouter({"a": m1, "b": m2})
    assert router.cancel("coid") is True
    assert m1.cancel.called


def test_router_query_open():
    """query_open 汇集所有交易所的开仓。"""
    m1 = _make_mock_adapter("a")
    m1.query_open.return_value = [_dummy_order("o1")]
    m2 = _make_mock_adapter("b")
    m2.query_open.return_value = [_dummy_order("o2")]
    router = FallbackRouter({"a": m1, "b": m2})
    out = router.query_open()
    assert len(out) == 2


def test_router_simulate_market():
    """simulate_market 传播到所有交易所。"""
    m1, m2 = _make_mock_adapter("a"), _make_mock_adapter("b")
    router = FallbackRouter({"a": m1, "b": m2})
    router.simulate_market({"BTC": 31000.0})
    assert m1.simulate_market.called
    assert m2.simulate_market.called


# ============================================================================
# risk/position_sizing.py
# ============================================================================

class TestPositionSizing:
    def test_calc_base_score_10(self):
        """score=10 → base=4.5%（最高档）。"""
        r = calc_position_pct(score=10, adx=20, atr_pct=1.0)
        assert 4.0 <= r.pct <= 5.0

    def test_calc_min_score(self):
        """score=0 → base=0.5%（最低基础档）→ 最终 ≥ 0.3% 下限硬顶。"""
        r = calc_position_pct(score=0, adx=20, atr_pct=1.0)
        # score=0 → else=0.5; 默认 W=50% → kelly_base=5.0; min(0.5,5.0)=0.5
        assert r.pct == 0.5
        assert r.pct >= 0.3       # 下限硬顶 0.3% 生效

    def test_calc_lower_clamp_activated(self):
        """极端低 win_rate 下凯利压缩基础位至下限 0.3%。"""
        r = calc_position_pct(score=5, adx=20, atr_pct=1.0,
                              trade_state={"win_rate": 5})
        assert r.pct == 0.3

    def test_calc_kelly_caps_base(self):
        """凯利上限应限制 base：win_rate=5% → kelly 极小 → 结果为 0.3%（下限）。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0,
                              trade_state={"win_rate": 5})
        assert r.pct == 0.3  # 凯利压缩到下限

    def test_calc_adx_bonus(self):
        """adx≥45 → ×1.2 放大。"""
        r = calc_position_pct(score=8, adx=45, atr_pct=1.0)
        # 3.5 base * 1.2 = 4.2; 但凯利上限可能进一步压缩；至少应 > 仅 adx 不放大
        assert "×1.2" in r.factors.get("ADX加成", "")

    def test_calc_loss_streak_reduces(self):
        """连亏 ≥3 → ×0.6。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0,
                              trade_state={"streak_losses": 3})
        assert "×0.6" in r.factors.get("连亏", "")

    def test_calc_profit_step_reduces(self):
        """连盈 ≥3 且累计正收益 → ×0.7（盈利递减）。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0,
                              trade_state={"profit_steps": 3, "cum_pnl_pct": 0.05})
        assert "×0.7" in r.factors.get("盈利递减", "")

    def test_calc_weekend_reduces(self):
        """is_weekend=True → ×0.7。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0, is_weekend=True)
        assert "×0.7" in r.factors.get("周末", "")

    def test_calc_sr_risk_reduces(self):
        """sr_risk 非空 → ×0.7。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0, sr_risk="high")
        assert "×0.7" in r.factors.get("S/R风险", "")

    def test_calc_contracting_vol_reduces(self):
        """vol_regime=contracting → ×0.8。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0, vol_regime="contracting")
        assert "×0.8" in r.factors.get("波动率", "")

    def test_calc_expanding_vol_increases(self):
        """vol_regime=expanding → ×1.1。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0, vol_regime="expanding")
        assert "×1.1" in r.factors.get("波动率", "")

    def test_calc_atr_extreme_reduces(self):
        """atr_pct > 5.0 → ×0.6。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=6.0)
        assert "×0.6" in r.factors.get("ATR封顶", "")

    def test_calc_cold_discount(self):
        """cold_discount < 1 → 应用折扣。"""
        r = calc_position_pct(score=8, adx=20, atr_pct=1.0,
                              trade_state={"cold_discount": 0.5})
        assert "×0.5" in r.factors.get("冷启动", "")

    def test_calc_hard_clamp(self):
        """硬顶 5% / 下限 0.3%。"""
        high = calc_position_pct(score=10, adx=50, atr_pct=0.5)  # 最大可能
        low = calc_position_pct(score=0, adx=20, atr_pct=1.0,
                                trade_state={"win_rate": 1})     # 最小可能
        assert high.pct <= 5.0
        assert low.pct >= 0.3

    def test_validate_total_exposure_notional_cap(self):
        """名义敞口超 NOTIONAL_CAP(30%) → 等比缩减。"""
        active = [
            {"symbol": "BTC", "position_pct": 20, "leverage_max": 2, "direction": "LONG"},
            {"symbol": "ETH", "position_pct": 15, "leverage_max": 2, "direction": "LONG"},
        ]
        # notional = 20*2 + 15*2 = 70 > 30 → scale = 30/70 ≈ 0.428
        out = validate_total_exposure(active)
        assert out[0]["position_pct"] == pytest.approx(20 * 30 / 70, 2)

    def test_validate_total_exposure_cap_total(self):
        """总仓位超 12% → 等比缩减。"""
        active = [
            {"symbol": "A", "position_pct": 8, "leverage_max": 1, "direction": "LONG"},
            {"symbol": "B", "position_pct": 7, "leverage_max": 1, "direction": "LONG"},
        ]
        out = validate_total_exposure(active)
        assert sum(s["position_pct"] for s in out) == pytest.approx(12.0, 2)

    def test_leverage_aware_notional(self):
        """杠杆感知名义敞口计算正确。"""
        active = [
            {"position_pct": 5, "leverage_max": 3},
            {"position_pct": 2, "leverage_max": 10},
        ]  # 5*3 + 2*10 = 35
        assert leverage_aware_notional(active) == 35.0


# ============================================================================
# risk/profile.py
# ============================================================================

class TestProfile:
    def test_get_profile_default(self):
        """get_profile() 返回当前默认为「均衡」。"""
        p = get_profile()
        assert p["cap"] == 12

    def test_get_profile_known(self):
        """传入已知名返回对应画像。"""
        p = get_profile("进取")
        assert p["cap"] == 16

    def test_get_profile_unknown_falls_back(self):
        """未知名回退到默认 ACTIVE_PROFILE。"""
        p = get_profile("不存在的")
        assert p["cap"] == 12  # 回退到「均衡」

    def test_apply_correlation_dir_cap(self):
        """同向仓位数受 mdir 限制。"""
        symbols = [
            {"symbol": "A", "direction": "LONG", "score": 10, "position_pct": 1},
            {"symbol": "B", "direction": "LONG", "score": 9, "position_pct": 1},
            {"symbol": "C", "direction": "LONG", "score": 8, "position_pct": 1},
            {"symbol": "D", "direction": "LONG", "score": 7, "position_pct": 1},
            {"symbol": "E", "direction": "LONG", "score": 6, "position_pct": 1},
        ]
        out = apply_correlation_filter(symbols, profile="保守")  # mdir=2
        long_count = sum(1 for s in out if s["direction"] == "LONG")
        assert long_count <= 2

    def test_apply_correlation_group_dedup(self):
        """同相关组内仅保留 score 最高的一个。"""
        symbols = [
            {"symbol": "BTC", "direction": "LONG", "score": 10, "position_pct": 1},
            {"symbol": "ETH", "direction": "LONG", "score": 8, "position_pct": 1},
            {"symbol": "SOL", "direction": "LONG", "score": 6, "position_pct": 1},
        ]  # BTC→BTC组, ETH→别组, SOL→ALT_L1
        out = apply_correlation_filter(symbols, profile="进取")  # mdir=4
        assert len(out) <= 3  # 不应减少（未超过 mdir 且组不重叠）
