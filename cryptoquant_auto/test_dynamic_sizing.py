from __future__ import annotations

from cryptoquant_auto.testnet_runner import (
    _target_notional, _ks_mult,
    SINGLE_CAP_PCT, MIN_NOTIONAL_USDT,
)


class _KS:
    """假 KillSwitch，仅暴露 .value（与真实 KillLevel 同接口）。"""
    def __init__(self, value):
        self.value = value


EQUITY = 5000.0


def test_ks_mult_full_mapping():
    # 与 _calc_leverage 同源的缩放系数
    assert _ks_mult(_KS(0)) == 1.0
    assert abs(_ks_mult(_KS(0.5)) - 0.8) < 1e-9
    assert _ks_mult(_KS(1.0)) == 0.5
    assert _ks_mult(_KS(2.0)) == 0.3
    assert _ks_mult(_KS(3.0)) == 0.1
    # 未知级别 -> 默认 1.0（不放大）
    assert _ks_mult(_KS(99)) == 1.0


def test_target_notional_single_cap_clamp():
    # 想开 50% 敞口 = 2500U，但单币上限 12% x 5000 = 600U
    dd = {"proposed_exposure": 0.5}
    n = _target_notional("BTC", dd, _KS(0), EQUITY)
    assert n == SINGLE_CAP_PCT * EQUITY


def test_target_notional_ks_shrink():
    # 10% 敞口 @ L0 -> 500U；@ L3(0.1) -> 50U
    dd = {"proposed_exposure": 0.10}
    n0 = _target_notional("BTC", dd, _KS(0), EQUITY)
    n3 = _target_notional("BTC", dd, _KS(3.0), EQUITY)
    assert abs(n0 - 500.0) < 1e-6
    assert abs(n3 - 50.0) < 1e-6


def test_target_notional_basic_scaling():
    # 8% 敞口 @ L0 -> 0.08 x 5000 = 400U（低于单币上限，直接命中）
    dd = {"proposed_exposure": 0.08}
    assert abs(_target_notional("BTC", dd, _KS(0), EQUITY) - 400.0) < 1e-6


def test_target_notional_no_exposure_zero():
    # 无 CVaR 建议 -> 返回 0，调用方据此回退旧路径
    assert _target_notional("BTC", {}, _KS(0), EQUITY) == 0.0
    assert _target_notional("BTC", {"proposed_exposure": 0.0}, _KS(0), EQUITY) == 0.0


def test_target_notional_equity_guard():
    assert _target_notional("BTC", {"proposed_exposure": 0.1}, _KS(0), 0.0) == 0.0


def test_target_notional_min_notional_unreachable_is_nonneg():
    # 极小敞口：比例极低 -> raw 很小，但仍 >= 0（由调用方 MIN_NOTIONAL 护栏判定跳过）
    dd = {"proposed_exposure": 0.0001}
    n = _target_notional("BTC", dd, _KS(0), EQUITY)
    assert n >= 0.0
    assert n < MIN_NOTIONAL_USDT
