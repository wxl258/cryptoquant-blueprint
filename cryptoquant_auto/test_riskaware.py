"""P1-13-quant 回归：CVaR 约束夏普（分母有界）、RiskAwareTrader 砍仓。"""
import numpy as np
import pytest

from cryptoquant_auto.sim.riskaware import (
    cvar, cvar_sharpe_score, VanillaTrader, RiskAwareTrader,
)


def test_cvar_is_negative_for_loss_tail():
    rng = np.random.default_rng(3)
    calm = rng.normal(0.001, 0.01, size=300)
    crash = np.concatenate([calm, rng.normal(-0.05, 0.03, size=60)])
    assert cvar(crash) < cvar(calm)


def test_cvar_sharpe_penalizes_breach():
    rng = np.random.default_rng(3)
    crash = np.concatenate([rng.normal(0.001, 0.01, 300),
                            rng.normal(-0.05, 0.03, 60)])
    score_v = cvar_sharpe_score(crash)               # 无预算
    score_a = cvar_sharpe_score(crash, cvar_budget=-0.02)  # 有预算
    assert score_a < score_v  # 越界重罚


def test_cvar_sharpe_denominator_guard_no_explosion():
    # P1-13/quant：cvar_budget 接近 0 时惩罚项不应爆炸（分母有下限）
    rng = np.random.default_rng(4)
    r = rng.normal(0.0, 0.02, 200)
    val = cvar_sharpe_score(r, cvar_budget=0.0)
    assert np.isfinite(val)
    assert abs(val) < 1e6


def test_risk_aware_trader_breaches_on_crash():
    rng = np.random.default_rng(3)
    crash = np.concatenate([rng.normal(0.001, 0.01, 300),
                            rng.normal(-0.05, 0.03, 60)])
    ra = RiskAwareTrader(cvar_budget=-0.02).decide(crash)
    assert ra["breach"] is True
    assert ra["action"] == "HOLD"
    assert ra["exposure"] == 0.0


def test_vanilla_trader_no_breach():
    rng = np.random.default_rng(5)
    calm = rng.normal(0.001, 0.01, 300)
    v = VanillaTrader().decide(calm)
    assert v["breach"] is False
    assert v["exposure"] == 1.0
