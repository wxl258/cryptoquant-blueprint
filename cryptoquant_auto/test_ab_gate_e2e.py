"""【P1-6】A/B 闸门硬拒 + signal→position→PnL 端到端集成测试。

覆盖两条圆桌要求：
  1) ab_harness 代码级硬拒：LLM 未过放行闸门(recommend_llm=False)时，
     assert_gate_passed 必须抛 GateRejected（Fail-closed，杜绝「只 return 不拒」）；
  2) 离线端到端：方向化信号(conviction) → CvarPositionOptimizer 产出仓位权重
     （受 SINGLE_CAP_PCT=0.04 单一真相源约束，呼应 P1-2）→ 价格兑现 → PnL 符号正确。
"""
from __future__ import annotations

import numpy as np

from cryptoquant_auto.sim.ab_harness import (
    controlled_ab,
    assert_gate_passed,
    GateRejected,
    make_synthetic_returns,
)
from cryptoquant_auto.risk.cvar_optimizer import CvarPositionOptimizer
from cryptoquant_auto.risk.gate import SINGLE_CAP_PCT


def test_gate_hard_reject_when_llm_loses():
    # LLM 明显无 edge（sr=0），规则有强 edge（sr=0.6）→ 不应放行
    rule = make_synthetic_returns(800, sr_target=0.6, seed=11)
    llm = make_synthetic_returns(800, sr_target=0.0, seed=12)
    res = controlled_ab(rule, llm, n_trials=4, periods_per_year=1)
    assert res.recommend_llm is False
    try:
        assert_gate_passed(res)
    except GateRejected:
        return  # 预期抛 GateRejected
    raise AssertionError("LLM 未过闸门却未抛 GateRejected（硬拒失效）")


def test_gate_passes_when_llm_wins():
    # LLM 明显更优且显著（sr=0.7 vs 0.05）→ 应放行，不抛异常
    rule = make_synthetic_returns(800, sr_target=0.05, seed=21)
    llm = make_synthetic_returns(800, sr_target=0.7, seed=22)
    res = controlled_ab(rule, llm, n_trials=4, periods_per_year=1)
    assert res.recommend_llm is True
    assert_gate_passed(res)  # 不应抛


def test_signal_to_position_to_pnl_e2e():
    # 强多 conviction → CVaR 权重（受单一真相源 0.04 约束）→ 上涨/下跌 PnL 符号正确
    syms = ["BTC", "ETH"]
    conviction = np.array([0.9, 0.6], float)        # 信号强度（=|μ|）
    # 方向化收益矩阵 R：方向=+1(看多)，全正 → 看多兑现
    R = np.array([
        [0.010, 0.008],
        [0.012, 0.009],
        [0.011, 0.007],
        [0.013, 0.010],
        [0.010, 0.008],
    ], float)
    # 默认 min_samples=24 > T=5 → 走确定性启发式路径（与 scipy 随机性解耦）
    opt = CvarPositionOptimizer(max_pos=SINGLE_CAP_PCT, total_cap=0.12)
    w = opt.solve(conviction, R, syms)

    # position 步骤：权重非空且受 SINGLE_CAP_PCT 单一真相源约束（P1-2）
    assert w, "CVaR 未产出任何仓位"
    assert all(0.0 <= v <= SINGLE_CAP_PCT + 1e-9 for v in w.values()), w

    fwd = np.array([0.02, 0.015], float)            # 下一期真实收益（看多兑现）
    pnl_long = float(sum(w[s] * fwd[i] for i, s in enumerate(syms)))
    assert pnl_long > 0, f"看多+上涨应得正 PnL, got {pnl_long}"

    pnl_short = float(sum(w[s] * (-fwd[i]) for i, s in enumerate(syms)))
    assert pnl_short < 0, f"看多+下跌应得负 PnL, got {pnl_short}"
