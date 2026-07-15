"""P2-5 · 年化口径统一政策回归锁。

审计结论（P2-5 前缀修复）：
  - 真实 1h 收益的「**报告口径**」统一 8760（sim/metrics.py、sim/evolution.py
    的 _sharpe/deflated_sharpe/probabilistic_sharpe 默认 8760）——这是对的。
  - A/B **显著性闸门**（controlled_ab + run_validation_0_5/2/3/4）刻意用
    单 bar SR（periods_per_year=1）。年化 ×√8760≈93.6 只单调放大 SR/DSR、
    不改变 Welch p，却会把边缘 per-bar edge 放大成假显著，破坏 DSR 显著性检验。
    → 闸门**不能**改成 8760，roundtable「A/B 也用 8760」是误判。

本测试锁死闸门默认单 bar 口径，并证明 periods_per_year 是该年化的唯一开关，
防止未来「好心」改默认 8760 引入假显著。
"""
import inspect
import math

import numpy as np
import pytest

from cryptoquant_auto.sim.ab_harness import controlled_ab


def test_controlled_ab_defaults_to_single_bar_sr():
    """A/B 闸门必须默认单 bar SR（periods_per_year=1）。

    【P2-5 锁】若有人把默认改成 8760，年化 ×93.6 会把噪声 edge 放大成假显著，
    令 recommend_llm 误放行。默认值即政策，锁死。
    """
    sig = inspect.signature(controlled_ab)
    assert sig.parameters["periods_per_year"].default == 1


def test_periods_per_year_is_the_unified_switch():
    """periods_per_year 必须真正生效：8760 下 sr 精确放大 √8760 倍。

    证明该参数是闸门年化的唯一开关；它同时驱动 DSR/PSR 与 detail.sr_*，
    改默认会立即改变所有 A/B 结论数值——故必须显式、刻意。
    """
    r = np.array([0.01, -0.01, 0.02, 0.0, -0.005])
    res_1 = controlled_ab(r, np.zeros_like(r), n_trials=1, periods_per_year=1)
    res_8760 = controlled_ab(r, np.zeros_like(r), n_trials=1, periods_per_year=8760)
    expected = float(r.mean() / (r.std(ddof=1) + 1e-12))
    assert res_1.detail["sr_rule"] == pytest.approx(expected)
    assert res_8760.detail["sr_rule"] == pytest.approx(expected * math.sqrt(8760))
