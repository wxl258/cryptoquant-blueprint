"""演化优化（DEAP NSGA-II）集成测试（蓝图路线图 第 4 周 D）。

验证：
  1) NSGA-II 在已知双目标帕累托问题上产出非支配前沿（真实 DEAP 路径）。
  2) DEAP 缺失时降级为随机采样+帕累托排序，仍产出非支配前沿且解在界内。
  3) CVaR 默认评估器单次评估返回 3 个有限目标。
  4) CVaR 演化端到端（小种群）产出界内解 + 目标维度正确。
  5) 返回的前沿确实是「非支配」的（帕累托性质成立）。
"""
import numpy as np
import pytest

from cryptoquant_auto.sim.evolution import (
    EvolutionEngine, Parameter, CVAR_PARAMS, make_cvar_evaluator, HAS_DEAP,
)


def _synthetic_evaluator():
    """minimize (x^2, (x-2)^2)，x∈[0,2]：真帕累托前沿为整个 [0,2]。"""
    def evaluate(individual):
        x = float(individual[0])
        return (x * x, (x - 2.0) ** 2)
    return evaluate


def _pairwise_nondominated(front):
    """front: [(param_dict, objectives), ...]；返回是否两两互不支配。"""
    objs = np.asarray([o for _, o in front], float)
    M = len(objs)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            diff = objs[j] - objs[i]
            if np.all(diff <= 0) and np.any(diff < 0):  # j 支配 i
                return False
    return True


def test_nsga2_finds_pareto_front():
    eng = EvolutionEngine([Parameter("x", 0.0, 2.0)], pop_size=30, n_gen=20, seed=1)
    front = eng.run(_synthetic_evaluator())
    assert front, "前沿不应为空"
    xs = [p["x"] for p, _ in front]
    assert all(0.0 <= x <= 2.0 for x in xs), "解应在参数界内"
    assert len(front) >= 2, "应找到多个非支配解（前沿有展布）"
    assert _pairwise_nondominated(front), "前沿应两两非支配"
    # 目标维度正确
    assert len(front[0][1]) == 2


def test_degrade_without_deap(monkeypatch):
    import cryptoquant_auto.sim.evolution as mod
    monkeypatch.setattr(mod, "HAS_DEAP", False)
    eng = EvolutionEngine([Parameter("x", 0.0, 2.0)], seed=2)
    front = eng.run(_synthetic_evaluator())
    assert front, "降级路径前沿不应为空"
    xs = [p["x"] for p, _ in front]
    assert all(0.0 <= x <= 2.0 for x in xs)
    assert _pairwise_nondominated(front)


def test_cvar_evaluator_one_eval():
    evaluate = make_cvar_evaluator()
    obj = evaluate([0.02, 0.12, 0.05, 0.05])   # budget, cap, max_pos, alpha
    assert len(obj) == 3
    assert all(np.isfinite(o) for o in obj), "目标应有限（无 NaN/inf）"


def test_cvar_evolution_end_to_end():
    # 小种群快速验证真实 CVaR 演化端到端可跑、产出界内解
    eng = EvolutionEngine(CVAR_PARAMS, pop_size=8, n_gen=4, seed=3)
    evaluate = make_cvar_evaluator()
    front = eng.run(evaluate)
    assert front, "CVaR 演化前沿不应为空"
    for p, o in front:
        assert CVAR_PARAMS[0].low <= p["cvar_budget"] <= CVAR_PARAMS[0].high
        assert CVAR_PARAMS[1].low <= p["total_cap"] <= CVAR_PARAMS[1].high
        assert CVAR_PARAMS[2].low <= p["max_pos"] <= CVAR_PARAMS[2].high
        assert CVAR_PARAMS[3].low <= p["alpha"] <= CVAR_PARAMS[3].high
        assert len(o) == 3 and all(np.isfinite(v) for v in o)
    assert _pairwise_nondominated(front), "CVaR 前沿应两两非支配"


def test_front_non_dominated_property():
    # 显式构造一个含支配关系的集合，确认引擎的帕累托工具能识别
    objs = np.array([
        [1.0, 1.0],   # 被 (0,0) 支配
        [0.0, 0.0],   # 非支配
        [0.5, 2.0],   # 非支配（与 (0,0) 互不强）
        [2.0, 0.5],   # 非支配
    ])
    mask = EvolutionEngine._non_dominated_mask(objs)
    # 仅 [0,0] 非支配；其余三个都被 [0,0] 支配
    assert list(mask) == [False, True, False, False]
