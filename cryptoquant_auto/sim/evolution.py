"""演化优化引擎（蓝图路线图 第 4 周 D · 任务：DEAP NSGA-II）。

用 NSGA-II 多目标遗传算法演化「策略/仓位参数」，在风险-收益权衡的
帕累托前沿上寻优。与 B.CVaR 形成闭环：默认演化 CVaR 仓位优化器超参
（cvar_budget / total_cap / max_pos / alpha），用真实 1h 历史收益 +
真实 CvarPositionOptimizer + 真实指标（夏普/回撤/CVaR）打分。

零依赖纪律：DEAP 缺失 → 退化为「随机采样 + 帕累托非支配排序」基线，
仍可产出可比的近似前沿（绝不抛错、行为有界）。演化是**离线调参**工具，
不进入实时决策路径；输出推荐参数集供人工/后续 cron 部署（env 可覆盖）。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except Exception:  # pragma: no cover - 无 DEAP 时走随机基线
    HAS_DEAP = False


# ---------------------------------------------------------------------------
# 参数空间
# ---------------------------------------------------------------------------
@dataclass
class Parameter:
    name: str
    low: float
    high: float
    integer: bool = False

    def clip(self, v: float) -> float:
        v = float(np.clip(v, self.low, self.high))
        return int(round(v)) if self.integer else v


# 默认演化空间：B.CVaR 仓位优化器超参（与 paper_runner 的 env 键对齐）
CVAR_PARAMS: List[Parameter] = [
    Parameter("cvar_budget", -0.05, -0.005),   # 尾部最差均值损失预算（越负越严）
    Parameter("total_cap", 0.06, 0.20),        # 总仓位硬上限
    Parameter("max_pos", 0.02, 0.08),          # 单币仓位上限
    Parameter("alpha", 0.01, 0.10),            # CVaR 尾部分位
]


# ---------------------------------------------------------------------------
# 本地指标（避免触及 metrics 私有函数；年化夏普与回撤，与 sim.metrics 同口径）
# ---------------------------------------------------------------------------
def _sharpe(equity: np.ndarray, periods_per_year: int = 8760) -> float:
    eq = np.asarray(equity, float)
    if len(eq) < 3:
        return 0.0
    rets = eq[1:] / eq[:-1] - 1.0
    mean = rets.mean()
    std = rets.std(ddof=1)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(periods_per_year))


def _max_dd(equity: np.ndarray) -> float:
    eq = np.asarray(equity, float)
    if len(eq) == 0:
        return 0.0
    peak = eq[0]
    mdd = 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1.0)
    return float(mdd)


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class EvolutionEngine:
    """通用 NSGA-II 演化引擎（DEAP 可选，缺失降级为随机帕累托基线）。

    evaluate(individual: List[float]) -> Sequence[float]，目标一律「越小越好」
    （最大化目标调用方自行取负）。返回帕累托前沿 [(param_dict, objectives), ...]。
    """

    def __init__(self, params: List[Parameter], pop_size: int = 40, n_gen: int = 20,
                 mu: float = 0.0, sigma: float = 0.15, indpb: float = 0.25,
                 tournsize: int = 3, seed: int = 42):
        self.params = params
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.tournsize = tournsize
        self.seed = seed
        self.rng = random.Random(seed)

    # ---- 个体 ↔ 参数 ----
    def _new_ind(self) -> List[float]:
        ind = []
        for p in self.params:
            if p.integer:
                ind.append(self.rng.randint(int(p.low), int(p.high)))
            else:
                ind.append(self.rng.uniform(p.low, p.high))
        return list(ind)

    def _to_dict(self, individual: Sequence[float]) -> dict:
        return {p.name: p.clip(individual[i]) for i, p in enumerate(self.params)}

    # ---- 帕累托非支配排序（纯 numpy；DEAP 缺失时作基线 + 输出整理）----
    @staticmethod
    def _non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
        arr = np.asarray(objectives, float)
        M = arr.shape[0]
        dominated = np.zeros(M, dtype=bool)
        for i in range(M):
            diff = arr - arr[i]
            le = np.all(diff <= 0, axis=1)
            lt = np.any(diff < 0, axis=1)
            dom = le & lt
            dom[i] = False
            if dom.any():
                dominated[i] = True
        return ~dominated

    def run(self, evaluate: Callable[[List[float]], Sequence[float]],
            verbose: bool = False) -> List[Tuple[dict, tuple]]:
        if HAS_DEAP:
            return self._run_deap(evaluate, verbose)
        return self._run_random(evaluate)

    # ---- DEAP / NSGA-II 路径 ----
    def _run_deap(self, evaluate, verbose) -> List[Tuple[dict, tuple]]:
        probe = list(evaluate(self._new_ind()))
        n_obj = len(probe)
        fit_name = f"FitnessMin_{n_obj}"
        ind_name = f"Individual_{n_obj}"
        if not hasattr(creator, fit_name):
            creator.create(fit_name, base.Fitness,
                           weights=tuple(-1.0 for _ in range(n_obj)))
        if not hasattr(creator, ind_name):
            creator.create(ind_name, list, fitness=getattr(creator, fit_name))

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate,
                         getattr(creator, ind_name), self._new_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: tuple(evaluate(list(ind))))
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian,
                         mu=self.mu, sigma=self.sigma, indpb=self.indpb)
        toolbox.register("select", tools.selNSGA2)

        pop = toolbox.population(n=self.pop_size)
        pop, _ = algorithms.eaMuPlusLambda(
            pop, toolbox, mu=self.pop_size, lambda_=self.pop_size,
            cxpb=0.7, mutpb=0.25, ngen=self.n_gen, verbose=verbose)
        fronts = tools.sortNondominated(pop, len(pop))
        pf = fronts[0]
        out = [(self._to_dict(list(ind)), tuple(ind.fitness.values)) for ind in pf]
        out.sort(key=lambda x: x[1][0])
        return out

    # ---- 无 DEAP 降级：随机采样 + 帕累托非支配 ----
    def _run_random(self, evaluate, n_samples: int = 200) -> List[Tuple[dict, tuple]]:
        inds = [self._new_ind() for _ in range(n_samples)]
        objs = [tuple(evaluate(inds[i])) for i in range(n_samples)]
        mask = self._non_dominated_mask(np.asarray(objs, float))
        out = [(self._to_dict(inds[i]), objs[i]) for i in range(len(inds)) if mask[i]]
        out.sort(key=lambda x: x[1][0])
        return out


# ---------------------------------------------------------------------------
# 默认评估器：演化 B.CVaR 超参（真实历史 + 真实优化器 + 真实指标）
# ---------------------------------------------------------------------------
def make_cvar_evaluator(symbols: Sequence[str] = ("BTC", "ETH", "SOL", "BNB", "XRP", "TRX"),
                        n_steps: int = 120, seed: int = 42) -> Callable[[List[float]], tuple]:
    """构造「CVaR 仓位超参 → 多目标(夏普↑, 回撤↓, CVaR↓)」评估器。

    对每一步合成一个单币方向信号，用候选 CVaR 超参经真实 CvarPositionOptimizer
    定仓，按真实 1h 收益累计权益，最后用真实指标算 (maximize 夏普, minimize 回撤,
    minimize CVaR) → 以「minimize」形式返回 (-sharpe, -mdd, cvar)。
    """
    from ..stage2_features import _load_history
    from ..risk.cvar_optimizer import CvarPositionOptimizer, cvar

    hist = _load_history()
    rets = {}
    for s in symbols:
        k1h = hist.get(s, {}).get("1h", [])
        if len(k1h) < n_steps + 1:
            continue
        closes = np.array([c["c"] for c in k1h], dtype=float)
        rets[s] = np.diff(np.log(closes))[-n_steps:]
    if not rets:
        raise RuntimeError("无足够历史收益构造 CVaR 评估器")
    minlen = min(len(v) for v in rets.values())

    rng = np.random.default_rng(seed)
    sig_sym = rng.integers(0, len(symbols), size=minlen)
    sig_dir = rng.choice([-1, 1], size=minlen)
    sig_conv = np.abs(rng.normal(0.5, 0.15, size=minlen))

    def evaluate(individual: List[float]) -> tuple:
        budget = CVAR_PARAMS[0].clip(individual[0])
        cap = CVAR_PARAMS[1].clip(individual[1])
        max_pos = CVAR_PARAMS[2].clip(individual[2])
        alpha = CVAR_PARAMS[3].clip(individual[3])
        opt = CvarPositionOptimizer(alpha=alpha, cvar_budget=budget,
                                    total_cap=cap, max_pos=max_pos, min_samples=24)
        eq = [1.0]
        for t in range(minlen):
            s = symbols[int(sig_sym[t])]
            di = int(sig_dir[t])
            conv = float(sig_conv[t])
            # 该币全历史方向化收益 (T,1) → 优化器据此估协方差与尾部 CVaR
            R_full = (di * rets[s]).reshape(-1, 1)
            w = opt.solve(np.array([conv]), R_full, [s])
            wv = w.get(s, 0.0)
            rp = wv * di * rets[s][t]   # 真实单步组合收益（用该步实际收益结算）
            eq.append(eq[-1] * (1.0 + rp))
        eqa = np.asarray(eq, float)
        sharpe = _sharpe(eqa)
        mdd = _max_dd(eqa)
        cv = cvar(np.diff(eqa), alpha)
        return (-sharpe, -mdd, cv)

    return evaluate


# ---------------------------------------------------------------------------
# CLI（离线调参；不进实时路径）
# ---------------------------------------------------------------------------
def run_cvar_evolution(pop_size: int = 30, n_gen: int = 15, seed: int = 42,
                       save: Optional[str] = None, verbose: bool = False):
    eng = EvolutionEngine(CVAR_PARAMS, pop_size=pop_size, n_gen=n_gen, seed=seed)
    evaluate = make_cvar_evaluator()
    front = eng.run(evaluate, verbose=verbose)
    print(f"# CVaR 演化帕累托前沿（{len(front)} 解，DEAP={HAS_DEAP}）")
    print("| # | cvar_budget | total_cap | max_pos | alpha | -Sharpe | -MaxDD | CVaR |")
    print("|---|-------------|-----------|----------|-------|---------|--------|------|")
    for i, (params, obj) in enumerate(front):
        print(f"| {i} | {params['cvar_budget']:.4f} | {params['total_cap']:.3f} | "
              f"{params['max_pos']:.3f} | {params['alpha']:.3f} | "
              f"{obj[0]:.3f} | {obj[1]:.4f} | {obj[2]:.4f} |")
    if save:
        with open(save, "w", encoding="utf-8") as f:
            json.dump([{"params": p, "objectives": list(o)} for p, o in front],
                      f, ensure_ascii=False, indent=2)
        print(f"\n已保存前沿 → {save}")
    return front


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CVaR 超参 NSGA-II 演化（离线调参）")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gen", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="保存前沿 JSON 路径")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run_cvar_evolution(pop_size=args.pop, n_gen=args.gen, seed=args.seed,
                       save=args.save, verbose=args.verbose)
