"""任务13 · GP 规则树进化 + NSGA-II 三目标 Pareto（零依赖纯 numpy）。

落点：在 factor_combiner.py 基础上扩展（详见 factor_combiner.py 末尾「阶段2 扩展」段，
本文件为实际实现并 re-export）。

设计：
  - GP 染色体 = 规则树（组合白名单特征 → 标量信号分）。节点：const/feat 叶 +
    add/sub/mul/neg/tanh/relu 算子。树的叶**只吃因果特征白名单**（任务12 产出），
    伪相关从入口掐死。
  - NSGA-II 三目标（蓝图 MOO3）：f1=-收益（最大化收益）/ f2=最大回撤 / f3=换手率，
    全部取「最小化」方向，求 Pareto 前沿。
  - 适应度用轻量代理（IS 段逐 bar 方向×前向收益）：GP 搜索要快；最终 Pareto 解
    必须在任务15 再过一遍阶段0.5（Purged+Embargo + DSR）才算进门。

零依赖纪律：仅 numpy + 包内模型。torch/LLM 不在此引入（后移至阶段3-4）。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import Signal, Direction
from ..stage2_features import FEATURE_NAMES


# ============================ 规则树 ============================
def make_random_tree(rng: np.random.Generator, max_depth: int,
                     n_feat: int, leaf_p: float = 0.4) -> Any:
    """随机生成规则树。leaf_p=长成叶子的概率。"""
    if max_depth <= 0 or rng.random() < leaf_p:
        if rng.random() < 0.25:
            return ("const", float(rng.uniform(-1.0, 1.0)))
        return ("feat", int(rng.integers(0, n_feat)))
    op = rng.choice(["add", "sub", "mul", "neg", "tanh", "relu"])
    if op in ("neg", "tanh", "relu"):
        return (op, make_random_tree(rng, max_depth - 1, n_feat, leaf_p))
    return (op, make_random_tree(rng, max_depth - 1, n_feat, leaf_p),
            make_random_tree(rng, max_depth - 1, n_feat, leaf_p))


def _safe(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(-1e6, min(1e6, x)))


def eval_tree(tree: Any, row: np.ndarray) -> float:
    """递归求值规则树，row 为白名单特征向量。"""
    if tree[0] == "const":
        return _safe(tree[1])
    if tree[0] == "feat":
        return _safe(row[tree[1]])
    if tree[0] == "neg":
        return _safe(-eval_tree(tree[1], row))
    if tree[0] == "tanh":
        return _safe(np.tanh(eval_tree(tree[1], row)))
    if tree[0] == "relu":
        return _safe(max(0.0, eval_tree(tree[1], row)))
    a = eval_tree(tree[1], row)
    if tree[0] in ("add", "sub", "mul"):
        b = eval_tree(tree[2], row)
        if tree[0] == "add":
            return _safe(a + b)
        if tree[0] == "sub":
            return _safe(a - b)
        return _safe(a * b)
    return 0.0


def tree_depth(tree: Any) -> int:
    if tree[0] in ("const", "feat"):
        return 0
    if tree[0] in ("neg", "tanh", "relu"):
        return 1 + tree_depth(tree[1])
    return 1 + max(tree_depth(tree[1]), tree_depth(tree[2]))


def _subtrees(tree: Any, path: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], Any]]:
    out = [(path, tree)]
    if tree[0] in ("neg", "tanh", "relu"):
        out += _subtrees(tree[1], path + (1,))
    elif tree[0] in ("add", "sub", "mul"):
        out += _subtrees(tree[1], path + (1,))
        out += _subtrees(tree[2], path + (2,))
    return out


def _replace(tree: Any, path: Tuple[int, ...], new: Any) -> Any:
    if path == ():
        return new
    op = tree[0]
    if op in ("neg", "tanh", "relu"):
        return (op, _replace(tree[1], path[1:], new))
    if path[0] == 1:
        return (op, _replace(tree[1], path[1:], new), tree[2])
    return (op, tree[1], _replace(tree[2], path[1:], new))


def crossover(rng: np.random.Generator, t1: Any, t2: Any) -> Tuple[Any, Any]:
    """交换两棵树各一个随机子树。"""
    s1 = [p for p, _ in _subtrees(t1) if p != ()]
    s2 = [p for p, _ in _subtrees(t2) if p != ()]
    if not s1 or not s2:
        return t1, t2
    p1 = s1[rng.integers(0, len(s1))]
    p2 = s2[rng.integers(0, len(s2))]
    sub1 = _get(t1, p1); sub2 = _get(t2, p2)
    return _replace(t1, p1, sub2), _replace(t2, p2, sub1)


def _get(tree: Any, path: Tuple[int, ...]) -> Any:
    for k in path:
        tree = tree[1] if k == 1 else tree[2]
    return tree


def mutate(rng: np.random.Generator, tree: Any, n_feat: int,
           max_depth: int = 3) -> Any:
    """随机替换一个子树为新的随机树（带深度惩罚，防爆搜索）。"""
    subs = [p for p, _ in _subtrees(tree) if p != ()]
    if not subs:
        return make_random_tree(rng, max_depth, n_feat)
    p = subs[rng.integers(0, len(subs))]
    new = make_random_tree(rng, max(1, max_depth - 1), n_feat)
    return _replace(tree, p, new)


# ============================ 适应度代理 ============================
def evaluate_tree(tree: Any, X: np.ndarray, y: np.ndarray
                  ) -> Tuple[float, float, float, Dict[str, float]]:
    """IS 轻量代理适应度。返回 (f1, f2, f3, meta)。

    f1 = -mean(pnl)       最大化收益
    f2 = max_drawdown     最小化回撤
    f3 = turnover_rate    最小化换手
    pnl_i = sign(score_i) * y_i（方向×真实前向收益，无成本代理）
    """
    n = X.shape[0]
    scores = np.array([eval_tree(tree, X[k]) for k in range(n)])
    dirs = np.sign(scores)
    pnl = dirs * y
    mean_ret = float(pnl.mean())
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    mdd = float((peak - cum).max()) if len(cum) else 0.0
    turn = float(np.mean(np.abs(np.diff(dirs)))) if n > 1 else 0.0
    meta = {"mean_return": mean_ret, "max_dd": mdd, "turnover": turn,
            "depth": tree_depth(tree)}
    return -mean_ret, mdd, turn, meta


# ============================ NSGA-II ============================
def _fast_non_dominated(objectives: np.ndarray) -> List[List[int]]:
    """快速非支配排序（最小化方向）。返回 fronts: list[list[idx]]。"""
    pop = objectives.shape[0]
    S = [[] for _ in range(pop)]
    ndom = [0] * pop
    fronts: List[List[int]] = [[]]
    for p in range(pop):
        for q in range(pop):
            if p == q:
                continue
            # p 支配 q：所有目标 <= 且至少一个 <
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                S[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    k = len(front)
    cd = np.zeros(k)
    if k <= 2:
        cd[:] = np.inf
        return cd
    m = objectives.shape[1]
    for obj in range(m):
        vals = objectives[front, obj]
        order = np.argsort(vals)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        rng = vals.max() - vals.min()
        if rng > 0:
            for rank in range(1, k - 1):
                cd[order[rank]] += (vals[order[rank + 1]] - vals[order[rank - 1]]) / rng
    return cd


@dataclass
class GPResult:
    pareto: List[Any] = field(default_factory=list)        # Pareto 前沿规则树
    pareto_meta: List[Dict[str, float]] = field(default_factory=list)
    best_by_return: Any = None
    best_meta: Dict[str, float] = field(default_factory=dict)
    whitelist: List[str] = field(default_factory=list)
    n_eval: int = 0


def evolve(X_whitelist: np.ndarray, y: np.ndarray, whitelist: List[str],
           pop_size: int = 30, generations: int = 12, max_depth: int = 3,
           seed: int = 0) -> GPResult:
    """GP + NSGA-II 进化（IS 段）。返回 Pareto 前沿。"""
    rng = np.random.default_rng(seed)
    n_feat = X_whitelist.shape[1]
    # 初始化种群
    pop = [make_random_tree(rng, max_depth, n_feat) for _ in range(pop_size)]
    objs = np.array([evaluate_tree(t, X_whitelist, y)[:3] for t in pop])
    metas = [evaluate_tree(t, X_whitelist, y)[3] for t in pop]
    n_eval = pop_size
    for gen in range(generations):
        # 非支配排序 + 拥挤度
        fronts = _fast_non_dominated(objs)
        crowd = np.zeros(len(pop))
        for fr in fronts:
            if fr:
                crowd[fr] = _crowding(objs, fr)
        # 精英选择（rank 优先，crowding 次之）
        order = sorted(range(len(pop)),
                       key=lambda i: (fronts_index(fronts, i), -crowd[i]))
        # 生成子代
        offspring = []
        off_objs = []
        while len(offspring) < pop_size:
            a = _tournament(rng, fronts, crowd, len(pop))
            b = _tournament(rng, fronts, crowd, len(pop))
            c1, c2 = crossover(rng, pop[a], pop[b])
            c1 = mutate(rng, c1, n_feat, max_depth)
            c2 = mutate(rng, c2, n_feat, max_depth)
            for c in (c1, c2):
                f1, f2, f3, m = evaluate_tree(c, X_whitelist, y)
                offspring.append(c); off_objs.append((f1, f2, f3)); metas.append(m)
                n_eval += 1
        # 合并选择下一代
        comb_pop = pop + offspring
        comb_obj = np.vstack([objs, np.array(off_objs)])
        comb_fronts = _fast_non_dominated(comb_obj)
        comb_crowd = np.zeros(len(comb_pop))
        for fr in comb_fronts:
            if fr:
                comb_crowd[fr] = _crowding(comb_obj, fr)
        next_order = sorted(range(len(comb_pop)),
                            key=lambda i: (fronts_index(comb_fronts, i), -comb_crowd[i]))
        pop = [comb_pop[i] for i in next_order[:pop_size]]
        objs = comb_obj[next_order[:pop_size]]
        metas = [metas[i] for i in next_order[:pop_size]]
    # Pareto 前沿（第一层）
    fronts = _fast_non_dominated(objs)
    pareto_idx = fronts[0] if fronts else list(range(len(pop)))
    pareto = [pop[i] for i in pareto_idx]
    pareto_meta = [metas[i] for i in pareto_idx]
    # 按收益挑一个代表解
    if pareto_meta:
        bi = max(range(len(pareto_meta)), key=lambda i: pareto_meta[i]["mean_return"])
    else:
        bi = 0
    return GPResult(pareto=pareto, pareto_meta=pareto_meta,
                    best_by_return=pareto[bi] if pareto else None,
                    best_meta=pareto_meta[bi] if pareto_meta else {},
                    whitelist=whitelist, n_eval=n_eval)


def fronts_index(fronts: List[List[int]], i: int) -> int:
    for rank, fr in enumerate(fronts):
        if i in fr:
            return rank
    return len(fronts)


def _tournament(rng: np.random.Generator, fronts: List[List[int]],
                crowd: np.ndarray, pop: int) -> int:
    a, b = rng.integers(0, pop), rng.integers(0, pop)
    ra, rb = fronts_index(fronts, a), fronts_index(fronts, b)
    if ra != rb:
        return a if ra < rb else b
    return a if crowd[a] >= crowd[b] else b


# ============================ 树 → 可回测 Signal ============================
def build_signals_from_tree(tree: Any, rows: List[dict], X: np.ndarray,
                            whitelist_idx: List[int]) -> List[Signal]:
    """把规则树转成真实前向路径 Signal（诚实回测用）。

    rows: stage2_features 的元数据结构（含 symbol/price/atr/forward）。
    X: 全特征矩阵；whitelist_idx 取对应列喂树。方向=sign(score)，置信=clamp(|score|)。
    """
    out: List[Signal] = []
    for k, r in enumerate(rows):
        feat = X[k, whitelist_idx]
        score = eval_tree(tree, feat)
        if abs(score) < 1e-6:
            continue
        direction = Direction.LONG if score > 0 else Direction.SHORT
        price = r["price"]
        atr = r["atr"] or price * 0.01
        mult = 2.0
        if direction is Direction.LONG:
            sl = price - mult * atr
            tp1 = price + abs(score) * 0 + mult * atr   # 简单 1R/2R 派生
            tp2 = price + 2 * mult * atr
        else:
            sl = price + mult * atr
            tp1 = price - mult * atr
            tp2 = price - 2 * mult * atr
        sig = Signal(
            symbol=r["symbol"], tf="1H", direction=direction,
            entry=round(price, 2), sl=round(sl, 2), tp1=round(tp1, 2),
            tp2=round(tp2, 2), rr=2.0,
            confidence=max(0.1, min(1.0, abs(score))),
            signal_id=f"{r['symbol']}_gp_{uuid.uuid4().hex[:8]}", atr=atr,
        )
        out.append(sig)
    return out
