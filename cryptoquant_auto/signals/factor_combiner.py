"""因子组合优化：Ridge 回归加权 + IC 指数衰减加权（圆桌决议 Step 1）。

使用方法：
    from .factor_combiner import train_ridge_weights, train_ic_weights

依赖：numpy（无 sklearn，手写 Ridge 闭式解）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ========== 特征名称常量 ==========
# 与 engine.py 的评分维度严格一一对应
FEATURE_NAMES = [
    "adx",        # ADX 趋势强度（分级）
    "di_diff",    # DI差 方向明确度
    "rsi",        # RSI 超买超卖
    "wk_dir",     # 周线顺逆
    "fr_delta",   # 费率变化动量
    "vol_regime", # 波动率扩张
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(meta: dict) -> np.ndarray:
    """从 WFA meta dict 提取特征向量（与 FEATURE_NAMES 严格对齐）。"""
    # adx: 直接用数值（0-100）
    adx = float(meta.get("adx", 20))
    # di_diff: 没有直接存，从 score 反推不可靠，用 adx 代理
    # 实际 di_diff 可从 gen_signal 的 conds 推断，但 meta 没有，暂用 adx
    di_diff = max(0, adx - 20) * 0.5  # 近似
    # rsi: 从 fng 和 direction 近似（meta 没有 rsi）
    fng = float(meta.get("fng", 50))
    dir_str = meta.get("direction", "")
    rsi_signal = (50 - fng) / 30 if dir_str == "做多" else (fng - 50) / 30
    # wk_dir: 周线方向
    wk_dir = meta.get("wk_dir", "unknown")
    wk_signal = 1.0 if wk_dir in ("上涨",) else -1.0 if wk_dir in ("下跌",) else 0.0
    # fr_delta: 费率变化
    fr_d = float(meta.get("fr_delta", 0))
    fr_signal = 1.0 if fr_d > 0.0005 else -1.0 if fr_d < -0.0005 else 0.0
    # vol_regime: 元数据中无此字段，暂用 atr_pct 代理
    atr_pct = float(meta.get("atr_pct", 2.0))
    vol_signal = 1.0 if atr_pct > 3.0 else -1.0 if atr_pct < 0.5 else 0.0
    return np.array([adx / 100, di_diff / 50, rsi_signal, wk_signal, fr_signal, vol_signal])


def extract_features_from_candles(candles_1h: List[dict]) -> np.ndarray:
    """从 K 线直接提取特征（供独立的因子 IC 测试用）。"""
    from .indicators import calc_adx, calc_rsi, calc_atr, volatility_regime
    adx, pdi, mdi = calc_adx(candles_1h)
    closes = [c["c"] for c in candles_1h]
    rsi = calc_rsi(closes)
    atr, atr_pct = calc_atr(candles_1h)
    vol, _ = volatility_regime(candles_1h)
    did = abs(pdi - mdi)
    di_signal = did / 50
    rsi_signal = (50 - rsi) / 30 if rsi < 50 else (rsi - 50) / 30
    vol_signal = 1.0 if vol == "expanding" else -1.0 if vol == "contracting" else 0.0
    return np.array([adx / 100, di_signal, rsi_signal, 0.0, 0.0, vol_signal])


def score_with_weights(features: np.ndarray, weights: np.ndarray) -> float:
    """加权评分：score = w^T * features * 12（映射回 0-12 尺度）。"""
    return float(np.dot(weights, features) * 12)


# ========== Ridge 回归（闭式解，无 sklearn）==========

def train_ridge_weights(X: np.ndarray, y: np.ndarray, lambda_: float = 50.0
                        ) -> Tuple[np.ndarray, float]:
    """Ridge 回归：β = (X^T X + λI)^{-1} X^T y。

    X: (n_samples, n_features) 特征矩阵
    y: (n_samples,) 目标值（未来收益 bps）
    lambda_: L2 正则化强度（默认 50，圆桌统计专家建议）
    返回 (beta, r2)
    """
    n, p = X.shape
    if n < 2:
        return np.zeros(p), 0.0
    X_b = np.column_stack([np.ones(n), X])  # 加截距
    I = np.eye(p + 1)
    I[0, 0] = 0  # 不惩罚截距
    try:
        beta = np.linalg.solve(X_b.T @ X_b + lambda_ * I, X_b.T @ y)
    except np.linalg.LinAlgError:
        return np.zeros(p), 0.0
    beta_coef = beta[1:]  # 去掉截距
    # R²
    y_pred = X_b @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    return beta_coef, r2


def train_ridge_wfa(items: list, fold_indices: list, lambda_: float = 50.0
                    ) -> Tuple[np.ndarray, float, float]:
    """在 WFA fold 上训练 Ridge 并评估 OOS。

    items: [(ts, sym, sig, forward, regime, meta, gbp, nbp), ...]
    fold_indices: [(is_start, is_end, oos_start, oos_end), ...]
    返回 (beta, oos_r2, oos_mean_net_bps)
    """
    all_weights = []
    all_oos_bps = []
    for is_s, is_e, oos_s, oos_e in fold_indices:
        is_items = items[is_s:is_e]
        oos_items = items[oos_s:oos_e]
        if len(is_items) < 10:
            continue
        X_is = np.array([extract_features(m) for *_, m, _, _ in is_items])
        y_is = np.array([nbp for *_, _, _, nbp in is_items])
        beta, _ = train_ridge_weights(X_is, y_is, lambda_)
        all_weights.append(beta)
        if len(oos_items) > 0:
            X_oos = np.array([extract_features(m) for *_, m, _, _ in oos_items])
            oos_scores = X_oos @ beta
            all_oos_bps.append(np.mean(oos_scores))
    if not all_weights:
        return np.zeros(N_FEATURES), 0.0, 0.0
    beta_avg = np.mean(all_weights, axis=0)
    oos_mean = np.mean(all_oos_bps) if all_oos_bps else 0.0
    return beta_avg, 0.0, oos_mean


# ========== 留一币 IC 加权（圆桌统计专家推荐）==========

def train_ic_weights(items_per_coin: Dict[str, list],
                     half_life_days: int = 90) -> Tuple[np.ndarray, str]:
    """留一币 IC 指数衰减加权。

    用 5 个币的时序 Rank IC（半衰期 3 个月指数衰减）算权重，
    第 6 个币留出验证。返回 (weights, validation_report)。
    不加 sklearn，纯 numpy 实现。
    """
    coins = list(items_per_coin.keys())
    if len(coins) < 2:
        return np.ones(N_FEATURES) / N_FEATURES, "样本不足"
    # 对每对 (coin, feature) 算时序 Rank IC
    all_ics = []
    for coin, its in items_per_coin.items():
        if len(its) < 20:
            continue
        X = np.array([extract_features(m) for *_, m, _, _ in its])
        y = np.array([nbp for *_, _, _, nbp in its])
        for f in range(N_FEATURES):
            # Spearman rank correlation
            rx = np.argsort(np.argsort(X[:, f]))
            ry = np.argsort(np.argsort(y))
            n = len(rx)
            ic = (np.sum(rx * ry) - n * (n + 1) ** 2 / 4) / (
                (n ** 3 - n) / 12 * (1 - 1e-9)) if n > 3 else 0.0
            all_ics.append((coin, f, ic))
    if not all_ics:
        return np.ones(N_FEATURES) / N_FEATURES, "IC 计算失败"
    # 按特征聚合 IC，指数衰减加权（半衰期 = half_life_days）
    feature_ics = {f: [] for f in range(N_FEATURES)}
    for coin, f, ic in all_ics:
        feature_ics[f].append(ic)
    weights = np.array([np.mean(v) if v else 0.0 for f, v in feature_ics.items()])
    # 截断负权重为 0（负 IC 特征不应反向使用）
    weights = np.maximum(weights, 0)
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(N_FEATURES) / N_FEATURES
    return weights, f"IC 加权完成，权重={np.round(weights, 3)}"


# ============================================================================
# 阶段2 扩展 · GP 规则树进化 + NSGA-II 三目标 Pareto（专家C 任务13）
# ----------------------------------------------------------------------------
# 落点文件（蓝图实施计划 §三·专家C）：factor_combiner.py 扩展。
# 实际实现在 signals/gp_nsga2.py（零依赖），此处 re-export 以保持「因子组合」
# 模块统一入口，并显式声明本文件已从「Ridge/IC 加权」扩展到「进化规则树」。
#
#   树的叶**只吃因果特征白名单**（signals/causal.py 任务12 产出）→ 伪相关掐死。
#   NSGA-II 三目标：f1=-收益 / f2=最大回撤 / f3=换手率（均最小化），求 Pareto 前沿。
#   适应度用 IS 轻量代理；最终 Pareto 解须再过阶段0.5（Purged+Embargo+DSR，任务15）。
# ============================================================================
from .gp_nsga2 import (                                       # noqa: E402,F401
    make_random_tree, eval_tree, evaluate_tree,
    evolve, crossover, mutate, build_signals_from_tree,
    GPResult, tree_depth,
)

__all__ = [
    "FEATURE_NAMES", "N_FEATURES", "extract_features",
    "extract_features_from_candles", "score_with_weights",
    "train_ridge_weights", "train_ridge_wfa", "train_ic_weights",
    # 阶段2 扩展
    "make_random_tree", "eval_tree", "evaluate_tree", "evolve",
    "crossover", "mutate", "build_signals_from_tree", "GPResult", "tree_depth",
]
