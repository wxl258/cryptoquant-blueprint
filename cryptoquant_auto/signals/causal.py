"""任务12 · 因果发现（PCMCI / NOTEARS 类，零依赖纯 numpy）。

目标：从候选特征（stage2_features.FEATURE_NAMES）中发现「真正驱动未来收益」
的因果特征，输出**因果特征白名单**——从入口掐死伪相关，供 GP/NSGA-II（任务13）
只吃因果特征。

方法（对应蓝图 arXiv 2408.09960「因果发现做稳定特征选择 + 不变因果结构」）：
  1. PCMCI 类单目标 PC：对每个候选特征做「y 与该特征 | 其余特征」的条件独立性
     检验（偏相关 + Fisher z），找出 y 的马尔可夫毯（直接父节点）= 直接因果驱动。
  2. 稳定特征选择（Stability Selection）：在 B 次 Bootstrap 重采样上重复 (1)，
     统计每个特征被选为父节点的频率 → 频率 ≥ 阈值 的特征进入稳定集。
  3. 不变因果结构（Invariance）：按 regime（TREND/RANGE/CRASH）分层，要求白名单
     特征的 y~x 斜率在 regime 间符号一致且变异系数 < tol —— 抗 regime 漂移，
     过滤掉「仅在某 regime 偶发显著」的伪因。
  4. NOTEARS 类 DAG（辅助）：线性 SEM + 增广拉格朗日解 DAG，给出因果序与全图，
     作交叉验证与文档；白名单主体由 (1)+(2)+(3) 稳健路径产出。

零依赖纪律：仅 numpy + math。不引 torch/sklearn/scipy。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ..stage2_features import FEATURE_NAMES


@dataclass
class CausalReport:
    whitelist: List[str] = field(default_factory=list)          # 因果特征白名单
    stable_freq: Dict[str, float] = field(default_factory=dict) # 各特征稳定选择频率
    invariant: Dict[str, bool] = field(default_factory=dict)    # 是否通过 regime 不变性
    parents_edges: Dict[str, float] = field(default_factory=dict)  # 父节点偏相关权重
    dag: Optional[np.ndarray] = None                            # NOTEARS 辅助 DAG（p+1×p+1）
    n_samples: int = 0
    method: str = "PCMCI-class PC + stability + invariance"

    @property
    def summary(self) -> str:
        wl = ", ".join(self.whitelist) if self.whitelist else "（空）"
        return (f"样本={self.n_samples} 白名单({len(self.whitelist)})=[{wl}] "
                f"稳定频率={ {k: round(v,2) for k,v in self.stable_freq.items()} }")


def _norm_ppf(p: float) -> float:
    """标准正态分位（Abramowitz-Stegun 近似，零依赖）。"""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
         3.75440866190742]
    p_low = 0.02425; p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _ols_residuals(target: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """target 对 preds（自动加截距）回归的残差。preds 可为空。"""
    n = len(target)
    if preds.shape[1] == 0:
        return target - target.mean()
    X = np.column_stack([np.ones(n), preds])
    try:
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    except np.linalg.LinAlgError:
        return target - target.mean()
    return target - X @ beta


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ma, mb = a.mean(), b.mean()
    da = a - ma; db = b - mb
    den = math.sqrt(float((da @ da)) * float((db @ db)) + 1e-12)
    return float(da @ db) / den


def _partial_corr(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> float:
    """a 与 b 在 Z 上的偏相关。"""
    ra = _ols_residuals(a, Z)
    rb = _ols_residuals(b, Z)
    return _corr(ra, rb)


class CausalDiscovery:
    """PCMCI 类因果发现 → 因果特征白名单（零依赖）。"""

    def __init__(self, feature_names: List[str] = FEATURE_NAMES,
                 alpha: float = 0.05, stability_boot: int = 50,
                 stability_thresh: float = 0.6, inv_tol: float = 0.6,
                 notears: bool = True, seed: int = 0):
        self.names = list(feature_names)
        self.p = len(self.names)
        self.alpha = alpha
        self.B = stability_boot
        self.stab_thresh = stability_thresh
        self.inv_tol = inv_tol
        self.use_notears = notears
        self.seed = seed
        self._zcrit = _norm_ppf(1 - alpha / 2)  # 双侧临界值（α=0.05 → ≈1.96）

    # ---- 单目标 PC：找出 y 的父节点（直接因果驱动）----
    def _parents_of_target(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        n, p = X.shape
        if n < p + 5:
            return []
        selected: List[int] = []
        for j in range(p):
            others = [k for k in range(p) if k != j]
            Z = X[:, others] if others else np.zeros((n, 0))
            r = _partial_corr(y, X[:, j], Z)
            r = max(-0.999999, min(0.999999, r))
            z = 0.5 * math.log((1 + r) / (1 - r))
            dof = n - len(others) - 3
            stat = math.sqrt(max(dof, 1)) * z
            if abs(stat) > self._zcrit:
                selected.append(j)
        return selected

    # ---- 稳定特征选择 ----
    def _stability_freq(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        freq = np.zeros(self.p)
        for b in range(self.B):
            idx = rng.integers(0, n, size=n)  # Bootstrap 重采样（有放回）
            sel = self._parents_of_target(X[idx], y[idx])
            for j in sel:
                freq[j] += 1
        return freq / self.B

    # ---- 不变因果结构：regime 间斜率符号一致 + 变异系数 < tol ----
    def _invariant_filter(self, cand: List[int], X: np.ndarray, y: np.ndarray,
                          regimes: np.ndarray) -> Dict[int, bool]:
        out: Dict[int, bool] = {}
        regimes_present = [r for r in ("TREND", "RANGE", "CRASH") if r in set(regimes)]
        for j in cand:
            xj = X[:, j]
            # 全样本斜率（标准化后 OLS）
            s0 = self._slope(xj, y)
            ok = True
            slopes = [s0]
            if regimes_present:
                signs = {math.copysign(1, s0)}
                for rg in regimes_present:
                    mask = regimes == rg
                    if mask.sum() < 10:
                        continue
                    sr = self._slope(xj[mask], y[mask])
                    slopes.append(sr)
                    signs.add(math.copysign(1, sr))
                if len(signs) > 1:        # 斜率符号跨 regime 翻转 → 非不变，丢弃
                    ok = False
                cv = np.std(slopes) / (abs(s0) + 1e-9)
                if cv > self.inv_tol:     # 斜率量级漂移过大 → 丢弃
                    ok = False
            out[j] = ok
        return out

    @staticmethod
    def _slope(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        xc = (x - x.mean()) / (x.std() + 1e-9)
        yc = (y - y.mean()) / (y.std() + 1e-9)
        return float(np.polyfit(xc, yc, 1)[0])

    # ---- NOTEARS 类 DAG（辅助，异常时静默降级）----
    def _fit_notears(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            return _notears_dag(X, seed=self.seed)
        except Exception:
            return None

    # ---- 主入口 ----
    def fit(self, X: np.ndarray, y: np.ndarray,
            regimes: Optional[np.ndarray] = None) -> CausalReport:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        assert p == self.p, f"特征数 {p} 与声明 {self.p} 不符"

        freq = self._stability_freq(X, y)
        stable_set = [j for j in range(p) if freq[j] >= self.stab_thresh]

        invariant: Dict[int, bool] = {j: True for j in stable_set}
        if regimes is not None:
            invariant = self._invariant_filter(stable_set, X, y, np.asarray(regimes))

        whitelist_idx = [j for j in stable_set if invariant.get(j, True)]
        whitelist = [self.names[j] for j in whitelist_idx]

        # 父节点偏相关权重（在全样本上）
        parents_edges: Dict[str, float] = {}
        for j in whitelist_idx:
            others = [k for k in range(p) if k != j]
            Z = X[:, others] if others else np.zeros((n, 0))
            parents_edges[self.names[j]] = round(_partial_corr(y, X[:, j], Z), 3)

        dag = self._fit_notears(np.column_stack([X, y])) if self.use_notears else None

        return CausalReport(
            whitelist=whitelist,
            stable_freq={self.names[j]: round(float(freq[j]), 3) for j in range(p)},
            invariant={self.names[j]: bool(invariant.get(j, True)) for j in whitelist_idx},
            parents_edges=parents_edges,
            dag=dag,
            n_samples=n,
        )


# ============================================================================
# NOTEARS（线性 SEM，增广拉格朗日 + Adam），零依赖辅助 DAG 求解。
# h(W) = tr((I + W∘W / p)^p) - p（Zheng et al. 2018 经典 DAG 约束）。
# ============================================================================
def _notears_dag(X: np.ndarray, lambda1: float = 0.05, max_outer: int = 40,
                 inner_steps: int = 60, lr: float = 0.02, seed: int = 0,
                 h_tol: float = 1e-2) -> Optional[np.ndarray]:
    n, p = X.shape
    Xc = X - X.mean(0)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(p, p))
    rho = 1.0
    alpha = 0.0
    m_w = np.zeros_like(W); m_b = np.zeros_like(W)  # Adam 动量
    for _ in range(max_outer):
        for t in range(inner_steps):
            M = np.eye(p) + (W * W) / p
            Mp = np.linalg.matrix_power(M, p - 1)       # M^{p-1}
            h = float(np.trace(np.linalg.matrix_power(M, p)) - p)
            G_h = 2.0 * Mp * W                            # ∇h = 2·M^{p-1}∘W
            g_iso = (Xc.T @ Xc @ W - Xc.T @ Xc) / n
            g_l1 = lambda1 * np.sign(W)
            g = g_iso + g_l1 + (rho * h + alpha) * G_h
            # Adam
            m_w = 0.9 * m_w + 0.1 * g
            m_b = 0.999 * m_b + 0.001 * (g * g)
            W = W - lr * m_w / (np.sqrt(m_b) + 1e-8)
        M = np.eye(p) + (W * W) / p
        h = float(np.trace(np.linalg.matrix_power(M, p)) - p)
        if h <= h_tol:
            break
        alpha = alpha + rho * h
        rho = min(rho * 2.0, 1e6)
    # 阈值化得到稀疏 DAG
    scale = Xc.std(0).mean() + 1e-9
    W_thr = np.where(np.abs(W) > 0.05 * scale, W, 0.0)
    np.fill_diagonal(W_thr, 0.0)
    return W_thr
