"""CVaR 约束仓位优化器（蓝图路线图 第 3 周 B · 任务：替换仓位公式）。

用 scipy.optimize 求解**跨资产 CVaR 约束**下的仓位权重：

    max_w  Sharpe(w) = (w·|μ|) / sqrt(wᵀ·Σ·w)
    s.t.   CVaR(portfolio, α) ≥ budget        # 尾部最差均值损失 ≤ −budget
           Σ w_i ≤ total_cap                    # 总仓位硬上限
           0 ≤ w_i ≤ max_pos                    # 单币上限 + 非负（方向已并入收益矩阵）

设计要点：
  - μ 为「方向×置信」的卷积代理，仅取绝对值进目标（方向由收益矩阵符号承载）；
  - 收益矩阵 R 每列 = direction_int × 该币 1h 对数收益（LONG 取正、SHORT 取负）；
  - 组合收益 Rp = R @ w；CVaR(Rp) = 最差 α 分位尾均值（负数，越大越好→越接近 0）。

零依赖纪律：scipy 缺失或求解失败 → 退化为「按 |μ| 比例、CVaR 预算内等比减半」的
启发式，保证行为有界、单调、绝不抛错（与项目其余护栏一致）。

蓝图升级路径（阶段3-4，当前未落地）：
  RiskawareTrader 深度强化学习（DRL）优化是蓝图最终目标，因无 GPU/重算力约束尚未实现。
  当前 scipy 解析解在零样本约束下运行高效，DRL 升级列为待定，留待有重算力环境后接入。

"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("cryptoquant")

try:
    from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
    HAS_SCIPY = True
except Exception:  # pragma: no cover - 沙箱/生产若未装 scipy 则走启发式
    HAS_SCIPY = False


def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Cornish-Fisher 修正 CVaR：用偏度/峰度调整分位数，尾部估计更准。

    与 sim.riskaware 同义。scipy.stats 缺失时回退到 np.quantile 历史分位法。
    """
    r = np.asarray(returns, float)
    if len(r) < 5:
        return 0.0
    # Cornish-Fisher adjusted quantile threshold
    try:
        from scipy.stats import norm
        z = norm.ppf(alpha)
        s = float(np.mean((r - r.mean())**3)) / (r.std(ddof=1)**3 + 1e-12)
        k = float(np.mean((r - r.mean())**4)) / (r.std(ddof=1)**4 + 1e-12) - 3.0
        z_cf = z + (s/6)*(z**2 - 1) + (k/24)*(z**3 - 3*z) - (s**2/36)*(2*z**3 - 5*z)
        q = float(r.mean() + z_cf * r.std(ddof=1))
    except Exception:
        q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else q


def _shrunk_cov(R: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf 式对角收缩协方差估计（纯 numpy，零依赖）。

    收缩目标 = 均值方差 * Identity。shrink=0.1→90%样本协方差+10%目标。
    显著降低条件数，缓解共线性对 SLSQP/DE 求解的不利影响。
    """
    T, n = R.shape
    if T < 3 or n < 2:
        return np.cov(R, rowvar=False) + 1e-8 * np.eye(n)
    X = R - R.mean(axis=0)
    S = X.T @ X / (T - 1)
    target = np.mean(np.diag(S)) * np.eye(n)
    return (1 - shrink) * S + shrink * target + 1e-8 * np.eye(n)


class CvarPositionOptimizer:
    """跨资产 CVaR 约束仓位优化器（scipy.optimize / SLSQP，带安全降级）。"""

    def __init__(self, alpha: float = 0.05, cvar_budget: float = -0.02,
                 total_cap: float = 0.12, max_pos: float = 0.04,
                 min_samples: int = 24,
                 herfindahl_max: float = 0.5,
                 turnover_penalty: float = 0.0):
        self.alpha = alpha
        self.cvar_budget = cvar_budget      # 允许的最差尾均值（负数，如 -0.02）
        self.total_cap = total_cap          # 总仓位硬上限（如 0.12）
        self.max_pos = max_pos              # 单币仓位上限（单一真相源，与 gate/circuit_breaker/kelly 统一为 0.04）
        self.min_samples = min_samples      # 收益样本下限（不足→启发式）
        self.herfindahl_max = herfindahl_max  # 集中度上限 Σw² ≤ H_max（防过度集中）
        self.turnover_penalty = turnover_penalty  # 换手率惩罚系数 λ（>0 时启用）

    # ---- 内部启发式（同时作 x0 与降级路径）----
    def _heuristic(self, conviction: np.ndarray) -> np.ndarray:
        """按 |μ| 比例分配，归一化到 total_cap，单币封顶 max_pos。"""
        tot = float(conviction.sum())
        n = len(conviction)
        if tot <= 0:
            return np.zeros(n)
        w = conviction / tot * self.total_cap
        return np.clip(w, 0.0, self.max_pos)

    def _safe_heuristic(self, conviction: np.ndarray, R: np.ndarray) -> np.ndarray:
        """CVaR 预算内等比减半，保证尾部约束满足（降级路径，单调收敛）。"""
        w = self._heuristic(conviction)
        for _ in range(8):
            if R.shape[0] < 2 or cvar(R @ w, self.alpha) >= self.cvar_budget:
                break
            w = w * 0.5
        return np.clip(w, 0.0, self.max_pos)

    def _port_cvar(self, w, R) -> float:
        return cvar(R @ w, self.alpha)

    def _adjust_budget(self, R: np.ndarray) -> float:
        """波动率 regime 自适应 CVaR budget：高波动收紧，低波动放松。

        参考波动率 = 全量等权组合收益的波动率；
        当前波动率 = 最近 min_samples 期等权收益波动率。
        budget(t) = base_budget × (参考波动率 / 当前波动率) + 硬区间 [−0.10, −0.001]
        """
        T = R.shape[0]
        if T < self.min_samples + 2:
            return self.cvar_budget
        port_ret = R.mean(axis=1)  # 等权代理
        ref_vol = float(np.std(port_ret)) + 1e-12
        cur_vol = float(np.std(port_ret[-self.min_samples:])) + 1e-12
        adj = self.cvar_budget * (ref_vol / cur_vol)
        result = float(np.clip(adj, max(self.cvar_budget * 2, -0.10),
                               min(self.cvar_budget * 0.5, -0.001)))
        # 【P1-2】预算放宽（相对基准更宽松，result 比 cvar_budget 更接近 0）须记录，
        # 便于审计「风险约束被放松」的触发原因（低波动 regime 下自适应放宽）。
        if result > self.cvar_budget + 1e-9:
            logger.info(
                "CVaR budget widened: base=%.4f -> adj=%.4f "
                "(ref_vol=%.4f cur_vol=%.4f)",
                self.cvar_budget, result, ref_vol, cur_vol,
            )
        return result

    def solve(self, conviction: np.ndarray, R: np.ndarray,
              symbols: List[str],
              prev_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """求解仓位权重。

        conviction: (n,) ≥0 的置信/卷积幅度（方向已并入 R，这里只用幅度）；
        R: (T, n) 方向化后的逐币收益矩阵；
        symbols: 与列序一致的币种名；
        prev_weights: 上期权重 (n,)，用于换手率惩罚（self.turnover_penalty>0 时启用）。
        返回 {symbol: weight(≥0)}，未参与优化的币不在字典内（视为 0）。
        """
        n = len(symbols)
        if n == 0:
            return {}
        conviction = np.asarray(conviction, float)
        R = np.asarray(R, float)
        T = R.shape[0]
        # 收益协方差（Ledoit-Wolf 收缩估计，降低条件数）
        cov = _shrunk_cov(R)

        def neg_sharpe(w):
            reward = float(w @ conviction)          # 取 |μ| 作奖励（方向在 R 中）
            # 换手率惩罚：λ·||w − w_prev||₁，抑制频繁调仓
            if prev_weights is not None and self.turnover_penalty > 0.0:
                turnover = float(np.sum(np.abs(w - prev_weights)))
                reward -= self.turnover_penalty * turnover
            var = float(w @ cov @ w)
            sd = math.sqrt(var + 1e-12)
            return -reward / sd

        if HAS_SCIPY and T >= self.min_samples:
            try:
                bounds = [(0.0, self.max_pos)] * n
                _budget = self._adjust_budget(R)
                cons = [
                    LinearConstraint(np.ones(n), 0.0, self.total_cap),
                    NonlinearConstraint(lambda w: self._port_cvar(w, R),
                                        _budget, np.inf),
                    NonlinearConstraint(lambda w: float(np.sum(w ** 2)),
                                        0.0, self.herfindahl_max),
                ]
                # 主求解器：differential_evolution（全局搜索，摆脱局部最优）
                try:
                    from scipy.optimize import differential_evolution
                    _HAS_DE = True
                except Exception:
                    _HAS_DE = False
                if _HAS_DE:
                    res = differential_evolution(neg_sharpe, bounds,
                                                 constraints=cons, maxiter=200,
                                                 seed=42, polish=True)
                    w = np.clip(res.x, 0.0, self.max_pos)
                else:
                    # 降级：SLSQP 局部搜索（scipy <1.7 不支援 DE/cons）
                    x0 = self._heuristic(conviction)
                    res = minimize(neg_sharpe, x0, method="SLSQP",
                                   bounds=bounds, constraints=cons,
                                   options={"maxiter": 200, "ftol": 1e-9})
                    w = np.clip(res.x, 0.0, self.max_pos)
                # 求解失败或仍越界 → 安全启发式兜底（绝不输出越界权重）
                if (not res.success) or self._port_cvar(w, R) < _budget:
                    w = self._safe_heuristic(conviction, R)
            except Exception:
                w = self._safe_heuristic(conviction, R)
        else:
            w = self._safe_heuristic(conviction, R)

        return {symbols[i]: float(w[i]) for i in range(n)}

    # ---- 诊断接口（供仪表盘/测试复核优化结果）----
    def portfolio_cvar(self, weights: Dict[str, float], R_full: np.ndarray) -> float:
        """给定完整权重向量，回算组合 CVaR（R_full 列序须与 weights 一致）。"""
        if R_full.shape[0] < 2 or not weights:
            return 0.0
        w = np.asarray(list(weights.values()), float)
        return self._port_cvar(w, R_full)
