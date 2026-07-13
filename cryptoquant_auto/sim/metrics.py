"""回测指标：胜率、净值、夏普、最大回撤、各币 edge。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class BacktestStats:
    n_trades: int = 0
    win_rate: float = 0.0
    net_pnl_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    per_coin_edge_bps: Dict[str, float] = field(default_factory=dict)
    gate_b: Dict[str, bool] = field(default_factory=dict)


def _sharpe(equity: List[float], periods_per_year: int = 8760) -> float:
    # 【C4 修复 · 2026-07-12】回测每步是 1h bar，年化应用 8760(365*24) 而非 252(日线)。
    # 原 252 会把夏普系统性低估 ~5.8 倍（回测专家复核发现）。
    if len(equity) < 3:
        return 0.0
    rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def _max_dd(equity: List[float]) -> float:
    peak = equity[0] if equity else 1.0
    mdd = 0.0
    for e in equity:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1)
    return mdd


def summarize(equity: List[float], trades: List[dict], gate_b: Dict[str, bool]) -> BacktestStats:
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    net = (equity[-1] / equity[0] - 1) if equity else 0.0
    per_coin: Dict[str, List[float]] = {}
    for t in trades:
        per_coin.setdefault(t["symbol"], []).append(t["pnl_bps"])
    edge = {c: (sum(v) / len(v)) for c, v in per_coin.items()}
    return BacktestStats(
        n_trades=n,
        win_rate=(len(wins) / n) if n else 0.0,
        net_pnl_pct=net,
        sharpe=_sharpe(equity),
        max_dd_pct=_max_dd(equity),
        per_coin_edge_bps=edge,
        gate_b=gate_b,
    )


# ============================================================================
# 阶段0.5 · 验证层（蓝图 Gate）：PSR / DSR(N) / BH-FDR(N)
# 所有 edge 验收闸门都走这里，防「骗自己」（过拟合 / 多重检验假阳性）。
# 纯 numpy/math，零外部依赖，与回测器同量级轻量。
# ============================================================================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _annualize(returns: "np.ndarray", periods_per_year: int) -> float:
    """单样本夏普比（已实现口径）。"""
    T = len(returns)
    if T < 2:
        return 0.0
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def probabilistic_sharpe(returns, sr0: float = 0.0, periods_per_year: int = 8760) -> float:
    """概率夏普比 PSR(sr0)：真实 SR 超过目标 sr0 的概率（Lopez de Prado）。

    PSR = Φ[ (sr - sr0)·√T / √(1 - skew·sr0 + (kurt-1)/4·sr0²) ]
    其中 sr 为样本夏普，T 为收益观测数。返回 [0,1]。
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 5:
        return 0.0
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    sr = mean / std * math.sqrt(periods_per_year)
    skew = float((((r - mean) / std) ** 3).mean())
    kurt = float((((r - mean) / std) ** 4).mean())
    denom = math.sqrt(max(1e-9, 1.0 - skew * sr0 + (kurt - 1.0) / 4.0 * sr0 ** 2))
    z = (sr - sr0) * math.sqrt(T) / denom
    return _norm_cdf(z)


def _expected_max_normal(n: int) -> float:
    """N 个独立标准正态样本最大值的期望（Gumbel 近似）。

    用于多重检验校正：N 次独立试验的最大 SR 期望 ≈ E[max]/√T。
    """
    if n <= 1:
        return 0.0
    ln = math.log(n)
    return math.sqrt(2.0 * ln) - (math.log(ln) + math.log(4.0 * math.pi)) / (2.0 * math.sqrt(2.0 * ln))


def deflated_sharpe(returns, n_trials: int = 1, sr0: float = 0.0,
                    periods_per_year: int = 8760) -> float:
    """紧缩夏普比 DSR(sr0; N)：多重检验校正后的「真实 SR > 阈值」概率。

    蓝图阶段0.5 核心闸门。阈值 sr*_N 取「N 次独立试验期望最大 SR」：
        sr*_N = sr0 + E[max of N Gaussians] / √T
    n_trials=1 时退化为 PSR(sr0)。N 越大，门槛越高 → 滤掉多重检验假阳性。
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 5:
        return 0.0
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    sr = mean / std * math.sqrt(periods_per_year)
    skew = float((((r - mean) / std) ** 3).mean())
    kurt = float((((r - mean) / std) ** 4).mean())
    gN = _expected_max_normal(max(1, n_trials))
    sr_star = sr0 + gN / math.sqrt(T)          # 多重检验调整后阈值
    denom = math.sqrt(max(1e-9, 1.0 - skew * sr_star + (kurt - 1.0) / 4.0 * sr_star ** 2))
    z = (sr - sr_star) * math.sqrt(T) / denom
    return _norm_cdf(z)


def bh_fdr(pvals, alpha: float = 0.10):
    """Benjamini-Hochberg FDR 修正（阶段0.5 多重检验控制）。

    输入原始 p 值列表，返回 (reject_mask, adjusted_pvals)。
    reject_mask[i]=True 表示第 i 个假设在 FDR 水平 α 下显著。
    """
    m = len(pvals)
    if m == 0:
        return [], []
    idx = sorted(range(m), key=lambda i: pvals[i])
    max_k = -1
    for k in range(m):
        if pvals[idx[k]] <= alpha * (k + 1) / m:
            max_k = k
    reject = [False] * m
    for k in range(max_k + 1):
        reject[idx[k]] = True
    adj = [0.0] * m
    cur = 1.0
    for k in reversed(range(m)):
        raw = pvals[idx[k]]
        cur = min(cur, raw * m / (k + 1), 1.0)
        adj[idx[k]] = cur
    return reject, adj

