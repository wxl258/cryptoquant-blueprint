"""受控 A/B 验证 harness（蓝图阶段0.5：LLM vs 规则 同流对比 + DSR 显著）。

给定同一 universes 的两组收益流（rule / llm），回答：
  - 哪组 DSR(N) 更高（多重检验校正后的 edge 验收）？
  - 两组差异是否统计显著（Welch t 检验双侧 p）？
  - 结论是否可据以「放行 LLM 替代规则」（否则回退规则）。

零依赖（numpy + math），可脱离真实数据用合成收益流跑通，便于 CI / 沙盒自检。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

from .metrics import deflated_sharpe, probabilistic_sharpe, bh_fdr


# ---------- 轻量 t 分布（双侧 p，无 scipy）----------
def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _lag1_acf(x: np.ndarray) -> float:
    """lag-1 自相关（时序非 iid 的代理）。"""
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    if x.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _eff_n(n: int, rho: float) -> float:
    """自相关下的有效样本量：方差膨胀因子 (1+ρ)/(1-ρ) 的倒数。

    ρ→±1 时退化为保守保护值，避免除零/极端自由度。
    """
    if abs(rho) >= 0.999:
        return max(2.0, n / 10.0)
    return max(2.0, n * (1.0 - rho) / (1.0 + rho))


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Welch 两样本 t 检验双侧 p 值（正确 t 分布）。

    【P1-16 修复】金融收益序列**非 iid**（自相关），朴素 Welch 把有效样本量当成
    名义 n，使 p 值被系统性低估（假显著、易误放行）。用 lag-1 自相关做 Newey-West
    风格修正：方差按 (1+ρ)/(1-ρ) 膨胀、自由度用有效样本量；iid 输入(ρ≈0)则与原值一致。
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    va = a.var(ddof=1); vb = b.var(ddof=1)
    ma = a.mean(); mb = b.mean()
    rho_a, rho_b = _lag1_acf(a), _lag1_acf(b)
    if abs(rho_a) < 0.999:
        va *= (1.0 + rho_a) / (1.0 - rho_a)
    if abs(rho_b) < 0.999:
        vb *= (1.0 + rho_b) / (1.0 - rho_b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 1.0
    t = (ma - mb) / se
    na_eff, nb_eff = _eff_n(na, rho_a), _eff_n(nb, rho_b)
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na_eff - 1) + (vb / nb) ** 2 / (nb_eff - 1)
    df = df_num / df_den if df_den > 0 else (na_eff + nb_eff - 2)
    df = max(1.0, min(df, na_eff + nb_eff - 2))
    x = df / (df + t * t)
    p_one = 0.5 * _betai(df / 2.0, 0.5, x)
    return float(min(1.0, 2.0 * p_one))


@dataclass
class ABResult:
    dsr_rule: float = 0.0
    dsr_llm: float = 0.0
    psr_rule: float = 0.0
    psr_llm: float = 0.0
    p_value: float = 1.0
    n_trials: int = 1
    winner: str = "tie"
    significant: bool = False
    recommend_llm: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


def controlled_ab(returns_rule, returns_llm, n_trials: int = 1,
                  sr0: float = 0.0, alpha: float = 0.05,
                  periods_per_year: int = 1) -> ABResult:
    """受控 A/B：同 universe 下 rule vs llm 收益流对比。

    returns_rule / returns_llm: 等长或近似等长的逐 bar 收益序列（numpy/列表）。
    n_trials: 该对比所代表的独立试验数（GP 代数×种群 → 多重检验校正）。
    periods_per_year: 年化周期数。合成自检统一传 1（直接吃单 bar SR），
        避免 1h 年化(×93.6) 把边缘 edge 压成采样噪声。
    结论：llm 在 DSR 更高 且 差异显著(p<alpha) 且 双方 DSR>0 时，才建议放行 LLM。
    """
    r = np.asarray(returns_rule, float)
    l = np.asarray(returns_llm, float)
    res = ABResult(n_trials=n_trials)
    res.dsr_rule = deflated_sharpe(r, n_trials=n_trials, sr0=sr0,
                                   periods_per_year=periods_per_year)
    res.dsr_llm = deflated_sharpe(l, n_trials=n_trials, sr0=sr0,
                                  periods_per_year=periods_per_year)
    res.psr_rule = probabilistic_sharpe(r, sr0=sr0, periods_per_year=periods_per_year)
    res.psr_llm = probabilistic_sharpe(l, sr0=sr0, periods_per_year=periods_per_year)
    res.p_value = _welch_p(r, l)
    res.significant = res.p_value < alpha
    if res.dsr_llm > res.dsr_rule:
        res.winner = "llm"
    elif res.dsr_rule > res.dsr_llm:
        res.winner = "rule"
    else:
        res.winner = "tie"
    # 放行条件：LLM 更优 + 显著 + 双方 DSR>0（有真实 edge，非噪声胜出）
    res.recommend_llm = (res.winner == "llm" and res.significant
                         and res.dsr_llm > 0 and res.dsr_rule > 0)
    res.detail = {
        "n_rule": int(len(r)), "n_llm": int(len(l)),
        "mean_rule": float(r.mean()), "mean_llm": float(l.mean()),
        "sr_rule": float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(periods_per_year)),
        "sr_llm": float(l.mean() / (l.std(ddof=1) + 1e-12) * math.sqrt(periods_per_year)),
    }
    return res


def make_synthetic_returns(n: int, sr_target: float, seed: int = 7,
                           vol: float = 0.01) -> np.ndarray:
    """生成目标「单 bar Sharpe 比」的合成收益（零均值正态）。

    sr_target 解释为**单 bar SR**（= 均值/波动），故 mean = sr_target·vol。
    调用方若要把它当年化收益，自行把 sr_target 乘 √年化周期数，或给 DSR/PSR
    传 periods_per_year。本 harness 自检时统一传 periods_per_year=1（即直接用
    单 bar SR），避免 1h 年化(×93.6) 把边缘 edge 压成采样噪声。
    """
    rng = np.random.default_rng(seed)
    mean = sr_target * vol
    return rng.normal(mean, vol, size=n)
