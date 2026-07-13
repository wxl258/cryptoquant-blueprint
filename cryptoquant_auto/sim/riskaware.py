"""CVaR 约束 + Sharpe 目标（蓝图阶段4 · 任务21 · 专家A）。

蓝图锚点：RiskawareTrader（arXiv 2511.11481）——下行风险优先的 DRL 组合优化。
沙盒现实：完整 DRL 需 torch 重算力（后移至阶段3-4 云）；原型用**纯 numpy 滚动窗口
策略**实现同一目标函数，验证「CVaR 约束如何改变动作」，且零依赖可跑。

目标函数：score = Sharpe − λ·max(0, CVaR − budget)
  - CVaR = 期望短缺（最差 α 尾均值，负数）
  - 当 CVaR 比预算更负（越界）→ 重罚；RiskAwareTrader 据此砍仓到 HOLD
  - 与阶段0.5 的 DSR/PSR、阶段1 的 SPCI 同属「稳健优先」护栏体系

零依赖纪律：仅 numpy/stdlib。torch 缺失亦可用；DRL 完整版留待云环境。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """历史 CVaR（期望短缺）：最差 α 分位尾部的均值（负数）。"""
    r = np.asarray(returns, float)
    if len(r) < 2:
        return 0.0
    q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else q


def cvar_sharpe_score(returns: np.ndarray, alpha: float = 0.05,
                      cvar_budget: Optional[float] = None,
                      periods_per_year: int = 1, penalty: float = 10.0) -> float:
    """CVaR 约束 Sharpe：无约束返回普通 Sharpe；有预算则越界重罚。

    cvar_budget 为允许的最差尾均值（负数，如 -0.02）。CVaR 比预算更负即越界。
    """
    r = np.asarray(returns, float)
    if len(r) < 2:
        return 0.0
    sr = float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(periods_per_year))
    if cvar_budget is None:
        return sr
    cv = cvar(r, alpha)
    breach = max(0.0, cvar_budget - cv)     # cv 更负 → breach>0
    return sr - penalty * (breach / abs(cvar_budget + 1e-12))


class VanillaTrader:
    """基线：动量方向 + 满仓，无 CVaR 砍仓（用于受控 A/B 对照）。"""

    def decide(self, recent: np.ndarray, forecast_point: Optional[np.ndarray] = None
               ) -> Dict:
        r = np.asarray(recent, float)
        mean = float(r.mean()) if len(r) else 0.0
        action = "LONG" if mean > 0 else ("SHORT" if mean < 0 else "HOLD")
        return {"action": action, "exposure": 1.0, "score": None,
                "cvar": cvar(r) if len(r) >= 2 else 0.0, "breach": False}


class RiskAwareTrader:
    """CVaR 约束交易员（滚动窗口）：CVaR 越界→砍仓 HOLD；否则按动量/预测方向 + 约束得分缩放暴露。"""

    def __init__(self, alpha: float = 0.05, cvar_budget: float = -0.02,
                 window: int = 60, penalty: float = 10.0):
        self.alpha = alpha
        self.cvar_budget = cvar_budget
        self.window = window
        self.penalty = penalty

    def decide(self, recent: np.ndarray, forecast_point: Optional[np.ndarray] = None
               ) -> Dict:
        r = np.asarray(recent, float)
        if len(r) < 5:
            return {"action": "HOLD", "exposure": 0.0, "score": 0.0,
                    "cvar": 0.0, "breach": False}
        cv = cvar(r, self.alpha)
        score = cvar_sharpe_score(r, self.alpha, self.cvar_budget,
                                  penalty=self.penalty)
        # CVaR 越界（更负）→ 砍仓
        if cv < self.cvar_budget:
            return {"action": "HOLD", "exposure": 0.0, "score": score,
                    "cvar": cv, "breach": True}
        # 方向：预测点趋势 > 动量
        if forecast_point is not None and len(forecast_point):
            mean = float(forecast_point[-1] - forecast_point[0])
        else:
            mean = float(r.mean())
        action = "LONG" if mean > 0 else ("SHORT" if mean < 0 else "HOLD")
        # 暴露随约束得分缩放（越安全给越高，最低 0.3）
        exp = float(np.clip(0.3 + 0.7 * max(0.0, score), 0.0, 1.0))
        return {"action": action, "exposure": exp, "score": score,
                "cvar": cv, "breach": False}
