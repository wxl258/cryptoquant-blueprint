"""正经凯利仓位（移植服务器 Kelly 优点，替换线性近似）。

分数凯利（half-Kelly）以概率化方式定仓：f* = p - (1-p)/b，f_eff = frac·f*。
再受单币上限与 regime 上限二次约束（由 gate 四闸校验）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KellyConfig:
    # 【C6 修复 · 2026-07-12】win_rate_est 默认 0.53 与实测(~0.23)严重不符，
    # 会导致 quarter-Kelly 名义上限超配（0.53 vs 0.23 → f* 差数倍）。
    # 默认改保守值 0.30（低于实测上限，宁小勿大），并允许 callers 注入 OOS 实测 (p,b)。
    # 注：0.30 仍为保守缺省；解锁放行前须经 WFA/回测反推真实 (p,b) 注入，
    # 不得直接用 0.53 乐观值（见审计 C6 / 红队复核）。
    win_rate_est: float = 0.30    # 历史/回测胜率估计（保守缺省，待注入实测）
    payoff_ratio_est: float = 2.0 # 平均盈利/平均亏损（赔率 b，待注入实测）
    kelly_frac: float = 0.25      # 分数凯利（0.25=quarter-Kelly，最稳健；圆桌共识二 P1a）
    calibrated: bool = False      # True=已用 OOS 实测 (p,b) 校准；False=仍用保守缺省


def kelly_fraction(p: float, b: float) -> float:
    """全凯利 f* = p - (1-p)/b，负数截 0。"""
    if b <= 0:
        return 0.0
    return max(0.0, p - (1.0 - p) / b)


def fractional_kelly(p: float, b: float, frac: float = 0.25) -> float:
    return max(0.0, frac * kelly_fraction(p, b))


def kelly_nominal(equity: float, p: float, b: float, frac: float = 0.25,
                   cap_pct: float = 0.04) -> float:
    """返回仓位名义（USD），受单币上限 cap_pct 约束。frac 默认 quarter-Kelly（P1a）。"""
    f = fractional_kelly(p, b, frac)
    return equity * min(f, cap_pct)
