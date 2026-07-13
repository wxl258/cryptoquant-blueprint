"""仓位计算（吸收生产系统 risk/control.py 优点 P0-C/P0-E）。

实现生产系统 7 步仓位链 calc_position_pct：
  base(评分0.5~4.5%) × 凯利预算上限(半凯利) × ADX加成 × 连亏递减 × 周末 × S/R × 波动率 × ATR封顶 × 冷启动
  → 硬顶5% / 下限0.3%
  注：连赢不再加仓——反凯利顺周期加注已被专家圆桌否决（共识#3/#7）。

另含：
  - 凯利风险预算上限 f* = W - (1-W)/RR
  - 杠杆感知名义敞口（v31.22）：总名义 ≤ NOTIONAL_CAP，VaR 风险加权 ≤ RISK_BUDGET_CAP
  - RISK_PROFILE 三档（保守/均衡/进取）

纯算法，零资金；trade_state 以 dict 注入（Mock 或历史回放），不依赖生产 trade_state 文件。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# RISK_PROFILE 三档（对齐生产系统）
RISK_PROFILES = {
    "保守": {"cap": 8, "mdir": 2, "min_score": 6, "min_adx": 28, "desc": "低风险: 仓位少、门槛高"},
    "均衡": {"cap": 12, "mdir": 3, "min_score": 5, "min_adx": 25, "desc": "当前默认配置"},
    "进取": {"cap": 16, "mdir": 4, "min_score": 4, "min_adx": 22, "desc": "高仓位、低门槛"},
}
ACTIVE_PROFILE = "均衡"
NOTIONAL_CAP = 30.0       # 总名义敞口(含杠杆)上限 %
RISK_BUDGET_CAP = 1.5     # VaR 风险加权敞口上限 %

# P1-3：赔率 b（盈亏比）回退默认值 —— 仅当调用方未传入实测值时使用。
# 警告：rr 必须来自回测 OOS 实测（estimate_pb），硬编码默认值是乐观假设，
# 不得作为实盘依据。WFA 流程已用 estimate_pb 实测 (p,b) 替换本默认值。
DEFAULT_PAYOFF_FALLBACK = 2.0
KELLY_FRAC = 0.25         # P1a：quarter-Kelly（最稳健，圆桌共识二）


@dataclass
class PositionResult:
    pct: float
    label: str
    factors: Dict[str, str] = field(default_factory=dict)


def calc_position_pct(score: float, adx: float, atr_pct: float,
                      trade_state: Optional[dict] = None,
                      is_weekend: bool = False, sr_risk: Optional[str] = None,
                      vol_regime: str = "stable", rr: float = DEFAULT_PAYOFF_FALLBACK,
                      profile: str = ACTIVE_PROFILE) -> PositionResult:
    """统一仓位计算（生产 7 步链，自包含重写）。

    rr（赔率 b）= 平均盈利/平均亏损，应由回测 OOS 实测（estimate_pb）注入；
    未注入时退化为 DEFAULT_PAYOFF_FALLBACK（P1-3：硬编码默认值仅作回退，非实盘依据）。
    """
    factors: Dict[str, str] = {}
    # 第1步: 基础仓位(评分映射 0.5%~4.5%)
    if score >= 10: base = 4.5
    elif score >= 9: base = 4.0
    elif score >= 8: base = 3.5
    elif score >= 7: base = 2.5
    elif score >= 6: base = 1.5
    elif score >= 5: base = 1.0
    else: base = 0.5
    factors["评分"] = f"{score}分→{base}%"

    # 凯利风险预算上限（quarter-Kelly：P1a 最稳健；全凯利回撤过深、p/b 皆为估计值易失真）
    ts = trade_state or {}
    W = max(0.0, min(1.0, (ts.get("win_rate", 50) or 50) / 100.0))
    kelly_f = KELLY_FRAC * (W - (1.0 - W) / max(float(rr), 0.5))
    kelly_base = max(0.3, min(5.0, kelly_f * 100.0))
    factors["凯利"] = f"f*={kelly_f:.3f}(quarter-Kelly, W={W:.0%},RR={rr}) 取小{base:.1f}/{kelly_base:.1f}"
    base = min(base, kelly_base)

    # ADX 加成
    if adx >= 45: base *= 1.2; factors["ADX加成"] = "×1.2(强趋势)"
    elif adx >= 35: base *= 1.1; factors["ADX加成"] = "×1.1(明确趋势)"
    else: factors["ADX加成"] = "×1.0"

    # 连亏递减（保留）；连赢不再加仓——反凯利顺周期加注已被专家否决（共识#3/#7）
    sl = ts.get("streak_losses", 0)
    if sl >= 3:
        base *= 0.6; factors["连亏"] = f"×0.6(连亏{sl}次)"
    elif sl >= 2:
        base *= 0.8; factors["连亏"] = f"×0.8(连亏{sl}次)"
    elif sl >= 1:
        base *= 0.9
    else:
        factors["连胜率"] = "×1.0(正常)"

    # 盈利递减（共识#7：盈利后应降暴露，而非顺周期加注）——与连亏递减对称、方向相反。
    # 近期累计收益为正且盈利笔数多 → 降低新仓比例（锁定已有浮盈、压低不确定性暴露）。
    profit_steps = ts.get("profit_steps", 0)      # 连续盈利笔数（或近期盈利笔数）
    cum_pnl_pct = ts.get("cum_pnl_pct", 0.0)      # 近期累计权益收益（正=盈利）
    if profit_steps >= 3 and cum_pnl_pct > 0:
        base *= 0.7; factors["盈利递减"] = f"×0.7(连盈{profit_steps}笔,累计+{cum_pnl_pct:.1%})"
    elif profit_steps >= 2 and cum_pnl_pct > 0:
        base *= 0.85; factors["盈利递减"] = f"×0.85(连盈{profit_steps}笔)"

    # 周末降权
    if is_weekend:
        base *= 0.7; factors["周末"] = "×0.7"

    # S/R 风险降权
    if sr_risk:
        base *= 0.7; factors["S/R风险"] = "×0.7"

    # 波动率
    if vol_regime == "contracting":
        base *= 0.8; factors["波动率"] = "×0.8(收敛)"
    elif vol_regime == "expanding":
        base *= 1.1; factors["波动率"] = "×1.1(扩张)"

    # ATR 动态封顶（atr_pct=ATR占价百分比，calc_atr 返回，常态1-4%；>3%高 / >5%极度）
    if atr_pct > 5.0:
        base *= 0.6; factors["ATR封顶"] = f"×0.6(ATR={atr_pct:.1f}%极度)"
    elif atr_pct > 3.0:
        base *= 0.8; factors["ATR封顶"] = f"×0.8(ATR={atr_pct:.1f}%高)"

    # 冷启动折扣（样本恢复中）
    cold = ts.get("cold_discount")
    if cold is not None and cold < 1.0:
        base *= cold; factors["冷启动"] = f"×{cold}(样本恢复中)"

    # 硬顶/下限
    pct = round(max(min(base, 5.0), 0.3), 1)
    if pct >= 4.0: label = f"🟢重仓 {pct}%"
    elif pct >= 2.5: label = f"🟢标准 {pct}%"
    elif pct >= 1.5: label = f"🟡半仓 {pct}%"
    elif pct >= 0.8: label = f"🟡轻仓 {pct}%"
    else: label = f"⚪观察 {pct}%"
    return PositionResult(pct=pct, label=label, factors=factors)


def validate_total_exposure(active: List[dict], profile: str = ACTIVE_PROFILE) -> List[dict]:
    """校验并执行仓位上限（吸收生产 validate_total_exposure）。

    active: [{symbol, position_pct, leverage_max, direction}]
    返回带修正后 position_pct 的列表（超出等比缩减）。
    """
    prof = RISK_PROFILES.get(profile, RISK_PROFILES[ACTIVE_PROFILE])
    cap_total = 12.0  # 总仓位硬上限（代码事实，非旧卡10%）
    # 1) 杠杆感知名义敞口（含杠杆）
    notional = sum(max(s.get("position_pct", 0), 0) * max(s.get("leverage_max", 1), 1) for s in active)
    if notional > NOTIONAL_CAP:
        scale = NOTIONAL_CAP / max(notional, 0.1)
        for s in active:
            s["position_pct"] = round(s["position_pct"] * scale, 2)
    # 2) 总仓位(无杠杆)超12% → 等比缩减
    total = sum(max(s.get("position_pct", 0), 0) for s in active)
    if total > cap_total:
        scale = cap_total / max(total, 0.1)
        for s in active:
            s["position_pct"] = round(s["position_pct"] * scale, 2)
    return active


def leverage_aware_notional(active: List[dict]) -> float:
    """总名义敞口(含杠杆)，供 VaR 风控。"""
    return round(sum(max(s.get("position_pct", 0), 0) * max(s.get("leverage_max", 1), 1)
                     for s in active), 2)
