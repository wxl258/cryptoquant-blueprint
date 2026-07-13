"""assert_pre_trade 四闸门：regime_cap / beta_agg / single_cap / thermo。

对应风控专家方案的 Fail-closed 前置拦截：任何一笔订单在下发前必须过四闸，
否则拒单（不依赖信号回传）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models import Order, Position, Direction

# 各币对 BTC 的 beta（相关性代理）；用于 β 聚合有效敞口
BETA: Dict[str, float] = {"BTC": 1.0, "ETH": 0.85, "BNB": 0.70, "SOL": 0.80, "XRP": 0.55}

TOTAL_CAP_PCT = 0.12     # 总仓位硬上限（名义/权益）
SINGLE_CAP_PCT = 0.04    # 单币上限


@dataclass
class GateConfig:
    equity: float = 100_000.0
    total_cap_pct: float = TOTAL_CAP_PCT
    single_cap_pct: float = SINGLE_CAP_PCT
    regime: str = "TREND"            # TREND | RANGE | CRASH
    beta_reject_coins: int = 3       # 同向 >= 此币数触发拒绝
    beta_reject_pct: float = 0.08    # 同向聚合有效敞口 >= 此比例拒绝
    thermo_max_ret: float = 0.05     # 单币近期收益超此值禁入（过热）
    min_ev: float = 0.0              # EV 期望值闸门下限（共识#2：负期望一律拒）
    enforce_gate_b: bool = True      # P1-4：成本盈亏平衡闸门（Gate B）是否生效（实盘 True / 回测测量 False）


@dataclass
class GateResult:
    ok: bool = True
    reasons: List[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)


def _regime_mult(regime: str) -> float:
    return {"TREND": 1.0, "RANGE": 0.5, "CRASH": 0.0}.get(regime, 1.0)


def _beta_sign(side: str) -> int:
    return 1 if side == "BUY" else -1


def compute_beta_exposure(positions: List[Position], candidate: Order | None = None) -> float:
    """β 聚合有效敞口 = Σ(名义_i × β_i × 方向sign)。"""
    exp = 0.0
    for p in positions:
        sign = p.direction.sign
        exp += p.qty * p.entry_price * BETA.get(p.symbol, 0.7) * sign
    if candidate is not None:
        sign = _beta_sign(candidate.side)
        exp += candidate.qty * candidate.price * BETA.get(candidate.symbol, 0.7) * sign
    return exp


def assert_pre_trade(order: Order, positions: List[Position], cfg: GateConfig,
                     recent_ret: Dict[str, float], ev_est: Optional[float] = None) -> GateResult:
    res = GateResult()
    nominal = order.qty * order.price

    # 1) single_cap
    if nominal > cfg.equity * cfg.single_cap_pct:
        res.fail(f"single_cap: {nominal/cfg.equity:.1%} > {cfg.single_cap_pct:.1%}")

    # 2) total_cap（受 regime 缩放）
    used = sum(p.qty * p.entry_price for p in positions)
    cap = cfg.equity * cfg.total_cap_pct * _regime_mult(cfg.regime)
    if used + nominal > cap:
        res.fail(f"total_cap: used {used/cfg.equity:.1%} + new {nominal/cfg.equity:.1%} > {cap/cfg.equity:.1%} (regime={cfg.regime})")

    # 3) regime_cap
    if cfg.regime == "CRASH":
        res.fail("regime CRASH: 禁止新开仓")

    # 4) beta_agg（同向 >=3 币且聚合 >=8% -> 拒同向加仓）
    beta_exp = compute_beta_exposure(positions, order)
    dirs = [p.direction.sign for p in positions] + [_beta_sign(order.side)]
    same_dir = len(set(dirs)) == 1
    n_coins = len({p.symbol for p in positions} | {order.symbol})
    if same_dir and n_coins >= cfg.beta_reject_coins and abs(beta_exp) >= cfg.beta_reject_pct * cfg.equity:
        res.fail(f"beta_agg: {n_coins}币同向, 有效敞口 {abs(beta_exp)/cfg.equity:.1%} >= {cfg.beta_reject_pct:.1%}")

    # 5) thermo（过热禁入）
    r = recent_ret.get(order.symbol, 0.0)
    if abs(r) > cfg.thermo_max_ret:
        res.fail(f"thermo: {order.symbol} 近期收益 {r:.1%} > {cfg.thermo_max_ret:.1%}")

    # 6) EV 期望值闸门（共识#2：正期望值 > 胜率；负期望一律拒，高胜率也不救）
    if ev_est is not None and ev_est < cfg.min_ev:
        res.fail(f"ev_negative: EV={ev_est:.4f} < {cfg.min_ev:.4f}")

    # 7) Gate B 成本盈亏平衡闸门（P1-4：实盘路径强制，回测测量可关）
    #    若「系统校准毛 edge − 最坏成本」<= 0，则该币在任何行情下净期望 ≤ 0，
    #    禁止开仓（fail-closed）。GROSS_EDGE_BPS 默认锁定为 -2.5bps（跨所5年OOS最保守锚点），
    #    故当前对所有币生效 → 系统在 edge 未经大样本复核前维持空仓（圆桌共识）。
    if cfg.enforce_gate_b:
        from ..risk.exec_cost import worst_case_pnl
        net_edge = worst_case_pnl(order.symbol)
        if net_edge <= 0:
            res.fail(f"gate_b: {order.symbol} 最坏成本净edge={net_edge:+.2f}bps<=0，不可交易")

    return res
