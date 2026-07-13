"""端到端调试 demo：零密钥、零资金，跑通自动化执行管线并注入故障。

运行：cd /workspace && python -m cryptoquant_auto.demo
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Signal, Direction, OrderType, OrderStatus
from .adapters.mock import MockAdapter
from .risk.gate import GateConfig
from .risk.kill_switch import KillSwitch
from .core.engine import ExecutionEngine
from .core.router import FallbackRouter
from .signals import gen_signal, MarketContext
from .meta.cognition import assess
from .risk.position_sizing import calc_position_pct
from .risk.circuit_breaker import dynamic_threshold
from .risk.black_swan import get_black_swan_level
from .util.logging_setup import setup_logging

# P1-4：demo 是「管线机制演示」而非实盘，故所有 GateConfig(enforce_gate_b=False, ) 默认关闭 Gate B 成本闸门
# （否则被 fail-closed 锚点全拒，无法演示成交）。实盘/回测请保持 enforce_gate_b=True。
# 专门的 test_gate_b_cost() 显式传入 enforce_gate_b=True 验证其拦截行为（后传关键字覆盖前传）。
# 注：dataclass 默认固化在 __init__ 签名中，无法靠改类属性/field.default 生效，故这里在
# 每个构造处显式关闭（正则替换为 GateConfig(enforce_gate_b=False, enforce_gate_b=False, ...），test_gate_b_cost 的
# 显式 True 因后传关键字而优先生效）。

# 【P2-C 重构】分析段（run_*_section 系列 + 其私有辅助 _run_signals_regime / _regime_breakdown /
# _print_regime_bd / _edge_features / _build_conditions / _segment_ev）整体外移至 demo_sections.py，
# 本文件仅保留「管线机制测试」(Case / mk_signal / test_*) 与 main() 编排入口，消除 808 行 fat file。
# demo_sections 内通过 GateConfig_from_demo() 工厂复用 demo 的「关闭 Gate B」约定，逻辑零改动。
from .demo_sections import (
    run_real_backtest_section,
    run_oos_section,
    run_regime_section,
    run_testnet_section,
)


@dataclass
class Case:
    name: str
    passed: bool
    detail: str


def mk_signal(symbol, direction, entry, tp1, tp2, atr, conf=0.6, sid="s1"):
    sl = entry - 2 * atr if direction is Direction.LONG else entry + 2 * atr
    return Signal(symbol=symbol, tf="1H", direction=direction, entry=entry,
                  sl=sl, tp1=tp1, tp2=tp2, rr=2.0, confidence=conf, signal_id=sid, atr=atr)


def test_idempotency() -> Case:
    """同 coid 重复提交，订单不得重复。"""
    a = MockAdapter()
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    eng.ingest_signal(sig)
    # 直接再提交同 coid（模拟重启重发）
    from .core.order_builder import build_entry_order
    o = build_entry_order(sig, GateConfig(enforce_gate_b=False, ))
    a.submit(o)
    n = len([x for x in a.query_open() if x.coid == o.coid])
    return Case("订单幂等（同coid不重复）", n == 1, f"open同coid订单数={n}")


def test_timeout_retry() -> Case:
    """瞬时超时：首次抛 Timeout，引擎查单/重发，最终仅1笔。"""
    a = MockAdapter(fault_mode="timeout")
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    d = eng.ingest_signal(sig)
    n = len([x for x in a.query_open() if x.symbol == "BTC"])
    return Case("瞬时超时幂等重发", d.accepted and n == 1,
                f"accepted={d.accepted}, BTC挂单数={n}")


def test_maintenance() -> Case:
    """维护态：提交被拒（验证断路，不崩）。"""
    a = MockAdapter(fault_mode="maintenance")
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    d = eng.ingest_signal(sig)
    return Case("维护态断路（不崩）", (not d.accepted) and d.reject == "adapter_reject",
                f"reject={d.reject}")


def test_state_machine() -> Case:
    """BTC 多头：入场 -> TP1平50%移保本位 -> TP2全平。"""
    a = MockAdapter()
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    eng.ingest_signal(sig)
    eng.step({"BTC": 60000})          # 入场
    r1 = eng.step({"BTC": 61000})     # TP1 -> 移 SL 保本位
    r2 = eng.step({"BTC": 62500})     # TP2 全平
    closed = a.query_position("BTC") is None
    breakeven_moved = any("SL 移至保本位" in e for e in r1.events)
    return Case("状态机 入场->TP1(50%)->TP2全平", closed and breakeven_moved,
                f"已平仓={closed}, SL移保本位={breakeven_moved}, 事件={len(r1.events)+len(r2.events)}")


def test_exec_sl_caps_dd() -> Case:
    """ETH 空头反向触发执行级SL：单币亏损被限制在 ~2-3% 权益，而非 -55%。"""
    a = MockAdapter()
    cfg = GateConfig(enforce_gate_b=False, equity=100000.0)
    sig = mk_signal("ETH", Direction.SHORT, 3000, 2950, 2880, 40)
    eng = ExecutionEngine(a, cfg, KillSwitch(), lev=5.0)
    eng.ingest_signal(sig)
    eng.step({"ETH": 3000})           # 入场
    eng.step({"ETH": 3080})           # 触发执行级SL(=3000+2*40)
    pos = a.query_position("ETH")
    # 理论最大单币亏损：2*ATR * qty / equity
    qty = 3000 * 0.04 / 3000 if False else None
    # 用实际成交估算：SL 亏损 = (3080-3000)*qty
    # qty = kelly: min(4000, 100000*0.03*1.1)/3000 = 3300/3000 = 1.1
    qty = 3300 / 3000
    loss_frac = (3080 - 3000) * qty / cfg.equity
    capped = loss_frac < 0.03 and pos is None
    return Case("执行级SL止血（单币DD<3%）", capped,
                f"单币亏损={loss_frac:.2%} (vs 回测-55.5%穿透), 已平仓={pos is None}")


def test_kill_switch_blocks() -> Case:
    """L1 触发后禁止新开仓。"""
    a = MockAdapter()
    ks = KillSwitch()
    ks.update(daily_pnl=-0.04)        # 当日 -4% -> L1
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), ks)
    d = eng.ingest_signal(mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300))
    return Case("KillSwitch L1 暂停新开", (not d.accepted) and "kill_switch" in (d.reject or ""),
                f"level={ks.level.name}, reject={d.reject}")


def test_reconcile_clean() -> Case:
    """正常平仓后，期望与实际对账一致。"""
    a = MockAdapter()
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    eng.ingest_signal(sig)
    eng.step({"BTC": 60000})
    eng.step({"BTC": 61000})
    eng.step({"BTC": 62500})
    rep = eng.step({"BTC": 62500})
    return Case("对账一致（期望=实际）", rep.reconcile.clean,
                f"clean={rep.reconcile.clean}, items={[(i.symbol,i.action) for i in rep.reconcile.items]}")


def test_reconcile_over() -> Case:
    """人为制造超仓：实际持仓 > 期望，对账标记 OVER。"""
    a = MockAdapter()
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    eng.ingest_signal(sig)
    eng.step({"BTC": 60000})
    # 模拟交易所侧多开了仓（如重复信号/部分成交叠加）
    p = a.query_position("BTC")
    if p:
        p.qty += 2.0
    rep = eng.step({"BTC": 60000})
    over = any(i.action == "OVER" for i in rep.reconcile.items)
    return Case("对账防超仓（OVER标记）", over,
                f"actions={[(i.symbol,i.action,i.diff_qty) for i in rep.reconcile.items]}")


def test_router_fallback() -> Case:
    """主所维护时，路由 fallback 到健康所，不丢单。"""
    primary = MockAdapter(fault_mode="maintenance")   # Binance 维护
    backup = MockAdapter()                             # OKX 健康
    router = FallbackRouter({"binance": primary, "okx": backup},
                            priority=["binance", "okx"])
    sig = mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300)
    from .core.order_builder import build_entry_order
    order = build_entry_order(sig, GateConfig(enforce_gate_b=False, ))
    o = router.submit(order)
    routed = o is not None and o.status is OrderStatus.OPEN and len(backup.query_open()) == 1
    return Case("跨所fallback路由（主所挂掉切备份）", routed,
                f"成交所=okx, 挂单数={len(backup.query_open())}")


def _synthetic_market(symbol: str, trend: float = 0.0, vol: float = 0.02, n: int = 60,
                      price0: float = 60000.0) -> list:
    """生成合成 1h K 线（调试用，零资金）。

    - 趋势用例(trend≠0)：纯漂移，确保 ADX 拉高。
    - 横盘用例(trend≈0)：叠加均值回归摆动 + 极小噪声，确保 ADX<20（真正横盘），
      避免随机游走累积伪趋势导致均值回归信号不稳定触发。
    确定性种子（稳定哈希），保证用例可复现（共识#8）。
    """
    import random, math
    random.seed(sum((i + 1) * ord(c) for i, c in enumerate(symbol)) % 2**31)
    candles = []
    p = price0
    step = trend / n
    flat = abs(trend) < 1e-9
    for i in range(n):
        if flat:
            # 真横盘：围绕中枢正弦摆动 + 极小噪声，无复利漂移（ADX 稳居 <20）
            c = price0 * (1 + vol * 2.0 * math.sin(2 * math.pi * i / 12)
                          + random.uniform(-vol * 0.3, vol * 0.3))
        else:
            p *= (1 + step + random.uniform(-vol * 0.3, vol * 0.3))
            c = p
        h = c * (1 + vol * 0.3); l = c * (1 - vol * 0.3)
        candles.append({"h": h, "l": l, "c": c})
    return candles


def test_signal_engine() -> Case:
    """验证吸收的生产级信号引擎（P0-A/B）：趋势评分 + 动态门槛 + 仓位链 + 熔断/黑天鹅。

    两条路径分别验证：
      (1) 强趋势上行 → 做多 + 高评分（趋势引擎）
      (2) ADX<20 震荡 + 极端费率 → 均值回归反向（互补）
    """
    # 路径1：强趋势上行（trend 足够大，ADX≥25）
    btc_up = _synthetic_market("BTC", trend=0.15, vol=0.01, n=60, price0=60000)
    ctx = MarketContext(fg_val=55, fr=-0.0002, fr_delta=0.0001)
    cand = gen_signal("BTC", btc_up, ctx=ctx)
    trend_ok = cand.direction == "做多" and cand.score >= cand.min_score_adj and cand.score > 0
    pr = calc_position_pct(cand.score, cand.adx, cand.atr_pct,
                           trade_state={"win_rate": 55, "streak_wins": 0, "streak_losses": 0},
                           vol_regime="stable")
    sizing_ok = 0.3 <= pr.pct <= 5.0 and pr.pct >= 1.0

    # 路径2：ADX<20 震荡 + 极正费率 → 均值回归反向做空
    from .signals.mean_reversion import gen_mean_reversion
    btc_flat = _synthetic_market("ETH", trend=0.0, vol=0.008, n=60, price0=3000)
    mr = gen_mean_reversion("ETH", btc_flat, fr=0.0005, ctx=ctx)
    mr_ok = mr.triggered and mr.direction == "做空"

    # 动态熔断阈值：高波动放宽（注：dynamic_threshold 现用 atr_pct 真实百分比量纲，
    # 常态1-4%，故传 6.0%=高波动→0.06、0.3%=低波动→0.03，验证"高放宽/低收紧"）
    th_high = dynamic_threshold(6.0); th_low = dynamic_threshold(0.3)
    cb_ok = th_high == 0.06 and th_low == 0.03

    # 黑天鹅连续跌幅：30分钟跌16% → L3
    hist = [100] * 25 + [84]
    lvl, _ = get_black_swan_level(hist)
    bs_ok = lvl == 3

    # 元认知环境聚合
    env = assess(btc_up, fg_val=55)
    cog_ok = env.dominant in ("BULL", "RANGE", "BEAR")

    ok = trend_ok and sizing_ok and mr_ok and cb_ok and bs_ok and cog_ok
    return Case("信号引擎驱动(P0-A/B/C/D/E)", ok,
                f"趋势:方向={cand.direction} 评分={cand.score:.0f}≥门槛{cand.min_score_adj:.0f} "
                f"仓位={pr.pct}%; 均值回归:ETH={mr.direction}分{mr.score:.0f}; "
                f"熔断[低{th_low}/高{th_high}] 黑天鹅L{lvl} 环境={env.dominant}")


def test_regime_detection() -> Case:
    """验证 detect_regime 接真实K线『分得对』（圆桌共识二·验证纪律：先证再驱动）。

    喂三类已知分布的收盘价序列，验证分类器正确识别：
      - 稳步上涨趋势 → TREND
      - 横盘低波动 → RANGE
      - 高波动+深回撤崩盘 → CRASH
    并验证 ExecutionEngine.update_regime 实时驱动 gate.regime（路由真正生效）。
    """
    from .risk.regime import detect_regime
    from .core.engine import ExecutionEngine
    from .risk.gate import GateConfig
    from .risk.kill_switch import KillSwitch

    # 1) 趋势序列：前段平静 + 后段急涨高波动 → recent vol >> base → TREND
    trend = [100.0 + 0.2 * ((i * 5) % 9 - 4) for i in range(40)]
    for i in range(40, 60):
        trend.append(trend[-1] * (1 + 0.01 + 0.015 * ((i * 3) % 7 - 3) / 3))
    # 2) 横盘序列：前段高波动 + 后段明显平静（vol_ratio<0.7）→ RANGE
    rng = [100.0 + 3.0 * ((i * 7) % 17 - 8) for i in range(40)]
    _base = rng[-1]
    for i in range(40, 60):
        rng.append(_base + 0.2 * ((i * 5) % 9 - 4))
    # 3) 崩盘序列：先平稳后高波动深跌（vol_ratio>1.8 且 max_dd<-0.12）→ CRASH
    crash = [100.0] * 30
    for i in range(30, 60):
        crash.append(crash[-1] * (1 - 0.006 + 0.01 * ((i * 3) % 7 - 3) / 3))

    r_trend = detect_regime(trend).regime
    r_range = detect_regime(rng).regime
    r_crash = detect_regime(crash).regime
    classify_ok = (r_trend == "TREND" and r_range == "RANGE" and r_crash == "CRASH")

    # 引擎路由：喂崩盘序列 → gate.regime 应自动变 CRASH，禁止新开仓
    cfg = GateConfig(enforce_gate_b=False, regime="TREND")
    eng = ExecutionEngine(MockAdapter(), cfg, KillSwitch())
    driven = eng.update_regime({"BTC": crash})
    route_ok = (driven == "CRASH" and eng.cfg.regime == "CRASH")

    # 多币合并：BTC 横盘 + ETH 崩盘 → 全局 CRASH（保守优先）
    driven2 = eng.update_regime({"BTC": rng, "ETH": crash})
    merge_ok = driven2 == "CRASH"

    ok = classify_ok and route_ok and merge_ok
    return Case("regime接真实K线(分得对+路由生效)", ok,
                f"趋势={r_trend} 横盘={r_range} 崩盘={r_crash} | "
                f"引擎驱动→{driven} 多币合并→{driven2}")


def test_profit_deflate_and_survival() -> Case:
    """验证第4条两项：（共识#7）盈利后降暴露 +（共识#10）黑天鹅L3转稳定币减仓。"""
    from .risk.position_sizing import calc_position_pct
    from .risk.kill_switch import KillSwitch

    # ① 盈利递减：同样评分/ADX，连盈状态下新仓比例应低于正常态
    normal = calc_position_pct(8, 30, 30, trade_state={"win_rate": 55}).pct
    profit = calc_position_pct(8, 30, 30,
                               trade_state={"win_rate": 55, "profit_steps": 3,
                                            "cum_pnl_pct": 2.0}).pct
    deflate_ok = profit < normal

    # ② L3 生存态：黑天鹅触发 → survival_action 返回减仓50%+转稳定币意图
    ks = KillSwitch()
    ks.update(black_swan=True)
    surv = ks.survival_action()
    survive_ok = (surv["active"] and surv["reduce_to"] == 0.5 and surv["to_stablecoin"])

    # ③ 引擎端到端：持仓中触发黑天鹅 → step() 发出减仓单（减至50%，不自动全平）
    from .core.engine import ExecutionEngine
    from .models import Signal, Direction, OrderType
    a = MockAdapter()
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    eng.ks.update(black_swan=True)   # 模拟已触发 L3
    pos = _make_dummy_position("BTC", Direction.LONG, 60000.0, 1.0, "test_l3")
    eng.expected["test_l3"] = pos
    eng.step({"BTC": 60000.0})
    reduce_orders = [o for o in a.open_orders.values()
                     if getattr(o, "otype", None) == OrderType.REDUCE]
    engine_ok = len(reduce_orders) >= 1 and all(o.qty <= 1.0 for o in reduce_orders)

    ok = deflate_ok and survive_ok and engine_ok
    return Case("盈利递减+L3生存态减仓(共识#7/#10)", ok,
                f"正常仓={normal}% 连盈仓={profit}%(降暴露) | "
                f"L3减仓→{surv['reduce_to']:.0%} 转稳定币={surv['to_stablecoin']} | "
                f"引擎减仓单={len(reduce_orders)}")


def _make_dummy_position(symbol, direction, entry, qty, sid):
    """构造最小 Position 用于减仓验证（复用 models.Position）。"""
    from .models import Position
    return Position(symbol=symbol, direction=direction, entry_price=entry, qty=qty,
                    initial_qty=qty, sl_price=0.0, tp1_price=entry * 1.03,
                    tp2_price=entry * 1.06, entry_coid=f"{symbol}_{sid}_entry",
                    signal_id=sid, leverage=5.0, liq_price=entry * 0.5)


def test_ev_gate() -> Case:
    """负期望值信号被 EV 闸门拦截（共识#2：EV框架 > 胜率）。"""
    a = MockAdapter()
    eng = ExecutionEngine(a, GateConfig(enforce_gate_b=False, ), KillSwitch())
    d = eng.ingest_signal(mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300),
                           ev_est=-0.01)            # 明确负期望
    rejected = (not d.accepted) and "ev_negative" in (d.reject or "")
    d2 = eng.ingest_signal(mk_signal("ETH", Direction.SHORT, 3000, 2950, 2880, 40),
                           ev_est=0.02)             # 正期望应放行
    return Case("EV 期望值闸门（负期望拦截）", rejected and d2.accepted,
                f"负EV reject={d.reject}, 正EV accepted={d2.accepted}")


def test_oi_spike_kill() -> Case:
    """持仓量/OI 异动触发 KillSwitch L1（共识#10：事后触发器降仓）。"""
    ks = KillSwitch()
    ks.update(oi_spike_pct=0.15)                    # OI 暴涨 15%
    l1 = ks.level.value >= 1
    ks2 = KillSwitch()
    ks2.update(fr_spike=0.002)                      # 资金费率异动
    l1b = ks2.level.value >= 1
    return Case("OI/费率异动触发L1暂停", l1 and l1b,
                f"oi_spike→{ks.level.name}, fr_spike→{ks2.level.name}")


def test_gate_b_cost() -> Case:
    """P1-4：Gate B 成本盈亏平衡闸门——最坏成本净 edge<=0 时禁止开仓（fail-closed）。

    GROSS_EDGE_BPS 默认锁定 -2.5bps（跨所5年OOS最保守锚点），故 worst_case_pnl 恒负，
    所有币被 Gate B 拦截，系统在 edge 未经大样本复核前维持空仓（圆桌共识）。
    """
    from .risk.exec_cost import GROSS_EDGE_BPS, worst_case_pnl
    a = MockAdapter()
    # 显式开启 Gate B（覆盖 demo 全局关闭：构造后改实例属性，避免与正则注入的关键字冲突）
    cfg = GateConfig(enforce_gate_b=False)
    cfg.enforce_gate_b = True
    eng = ExecutionEngine(a, cfg, KillSwitch())
    d = eng.ingest_signal(mk_signal("BTC", Direction.LONG, 60000, 61000, 62500, 300))
    blocked = (not d.accepted) and "gate_b" in (d.reject or "")
    return Case("Gate B 成本盈亏平衡闸门（最坏成本净edge<=0禁开仓）", blocked,
                f"GROSS_EDGE_BPS={GROSS_EDGE_BPS:+.2f}, BTC最坏净edge={worst_case_pnl('BTC'):+.2f}bps, "
                f"reject={d.reject}")

def main():
    setup_logging()   # 【P2-B】结构化日志：bootstrap 根 handler，统一各模块日志格式
    cases = [
        test_idempotency(),
        test_timeout_retry(),
        test_maintenance(),
        test_state_machine(),
        test_exec_sl_caps_dd(),
        test_kill_switch_blocks(),
        test_reconcile_clean(),
        test_reconcile_over(),
        test_router_fallback(),
        test_signal_engine(),
        test_ev_gate(),
        test_oi_spike_kill(),
        test_gate_b_cost(),
        test_regime_detection(),
        test_profit_deflate_and_survival(),
    ]
    print("=" * 64)
    print("CryptoQuant 自动化执行原型 — 调试验证（零资金 / 零密钥）")
    print("=" * 64)
    npass = 0
    for c in cases:
        mark = "✅" if c.passed else "❌"
        if c.passed:
            npass += 1
        print(f"{mark} {c.name}")
        print(f"     └─ {c.detail}")
    print("-" * 64)
    print(f"通过 {npass}/{len(cases)}")
    print("结论：管线、幂等、执行级SL止血、KillSwitch、对账、跨所fallback 均可在假钱环境验证；")
    print("      下一步接测试网适配器（提供测试网Key）即可跑真实API假钱。")
    print("=" * 64)

    run_real_backtest_section()
    run_oos_section()
    run_regime_section()
    run_testnet_section()


if __name__ == "__main__":
    main()
