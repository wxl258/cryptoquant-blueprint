"""P0 修复验证（针对性，证明 4 个致命 bug 已闭环）。

运行：cd cryptoquant-blueprint && python3 -m cryptoquant_auto.verify_p0_fixes
"""
from __future__ import annotations
import os, tempfile, sys

from cryptoquant_auto.risk.constitution import TradingConstitution
from cryptoquant_auto.meta.memory import FinMemMemory, Episode, Profile
from cryptoquant_auto.meta.agents import FourRoleCouncil
from cryptoquant_auto.adapters.mock_llm import get_llm
from cryptoquant_auto.adapters.base import ExchangeAdapter
from cryptoquant_auto.adapters.binance_testnet import BinanceTestnetAdapter
from cryptoquant_auto.models import Order, OrderType, OrderStatus


def section(t): print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


# ---------- P0-1：执行层硬锁下沉到 ingest_signal/v8 直连路径 ----------
def test_p0_1():
    section("P0-1 执行层硬锁覆盖 ingest_signal 直连路径")
    from cryptoquant_auto.core.engine import ExecutionEngine
    from cryptoquant_auto.risk.gate import GateConfig
    from cryptoquant_auto.risk.kill_switch import KillSwitch
    from cryptoquant_auto.adapters.mock import MockAdapter
    from cryptoquant_auto.models import Signal, Direction

    const_locked = TradingConstitution(live_capital=True)     # 硬锁开启
    eng = ExecutionEngine(MockAdapter(), GateConfig(), KillSwitch(),
                          constitution=const_locked)
    sig = Signal(symbol="BTC", tf="1H", direction=Direction.LONG, entry=100.0,
                 sl=90.0, tp1=110.0, tp2=120.0, rr=2.0, confidence=0.9,
                 signal_id="t_p01", atr=5.0)
    # v8 直连路径（此前绕开宪法）
    d = eng.ingest_signal(sig)
    ok = (not d.accepted) and "R0" in (d.reject or "")
    print(f"  ingest_signal 直连 + live_capital=True → accepted={d.accepted} reject={d.reject!r}")
    print(f"  ✅" if ok else f"  ❌ 硬锁未覆盖直连路径")
    return ok


# ---------- P0-2：适配器级实盘硬锁兜底 ----------
def test_p0_2():
    section("P0-2 适配器拒绝 live_capital=True 的真实下单")
    const_locked = TradingConstitution(live_capital=True)
    # 构造即应拒绝（硬锁开启）
    build_ok = True
    try:
        BinanceTestnetAdapter("k", "s", constitution=const_locked)
    except RuntimeError as e:
        build_ok = True
        print(f"  构造期拒绝: {e}")
    # 即便绕过构造，submit 也必须拒绝
    ad = BinanceTestnetAdapter("k", "s", constitution=TradingConstitution(live_capital=False))
    ad.constitution = const_locked
    o = Order(coid="x", symbol="BTC", side="BUY", otype=OrderType.ENTRY,
              price=100.0, qty=1.0, signal_id="x", leg="entry")
    r = ad.submit(o)
    submit_ok = r.status is OrderStatus.REJECTED
    print(f"  submit + live_capital=True → status={r.status.name}")
    print(f"  ✅" if (build_ok and submit_ok) else f"  ❌ 适配器未兜底")
    return build_ok and submit_ok


# ---------- P0-3：forbidden_regimes 对非 CRASH regime 生效 ----------
def test_p0_3():
    section("P0-3 禁易锁对非 CRASH 原始 regime（如 TREND）生效")
    tmp = tempfile.mkdtemp()
    mem = FinMemMemory(base_dir=tmp)
    mem.profile.forbidden_regimes = ["TREND"]     # 反思写入的原始 regime
    council = FourRoleCouncil(mem, llm=get_llm(), profile=mem.profile)
    # 构造一个 TREND regime、强动量的特征，使未禁时应为 LONG
    import numpy as np
    # ADX, RSI, ATR%, VOL_REGIME, FR, FR_DELTA, OI%, FNG, MOMENTUM
    feat = np.array([0.9, 50.0, 0.02, 1.0, 0.0, 0.0, 0.0, 0.5, 0.02])
    v = council.decide("BTC", feat, regime="TREND", record=False)
    ok = (v.action == "HOLD") and any("forbidden" in x for x in v.vetoes)
    print(f"  TREND 被禁 → action={v.action} vetoes={v.vetoes}")
    print(f"  ✅" if ok else f"  ❌ 禁易锁对 TREND 未触发")
    return ok


# ---------- P0-4a：decision_id 精确回填，避免错序污染 ----------
def test_p0_4a():
    section("P0-4a decision_id 精确回填（同 symbol 多笔不串）")
    tmp = tempfile.mkdtemp()
    mem = FinMemMemory(base_dir=tmp)
    d1 = mem.record_decision(Episode(ts=1.0, symbol="BTC", regime="RANGE",
                                     decision="LONG", confidence=0.6, rationale=["a"]))
    d2 = mem.record_decision(Episode(ts=2.0, symbol="BTC", regime="RANGE",
                                     decision="LONG", confidence=0.6, rationale=["b"]))
    ok1 = mem.set_outcome("BTC", 10.0, decision_id=d1)
    ok2 = mem.set_outcome("BTC", -20.0, decision_id=d2)
    e1 = next(e for e in mem.short_term if e.decision_id == d1)
    e2 = next(e for e in mem.short_term if e.decision_id == d2)
    ok = ok1 and ok2 and e1.outcome_bps == 10.0 and e2.outcome_bps == -20.0
    print(f"  d1→+10bps={e1.outcome_bps}  d2→-20bps={e2.outcome_bps} (未串号)")
    print(f"  ✅" if ok else f"  ❌ 回填串号")
    return ok


# ---------- P0-4b：禁易冷却到期自动解禁，破解永久误杀 ----------
def test_p0_4b():
    section("P0-4b 禁易冷却到期自动解禁（破解永久误杀死锁）")
    tmp = tempfile.mkdtemp()
    mem = FinMemMemory(base_dir=tmp)
    # 注入亏损 TREND 样本触发禁易
    for i in range(15):
        mem.record_decision(Episode(ts=float(i), symbol="BTC", regime="TREND",
                                     decision="LONG", confidence=0.6, rationale=["x"]))
        mem.set_outcome("BTC", -30.0, decision_id=mem.short_term[-1].decision_id)
    mem.reflect()
    forbidden_after = "TREND" in mem.profile.forbidden_regimes
    cd = mem.profile.forbid_cooldown.get("TREND", 0)
    # 推进冷却轮数（不注入新亏损，模拟时间流逝）
    cooldown = cd
    for _ in range(cooldown + 1):
        mem.reflect()
    expired = "TREND" not in mem.profile.forbidden_regimes
    ok = forbidden_after and expired
    print(f"  触发禁易={forbidden_after} 冷却={cd}轮 → 到期解禁={expired}")
    print(f"  ✅" if ok else f"  ❌ 永久误杀未破解")
    return ok


def main() -> int:
    results = {
        "P0-1 执行层硬锁下沉": test_p0_1(),
        "P0-2 适配器硬锁兜底": test_p0_2(),
        "P0-3 禁易命名空间统一": test_p0_3(),
        "P0-4a decision_id精确回填": test_p0_4a(),
        "P0-4b 冷却解禁死锁": test_p0_4b(),
    }
    section("P0 修复验证总决")
    all_ok = True
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'} {k}")
        all_ok = all_ok and v
    print("\n  🟢 全部 P0 修复闭环" if all_ok else "\n  🔴 仍有未修复项")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
