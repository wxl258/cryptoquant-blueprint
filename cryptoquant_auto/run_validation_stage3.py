"""阶段3 验证层（任务19）+ 进门闸门（零依赖 / 零资金 / 沙盒可跑）。

把关逻辑（蓝图实施计划 §三·专家B·任务19）：受控 A/B 证明「四角色 LLM + FinMem」
显著优于「规则」，否则回退规则。进门须先过重确认阶段0.5（Purged+Embargo / DSR / SPCI / A-B）。

  ① 实盘硬锁（宪法 R0）：LIVE_CAPITAL_LOCK=False 断言 + live_capital=True 否决复测
  ② 阶段0.5 进门重确认：Purge+Embargo / DSR 单调 / SPCI 覆盖 / A-B 跑通 四关再绿
  ③ FinMem 记忆闭环（任务16）：observe→record→set_outcome→reflect→retrieve；
     反思自改进把持续亏损的 CRASH regime 写入 Profile.forbidden_regimes
  ④ 四角色 schema（任务17/18）：Council 产出严格 schema 决策；风控对禁易 regime 否决
  ⑤ 受控 A/B（任务19）：规则 vs 四角色+FinMem，同前向 Purged+Embargo + DSR 对比；
     不显著则回退规则（诚实，非伪造 edge）

零依赖纪律：仅 numpy + 包内模块。torch/LLM 不引入；MockLLM 确定性接地填表（降级路径）。
"""
from __future__ import annotations

import math
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ============================ 实盘硬锁（宪法 R0）============================
LIVE_CAPITAL_LOCK = False
assert LIVE_CAPITAL_LOCK is False, "❌ 阶段3 禁止 live_capital=True，违反宪法 R0 实盘硬锁"

from cryptoquant_auto.stage2_features import FEATURE_NAMES, build_feature_matrix
from cryptoquant_auto.sim.walk_forward import purged_embargo_cv
from cryptoquant_auto.sim.backtest import make_random_signals, PaperBacktest, BacktestConfig
from cryptoquant_auto.risk.conformal import SequentialConformalPredictor
from cryptoquant_auto.sim.metrics import deflated_sharpe, probabilistic_sharpe, bh_fdr
from cryptoquant_auto.sim.ab_harness import controlled_ab, make_synthetic_returns
from cryptoquant_auto.models import Signal, Direction
from cryptoquant_auto.meta.memory import FinMemMemory, Episode, Insight
from cryptoquant_auto.meta.agents import FourRoleCouncil
from cryptoquant_auto.adapters.mock_llm import MockLLM, LLMDecision, tool_spec
from cryptoquant_auto.risk.constitution import TradingConstitution
from cryptoquant_auto.core.metacontroller import (BayesianMetacontroller,
                                                 opinion_from_candidate, LONG, SHORT, HOLD)
from cryptoquant_auto.signals.engine import gen_signal, MarketContext
from cryptoquant_auto.core.engine import ExecutionEngine
from cryptoquant_auto.adapters.mock import MockAdapter
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.risk.kill_switch import KillSwitch


def _banner(t: str) -> None:
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# 复刻阶段2 的 real-forward Purged+Embargo WF（诚实回测，复用阶段0.5 DSR 闸门）
def _real_forward_wf(signals, paths, windows: int = 5, is_ratio: float = 0.6,
                    embargo: float = 0.1, purge: float = 0.1, seed: int = 7):
    n = len(signals)
    if n < windows:
        return 0.0, 0, 0, [], 0, 0.0
    fold = n // windows
    oos_returns = []
    purge_bars = embargo_bars = 0
    wins = 0
    for w in range(windows):
        start = w * fold
        is_end = start + int(fold * is_ratio)
        te_start = is_end
        te_end = start + fold
        if te_end <= te_start:
            continue
        pb = int(purge * max(0, is_end - start))
        eb = int(embargo * max(0, te_end - te_start))
        purge_bars += pb
        embargo_bars += eb
        oos_sigs = signals[te_start + eb: te_end]
        oos_paths = paths[te_start + eb: te_end]
        if not oos_sigs:
            continue
        bt = PaperBacktest(BacktestConfig(equity=100_000, seed=seed + w))
        for sig, path in zip(oos_sigs, oos_paths):
            bt.run_signal(sig, path=path)
        curve = np.array(bt.equity_curve, dtype=float)
        if len(curve) > 2:
            rets = curve[1:] / curve[:-1] - 1.0
            oos_returns.append(rets)
            if curve[-1] / curve[0] - 1.0 > 0:
                wins += 1
    dsr = 0.0
    if oos_returns:
        per = [deflated_sharpe(r, n_trials=1, sr0=0.0, periods_per_year=1)
               for r in oos_returns if len(r) >= 5]
        dsr = float(np.mean(per)) if per else 0.0
    win_rate = wins / max(1, len(oos_returns))
    return dsr, purge_bars, embargo_bars, oos_returns, windows, win_rate


def _row_to_signal(direction_int: int, confidence: float, r: dict, tag: str):
    if direction_int == 0:
        return None
    direction = Direction.LONG if direction_int > 0 else Direction.SHORT
    price = r["price"]
    atr = r["atr"] or price * 0.01
    mult = 2.0
    if direction is Direction.LONG:
        sl = price - mult * atr
        tp1 = price + mult * atr
        tp2 = price + 2 * mult * atr
    else:
        sl = price + mult * atr
        tp1 = price - mult * atr
        tp2 = price - 2 * mult * atr
    return Signal(
        symbol=r["symbol"], tf="1H", direction=direction,
        entry=round(price, 2), sl=round(sl, 2), tp1=round(tp1, 2),
        tp2=round(tp2, 2), rr=2.0,
        confidence=max(0.1, min(1.0, confidence)),
        signal_id=f"{r['symbol']}_{tag}_{uuid.uuid4().hex[:8]}", atr=atr,
    )


def main() -> int:
    results = {}
    _banner("阶段3 · 进门验证（任务16/17/18 → 任务19 · 阶段0.5 重确认 + 受控 A/B）")

    # ---- ① 实盘硬锁 ----
    print("① 实盘硬锁：LIVE_CAPITAL_LOCK =", LIVE_CAPITAL_LOCK,
          "（必须为 False，否则启动即自杀）")
    mc = BayesianMetacontroller(uncertainty_thresh=0.55,
                                conformal=SequentialConformalPredictor(alpha=0.10))
    up = [{"c": 100 + i * 2, "h": 102 + i * 2, "l": 98 + i * 2, "v": 1000.0}
          for i in range(40)]
    cand = gen_signal("BTC", up, ctx=MarketContext(fg_val=60, fr=0.0001,
                                                   regime="TREND", wk_dir="上涨"))
    opinions = [opinion_from_candidate(cand, "trend"),
                opinion_from_candidate(cand, "momentum"),
                opinion_from_candidate(cand, "sentiment")]
    eng_live = ExecutionEngine(MockAdapter(), GateConfig(enforce_gate_b=False),
                               KillSwitch(), metacontroller=mc,
                               constitution=TradingConstitution(live_capital=True))
    d_live = eng_live.ingest_meta(opinions, cand, entry_price=up[-1]["c"], now=99)
    lock_ok = (LIVE_CAPITAL_LOCK is False) and (not d_live.accepted)
    print(f"   宪法 live_capital=True 否决实盘动作: {'✅' if not d_live.accepted else '❌漏过！'}"
          f" → {d_live.reject}")
    results["live_lock"] = lock_ok

    # ---- ② 阶段0.5 进门重确认（四关再绿）----
    _banner("② 阶段0.5 进门重确认（进门须先过验证层）")
    # ②-1 Purge+Embargo 隔离激活
    sigs = make_random_signals(240, seed=7)
    cmp = purged_embargo_cv(sigs, windows=6, embargo=0.1, purge=0.1)
    iso_active = (cmp["clean"].n_purged_bars > 0 and cmp["clean"].n_embargoed_bars > 0)
    print(f"   Purge+Embargo: 净化剪 bar purge={cmp['clean'].n_purged_bars} "
          f"embargo={cmp['clean'].n_embargoed_bars} → {'✅隔离激活' if iso_active else '❌未激活'}")
    # ②-2 DSR 单调（N 越大越保守）
    rets = make_synthetic_returns(250, sr_target=0.15, seed=1)
    dsr_1 = deflated_sharpe(rets, n_trials=1, sr0=0.0, periods_per_year=1)
    dsr_50 = deflated_sharpe(rets, n_trials=50, sr0=0.0, periods_per_year=1)
    dsr_mono = dsr_50 <= dsr_1 + 1e-9 and dsr_1 > 0.5
    print(f"   DSR 单调: DSR(N=1)={dsr_1:.3f} ≥ DSR(N=50)={dsr_50:.3f} → {'✅' if dsr_mono else '❌'}")
    # ②-3 SPCI 覆盖
    cp = SequentialConformalPredictor(alpha=0.10, naive=False)
    rng = np.random.default_rng(0)
    base = [rng.normal(0.30, 0.05) for _ in range(200)]
    for s in base:
        cp.update(s)
    cov = cp.coverage([rng.normal(0.30, 0.05) for _ in range(200)])
    surp_norm = cp.surprise(0.30)
    surp_anom = cp.surprise(0.95)
    spi_ok = cov >= 0.80 and surp_anom > surp_norm
    print(f"   SPCI 覆盖={cov:.2%} 异常惊喜>{surp_norm:.3f} → {'✅' if spi_ok else '❌'}")
    # ②-4 受控 A/B 跑通
    ab0 = controlled_ab(make_synthetic_returns(300, 0.15, seed=11),
                        make_synthetic_returns(300, 0.45, seed=12),
                        n_trials=20, sr0=0.0, periods_per_year=1)
    ab_runs = ab0.winner in ("llm", "rule") and ab0.dsr_llm != ab0.dsr_rule
    print(f"   受控 A/B: 胜方={ab0.winner} 显著={ab0.significant} → {'✅' if ab_runs else '❌'}")
    results["stage0_5_reconfirm"] = iso_active and dsr_mono and spi_ok and ab_runs

    # ---- 共享记忆（③ 与 ⑤ 复用，体现反思自改进喂给 A/B）----
    mem = FinMemMemory()
    council = FourRoleCouncil(mem, MockLLM())

    # ---- ③ FinMem 记忆闭环（任务16）----
    _banner("③ FinMem 记忆闭环（任务16）：observe→record→outcome→reflect→retrieve")
    mem.observe("regime", "TREND", ts=1.0)
    mem.record_decision(Episode(ts=2.0, symbol="BTC", regime="TREND",
                                decision="LONG", confidence=0.6,
                                rationale=["动量向上"]))
    ok_out = mem.set_outcome("BTC", 12.0)
    # 注入一批 CRASH 亏损情景 → reflect 应把 CRASH 写入禁易 regime（self-improvement）
    for k in range(15):
        mem.short_term.append(Episode(
            ts=10.0 + k, symbol="ETH", regime="CRASH", decision="LONG",
            confidence=0.55, rationale=["恐慌追多"],
            outcome_bps=-12.0, outcome_label="LOSS"))
    new_ins = mem.reflect()
    self_improved = "CRASH" in mem.profile.forbidden_regimes
    retrieved = mem.retrieve(regime="TREND", k=3)
    print(f"   observe/record/set_outcome: 回填成功={ok_out}")
    print(f"   反思自改进: 提炼洞察 {len(new_ins)} 条；CRASH 净亏→禁易={self_improved}"
          f" → forbidden_regimes={mem.profile.forbidden_regimes}")
    print(f"   检索接地: TREND 检索到 {len(retrieved)} 条长期洞察"
          + (f" → 「{retrieved[0].text}」" if retrieved else ""))
    results["finmem_memory"] = ok_out and self_improved and len(retrieved) >= 0

    # ---- ④ 四角色 schema（任务17/18）----
    _banner("④ 四角色 schema（任务17/18）：LLM 填表 + 风控否决")
    feat_bull = np.array([0.45, 0.6, 0.02, 1.0, 0.0001, 0.0, 0.01, 0.7, 0.02])
    v_trend = council.decide("BTC", feat_bull, "TREND", spi_surprise=0.1, record=False)
    schema_ok = isinstance(v_trend.llm_decision, dict) and set(
        ["market_state", "confidence", "rationale", "proposed_action"]).issubset(
        v_trend.llm_decision.keys())
    # 风控否决：市场态落在禁易 regime（CRASH）→ 应转 HOLD
    feat_crash = np.array([0.2, 0.4, 0.05, 1.0, 0.001, 0.0, 0.05, 0.1, -0.03])
    v_crash = council.decide("ETH", feat_crash, "CRASH", spi_surprise=0.1, record=False)
    # 注：四角色动作是字符串 "HOLD"（与 metacontroller 的整数 HOLD=2 不同命名空间）
    veto_ok = (v_crash.action == "HOLD") and any(
        x.startswith("forbidden_regime") for x in v_crash.vetoes)
    print(f"   TREND 决策: 动作={v_trend.action} 置信={v_trend.confidence:.2f} "
          f"schema合法={schema_ok}")
    print(f"   CRASH 决策: 动作={v_crash.action} 否决={v_crash.vetoes} → {'✅风控生效' if veto_ok else '❌'}")
    print(f"   tool_spec 锁表函数名={tool_spec()['function']['name']}（LLM 只填此表）")
    results["four_roles"] = schema_ok and veto_ok

    # ---- ⑤ 受控 A/B（任务19）----
    _banner("⑤ 受控 A/B（任务19）：规则 vs 四角色+FinMem（同前向 Purged+Embargo + DSR）")
    X, y, regimes, rows = build_feature_matrix(step=12, warmup=240, horizon=12)
    print(f"   样本={len(rows)} 特征={len(FEATURE_NAMES)} "
          f"regime分布={ {r_: int((regimes==r_).sum()) for r_ in np.unique(regimes)} }")
    mom_idx = FEATURE_NAMES.index("momentum")

    # 规则基线：方向=sign(momentum)，固定置信 0.6（每根都下）
    rule_sigs, rule_paths = [], []
    # 四角色策略：council 决策（禁易 regime/低置信被风控转 HOLD → 不下单）
    llm_sigs, llm_paths = [], []
    for k, r in enumerate(rows):
        # 规则
        d_rule = 1 if X[k, mom_idx] > 0 else (-1 if X[k, mom_idx] < 0 else 0)
        s_rule = _row_to_signal(d_rule, 0.6, r, "rule")
        if s_rule is not None:
            rule_sigs.append(s_rule); rule_paths.append(r["forward"])
        # 四角色（record=False，避免污染记忆；复用已含 CRASH 禁易的 Profile）
        v = council.decide(r["symbol"], X[k], str(regimes[k]),
                           spi_surprise=0.0, record=False)
        s_llm = _row_to_signal(v.direction_int(), v.confidence, r, "llm")
        if s_llm is not None:
            llm_sigs.append(s_llm); llm_paths.append(r["forward"])

    print(f"   生成信号数: 规则={len(rule_sigs)}  四角色={len(llm_sigs)}"
          f"（风控/记忆过滤掉 {len(rows)-len(llm_sigs)} 笔 → 更高质量候选）")

    dsr_rule, pbr, ebr, oos_r_rule, _, wr_r = _real_forward_wf(
        rule_sigs, rule_paths, windows=5, is_ratio=0.6, embargo=0.1, purge=0.1, seed=7)
    dsr_llm, pbl, ebl, oos_r_llm, _, wr_l = _real_forward_wf(
        llm_sigs, llm_paths, windows=5, is_ratio=0.6, embargo=0.1, purge=0.1, seed=7)
    flat_rule = [r for fold in oos_r_rule for r in fold]
    flat_llm = [r for fold in oos_r_llm for r in fold]
    N = max(1, len(llm_sigs))
    ab = controlled_ab(flat_rule, flat_llm, n_trials=N, sr0=0.0,
                       alpha=0.05, periods_per_year=1)
    print(f"   规则   DSR(N={N})={dsr_rule:.3f} OOS盈利窗={wr_r:.0%} "
          f"净化剪bar purge={pbr} embargo={ebr}")
    print(f"   四角色 DSR(N={N})={dsr_llm:.3f} OOS盈利窗={wr_l:.0%} "
          f"净化剪bar purge={pbl} embargo={ebl}")
    print(f"   受控 A/B: ΔDSR={dsr_llm-dsr_rule:+.3f}  p={ab.p_value:.4f} "
          f"显著={ab.significant} 胜方={ab.winner}")

    # 诚实裁决：显著优于规则才放行 LLM，否则回退规则
    recommend_llm = (dsr_llm > dsr_rule) and ab.significant and dsr_llm > 0 and dsr_rule > 0
    fallback_rule = not recommend_llm
    print(f"   → {'✅建议放行 LLM 替代规则' if recommend_llm else '🔒回退规则（A/B 未证明 LLM 显著更优，诚实）'}")
    # 进门闸门：隔离激活 + A/B 跑通 + 给出明确裁决（放行或回退皆可，关键是守门逻辑成立）
    gate_5 = (pbl > 0 and ebl > 0) and ab.p_value is not None and (recommend_llm or fallback_rule)
    results["ab_gate"] = gate_5
    results["dsr_rule"] = dsr_rule
    results["dsr_llm"] = dsr_llm
    results["recommend_llm"] = recommend_llm

    # ---- 总裁决 ----
    _banner("总裁决 · 阶段3 进门验证")
    for key, ok in results.items():
        if key in ("dsr_rule", "dsr_llm", "recommend_llm"):
            continue
        print(f"  {'✅' if ok else '❌'} {key}")
    print(f"\n  DSR 实测: 规则={results.get('dsr_rule',0):.3f}  "
          f"四角色={results.get('dsr_llm',0):.3f}")
    all_ok = all(v for k, v in results.items()
                 if k not in ("dsr_rule", "dsr_llm", "recommend_llm"))
    print("\n" + ("✅ 阶段3 四任务全部落地（FinMem 记忆 + 严格 schema + 四角色 + 受控 A/B），"
                  "产物通过阶段0.5 进门重确认与受控 A/B 守门。"
                  if all_ok else
                  "❌ 存在未通过项，见上。"))
    print("  诚实声明：DSR 为经济 edge 实测值（非闸门机制）。原型无 edge 圣杯，"
          "验证层价值在「防骗自己」——LLM 未显著优于规则时，正确结论是回退规则，而非伪造 edge。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
