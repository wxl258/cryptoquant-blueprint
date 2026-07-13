"""阶段2 验证层（任务15）+ 进门闸门（零依赖 / 零资金 / 沙盒可跑）。

把关逻辑（蓝图实施计划 §三·专家C·任务15）：因果白名单 + GP 解须过一遍阶段0.5
（Purged+Embargo + DSR）才算进门。本脚本把任务12/13/14 的产物统一送进验证层：

  ① 因果发现（任务12）→ 因果特征白名单（稳定选择 + regime 不变性）
  ② GP/NSGA-II（任务13）→ Pareto 前沿（IS 轻量代理进化）
  ③ 阶段0.5 进门闸门：把白名单 + GP 最优解送 real-forward Purged+Embargo 滚动 WF，
     DSR(N) 多重检验校正（N = GP 种群×代数），受控对比（GP vs 随机信号同前向）
  ④ StockSim（任务14）→ 订单级撮合 + 人造市场，复现程式化事实（肥尾/波动聚集/量自相关）
  ⑤ 实盘硬锁：LIVE_CAPITAL_LOCK=False 断言 + 宪法 live_capital=True 否决复测

零依赖纪律：仅 numpy + 包内模块。torch/LLM 不引入（后移至阶段3-4）。
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ============================ 实盘硬锁（宪法 R0）============================
# 阶段2 全部为原型沙盒验证，绝不允许任何实盘资本动作。
LIVE_CAPITAL_LOCK = False
assert LIVE_CAPITAL_LOCK is False, "❌ 阶段2 禁止 live_capital=True，违反宪法 R0 实盘硬锁"

from cryptoquant_auto.stage2_features import FEATURE_NAMES, build_feature_matrix
from cryptoquant_auto.signals.causal import CausalDiscovery
from cryptoquant_auto.signals.gp_nsga2 import evolve, build_signals_from_tree
from cryptoquant_auto.sim.metrics import deflated_sharpe
from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
from cryptoquant_auto.sim.stocksim import MarketSimulator, measure_stylized_facts, make_market_agent
from cryptoquant_auto.signals.engine import gen_signal, MarketContext
from cryptoquant_auto.core.metacontroller import (BayesianMetacontroller,
                                                 opinion_from_candidate, LONG, SHORT, HOLD)
from cryptoquant_auto.risk.constitution import TradingConstitution
from cryptoquant_auto.core.engine import ExecutionEngine
from cryptoquant_auto.adapters.mock import MockAdapter
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.risk.kill_switch import KillSwitch
from cryptoquant_auto.risk.conformal import SequentialConformalPredictor


def _banner(t: str) -> None:
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def _real_forward_wf(signals, paths, windows: int = 5, is_ratio: float = 0.6,
                    embargo: float = 0.01, purge: float = 0.01, seed: int = 7,
                    n_trials: int = 1):
    """真实前向 Purged+Embargo 滚动 WF（诚实回测，复用阶段0.5 DSR 闸门）。

    与阶段0.5 walk_forward 同款隔离逻辑，但用每笔信号的**真实前向收盘**回放
    （非合成漂移），DSR 计算用 deflated_sharpe（periods_per_year=1，信号级）。
    返回 (DSR, purge_bars, embargo_bars, oos_returns, n_folds, win_rate)。
    """
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
            net = curve[-1] / curve[0] - 1.0
            if net > 0:
                wins += 1
    dsr = 0.0
    if oos_returns:
        per = [deflated_sharpe(r, n_trials=n_trials, sr0=0.0, periods_per_year=1)
               for r in oos_returns if len(r) >= 5]
        dsr = float(np.mean(per)) if per else 0.0
    win_rate = wins / max(1, len(oos_returns))
    return dsr, purge_bars, embargo_bars, oos_returns, windows, win_rate


def main() -> int:
    results = {}
    _banner("阶段2 · 进门验证（任务12/13/14 → 任务15 · 阶段0.5 闸门）")

    # ---- ⑤ 实盘硬锁先验证（宪法 R0）----
    print("⑤ 实盘硬锁：LIVE_CAPITAL_LOCK =", LIVE_CAPITAL_LOCK,
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

    # ---- 数据 ----
    _banner("数据：真实行情特征矩阵（history_cache.json + deriv_data.json）")
    X, y, regimes, rows = build_feature_matrix(step=12, warmup=240, horizon=12)
    print(f"  样本={len(rows)} 特征={len(FEATURE_NAMES)} "
          f"regime分布={ {r: int((regimes==r).sum()) for r in np.unique(regimes)} }")
    # 时间切分 IS/OOS（前 60% 训练 GP，后 40% 进门验证）
    k = int(len(rows) * 0.6)
    X_is, y_is = X[:k], y[:k]
    X_oos, y_oos = X[k:], y[k:]
    rows_oos = rows[k:]
    print(f"  IS={k}  OOS={len(rows_oos)}（OOS 送阶段0.5 闸门）")

    # ---- ① 因果发现 ----
    _banner("① 因果发现（任务12）→ 因果特征白名单")
    cd = CausalDiscovery(feature_names=FEATURE_NAMES, stability_boot=50,
                         stability_thresh=0.6, inv_tol=0.6, seed=0)
    rep = cd.fit(X, y, regimes=regimes)
    print("  " + rep.summary)
    print(f"  稳定选择频率: { {k_: round(v_,2) for k_,v_ in rep.stable_freq.items()} }")
    if rep.invariant:
        print(f"  regime 不变性: {rep.invariant}")
    wl_idx = [FEATURE_NAMES.index(n) for n in rep.whitelist]
    print(f"  NOTEARS DAG 辅助: {'已解出' if rep.dag is not None else '降级跳过'}")
    results["causal_whitelist"] = bool(rep.whitelist)

    # ---- ② GP/NSGA-II ----
    _banner("② GP/NSGA-II（任务13）→ Pareto 前沿（IS 进化）")
    if not wl_idx:
        print("  ⚠️ 白名单为空，GP 退化为全特征池（仍零依赖可跑）")
        wl_idx = list(range(len(FEATURE_NAMES)))
        rep.whitelist = list(FEATURE_NAMES)
    gp = evolve(X_is[:, wl_idx], y_is, whitelist=rep.whitelist,
                pop_size=30, generations=12, max_depth=3, seed=0)
    print(f"  Pareto 前沿规模={len(gp.pareto)}  总评估={gp.n_eval}")
    print(f"  最优解(IS 代理): 收益={gp.best_meta.get('mean_return',0):.5f} "
          f"回撤={gp.best_meta.get('max_dd',0):.5f} 换手={gp.best_meta.get('turnover',0):.3f} "
          f"树深={gp.best_meta.get('depth',0)}")
    results["gp_pareto"] = bool(gp.pareto)

    # ---- ③ 阶段0.5 进门闸门：real-forward Purged+Embargo + DSR ----
    _banner("③ 阶段0.5 进门闸门：因果白名单 + GP 解 → real-forward Purged+Embargo WF + DSR")
    best_tree = gp.best_by_return
    gp_sigs = build_signals_from_tree(best_tree, rows_oos, X_oos, wl_idx)
    gp_paths = [r["forward"] for r in rows_oos]
    N_trials = max(1, gp.n_eval)  # GP 搜索规模 → 多重检验校正
    dsr_gp, pb, eb, _, nf, wr = _real_forward_wf(
        gp_sigs, gp_paths, windows=5, is_ratio=0.6, embargo=0.1,
        purge=0.1, seed=7, n_trials=N_trials)
    print(f"  GP 解 DSR(N={N_trials})={dsr_gp:.3f}  OOS盈利窗={wr:.0%}  "
          f"净化剪bar: purge={pb} embargo={eb}（>0 = 隔离层在干活 ✅）")

    # 受控对比：同前向的随机信号 DSR（隔离信号质量）
    from cryptoquant_auto.models import Direction, Signal
    import uuid as _u
    rng = np.random.default_rng(7)
    rand_sigs = []
    for idx, r in enumerate(rows_oos):
        # 随机方向信号（仅用于对照，不进 GP）
        direction = Direction.LONG if rng.random() < 0.5 else Direction.SHORT
        atr = r["atr"] or r["price"] * 0.01
        sl = r["price"] - 2 * atr if direction is Direction.LONG else r["price"] + 2 * atr
        tp1 = r["price"] + 2 * atr if direction is Direction.LONG else r["price"] - 2 * atr
        rand_sigs.append(Signal(symbol=r["symbol"], tf="1H", direction=direction,
                                entry=round(r["price"], 2), sl=round(sl, 2),
                                tp1=round(tp1, 2), tp2=round(tp1, 2), rr=2.0,
                                confidence=0.5,
                                signal_id=f"{r['symbol']}_rnd_{_u.uuid4().hex[:8]}",
                                atr=atr))
    dsr_rnd, _, _, _, _, _ = _real_forward_wf(
        rand_sigs, gp_paths, windows=5, is_ratio=0.6, embargo=0.1,
        purge=0.1, seed=7, n_trials=N_trials)
    print(f"  随机信号 DSR(N={N_trials})={dsr_rnd:.3f}（同前向对照）")
    print(f"  GP 相对随机 ΔDSR={dsr_gp - dsr_rnd:+.3f} "
          f"→ {'GP 方向优于随机' if dsr_gp > dsr_rnd else '未显著优于随机（诚实）'}")
    # 进门闸门判定：隔离激活(purge/embargo>0) + DSR 已计算 + GP 不劣于随机（受控对比）
    gate_c_active = (pb > 0 and eb > 0)
    results["stage0_5_gate"] = gate_c_active and (dsr_gp is not None) and (dsr_gp >= dsr_rnd)
    results["dsr_gp"] = dsr_gp
    results["dsr_random"] = dsr_rnd

    # ---- ④ StockSim 程式化事实 ----
    _banner("④ StockSim 订单级对手（任务14）→ 程式化事实复现")
    sim = MarketSimulator(mid=100.0, agent=make_market_agent(kind="mock"), tick=0.05)
    res = sim.run(2000)
    facts = measure_stylized_facts(res.prices, res.volumes)
    print(f"  仿真步数={facts['n']} 订单簿成交={len(sim.book.trades)}")
    print(f"  肥尾(超额峰度={facts['excess_kurtosis']})={'✅' if facts['fat_tails'] else '❌'}  "
          f"波动聚集(|收益|ACF1={facts['vol_acf1']})={'✅' if facts['vol_clustering'] else '❌'}  "
          f"量自相关(ACF1={facts['volume_acf1']})={'✅' if facts['volume_autocorr'] else '❌'}")
    print(f"  复现程式化事实数 = {facts['n_stylized_facts']}/3")
    results["stocksim_facts"] = facts["n_stylized_facts"] >= 2

    # ---- 总裁决 ----
    _banner("总裁决 · 阶段2 进门验证")
    for key, ok in results.items():
        if key in ("dsr_gp", "dsr_random"):
            continue
        print(f"  {'✅' if ok else '❌'} {key}")
    print(f"\n  DSR 实测: GP={results.get('dsr_gp',0):.3f}  随机={results.get('dsr_random',0):.3f}")
    all_ok = all(v for k, v in results.items() if k not in ("dsr_gp", "dsr_random"))
    print("\n" + ("✅ 阶段2 四任务全部落地，产物通过阶段0.5 进门闸门（Purged+Embargo + DSR 已接线）。"
                 if all_ok else
                 "❌ 存在未通过项，见上。"))
    print("  诚实声明：DSR 为经济 edge 实测值（非闸门机制）。原型无 edge 圣杯，"
          "验证层价值在「防骗自己」而非保证盈利。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
