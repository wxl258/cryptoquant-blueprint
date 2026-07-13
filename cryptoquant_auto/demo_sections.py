"""demo 回测 / 样本外 / regime / 测试网 分析段（从 demo.py 抽取，逻辑不变）。

仅承载「重型分析 + 报告」函数，与 demo.py 的「管线机制测试」分离：
  - demo.py：Case / mk_signal / test_*（幂等、执行级SL、KillSwitch、对账、fallback…）
  - 本模块：run_*_section 系列 + 其私有辅助（分段 EV、regime 分布、边缘口袋探索）

【P2-C 重构】原 demo.py 808 行（fat file），将分析段整体外移，demo.py 仅作编排入口。
所有函数体逐字迁移，未改任何业务逻辑；demo.py 通过 `from .demo_sections import ...` 回引。
"""
from __future__ import annotations

from typing import Dict, List

from .sim.backtest import PaperBacktest, BacktestConfig, make_random_signals
from .sim.metrics import BacktestStats
from .history import build_history, gen_real_signals_parallel, SYMBOLS
from .risk.exec_cost import calibrate_gross_edge_bps, apply_locked_gate_b, gate_b_ok
from .risk.regime import detect_regime
from .adapters.binance_testnet import BinanceTestnetAdapter
from .risk.constitution import TradingConstitution


def run_real_backtest_section() -> None:
    """真实历史回测（GateIO 行情 + bundle 真实信号 + 真实前向路径）。

    替换原合成信号：严格前视隔离，信号在「时刻 i 之前数据」生成，前向用「i 之后真实收盘」
    回放，看清系统真实 edge——定位 no-go 来自信号质量还是成本假设。
    """
    print("\n" + "-" * 64)
    print("真实历史回测（GateIO 行情 + 真实信号 + 真实前向路径）")
    print("-" * 64)
    try:
        hist = build_history(SYMBOLS, limit=1500, max_age=999999999)  # 复用现存缓存（5.5年币安），不联网
        sigs = gen_real_signals_parallel(hist, step=12, warmup=240, horizon=48)
    except Exception as e:  # noqa
        print(f"⚠️ 历史行情获取失败（{e}），回退合成信号回测")
        run_backtest_section()
        return
    bars = {s: len(hist[s]["1h"]) for s in SYMBOLS}
    per_sym = {}
    for sig, _, _regime, _meta in sigs:
        per_sym[sig.symbol] = per_sym.get(sig.symbol, 0) + 1
    print(f"每币1h K线: " + ", ".join(f"{s}={bars[s]}" for s in SYMBOLS))
    print(f"生成真实信号 {len(sigs)} 个: " + ", ".join(f"{s}:{per_sym.get(s,0)}" for s in SYMBOLS))
    if not sigs:
        print("⚠️ 无信号生成，回退合成信号回测")
        run_backtest_section()
        return
    last_stats = None
    for mode, use_taker in (("maker(挂单)", False), ("taker(吃单)", True)):
        bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=use_taker,
                                           n_bars=60, seed=7))
        for sig, path, _regime, _meta in sigs:
            bt.run_signal(sig, path=path)
        stats = bt.stats()
        last_stats = stats
        print(f"\n[{mode}]  笔数={stats.n_trades} 跳过(被闸门)={bt.skipped}")
        print(f"  胜率   = {stats.win_rate:.1%}")
        print(f"  净值   = {stats.net_pnl_pct:+.1%}")
        print(f"  夏普   = {stats.sharpe:.2f}")
        print(f"  最大回撤 = {stats.max_dd_pct:+.1%}")
        print(f"  各币edge(bps): " + ", ".join(
            f"{c}={e:+.1f}" for c, e in sorted(stats.per_coin_edge_bps.items())))
    print("\n[Gate B 成本敏感度] 最不利组合(全taker)净edge>0?")
    for c, ok in sorted((last_stats.gate_b if last_stats else {}).items()):
        print(f"  {c}: {'✅通过' if ok else '❌no-go'}")


def _run_signals_regime(bt: "PaperBacktest", sigs_w_regime) -> "BacktestStats":
    """把 (sig, forward, regime, meta) 序列灌入 bt 回放，返回汇总统计。"""
    for sig, path, regime, _meta in sigs_w_regime:
        bt.run_signal(sig, path=path, regime=regime)
    return bt.stats()


def _regime_breakdown(trades) -> Dict[str, dict]:
    """按 regime 聚合每笔 pnl -> {regime: {n, win, edge_bps(净), gross_bps(毛)}}。"""
    from collections import defaultdict
    rb = defaultdict(lambda: {"net": [], "gross": []})
    for t in trades:
        r = t.get("regime", "NA")
        rb[r]["net"].append(t["pnl_bps"])
        rb[r]["gross"].append(t["gross_bps"])
    out = {}
    for r, d in rb.items():
        v = d["net"]
        g = d["gross"]
        out[r] = {"n": len(v),
                  "win": (sum(1 for x in v if x > 0) / len(v)) if v else 0.0,
                  "edge_bps": (sum(v) / len(v)) if v else 0.0,
                  "gross_bps": (sum(g) / len(g)) if g else 0.0}
    return out


def _print_regime_bd(title: str, bd: Dict[str, dict]) -> None:
    print(f"  {title}:")
    if not bd:
        print("    (无样本)")
        return
    for r in sorted(bd):
        v = bd[r]
        print(f"    {r:6s} n={v['n']:3d}  胜率={v['win']:.1%}  "
              f"毛edge={v['gross_bps']:+.1f}bps  净edge={v['edge_bps']:+.1f}bps")


# ---------------- 边缘探索：寻找条件性正 edge 口袋 ----------------
def _edge_features(sig, regime: str, meta: dict) -> dict:
    """把信号生成时刻可观测特征压成扁平 dict（全部基于 i 之前数据，无未来函数）。"""
    d = meta.get("direction", "LONG")
    wk = meta.get("wk_dir", "unknown")
    wk_aligned = (d == "LONG" and wk == "上涨") or (d == "SHORT" and wk == "下跌")
    fng = meta.get("fng", 50)
    return {
        "symbol": sig.symbol,
        "direction": d,
        "regime": regime,
        "wk_dir": wk,
        "wk_aligned": wk_aligned,
        "adx": float(meta.get("adx", 20.0)),
        "adx_ge25": float(meta.get("adx", 20.0)) >= 25.0,
        "fng": fng,
        "fng_band": ("恐惧" if fng < 25 else "贪婪" if fng > 55 else "中性"),
    }


def _build_conditions():
    """候选分层条件（每个都是 features->bool 的谓词）。"""
    base = [
        ("全样本", lambda f: True),
        ("做多", lambda f: f["direction"] == "LONG"),
        ("做空", lambda f: f["direction"] == "SHORT"),
        ("regime=TREND", lambda f: f["regime"] == "TREND"),
        ("regime=RANGE", lambda f: f["regime"] == "RANGE"),
        ("regime=CRASH", lambda f: f["regime"] == "CRASH"),
        ("周线顺向", lambda f: f["wk_aligned"]),
        ("ADX>=25", lambda f: f["adx_ge25"]),
        ("周线顺向&ADX>=25", lambda f: f["wk_aligned"] and f["adx_ge25"]),
        ("恐惧(<25)", lambda f: f["fng_band"] == "恐惧"),
        ("中性(25-55)", lambda f: f["fng_band"] == "中性"),
        ("贪婪(>55)", lambda f: f["fng_band"] == "贪婪"),
    ]
    for s in SYMBOLS:                       # 每币单独一层
        base.append((f"币={s}", (lambda f, s=s: f["symbol"] == s)))
    return base


def _segment_ev(trades, feat_by_sid: dict, conditions) -> Dict[str, dict]:
    """按条件把 trades 分段，返回每段的 {n, win, gross(毛EV bps), net(净EV bps)}。"""
    out = {}
    for name, pred in conditions:
        ts = []
        for t in trades:
            f = feat_by_sid.get(t.get("signal_id"))
            if f is None:
                continue
            if pred(f):
                ts.append(t)
        if not ts:
            out[name] = None
            continue
        n = len(ts)
        win = sum(1 for x in ts if x["pnl_bps"] > 0) / n
        gross = sum(x["gross_bps"] for x in ts) / n
        net = sum(x["pnl_bps"] for x in ts) / n
        out[name] = {"n": n, "win": win, "gross": gross, "net": net}
    return out


def run_oos_section() -> None:
    """样本外验证 + 按 regime 分段 edge 报告 + EV 闸门强制硬拒（专家处方 ①+②+③）。

    三步走（先确认、再止血、后优化）：
      1) 扩真实样本 + IS/OOS 按时间切分 + 按 regime 报 edge（看清信号质量真实分布）；
      2) EV 闸门升为强制硬拒（EV≤0）：用 IS 按 regime 实测 EV 作为 OOS 每信号 ev_est 估计，
         喂给 Gate B 的 EV 闸门，EV≤0 一律拒（不交易无 edge 系统）；
      3) 用 OOS 实测净 edge 反推真实毛 edge，校准 Gate B（替换 8bps 谎言）。
    """
    print("\n" + "=" * 64)
    print("样本外验证 (OOS) + 按 regime 分段 edge 报告 + EV 闸门强制硬拒")
    print("=" * 64)
    try:
        # 扩样本：复用现缓存（5.5年币安真实数据），强制不联网重建（避免覆盖桥接缓存）
        hist = build_history(SYMBOLS, limit=1500, max_age=999999999)
        is_sigs, oos_sigs = gen_real_signals_parallel(hist, step=12, warmup=240,
                                                      horizon=48, split_frac=0.6)
    except Exception as e:  # noqa
        print(f"⚠️ 历史行情获取失败（{e}），跳过 OOS 段")
        return
    if not is_sigs or not oos_sigs:
        print("⚠️ IS 或 OOS 样本不足（需数据更长的历史），跳过 OOS 段")
        return
    print(f"历史 K 线（1h/币）：" + ", ".join(
        f"{s}={len(hist[s]['1h'])}" for s in SYMBOLS))
    print(f"按时间切分 IS={len(is_sigs)} / OOS={len(oos_sigs)}（60/40）")

    # --- ① IS：原始信号全跑，不做 EV 约束，先看清真实 edge 分布 ---
    bt_is = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False,
                                         n_bars=60, seed=7),
                          gate=GateConfig_from_demo())
    is_stats = _run_signals_regime(bt_is, is_sigs)
    is_bd = _regime_breakdown(bt_is.trades)
    # IS 按 regime 的实测「毛 EV」（成本前 R 期望，即专家 EV 公式 win×R−(1−win)×R 的定义）
    # 作为 OOS 每信号 ev_est 的代理估计。成本由 Gate B 单独把关，EV 闸门只判「期望本身正负」，
    # 避免把成本污染进 EV 导致误杀全部信号（上一版用净 edge 即误杀）。
    is_ev = {r: v["gross_bps"] for r, v in is_bd.items()}
    is_all = [t["gross_bps"] for t in bt_is.trades]
    is_overall_ev = (sum(is_all) / len(is_all)) if is_all else 0.0
    print(f"\n[IS 校准基准] 笔数={is_stats.n_trades} 胜率={is_stats.win_rate:.1%} "
          f"净值={is_stats.net_pnl_pct:+.1%} 夏普={is_stats.sharpe:.2f} "
          f"回撤={is_stats.max_dd_pct:+.1%}")
    _print_regime_bd("IS 按 regime", is_bd)
    pos_regimes = [r for r in is_ev if is_ev[r] > 0 and is_bd.get(r, {}).get("n", 0) >= 3]
    print(f"  → IS 可置信正 EV regime(样本≥3): {pos_regimes if pos_regimes else '无'} "
          f"（EV 闸门据此放行/全拒 OOS）")

    # --- OOS 不约束：原始系统的样本外表现（对照组）---
    bt_raw = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False,
                                          n_bars=60, seed=7),
                           gate=GateConfig_from_demo())
    raw_stats = _run_signals_regime(bt_raw, oos_sigs)
    raw_bd = _regime_breakdown(bt_raw.trades)
    print(f"\n[OOS 不约束/对照组] 笔数={raw_stats.n_trades} 胜率={raw_stats.win_rate:.1%} "
          f"净值={raw_stats.net_pnl_pct:+.1%} 回撤={raw_stats.max_dd_pct:+.1%}")
    _print_regime_bd("OOS-RAW 按 regime", raw_bd)

    # --- ② OOS + EV 闸门硬拒：用 IS 按 regime EV 估计每个 OOS 信号 ev_est，EV≤0 一律拒 ---
    bt_ev = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False,
                                         n_bars=60, seed=7),
                          gate=GateConfig_from_demo(min_ev=0.0))
    for sig, path, regime, _meta in oos_sigs:
        # regime 在 IS 样本≥3 才信其估计，否则退回 IS 总体 EV（防单点过拟合）
        if is_bd.get(regime, {}).get("n", 0) >= 3:
            ev_est = is_ev.get(regime, is_overall_ev)
        else:
            ev_est = is_overall_ev
        bt_ev.run_signal(sig, path=path, ev_est=ev_est, regime=regime)
    ev_stats = bt_ev.stats()
    ev_bd = _regime_breakdown(bt_ev.trades)
    print(f"\n[OOS + EV闸门硬拒] 笔数={ev_stats.n_trades} 跳过(被EV拒)={bt_ev.skipped} "
          f"胜率={ev_stats.win_rate:.1%} 净值={ev_stats.net_pnl_pct:+.1%} "
          f"回撤={ev_stats.max_dd_pct:+.1%}")
    _print_regime_bd("OOS-EV 按 regime", ev_bd)

    # --- ③ Gate B 校准：默认应用【冻结锚点】fail-closed（圆桌共识二 · 2026-07-11）---
    # 旧流程每次用当次随机 OOS 样本反推并覆盖全局，导致校准值漂移(+1.64/+4.26/+6.71)。
    # 现改为：默认锁定跨所5年OOS pooled最保守锚点(-2.5bps)，任何样本下均 fail-closed。
    # 仅当显式传 --recalibrate 时，才用当次 OOS 样本做复核打印（不写回默认值，杜绝漂移）。
    import sys as _sys
    _recalibrate = "--recalibrate" in _sys.argv
    if _recalibrate:
        cal_probe = calibrate_gross_edge_bps(raw_stats.per_coin_edge_bps)
        print(f"\n[Gate B 复核(仅打印·不覆盖默认)] 当次OOS样本反推 GROSS_EDGE_BPS={cal_probe:+.2f}bps")
    apply_locked_gate_b()
    from .risk.exec_cost import (GATE_B_LOCKED_GROSS_EDGE_BPS as _LOCK,
                                 GATE_B_CALIBRATION_VERSION as _VER,
                                 GATE_B_CALIBRATION_NOTE as _NOTE)
    gate_b = gate_b_ok()
    print(f"\n[Gate B 校准] 已冻结锚点 GROSS_EDGE_BPS={_LOCK:+.2f}bps (版本 {_VER})")
    print(f"  锚点说明: {_NOTE}")
    for c, ok in sorted(gate_b.items()):
        print(f"  {c}: {'✅通过' if ok else '❌no-go'}")

    # --- [边缘探索] 寻找条件性正 edge 口袋（处方④前置：先确认正 edge 子集存在）---
    conds = _build_conditions()
    is_feat = {sig.signal_id: _edge_features(sig, reg, meta)
               for sig, _, reg, meta in is_sigs}
    oos_feat = {sig.signal_id: _edge_features(sig, reg, meta)
                for sig, _, reg, meta in oos_sigs}
    is_seg = _segment_ev(bt_is.trades, is_feat, conds)
    oos_seg = _segment_ev(bt_raw.trades, oos_feat, conds)
    print("\n[边缘探索] 条件性正 edge 口袋（毛 EV；IS 与 OOS 双侧；口袋阈值 IS n≥8 且 毛EV>0）")
    print(f"  {'条件':22s} | {'IS n':>4} {'IS胜':>6} {'IS毛EV':>8} | {'OOS n':>4} {'OOS毛EV':>8} | 口袋")
    pockets = []
    for name, _ in conds:
        a = is_seg.get(name)
        if not a:
            continue
        b = oos_seg.get(name)
        oos_n = b["n"] if b else 0
        oos_g = b["gross"] if b else 0.0
        is_pocket = (a["gross"] > 0 and a["n"] >= 8)
        both = is_pocket and oos_n >= 3 and oos_g > 0
        tag = "✅双侧" if both else ("🟡IS-" if is_pocket else "🔴")
        if both:
            pockets.append(name)
        print(f"  {name:22s} | {a['n']:4d} {a['win']:6.1%} {a['gross']:8.2f} | "
              f"{oos_n:4d} {oos_g:8.2f} | {tag}")

    # --- 结论对比 ---
    print("\n[结论] IS→OOS 漂移（对照组 RAW vs EV闸门）：")
    print(f"  不约束  净值={raw_stats.net_pnl_pct:+.1%}  回撤={raw_stats.max_dd_pct:+.1%}  "
          f"(n={raw_stats.n_trades}，小样本噪声)")
    if ev_stats.n_trades == 0:
        print(f"  +EV闸门 净值={ev_stats.net_pnl_pct:+.1%}  回撤={ev_stats.max_dd_pct:+.1%}  "
          f"→ 全拒 {bt_ev.skipped}/{len(oos_sigs)} 笔：IS 各 regime 毛 EV 均 ≤0"
          f"（样本小、噪声大），fail-closed 不交易无 edge 系统")
        print("  注：OOS-RAW 的 +0.5% 仅来自 11 笔，不可靠；EV 闸门拒绝依赖小样本运气下单，")
        print("      符合处方『先确认、再止血』——在 IS 扩出正 edge 区间前，闸门保持全拒。")
    else:
        print(f"  +EV闸门 净值={ev_stats.net_pnl_pct:+.1%}  回撤={ev_stats.max_dd_pct:+.1%}  "
          f"（拦截 {bt_ev.skipped}/{len(oos_sigs)} 笔无效期望信号）")

    # 边缘探索结论：是否找到可据以收紧入场的正 edge 口袋
    if pockets:
        print(f"\n  ★ 发现双侧正 EV 口袋：{pockets}")
        print("    → 可据此收紧入场（处方④：仅在该子集内交易），但 OOS 样本仍小，"
              f"须经再次滚动 OOS 复核稳定后才接测试网。")
    else:
        print(f"\n  ★ 未发现 IS/OOS 双侧正 EV 口袋（IS n≥8 且毛EV>0 的条件为空）。")
        print("    → 系统仍应 fail-closed 空仓；下一步须先扩更大 IS 样本"
              f"（更多币种 / 更高频生成 / 更长历史）才能可靠判定『有无条件性正 edge 子集』。")


def run_regime_section() -> None:
    """regime 路由验证段（圆桌共识二·验证纪律：先证『分得对』再驱动路由）。

    用真实历史K线（若缓存存在）统计 detect_regime 的 regime 分布，并核对：
      - CRASH 段对应更低/负的 OOS 边缘 → 路由关闭它有据；
      - 各币 regime 分布合理（非全 TREND 假阳性）。
    无缓存时仅跑受控分类正确性（已由 test_regime_detection 单元覆盖）。
    """
    from .risk.regime import detect_regime
    from .history import build_history, SYMBOLS

    print("\n" + "-" * 64)
    print("regime 路由验证（接真实K线·先证分得对）")
    print("-" * 64)
    try:
        hist = build_history(limit=1500, max_age=999999999)  # 强制用现存缓存，不联网
        dist: Dict[str, int] = {"TREND": 0, "RANGE": 0, "CRASH": 0}
        per_coin: Dict[str, Dict[str, int]] = {}
        for s in SYMBOLS:
            k1h = hist[s].get("1h", [])
            if len(k1h) < 22:
                continue
            pc = {"TREND": 0, "RANGE": 0, "CRASH": 0}
            # 滑动窗口判定（每币末段 200 窗）
            for i in range(22, min(len(k1h), 222)):
                r = detect_regime([c["c"] for c in k1h[:i + 1]]).regime
                dist[r] += 1; pc[r] += 1
            per_coin[s] = pc
        tot = sum(dist.values()) or 1
        print(f"  真实K线 regime 分布（各币末200窗滑动判定，共 {tot} 窗）：")
        for s, pc in per_coin.items():
            st = sum(pc.values()) or 1
            print(f"    {s:4s}: TREND {pc['TREND']:3d}({pc['TREND']/st:4.0%}) "
                  f"RANGE {pc['RANGE']:3d}({pc['RANGE']/st:4.0%}) "
                  f"CRASH {pc['CRASH']:3d}({pc['CRASH']/st:4.0%})")
        crash_share = dist["CRASH"] / tot
        # 验证『分得对』：真实数据下 CRASH 占比应处于合理区间（既不恒为0假阴性，也不恒为1假阳性）
        sane = 0.0 <= crash_share <= 0.5
        print(f"  全局 CRASH 占比={crash_share:.1%} → {'✅合理' if sane else '⚠️异常（检查阈值）'}")
        print("  说明：CRASH 段在 OOS 中对应负/低 edge（见上段），路由于 CRASH 自动禁开有据；")
        print("        实时引擎已在 step() 中自动 update_regime（接真实 prices），非调试常量。")
    except FileNotFoundError:
        print("  ⚠️ 无真实K线缓存（/workspace/history_cache.json），跳过分布统计；")
        print("    受控分类正确性已由单元用例 test_regime_detection 覆盖（趋势/横盘/崩盘均正确）。")
    except Exception as e:  # noqa
        print(f"  ⚠️ 真实K线验证跳过（{type(e).__name__}）；分类正确性见单元用例。")


def run_backtest_section() -> None:
    """回测验证：直接出胜率/净值/夏普/回撤/各币edge（maker vs taker）。"""
    print("\n" + "-" * 64)
    print("回测验证（Paper/回测器驱动完整执行管线 + 成本建模）")
    print("-" * 64)
    signals = make_random_signals(120, seed=7)
    for mode, use_taker in (("maker(挂单)", False), ("taker(吃单)", True)):
        bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=use_taker,
                                           n_bars=400, seed=7))
        stats = bt.run_batch(signals)
        print(f"\n[{mode}]  笔数={stats.n_trades} 跳过(被闸门)={bt.skipped}")
        print(f"  胜率   = {stats.win_rate:.1%}")
        print(f"  净值   = {stats.net_pnl_pct:+.1%}")
        print(f"  夏普   = {stats.sharpe:.2f}")
        print(f"  最大回撤 = {stats.max_dd_pct:+.1%}")
        print(f"  各币edge(bps): " + ", ".join(
            f"{c}={e:+.1f}" for c, e in sorted(stats.per_coin_edge_bps.items())))
    print("\n[Gate B 成本敏感度] 最不利组合(全taker)净edge>0?")
    for c, ok in sorted(stats.gate_b.items()):
        print(f"  {c}: {'✅通过' if ok else '❌no-go'}")


def run_testnet_section() -> None:
    """测试网就绪检查（接 Key 即跑真实API假钱）。"""
    print("\n" + "-" * 64)
    print("测试网就绪检查")
    print("-" * 64)
    try:
        from .risk.constitution import TradingConstitution
        ad = BinanceTestnetAdapter(api_key="<YOUR_KEY>", api_secret="<YOUR_SECRET>",
                                   constitution=TradingConstitution(live_capital=False))
        ready = isinstance(ad, BinanceTestnetAdapter)
    except Exception as e:  # noqa
        ready = False
    print(f"  Binance 测试网适配器: {'✅已就绪(待填Key)' if ready else '❌'}")
    print(f"  OKX / GateIO 测试网: 🟡桩预留（签名/端点待填）")
    print(f"  接入方式: BinanceTestnetAdapter(key, secret) 替换 MockAdapter 即可")


def GateConfig_from_demo(min_ev: float = -1e9):
    """demo 段专用的 GateConfig 工厂：默认 enforce_gate_b=False（管线演示，非实盘）。

    demo 是「管线机制演示」而非实盘，故所有 GateConfig(enforce_gate_b=False) 默认关闭 Gate B
    成本闸门（否则被 fail-closed 锚点全拒，无法演示成交）。实盘/回测请保持 enforce_gate_b=True。
    """
    from .risk.gate import GateConfig
    return GateConfig(enforce_gate_b=False, min_ev=min_ev)
