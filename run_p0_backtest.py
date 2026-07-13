"""P0 修复集成（并行版）：把逐期资金费率 + OI 接入信号引擎，跑对照回测。

原回测 bug：history.py 把资金费率存成【常量标量】，gen_real_signals 里
fr_delta=0.0、oi_pct=0.0 —— 资金费率/持仓量维度从未触发。本脚本用真实
时间序列补回 fr/fr_delta/oi_pct（复刻 demo 的 imap_unordered 并行 + 严格前视隔离）。
"""
import json, sys, bisect, multiprocessing as mp
sys.path.insert(0, "/workspace/cryptoquant_extract")

from cryptoquant_auto.signals import generate_signals, MarketContext
from cryptoquant_auto.signals.indicators import calc_adx
from cryptoquant_auto.meta.cognition import assess
from cryptoquant_auto.risk.regime import detect_regime
from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.history import SYMBOLS
from cryptoquant_auto.demo import (_regime_breakdown, _build_conditions,
                                    _segment_ev, _edge_features, _print_regime_bd)

hist = json.load(open("/workspace/history_cache.json"))
deriv = json.load(open("/workspace/deriv_data.json"))

# 时间序列：[ts, val] 升序
fr_map = {s: sorted(deriv[s]["fr"], key=lambda x: x[0]) for s in SYMBOLS}
oi_map = {s: sorted(deriv[s]["oi"], key=lambda x: x[0]) for s in SYMBOLS}
fr_ts = {s: [x[0] for x in fr_map[s]] for s in SYMBOLS}
fr_v = {s: [x[1] for x in fr_map[s]] for s in SYMBOLS}
oi_ts = {s: [x[0] for x in oi_map[s]] for s in SYMBOLS}
oi_v = {s: [x[1] for x in oi_map[s]] for s in SYMBOLS}


def _lookup(ts_arr, v_arr, ts):
    i = bisect.bisect_right(ts_arr, ts) - 1
    return v_arr[i] if i >= 0 else (v_arr[0] if v_arr else 0.0)


def _p0_worker(args):
    sym, i, k1h_hist, k4h_full, k1w_full, fng = args
    ti = k1h_hist[-1]["t"] * 1000
    w1h = k1h_hist
    w4h = [x for x in k4h_full if x["t"] <= ti // 1000]
    w1w = [x for x in k1w_full if x["t"] <= ti // 1000]
    price = w1h[-1]["c"]
    day = ti // 1000 // 86400 * 86400
    fg = fng.get(day, 50)
    wk = w1w[-30:] if len(w1w) >= 30 else w1w
    wk_adx, wk_pdi, wk_mdi = (calc_adx(wk) if len(wk) >= 2 else (20.0, 20.0, 20.0))
    wk_dir = ("上涨" if wk_pdi > wk_mdi else "下跌" if wk_mdi > wk_pdi else "unknown")
    env = assess([c for c in w1h[-24:]], fg_val=fg)
    # ★ P0：真实逐期资金费率 / OI
    fr_now = _lookup(fr_ts[sym], fr_v[sym], ti)
    fr_old = _lookup(fr_ts[sym], fr_v[sym], ti - 3 * 86400 * 1000)
    frd = fr_now - fr_old
    oi_now = _lookup(oi_ts[sym], oi_v[sym], ti)
    oi_old = _lookup(oi_ts[sym], oi_v[sym], ti - 86400 * 1000)
    oip = (oi_now - oi_old) / oi_old if oi_old else 0.0
    ctx = MarketContext(fg_val=fg, fr=fr_now, fr_delta=frd, oi_pct=oip,
                        wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir),
                        market_state=env.dominant)
    market_data = {sym: {"1h": w1h, "4h": w4h, "1w": w1w,
                         "fr": fr_now, "fr_delta": frd}}
    sigs = generate_signals([sym], market_data, ctx=ctx,
                            price_map={sym: price}, tf="1H")
    if not sigs:
        return (sym, i, None)
    regime = detect_regime([c["c"] for c in w1h]).regime
    adx_v = calc_adx(w1h)[0] if len(w1h) >= 15 else 20.0
    sig0 = sigs[0]
    meta = {"wk_dir": wk_dir, "adx": adx_v, "fng": fg,
            "conf": sig0.confidence,
            "atr_pct": (sig0.atr / price * 100) if price else 0.0,
            "direction": sig0.direction.value,
            "fr": fr_now, "fr_delta": frd, "oi_pct": oip}
    return (sym, i, (sig0, regime, meta))


def gen_p0_parallel(step=12, warmup=240, horizon=48, split_frac=None,
                    max_workers=None):
    max_workers = max_workers or mp.cpu_count() or 4
    tasks = []
    for s in SYMBOLS:
        k1h = hist[s]["1h"]
        k4h_full = hist[s]["4h"]
        k1w_full = hist[s]["1w"]
        fng = hist[s].get("fng", {})
        n = len(k1h)
        if n < warmup + horizon + 1:
            continue
        for i in range(warmup, n - horizon, step):
            tasks.append((s, i, k1h[:i + 1], k4h_full, k1w_full, fng))
    core_map = {}
    with mp.Pool(processes=max_workers) as pool:
        for sym, i, res in pool.imap_unordered(_p0_worker, tasks, chunksize=8):
            if res is not None:
                core_map[(sym, i)] = res
    out_map = {}
    for (sym, i), (sig0, regime, meta) in core_map.items():
        k1h_full = hist[sym]["1h"]
        forward = [x["c"] for x in k1h_full[i + 1:i + 1 + horizon]]
        if forward:
            out_map[(sym, i)] = (sig0, forward, regime, meta)
    sym_order = {s: i for i, s in enumerate(SYMBOLS)}
    out = [out_map[k] for k in sorted(out_map,
                                      key=lambda kv: (sym_order.get(kv[0], 999), kv[1]))]
    if split_frac is None:
        return out
    k = max(1, int(len(out) * float(split_frac)))
    return out[:k], out[k:]


print("=" * 64)
print("P0 修复对照回测：真实逐期资金费率 + OI 接入信号引擎（并行）")
print("=" * 64)
sigs = gen_p0_parallel(step=12, warmup=240, horizon=48)
per_sym = {}
fr_range = []
for sig, _, _, meta in sigs:
    per_sym[sig.symbol] = per_sym.get(sig.symbol, 0) + 1
    fr_range.append(meta["fr"])
print(f"生成信号 {len(sigs)} 个: " + ", ".join(f"{s}:{per_sym.get(s,0)}" for s in SYMBOLS))
print(f"（fr 真实序列覆盖全5.5年 | oi 近30天, 其余回退0）")
print(f"fr 样本: min={min(fr_range):.5f} max={max(fr_range):.5f} | "
      f"首信号 fr={sigs[0][3]['fr']:.5f} fr_delta={sigs[0][3]['fr_delta']:+.5f} "
      f"oi_pct={sigs[0][3]['oi_pct']:+.3%}")

is_sigs, oos_sigs = gen_p0_parallel(step=12, warmup=240, horizon=48, split_frac=0.6)

# --- 全样本 maker 回测 ---
bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                   gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
for sig, path, regime, meta in sigs:
    bt.run_signal(sig, path=path, regime=regime)
st = bt.stats()
print(f"\n[全样本 maker] 笔数={st.n_trades} 胜率={st.win_rate:.1%} "
      f"净值={st.net_pnl_pct:+.1%} 夏普={st.sharpe:.2f} 回撤={st.max_dd_pct:+.1%}")
print("  各币edge(bps): " + ", ".join(f"{c}={e:+.1f}" for c, e in sorted(st.per_coin_edge_bps.items())))

# --- IS / OOS ---
bt_is = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                      gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
for sig, path, regime, meta in is_sigs:
    bt_is.run_signal(sig, path=path, regime=regime)
is_stats = bt_is.stats()
is_bd = _regime_breakdown(bt_is.trades)
is_ev = {r: v["gross_bps"] for r, v in is_bd.items()}
is_all = [t["gross_bps"] for t in bt_is.trades]
is_overall_ev = (sum(is_all) / len(is_all)) if is_all else 0.0
print(f"\n[IS] 笔数={is_stats.n_trades} 胜率={is_stats.win_rate:.1%} "
      f"净值={is_stats.net_pnl_pct:+.1%} 夏普={is_stats.sharpe:.2f} 回撤={is_stats.max_dd_pct:+.1%}")
_print_regime_bd("IS 按 regime", is_bd)

bt_raw = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                       gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
for sig, path, regime, meta in oos_sigs:
    bt_raw.run_signal(sig, path=path, regime=regime)
raw_stats = bt_raw.stats()
raw_bd = _regime_breakdown(bt_raw.trades)
print(f"\n[OOS 不约束] 笔数={raw_stats.n_trades} 胜率={raw_stats.win_rate:.1%} "
      f"净值={raw_stats.net_pnl_pct:+.1%} 回撤={raw_stats.max_dd_pct:+.1%}")
_print_regime_bd("OOS-RAW 按 regime", raw_bd)

bt_ev = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                      gate=GateConfig(min_ev=0.0, enforce_gate_b=False))
for sig, path, regime, meta in oos_sigs:
    ev_est = is_ev.get(regime, is_overall_ev) if is_bd.get(regime, {}).get("n", 0) >= 3 else is_overall_ev
    bt_ev.run_signal(sig, path=path, ev_est=ev_est, regime=regime)
ev_stats = bt_ev.stats()
print(f"\n[OOS + EV闸门] 笔数={ev_stats.n_trades} 跳过={bt_ev.skipped} 胜率={ev_stats.win_rate:.1%} "
      f"净值={ev_stats.net_pnl_pct:+.1%} 回撤={ev_stats.max_dd_pct:+.1%}")

# --- 边缘探索 ---
conds = _build_conditions()
is_feat = {sig.signal_id: _edge_features(sig, reg, meta) for sig, _, reg, meta in is_sigs}
oos_feat = {sig.signal_id: _edge_features(sig, reg, meta) for sig, _, reg, meta in oos_sigs}
is_seg = _segment_ev(bt_is.trades, is_feat, conds)
oos_seg = _segment_ev(bt_raw.trades, oos_feat, conds)
print("\n[边缘探索] 条件性正 edge 口袋（毛EV；IS/OOS 双侧；IS n≥8 且 毛EV>0）")
print(f"  {'条件':22s} | {'IS n':>4} {'IS毛EV':>8} | {'OOS n':>4} {'OOS毛EV':>8} | 口袋")
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
    print(f"  {name:22s} | {a['n']:4d} {a['gross']:8.2f} | {oos_n:4d} {oos_g:8.2f} | {tag}")

print("\n" + "=" * 64)
print("对照基线（上次全量回测，无真实 fr/oi）：")
print("  maker: 胜率22.8% 净值-12.0% 夏普-2.81 | OOS不约束: 净值-0.4% n=105")
print("P0（真实资金费率+OI 接入）：")
print(f"  全样本: 胜率{st.win_rate:.1%} 净值{st.net_pnl_pct:+.1%} 夏普{st.sharpe:.2f}")
print(f"  IS: 胜率{is_stats.win_rate:.1%} 净值{is_stats.net_pnl_pct:+.1%} 夏普{is_stats.sharpe:.2f}")
print(f"  OOS不约束: 胜率{raw_stats.win_rate:.1%} 净值{raw_stats.net_pnl_pct:+.1%} 回撤{raw_stats.max_dd_pct:+.1%}")
print(f"  OOS+EV闸门: 笔数{ev_stats.n_trades} 跳过{bt_ev.skipped}")
print(f"  双侧正EV口袋: {pockets if pockets else '无'}")
print("=" * 64)
