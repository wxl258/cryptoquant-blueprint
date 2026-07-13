"""P0 修复对照回测（路径修正版）：复用原 run_p0_backtest.py 逻辑，
修正硬编码路径 /workspace/cryptoquant_extract 与 /workspace/*.json 为相对脚本目录。
依赖已修复的回测器（C1/C4 修正生效）。"""
import os
import json
import bisect
import multiprocessing as mp

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(_PKG_DIR, "cryptoquant_auto")
import sys
sys.path.insert(0, _PKG_DIR)

from cryptoquant_auto.signals import generate_signals, MarketContext
from cryptoquant_auto.signals.indicators import calc_adx
from cryptoquant_auto.meta.cognition import assess
from cryptoquant_auto.risk.regime import detect_regime
from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.history import SYMBOLS
from cryptoquant_auto.demo import (_regime_breakdown, _build_conditions,
                                    _segment_ev, _edge_features, _print_regime_bd)

_HIST_CACHE = os.path.join(_PKG_DIR, "history_cache.json")
_DERIV_DATA = os.path.join(_PKG_DIR, "deriv_data.json")
hist = json.load(open(_HIST_CACHE))
deriv = json.load(open(_DERIV_DATA))

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
    fr_now = _lookup(fr_ts[sym], fr_v[sym], ti)
    fr_old = _lookup(fr_ts[sym], fr_v[sym], ti - 3 * 86400 * 1000)
    frd = fr_now - fr_old
    oi_now = _lookup(oi_ts[sym], oi_v[sym], ti)
    oi_old = _lookup(oi_ts[sym], oi_v[sym], ti - 86400 * 1000)
    oip = (oi_now - oi_old) / oi_old if oi_old else 0.0
    regime = detect_regime([c["c"] for c in w1h]).regime
    ctx = MarketContext(fg_val=fg, fr=fr_now, fr_delta=frd, oi_pct=oip,
                        wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir),
                        market_state=env.dominant, regime=regime)
    market_data = {sym: {"1h": w1h, "4h": w4h, "1w": w1w,
                         "fr": fr_now, "fr_delta": frd}}
    sigs = generate_signals([sym], market_data, ctx=ctx,
                            price_map={sym: price}, tf="1H")
    if not sigs:
        return (sym, i, None)
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


if __name__ == "__main__":
    print("=" * 64)
    print("P0 修复对照回测（路径修正版，用修复后回测器）")
    print("=" * 64)
    sigs = gen_p0_parallel(step=12, warmup=240, horizon=48)
    per_sym = {}
    fr_range = []
    for sig, _, _, meta in sigs:
        per_sym[sig.symbol] = per_sym.get(sig.symbol, 0) + 1
        fr_range.append(meta["fr"])
    print(f"生成信号 {len(sigs)} 个: " + ", ".join(f"{s}:{per_sym.get(s,0)}" for s in SYMBOLS))
    print(f"fr 样本: min={min(fr_range):.5f} max={max(fr_range):.5f}")

    # --- IS/OOS 切分 ---
    is_sigs, oos_sigs = gen_p0_parallel(step=12, warmup=240, horizon=48, split_frac=0.6)

    # --- 全样本 maker 回测（用修复后回测器，C1/C3/C4/C5 均生效）---
    from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
    from cryptoquant_auto.risk.gate import GateConfig

    bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                       gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
    for sig, path, regime, meta in sigs:
        bt.run_signal(sig, path=path, regime=regime)
    st = bt.stats()
    print(f"\n[全样本 maker] 笔数={st.n_trades} 胜率={st.win_rate:.1%} "
          f"净值={st.net_pnl_pct:+.1%} 夏普={st.sharpe:.2f} 回撤={st.max_dd_pct:+.1%}")
    print("  各币edge(bps): " + ", ".join(f"{c}={e:+.1f}" for c, e in sorted(st.per_coin_edge_bps.items())))

    # --- IS ---
    bt_is = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                          gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
    for sig, path, regime, meta in is_sigs:
        bt_is.run_signal(sig, path=path, regime=regime)
    is_stats = bt_is.stats()
    is_bd = _regime_breakdown(bt_is.trades)
    print(f"\n[IS] 笔数={is_stats.n_trades} 胜率={is_stats.win_rate:.1%} "
          f"净值={is_stats.net_pnl_pct:+.1%} 夏普={is_stats.sharpe:.2f} 回撤={is_stats.max_dd_pct:+.1%}")
    _print_regime_bd("IS 按 regime", is_bd)

    # --- OOS（不约束）---
    bt_raw = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                           gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
    for sig, path, regime, meta in oos_sigs:
        bt_raw.run_signal(sig, path=path, regime=regime)
    raw_stats = bt_raw.stats()
    raw_bd = _regime_breakdown(bt_raw.trades)
    print(f"\n[OOS 不约束] 笔数={raw_stats.n_trades} 胜率={raw_stats.win_rate:.1%} "
          f"净值={raw_stats.net_pnl_pct:+.1%} 回撤={raw_stats.max_dd_pct:+.1%}")
    _print_regime_bd("OOS 按 regime", raw_bd)

    # --- 结论 ---
    print("\n" + "=" * 64)
    print("裁决")
    print("=" * 64)
    print(f"全样本: {st.net_pnl_pct:+.1%} | IS: {is_stats.net_pnl_pct:+.1%} | OOS: {raw_stats.net_pnl_pct:+.1%}")
    if raw_stats.net_pnl_pct < 0:
        print("❌ OOS 净负 → 系统在当前5因子权重下无可持续正 edge，维持 fail-closed")
        print("   建议：因子发现（重做加权）。C5解锁条件不满足。")
    else:
        print("✅ OOS 净正，但需 WFA 交叉确认后再决策是否解锁 Gate B")

