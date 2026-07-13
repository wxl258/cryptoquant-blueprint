"""圆桌补充分析：做多为什么亏？有没有可修复的办法？

从 WFA 缓存信号中，对做多信号做子集分析：
  1. 做多信号的整体 edge 分布
  2. 做多 × regime：TREND/RANGE/CRASH
  3. 做多 × ADX：强趋势/弱趋势
  4. 做多 × fr：正费/负费/极端负费
  5. 做多 × 周线方向：上涨/下跌/unknown
  6. 做多 × 币种差异

目标是找出：有没有哪个做多子集有正 edge？做多亏钱是
(A) 所有做多都亏 → 做多因子无效
(B) 部分做多亏（fr偏负时被误推入场的低质信号）拖累整体
"""
import json, statistics, sys, pickle, math
sys.path.insert(0, "/workspace/cryptoquant_extract")
from cryptoquant_auto.history import SYMBOLS
from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
from cryptoquant_auto.risk.gate import GateConfig

hist = json.load(open("/workspace/history_cache.json"))

def bh_fdr(pvals, alpha=0.10):
    m = len(pvals)
    if m == 0: return [], []
    sorted_idx = sorted(range(m), key=lambda i: pvals[i])
    max_k = -1
    for k in range(m):
        if pvals[sorted_idx[k]] <= alpha * (k + 1) / m:
            max_k = k
    reject = [False] * m
    for k in range(max_k + 1):
        reject[sorted_idx[k]] = True
    adj = [0.0] * m
    cur_min = 1.0
    for k in reversed(range(m)):
        raw = pvals[sorted_idx[k]]
        corrected = min(raw * m / (k + 1), 1.0)
        cur_min = min(cur_min, corrected)
        adj[sorted_idx[k]] = cur_min
    return adj, reject

def regen_signals_with_bps():
    """从原始数据生成做多信号的逐笔 gross_bps，含子条件标签。
    
    只跑做多信号（方向 == "做多"），大幅加快速度。
    """
    from cryptoquant_auto.signals.engine import gen_signal, MarketContext
    from cryptoquant_auto.signals.indicators import calc_adx
    from cryptoquant_auto.signals.generator import candidate_to_signal
    from cryptoquant_auto.meta.cognition import assess
    from cryptoquant_auto.risk.regime import detect_regime
    
    long_signals = []
    for s in SYMBOLS:
        k1h = hist[s]["1h"]; n = len(k1h)
        if n < 240 + 48 + 1: continue
        for i in range(240, n - 48, 12):
            w1h = k1h[:i+1]; ti = w1h[-1]["t"]
            w4h = [x for x in hist[s]["4h"] if x["t"] <= ti]
            w1w = [x for x in hist[s]["1w"] if x["t"] <= ti]
            price = w1h[-1]["c"]
            day = ti // 86400 * 86400
            fg = hist[s].get("fng", {}).get(day, 50)
            wk = w1w[-30:] if len(w1w) >= 30 else w1w
            wk_adx, wk_pdi, wk_mdi = (calc_adx(wk) if len(wk) >= 2 else (20.0, 20.0, 20.0))
            wk_dir = ("上涨" if wk_pdi > wk_mdi else "下跌" if wk_mdi > wk_pdi else "unknown")
            env = assess([c for c in w1h[-24:]], fg_val=fg)
            # fr 从何处获取？用默认 0 跳过（不影响做多分析）
            ctx = MarketContext(fg_val=fg, fr=0.0, fr_delta=0.0, oi_pct=0.0,
                                wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir), market_state=env.dominant)
            cand = gen_signal(s, w1h, w4h, w1w, ctx=ctx)
            if not cand.passed or cand.direction != "做多": continue
            
            # fr 真实值
            import bisect
            deriv = json.load(open("/workspace/deriv_data.json"))
            fr_data = sorted(deriv[s]["fr"], key=lambda x: x[0])
            fr_ts = [x[0] for x in fr_data]
            fr_vl = [x[1] for x in fr_data]
            ti_ms = ti * 1000
            idx = bisect.bisect_right(fr_ts, ti_ms) - 1
            fr_now = fr_vl[idx] if idx >= 0 else 0.0
            
            sig = candidate_to_signal(cand, price)
            if sig is None: continue
            forward = [x["c"] for x in k1h[i+1:i+1+48]]
            if len(forward) < 5: continue
            bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                               gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
            bt.run_signal(sig, path=forward)
            gross_bps = bt.trades[-1]["gross_bps"] if bt.trades else 0.0
            
            regime = detect_regime([c["c"] for c in w1h]).regime
            adx_v = calc_adx(w1h)[0] if len(w1h) >= 15 else 20.0
            long_signals.append({
                "gross_bps": gross_bps, "regime": regime,
                "adx": adx_v, "fr": fr_now,
                "wk_dir": wk_dir, "symbol": s,
            })
    return long_signals

print("="*72)
print("圆桌·做多亏损根因分析")
print("="*72)

long_sigs = regen_signals_with_bps()
print(f"\n做多信号总数: {len(long_sigs)}")
all_gbp = [s["gross_bps"] for s in long_sigs]
print(f"做多全体: mean={statistics.mean(all_gbp):+.2f} bps, "
      f"median={statistics.median(all_gbp):+.2f} bps")

# ---- 子集分析 ----
conditions = [
    ("全部做多",       lambda s: True),
    ("做多+TREND",    lambda s: s["regime"] == "TREND"),
    ("做多+RANGE",    lambda s: s["regime"] == "RANGE"),
    ("做多+CRASH",    lambda s: s["regime"] == "CRASH"),
    ("做多+ADX≥35",   lambda s: s["adx"] >= 35),
    ("做多+ADX≥25<35", lambda s: 25 <= s["adx"] < 35),
    ("做多+ADX<25",   lambda s: s["adx"] < 25),
    ("做多+fr<-0.001", lambda s: s["fr"] < -0.001),
    ("做多+fr>-0.001&<0", lambda s: -0.001 <= s["fr"] < 0),
    ("做多+fr≥0",     lambda s: s["fr"] >= 0),
    ("做多+周线上涨",   lambda s: s["wk_dir"] == "上涨"),
    ("做多+周线下跌",   lambda s: s["wk_dir"] == "下跌"),
    ("做多+周线unknown", lambda s: s["wk_dir"] == "unknown"),
    ("做多+BTC",       lambda s: s["symbol"] == "BTC"),
    ("做多+ETH",       lambda s: s["symbol"] == "ETH"),
    ("做多+SOL",       lambda s: s["symbol"] == "SOL"),
    ("做多+BNB",       lambda s: s["symbol"] == "BNB"),
    ("做多+XRP",       lambda s: s["symbol"] == "XRP"),
    ("做多+TRX",       lambda s: s["symbol"] == "TRX"),
]

print(f"\n{'条件':24s} | {'数量':>5} | {'均值':>8} | {'正次数':>5} | 结论")
print("-"*60)
for name, pred in conditions:
    matched = [s for s in long_sigs if pred(s)]
    if not matched: continue
    gbp = [s["gross_bps"] for s in matched]
    mn = statistics.mean(gbp)
    pos = sum(1 for v in gbp if v > 0)
    if mn > 0 and pos >= len(matched)*0.3:
        tag = "✅正"
    elif mn > 0:
        tag = "🟡微弱正"
    elif pos == 0:
        tag = "🔴全亏"
    else:
        tag = "🔴负"
    print(f"{name:24s} | {len(matched):5d} | {mn:+8.2f} | {pos:3d}/{len(matched):<3d} | {tag}")

print("\n" + "="*72)
print("关键判断：做多亏损是可修复的，还是不可修复的？")
print("如果某个子集（如做多+ADX≥35、做多+TREND）有正 edge")
print("→ 做多是 '被低质信号拖累'，可修")
print("如果所有子集都负 edge")
print("→ 做多因子整体无效")
print("="*72)
