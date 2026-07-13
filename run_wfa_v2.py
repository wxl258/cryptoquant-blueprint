"""P1a：Walk-Forward 验证（滚动 IS/OOS × 条件 edge 一致性 + BH-FDR）

方法：
  1. 生成全量信号（含时间戳 + 前向路径）— 复用 gen_p0_parallel
  2. 逐笔算 gross_bps（用 PaperBacktest 单笔跑 forward path）
  3. 时间排序，切 K 个滚动 fold（各 60% IS / 40% OOS）
  4. 每 fold：跑条件分析 → 记录各条件 OOS gEV
  5. K fold 聚合：单侧 t 检验 H0: gEV ≤ 0 → p 值
  6. BH-FDR 多重检验修正
  7. 报告：哪些条件跨 fold 有可持续正 edge
"""
import json, math, statistics, sys, bisect, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 脚本目录（含 cryptoquant_auto 包）
from cryptoquant_auto.signals.engine import gen_signal, MarketContext
from cryptoquant_auto.signals.generator import candidate_to_signal
from cryptoquant_auto.signals.indicators import calc_adx, calc_vol_price_divergence
from cryptoquant_auto.meta.cognition import assess
from cryptoquant_auto.risk.regime import detect_regime
from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.history import SYMBOLS

# 数据/缓存路径：相对脚本目录，修复 P2-6 路径硬编码（/workspace、/root 错位刷新不生效）
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_HIST_CACHE = os.path.join(_PKG_DIR, "history_cache.json")
_DERIV_DATA = os.path.join(_PKG_DIR, "deriv_data.json")
# P2-4：带版本号的缓存文件名，避免方法论变更后 stale-cache 污染结论
WFA_CACHE_VERSION = "v4_net_nonoverlap"
SIGNAL_CACHE = os.path.join(_PKG_DIR, f"wfa_signals_{WFA_CACHE_VERSION}.pkl")

# 自包含条件（匹配 meta dict 格式）
CONDITIONS = [
    ("全样本", lambda m: True),
    ("做多", lambda m: m["direction"] == "做多"),
    ("做空", lambda m: m["direction"] == "做空"),
    ("regime=TREND", lambda m: m["regime"] == "TREND"),
    ("regime=RANGE", lambda m: m["regime"] == "RANGE"),
    ("regime=CRASH", lambda m: m["regime"] == "CRASH"),
]
# 周线顺向：做多且周线上涨 或 做空且周线下跌
CONDITIONS.append(("周线顺向", lambda m:
    (m["direction"] == "做多" and m.get("wk_dir") == "上涨") or
    (m["direction"] == "做空" and m.get("wk_dir") == "下跌")))
CONDITIONS.append(("ADX>=25", lambda m: m.get("adx", 0) >= 25))
CONDITIONS.append(("周线顺向&ADX>=25", lambda m:
    ((m["direction"] == "做多" and m.get("wk_dir") == "上涨") or
     (m["direction"] == "做空" and m.get("wk_dir") == "下跌"))
    and m.get("adx", 0) >= 25))

# Phase 1: F2 量价背离条件
CONDITIONS.append(("F2看涨背离", lambda m: m.get("f2", 0) == 1.0))   # 价跌量增→看涨
CONDITIONS.append(("F2看跌背离", lambda m: m.get("f2", 0) == -1.0))  # 价涨量缩→看跌
CONDITIONS.append(("F2任意背离", lambda m: m.get("f2", 0) != 0.0))   # 有任何背离



# ============ 信号生成 ============
hist = json.load(open(_HIST_CACHE))
deriv = json.load(open(_DERIV_DATA))
fr_map = {s: sorted(deriv[s]["fr"], key=lambda x: x[0]) for s in SYMBOLS}
fr_ts = {s: [x[0] for x in fr_map[s]] for s in SYMBOLS}
fr_v =  {s: [x[1] for x in fr_map[s]] for s in SYMBOLS}
oi_map = {s: sorted(deriv[s]["oi"], key=lambda x: x[0]) for s in SYMBOLS}
oi_ts = {s: [x[0] for x in oi_map[s]] for s in SYMBOLS}
oi_v =  {s: [x[1] for x in oi_map[s]] for s in SYMBOLS}

def _lookup(ts_arr, v_arr, ts):
    i = bisect.bisect_right(ts_arr, ts) - 1
    return v_arr[i] if i >= 0 else (v_arr[0] if v_arr else 0.0)

def compute_edge_bps(sig, forward):
    """用 PaperBacktest 跑单信号前向路径，返回 (gross_bps, net_bps)。

    - gross_bps：成本前毛 edge（EV 闸门用）。
    - net_bps：扣除 maker 费 + 资金费后的净 edge（P0-5：回测判定必须以 net edge 为准，
      否则 +0.71bps < 单笔 maker+funding 成本(~2~5bps)，经济上不可交易）。
    """
    bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                       gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
    bt.run_signal(sig, path=forward)
    if bt.trades:
        t = bt.trades[-1]
        return t["gross_bps"], t["pnl_bps"]
    return 0.0, 0.0


def gen_all_signals(use_cache=True):
    """生成全量信号（串行，复用 P0 信号生成逻辑）。
    
    返回 [(timestamp, sym, sig, forward, regime, meta, gross_bps)]
    """
    if use_cache and os.path.exists(SIGNAL_CACHE):
        with open(SIGNAL_CACHE, "rb") as f:
            cached = pickle.load(f)
        print(f"从缓存加载 {len(cached)} 个信号")
        return cached
    
    items = []
    total = 0
    for s in SYMBOLS:
        k1h = hist[s]["1h"]; k4h = hist[s]["4h"]; k1w = hist[s]["1w"]; fng = hist[s].get("fng", {})
        n = len(k1h)
        if n < 240 + 48 + 1: continue
        for i in range(240, n - 48, 12):
            total += 1
            w1h = k1h[:i+1]
            ti = w1h[-1]["t"]
            w4h = [x for x in k4h if x["t"] <= ti]
            w1w = [x for x in k1w if x["t"] <= ti]
            price = w1h[-1]["c"]
            day = ti // 86400 * 86400
            fg = fng.get(day, 50)
            wk = w1w[-30:] if len(w1w) >= 30 else w1w
            wk_adx, wk_pdi, wk_mdi = (calc_adx(wk) if len(wk) >= 2 else (20.0, 20.0, 20.0))
            wk_dir = ("上涨" if wk_pdi > wk_mdi else "下跌" if wk_mdi > wk_pdi else "unknown")
            env = assess([c for c in w1h[-24:]], fg_val=fg)
            ti_ms = ti * 1000
            fr_now = _lookup(fr_ts[s], fr_v[s], ti_ms)
            fr_old = _lookup(fr_ts[s], fr_v[s], ti_ms - 3 * 86400 * 1000)
            oi_now = _lookup(oi_ts[s], oi_v[s], ti_ms)
            oi_old = _lookup(oi_ts[s], oi_v[s], ti_ms - 86400 * 1000)
            ctx = MarketContext(fg_val=fg, fr=fr_now, fr_delta=fr_now - fr_old,
                                oi_pct=(oi_now - oi_old) / oi_old if oi_old else 0.0,
                                wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir),
                                market_state=env.dominant,
                                regime=detect_regime([c["c"] for c in w1h]).regime)
            cand = gen_signal(s, w1h, w4h, w1w, ctx=ctx)
            if not cand.passed: continue
            regime = detect_regime([c["c"] for c in w1h]).regime
            adx_v = calc_adx(w1h)[0] if len(w1h) >= 15 else 20.0
            forward = [x["c"] for x in k1h[i+1:i+1+48]]
            if len(forward) < 5: continue
            meta = {"wk_dir": wk_dir, "adx": adx_v, "fng": fg, "direction": cand.direction,
                    "score": cand.score, "min_score_adj": cand.min_score_adj,
                    "fr": fr_now, "fr_delta": fr_now - fr_old,
                    "f2": calc_vol_price_divergence(w1h, n=24)}
            # 算 edge（直接从前向路径）
            sig = candidate_to_signal(cand, price)
            if sig is None: continue
            gross_bps, net_bps = compute_edge_bps(sig, forward)
            items.append((ti, s, sig, forward, regime, meta, gross_bps, net_bps))
        
        print(f"  {s}: {len([x for x in items if x[1]==s])} 信号")
    
    items.sort(key=lambda x: x[0])
    print(f"总共生成 {len(items)} 信号（扫了 {total} 窗口）")
    
    # 缓存
    with open(SIGNAL_CACHE, "wb") as f:
        pickle.dump(items, f)
    return items


# ============ BH-FDR ============
def bh_fdr(pvals, alpha=0.10):
    """Benjamini-Hochberg FDR 修正。"""
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
    # 调整后 p 值
    adj = [0.0] * m
    cur_min = 1.0
    for k in reversed(range(m)):
        raw = pvals[sorted_idx[k]]
        corrected = min(raw * m / (k + 1), 1.0)
        cur_min = min(cur_min, corrected)
        adj[sorted_idx[k]] = cur_min
    return adj, reject


# ============ T 检验（P2-3：正确 t 分布，非正态近似）============
def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 的连分式（Numerical Recipes Lentz 法）。"""
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 I_x(a,b)。"""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_p_value(t: float, df: int) -> float:
    """单侧 P(T_df >= t)，t>0。基于 I_x(df/2, 1/2) 关系。"""
    if t <= 0: return 1.0
    x = df / (df + t * t)
    return 0.5 * _betai(df / 2.0, 0.5, x)


def t_test_one_sided(vals, mu0=0.0):
    """单侧单样本 t 检验 H0: mean(vals) <= mu0 → p 值（正确 t 分布，P2-3）。"""
    n = len(vals)
    if n < 2: return 1.0
    mean = statistics.mean(vals)
    if mean <= mu0: return 1.0
    try:
        var = statistics.variance(vals, mean)
    except statistics.StatisticsError:
        return 1.0
    if var <= 0: return 1.0
    se = math.sqrt(var / n)
    t = (mean - mu0) / se
    df = n - 1
    return max(1e-12, min(_t_p_value(t, df), 1.0))


# ============ 主流程 ============
def main():
    print("=" * 72)
    print("P1a: 滚动 Walk-Forward × 条件 edge 一致性 + BH-FDR")
    print("=" * 72)
    
    items = gen_all_signals()
    print(f"\n总信号数: {len(items)}")
    print(f"时间范围: {items[0][0]} → {items[-1][0]} ({(items[-1][0]-items[0][0])/86400:.0f} 天)")
    
    # 汇总 edge 分布（P0-5：同时报告 gross 与 net，判定以 net 为准）
    all_gbp = [x[6] for x in items]
    all_nbp = [x[7] for x in items]
    print(f"全样本 gross_bps: mean={statistics.mean(all_gbp):+.2f} "
          f"median={statistics.median(all_gbp):+.2f} "
          f"min={min(all_gbp):+.2f} max={max(all_gbp):+.2f}")
    print(f"全样本 net_bps  : mean={statistics.mean(all_nbp):+.2f} "
          f"median={statistics.median(all_nbp):+.2f} "
          f"min={min(all_nbp):+.2f} max={max(all_nbp):+.2f}")
    print(f"  → 成本吞噬: gross 较 net 多 {statistics.mean(all_gbp)-statistics.mean(all_nbp):+.2f} bps/笔")

    conds = CONDITIONS
    cond_names = [c[0] for c in conds]

    # K 个非重叠折叠（P0-4：修复原 50% 重叠 → 相邻 fold OOS 落入下一 fold IS 致泄漏）
    K = 6
    fold = max(1, len(items) // K)                 # 非重叠块大小
    is_frac = 0.65

    # IS 用均值；OOS 逐笔入池（P1-5：逐笔显著性，拒绝 n=6 折叠均值 t 检验）
    cond_is_gbp  = {name: [] for name in cond_names}
    cond_is_nbp  = {name: [] for name in cond_names}
    cond_oos_gbp = {name: [] for name in cond_names}   # 每元素 = 该 fold OOS 逐笔 gross 列表
    cond_oos_nbp = {name: [] for name in cond_names}   # 每元素 = 该 fold OOS 逐笔 net 列表
    fold_records = []

    for fk in range(K):
        s0 = fk * fold
        s1 = (fk + 1) * fold if fk < K - 1 else len(items)   # 末块吃余数
        if s1 - s0 < 12:
            continue
        split_i = s0 + int((s1 - s0) * is_frac)
        is_sigs = items[s0:split_i]
        oos_sigs = items[split_i:s1]

        # 构建条件匹配 dict（含 gross/net 双 edge）
        def _features(ti, sym, sig, adj, regime, meta, gbp, nbp):
            return {**meta, "regime": regime, "gross_bps": gbp, "net_bps": nbp}

        is_feats = [_features(*x) for x in is_sigs]
        oos_feats = [_features(*x) for x in oos_sigs]

        fold_records.append((fk, s0, split_i, s1, len(is_feats), len(oos_feats),
                             is_sigs[0][0] if is_sigs else 0,
                             oos_sigs[-1][0] if oos_sigs else 0))

        for name, pred in conds:
            is_gbp = [f["gross_bps"] for f in is_feats if pred(f)]
            is_nbp = [f["net_bps"] for f in is_feats if pred(f)]
            oo_gbp = [f["gross_bps"] for f in oos_feats if pred(f)]
            oo_nbp = [f["net_bps"] for f in oos_feats if pred(f)]
            cond_is_gbp[name].append(statistics.mean(is_gbp) if is_gbp else None)
            cond_is_nbp[name].append(statistics.mean(is_nbp) if is_nbp else None)
            cond_oos_gbp[name].append(oo_gbp)
            cond_oos_nbp[name].append(oo_nbp)

    print(f"\n运行 {len(fold_records)} 个非重叠 fold（块大小≈{fold}）")
    for r in fold_records:
        fk, s0, s1, s2, n_is, n_oos, ts0, ts1 = r
        print(f"  Fold {fk}: sig[{s0}-{s1}-{s2}] IS={n_is} OOS={n_oos} ts={ts0}→{ts1}")

    # ===== 聚合分析（P0-5 net 为准；P1-5 逐笔显著性）=====
    print("\n" + "-" * 80)
    print(f"{'条件':22s} | {'IS净':>7} | {'OOS净':>8} | {'OOS笔':>6} | {'OOS>0':>6} | {'p值':>7} | FDR | 结论")
    print("-" * 80)

    # 逐条件：聚合所有 fold 的 OOS 逐笔净 edge，做单样本单侧 t 检验（P1-5）
    def _pool(name):
        return [v for lst in cond_oos_nbp[name] if lst for v in lst]

    pvals = []
    for name in cond_names:
        pool = _pool(name)
        pvals.append(t_test_one_sided(pool) if len(pool) >= 8 else 1.0)

    adj_pvals, rejects = bh_fdr(pvals, alpha=0.10)

    for i, name in enumerate(cond_names):
        pool = _pool(name)
        is_v  = [v for v in cond_is_nbp[name] if v is not None]
        mean_is = statistics.mean(is_v) if is_v else 0.0
        mean_oos = statistics.mean(pool) if pool else 0.0
        n_oos = len(pool)
        n_pos = sum(1 for v in pool if v > 0)

        pv = pvals[i]
        apv = adj_pvals[i]
        reject = rejects[i]

        is_positive  = mean_is > 0 and mean_oos > 0 and n_oos >= 30 and n_pos >= n_oos * 0.5
        any_positive = mean_oos > 0 and n_pos >= 1

        if reject and is_positive:
            tag = "✅正·通过FDR"
        elif reject and any_positive:
            tag = "🟡边缘·通过FDR"
        elif is_positive:
            tag = "🟡正·未过FDR"
        elif n_pos == 0:
            tag = "🔴全负"
        elif mean_oos <= 0:
            tag = "🔴净负"
        else:
            tag = "🟡微弱正"

        print(f"{name:22s} | {mean_is:+7.2f} | {mean_oos:+8.2f} | {n_oos:6d} | "
              f"{n_pos:6d} | {pv:>7.4f} | {apv:>6.4f} | {tag}")

    print("-" * 80)
    n_rej = sum(1 for r in rejects if r)
    print(f"BH-FDR(α=0.10) 通过: {n_rej}/{len(cond_names)} 个条件")

    # 最终裁决（以 net edge 为准）
    has_edge = sum(1 for i, name in enumerate(cond_names)
                   if rejects[i] and
                   statistics.mean(_pool(name) or [0]) > 0)

    print(f"\n{'='*80}")
    if has_edge >= 2:
        print(f"裁决: ✅ 发现 {has_edge} 个条件具有跨 fold 可持续正净 edge → 可接 Kelly + HMM")
    elif has_edge >= 1:
        print(f"裁决: 🟡 发现 {has_edge} 个边缘条件 → 需更多数据确认 → 建议扩面后重跑")
    else:
        print(f"裁决: ❌ 无跨 fold 可持续正净 edge → 当前 5 因子权重在加密市场为净负 → 建议因子发现")
        print(f"      (全样本 net_bps = {statistics.mean(all_nbp):+.2f}, gross_bps = {statistics.mean(all_gbp):+.2f}; 成本吞噬致净负)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
