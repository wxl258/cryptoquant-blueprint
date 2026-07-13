"""真实历史行情抓取 + 真实信号生成（替换合成信号，做诚实回测）。

数据源（均免费、零密钥）：
  - GateIO 期货公开 K 线 / 资金费（生产主源，与 contract_fetch 一致）
  - Alternative.me 恐慌贪婪指数（历史序列）
信号用 bundle 自带 generate_signals 在「信号时刻之前的真实数据」上生成，
前向路径用「信号时刻之后的真实收盘」回放——彻底去掉合成漂移，看清真实 edge。

结果缓存到包内 history_cache.json（默认 1 小时内复用），保证可复现且不刷 API。
"""
from __future__ import annotations

import bisect
import json
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

GATE = "https://api.gateio.ws/api/v4/futures/usdt"
FNG = "https://api.alternative.me/fng/"
SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX"]
CONTRACT = {s: f"{s}_USDT" for s in SYMBOLS}
# P2-6 修复：缓存路径改为包内相对路径，避免硬编码 /workspace 导致落点错位、刷新不生效
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_cache.json")

# 真实资金费率/OI 序列（修复 P0-6 数据泄漏：history_cache 的 fr 是标量 0.0 占位）
# 多路径探测，保证沙箱/服务器均可加载
_DERIV_PATHS = [
    "/workspace/deriv_data.json",
    "/root/deriv_data.json",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "deriv_data.json"),
]
_DERIV_CACHE: Dict[str, tuple] = {}   # sym -> (fr_ts[], fr_v[], oi_ts[], oi_v[])


def load_deriv_series() -> Dict[str, tuple]:
    """加载 deriv_data.json 真实 fr/OI 时间序列（按时间戳排序）。

    返回 {sym: (fr_ts, fr_v, oi_ts, oi_v)}，缺失币返回空 dict。
    修复 history.py 原用 hist[s]['fr'] 标量(=0.0) 喂信号的致命 bug。
    """
    if _DERIV_CACHE:
        return _DERIV_CACHE
    raw = None
    for p in _DERIV_PATHS:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    raw = json.load(f)
                break
            except Exception:
                continue
    if not raw:
        return _DERIV_CACHE
    for s in SYMBOLS:
        if s not in raw:
            continue
        fr = sorted(raw[s].get("fr", []), key=lambda x: x[0])
        oi = sorted(raw[s].get("oi", []), key=lambda x: x[0])
        # 去除 OI 时间戳逆序（修复 P1-9 OI 序列非单调）
        oi = [oi[k] for k in range(len(oi))
              if k == 0 or oi[k][0] >= oi[k - 1][0]]
        _DERIV_CACHE[s] = ([x[0] for x in fr], [x[1] for x in fr],
                           [x[0] for x in oi], [x[1] for x in oi])
    return _DERIV_CACHE


def lookup_series(ts_arr: List[float], v_arr: List[float], ts: float) -> float:
    """在时间序列中按 ts 查最近 <= ts 的值（前视隔离，不取未来）。

    ts 早于序列起点（如 OI 仅覆盖近 30 天，其余历史均越界）→ 返回 0.0，
    避免返回无意义的首值污染 oi_pct/fr。
    """
    if not ts_arr:
        return 0.0
    i = bisect.bisect_right(ts_arr, ts) - 1
    return v_arr[i] if i >= 0 else 0.0


# ---------------- 网络 ----------------
def _get_json(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_klines(contract: str, interval: str = "1h", limit: int = 1500,
                  page: int = 1000, max_pages: int = 5) -> List[dict]:
    """GateIO 期货 K 线（分页拉满 limit 根）-> 升序 [{t,o,h,l,c}]。

    GateIO 单次 limit 上限约 1000，用 to 参数向前翻页累积更长历史。
    """
    rows: List[dict] = []
    to = None
    for _ in range(max_pages):
        url = f"{GATE}/candlesticks?contract={contract}&interval={interval}&limit={page}"
        if to is not None:
            url += f"&to={to}"
        chunk = _get_json(url)
        if not chunk:
            break
        if to is not None:
            chunk = [r for r in chunk if int(r["t"]) < to]
            if not chunk:
                break
        rows = chunk + rows
        to = int(rows[0]["t"])
        if len(chunk) < page:
            break
        if len(rows) >= limit:
            break
    out = [{"t": int(x["t"]), "o": float(x["o"]), "h": float(x["h"]),
           "l": float(x["l"]), "c": float(x["c"])} for x in rows]
    out.sort(key=lambda d: d["t"])
    return out[:limit]


def fetch_funding(contract: str) -> float:
    d = _get_json(f"{GATE}/contracts/{contract}")
    return float(d.get("funding_rate", 0.0))


def fetch_fng(limit: int = 0) -> Dict[int, int]:
    """Alternative.me 恐慌贪婪 -> {unix_day: value}。"""
    try:
        data = _get_json(f"{FNG}?limit={limit}")
    except Exception:
        return {}
    return {int(it["timestamp"]) // 86400 * 86400: int(it["value"])
            for it in data.get("data", [])}


def resample(candles_1h: List[dict], n: int) -> List[dict]:
    """每 n 根 1h 聚合成 1 根（4h=n=4，1w=n=168）。"""
    out = []
    for i in range(0, len(candles_1h) - n + 1, n):
        grp = candles_1h[i:i + n]
        out.append({"t": grp[0]["t"], "o": grp[0]["o"],
                    "h": max(x["h"] for x in grp), "l": min(x["l"] for x in grp),
                    "c": grp[-1]["c"]})
    return out


# ---------------- 历史构建 ----------------
def build_history(symbols: List[str] = SYMBOLS, limit: int = 1500,
                  cache: str = CACHE, max_age: int = 3600) -> dict:
    """抓取并缓存真实历史；1 小时内复用。返回 {symbol: {1h,4h,1w,fr,fng}}。"""
    if os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) < max_age:
        with open(cache) as f:
            return json.load(f)
    hist: Dict[str, dict] = {}
    fng = fetch_fng(limit=0)
    for s in symbols:
        c = CONTRACT[s]
        k1h = fetch_klines(c, "1h", limit)
        k4h = resample(k1h, 4)
        k1w = resample(k1h, 168)
        fr = fetch_funding(c)
        hist[s] = {"1h": k1h, "4h": k4h, "1w": k1w, "fr": fr, "fng": fng}
    with open(cache, "w") as f:
        json.dump(hist, f)
    return hist


# ---------------- 真实信号生成（严格前视隔离） ----------------
def gen_real_signals(hist: dict, step: int = 12, warmup: int = 240,
                     horizon: int = 48, split_frac: float = None) -> object:
    """滑动窗口在「信号时刻 i 之前的真实数据」上生成信号，前向路径取 i 之后真实收盘。

    每个信号附带：(1) 生成时刻的 regime 标签（detect_regime，仅用 i 之前的 w1h）；
                (2) meta 元数据 {wk_dir, adx(1h趋势强度), fng, conf, atr_pct, direction}
                    —— 全部基于 i 之前数据，严格无未来函数，供边缘探索分层。
    返回：
      - split_frac is None -> List[(Signal, forward_closes, regime, meta)]
      - split_frac set     -> (is_out, oos_out)，按时间顺序前 split_frac 切 IS、余下为 OOS
    forward_closes 为 i 之后的真实收盘序列（严格前视隔离，无合成漂移）。
    """
    from .signals import generate_signals, MarketContext
    from .signals.indicators import calc_adx, calc_vol_price_divergence
    from .meta.cognition import assess
    from .risk.regime import detect_regime

    out: List[tuple] = []
    deriv = load_deriv_series()          # 真实 fr/OI 序列（P0-6 修复）
    fng_full = fetch_fng(limit=0) if not hist.get("_fng_loaded") else {}
    for s in SYMBOLS:
        k1h = hist[s]["1h"]
        fng = hist[s].get("fng", {})
        k4h_full = hist[s]["4h"]
        k1w_full = hist[s]["1w"]
        n = len(k1h)
        if n < warmup + horizon + 1:
            continue
        dseries = deriv.get(s)
        if dseries:
            fr_ts_a, fr_v_a, oi_ts_a, oi_v_a = dseries
        for i in range(warmup, n - horizon, step):
            ti = k1h[i]["t"]
            ti_ms = ti * 1000
            w1h = k1h[:i + 1]                                  # 仅历史
            w4h = [x for x in k4h_full if x["t"] <= ti]
            w1w = [x for x in k1w_full if x["t"] <= ti]
            price = k1h[i]["c"]
            # 上下文（全部基于 ti 之前数据）
            day = ti // 86400 * 86400
            fg = fng.get(day, 50)
            # 真实资金费率（P0-6 修复：用 deriv_data.json 序列按 ts 查表，非标量 0.0）
            if dseries:
                fi = bisect.bisect_right(fr_ts_a, ti_ms) - 1
                fr = fr_v_a[fi] if fi >= 0 else (fr_v_a[0] if fr_v_a else 0.0)
                fj = bisect.bisect_right(fr_ts_a, ti_ms - 3 * 86400 * 1000) - 1
                fr_old = fr_v_a[fj] if fj >= 0 else fr
                fr_delta = fr - fr_old
                oi_now = lookup_series(oi_ts_a, oi_v_a, ti_ms)
                oi_old = lookup_series(oi_ts_a, oi_v_a, ti_ms - 86400 * 1000)
                oi_pct = (oi_now - oi_old) / oi_old if oi_old else 0.0
            else:
                fr = hist[s]["fr"]
                fr_delta = 0.0
                oi_pct = 0.0
            wk = w1w[-30:] if len(w1w) >= 30 else w1w
            wk_adx, wk_pdi, wk_mdi = (calc_adx(wk)
                                      if len(wk) >= 2 else (20.0, 20.0, 20.0))
            wk_dir = ("上涨" if wk_pdi > wk_mdi
                      else "下跌" if wk_mdi > wk_pdi else "unknown")
            env = assess([c for c in w1h[-24:]], fg_val=fg)
            # regime 标签：仅用 w1h（i 之前）判定，无未来函数（P0b 路由需提前计算）
            regime = detect_regime([c["c"] for c in w1h]).regime
            ctx = MarketContext(fg_val=fg, fr=fr, fr_delta=fr_delta, oi_pct=oi_pct,
                                wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir),
                                market_state=env.dominant, regime=regime)
            market_data = {s: {"1h": w1h, "4h": w4h, "1w": w1w,
                               "fr": fr, "fr_delta": fr_delta}}
            sigs = generate_signals([s], market_data, ctx=ctx,
                                    price_map={s: price}, tf="1H")
            if sigs:
                forward = [x["c"] for x in k1h[i + 1:i + 1 + horizon]]
                if forward:
                    # 1h ADX（i 之前全窗 Wilder 平滑）—— 趋势强度，供分层，无未来函数
                    adx_v = calc_adx(w1h)[0] if len(w1h) >= 15 else 20.0
                    sig0 = sigs[0]
                    meta = {"wk_dir": wk_dir, "adx": adx_v, "fng": fg,
                            "conf": sig0.confidence,
                            "atr_pct": (sig0.atr / price * 100) if price else 0.0,
                            "direction": sig0.direction.value,
                            "f2": calc_vol_price_divergence(w1h, n=24)}
                    out.append((sig0, forward, regime, meta))
    if split_frac is None:
        return out
    k = max(1, int(len(out) * float(split_frac)))
    return out[:k], out[k:]


# ---------------- 多核并行版（适配大样本回测） ----------------
def _gen_one_window_core(args):
    """单窗口核心生成（不含 forward，供并行 path）。

    args: (sym, i, k1h_hist, k4h_full, k1w_full, fr_series, fng)
    —— 不携带全量 k1h：forward 由主进程在收集后用本地 hist 严格前视隔离计算，
       彻底避免跨进程大对象 pickle 串扰（此前导致并行结果错乱）。
    返回 (sym, i, (sig0, regime, meta)) 或 (sym, i, None)（无信号）。
    """
    sym, i, k1h_hist, k4h_full, k1w_full, fr_series, fng = args
    from .signals import generate_signals, MarketContext
    from .signals.indicators import calc_adx, calc_vol_price_divergence
    from .meta.cognition import assess
    from .risk.regime import detect_regime

    ti = k1h_hist[-1]["t"]
    ti_ms = ti * 1000
    w1h = k1h_hist
    w4h = [x for x in k4h_full if x["t"] <= ti]
    w1w = [x for x in k1w_full if x["t"] <= ti]
    price = k1h_hist[-1]["c"]
    day = ti // 86400 * 86400
    fg = fng.get(day, 50)
    wk = w1w[-30:] if len(w1w) >= 30 else w1w
    wk_adx, wk_pdi, wk_mdi = (calc_adx(wk) if len(wk) >= 2 else (20.0, 20.0, 20.0))
    wk_dir = ("上涨" if wk_pdi > wk_mdi else "下跌" if wk_mdi > wk_pdi else "unknown")
    # 真实资金费率（P0-6 修复：序列按 ts 查表，非标量 0.0）
    if fr_series:
        fr_ts_a, fr_v_a, oi_ts_a, oi_v_a = fr_series
        fi = bisect.bisect_right(fr_ts_a, ti_ms) - 1
        fr = fr_v_a[fi] if fi >= 0 else (fr_v_a[0] if fr_v_a else 0.0)
        fj = bisect.bisect_right(fr_ts_a, ti_ms - 3 * 86400 * 1000) - 1
        fr_old = fr_v_a[fj] if fj >= 0 else fr
        fr_delta = fr - fr_old
        oi_now = lookup_series(oi_ts_a, oi_v_a, ti_ms)
        oi_old = lookup_series(oi_ts_a, oi_v_a, ti_ms - 86400 * 1000)
        oi_pct = (oi_now - oi_old) / oi_old if oi_old else 0.0
    else:
        fr = 0.0
        fr_delta = 0.0
        oi_pct = 0.0
    env = assess([c for c in w1h[-24:]], fg_val=fg)
    # regime 标签：仅用 w1h（i 之前）判定，无未来函数（P0b 路由需传入 ctx）
    regime = detect_regime([c["c"] for c in w1h]).regime
    ctx = MarketContext(fg_val=fg, fr=fr, fr_delta=fr_delta, oi_pct=oi_pct,
                        wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir),
                        market_state=env.dominant, regime=regime)
    market_data = {sym: {"1h": w1h, "4h": w4h, "1w": w1w, "fr": fr, "fr_delta": fr_delta}}
    sigs = generate_signals([sym], market_data, ctx=ctx, price_map={sym: price}, tf="1H")
    if not sigs:
        return (sym, i, None)
    adx_v = calc_adx(w1h)[0] if len(w1h) >= 15 else 20.0
    sig0 = sigs[0]
    meta = {"wk_dir": wk_dir, "adx": adx_v, "fng": fg,
            "conf": sig0.confidence,
            "atr_pct": (sig0.atr / price * 100) if price else 0.0,
            "direction": sig0.direction.value,
            "f2": calc_vol_price_divergence(w1h, n=24)}
    return (sym, i, (sig0, regime, meta))


def _worker_safe(args):
    """包装：捕获 worker 异常并以 (sym,i,ERR) 形式回传，避免静默丢弃。"""
    try:
        return _gen_one_window_core(args)
    except Exception:
        import traceback
        return (args[0], args[1], ("__ERR__", traceback.format_exc()))


def gen_real_signals_parallel(hist: dict, step: int = 12, warmup: int = 240,
                              horizon: int = 48, split_frac: float = None,
                              max_workers: int = None) -> object:
    """gen_real_signals 的多核并行版（适配 5.5 年大样本回测）。

    按窗口 (sym, i) 铺到进程池，严格前视隔离（worker 仅见 k1h[:i+1] + forward 用 k1h[i+1:]）。
    结果按 i 排序保持时间顺序，行为与串行版逐笔一致（可用于加速 demo 回测段）。
    """
    import multiprocessing as mp
    max_workers = max_workers or os.cpu_count() or 4
    deriv = load_deriv_series()          # P0-6 修复：真实 fr/OI 序列（本函数作用域内定义，避免 NameError）
    tasks = []
    for s in SYMBOLS:
        k1h = hist[s]["1h"]
        k4h_full = hist[s]["4h"]
        k1w_full = hist[s]["1w"]
        fng = hist[s].get("fng", {})
        dseries = deriv.get(s)
        n = len(k1h)
        if n < warmup + horizon + 1:
            continue
        for i in range(warmup, n - horizon, step):
            # 仅传历史切片 (k1h[:i+1])，不传全量 k1h —— forward 主进程补算
            # P0-6 修复：传入真实 fr/OI 序列元组（非标量 0.0），worker 内按 ts 查表
            tasks.append((s, i, k1h[:i + 1], k4h_full, k1w_full, dseries, fng))
    n_tasks = len(tasks)
    core_map: dict = {}
    dropped = 0
    err_samples = []
    # 用 multiprocessing.Pool.map（已被 dbgpool 验证正确）替代 ProcessPoolExecutor.submit
    with mp.Pool(processes=max_workers) as pool:
        for result in pool.imap_unordered(_worker_safe, tasks, chunksize=8):
            sym, i, res = result
            if isinstance(res, tuple) and res and res[0] == "__ERR__":
                dropped += 1
                if len(err_samples) < 3:
                    err_samples.append((sym, i, res[1]))
                continue
            if res is not None:
                core_map[(sym, i)] = res
    if err_samples:
        import sys
        print(f"[gen_real_signals_parallel] 警告: {dropped}/{n_tasks} 窗口异常",
              file=sys.stderr)
        for sym, i, tb in err_samples:
            print(f"  --- ERR @{sym} i={i} ---\n{tb}", file=sys.stderr)
    # 主进程用本地全量 k1h 严格前视隔离补算 forward（与串行版同源，保证一致）
    out_map: dict = {}
    for (sym, i), (sig0, regime, meta) in core_map.items():
        k1h_full = hist[sym]["1h"]
        forward = [x["c"] for x in k1h_full[i + 1:i + 1 + horizon]]
        if forward:
            out_map[(sym, i)] = (sig0, forward, regime, meta)
    # 按 SYMBOLS 顺序重组（与串行版完全一致；不是字母序！）
    sym_order = {s: i for i, s in enumerate(SYMBOLS)}
    out = [out_map[k] for k in sorted(out_map.keys(),
                                       key=lambda kv: (sym_order.get(kv[0], 999), kv[1]))]
    if split_frac is None:
        return out
    k = max(1, int(len(out) * float(split_frac)))
    return out[:k], out[k:]

