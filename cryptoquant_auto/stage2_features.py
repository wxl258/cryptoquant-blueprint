"""阶段2 真实数据特征矩阵构建（零依赖，严格前视隔离）。

从 history_cache.json（真实 K 线）+ deriv_data.json（真实 fr/OI）构造
因果发现 / GP 进化用的特征矩阵。所有特征仅用「信号时刻之前」数据，无未来函数。

落点：阶段2 Task 12/13 共享数据管线。纯 numpy + 包内 indicators，无新依赖。
"""
from __future__ import annotations

import bisect
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .signals.indicators import calc_adx, calc_rsi, calc_atr, volatility_regime


# 候选因果变量（即「可能被发现为因」的特征白名单候选池）
FEATURE_NAMES = [
    "adx",        # 1h ADX 趋势强度（归一 0-1）
    "rsi",        # 1h RSI 超买超卖（归一 0-1）
    "atr_pct",    # ATR 占价百分比（归一 0-1）
    "vol_regime", # 波动率扩张信号（+1 扩张 / -1 收敛 / 0 平稳）
    "fr",         # 真实资金费率
    "fr_delta",   # 费率变化（3 日）
    "oi_pct",     # 持仓量变化%
    "fng",        # 恐慌贪婪指数（归一 0-1）
    "momentum",   # 12 根 log 动量
]
N_FEATURES = len(FEATURE_NAMES)


def _load_history() -> dict:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_cache.json")
    with open(p) as f:
        return json.load(f)


def _load_deriv() -> dict:
    base = os.path.dirname(os.path.abspath(__file__))
    cand = [
        os.path.join(base, "..", "deriv_data.json"),
        "/workspace/deriv_data.json",
        "/root/deriv_data.json",
    ]
    for p in cand:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                continue
    return {}


def _lookup(ts_arr: List[float], v_arr: List[float], ts: float) -> float:
    """按 ts 查最近 <= ts 的值（前视隔离，不取未来）。"""
    if not ts_arr:
        return 0.0
    i = bisect.bisect_right(ts_arr, ts) - 1
    return v_arr[i] if i >= 0 else 0.0


def _regime_of(closes: List[float]) -> str:
    """轻量 regime 判定（仅用历史收盘，无未来函数）。

    优先用 risk.regime.detect_regime；失败则退回简单波动率阈值分类。
    """
    try:
        from .risk.regime import detect_regime
        return detect_regime(closes).regime
    except Exception:
        if len(closes) < 30:
            return "unknown"
        rets = np.diff(np.log(np.array(closes, dtype=float)[-30:]))
        vol = float(np.std(rets))
        if vol > 0.03:
            return "CRASH"
        if vol < 0.008:
            return "RANGE"
        return "TREND"


def assemble_feature(closes: List[float], w: List[dict], fr: float, fr_delta: float,
                     oi_pct: float, fg: float, i: int) -> List[float]:
    """把单根窗口算成 9 维特征向量（FEATURE_NAMES 顺序）。

    抽取为单一函数，供 build_feature_matrix（历史回放）与 paper_runner（实时/最新窗口）
    共用，确保两条路径特征口径一致、无漂移。
    """
    adx, _, _ = calc_adx(w)
    rsi = calc_rsi(closes)
    atr, atr_pct = calc_atr(w)
    vreg, _ = volatility_regime(w)
    vreg_sig = 1.0 if vreg == "expanding" else (-1.0 if vreg == "contracting" else 0.0)
    mom = math.log(closes[-1] / closes[-13]) if i >= 12 and closes[-13] > 0 else 0.0
    return [adx / 100.0, rsi / 100.0, atr_pct / 100.0, vreg_sig,
            fr, fr_delta, oi_pct, fg / 100.0, mom]


def build_feature_matrix(symbols: Optional[List[str]] = None, step: int = 12,
                         warmup: int = 240, horizon: int = 12,
                         max_rows: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """构造阶段2 特征矩阵（严格前视隔离）。

    返回：
      X      : (n, N_FEATURES) 特征矩阵（仅用 i 之前数据）
      y      : (n,) 目标 = i→i+horizon 前向真实收益（OOS 预测对象）
      regimes: (n,) 每个样本生成时刻的 regime 标签（TREND/RANGE/CRASH/unknown）
      rows   : list[dict] 每样本元数据 {symbol,t,price,atr,fr,forward}，
               forward 为 i+1..i+horizon 真实收盘序列（供诚实回测/验证）
    """
    hist = _load_history()
    deriv = _load_deriv()
    symbols = symbols or list(hist.keys())
    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    regimes: List[str] = []
    rows: List[dict] = []

    for sym in symbols:
        if sym not in hist:
            continue
        k1h = hist[sym]["1h"]
        fng = hist[sym].get("fng", {})
        dseries = deriv.get(sym)
        fr_ts = fr_v = oi_ts = oi_v = ([], [], [], [])
        if dseries:
            fr = sorted(dseries.get("fr", []), key=lambda x: x[0])
            oi = sorted(dseries.get("oi", []), key=lambda x: x[0])
            # 去 OI 时间戳逆序（与 history.py 同款修复）
            oi = [oi[k] for k in range(len(oi))
                  if k == 0 or oi[k][0] >= oi[k - 1][0]]
            fr_ts = [x[0] for x in fr]; fr_v = [x[1] for x in fr]
            oi_ts = [x[0] for x in oi]; oi_v = [x[1] for x in oi]
        n = len(k1h)
        if n < warmup + horizon + 1:
            continue
        for i in range(warmup, n - horizon, step):
            w = k1h[:i + 1]
            closes = [c["c"] for c in w]
            ti = k1h[i]["t"]; ti_ms = ti * 1000
            day = ti // 86400 * 86400
            fg = fng.get(day, 50)
            if fr_ts:
                fi = bisect.bisect_right(fr_ts, ti_ms) - 1
                fr = fr_v[fi] if fi >= 0 else (fr_v[0] if fr_v else 0.0)
                fj = bisect.bisect_right(fr_ts, ti_ms - 3 * 86400 * 1000) - 1
                fr_old = fr_v[fj] if fj >= 0 else fr
                fr_delta = fr - fr_old
                oi_now = _lookup(oi_ts, oi_v, ti_ms)
                oi_old = _lookup(oi_ts, oi_v, ti_ms - 86400 * 1000)
                oi_pct = (oi_now - oi_old) / oi_old if oi_old else 0.0
            else:
                fr = 0.0; fr_delta = 0.0; oi_pct = 0.0
            feat = assemble_feature(closes, w, fr, fr_delta, oi_pct, fg, i)
            fwd = (k1h[i + horizon]["c"] - k1h[i]["c"]) / k1h[i]["c"]
            regime = _regime_of(closes)
            X_rows.append(feat)
            y_rows.append(fwd)
            regimes.append(regime)
            rows.append({
                "symbol": sym, "t": ti, "price": k1h[i]["c"], "atr": 0.0,
                "fr": fr, "forward": [c["c"] for c in k1h[i + 1:i + 1 + horizon]],
            })
            if max_rows and len(X_rows) >= max_rows:
                break
        if max_rows and len(X_rows) >= max_rows:
            break

    return (np.array(X_rows, dtype=float), np.array(y_rows, dtype=float),
            np.array(regimes), rows)
