"""因果发现特征筛选（Blueprint v0.3 · 专家C方案）。

用 Granger 因果检验 + 滚动窗口稳定性筛选，从现有 9 维特征池中选出
「对 1h 收益率具有稳定预报能力」的特征子集，剔除伪相关和噪声特征。

接口：
  - get_causal_features() -> List[str]   # 筛选后的特征名列表
  - discover() -> dict                     # 完整诊断结果

使用方式：
  from .causal_discovery import get_causal_features
  causal_feats = get_causal_features()

回退纪律：
  若 statsmodels 未装 / 数据不足 / 任何异常 → 返回全量特征（不卡管线）。
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stage2_features import assemble_feature, FEATURE_NAMES, _load_history, _load_deriv

logger = logging.getLogger("cryptoquant.causal")

# 持久化缓存路径（与 finmem 同目录）
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "causal_features.json")

# 默认参数
MAX_LAG = 12           # Granger 最大滞后（1h bar × 12 = 12h）
SIGNIFICANCE = 0.05    # p 值门槛
STABILITY_WIN = 10     # 滚动窗口数
STABILITY_RATIO = 0.5  # 通过窗口比例 ≥50% 才算稳定（服务器实测：momentum/vol_regime 5/10）


def _make_features_for_symbol(symbol: str) -> Tuple[np.ndarray, np.ndarray]:
    """对一个币构造完整的特征矩阵 X 与目标向量 y（下一 bar 收益率）。"""
    hist = _load_history()
    entry = hist.get(symbol)
    if not entry:
        return np.array([]), np.array([])
    k1h = entry.get("1h", [])
    if len(k1h) < 260:  # 至少需 ~260 根 bar（~11 天）
        return np.array([]), np.array([])
    deriv = _load_deriv()
    features: List[np.ndarray] = []
    targets: List[float] = []
    # 从 warmup 之后开始构建，确保有足够回看窗口
    warmup = 240
    for i in range(warmup, len(k1h) - 1):
        w = k1h[:i + 1]
        closes = [c["c"] for c in w]
        fr = 0.0
        d = deriv.get(symbol)
        if d:
            fr_list = sorted(d.get("fr", []), key=lambda x: x[0])
            if fr_list:
                fr = float(fr_list[-1][1])
            oi_list = sorted(d.get("oi", []), key=lambda x: x[0])
        fng_data = entry.get("fng", {})
        day = (k1h[i]["t"] // 86400) * 86400
        fg = fng_data.get(day, 50)
        feat = assemble_feature(closes, w, fr, 0.0, 0.0, fg, i)
        features.append(np.array(feat, dtype=float))
        # 目标：下一 bar 收益率（log return）
        if i + 1 < len(k1h):
            ret = math.log(k1h[i + 1]["c"] / k1h[i]["c"])
            targets.append(ret)
    if not features or len(features) < MAX_LAG + 10:
        return np.array([]), np.array([])
    return np.array(features), np.array(targets)


def _granger_pvalues(X: np.ndarray, y: np.ndarray,
                     max_lag: int = MAX_LAG) -> Dict[int, float]:
    """对单特征做 Granger 因果检验，返回各滞后阶数的最小 p 值。"""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {}
    # 拼接为 (y, X) 两列：检测 X → y
    data = np.column_stack([y, X])
    if data.shape[0] < max_lag * 3:
        return {}
    try:
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception:
        return {}
    pvals = {}
    for lag, r in result.items():
        # 使用 ssr 卡方检验的 p 值
        p = min(
            r[0].get("ssr_chi2test", (1, 1))[1],
            r[0].get("params_ftest", (1, 1))[1],
        )
        pvals[lag] = float(p)
    return pvals


def discover(symbol: str = "BTC", force: bool = False) -> dict:
    """对指定币做完整因果发现，返回诊断结果。

    结果格式：
      {
        "symbol": "BTC",
        "features": [...]        # 通过筛选的特征名
        "details": {feat: {best_lag, best_p, windows_pass, windows_total}}
      }
    """
    X_all, y = _make_features_for_symbol(symbol)
    if X_all.shape[0] == 0 or y.shape[0] == 0:
        logger.warning("数据不足，无法做因果发现（%s）", symbol)
        return {"symbol": symbol, "features": list(FEATURE_NAMES), "details": {}}

    n_feats = X_all.shape[1]
    details: Dict[str, dict] = {}
    stable_feats: List[str] = []

    # 滚动窗口稳定性检验
    win_size = max(X_all.shape[0] // STABILITY_WIN, MAX_LAG * 3)
    for fi in range(n_feats):
        feat_name = FEATURE_NAMES[fi]
        Xi = X_all[:, fi]
        windows_passed = 0
        best_lag = 0
        best_p = 1.0
        for wi in range(STABILITY_WIN):
            start = wi * win_size
            end = start + win_size
            if end > X_all.shape[0]:
                break
            Xw = Xi[start:end]
            yw = y[start:end]
            if len(Xw) < MAX_LAG * 3:
                continue
            pvals = _granger_pvalues(Xw, yw, MAX_LAG)
            if not pvals:
                continue
            min_p = min(pvals.values())
            if min_p < SIGNIFICANCE:
                windows_passed += 1
            if min_p < best_p:
                best_p = min_p
                best_lag = min(pvals, key=pvals.get)

        details[feat_name] = {
            "best_lag": int(best_lag),
            "best_p": round(float(best_p), 6),
            "windows_pass": windows_passed,
            "windows_total": STABILITY_WIN,
        }
        if windows_passed >= STABILITY_WIN * STABILITY_RATIO:
            stable_feats.append(feat_name)

    if not stable_feats:
        # 容错：一个都没通过就保留特征不削减
        stable_feats = list(FEATURE_NAMES)

    result = {"symbol": symbol, "features": stable_feats, "details": details}
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("因果发现完成: %s 稳定 %d/%d 特征", symbol, len(stable_feats), n_feats)
    return result


def get_causal_features(symbol: str = "BTC",
                        max_age: float = 86400 * 3) -> List[str]:
    """获取缓存的因果筛选后特征列表。

    缓存 > max_age（默认 3 天）自动重跑。
    任何异常 → 返回全量 FEATURE_NAMES（不卡管线）。
    """
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE) as f:
                data = json.load(f)
            age = time.time() - os.path.getmtime(_CACHE_FILE)
            if age < max_age and data.get("symbol") == symbol:
                feats = data.get("features", [])
                if feats:
                    return feats
        # 缓存过旧/缺失 → 跑发现
        result = discover(symbol)
        feats = result.get("features", [])
        return feats if feats else list(FEATURE_NAMES)
    except Exception as e:
        logger.warning("因果发现异常（回退全量特征）：%s", e)
        return list(FEATURE_NAMES)
