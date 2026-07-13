"""自包含技术指标（生产系统 indicators/all 的精简提取）。

仅保留信号引擎真正用到的 4 个指标，纯 numpy 实现，零外部依赖。
生产系统优点（P0-A/P0-B）的吸收落点之一。

Phase 1 (2026-07-12): 新增量价因子 F1-F4，利用 K 线 volume 字段（v）。
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional


def calc_adx(candles: List[dict], period: int = 14) -> Tuple[float, float, float]:
    """ADX / +DI / -DI。candles: [{h,l,c}]（按时间升序）。
    不足周期返回 (20,20,20) 视为中性。
    """
    if len(candles) < period + 1:
        return 20.0, 20.0, 20.0
    tr_l, pdm_l, mdm_l = [], [], []
    for i in range(1, len(candles)):
        pc, ph, pl = candles[i - 1]["c"], candles[i - 1]["h"], candles[i - 1]["l"]
        h, l, c = candles[i]["h"], candles[i]["l"], candles[i]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        up = max(h - candles[i - 1]["h"], 0.0)
        dn = max(candles[i - 1]["l"] - l, 0.0)
        tr_l.append(tr); pdm_l.append(up); mdm_l.append(dn)
    # Wilder 平滑
    atr = sum(tr_l[:period]) / period
    pdi = sum(pdm_l[:period]) / period
    mdi = sum(mdm_l[:period]) / period
    for i in range(period, len(tr_l)):
        atr = (atr * (period - 1) + tr_l[i]) / period
        pdi = (pdi * (period - 1) + pdm_l[i]) / period
        mdi = (mdi * (period - 1) + mdm_l[i]) / period
    if atr == 0:
        return 20.0, 20.0, 20.0
    pdi_n = 100 * pdi / atr
    mdi_n = 100 * mdi / atr
    dx = 100 * abs(pdi_n - mdi_n) / (pdi_n + mdi_n + 1e-9)
    adx = dx
    return round(adx, 1), round(pdi_n, 1), round(mdi_n, 1)


def calc_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0)); losses.append(max(-d, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0
    rs = ag / al
    return round(100 - 100 / (1 + rs), 1)


def calc_atr(candles: List[dict], period: int = 14) -> Tuple[float, float]:
    """返回 (ATR绝对值, ATR%)。candles: [{h,l,c}]。"""
    if len(candles) < 2:
        return 0.0, 0.0
    tr_l = []
    for i in range(1, len(candles)):
        pc = candles[i - 1]["c"]
        h, l, c = candles[i]["h"], candles[i]["l"], candles[i]["c"]
        tr_l.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(tr_l[:period]) / min(period, len(tr_l))
    for i in range(period, len(tr_l)):
        atr = (atr * (period - 1) + tr_l[i]) / period
    last_c = candles[-1]["c"]
    atr_pct = (atr / last_c * 100.0) if last_c else 0.0
    return round(atr, 4), round(atr_pct, 2)


def volatility_regime(candles: List[dict], period: int = 20) -> Tuple[str, str]:
    """波动率方向：contracting / expanding / stable（基于 ATR% 斜率）。"""
    if len(candles) < period * 2:
        return "stable", "样本不足"
    half = period
    a1 = sum((candles[i]["h"] - candles[i]["l"]) for i in range(len(candles) - half, len(candles))) / half
    a2 = sum((candles[i]["h"] - candles[i]["l"]) for i in range(len(candles) - 2 * half, len(candles) - half)) / half
    if a1 <= a2 * 0.85:
        return "contracting", f"ATR区间收窄({a1:.1f}<{a2:.1f})"
    if a1 >= a2 * 1.15:
        return "expanding", f"ATR区间扩张({a1:.1f}>{a2:.1f})"
    return "stable", "波动平稳"


# ============================================================================
# Phase 1: 量价因子 (Volume Factors)
# 利用 K 线 volume 字段 (v)，当前信号引擎完全未使用此维度。
# ============================================================================

def _sma(values: List[float], period: int) -> List[float]:
    """简单移动平均，返回与 values 同长的数组，不足 period 返回 0。"""
    out: List[float] = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(0.0)
        else:
            out.append(sum(values[i - period + 1:i + 1]) / period)
    return out


def _stdev(values: List[float], period: int) -> List[float]:
    """滚动标准差，返回与 values 同长。"""
    out: List[float] = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(0.0)
        else:
            seg = values[i - period + 1:i + 1]
            m = sum(seg) / period
            var = sum((x - m) ** 2 for x in seg) / period
            out.append(math.sqrt(var))
    return out


def _wilder_smooth(values: List[float], period: int) -> List[float]:
    """Wilder 平滑（RMA）。"""
    out: List[float] = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(0.0)
        elif i == period - 1:
            out.append(sum(values[:period]) / period)
        else:
            out.append((out[-1] * (period - 1) + values[i]) / period)
    return out


# ---- F1: OBV 趋势强度 ----
def calc_obv_adx(candles: List[dict], period: int = 14) -> Tuple[float, float, float]:
    """OBV 的 ADX 值（量能趋势强度）。

    对 OBV 序列计算 ADX，判断成交量趋势是否确立。
    - obv_adx >= 25: 量能趋势确立（量价同向信号可靠）
    - obv_adx >= 35: 量能趋势明确
    - obv_adx < 25: 量能无方向
    obv_pdi > obv_mdi: OBV 上升趋势；obv_mdi > obv_pdi: OBV 下降趋势。
    candles: [{h,l,c,v}]。
    """
    if len(candles) < period + 2:
        return 20.0, 20.0, 20.0
    obv = 0.0
    obv_series: List[float] = []
    for i in range(1, len(candles)):
        if candles[i]["c"] > candles[i - 1]["c"]:
            obv += candles[i].get("v", 0)
        elif candles[i]["c"] < candles[i - 1]["c"]:
            obv -= candles[i].get("v", 0)
        obv_series.append(obv)
    # 对 OBV 序列套 ADX 模板（用 h=l=obv 让 TR=0, 实际只用 c 方向）
    # 简化：直接用 Wilder 平滑估算 OBV 的上升/下降量
    if len(obv_series) < period + 1:
        return 20.0, 20.0, 20.0
    ups = [max(obv_series[i] - obv_series[i - 1], 0.0) for i in range(1, len(obv_series))]
    dns = [max(obv_series[i - 1] - obv_series[i], 0.0) for i in range(1, len(obv_series))]
    # Wilder 平滑上升/下降量
    au = _wilder_smooth(ups, period)
    ad = _wilder_smooth(dns, period)
    pdi_obv = au[-1] if au else 20.0
    mdi_obv = ad[-1] if ad else 20.0
    if pdi_obv + mdi_obv < 1e-9:
        return 20.0, 20.0, 20.0
    # DI 差归一化到 0-100
    dx_val = 100.0 * abs(pdi_obv - mdi_obv) / (pdi_obv + mdi_obv)
    # ADX 是对 DX 的 Wilder 平滑
    dx_series = [100.0 * abs(au[i] - ad[i]) / (au[i] + ad[i] + 1e-9)
                 for i in range(min(len(au), len(ad)))]
    adx_val = _wilder_smooth(dx_series, period)[-1] if len(dx_series) >= period else 20.0
    return round(adx_val, 1), round(pdi_obv, 1), round(mdi_obv, 1)


# ---- F2: 量价背离 (Price-Volume Divergence) ----
def calc_vol_price_divergence(candles: List[dict], n: int = 24) -> float:
    """量价背离信号。

    比较近 n 根 K 线的价格趋势与成交量趋势方向。
    返回: +1 看涨背离(价跌量增→反转做多), -1 看跌背离(价涨量缩→反转做空),
         0 无背离。
    """
    if len(candles) < n + 2:
        return 0.0
    closes = [c["c"] for c in candles[-n:]]
    volumes = [c.get("v", 0) for c in candles[-n:]]
    # 价格趋势: 线性回归斜率（简化: 首末方向）
    price_dir = 1 if closes[-1] > closes[0] else -1 if closes[-1] < closes[0] else 0
    vol_dir = 1 if volumes[-1] > volumes[0] else -1 if volumes[-1] < volumes[0] else 0
    # 背离: 价涨量缩=看跌(-1), 价跌量增=看涨(+1)
    if price_dir > 0 and vol_dir < 0:
        return -1.0  # 看跌背离
    if price_dir < 0 and vol_dir > 0:
        return +1.0  # 看涨背离
    return 0.0


# ---- F3: Volume Confirmation Ratio ----
def calc_vol_confirmation(candles: List[dict], period: int = 20,
                          threshold: float = 1.5) -> Tuple[int, float]:
    """成交量确认信号。

    当前成交量相对均值的倍数。放量 = 趋势可信。
    返回 (signal, ratio):
      signal=+1 放量上涨(做多确认)，-1 放量下跌(做空确认)，0 无量/缩量。
      ratio = 当前成交量 / 均量。
    """
    if len(candles) < period + 2:
        return 0, 0.0
    volumes = [c.get("v", 0) for c in candles]
    avg_vol = sum(volumes[-period - 1:-1]) / period
    current_vol = volumes[-1]
    ratio = current_vol / max(avg_vol, 1e-9)
    price_up = candles[-1]["c"] > candles[-2]["c"]
    if ratio >= threshold:
        return (1 if price_up else -1), round(ratio, 2)
    return 0, round(ratio, 2)


# ---- F4: VWAP 偏离 ----
def calc_vwap_deviation(candles: List[dict], lookback: int = 24) -> Tuple[float, float]:
    """VWAP 偏离（均值回归信号）。

    返回 (z_deviation, atr_norm_deviation):
      z_deviation: 当前价相对 VWAP 的 z-score 偏离。
      atr_norm_deviation: 以 ATR 归一化的偏离（单位: ATR）。
    当 |z_deviation| > 2.0 或 |atr_norm| > 2.0 时视为过度偏离→均值回归信号。
    """
    if len(candles) < lookback + 2:
        return 0.0, 0.0
    seg = candles[-lookback:]
    total_v = sum(c.get("v", 0) for c in seg) or 1e-9
    vwap = sum(c["c"] * c.get("v", 0) for c in seg) / total_v
    last_c = candles[-1]["c"]
    # z-score: (last - mean) / std of price
    closes = [c["c"] for c in seg]
    mean_c = sum(closes) / len(closes)
    var_c = sum((x - mean_c) ** 2 for x in closes) / len(closes)
    std_c = math.sqrt(var_c) or 1e-9
    z_dev = (last_c - vwap) / std_c
    # ATR 归一化
    atr, _ = calc_atr(candles)
    atr_norm = (last_c - vwap) / max(atr, 1e-9)
    return round(z_dev, 2), round(atr_norm, 2)
