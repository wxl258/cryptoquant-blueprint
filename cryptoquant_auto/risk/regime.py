"""Regime 检测（移植服务器 Regime 条件仓位 优点，接上 gate 的 regime_cap）。

基于价格序列波动率/回撤判定 TREND/RANGE/CRASH，喂给 gate.GateConfig.regime。
已实现：实时执行路径由 ExecutionEngine.update_regime(prices) 每帧驱动；
回测路径由 history.gen_real_signals 用真实1h收盘价判定（严格无未来函数）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RegimeResult:
    regime: str                    # TREND | RANGE | CRASH
    vol_ratio: float = 1.0
    max_dd: float = 0.0


def detect_regime(prices: List[float], win: int = 20) -> RegimeResult:
    """无数据时安全返回 TREND（与默认一致）。

    判定基于「近期最大回撤」为主、波动率为辅：
      - CRASH: 近期回撤 < -12% 且波动显著放大 → 禁开做空/做多半量
      - RANGE: 价格在区间内窄幅震荡（振幅 < 6% 且回撤平缓）→ 双向禁开
      - TREND: 其余（含持续单边市）→ 双向全开
    修复旧版「纯波动比」误判：单边跌因波动低被错判为 RANGE。
    """
    if len(prices) < win + 2:
        return RegimeResult("TREND")
    recent = prices[-win:]
    vol_recent = _std(recent)
    # 近期最大回撤（RANGE 判定用）：只看最近 win 棒，RANGE 须「近期平静」。
    peak = recent[0]
    max_dd_recent = 0.0
    for p in recent:
        peak = max(peak, p)
        max_dd_recent = min(max_dd_recent, p / peak - 1)
    # CRASH 判定回撤：用更长回看窗（≥60 棒 / 3×win），避免崩盘在 20 棒窗口内仅显 -10%
    # 而漏判 CRASH（崩盘回撤是累积量，须跨更长窗口度量，否则 fail-safe 不触发）。
    crash_win = max(win * 3, 60)
    dd_win = prices[-crash_win:] if len(prices) >= crash_win else prices
    peak = dd_win[0]
    max_dd = 0.0
    for p in dd_win:
        peak = max(peak, p)
        max_dd = min(max_dd, p / peak - 1)
    # 区间振幅（衡量震荡程度，不受单边趋势方向影响）
    lo = min(recent)
    hi = max(recent)
    rng = (hi - lo) / hi if hi > 0 else 0.0
    # 方向斜率（近期首末变化，识别单调趋势，避免缓跌被误判 RANGE）
    slope = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0.0
    mean_recent = sum(recent) / len(recent) if recent else 0.0
    rel_vol = vol_recent / mean_recent if mean_recent > 0 else 0.0
    # CRASH：深回撤(长窗) + 近期确实在跌(负斜率)或高波动 → 排除「古老峰值造成的历史回撤」
    if max_dd < -0.12 and (slope < -0.01 or rel_vol > 0.02):
        return RegimeResult("CRASH", vol_recent, max_dd)
    # RANGE 仅当「振幅窄 且 无方向 且 近期无深回撤」；用近期回撤(非长窗)判定
    if rng < 0.05 and abs(slope) < 0.02 and max_dd_recent > -0.05:
        return RegimeResult("RANGE", vol_recent, max_dd_recent)
    return RegimeResult("TREND", vol_recent, max_dd)


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return var ** 0.5
