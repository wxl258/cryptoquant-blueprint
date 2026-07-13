"""黑天鹅检测（移植服务器 black_swan_state 优点）。

纯事件探测器，命中即强制 KillSwitch 进 L3（复用现有 black_swan 字段）。
Fail-closed：只探测、只告警/升级，绝不自动平仓。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Candle:
    symbol: str
    open: float
    close: float
    high: float
    low: float


def detect_black_swan(candles_1m: Dict[str, Candle], cross_spread_pct: float = 0.0,
                      btc_dump_pct: float = 0.05) -> (bool, str):
    """返回 (是否黑天鹅, 原因)。无数据时安全返回 (False, '')。"""
    if not candles_1m:
        return False, ""
    btc = candles_1m.get("BTC")
    if btc is not None and btc.open > 0:
        drop = (btc.open - btc.close) / btc.open
        if drop > btc_dump_pct:
            return True, f"btc_dump_{drop:.1%}"
    if cross_spread_pct > 0.015:   # 跨所价差飙升
        return True, f"spread_spike_{cross_spread_pct:.1%}"
    return False, ""


# ===== 生产系统优点吸收（P0-D）：分层连续跌幅检测 =====
def get_black_swan_level(btc_price_history: list) -> (int, str):
    """黑天鹅分层（吸收服务器 black_swan.get_black_swan_level）。

    用连续跌幅（非 max-min，避免假阳性）：
      - L3: 30分钟跌幅 ≥ 15%
      - L2: 15分钟跌幅 ≥ 8%
      - L1: 5分钟跌幅  ≥ 3%
    返回 (level 0-3, 原因)。history 为按时间升序的价格列表。
    """
    n = len(btc_price_history)
    if n < 2:
        return 0, ""
    # L3: 30分钟（取最近 ~30 个采样点，不足则全段）
    win = min(30, n)
    if win >= 2:
        start, end = btc_price_history[-win], btc_price_history[-1]
        if start > 0:
            d = (start - end) / start * 100
            if d >= 15:
                return 3, f"30m跌幅{d:.1f}%≥15% (L3)"
    # L2: 15分钟
    win = min(15, n)
    if win >= 2:
        start, end = btc_price_history[-win], btc_price_history[-1]
        if start > 0:
            d = (start - end) / start * 100
            if d >= 8:
                return 2, f"15m跌幅{d:.1f}%≥8% (L2)"
    # L1: 5分钟
    win = min(5, n)
    if win >= 2:
        start, end = btc_price_history[-win], btc_price_history[-1]
        if start > 0:
            d = (start - end) / start * 100
            if d >= 3:
                return 1, f"5m跌幅{d:.1f}%≥3% (L1)"
    return 0, ""
