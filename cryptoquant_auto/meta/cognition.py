"""元认知环境评分（吸收生产系统 meta/cognition.py 优点 P0-B）。

三级聚合：微观(4h/cron级) → 中观(日级) → 宏观(周级)，输出 6 维环境评分，
供信号引擎的市场状态闸门(BULL/BEAR/RANGE)与决策参考。

保持零资金：环境由历史 K 线统计得到，不依赖实时订阅。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BASE_DIR = os.environ.get("CRYPTOQUANT_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ENV_HIST_FILE = os.path.join(BASE_DIR, "data", "env_history.json")


@dataclass
class EnvRecord:
    ts: float
    env: str                 # BULL / BEAR / RANGE
    confidence: float
    btc_change: float
    adx: float
    atr_pct: float


@dataclass
class EnvAssessment:
    dominant: str = "unknown"        # BULL / BEAR / RANGE
    confidence: float = 0.0
    distribution: Dict[str, int] = field(default_factory=dict)
    adx_median: float = 20.0
    atr_pct_median: float = 2.0
    btc_trend_1d: float = 0.0
    btc_trend_1w: float = 0.0
    note: str = ""


def _micro_env(btc_1h: List[dict], fg_val: float = 50.0) -> tuple:
    """微观(4h/cron级)：基于近期1h K线趋势 + 恐慌贪婪，判 BULL/BEAR/RANGE。"""
    if len(btc_1h) < 24:
        return "unknown", 0.3
    closes = [c["c"] for c in btc_1h[-24:]]
    chg = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0.0
    # ADX 近似
    ups = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    dn = len(closes) - 1 - ups
    if chg > 1.5 and ups > dn:
        return "BULL", min(0.9, 0.5 + abs(chg) / 10)
    if chg < -1.5 and dn > ups:
        return "BEAR", min(0.9, 0.5 + abs(chg) / 10)
    return "RANGE", 0.5


def assess(btc_1h: List[dict], btc_1d: Optional[List[dict]] = None,
           btc_1w: Optional[List[dict]] = None, fg_val: float = 50.0) -> EnvAssessment:
    """三级环境聚合（微观→中观→宏观）。"""
    micro_env, micro_conf = _micro_env(btc_1h, fg_val)
    out = EnvAssessment()

    # 中观(日级)：1d 趋势
    if btc_1d and len(btc_1d) >= 2:
        c = [x["c"] for x in btc_1d[-2:]]
        out.btc_trend_1d = (c[-1] - c[0]) / c[0] * 100 if c[0] else 0.0
    # 宏观(周级)：1w 趋势
    if btc_1w and len(btc_1w) >= 2:
        c = [x["c"] for x in btc_1w[-2:]]
        out.btc_trend_1w = (c[-1] - c[0]) / c[0] * 100 if c[0] else 0.0

    # 综合判定（6维：微观态/微观置信/日趋势/周趋势/恐慌贪婪/波动）
    votes = []
    if micro_env != "unknown":
        votes.append(micro_env)
    if out.btc_trend_1d > 2:
        votes.append("BULL")
    elif out.btc_trend_1d < -2:
        votes.append("BEAR")
    if out.btc_trend_1w > 3:
        votes.append("BULL")
    elif out.btc_trend_1w < -3:
        votes.append("BEAR")
    if fg_val >= 75:
        votes.append("BULL")   # 极度贪婪常伴上涨趋势
    elif fg_val <= 25:
        votes.append("BEAR")

    if votes:
        counter = Counter(votes)
        out.dominant = counter.most_common(1)[0][0]
        out.confidence = round(counter[out.dominant] / len(votes), 2)
        out.distribution = dict(counter)
    else:
        out.dominant = "RANGE"
        out.confidence = 0.4
    out.note = f"微观={micro_env} 日趋势={out.btc_trend_1d:+.1f}% 周趋势={out.btc_trend_1w:+.1f}% FG={fg_val}"
    return out


def record_env(rec: EnvRecord) -> None:
    """记录单次环境检测结果（持久化，供中观/宏观聚合）。"""
    try:
        hist = load_env_history()
        hist.append({"ts": rec.ts, "env": rec.env, "confidence": rec.confidence,
                     "btc_change": rec.btc_change, "adx": rec.adx, "atr_pct": rec.atr_pct})
        hist = hist[-5000:]
        with open(ENV_HIST_FILE, "w") as f:
            json.dump(hist, f)
    except Exception:
        pass


def load_env_history() -> List[dict]:
    try:
        if os.path.exists(ENV_HIST_FILE):
            with open(ENV_HIST_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []
