"""各币行情路径生成（回测用）。

合成但按各币波动量级校准，用于驱动引擎成交、统计胜率/净值/回撤。
真实回测可替换为历史 K 线回放（同接口：iterable of (symbol->price)）。
"""
from __future__ import annotations

import random
from typing import Dict, List

# 每根 bar 的波动量级（合成校准）
COIN_VOL: Dict[str, float] = {"BTC": 0.012, "ETH": 0.016, "BNB": 0.020,
                              "SOL": 0.026, "XRP": 0.030}


def make_path(start: float, coin: str, n_bars: int, drift: float = 0.0,
              seed: int = None) -> List[float]:
    """生成单币价格路径（几何随机游走）。drift 为整段累计偏倚。"""
    rnd = random.Random(seed)
    vol = COIN_VOL.get(coin, 0.02)
    prices = [start]
    per_bar_drift = drift / n_bars
    for _ in range(n_bars):
        ret = per_bar_drift + vol * rnd.gauss(0, 1)
        prices.append(prices[-1] * (1 + ret))
    return prices


def path_until_exit(start: float, coin: str, entry: float, sl: float, tp1: float,
                    tp2: float, direction: str, max_bars: int = 500,
                    seed: int = None) -> List[float]:
    """生成路径直到触发 entry/sl/tp 之一（用于单笔信号回放）。"""
    rnd = random.Random(seed)
    vol = COIN_VOL.get(coin, 0.02)
    prices = [start]
    for _ in range(max_bars):
        ret = vol * rnd.gauss(0, 1)
        p = prices[-1] * (1 + ret)
        prices.append(p)
        if direction == "BUY":
            if p <= sl or p >= tp1:
                break
        else:
            if p >= sl or p <= tp1:
                break
    return prices
