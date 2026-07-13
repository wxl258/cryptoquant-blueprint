"""BinancePublicDataSource 离线解析自检（沙箱安全，不触外网）。

币安公网被 GFW 屏蔽，沙箱无法 live 跑；本测试用 monkeypatch 打桩
fapi/v1/klines、fundingRate、openInterestHist、alternative.me FNG 四个端点，
验证快照结构（feat 9 维 / regime / price / ts）与币安数组格式解析正确。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cryptoquant_auto.paper_runner import BinancePublicDataSource, FEATURE_NAMES


def _fake_klines(n: int = 250) -> list:
    """币安 klines 数组格式：[openTime(ms), o, h, l, c, v, closeTime(ms), ...]"""
    base = 1_700_000_000_000  # 某 ms 时间戳
    rows = []
    price = 30000.0
    for k in range(n):
        o = price
        c = price * (1 + 0.001 * math.sin(k / 7.0))
        h = max(o, c) * 1.0005
        l = min(o, c) * 0.9995
        v = 100.0 + 10 * math.cos(k / 3.0)
        rows.append([
            base + k * 3_600_000,      # openTime
            f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}",  # o h l c
            f"{v:.4f}",                # volume
            base + (k + 1) * 3_600_000,  # closeTime
            "0.0", "0", "0.0", "0.0", "0",   # 其余占位字段
        ])
        price = c
    return rows


_FAKE = {
    "klines": _fake_klines(250),
    "fr": [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": 123}],
    "oi": [
        {"symbol": "BTCUSDT", "sumOpenInterest": "100.0", "sumOpenInterestValue": "3e6", "timestamp": 1},
        {"symbol": "BTCUSDT", "sumOpenInterest": "110.0", "sumOpenInterestValue": "3.3e6", "timestamp": 2},
    ],
    "fng": {"data": [{"value": "72", "value_classification": "Greed"}]},
}


def _fake_get_json(url: str, timeout: float = 10.0):
    if "klines" in url:
        return _FAKE["klines"]
    if "fundingRate" in url:
        return _FAKE["fr"]
    if "openInterestHist" in url:
        return _FAKE["oi"]
    if "fng" in url:
        return _FAKE["fng"]
    raise AssertionError(f"未预期 URL: {url}")


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(BinancePublicDataSource, "_get_json", staticmethod(_fake_get_json))
    return BinancePublicDataSource(warmup=240, limit=500)


def test_snapshot_structure_and_parsing(patched):
    snap = patched.snapshot()
    # 12 个币全部应成功解析（mock 数据充足）
    assert set(snap.keys()) == set(patched.SYMBOLS)
    for sym, d in snap.items():
        assert isinstance(d["feat"], np.ndarray)
        assert d["feat"].shape == (len(FEATURE_NAMES),)  # 9 维
        assert isinstance(d["regime"], str) and d["regime"]
        assert isinstance(d["price"], float) and d["price"] > 0
        assert isinstance(d["ts"], int) and d["ts"] > 0


def test_klines_ms_timestamp_converted(patched):
    snap = patched.snapshot()
    d = snap["BTC"]
    # 最后一个 k 线的 openTime 应为 ms→s 转换后的秒级时间戳
    last = _FAKE["klines"][-1]
    assert d["ts"] == int(last[0]) // 1000


def test_funding_rate_and_oi_parsed(patched):
    snap = patched.snapshot()
    d = snap["BTC"]
    # fr 维度（FEATURE_NAMES 中索引 4）= fundingRate 值
    feat = dict(zip(FEATURE_NAMES, d["feat"]))
    assert feat["fr"] == pytest.approx(0.0001, rel=1e-6)
    # oi_pct 维度（索引 6）= (110-100)/100 = 0.1
    assert feat["oi_pct"] == pytest.approx(0.1, rel=1e-6)


def test_symbols_and_failsoft(patched):
    # symbols 列表完整
    assert len(patched.symbols()) == 12
