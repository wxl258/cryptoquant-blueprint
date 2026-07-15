"""【P2-3】FG 按 00:00 UTC 发布时刻对齐边界的回归锁。

fetch_fng 必须把任意时间戳 floor 到 UTC 午夜，键与 gen_real_signals 中
day = ti // 86400 * 86400 同口径；否则信号日会与 FNG 日错位。
"""
from __future__ import annotations

import pytest

from cryptoquant_auto.history import fetch_fng

# 2025-07-14 00:00:00 UTC（真实 UTC 午夜，= 20283 × 86400）
DAY_A = 1752451200
# 2025-07-15 00:00:00 UTC（DAY_A + 86400）
DAY_B = 1752537600


def test_fetch_fng_aligns_to_utc_midnight(monkeypatch):
    # 非午夜时间戳(DAY_A + 3h30m)应 floor 到 DAY_A 午夜；精确午夜保持不变。
    raw = {"data": [
        {"timestamp": DAY_A + 3 * 3600 + 30 * 60, "value": 42},  # DAY_A 03:30 UTC
        {"timestamp": DAY_B, "value": 55},                        # DAY_B 00:00 UTC
    ]}
    monkeypatch.setattr("cryptoquant_auto.history._get_json", lambda url: raw)

    fng = fetch_fng(limit=0)
    assert fng.get(DAY_A) == 42, "非午夜时间戳未 floor 到 UTC 午夜"
    assert fng.get(DAY_B) == 55, "精确午夜键丢失"
    # 不应出现非午夜键
    assert all(k % 86400 == 0 for k in fng), "存在非 UTC 午夜的键"


def test_fetch_fng_empty_on_error(monkeypatch):
    def _boom(url):
        raise RuntimeError("network")
    monkeypatch.setattr("cryptoquant_auto.history._get_json", _boom)
    assert fetch_fng(limit=0) == {}, "FNG 拉取失败应返回空 dict（不抛）"
