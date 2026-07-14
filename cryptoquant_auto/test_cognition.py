"""P1-21 回归：元认知环境评估 + 持久化根与 memory/reflection 一致。"""
import os

import pytest

from cryptoquant_auto.meta import cognition
from cryptoquant_auto.meta.cognition import assess, EnvRecord, record_env, load_env_history


def test_assess_returns_dominant_env():
    # 上涨趋势 K 线 → 偏 BULL
    closes = [{"c": 100 + i} for i in range(30)]
    out = assess(closes, fg_val=50.0)
    assert out.dominant in ("BULL", "BEAR", "RANGE")
    assert 0.0 <= out.confidence <= 1.0


def test_assess_extreme_fear_bear():
    closes = [{"c": 100 - i * 0.5} for i in range(30)]
    out = assess(closes, fg_val=10.0)  # 极端恐惧
    assert out.dominant in ("BEAR", "RANGE")


def test_persistence_root_is_package_dir():
    # P1-21：cognition 基目录必须与 memory/reflection 一致（包目录）
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(cognition.__file__)))  # cryptoquant_auto
    assert os.path.dirname(cognition.ENV_HIST_FILE) == os.path.join(pkg, "data")


def test_env_history_roundtrip(tmp_path, monkeypatch):
    # 重定向 ENV_HIST_FILE 到临时目录（父目录须存在），验证写入/读取不静默失败
    target = tmp_path / "env_history.json"
    monkeypatch.setattr(cognition, "ENV_HIST_FILE", str(target))
    record_env(EnvRecord(ts=1.0, env="BULL", confidence=0.7,
                          btc_change=2.0, adx=25.0, atr_pct=2.0))
    hist = load_env_history()
    assert isinstance(hist, list) and len(hist) >= 1
    assert hist[-1]["env"] == "BULL"
