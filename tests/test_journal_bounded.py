"""paper_journal.jsonl 有界化回收单测（防 5min cron 长期运行撑满磁盘）。

复现场景：审计日志 append-only 长期无限增长。验证 _write_outputs 在超过
JOURNAL_MAX_BYTES 阈值时仅保留最近 JOURNAL_KEEP 行，且在阈值内不误截。
"""
from __future__ import annotations

import json

import pytest

import cryptoquant_auto.paper_runner as pr


def _full_record(symbol: str) -> dict:
    """凑齐 _write_dashboard 需要的字段，避免仪表盘渲染报错。"""
    return {
        "symbol": symbol, "regime": "RANGE", "market_state": "RANGE",
        "action": "HOLD", "confidence": 0.5, "rationale": ["x"],
        "vetoes": [], "constitution_compliant": True, "violations": [],
        "llm": "mock", "price": 1.0, "ts": 0,
    }


def test_no_trim_when_under_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "PAPER_DIR", str(tmp_path))
    monkeypatch.setattr(pr, "JOURNAL_MAX_BYTES", 10 ** 9)   # 阈值极高 → 不触发
    monkeypatch.setattr(pr, "JOURNAL_KEEP", 10 ** 9)
    jp = tmp_path / "paper_journal.jsonl"
    jp.write_text("\n".join(json.dumps({"n": i}) for i in range(5)) + "\n")
    pr._write_outputs([_full_record("BTC")])
    lines = jp.read_text().splitlines()
    assert len(lines) == 6          # 5 旧 + 1 新，未被截断
    assert json.loads(lines[-1])["symbol"] == "BTC"


def test_trim_keeps_recent_when_over_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "PAPER_DIR", str(tmp_path))
    monkeypatch.setattr(pr, "JOURNAL_MAX_BYTES", 1000)   # 1KB 阈值
    monkeypatch.setattr(pr, "JOURNAL_KEEP", 50)
    jp = tmp_path / "paper_journal.jsonl"
    # 预填 200 行（每行 ~10B → ~2KB，超过 1KB 阈值）
    jp.write_text("\n".join(json.dumps({"n": i}) for i in range(200)) + "\n")
    pr._write_outputs([_full_record("BTC")])   # 追加 1 行 → 共 201 行
    lines = jp.read_text().splitlines()
    assert len(lines) == 50                    # 截留最近 50 行
    assert json.loads(lines[-1])["symbol"] == "BTC"   # 最新记录保留在末尾
    assert json.loads(lines[0])["n"] == 151    # 丢弃最旧的 0..150，保留 151..200
