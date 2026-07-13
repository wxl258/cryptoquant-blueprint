"""P0-4 / P0-4b / P1-21 回归：FinMem 分层记忆、决策回填、反思自改进、持久化根。"""
import os

import pytest

from cryptoquant_auto.meta.memory import (
    FinMemMemory, Profile, Episode, REFLECT_MIN_N,
)


def _mem(tmp_path):
    return FinMemMemory(base_dir=str(tmp_path))


def test_record_decision_returns_unique_id(tmp_path):
    mem = _mem(tmp_path)
    ep = Episode(ts=1.0, symbol="BTC", regime="TREND",
                 decision="LONG", confidence=0.6, rationale=["r"])
    did = mem.record_decision(ep)
    assert isinstance(did, str) and len(did) > 0
    # 唯一性
    did2 = mem.record_decision(Episode(ts=2.0, symbol="ETH", regime="RANGE",
                                        decision="SHORT", confidence=0.5, rationale=["r"]))
    assert did != did2


def test_set_outcome_by_decision_id(tmp_path):
    mem = _mem(tmp_path)
    did = mem.record_decision(Episode(ts=1.0, symbol="BTC", regime="TREND",
                                       decision="LONG", confidence=0.6, rationale=["r"]))
    ok = mem.set_outcome("BTC", 12.5, decision_id=did)
    assert ok is True
    # 回填后该情景 outcome_label 应为 WIN（正 bps）
    target = [e for e in mem.short_term if e.decision_id == did][0]
    assert target.outcome_bps == 12.5
    assert target.outcome_label == "WIN"


def test_set_outcome_unknown_id_returns_false(tmp_path):
    mem = _mem(tmp_path)
    assert mem.set_outcome("BTC", 5.0, decision_id="nope") is False


def test_reflect_forbids_losing_regime_with_cooldown(tmp_path):
    mem = _mem(tmp_path)
    # 注入足够多的 TREND 亏损样本（>= REFLECT_MIN_N）
    for i in range(REFLECT_MIN_N + 2):
        mem.record_decision(Episode(ts=float(i), symbol="BTC", regime="TREND",
                                     decision="LONG", confidence=0.6, rationale=["r"]))
        mem.set_outcome("BTC", -10.0, decision_id=mem.short_term[-1].decision_id)
    mem.reflect()
    assert "TREND" in mem.profile.forbidden_regimes
    assert mem.profile.forbid_cooldown.get("TREND", 0) > 0


def test_forbidden_regime_auto_unforbid_after_cooldown(tmp_path):
    """冷却到期 + 数据改善 → 解禁（且 forbidden_at 清除）。

    注意：reflect 在冷却到期后会用『当前』短期记忆重新评估；若数据仍亏损会 re-forbid
    （正确风险管理）。故此处冷却倒计时期间把亏损样本替换为盈利样本，模拟「数据已改善」，
    验证冷却机制确能释放 regime。
    """
    mem = _mem(tmp_path)
    for i in range(REFLECT_MIN_N + 2):
        mem.record_decision(Episode(ts=float(i), symbol="BTC", regime="TREND",
                                     decision="LONG", confidence=0.6, rationale=["r"]))
        mem.set_outcome("BTC", -10.0, decision_id=mem.short_term[-1].decision_id)
    mem.reflect()
    assert "TREND" in mem.profile.forbidden_regimes
    cooldown = mem.profile.forbid_cooldown["TREND"]
    # 替换为盈利样本（模拟数据改善）；ts 取更大值避免被旧 forbidden_at cutoff 误判
    mem.short_term.clear()
    for i in range(REFLECT_MIN_N + 2):
        mem.record_decision(Episode(ts=100.0 + float(i), symbol="BTC", regime="TREND",
                                     decision="LONG", confidence=0.6, rationale=["r"]))
        mem.set_outcome("BTC", 15.0, decision_id=mem.short_term[-1].decision_id)
    for _ in range(cooldown + 1):
        mem.reflect()
    assert "TREND" not in mem.profile.forbidden_regimes
    assert "TREND" not in mem.profile.forbidden_at  # 解禁须清 forbidden_at（P2 修复）


def test_forbidden_at_cleared_on_unforbid(tmp_path):
    """P2 回归：解禁时须清除 forbidden_at[regime]。

    根因（P1 会话遗留）：原逻辑解除禁易只移出 forbidden_regimes，未清 forbidden_at
    （=禁易时刻 max ts）。该 stale 值使后续 reflect 的 fresh = e.ts > forbidden_at 过滤
    对 ts 更小的样本恒为空 → 该 regime 一旦被禁过就「永远无法再次触发禁易」（死锁），
    stage3 的 CRASH 小 ts 注入即被此死锁屏蔽。修复后，冷却到期解禁须同步 pop forbidden_at。
    """
    mem = _mem(tmp_path)
    for i in range(REFLECT_MIN_N + 2):
        mem.record_decision(Episode(ts=float(i), symbol="BTC", regime="CRASH",
                                     decision="LONG", confidence=0.6, rationale=["r"]))
        mem.set_outcome("BTC", -12.0, decision_id=mem.short_term[-1].decision_id)
    mem.reflect()
    assert "CRASH" in mem.profile.forbidden_regimes
    assert "CRASH" in mem.profile.forbidden_at  # 禁易时记录时刻
    cooldown = mem.profile.forbid_cooldown["CRASH"]
    # 数据改善（盈利）→ 冷却到期解禁
    mem.short_term.clear()
    for i in range(REFLECT_MIN_N + 2):
        mem.record_decision(Episode(ts=100.0 + float(i), symbol="BTC", regime="CRASH",
                                     decision="LONG", confidence=0.6, rationale=["r"]))
        mem.set_outcome("BTC", 15.0, decision_id=mem.short_term[-1].decision_id)
    for _ in range(cooldown + 1):
        mem.reflect()
    assert "CRASH" not in mem.profile.forbidden_regimes
    assert "CRASH" not in mem.profile.forbidden_at  # 解禁须清 forbidden_at（P2 修复点）


def test_persistence_files_written_under_base_dir(tmp_path):
    mem = _mem(tmp_path)
    mem.record_decision(Episode(ts=1.0, symbol="BTC", regime="TREND",
                                decision="LONG", confidence=0.6, rationale=["r"]))
    data_dir = tmp_path / "data"
    assert data_dir.exists()
    assert (data_dir / "finmem_profile.json").exists()
    assert (data_dir / "finmem_shortterm.json").exists()


def test_persistence_root_is_package_dir():
    # P1-21：memory 持久化根应为包目录（与 cognition/reflection 一致）
    import cryptoquant_auto.meta.memory as m
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(m.__file__)))  # cryptoquant_auto
    assert m._BASE_DIR == pkg
