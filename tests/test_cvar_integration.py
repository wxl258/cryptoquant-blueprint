"""CVaR 约束仓位优化集成测试（蓝图路线图 第 3 周 B · 任务：替换仓位公式）。

验证：
  1) CvarPositionOptimizer(scipy) 产出满足 CVaR 预算 + 总仓/单币上限的权重。
  2) 崩溃样本下，优化器通过减仓/零仓把组合 CVaR 拉回预算内（不输出越界权重）。
  3) scipy 缺失时的降级路径仍满足约束（monkeypatch HAS_SCIPY=False）。
  4) run_once 携带 proposed_exposure / cvar_pct 产出；cvar_optimizer=None 时
     退化为全 0（向后兼容、fail-closed：无优化器→无仓位，而非风险误配）。
"""
import numpy as np
import pytest

from cryptoquant_auto.risk.cvar_optimizer import CvarPositionOptimizer, cvar


def _solve(alpha=0.05, budget=-0.02, cap=0.12, max_pos=0.05, seed=11):
    rng = np.random.default_rng(seed)
    syms = ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX"]
    R = rng.normal(0.0, 0.02, size=(200, 6))
    conv = np.abs(rng.normal(0.5, 0.2, size=6))
    opt = CvarPositionOptimizer(alpha=alpha, cvar_budget=budget,
                                total_cap=cap, max_pos=max_pos)
    w = opt.solve(conv, R, syms)
    wv = np.array([w[s] for s in syms])
    pcvar = cvar(R @ wv, alpha)
    return w, wv, pcvar, cap, max_pos, budget


def test_optimizer_satisfies_constraints():
    w, wv, pcvar, cap, max_pos, budget = _solve()
    assert pcvar >= budget - 1e-6, f"组合 CVaR {pcvar} 应≥预算 {budget}"
    assert wv.sum() <= cap + 1e-6, "总仓应≤上限"
    assert wv.max() <= max_pos + 1e-6, "单币应≤上限"
    assert (wv >= -1e-9).all(), "权重应非负"


def test_optimizer_respects_crash_tail():
    # 构造一个尾部极差的资产，验证优化器不会给越界权重
    rng = np.random.default_rng(13)
    syms = ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX"]
    R = rng.normal(0.0, 0.01, size=(300, 6))
    R[:, 0] = np.concatenate([R[:, 0][:240], rng.normal(-0.08, 0.04, 60)])  # 极端负尾
    conv = np.abs(rng.normal(0.5, 0.2, size=6))
    opt = CvarPositionOptimizer(cvar_budget=-0.01, total_cap=0.12, max_pos=0.05)
    w = opt.solve(conv, R, syms)
    wv = np.array([w[s] for s in syms])
    pcvar = cvar(R @ wv, 0.05)
    assert pcvar >= -0.01 - 1e-6, f"组合 CVaR {pcvar} 应≥预算 -0.01（崩溃尾被约束）"


def test_degrade_path_without_scipy(monkeypatch):
    # 强制走启发式降级，验证仍满足 CVaR 预算（绝不抛错、约束有界）
    import cryptoquant_auto.risk.cvar_optimizer as mod
    monkeypatch.setattr(mod, "HAS_SCIPY", False)
    rng = np.random.default_rng(17)
    syms = ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX"]
    R = rng.normal(0.0, 0.02, size=(200, 6))
    conv = np.abs(rng.normal(0.5, 0.2, size=6))
    opt = mod.CvarPositionOptimizer(cvar_budget=-0.02, total_cap=0.12, max_pos=0.05)
    w = opt.solve(conv, R, syms)
    wv = np.array([w[s] for s in syms])
    pcvar = mod.cvar(R @ wv, 0.05)
    assert pcvar >= -0.02 - 1e-6, "降级路径组合 CVaR 也应≥预算"
    assert wv.sum() <= 0.1201


def _fake_council_and_source():
    """构造最小 fake council + DataSource，驱动 run_once（无需真实 LLM / 网络）。"""
    from cryptoquant_auto.paper_runner import DataSource

    class FakeDecision:
        def __init__(self, action, conf):
            self.action = action
            self.confidence = conf
            self.rationale = ["fake"]
            self.vetoes = []
            self.market_state = "RANGE"
            self.llm_decision = {}
            self.analyst_drivers = []
            self.decision_id = ""

        def direction_int(self):
            return 1 if self.action == "LONG" else (-1 if self.action == "SHORT" else 0)

    class FakeCouncil:
        def __init__(self):
            self.llm = object()  # 非 RealLLM → 标记 'mock'

        def decide(self, sym, feat, regime, record=True, conformal=None,
                   feature_names=None, forecast=None):
            action = "HOLD" if sym == "ETH" else "LONG"
            return FakeDecision(action, 0.6)

    class FakeSource(DataSource):
        def symbols(self):
            return ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX"]

        def snapshot(self):
            out = {}
            for i, s in enumerate(self.symbols()):
                out[s] = {"feat": np.zeros(9), "regime": "TREND",
                          "price": 100.0 + i, "ts": 1700000000 + i}
            return out

    return FakeCouncil(), FakeSource()


def test_run_once_emits_proposed_exposure():
    from cryptoquant_auto.paper_runner import run_once, _make_cvar_optimizer
    from cryptoquant_auto.risk.constitution import TradingConstitution
    from cryptoquant_auto.meta.memory import FinMemMemory

    council, source = _fake_council_and_source()
    const = TradingConstitution(live_capital=False)
    mem = FinMemMemory()
    opt = _make_cvar_optimizer()
    recs = run_once(source, council, const, mem, cvar_optimizer=opt)
    assert len(recs) == 6
    for r in recs:
        assert "proposed_exposure" in r and "cvar_pct" in r
        assert isinstance(r["proposed_exposure"], float)
    longs = [r for r in recs if r["action"] != "HOLD"]
    assert any(r["proposed_exposure"] > 0.0 for r in longs), \
        "LONG 币种应获得正仓位（优化器确实在分配）"


def test_run_once_no_optimizer_backward_compat():
    from cryptoquant_auto.paper_runner import run_once
    from cryptoquant_auto.risk.constitution import TradingConstitution
    from cryptoquant_auto.meta.memory import FinMemMemory

    council, source = _fake_council_and_source()
    const = TradingConstitution(live_capital=False)
    mem = FinMemMemory()
    recs = run_once(source, council, const, mem, cvar_optimizer=None)
    assert len(recs) == 6
    for r in recs:
        # 无优化器 → 仓位权重退化为 0（fail-closed：无仓位而非风险误配），向后兼容；
        # cvar_pct 是该币尾部风险诊断指标，与优化器无关，仍如实显示。
        assert r["proposed_exposure"] == 0.0
