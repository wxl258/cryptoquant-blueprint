"""P2-D 回测验证回归：PaperBacktest 管线 + 统计产出 + 关键风控闸门。

覆盖：
  - 基础回放与统计产出（既有）
  - EV 期望值闸门：负期望一律拒、不评估则放行（共识#2：EV > 胜率）
  - Gate B 成本盈亏平衡闸门：最坏成本净edge<=0 时全拒（fail-closed 锚点）
  - 可复现性：同 seed + 同信号 → 统计完全一致（共识#8）
  - maker/taker 双侧均可产出统计
  - stats 暴露 per_coin_edge_bps 与 gate_b 字段
"""
import pytest

from cryptoquant_auto.sim.backtest import (
    PaperBacktest, BacktestConfig, make_random_signals,
)
from cryptoquant_auto.sim.metrics import summarize
from cryptoquant_auto.risk.gate import GateConfig


def _one_signal():
    return make_random_signals(1, seed=5)[0]


def test_paper_backtest_runs_and_reports():
    sigs = make_random_signals(240, seed=7)
    bt = PaperBacktest(BacktestConfig(equity=100_000, seed=7))
    bt.run_batch(sigs)
    stats = bt.stats()
    assert stats.n_trades >= 0
    assert isinstance(stats.net_pnl_pct, float)
    # 回撤非正（亏损方向）；max_dd_pct 为 <=0 的负值或 0
    assert stats.max_dd_pct <= 0.0


def test_summarize_returns_backtest_stats():
    sigs = make_random_signals(120, seed=9)
    bt = PaperBacktest(BacktestConfig(equity=100_000, seed=9))
    bt.run_batch(sigs)
    equity = bt.equity_curve
    trades = bt.trades
    s = summarize(equity, trades, gate_b={})
    assert hasattr(s, "sharpe")
    assert hasattr(s, "win_rate")


def test_ev_gate_rejects_negative_expectation():
    """EV 闸门（共识#2）：ev_est 明确负 → 拒（计入 skipped）；不评估 → 放行。

    同信号、同 gate（仅 ev_est 差），保证差异完全由 EV 闸门造成（因果隔离）。
    """
    sig = _one_signal()
    common = dict(equity=100_000, enforce_gate_b=False, min_ev=0.0)

    bt_none = PaperBacktest(BacktestConfig(equity=100_000, seed=5),
                            gate=GateConfig(**common))
    bt_none.run_signal(sig, ev_est=None)            # 不评估 → 应放行

    bt_neg = PaperBacktest(BacktestConfig(equity=100_000, seed=5),
                           gate=GateConfig(**common))
    bt_neg.run_signal(sig, ev_est=-0.01)            # 明确负期望 → 应拒

    assert bt_none.skipped == 0 and len(bt_none.trades) == 1
    assert bt_neg.skipped == 1 and len(bt_neg.trades) == 0


def test_gate_b_fail_closed_blocks_all():
    """Gate B 成本盈亏平衡闸门（P1-4 fail-closed）：最坏净edge<=0 时所有币禁开。

    GROSS_EDGE_BPS 锁定 -2.5bps（跨所5年OOS最保守锚点），故对全部币生效，
    系统在 edge 未经大样本复核前维持空仓（圆桌共识）。
    """
    sigs = make_random_signals(40, seed=3)
    bt = PaperBacktest(BacktestConfig(equity=100_000, seed=3),
                       gate=GateConfig(equity=100_000, enforce_gate_b=True))
    for s in sigs:
        bt.run_signal(s)
    assert bt.skipped == len(sigs)
    assert len(bt.trades) == 0


def test_reproducible_same_seed():
    """可复现性（共识#8）：同 seed + 同信号序列 → 统计完全一致。"""
    sigs = make_random_signals(120, seed=7)

    def run():
        bt = PaperBacktest(BacktestConfig(equity=100_000, seed=7, use_taker=False))
        bt.run_batch(sigs)
        return bt.stats()

    a, b = run(), run()
    assert a.n_trades == b.n_trades
    assert a.net_pnl_pct == b.net_pnl_pct
    assert a.sharpe == b.sharpe


def test_maker_vs_taker_both_reported():
    sigs = make_random_signals(120, seed=7)
    m = PaperBacktest(BacktestConfig(equity=100_000, seed=7, use_taker=False)).run_batch(sigs)
    t = PaperBacktest(BacktestConfig(equity=100_000, seed=7, use_taker=True)).run_batch(sigs)
    for s in (m, t):
        assert isinstance(s.sharpe, float)
        assert isinstance(s.win_rate, float)
        assert isinstance(s.net_pnl_pct, float)


def test_stats_expose_edge_and_gate_b():
    sigs = make_random_signals(120, seed=7)
    bt = PaperBacktest(BacktestConfig(equity=100_000, seed=7))
    bt.run_batch(sigs)
    st = bt.stats()
    assert isinstance(st.per_coin_edge_bps, dict) and len(st.per_coin_edge_bps) > 0
    assert isinstance(st.gate_b, dict)
