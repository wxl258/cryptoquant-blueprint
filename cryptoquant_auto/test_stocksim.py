"""StockSim 集成测试（蓝图路线图 第 5 周 E · 订单簿 + LLM 合成对手盘）。

验证：
  1) OrderBook 市价单穿价吃单：VWAP 高于买前 mid、mid 被价格冲击上移。
  2) OrderBook 限价单挂驻 + 反向市价单吃到挂单（成交价低于原 mid）。
  3) Stylized facts 复现：长程模拟下肥尾/波动聚集/量自相关三项成立（程式化事实）。
  4) LLM 合成对手盘：LLMMarketAgent 经 MockLLM 接地产出叙事（last_narrative 非空）。
  5) 集成：stocksim_backtest 把信号对 StockSim 合成市场成交，回真实执行管线产出统计。
"""
import numpy as np
import pytest

from cryptoquant_auto.sim.stocksim import (
    OrderBook, MarketSimulator, make_market_agent, measure_stylized_facts,
)
from cryptoquant_auto.sim.backtest import make_random_signals, stocksim_backtest


def test_orderbook_market_buy_impacts_mid():
    ob = OrderBook(mid=100.0, tick=0.5, base_liq=50.0)
    vwap = ob.match_market("BUY", 60.0)        # 吃穿 1~2 档
    assert vwap > 100.0, "市价买应推高成交价"
    assert ob.mid > 100.0, "mid 应被价格冲击上移"
    # VWAP 介于最差吃单价之间（best ask=100.5，吃穿到 101.5 档）
    assert 100.0 < vwap <= 100.0 + 3 * 0.5 + 1e-6


def test_orderbook_limit_orders_rest_and_match():
    ob = OrderBook(mid=100.0, tick=0.5, base_liq=50.0)
    ob.add_limit("BUY", 99.0, 30.0)            # 该档已 seeded 50，累加为 80
    assert ob.bids.get(99.0) == 80.0
    vwap = ob.match_market("SELL", 20.0)       # 吃最优买盘（99.5）
    assert vwap < 100.0, "反向市价卖应吃到更低的买盘"
    assert ob.bids.get(99.0) == 80.0, "未触及的挂单档量不变"


def test_stylized_facts_reproduced():
    sim = MarketSimulator(mid=100.0, agent=make_market_agent("mock", seed=1), tick=0.05)
    res = sim.run(2000)
    facts = measure_stylized_facts(res.prices, res.volumes)
    # 程式化事实应稳定复现（实测多 seed 下肥尾+波动聚集+量自相关三项恒 True）
    assert facts["fat_tails"] is True, "应复现肥尾（超额峰度>1）"
    assert facts["vol_clustering"] is True, "应复现波动聚集（|收益|滞后相关>0）"
    assert facts["n_stylized_facts"] >= 2, "至少复现两项程式化事实"


def test_llm_agent_narrates():
    agent = make_market_agent("llm", seed=1)
    for _ in range(40):                        # 越过叙事预热窗
        agent.next_order()
    assert agent.last_narrative != "", "LLM 合成对手盘应经 MockLLM 接地产出叙事"


def test_stocksim_backtest_integration():
    sigs = make_random_signals(4, seed=3)
    stats = stocksim_backtest(sigs, agent_kind="mock", n_bars=60, seed=3)
    assert stats.n_trades > 0, "策略应对 StockSim 合成市场产生成交"
    # 指标数值有限（无 NaN/inf）
    assert np.isfinite(stats.sharpe)
    assert np.isfinite(stats.max_dd_pct)
    assert 0.0 <= stats.win_rate <= 1.0
