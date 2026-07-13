"""任务14 · StockSim 订单级对手盘（零依赖纯 numpy）。

落点：sim/ 扩展（蓝图实施计划 §三·专家C）。复现程式化事实（肥尾 / 波动聚集 /
成交量自相关），让原型策略在「会复现真实统计特征」的市场里练，平移更稳。

组成：
  - OrderBook：限价/市价订单 + 价格-时间优先撮合（FIFO 跨价位），成交带记录 trade tape。
  - MarketSimulator：把对手盘订单注入订单簿，驱动价格形成，输出价格路径。
  - MockMarketAgent：零依赖统计占位（GARCH 波动率 + 羊群/动量/均值回归 + 偶发跳跃），
    复现肥尾与波动聚集。
  - LLMMarketAgent：阶段4 升级的 LLM 人造市场智能体——用 MockLLM 对近期收益/波动接地产出
    叙事并偏置订单流；仍保留 GARCH 基底，故程式化事实可持续复现（make_market_agent("llm") 接入）。
  - measure_stylized_facts：量化肥尾(超额峰度) / 波动聚集(|收益|滞后自相关) /
    成交量自相关，给出是否复现的判定。

零依赖纪律：仅 numpy。蓝图提到「LLM 人造市场」需 torch/API，按阶段铁律后移至
阶段3-4；本原型用 MockMarketAgent 验证概念，并保留降级路径
（make_market_agent(kind='llm') 在缺少 LLM 依赖时自动回退 mock，绝不破坏零依赖可跑）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================ 订单簿 ============================
class OrderBook:
    """极简订单簿：每个价位聚合挂单量，价格-时间优先撮合（最优价先吃）。"""

    def __init__(self, mid: float, tick: float = 0.5, base_liq: float = 50.0):
        self.mid = float(mid)
        self.tick = float(tick)
        self.base_liq = float(base_liq)
        self.bids: Dict[float, float] = {}   # price -> qty
        self.asks: Dict[float, float] = {}
        self.trades: List[Tuple[float, float, str, int]] = []  # (price, qty, side, seq)
        self.seq = 0
        self._seed_book()

    def _seed_book(self) -> None:
        for k in range(1, 11):
            self.bids[self.mid - k * self.tick] = self.base_liq
            self.asks[self.mid + k * self.tick] = self.base_liq

    def _best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def _best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def add_limit(self, side: str, price: float, qty: float) -> None:
        if qty <= 0:
            return
        book = self.bids if side == "BUY" else self.asks
        book[price] = book.get(price, 0.0) + qty

    def _replenish(self) -> None:
        """每步补充流动性，维持簿厚度（背景挂单）。"""
        bb, ba = self._best_bid(), self._best_ask()
        if bb is None or self.mid - bb > 10 * self.tick:
            self.bids[self.mid - self.tick] = self.base_liq
        if ba is None or ba - self.mid > 10 * self.tick:
            self.asks[self.mid + self.tick] = self.base_liq

    def match_market(self, side: str, qty: float) -> float:
        """市价单吃单，返回 VWAP 成交价；价格冲击写入 mid。"""
        book = self.asks if side == "BUY" else self.bids
        remaining = float(qty)
        vwap_sum = 0.0
        vwap_qty = 0.0
        while remaining > 1e-9 and book:
            best = min(book) if side == "BUY" else max(book)
            avail = book[best]
            take = min(remaining, avail)
            self.trades.append((best, take, side, self.seq))
            self.seq += 1
            vwap_sum += best * take
            vwap_qty += take
            remaining -= take
            if abs(book[best] - take) < 1e-9:
                del book[best]
            else:
                book[best] -= take
            # 价格冲击：吃单推高/压低 mid
            self.mid = best
        if remaining > 1e-9:
            # 簿被吃穿 → 外延价格（冲击），用 tick 步进补足
            step = self.tick if side == "BUY" else -self.tick
            self.mid += step * (1 + remaining / self.base_liq)
        self._replenish()
        return (vwap_sum / vwap_qty) if vwap_qty > 0 else self.mid


# ============================ 人造市场智能体 ============================
class MockMarketAgent:
    """LLM 人造市场智能体（零依赖占位）。

    用 GARCH 波动率状态 + 羊群/动量/均值回归 + 偶发跳跃，复现程式化事实。
    蓝图中的真实 LLM 驱动在阶段3-4 接入（torch/API），此处提供等价统计行为占位。
    """

    def __init__(self, vol0: float = 0.002, garch_a: float = 0.02,
                 garch_b: float = 0.94, jump_prob: float = 0.01,
                 jump_size: float = 0.012, herding: float = 0.3,
                 momentum: float = 0.2, seed: int = 0):
        self.vol = vol0
        self.garch_a = garch_a
        self.garch_b = garch_b
        self.jump_prob = jump_prob
        self.jump_size = jump_size
        self.herding = herding
        self.momentum = momentum
        self.rng = np.random.default_rng(seed)
        self.last_side = 0.0
        self.last_ret = 0.0
        self.activity = 1.0          # 成交量活跃度（AR(1)）→ 复现成交量自相关

    def _step_activity(self) -> None:
        """成交量活跃度 AR(1)：缓慢起伏 → 成交量序列自相关（程式化事实之一）。"""
        self.activity = max(0.2, 0.7 * self.activity + 0.3 * abs(self.rng.normal(0.9, 0.5)))

    def next_order(self) -> Tuple[str, float]:
        """生成下一笔对手盘订单（side, signed_size）。size 含波动率缩放。"""
        # GARCH 波动率更新（波动聚集来源）
        self.vol = (self.garch_a + self.garch_b * self.vol
                    + self.garch_a * abs(self.last_ret))
        self.vol = max(1e-4, min(self.vol, 0.05))
        # 方向：羊群(跟随上笔) + 动量(跟随上笔收益) + 噪声
        noise = self.rng.normal(0, 1.0)
        dir_score = (self.herding * self.last_side
                     + self.momentum * np.sign(self.last_ret)
                     + noise)
        side = "BUY" if dir_score > 0 else "SELL"
        self.last_side = 1.0 if side == "BUY" else -1.0
        # 成交量：AR(1) 活跃度 + 加法小抖动（加法噪声对自相关稀释远小于乘法噪声），
        # 使成交量序列呈现正自相关（程式化事实之一）。
        self._step_activity()
        size = max(0.1, self.activity + 0.05 * self.rng.normal(0, 1.0))
        # 偶发跳跃（肥尾来源）
        if self.rng.random() < self.jump_prob:
            size += abs(self.rng.normal(0, 1.0)) * (self.jump_size / self.vol + 5.0)
            side = "BUY" if self.rng.random() < 0.5 else "SELL"
            self.last_side = 1.0 if side == "BUY" else -1.0
        return side, max(0.1, size)


# ============================ 仿真器 ============================
@dataclass
class SimResult:
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)


class MarketSimulator:
    """把对手盘订单注入订单簿，驱动价格形成。"""

    def __init__(self, mid: float = 100.0, agent: Optional[MockMarketAgent] = None,
                 tick: float = 0.05):
        self.book = OrderBook(mid, tick=tick)
        self.agent = agent or MockMarketAgent()
        self.prices: List[float] = [mid]
        self.volumes: List[float] = [0.0]
        self.returns: List[float] = []

    def step(self) -> float:
        side, size = self.agent.next_order()
        vwap = self.book.match_market(side, size)
        self.prices.append(vwap)
        self.volumes.append(size)
        ret = math.log(vwap / self.prices[-2]) if self.prices[-2] > 0 else 0.0
        self.agent.last_ret = ret
        self.returns.append(ret)
        return vwap

    def run(self, n_steps: int) -> SimResult:
        for _ in range(n_steps):
            self.step()
        return SimResult(prices=self.prices[:], volumes=self.volumes[:],
                         returns=self.returns[:])


# ============================ 程式化事实度量 ============================
def _acf1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    if x.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def measure_stylized_facts(prices: List[float], volumes: List[float]
                           ) -> Dict[str, object]:
    """量化 StockSim 是否复现程式化事实。"""
    rets = np.diff(np.log(np.asarray(prices, float) + 1e-12))
    n = len(rets)
    # 肥尾：超额峰度
    if n > 4 and rets.std(ddof=1) > 0:
        m2 = (rets ** 2).mean()
        m4 = (rets ** 4).mean()
        kurt = float(m4 / (m2 ** 2)) - 3.0
    else:
        kurt = 0.0
    # 波动聚集：|收益| 滞后1 自相关
    acf_abs = _acf1(np.abs(rets))
    # 成交量自相关
    acf_vol = _acf1(np.asarray(volumes, float))
    return {
        "n": n,
        "excess_kurtosis": round(kurt, 3),
        "fat_tails": kurt > 1.0,                 # 超额峰度 > 1 → 肥尾
        "vol_acf1": round(acf_abs, 3),
        "vol_clustering": acf_abs > 0.05,        # |收益|滞后相关 > 0 → 波动聚集
        "volume_acf1": round(acf_vol, 3),
        "volume_autocorr": acf_vol > 0.05,
        "n_stylized_facts": int(kurt > 1.0) + int(acf_abs > 0.05) + int(acf_vol > 0.05),
    }


class LLMMarketAgent(MockMarketAgent):
    """LLM 人造市场智能体（阶段4 升级）：用 MockLLM 对近期收益/波动接地产出
    市场态+情绪叙事，偏置订单流方向，使仿真市场「响应」LLM 叙事。

    仍保留 GARCH 基底 → 程式化事实（肥尾/波动聚集/量自相关）可持续复现。
    降级：MockLLM 不可用时 self._llm=None，next_order 退化为 MockMarketAgent 行为（无偏置），
    保证零依赖可跑（纪律：torch/LLM 缺失→numpy 降级）。
    """

    def __init__(self, vol0: float = 0.002, garch_a: float = 0.02,
                 garch_b: float = 0.94, jump_prob: float = 0.01,
                 jump_size: float = 0.012, herding: float = 0.3,
                 momentum: float = 0.2, seed: int = 0,
                 llm_influence: float = 0.5, window: int = 30):
        super().__init__(vol0=vol0, garch_a=garch_a, garch_b=garch_b,
                         jump_prob=jump_prob, jump_size=jump_size,
                         herding=herding, momentum=momentum, seed=seed)
        self.llm_influence = llm_influence
        self.window = window
        self._ret_buf: List[float] = []
        self._sentiment = 0.0
        self._conf = 0.0
        self.last_narrative = ""
        try:
            from ..adapters.mock_llm import MockLLM, CouncilContext
            self._llm = MockLLM()
            self._ctx_cls = CouncilContext
        except Exception:
            self._llm = None
            self._ctx_cls = None

    def _maybe_narrate(self) -> None:
        """用上一笔收益更新缓冲，并调 MockLLM 产出叙事（接地）。"""
        if hasattr(self, "last_ret"):
            self._ret_buf.append(self.last_ret)
        if len(self._ret_buf) > self.window:
            self._ret_buf = self._ret_buf[-self.window:]
        self._sentiment = 0.0
        self._conf = 0.0
        self.last_narrative = ""
        if self._llm is None or len(self._ret_buf) < 5:
            return
        buf = np.array(self._ret_buf, float)
        mean = float(buf.mean())
        vol = float(buf.std() + 1e-6)
        regime = "TREND" if abs(mean) > vol else "RANGE"
        fused = "LONG" if mean > 0 else ("SHORT" if mean < 0 else "HOLD")
        base_conf = float(np.clip(abs(mean) / (vol * 4 + 1e-6), 0.0, 1.0))
        support = ["近期动量向上"] if mean > 0 else (["近期动量向下"] if mean < 0 else [])
        contra = ["波动偏高"] if vol > 0.01 else []
        ctx = self._ctx_cls(symbol="SIM", regime=regime, market_state=regime,
                            fused_action=fused, base_confidence=base_conf,
                            support=support, contra=contra,
                            retrieved_insights=[], spi_surprise=0.0)
        dec = self._llm.produce(ctx)
        self._sentiment = (1.0 if dec.proposed_action == "LONG"
                           else -1.0 if dec.proposed_action == "SHORT" else 0.0)
        self._conf = dec.confidence
        self.last_narrative = f"{dec.market_state}/{dec.proposed_action}({dec.confidence:.2f})"

    def next_order(self):
        self._maybe_narrate()
        if self._llm is not None and self._sentiment != 0.0:
            # 把 LLM 叙事偏置注入订单流方向（羊群/动量）
            bias = self._sentiment * self._conf * self.llm_influence
            old_h, old_m = self.herding, self.momentum
            self.herding += bias
            self.momentum += bias
            side, size = super().next_order()
            self.herding, self.momentum = old_h, old_m
            return side, size
        return super().next_order()


def make_market_agent(kind: str = "mock", **kw) -> MockMarketAgent:
    """工厂：kind='mock' 零依赖占位；kind='llm' 用 LLM 接地市场智能体。

    LLMMarketAgent 内部已含降级（MockLLM 缺失→退化为 mock 行为），保证零依赖可跑。
    """
    if kind == "mock":
        return MockMarketAgent(**kw)
    # 阶段4：LLM 驱动市场；缺失依赖时 LLMMarketAgent 自动退化为 mock 行为
    try:
        return LLMMarketAgent(**kw)
    except Exception:
        return MockMarketAgent()
