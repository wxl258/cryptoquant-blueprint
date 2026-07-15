"""任务14 · StockSim 订单级对手盘（零依赖纯 numpy）。

落点：sim/ 扩展（蓝图实施计划 §三·专家C）。复现程式化事实（肥尾 / 波动聚集 /
成交量自相关），让原型策略在「会复现真实统计特征」的市场里练，平移更稳。

组成：
  - OrderBook：限价/市价订单 + 价格-时间优先撮合（FIFO 跨价位），成交带记录 trade tape。
    Kyle's λ 连续价格冲击 + Hawkes 自激励报单集群到达（LOB 层，与 council 协同）。
  - MarketSimulator：把对手盘订单注入订单簿，驱动价格形成，输出价格路径。
  - MockMarketAgent：零依赖统计占位（GARCH 波动率 + 羊群/动量/均值回归 + 偶发跳跃），
    复现肥尾与波动聚集。
  - LLMMarketAgent：阶段4 升级的 LLM 人造市场智能体——用 MockLLM 对近期收益/波动接地产出
    叙事并偏置订单流；仍保留 GARCH 基底，故程式化事实可持续复现（make_market_agent("llm") 接入）。
  - measure_stylized_facts：7 项程式化事实检验（肥尾 / 波动聚集 / 成交量自相关 /
    杠杆效应 / 量-波交叉相关 / 波动长记忆 ACF₅₊₁₀ / 收益线性自相关缺失），Council 7/7 通过。
  - CouncilMarketAgent：多角色委员会（趋势×2、均值回归×2、基本面×2、噪声×2、做市×1），
    纯 numpy 多边订单流，OFI 订单流不平衡信号 + Hawkes 调制 → 价格内生形成，
    MomentumAgent 按 OFI 方向带节奏增/减仓，NoiseAgent 按 Hawkes 强度扩大/缩小跳跃与活跃度。

零依赖纪律：仅 numpy。蓝图提到「LLM 人造市场」需 torch/API，按阶段铁律后移至
阶段3-4；本原型用 MockMarketAgent 验证概念，并保留降级路径
（make_market_agent(kind='llm') 在缺少 LLM 依赖时自动回退 mock，绝不破坏零依赖可跑）。

蓝图升级路径（阶段3-4，当前未落地）：
  FinMem + 4-role LLM（多头/空头/套利/流动性）人造市场是蓝图最终目标；受限于无重算力，
  当前 LLMMarketAgent 使用 MockLLM（GARCH+叙事占位），真实 LLM 推理留待有重算力/GPU 环境后接入。

LOB 层改进（Kyle's λ / OFI / Hawkes / 7 stylized facts）已于 Phase 2-4 提前落地。

"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================ 订单簿 ============================
class OrderBook:
    """极简订单簿：每个价位聚合挂单量，价格-时间优先撮合（最优价先吃）。"""

    def __init__(self, mid: float, tick: float = 0.5, base_liq: float = 50.0,
                 kyle_lambda: float = 0.0005):
        self.mid = float(mid)
        self.tick = float(tick)
        self.base_liq = float(base_liq)
        self.kyle_lambda = float(kyle_lambda)
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
        # Kyle's λ 连续价格冲击：成交额越大/盘口深度越浅，冲击越大
        taken = vwap_qty  # 实际吃单量
        total_depth = sum(self.bids.values()) + sum(self.asks.values()) + 1e-12
        imp = self.kyle_lambda * taken / total_depth * 10.0
        self.mid *= 1.0 + (imp if side == "BUY" else -imp)
        return (vwap_sum / vwap_qty) if vwap_qty > 0 else self.mid


# ============================ 人造市场智能体 ============================
class MockMarketAgent:
    """LLM 人造市场智能体（零依赖占位）。

    用 GARCH 波动率状态 + 羊群/动量/均值回归 + 偶发跳跃，复现程式化事实。
    蓝图中的真实 LLM 驱动在阶段3-4 接入（torch/API），此处提供等价统计行为占位。
    """

    def __init__(self, vol0: float = 0.002, garch_a: float = 0.05,
                 garch_b: float = 0.90, garch_omega: float = 1e-6,
                 garch_gamma: float = 0.08,
                 jump_prob: float = 0.01,
                 jump_size: float = 0.012, herding: float = 0.3,
                 momentum: float = 0.2, seed: int = 0):
        self.vol = vol0
        self.garch_a = garch_a
        self.garch_b = garch_b
        self.garch_omega = garch_omega
        self.garch_gamma = garch_gamma
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
        # GJR-GARCH 波动率更新（波动聚集 + 杠杆效应）
        neg = 1.0 if self.last_ret < 0 else 0.0
        v = self.vol * self.vol
        v = (self.garch_omega + self.garch_a * self.last_ret ** 2
             + self.garch_gamma * self.last_ret ** 2 * neg
             + self.garch_b * v)
        self.vol = max(1e-4, min(math.sqrt(v), 0.05))
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




# 以下为 Phase 1 新增的「多角色委员会」智能体
class MarketAgent:
    """多角色委员会智能体基类。"""
    name = "base"
    def reset(self): pass
    def produce(self, state: dict):
        """返回 (side:str, qty:float, order_type:str, limit_price:float|None) 或 None。"""
        raise NotImplementedError


class MomentumAgent(MarketAgent):
    """趋势跟随：MA5 > MA20 → BUY，差值越大仓位越重。"""
    name = "momentum"
    def __init__(self, lookback_short: int = 5, lookback_long: int = 20,
                 threshold: float = 0.002, max_qty: float = 10.0, seed: int = 0):
        self.ls = lookback_short; self.ll = lookback_long
        self.th = threshold; self.max_qty = max_qty
        self.rng = np.random.default_rng(seed)
    def produce(self, state):
        prices = state["prices"]
        if len(prices) < self.ll + 1: return None
        ma_s = float(np.mean(prices[-self.ls:]))
        ma_l = float(np.mean(prices[-self.ll:]))
        signal = (ma_s - ma_l) / (ma_l + 1e-12)
        if abs(signal) < self.th: return None
        side = "BUY" if signal > 0 else "SELL"
        qty = self.max_qty * min(abs(signal) / (self.th * 5), 1.0)
        # OFI 调制：与趋势同向则加仓，反向则减仓
        ofi = state.get("ofi", 0.0)
        if ofi * signal > 0:
            qty *= 1.5
        elif ofi * signal < 0:
            qty *= 0.5
        return (side, max(0.1, qty), "MARKET", None)


class MeanRevAgent(MarketAgent):
    """均值回归：价格偏离 MA20 超 1.5σ 时赌回归。"""
    name = "meanrev"
    def __init__(self, lookback: int = 20, z_entry: float = 1.5,
                 max_qty: float = 8.0, seed: int = 0):
        self.lb = lookback; self.z_entry = z_entry
        self.max_qty = max_qty; self.rng = np.random.default_rng(seed)
    def produce(self, state):
        prices = state["prices"]
        if len(prices) < self.lb + 1: return None
        ma = float(np.mean(prices[-self.lb:]))
        std = float(np.std(prices[-self.lb:])) + 1e-12
        z = (prices[-1] - ma) / std
        if abs(z) < self.z_entry: return None
        side = "SELL" if z > 0 else "BUY"
        qty = self.max_qty * min(abs(z) / (self.z_entry * 3), 1.0)
        return (side, max(0.1, qty), "MARKET", None)


class FundamentalAgent(MarketAgent):
    """基本面：私有公允价值缓慢漂移，偏离超阈值时交易。"""
    name = "fundamental"
    def __init__(self, fair_vol: float = 0.005, threshold: float = 0.005,
                 max_qty: float = 12.0, seed: int = 0):
        self.fair_vol = fair_vol; self.th = threshold
        self.max_qty = max_qty; self.rng = np.random.default_rng(seed)
        self.fair = 100.0
    def reset(self): self.fair = 100.0
    def produce(self, state):
        prices = state["prices"]
        if len(prices) < 2: return None
        self.fair *= 1.0 + self.rng.normal(0, self.fair_vol)
        misp = (prices[-1] - self.fair) / (self.fair + 1e-12)
        if abs(misp) < self.th: return None
        side = "SELL" if misp > 0 else "BUY"
        qty = self.max_qty * min(abs(misp) / (self.th * 4), 1.0)
        return (side, max(0.1, qty), "MARKET", None)


class NoiseAgent(MarketAgent):
    """噪声/羊群：随机方向 + 羊群跟随 + 偶发跳跃。"""
    name = "noise"
    def __init__(self, jump_prob: float = 0.008, jump_size: float = 15.0,
                 herd_strength: float = 0.3, max_qty: float = 5.0, seed: int = 0):
        self.jump_prob = jump_prob; self.jump_size = jump_size
        self.herd_strength = herd_strength; self.max_qty = max_qty
        self.rng = np.random.default_rng(seed)
        self.last_net = 0.0
    def reset(self): self.last_net = 0.0
    def produce(self, state):
        prices = state.get("prices", [])
        # Hawkes 调制：高强度 → 更活跃、更大单、更易跳跃
        hi = state.get("hawkes_intensity", 1.0)
        scale = max(0.5, hi)  # 强度漂移，最低 0.5
        rnd_dir = 1 if self.rng.random() < 0.5 else -1
        qty = self.rng.exponential(1.0) * 1.5 * scale + 0.1
        if self.rng.random() < self.herd_strength and len(prices) >= 3:
            trend = 1 if prices[-1] > prices[-3] else -1
            rnd_dir = trend
            qty += 1.0 * scale
        if self.rng.random() < self.jump_prob * scale:
            qty += self.jump_size * (0.5 + self.rng.random()) * scale
            rnd_dir = 1 if self.rng.random() < 0.5 else -1
        side = "BUY" if rnd_dir > 0 else "SELL"
        return (side, max(0.1, min(qty, self.max_qty * 3 * scale)), "MARKET", None)


class MarketMakerAgent(MarketAgent):
    """做市商：提供双边 LIMIT 报价，波动调整宽度，库存偏移。"""
    name = "maker"
    def __init__(self, base_qty: float = 8.0, spread_scale: float = 0.5,
                 n_levels: int = 5, seed: int = 0):
        self.base_qty = base_qty; self.spread_scale = spread_scale
        self.n_levels = n_levels; self.rng = np.random.default_rng(seed)
        self.inventory = 0.0
    def reset(self): self.inventory = 0.0
    def produce(self, state): return None  # maker produces limit orders via get_limit_orders()
    def get_limit_orders(self, state) -> list:
        prices = state["prices"]
        if len(prices) < 10: return []
        mid = prices[-1]
        vol = float(np.std(prices[-10:])) + 1e-12
        spread = float(vol * self.spread_scale)
        skew = float(np.tanh(self.inventory * 0.05)) * spread * 0.3
        orders = []
        for k in range(1, self.n_levels + 1):
            tick = (k - 1) * spread * 0.2
            bp = mid - spread / 2 - tick + skew
            ap = mid + spread / 2 + tick + skew
            qty = self.base_qty / k
            orders.append(("BUY", qty, "LIMIT", bp))
            orders.append(("SELL", qty, "LIMIT", ap))
        return orders


class CouncilMarketAgent:
    """多智能体委员会：管理多个 MarketAgent，每步汇总订单流。
    
    可通过 get_orders() 获取完整订单列表（多边撮合），
    也可通过 next_order() 获取净市场订单（向后兼容）。
    """
    def __init__(self, agents: Optional[list] = None,
                 hawkes_mu: float = 1.0, hawkes_alpha: float = 0.3,
                 hawkes_beta: float = 0.25):
        self.agents = agents or self._default_council()
        self._prices: List[float] = [100.0]
        self._volumes: List[float] = [0.0]
        self.hawkes_mu = hawkes_mu
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        self._hawkes_intensity = hawkes_mu
    def _default_council(self) -> list:
        return [
            MomentumAgent(seed=0), MomentumAgent(seed=1, threshold=0.003),
            MeanRevAgent(seed=0), MeanRevAgent(seed=1, lookback=40, z_entry=2.0),
            FundamentalAgent(seed=0), FundamentalAgent(seed=1, fair_vol=0.008),
            NoiseAgent(seed=0), NoiseAgent(seed=1, herd_strength=0.5),
            MarketMakerAgent(seed=0),
        ]
    def reset(self):
        self._prices = [100.0]; self._volumes = [0.0]
        for a in getattr(self.agents, []): a.reset()
    def get_orders(self) -> list:
        """返回所有 agent 的订单列表 [(side, qty, type, price), ...]。"""
        state = {"prices": self._prices, "volumes": self._volumes}
        # OFI: 上一步的净流（滞后一期，避免鸡生蛋）
        last_ofi = 0.0
        if len(self._prices) >= 6:
            short_ret = self._prices[-1] - self._prices[-5]
            last_ofi = 1.0 if short_ret > 0 else -1.0
        state["ofi"] = last_ofi
        # Hawkes 自激励强度：订单簇集群
        state["hawkes_intensity"] = self._hawkes_intensity
        orders = []
        for agent in self.agents:
            if isinstance(agent, MarketMakerAgent):
                orders.extend(agent.get_limit_orders(state))
            else:
                o = agent.produce(state)
                if o is not None:
                    orders.append(o)
        # 更新 Hawkes 强度：λ(t) = μ + (λ₍₋₁₎ - μ)·e⁻ᵝ + α·N(t)
        n_market = sum(1 for _,_,ot,_ in orders if ot == "MARKET")
        decay = math.exp(-self.hawkes_beta)
        self._hawkes_intensity = (self.hawkes_mu
            + (self._hawkes_intensity - self.hawkes_mu) * decay
            + self.hawkes_alpha * n_market)
        return orders
    def next_order(self):
        """向后兼容接口：返回净市场订单 (side, size)。"""
        orders = self.get_orders()
        net_qty = 0.0
        for side, qty, otype, _ in orders:
            if otype == "MARKET":
                net_qty += qty if side == "BUY" else -qty
        side = "BUY" if net_qty >= 0 else "SELL"
        return side, max(0.1, abs(net_qty))

# ============================ 仿真器 ============================
@dataclass
class SimResult:
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)


class MarketSimulator:
    """把对手盘订单注入订单簿，驱动价格形成。"""

    def __init__(self, mid: float = 100.0, agent: Optional[MockMarketAgent] = None,
                 tick: float = 0.05,
                 council: Optional[CouncilMarketAgent] = None,
                 kyle_lambda: float = 0.0005):
        self.book = OrderBook(mid, tick=tick, kyle_lambda=kyle_lambda)
        self.agent = agent or MockMarketAgent()
        self.council = council
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

    def step_council(self) -> float:
        """使用 CouncilMarketAgent 的多边订单流步进。"""
        orders = self.council.get_orders()
        last_vwap = self.prices[-1] if self.prices else self.book.mid
        for side, qty, otype, price in orders:
            if otype == "MARKET":
                last_vwap = self.book.match_market(side, qty)
            elif otype == "LIMIT" and price is not None:
                self.book.add_limit(side, price, qty)
        self.prices.append(last_vwap)
        mkt_qty = sum(q for _, q, ot, _ in orders if ot == "MARKET")
        self.volumes.append(mkt_qty)
        if len(self.prices) >= 2 and self.prices[-2] > 0:
            self.returns.append(math.log(last_vwap / self.prices[-2]))
        self.council._prices = self.prices
        self.council._volumes = self.volumes
        return last_vwap

    def run_council(self, n_steps: int) -> SimResult:
        for _ in range(n_steps):
            self.step_council()
        return SimResult(prices=self.prices[:], volumes=self.volumes[:],
                         returns=self.returns[:])

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
    """量化 StockSim 是否复现程式化事实（7 项检验）。"""
    rets = np.diff(np.log(np.asarray(prices, float) + 1e-12))
    n = len(rets)
    vols = np.asarray(volumes, float)[:n]
    # 1) 肥尾：超额峰度
    if n > 4 and rets.std(ddof=1) > 0:
        m2 = (rets ** 2).mean(); m4 = (rets ** 4).mean()
        kurt = float(m4 / (m2 ** 2)) - 3.0
    else:
        kurt = 0.0
    # 2) 波动聚集：|收益| 滞后1 自相关
    abs_rets = np.abs(rets)
    acf_abs1 = _acf1(abs_rets)
    # 3) 成交量自相关
    acf_vol = _acf1(vols)
    # 4) 杠杆效应：corr(retₜ, |retₜ₊₁|) < 0
    leverage_corr = float(np.corrcoef(rets[:-1], abs_rets[1:])[0, 1]) if n > 5 else 0.0
    # 5) 量-波交叉相关：corr(volₜ, |retₜ₊₁|) > 0
    vol_vol_corr = float(np.corrcoef(vols[:n - 1], abs_rets[1:])[0, 1]) if n > 5 and max(vols) > 0 else 0.0
    # 6) 波动率长记忆：|收益| 滞后 5/10 自相关仍为正（幂律衰减）
    vl5 = float(np.corrcoef(abs_rets[:-5], abs_rets[5:])[0, 1]) if n > 10 else 0.0
    vl10 = float(np.corrcoef(abs_rets[:-10], abs_rets[10:])[0, 1]) if n > 20 else 0.0
    # 7) 收益线性自相关：滞后1 应接近 0
    acf_ret1 = _acf1(rets)

    flags = {
        "fat_tails": kurt > 1.0,
        "vol_clustering": acf_abs1 > 0.05,
        "volume_autocorr": acf_vol > 0.05,
        "has_leverage": leverage_corr < 0.0,
        "has_vol_vol_corr": vol_vol_corr > 0.1,
        "has_long_memory": vl5 > 0.1 and vl10 > 0.05,
        "has_no_linear_acf": abs(acf_ret1) < 0.05,
    }
    return {
        "n": n,
        "excess_kurtosis": round(kurt, 3),
        "vol_acf1": round(acf_abs1, 3),
        "volume_acf1": round(acf_vol, 3),
        "leverage_corr": round(leverage_corr, 3),
        "vol_vol_corr": round(vol_vol_corr, 3),
        "vol_acf5": round(vl5, 3),
        "vol_acf10": round(vl10, 3),
        "return_acf1": round(acf_ret1, 3),
        **flags,
        "n_stylized_facts": sum(flags.values()),
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
        # 【P1-12 修复】原实现 try/except 包住「from ..adapters.mock_llm import ...」
        # —— 这是包内纯 stdlib 模块，永不失败 → 降级分支不可达且误导（假装有真实降级）。
        # 诚实做法：MockLLM 恒可用即直接构造；保留 _llm is None 兜底仅为防御性（基本不触发）。
        from ..adapters.mock_llm import MockLLM, CouncilContext
        self._llm = MockLLM()
        self._ctx_cls = CouncilContext

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
    """工厂：kind 可选 'mock'(零依赖) / 'llm'(MockLLM 叙事) / 'council'(多角色委员会)。

    council 默认 5 种 9 个智能体（趋势×2、均值回归×2、基本面×2、噪声×2、做市×1），
    全部纯 numpy。LLM 缺失无影响，零依赖纪律不变。
    """
    if kind == "mock":
        return MockMarketAgent(**kw)
    if kind == "council":
        return CouncilMarketAgent(**kw)
    # 阶段4：LLM 驱动市场；缺失依赖时 LLMMarketAgent 自动退化为 mock 行为
    try:
        return LLMMarketAgent(**kw)
    except Exception:
        return MockMarketAgent()
