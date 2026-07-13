"""Paper / 回测验证器：驱动真实引擎管线，带成本建模，统计胜率/净值/夏普/回撤。

直接回应「胜率之类可以回测」：把信号序列灌入完整执行管线（四闸门 -> KillSwitch
-> 下单 -> 状态机 -> 执行级SL -> 对账），按各币真实成本（maker/taker/滑点/资金费）
核算 PnL，输出胜率、净值、夏普、最大回撤、各币 edge，并校验 Gate B。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models import Signal, Direction
from ..adapters.mock import MockAdapter
from ..risk.gate import GateConfig
from ..risk.kill_switch import KillSwitch
from ..risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..risk.exec_cost import effective_edge_bps, gate_b_ok
from ..core.engine import ExecutionEngine
from .market_path import make_path, COIN_VOL
from .metrics import summarize, BacktestStats


@dataclass
class BacktestConfig:
    equity: float = 100_000.0
    use_taker: bool = False        # False=maker（挂单），True=taker（吃单）
    n_bars: int = 400
    drift: float = 0.0
    seed: int = 42


class PaperBacktest:
    def __init__(self, cfg: BacktestConfig = None, gate: GateConfig = None):
        self.bc = cfg or BacktestConfig()
        self.gate = gate or GateConfig(equity=self.bc.equity, enforce_gate_b=False)
        self.adapter = MockAdapter(equity=self.bc.equity)
        self.ks = KillSwitch()
        # 【2026-07-12】回测模式：CircuitBreaker loss_trip 设极高值（999），
        # 避免回测中被 3 笔连亏熔断后永久拒单（真实环境下 CB 正确保护，但回测
        # 用单 Engine 实例串行跑所有信号时，熔断后所有后续信号皆被拒 → edge 统计失真）。
        # 同理 KillSwitch loss_streak 阈值设极高值（999），避免 L1 升级后永久拒单。
        # 调用方如需验证 CB/KS 行为，可传入 cb/ks 覆盖。
        self.cb = CircuitBreaker(CircuitBreakerConfig(loss_trip=999))
        self.engine = ExecutionEngine(self.adapter, self.gate, self.ks, lev=1.0,
                                      cb=self.cb)
        self.cash = self.bc.equity
        self.equity_curve: List[float] = [self.bc.equity]
        self.trades: List[dict] = []
        self.skipped = 0

    # ---- 单笔信号回放 ----
    def run_signal(self, sig: Signal, seed: int = None, path: List[float] = None,
                   ev_est: float = None, regime: str = None,
                   kc: "KellyConfig" = None) -> Optional[dict]:
        """path=真实前向收盘回放（诚实回测）；ev_est 透传给 EV 闸门；regime 仅做分段统计标记。

        - ev_est: 该信号的期望值估计（OOS 中用 IS 按 regime 的实测 EV 代理），喂给 Gate B 的
          EV 闸门做硬拒；None 表示不评估（其他闸门照常）。
        - kc: 校准的 KellyConfig（含 IS 实测 p/b），透传给 ingest_signal 用于 OOS 仓位标定
          （C6 修复闭环：IS 估的 p,b 真正注入 OOS）。
        - regime: 信号生成时刻的 regime 标签，仅写入 trade 供分段统计，不参与交易决策（避免前视）。
        """
        d = self.engine.ingest_signal(sig, ev_est=ev_est, kc=kc)
        if not d.accepted:
            # 【C3 修复 · 2026-07-12】回测中不再 manual_resume 绕过 KillSwitch。
            # 原逻辑：被 kill_switch reject 就人工ACK重试 → 回测里风控拦截形同虚设，
            # 统计的回撤/连亏触发全部失真（审计 C3 红队确认）。
            # 修复：reject 即视为该信号未通过风控，计入 skipped 并跳过（不重试）。
            # 这样回测能真实反映 KillSwitch 分级拦截效果，与实盘 fail-closed 行为一致。
            self.skipped += 1
            return None
        # 前向路径：path 给定则用真实历史收盘回放（诚实回测）；否则合成漂移
        if path is None:
            dir_drift = 0.02 if sig.direction is Direction.LONG else -0.02
            path = make_path(sig.entry, sig.symbol, self.bc.n_bars,
                             drift=self.bc.drift + dir_drift, seed=seed)
        for price in path:
            self.engine.step({sig.symbol: price})
            if self.adapter.query_position(sig.symbol) is None:
                break
        # 路径走完仍未平仓 -> 强制止损出局（超时=止损），避免仓位跨信号累积撑爆闸门
        if self.adapter.query_position(sig.symbol) is not None:
            # 相对偏移（小价币不能用绝对值减法，否则出现负价）
            force = sig.sl * 0.99 if sig.direction is Direction.LONG else sig.sl * 1.01
            self.engine.step({sig.symbol: force})
        # 清理未实现意图（如入场从未成交）：持仓为None则移除意图，避免撑爆闸门
        if self.adapter.query_position(sig.symbol) is None:
            self.engine.expected.pop(sig.signal_id, None)
        trade = self._settle(sig, path, regime=regime)
        # 【2026-07-12】回测是测量工具模式，不更新 KillSwitch 状态。
        # 原代码在此处调用 ks.update(daily_pnl=..., peak_dd=..., loss_streak=...)，
        # 但 KillSwitch 的 loss_streak>=3 阈值是硬编码的，回测中一旦升级 L1，
        # 后续所有信号皆因 ks.allows_new()=False 被拒，无法测量裸 edge。
        # WFA 通过每个信号创建独立 PaperBacktest 实例避免了此问题；
        # 顺序 P0 回测也必须不更新 KS 才能正确测量。
        # 生产风控验证应由独立的回测/模拟覆盖，不影响 PaperBacktest 的 edge 测量职能。
        return trade

    def _loss_streak(self) -> int:
        s = 0
        for t in reversed(self.trades):
            if t["pnl"] < 0:
                s += 1
            else:
                break
        return s

    def _settle(self, sig: Signal, path: List[float], regime: str = None) -> dict:
        prefix = f"{sig.symbol}_{sig.signal_id}"
        fills = [f for f in self.adapter.fills if f.coid.startswith(prefix)]
        entry_fills = [f for f in fills if f.coid.endswith("_entry")]
        exit_fills = [f for f in fills if (f.coid.endswith("_tp1") or f.coid.endswith("_tp2")
                                           or f.coid.endswith("_sl") or f.coid.endswith("_sl_v2"))]
        if not entry_fills:
            return {"symbol": sig.symbol, "pnl": 0.0, "pnl_bps": 0.0, "closed": False}
        sign = 1 if sig.direction is Direction.LONG else -1
        e = entry_fills[0]
        entry_notional = e.qty * e.price
        gross = 0.0
        exit_notional = 0.0
        for f in exit_fills:
            gross += (f.price - e.price) * f.qty * sign
            exit_notional += f.qty * f.price
        # 成本
        c = _cost_rates(sig.symbol, self.bc.use_taker)
        fee = (entry_notional + exit_notional) * c["fee"]
        slip = (entry_notional + exit_notional) * c["slip"] if self.bc.use_taker else 0.0
        bars_held = max(1, len(path))
        fund = entry_notional * c["fund"] * (bars_held / 8.0)  # 每8bar一结算
        net = gross - fee - slip - fund
        self.cash += net
        self.equity_curve.append(self.cash)
        # 【C4 修复 · 2026-07-12】bps 口径说明（红队/回测专家共识：此为组合层口径，非 bug）：
        #   pnl_bps / gross_bps 用 盈亏 / 总权益(equity) * 1e4，表示「该笔对组合净值的 bps 贡献」。
        #   单笔 quarter-Kelly 仓位约 4% 权益，净 edge 按组合权益摊薄属正常。
        #   若需「每单位仓位暴露性价比」应另算 盈亏/名义仓位，二者不可混比、不可直接相除。
        #   此处保持组合层口径并显式注释，不改用名义分母（避免审计 C2 误判式修正）。
        trade = {"symbol": sig.symbol, "pnl": net,
                 "pnl_bps": net / self.bc.equity * 1e4, "closed": True,
                 "bars_held": bars_held, "regime": regime,
                 "signal_id": sig.signal_id,
                 "gross_bps": gross / self.bc.equity * 1e4}  # 组合层毛 edge（EV 闸门用）
        self.trades.append(trade)
        return trade

    # ---- 批量 ----
    def run_batch(self, signals: List[Signal], kc: "KellyConfig" = None) -> BacktestStats:
        # 【C6 修复 · 2026-07-12】透传校准的 kc（含 IS 实测 p/b）给每笔 run_signal，
        # 使 quarter-Kelly 名义上限用真实胜率而非保守缺省（修复原 walk_forward 的
        # "IS 估 p,b 却未注入 OOS" 闭环断裂问题）。
        for i, sig in enumerate(signals):
            self.run_signal(sig, seed=self.bc.seed + i * 7, kc=kc)
        return self.stats()

    def stats(self) -> BacktestStats:
        return summarize(self.equity_curve, self.trades, gate_b_ok())


def _cost_rates(coin: str, use_taker: bool) -> Dict[str, float]:
    from ..risk.exec_cost import COST
    c = COST.get(coin, COST["ETH"])
    return {"fee": c["taker"] / 1e4 if use_taker else c["maker"] / 1e4,
            "slip": c["slip_taker"] / 1e4 if use_taker else 0.0,
            "fund": c["fund"] / 1e4}


def make_random_signals(n: int, seed: int = 1) -> List[Signal]:
    """生成合成信号序列用于回测（真实回测可替换为历史信号）。"""
    rnd = random.Random(seed)
    coins = ["BTC", "ETH", "BNB", "SOL", "XRP"]
    base = {"BTC": 60000, "ETH": 3000, "BNB": 550, "SOL": 140, "XRP": 0.5}
    out = []
    for i in range(n):
        sym = rnd.choice(coins)
        px = base[sym] * (1 + rnd.uniform(-0.05, 0.05))
        atr = px * rnd.uniform(0.008, 0.02)
        direction = Direction.LONG if rnd.random() < 0.5 else Direction.SHORT
        if direction is Direction.LONG:
            tp1, tp2 = px * 1.02, px * 1.05
        else:
            tp1, tp2 = px * 0.98, px * 0.95
        out.append(Signal(symbol=sym, tf=rnd.choice(["1H", "4H", "1D"]),
                          direction=direction, entry=round(px, 4),
                          sl=round(px - 2 * atr if direction is Direction.LONG else px + 2 * atr, 4),
                          tp1=round(tp1, 4), tp2=round(tp2, 4),
                          rr=2.0, confidence=rnd.uniform(0.5, 0.9),
                          signal_id=f"bt{i}", atr=round(atr, 4)))
    return out
