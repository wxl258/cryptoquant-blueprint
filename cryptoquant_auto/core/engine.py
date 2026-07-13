"""ExecutionEngine：信号 -> 风险闸门 -> 下单 -> 成交 -> 对账 闭环编排。

调试版：用 MockAdapter 跑通整条管线；真实部署时把 MockAdapter 换成
测试网/实盘适配器即可。Fail-closed：恢复必须 manual_resume()。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models import Signal, Order, OrderStatus, OrderType, Position, Direction, Fill
from ..adapters.base import ExchangeAdapter
from ..risk.gate import GateConfig, assert_pre_trade
from ..risk.kill_switch import KillSwitch
from ..risk.kelly import KellyConfig
from ..risk.exec_sl import estimate_liquidation_price, move_to_breakeven
from ..risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..risk.signal_filter import SignalQualityGate, SignalQualityConfig, MarketSnapshot
from ..risk.liquidation_guard import check_liquidation_guard
from ..risk.black_swan import detect_black_swan, Candle
from ..risk.regime import detect_regime
from .order_builder import build_entry_order, build_tp_sl_orders
from .reconcile import reconcile, ReconcileReport


@dataclass
class Decision:
    accepted: bool = False
    order: Optional[Order] = None
    reject: Optional[str] = None


@dataclass
class StepReport:
    processed_fills: int = 0
    reconcile: Optional[ReconcileReport] = None
    events: List[str] = field(default_factory=list)


class ExecutionEngine:
    def __init__(self, adapter: ExchangeAdapter, cfg: GateConfig, ks: KillSwitch,
                 lev: float = 5.0, maker_mode: bool = True,
                 cb: CircuitBreaker = None, quality_gate: SignalQualityGate = None,
                 metacontroller=None, constitution=None):
        self.adapter = adapter
        self.cfg = cfg
        self.ks = ks
        self.lev = lev
        self.maker_mode = maker_mode
        self.cb = cb or CircuitBreaker()                       # 硬熔断（服务器优点）
        self.quality_gate = quality_gate or SignalQualityGate()  # 信号质量前置（服务器优点）
        # 阶段1 脊柱：贝叶斯融合器 + 交易宪法（蓝图）。可选；接了才启用 ingest_meta。
        self.metacontroller = metacontroller
        self.constitution = constitution
        self.expected: Dict[str, Position] = {}     # 意图持仓，按 signal_id 键（跨信号隔离）
        self.processed: set[str] = set()            # 已处理的 fill coid
        self.recent_ret: Dict[str, float] = {}       # 各币近期收益（thermo 闸门）
        self._now: float = 0.0
        self._bar_interval: float = 3600.0          # 每步推进的时钟（秒，默认 1h K）
        self.current_regime: str = getattr(cfg, "regime", "TREND")  # 实时 regime（接真实K线）
        # P0-2/P0-3 修复：引擎自身维护风险度量，驱动 KillSwitch / CircuitBreaker
        self._equity_base: float = getattr(cfg, "equity", 100_000.0)
        self._realized: float = 0.0                  # 累计已实现权益（含杠杆名义盈亏）
        self._peak_equity: float = self._equity_base
        self._loss_streak: int = 0
        # 【P1-9 修复】日间盈亏与日界键：daily_pnl 必须按自然日重置，否则全程累计
        # 会让长期微亏误触 L2(-0.05)，阈值基准错误。
        self._realized_day: float = 0.0
        self._day_key: Optional[str] = None
        self._win_streak: int = 0
        self._price_buf: list = []                    # 【C8】近期价格缓冲，估 ATR sigma 兜底

    # ---- 实时 regime 路由（圆桌共识二 · 2026-07-11：接真实K线，非调试常量）----
    def update_regime(self, prices: dict) -> str:
        """用真实行情（各币收盘价序列）驱动 detect_regime，动态更新 gate 的 regime。

        圆桌纪律：regime 必须由真实价格序列判定，禁止写死常量。CRASH 时 gate
        自动「禁止新开仓」（_regime_mult=0），RANGE 半仓，TREND 全仓 —— 路由真正生效。
        prices: {symbol: [close, ...]}，取每币最近一段序列喂 detect_regime。
        返回更新后的 regime（供审计）。
        """
        best_regime = self.current_regime
        # 多币合并判定：任一币进入 CRASH 即全局 CRASH（保守优先）；否则取波动最高的币的 regime
        for sym, series in prices.items():
            if not isinstance(series, (list, tuple)) or len(series) < 22:
                continue
            r = detect_regime(list(series)).regime
            if r == "CRASH":
                best_regime = "CRASH"
                break
            if r == "RANGE" and best_regime == "TREND":
                best_regime = "RANGE"
        self.current_regime = best_regime
        # 写回 gate 配置，使四闸门（regime_cap / total_cap 缩放）实时生效
        if hasattr(self.cfg, "regime"):
            self.cfg.regime = best_regime
        return best_regime

    # ---- 信号入口（v8 原路径：单信号 → 四闸门 → 下单）----
    def ingest_signal(self, sig: Signal, snapshot: MarketSnapshot = None,
                      ev_est: Optional[float] = None, now: float = None,
                      kc: "KellyConfig" = None) -> Decision:
        return self._submit_signal(sig, snapshot, ev_est, now, kc)

    # ---- 脊柱入口（蓝图阶段1）：多源意见 → 融合 → 宪法校验 → 动作 ----
    def ingest_meta(self, opinions, candidate, entry_price: float,
                    proposed_exposure: Optional[float] = None,
                    snapshot: MarketSnapshot = None, ev_est: Optional[float] = None,
                    now: float = None, kc: "KellyConfig" = None) -> Decision:
        """把贝叶斯脊柱接到执行管线：metacontroller 融合 → 宪法校验 → 建 v8 Signal → 下单。

        仅在 __init__ 注入了 metacontroller + constitution 时可用；否则明确拒。
        - 宪法否决（含 live_capital 硬锁）→ 拒绝，不碰资金；
        - 融合结果为观望（软降级）→ 拒绝（meta_hold），不建仓；
        - 否则把 MetaDecision 落成 v8 Signal，复用全部 v8 风险闸门与下单幂等。
        """
        if self.metacontroller is None or self.constitution is None:
            return Decision(reject="meta_not_configured")
        # 动作常量（与 metacontroller 的 LONG/SHORT/HOLD=0/1/2 对齐）
        from .metacontroller import LONG, SHORT, HOLD
        meta = self.metacontroller.decide(opinions, symbol=candidate.symbol,
                                          proposed_exposure=proposed_exposure)
        verdict = self.constitution.check(meta)
        meta.constitution_ok = verdict.compliant
        meta.violations = verdict.violations
        if not verdict.compliant:
            return Decision(reject="|".join(verdict.violations) or "constitution_reject")
        if meta.action == HOLD:
            return Decision(reject="meta_hold")
        # 把 MetaDecision 落成 v8 Signal（方向 + 由 candidate.atr 推导 SL/TP）
        direction = Direction.LONG if meta.action == LONG else Direction.SHORT
        atr = candidate.atr or max(entry_price * 0.01, 1.0)
        sl_dist = 2.0 * atr
        sign = 1 if direction is Direction.LONG else -1
        sl = entry_price - sign * sl_dist
        tp1 = entry_price + sign * sl_dist * candidate.rr
        tp2 = entry_price + sign * sl_dist * candidate.rr * 1.5
        sig = Signal(symbol=candidate.symbol, tf="1H", direction=direction,
                     entry=entry_price, sl=sl, tp1=tp1, tp2=tp2, rr=candidate.rr,
                     confidence=meta.confidence,
                     signal_id=f"meta_{candidate.symbol}_{int((now or 0))}",
                     atr=atr)
        d = self._submit_signal(sig, snapshot, ev_est, now, kc)
        d.meta = meta                       # 挂载脊柱决策，便于可观测/审计
        return d

    def _submit_signal(self, sig: Signal, snapshot: MarketSnapshot = None,
                       ev_est: Optional[float] = None, now: float = None,
                       kc: "KellyConfig" = None) -> Decision:
        # 【P0-1 修复】R0 宪法硬锁下沉到执行入口：此前 constitution.check 仅由
        # ingest_meta 调用，v8 ingest_signal 直连路径（→本方法）完全绕开 → 硬锁形同虚设。
        # 现于所有下单路径的统一入口强制否决，覆盖 ingest_signal / ingest_meta 两条链路。
        if self.constitution is not None and self.constitution.live_capital:
            return Decision(reject="R0:live_capital=True 禁止任何实盘动作（原型仅沙盒）")
        if now is not None:
            self._now = now                            # P0-3: 推进时钟
        self.cb.feed_signal(self._now)                  # 信号源活跃，刷新熔断 stall
        if self.cb.tripped:
            return Decision(reject=f"circuit:{self.cb.reason}")   # 硬熔断优先
        if not self.ks.allows_new():
            return Decision(reject=f"kill_switch:{self.ks.level.name}")
        qr = self.quality_gate.check(sig, snapshot)     # 信号质量前置闸门（四闸门前）
        if not qr.ok:
            return Decision(reject="|".join(qr.reasons))
        # 【C6 修复】kc=已校准的 KellyConfig（含 OOS 实测 p/b）才用真实胜率；
        # 不传则 kelly_size 内部用保守缺省（0.30），避免 0.53 乐观超配。
        order = build_entry_order(sig, self.cfg, self.maker_mode, kc=kc)
        res = assert_pre_trade(order, list(self.expected.values()), self.cfg,
                               self.recent_ret, ev_est)
        if not res.ok:
            return Decision(reject="|".join(res.reasons))
        # 幂等提交（超时视为未知，按 coid 查单/重发，绝不盲重复）
        try:
            o = self.adapter.submit(order)
        except TimeoutError:
            existing = [x for x in self.adapter.query_open() if x.coid == order.coid]
            if existing:
                o = existing[0]
            else:
                try:
                    o = self.adapter.submit(order)
                except TimeoutError:
                    return Decision(reject="submit_timeout_pending_verify")
        if o.status is OrderStatus.REJECTED:
            return Decision(reject="adapter_reject")
        # P2-2 修复：实际杠杆由订单名义/权益推导，取代硬编码 lev=5.0。
        # quarter-Kelly 下单名义≈4%权益 → 实际杠杆≈1x，无交易所强平（风险由 SL 控），
        # 此时 liq_price=None，liquidation_guard 自动跳过；仅当实际杠杆>1 才建模强平价。
        notional = order.qty * order.price
        actual_lev = notional / max(self._equity_base, 1.0)
        if actual_lev <= 1.0:
            liq = None
        else:
            liq = estimate_liquidation_price(sig.entry, actual_lev, sig.direction)
        # 记录意图持仓（按 signal_id 隔离，避免同币种跨信号碰撞）
        self.expected[sig.signal_id] = Position(
            symbol=sig.symbol, direction=sig.direction, entry_price=sig.entry,
            qty=order.qty, initial_qty=order.qty, sl_price=0.0,
            tp1_price=sig.tp1, tp2_price=sig.tp2, entry_coid=order.coid,
            signal_id=sig.signal_id,
            liq_price=liq, leverage=actual_lev,
        )
        return Decision(accepted=True, order=o)

    # ---- 行情推进 + 状态机 ----
    def step(self, prices: dict, candles_1m: Dict[str, Candle] = None,
             cross_spread: float = 0.0, now: float = None,
             anomaly: dict = None) -> StepReport:
        # 【C8 修复 · 2026-07-12】anomaly: {oi_spike_pct, fr_spike, atr_sigma_spike}
        # 由上层（实盘 WS / 回测 deriv 数据）提供异动值，透传给 KillSwitch 触发
        # 共识#10「OI/FR/ATR 异动即暂停新开」。原实现 _feed_kill_switch 从不传参 → 该分支永不触发。
        # P0-3 修复：仅当显式传入 now（生产实时路径）才推进时钟并检测信号中断熔断；
        # 回测不传 now 时保持 _now 不变，避免按 bar 步进触发 stall 误熔断（不回归）。
        if now is not None:
            self._now = now
            self._rollover_day()                         # 【P1-9】日界检测→重置日间盈亏
            self.cb.check_stall(self._now)              # 信号源中断熔断
            if self.cb.tripped:
                return StepReport(events=[f"circuit_trip:{self.cb.reason}"])
        self.update_regime(prices)                      # 实时 regime 路由（接真实K线）
        self.adapter.simulate_market(prices)
        report = StepReport()
        for f in self.adapter.fills:
            if f.coid in self.processed:
                continue
            self.processed.add(f.coid)
            report.processed_fills += 1
            self._on_fill(f, report)
        # 黑天鹅探测 -> 强制 KillSwitch L3（服务器优点）
        swan, reason = detect_black_swan(candles_1m or {}, cross_spread)
        if swan:
            self.ks.update(black_swan=True)
            report.events.append(f"[black_swan] {reason} -> L3")
        # 记录近期价格用于引擎侧 ATR sigma 估算（若上层未提供 anomaly）
        self._price_buf.append(prices)
        if len(self._price_buf) > 30:
            self._price_buf.pop(0)
        # P0-2 修复：用引擎自维护的实时度量驱动 KillSwitch（不再恒 True）
        self._feed_kill_switch(report, anomaly=anomaly)
        # 生存态减仓执行（圆桌共识#10：L3 → 转稳定币意图 + 持仓等比减至50%）
        surv = self.ks.survival_action()
        if surv["active"]:
            from .order_builder import build_reduce_order
            for sid, pos in list(self.expected.items()):
                sig = self._sig_from_expected(sid)
                if sig is None:
                    continue
                entry_o = Order(
                    coid=f"{pos.symbol}_{sid}_entry", symbol=pos.symbol,
                    side="BUY" if pos.direction is Direction.LONG else "SELL",
                    otype=OrderType.ENTRY, price=pos.entry_price, qty=pos.initial_qty,
                    signal_id=sid, leg="entry",
                )
                ro = build_reduce_order(sig, entry_o, reduce_to=surv["reduce_to"])
                try:
                    self.adapter.submit(ro)
                    report.events.append(
                        f"[{pos.symbol}] 生存态减仓→{surv['reduce_to']:.0%}"
                        f"{' + 转稳定币' if surv['to_stablecoin'] else ''}")
                except Exception:
                    pass
        # 强平价预警巡检（仅告警，不改仓；服务器优点）
        alerts = check_liquidation_guard(list(self.expected.values()), prices)
        for a in alerts:
            report.events.append(f"[liq_guard] {a}")
        report.reconcile = reconcile(list(self.expected.values()), self.adapter.query_positions())
        return report

    def _rollover_day(self) -> None:
        """【P1-9】按自然日重置日间盈亏：daily_pnl 必须基于当日盈亏而非全程累计。"""
        import datetime as _dt
        key = _dt.datetime.utcfromtimestamp(self._now).strftime("%Y-%m-%d")
        if self._day_key is None:
            self._day_key = key
        elif key != self._day_key:
            self._day_key = key
            self._realized_day = 0.0

    def _feed_kill_switch(self, report: StepReport, anomaly: dict = None) -> None:
        """用引擎自维护的权益/连亏/保证金度量刷新 KillSwitch（P0-2 修复核心）。

        【C8 修复】anomaly={oi_spike_pct,fr_spike,atr_sigma_spike} 透传给 ks.update，
        激活共识#10「异动即暂停新开」；若上层未提供，则用近期价格缓冲估算 ATR sigma 兜底。
        """
        peak_dd = (self._peak_equity - self._equity_base) / self._equity_base
        peak_dd = min(0.0, peak_dd)
        # 【P1-9】daily_pnl 用「当日」盈亏，而非全程累计（_realized_day 在日界重置）
        daily_pnl = self._realized_day / self._equity_base
        # 保证金率：适配器暴露则用之，否则维持安全默认（不谎报风险）
        margin_ratio = getattr(self.adapter, "margin_ratio", 99.0)
        # 异动值：优先用上层显式传入，否则用价格缓冲估算 ATR sigma
        an = anomaly or {}
        oi_spike = an.get("oi_spike_pct", 0.0)
        fr_spike = an.get("fr_spike", 0.0)
        atr_sigma = an.get("atr_sigma_spike", self._est_atr_sigma())
        if (abs(daily_pnl) > 1e-12 or self._loss_streak or peak_dd < 0
                or oi_spike > 0 or fr_spike > 0 or atr_sigma > 0):
            self.ks.update(daily_pnl=daily_pnl, peak_dd=peak_dd,
                           loss_streak=self._loss_streak, margin_ratio=margin_ratio,
                           oi_spike_pct=oi_spike, fr_spike=fr_spike,
                           atr_sigma_spike=atr_sigma)
            if not self.ks.allows_new():
                report.events.append(f"[kill_switch] 升级至 {self.ks.level.name}")

    def _est_atr_sigma(self) -> float:
        """用近期价格缓冲估算 ATR sigma（兜底，当上层未提供 anomaly 时）。"""
        if len(self._price_buf) < 10:
            return 0.0
        # 取 BTC 或任意首币的近期收盘价序列估算波动率 sigma
        series = []
        for pmap in self._price_buf:
            v = next(iter(pmap.values()), None)
            if v is not None:
                series.append(v)
        if len(series) < 10:
            return 0.0
        rets = [abs(series[i] - series[i - 1]) / series[i - 1] for i in range(1, len(series))]
        if not rets:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        sigma = var ** 0.5
        # 转 sigma 倍数（相对每日典型波动），粗略：>3 倍视为异动
        return sigma / max(mean, 1e-9) if mean > 0 else 0.0

    def _on_fill(self, f, report: StepReport) -> None:
        # coid 形如 symbol_signalid_leg（三部分以下划线分隔）
        parts = f.coid.split("_")
        sym = parts[0]
        leg = parts[-1]
        sig_id = "_".join(parts[1:-1])
        order = next((o for o in self.adapter.query_open() if o.coid == f.coid), None)
        if order is None:
            return
        pos = self.expected.get(sig_id)   # 按 signal_id 隔离，避免同币种跨信号碰撞

        if leg == "entry":
            # 入场成交 -> 挂 TP1/TP2/执行级SL 子单
            if pos is None:
                return
            sig = self._sig_from_expected(sig_id)
            if sig is None:
                return
            children = build_tp_sl_orders(sig, order, atr=sig.atr or max(sig.entry * 0.005, 1.0))
            for c in children:
                self.adapter.submit(c)
            report.events.append(f"[{sym}] 入场成交@{f.price:.2f}，挂 TP1/TP2/SL")
        elif leg == "tp1":
            # TP1 平50% -> 移 SL 至保本位
            if pos is None or pos.signal_id != sig_id:
                return
            report.events.append(f"[{sym}] TP1 平50%@{f.price:.2f}，SL 移至保本位")
            old_sl_coid = f"{sym}_{sig_id}_sl"
            old_sl_order = next((o for o in self.adapter.query_open() if o.coid == old_sl_coid), None)
            current_sl = old_sl_order.price if old_sl_order else pos.entry_price
            breakeven = move_to_breakeven(current_sl, pos.entry_price, pos.direction, fee_buffer=0.0)
            side = "BUY" if pos.direction is Direction.SHORT else "SELL"
            new_sl = Order(coid=f"{sym}_{sig_id}_sl_v2", symbol=sym, side=side,
                           otype=OrderType.SL, price=breakeven, qty=order.qty * 0.5,
                           signal_id=sig_id, leg="sl", parent_coid=order.coid)
            self.adapter.cancel(old_sl_coid)
            self.adapter.submit(new_sl)
            self.expected[sig_id].state = "BREAKEVEN"
        elif leg in ("tp2", "sl"):
            # 全部平仓 -> 清理该信号剩余子单，移除意图持仓
            if pos is None or pos.signal_id != sig_id:
                return
            action = "TP2 全平" if leg == "tp2" else "SL 止损全平"
            report.events.append(f"[{sym}] {action}@{f.price:.2f}")
            # P0-2/P0-3 修复：记录已实现盈亏，驱动 KillSwitch 连亏/回撤 + CircuitBreaker 连亏熔断
            sign = 1 if pos.direction is Direction.LONG else -1
            pnl_frac = (f.price - pos.entry_price) * f.qty * sign / max(self._equity_base, 1.0)
            self._realized += pnl_frac
            self._realized_day += pnl_frac          # 【P1-9】同步累计日间盈亏
            self._peak_equity = max(self._peak_equity, self._equity_base + self._realized)
            if pnl_frac < 0:
                self._loss_streak += 1
                self._win_streak = 0
            else:
                self._loss_streak = 0
                self._win_streak += 1
            self.cb.on_trade_close(pnl_frac)            # P0-3: 连亏/异常硬熔断
            if self.cb.tripped:
                report.events.append(f"[circuit] {self.cb.reason}")
            for o in self.adapter.query_open():
                if o.coid.startswith(f"{sym}_{sig_id}_") and o.coid != f.coid:
                    self.adapter.cancel(o.coid)
            self.expected.pop(sig_id, None)

    def _sig_from_expected(self, sig_id: str) -> Optional[Signal]:
        # 从意图持仓反推一个最小 Signal 用于构建子单（调试简化）
        p = self.expected.get(sig_id)
        if p is None:
            return None
        return Signal(
            symbol=p.symbol, tf="1H", direction=p.direction, entry=p.entry_price,
            sl=p.sl_price or p.entry_price, tp1=p.tp1_price, tp2=p.tp2_price,
            rr=2.0, confidence=0.6, signal_id=sig_id,
            atr=max(p.entry_price * 0.01, 1.0),
        )

    def manual_resume(self) -> None:
        """Fail-closed：人工 ACK 恢复。

        【P1-6 修复】除 KillSwitch 自身复位外，必须同步清零引擎侧累计权益度量
        （_realized / _peak_equity / _loss_streak / _realized_day），否则下一轮
        _feed_kill_switch 仍以「累计亏损」重判 daily_pnl/peak_dd → 立刻重新跳闸 L1/L2，
        人工 ACK 形同虚设。
        """
        self.ks.manual_resume()
        self._realized = 0.0
        self._peak_equity = self._equity_base
        self._loss_streak = 0
        self._win_streak = 0
        self._realized_day = 0.0
        self._day_key = None
