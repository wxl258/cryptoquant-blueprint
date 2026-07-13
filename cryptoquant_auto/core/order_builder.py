"""信号 -> 订单 构建（含入场单与 TP/SL 子单，确定性 coid）。"""
from __future__ import annotations

from ..models import Signal, Order, OrderType, Direction
from ..risk.gate import GateConfig
from ..risk.exec_sl import exec_sl_price


def kelly_size(sig: Signal, cfg: GateConfig, kc: "KellyConfig" = None,
               score: float = None, adx: float = None, atr_pct: float = None,
               trade_state: dict = None) -> float:
    """quarter-Kelly 仓位（P1a）：用 KellyConfig 估名义，受单币上限 + 置信度缩放。

    【C6 修复 · 2026-07-12】支持注入 OOS 实测 (p,b)：调用方传入已校准的 kc
    （kc.calibrated=True）即可用真实胜率算名义上限；不传则用保守缺省
    （KellyConfig() 默认 win_rate_est=0.30，非乐观 0.53）。
    【C8 修复 · 2026-07-12】接通 position_sizing.calc_position_pct 7步链（原死代码）：
    在 quarter-Kelly 名义上限基础上，用 score/adx/atr_pct/trade_state 做二次缩放
    （ADX加成/连亏递减/盈利递减/周末/SR/波动率/ATR封顶），取 min 约束，不超配。
    返回以「币」为单位的 qty。
    """
    from ..risk.kelly import kelly_nominal, KellyConfig
    from ..risk.position_sizing import calc_position_pct
    kc = kc or KellyConfig()  # 默认保守缺省（已非 0.53 乐观值）
    if not kc.calibrated:
        # 未校准：仅用保守缺省，且强制 quarter-Kelly 收敛（防超配）
        kc = KellyConfig(win_rate_est=min(kc.win_rate_est, 0.30),
                         payoff_ratio_est=kc.payoff_ratio_est,
                         kelly_frac=kc.kelly_frac)
    conf_mult = 0.5 + sig.confidence          # 0.5~1.5
    # quarter-Kelly 名义上限（frac=kc.kelly_frac=0.25）
    kelly_nom = kelly_nominal(cfg.equity, kc.win_rate_est, kc.payoff_ratio_est,
                              frac=kc.kelly_frac, cap_pct=cfg.single_cap_pct)
    nominal = min(kelly_nom * conf_mult, cfg.equity * cfg.single_cap_pct)
    nominal = max(nominal, cfg.equity * 0.003)   # 下限 0.3%
    # 【C8】7步链二次缩放：仅当提供 score/adx/atr_pct 时才启用（避免无数据硬算）
    if score is not None and adx is not None:
        ps_res = calc_position_pct(
            score=score, adx=adx,
            atr_pct=(atr_pct if atr_pct is not None else (sig.atr / sig.entry * 100 if sig.entry else 0.0)),
            trade_state=trade_state or {},
            rr=getattr(kc, "payoff_ratio_est", 2.0),
        )
        # ps_res.pct 是占权益百分比；换算名义与 kelly 取 min
        ps_nominal = cfg.equity * ps_res.pct / 100.0
        nominal = min(nominal, ps_nominal)
    return nominal / sig.entry


def build_entry_order(sig: Signal, cfg: GateConfig, maker_mode: bool = True,
                      kc: "KellyConfig" = None) -> Order:
    side = "BUY" if sig.direction is Direction.LONG else "SELL"
    qty = kelly_size(sig, cfg, kc=kc)
    # 注：mock 下单以信号入场价挂单；真实所由 post_only(GTX) 保证不吃 taker。
    return Order(
        coid=f"{sig.symbol}_{sig.signal_id}_entry",
        symbol=sig.symbol,
        side=side,
        otype=OrderType.ENTRY,
        price=sig.entry,
        qty=qty,
        signal_id=sig.signal_id,
        leg="entry",
        post_only=maker_mode,
    )


def build_reduce_order(sig: Signal, entry_order: Order, reduce_to: float = 0.5) -> "Order":
    """生存态减仓单：方向与原持仓相反，数量缩减至 reduce_to 比例（圆桌共识#10）。

    Fail-closed：仅减仓、不自动全平；to_stablecoin 意图由调用方标记。
    """
    side = "SELL" if sig.direction is Direction.LONG else "BUY"   # 平仓方向=开仓反向
    return Order(
        coid=f"{sig.symbol}_{sig.signal_id}_reduce",
        symbol=sig.symbol,
        side=side,
        otype=OrderType.REDUCE,
        price=sig.entry,
        qty=entry_order.qty * (1.0 - reduce_to),
        signal_id=sig.signal_id,
        leg="reduce",
        parent_coid=entry_order.coid,
    )


def build_tp_sl_orders(sig: Signal, entry_order: Order, atr: float | None = None) -> list[Order]:
    """入场成交后挂 TP1/TP2/执行级SL 子单。

    【C1 修复 · 2026-07-12】子单 qty 必须按平仓比例拆分，不能都用全量 entry_order.qty：
      - TP1 = 平 50%（qty = 0.5 * entry）
      - TP2 = 平剩余 50%（qty = 0.5 * entry）
      - SL  = 兜底平仓（qty = entry，覆盖未成交情况）
    原实现三子单均用全量 → _close_or_reduce 把 pos.qty 一次减到 0 → 后续 leg 永不触发，
    状态机崩坏（审计 C1 红队复核确认）。
    """
    atr = atr or sig.atr
    side = "BUY" if sig.direction is Direction.SHORT else "SELL"   # 平仓方向与开仓相反
    sl = exec_sl_price(sig.entry, atr, sig.direction, k=2.0) if atr else sig.sl
    qty_full = entry_order.qty
    qty_half = qty_full * 0.5
    children = []
    # TP1 平 50%，TP2 平剩余 50%，SL 用全量兜底（防 TP 未触发时仍保本/止损）
    for leg, price, qty in (("tp1", sig.tp1, qty_half),
                             ("tp2", sig.tp2, qty_half),
                             ("sl", sl, qty_full)):
        children.append(Order(
            coid=f"{sig.symbol}_{sig.signal_id}_{leg}",
            symbol=sig.symbol,
            side=side,
            otype=OrderType.TP if leg.startswith("tp") else OrderType.SL,
            price=price,
            qty=qty,
            signal_id=sig.signal_id,
            leg=leg,
            parent_coid=entry_order.coid,
        ))
    return children
