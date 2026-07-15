"""测试网真实模拟交易器（Binance USDT-M 测试网，假钱真成交）。

流程（每次 cron 调用）：
  1. 读取 paper/paper_state.json 获取最新 AI 信号（LONG/SHORT/HOLD）
  2. 对比测试网当前持仓方向
  3. 方向一致 → 不操作；方向反转 → 市价平旧仓 → 市价开新方向
  4. 跟踪每笔成交、持仓规模、浮动/已实现盈亏
  5. 输出 paper/testnet_state.json + paper/testnet_dashboard.md

密钥：.testnet_secret（export CRYPTOQUANT_TESTNET_KEY / CRYPTOQUANT_TESTNET_SECRET）
安全：脚本内含 testnet.binancefuture.com 硬检查，改动 BASE 即拒绝构建。
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, time as _dtime
from typing import Dict, List, Optional, Tuple

import requests
import urllib.request

from .adapters.binance_testnet import BinanceTestnetAdapter
from .models import Direction, OrderType, Order
from .risk.constitution import TradingConstitution
from .risk.gate import GateConfig, assert_pre_trade
from .risk.kill_switch import KillSwitch

logger = logging.getLogger("cryptoquant.testnet")

# 配置参数（可通过环境变量覆盖）
POS_SIZE_USDT = int(os.getenv("CRYPTOQUANT_POS_SIZE", "100"))   # 每币基柱名义 USDT（保证金基数）


# === Option B: CVaR 动态仓位护栏（proposed_exposure 接入执行端） ===
# 名义敞口由信号侧 CVaR proposed_exposure 推导；杠杆仅设保证金不回乘。
SINGLE_CAP_PCT = float(os.getenv("CRYPTOQUANT_SINGLE_CAP_PCT", "0.12"))       # 单币名义敞口上限 12%
PORTFOLIO_CAP_PCT = float(os.getenv("CRYPTOQUANT_PORTFOLIO_CAP_PCT", "1.0"))   # 组合名义敞口上限 100%
MIN_NOTIONAL_USDT = float(os.getenv("CRYPTOQUANT_MIN_NOTIONAL", "5.0"))        # 最小名义敞口护栏 5U
EPS_EXPOSURE = 1e-9    # 敞口比较极小值，避免浮点抖动
DYNAMIC_SIZING = os.getenv("CRYPTOQUANT_DYNAMIC_SIZING", "1") in ("1", "true", "True", "yes", "on")
LEVERAGE = int(os.getenv("CRYPTOQUANT_LEVERAGE", "2"))           # 目标杠杆倍数（会通过 API 通知币安）
# 测试网账户权益（gate 单币/总仓比例 + KillSwitch 日亏比例的基准）。
# 默认 5000 → 固定 200USDT 单币 ≈ 4% 对齐 SINGLE_CAP_PCT，使闸门阈值有意义。
# 注意：POS_SIZE_USDT × LEVERAGE 决定名义敞口，实际保证金 = POS_SIZE_USDT。
EQUITY_USDT = float(os.getenv("CRYPTOQUANT_EQUITY_USDT", "5000"))

# SL/TP 参数（P0/P2 止盈止损）
SL_ATR_K = 2.0                 # 止损距离 = k × atr_est
DEFAULT_ATR_PCT = 0.02         # 默认 ATR 占价百分比（无 atr 数据时回退）
TP_PCT = 0.08                  # 止盈百分比（相对入场价）
BREAKEVEN_PROFIT_PCT = 0.03    # 浮盈达此比例移 SL 到保本位

# 币安 USDT-M 合约 LOT_SIZE 步长（api/v1/exchangeInfo 可查，硬编码省一次 API 调用）
_STEP_SIZES = {
    "BTC": 0.001, "ETH": 0.01, "SOL": 0.1,
    "BNB": 0.01, "XRP": 1.0, "TRX": 10.0,
    "DOGE": 1.0, "ADA": 1.0, "AVAX": 0.01,
    "LINK": 0.01, "TON": 0.1, "SUI": 0.1,
}

# 限价单价格步长（PRICE_FILTER.tickSize，粗略但够用）
_PRICE_STEPS = {
    "BTC": 0.1, "ETH": 0.01, "SOL": 0.001,
    "BNB": 0.01, "XRP": 0.0001, "TRX": 0.00001,
    "DOGE": 0.00001, "ADA": 0.0001, "AVAX": 0.01,
    "LINK": 0.001, "TON": 0.001, "SUI": 0.001,
}


def _fmt_qty(symbol: str, qty: float) -> float:
    """按步长向下取整，拒绝超精度错误（-1111）。"""
    step = _STEP_SIZES.get(symbol, 0.001)
    adjusted = int(qty / step) * step
    precision = max(0, -int(math.floor(math.log10(step))))
    return round(adjusted, precision)


def _fmt_price(symbol: str, price: float) -> float:
    """按 PRICE_FILTER.tickSize 截断价格精度（防 -4014）。"""
    step = _PRICE_STEPS.get(symbol, 0.01)
    precision = max(0, -int(math.floor(math.log10(step))))
    return round(price, precision)

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")


def _calc_leverage(confidence: float, regime: str, ks_level) -> int:
    """动态杠杆：行情越好（高置信、强趋势、无风险）→ 杠杆越高。

    公式：leverage = 10 × regime_mult × conf_mult × ks_mult [clamped 1~20]
    - regime_mult: TREND=1.5 / RANGE=1.0 / CRASH=0.5
    - conf_mult: 0.5 + confidence (conf 0~1 → mult 0.5~1.5)
    - ks_mult: L0=1.0 / WARN=0.8 / L1=0.5 / L2=0.3 / L3=0.1

    示例：
      TREND+高置信(0.8)+L0  = 10×1.5×1.3×1.0 = 19.5 → 20x 🟢
      RANGE+中置信(0.4)+L0  = 10×1.0×0.9×1.0 = 9 → 9x 🟡
      CRASH+低置信(0.2)+L2  = 10×0.5×0.7×0.3 = 1.05 → 1x 🔴
    """
    regime_mult = {"TREND": 1.5, "RANGE": 1.0, "CRASH": 0.5}.get(regime, 1.0)
    conf_mult = max(0.5, min(1.5, 0.5 + confidence))
    ks_val = ks_level.value if hasattr(ks_level, "value") else float(ks_level)
    ks_mult = {0: 1.0, 0.5: 0.8, 1.0: 0.5, 2.0: 0.3, 3.0: 0.1}.get(ks_val, 1.0)
    raw = 10.0 * regime_mult * conf_mult * ks_mult
    return max(1, min(20, int(round(raw))))



def _ks_mult(ks_level) -> float:
    """KillSwitch 级别 -> 敞口缩放系数（与 _calc_leverage 同源）。"""
    ks_val = ks_level.value if hasattr(ks_level, "value") else float(ks_level)
    return {0: 1.0, 0.5: 0.8, 1.0: 0.5, 2.0: 0.3, 3.0: 0.1}.get(ks_val, 1.0)


def _target_notional(sym: str, details: dict, ks_level, equity: float) -> float:
    """Option B: 由 CVaR proposed_exposure 推导目标名义敞口（纯函数，可单测）。

    名义敞口 = clamp(proposed_exposure * equity * _ks_mult(level), 0, SINGLE_CAP_PCT*equity)
    - 杠杆不回乘：返回值为名义敞口，调用方直接 notional/price 算 qty；
      杠杆仅经 adapter._ensure_leverage 设保证金占用（margin = notional/lev）。
    - 单币护栏：不超过 SINGLE_CAP_PCT * equity。
    """
    if equity <= 0:
        return 0.0
    expo = float((details or {}).get("proposed_exposure", 0.0))
    if expo <= 0:
        return 0.0
    cap_single = SINGLE_CAP_PCT * equity
    raw = expo * equity * _ks_mult(ks_level)
    return min(max(raw, 0.0), cap_single)


def _load_paper_signals() -> tuple[Dict[str, str], Dict[str, dict]]:
    """解析 paper_state.json 产出 (signals, details)。

    signals: {symbol: LONG/SHORT/HOLD}
    details: {symbol: {confidence, regime, proposed_exposure, cvar_pct}}
    """
    path = os.path.join(PAPER_DIR, "paper_state.json")
    if not os.path.exists(path):
        logger.warning("paper_state.json 不存在，跳过本轮测试网同步")
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    signals: Dict[str, str] = {}
    details: Dict[str, dict] = {}
    for r in data.get("records", []):
        sym = r["symbol"]
        act = r.get("action", "HOLD")
        if act in ("LONG", "SHORT", "HOLD"):
            signals[sym] = act
        else:
            signals[sym] = "HOLD"
        details[sym] = {
            "confidence": r.get("confidence", 0.3),
            "regime": r.get("regime", "RANGE"),
            "proposed_exposure": r.get("proposed_exposure", 0.0),
            "cvar_pct": r.get("cvar_pct", 0.0),
        }
    return signals, details


def _submit_sl_order(adapter, symbol: str, side: str, entry: float,
                     direction, qty: float) -> None:
    """开仓后挂 STOP_MARKET 止损条件单（P0 止损）。

    sl_price = exec_sl_price(entry, atr_est, direction, k=SL_ATR_K)
    无实际 atr 数据时用 entry × DEFAULT_ATR_PCT 估算。
    """
    from .risk.exec_sl import exec_sl_price
    atr_est = entry * DEFAULT_ATR_PCT
    sl = exec_sl_price(entry, atr_est, direction, k=SL_ATR_K)
    sl_fmt = _fmt_price(symbol, sl)
    sl_side = "SELL" if side == "BUY" else "BUY"
    order = Order(coid=f"sl_{symbol}_{int(time.time())}", symbol=symbol,
                  side=sl_side, otype=OrderType.SL, price=sl_fmt, qty=qty,
                  signal_id=f"sl_{symbol}")
    result = adapter.submit(order)
    if result.status == OrderStatus.REJECTED:
        logger.warning("  [SL] %s 止损挂单被拒 sl=%.2f", symbol, sl)
    else:
        logger.info("  [SL] %s 止损已挂 @%.2f (ATR≈%.4f k=%.1f)", symbol, sl, atr_est, SL_ATR_K)


def _submit_tp_order(adapter, symbol: str, side: str, entry: float,
                     qty: float) -> None:
    """开仓后挂 TAKE_PROFIT_MARKET 止盈条件单（P2 止盈）。"""
    tp = entry * (1 + TP_PCT) if side == "BUY" else entry * (1 - TP_PCT)
    tp_fmt = _fmt_price(symbol, tp)
    tp_side = "SELL" if side == "BUY" else "BUY"
    order = Order(coid=f"tp_{symbol}_{int(time.time())}", symbol=symbol,
                  side=tp_side, otype=OrderType.TP, price=tp_fmt, qty=qty,
                  signal_id=f"tp_{symbol}")
    result = adapter.submit(order)
    if result.status == OrderStatus.REJECTED:
        logger.warning("  [TP] %s 止盈挂单被拒 tp=%.2f", symbol, tp)
    else:
        logger.info("  [TP] %s 止盈已挂 @%.2f (+%.1f%%)", symbol, tp, TP_PCT * 100)


def _close_stale_positions(adapter: BinanceTestnetAdapter,
                            prices: Dict[str, float],
                            kill_switch: KillSwitch,
                            trades: List[dict]) -> None:
    """解析 paper_state.json 产出 (signals, details)。

    signals: {symbol: LONG/SHORT/HOLD}
    details: {symbol: {confidence, regime, proposed_exposure, cvar_pct}}
    """
    path = os.path.join(PAPER_DIR, "paper_state.json")
    if not os.path.exists(path):
        logger.warning("paper_state.json 不存在，跳过本轮测试网同步")
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    signals: Dict[str, str] = {}
    details: Dict[str, dict] = {}
    for r in data.get("records", []):
        sym = r["symbol"]
        act = r.get("action", "HOLD")
        if act in ("LONG", "SHORT", "HOLD"):
            signals[sym] = act
        else:
            signals[sym] = "HOLD"
        details[sym] = {
            "confidence": r.get("confidence", 0.3),
            "regime": r.get("regime", "RANGE"),
            "proposed_exposure": r.get("proposed_exposure", 0.0),
            "cvar_pct": r.get("cvar_pct", 0.0),
        }
    return signals, details


def _fetch_prices(symbols: List[str]) -> Dict[str, float]:
    """从币安公开行情拉最新价格（与 paper_runner 同源）。"""
    prices: Dict[str, float] = {}
    for s in symbols:
        try:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={s}USDT"
            req = urllib.request.Request(url, headers={"User-Agent": "cryptoquant-testnet/0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode("utf-8"))
                prices[s] = float(data["price"])
        except Exception as e:
            logger.warning("[prices] %s 获取失败: %s", s, e)
    return prices


def _load_fill_history() -> List[dict]:
    """读取跨 run 成交历史（paper/fill_history.json）。"""
    path = os.path.join(PAPER_DIR, "fill_history.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return []
    return []


MAX_FILLS = 200  # fill_history 最大条目（P2-6 清理）


def _save_fill_history(history: List[dict]) -> None:
    path = os.path.join(PAPER_DIR, "fill_history.json")
    if len(history) > MAX_FILLS:
        history = history[-MAX_FILLS:]
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False)


def _daily_realized_pnl() -> float:
    """当日（自然日 00:00 起）已平仓回合的真实盈亏（USDT）。供 KillSwitch 日亏熔断用。

    FIFO 逐币匹配 BUY/SELL，仅统计「当日发生平仓(SELL)且对应 BUY 已匹配」的回合盈亏，
    并扣平仓腿双边手续费；仍持有仓位的建仓成本不计入。与行业「已实现盈亏」一致，
    避免把开仓占用的本金误记为当日亏损、虚假触发 L2_REDUCE（修复纯开仓日误冻）。
    """
    history = _load_fill_history()
    today_start = datetime.combine(datetime.now().date(), _dtime.min).timestamp()
    lots: Dict[str, list] = {}            # symbol -> [[price, qty], ...] 未平 BUY 批次
    daily = 0.0
    for f in sorted(history, key=lambda x: float(x.get("ts", 0))):
        ts = float(f.get("ts", 0))
        sym = f["symbol"]
        price = float(f["price"])
        qty = float(f["qty"])
        notional = price * qty
        if f["side"] == "BUY":
            lots.setdefault(sym, []).append([price, qty])
        else:  # SELL 平仓：与最早未平 BUY 批次 FIFO 配对
            remaining = qty
            for lot in lots.get(sym, []):
                if remaining <= 0:
                    break
                take = min(lot[1], remaining)
                if ts >= today_start:
                    daily += (price - lot[0]) * take
                lot[1] -= take
                remaining -= take
            if ts >= today_start:
                daily -= notional * FEE_RATE_TAKER
    return daily


def _close_position(adapter: BinanceTestnetAdapter, sym: str,
                    cur, price: float, trades: List[dict]) -> None:
    """按真实持仓量平仓（激进限价），记录回执。"""
    cur_dir = "LONG" if cur.direction == Direction.LONG else "SHORT"
    side = "SELL" if cur_dir == "LONG" else "BUY"
    qty = round(abs(cur.qty), 6)
    if qty <= 0:
        return
    # 市价单平仓：测试网部分交易对 PRICE_FILTER 过期(minPrice 远高于现价)，
    # 限价单会被价格过滤器拒绝(-4014)；市价单绕开价格过滤器且实测可成交。
    resp = adapter.submit_market(sym, side, qty)
    trades.append({"symbol": sym, "action": "CLOSE",
                   "side": side, "qty": qty,
                   "resp": resp.get("status", "ERROR"),
                   "ok": bool(resp.get("ok", False)),
                   "code": resp.get("code")})
    logger.info("  [testnet] %s 平仓 %s qty=%s", sym, side, qty)


# 【P2-2】手续费：Binance USDT-M 吃单费率 0.0004（测试网同口径）。市价单为吃单。
# 盈亏计入双边手续费，使仪表盘盈亏可信（此前忽略手续费，盈亏虚高）。
FEE_RATE_TAKER = 0.0004


def _calc_pnl(adapter: BinanceTestnetAdapter,
              prices: Optional[Dict[str, float]] = None) -> dict:
    """汇总测试网 P&L；成交数与已实现盈亏跨 run 持久化（修复'累计'口径）。

    【P2-2】已实现盈亏扣双边手续费：realized = Σ(SELL proceeds) − Σ(BUY cost) − Σ(fee)。
    未实现盈亏为毛 MTM（开仓费已在已实现的买入腿扣除），与行业惯例一致。
    """
    if prices is None:
        prices = {}
    run_fills = getattr(adapter, "fills", [])
    history = _load_fill_history()
    for f in run_fills:
        history.append({"coid": f.coid, "symbol": f.symbol, "side": f.side,
                        "price": float(f.price), "qty": float(f.qty),
                        "ts": float(f.ts)})
    _save_fill_history(history)
    logger.info("[debug] run_fills=%d history=%d positions=%d",
                len(run_fills), len(history), len(adapter.query_positions()))
    # 【P0 符号修复】已实现 = Σ(SELL proceeds) − Σ(BUY cost) − 双边手续费；
    # SELL 正、BUY 负，末位不再取负。与 _daily_realized_pnl 同口径。
    buy_cost = 0.0
    sell_proceeds = 0.0
    total_fee = 0.0
    for f in history:
        notional = f["price"] * f["qty"]
        total_fee += notional * FEE_RATE_TAKER
        if f["side"] == "SELL":
            sell_proceeds += notional
        else:
            buy_cost += notional
    realized = sell_proceeds - buy_cost - total_fee
    positions = adapter.query_positions()
    pos_details = []
    unrealized_total = 0.0
    for p in positions:
        qty = float(p.qty)
        entry = float(p.entry_price)
        direction = p.direction.value if hasattr(p.direction, "value") else str(p.direction)
        mark = prices.get(p.symbol)
        if mark is not None and qty > 1e-9:
            upnl = (entry - mark) * qty if direction == "SHORT" else (mark - entry) * qty
        else:
            upnl = 0.0
        unrealized_total += upnl
        pos_details.append({
            "symbol": p.symbol, "direction": direction,
            "qty": qty, "entry_price": entry,
            "unrealized_pnl": round(upnl, 2),
        })
    return {"realized_pnl": round(realized, 2), "unrealized_pnl": round(unrealized_total, 2),
            "total_fees": round(total_fee, 2),
            "positions": pos_details, "total_fills": len(history)}


def sync_positions(adapter: BinanceTestnetAdapter,
                    signals: Dict[str, str],
                    prices: Dict[str, float],
                    kill_switch: KillSwitch,
                    gate_cfg: GateConfig,
                    details: Dict[str, dict] = None) -> List[dict]:
    """根据最新信号同步测试网持仓（跨 run 持久化真实持仓/挂单）。

    修复：先拉测试网真实持仓与挂单，否则每轮从空开始会重复开单、无力平仓历史单；
    持仓/挂单按真实状态决策，冲突挂单先撤。
    """
    trades: List[dict] = []
    # 【P0-1 修复】刷新失败 → 返回 None（main 据此跳过写入脏数据）；
    # 此前返回空 trades → _write_outputs 仍被调用 → 空仓位覆盖 testnet_state.json。
    if adapter.refresh_positions() is None or adapter.refresh_open_orders() is None:
        logger.error("[testnet] 持仓/挂单刷新失败, 跳过本轮, 避免盲目重复开单")
        return None
    positions_map = {p.symbol: p for p in adapter.query_positions()}
    open_map = {o.symbol: o for o in adapter.query_open()}

    portfolio_used = 0.0    # 组合累计名义敞口累加器（Option B 护栏）
    for sym, action in signals.items():
        if sym not in prices or not prices[sym]:
            continue
        price = prices[sym]
        # 动态杠杆：根据行情/置信/KillSwitch 算最佳杠杆
        dd = (details or {}).get(sym, {})
        lev = _calc_leverage(dd.get("confidence", 0.3), dd.get("regime", "RANGE"),
                            kill_switch.level)
        adapter._ensure_leverage(sym, lev)
        cur = positions_map.get(sym)
        cur_dir = None
        if cur and cur.qty > 1e-9:
            cur_dir = "LONG" if cur.direction == Direction.LONG else "SHORT"
        op = open_map.get(sym)
        open_dir = ("LONG" if op.side == "BUY" else "SHORT") if op else None

        # 信号 HOLD → 平仓出场 + 撤销该币挂单
        if action == "HOLD":
            if cur_dir:
                _close_position(adapter, sym, cur, price, trades)
            if op:
                adapter.cancel(op.coid)
            continue

        # 信号与持仓方向一致 → 跳过（不重复开）
        if cur_dir == action:
            # 【P0/P2】检查浮盈→移SL到保本位（如果当前没挂或已触发）
            if cur_dir and cur:
                # 尝试从 open_map 查当前 SL 订单
                current_sl_coid = f"sl_{sym}_"
                has_old_sl = any(k.startswith(current_sl_coid) or "sl" in k for k in open_map)
                upnl_pct = (price - cur.entry_price) / cur.entry_price if cur_dir == "LONG" else (cur.entry_price - price) / cur.entry_price
                if upnl_pct >= BREAKEVEN_PROFIT_PCT and has_old_sl:
                    # 取消旧 SL，挂保本 SL
                    logger.info("  [breakeven] %s 浮盈+%.1f%%, SL 移至保本位", sym, upnl_pct * 100)
                    for coid, o in list(adapter.query_open().items() if hasattr(adapter.query_open(), 'items') else []):
                        if o.coid.startswith(f"sl_{sym}_") or "sl" in o.coid:
                            adapter.cancel(o.coid)
                    be_sl = cur.entry_price + 0.0  # 保本（多头）
                    order = Order(coid=f"sl_{sym}_be_{int(time.time())}", symbol=sym,
                                  side="SELL", otype=OrderType.SL, price=_fmt_price(sym, be_sl),
                                  qty=abs(cur.qty), signal_id=f"sl_{sym}")
                    adapter.submit(order)
            continue

        # 【P0-1】KillSwitch 熔断：L1+ 暂停新开仓；L2+ 同时平现有仓位（真减仓）
        if not kill_switch.allows_new():
            logger.warning("  [testnet] KillSwitch=%s 暂停新开, 跳过 %s 开仓",
                           kill_switch.level.name, sym)
            # 【P0-2】L2+ 减仓语义：即使信号与持仓同方向也强制平仓
            if kill_switch.level.value >= 2.0 and cur_dir:
                _close_position(adapter, sym, cur, price, trades)
                logger.warning("  [testnet]   L2+ 减仓 %s (KillSwitch=%s)", sym, kill_switch.level.name)
            continue

        # 方向反转 → 先平旧仓 + 撤冲突挂单，再开新仓
        if cur_dir:
            _close_position(adapter, sym, cur, price, trades)
        if open_dir and open_dir != action:
            adapter.cancel(op.coid)

        # 开新仓（市价单，绕开测试网过期价格过滤器）；已有同方向持仓则跳过，避免重复
        open_side = "BUY" if action == "LONG" else "SELL"
        if open_dir == action:
            logger.info("  [testnet] %s 已有同方向挂单, 跳过开仓", sym)
            continue
        # === Option B: CVaR 动态仓位（proposed_exposure 定名义敞口，杠杆不回乘） ===
        if DYNAMIC_SIZING and float((dd or {}).get("proposed_exposure", 0.0)) > EPS_EXPOSURE:
            notional = _target_notional(sym, dd, kill_switch.level, EQUITY_USDT)
            # 组合护栏：剩余可分配名义敞口
            remaining = PORTFOLIO_CAP_PCT * EQUITY_USDT - portfolio_used
            notional = min(notional, max(0.0, remaining))
            if notional >= MIN_NOTIONAL_USDT:
                qty = _fmt_qty(sym, notional / price)   # 杠杆不回乘
                portfolio_used += notional
                logger.info("  [dynamic] %s 目标名义=%.2fU 杠杆=%dx 组合累计=%.2fU",
                            sym, notional, lev, portfolio_used)
            else:
                logger.info("  [testnet] %s 组合护栏耗尽/低于最小护栏, 跳过开仓", sym)
                continue
        else:
            # 无 CVaR 建议或动态关闭 -> 回退旧路径（POS_SIZE x lev）
            qty = _fmt_qty(sym, POS_SIZE_USDT * lev / price)
        if qty < 0.001:
            logger.warning("  [testnet] %s qty=%s 过小跳过", sym, qty)
            continue
        # 【P0-1】开仓前过四闸门（单币/总仓/regime/beta/过热）；Gate B 默认关闭，
        # 仅作纪律护栏，不触发"edge 未校准→全空仓"。
        order = Order(coid=f"sig_{sym}", symbol=sym, side=open_side, qty=qty,
                      price=price, otype=OrderType.ENTRY, signal_id=f"sig_{sym}")
        gate_res = assert_pre_trade(order, list(positions_map.values()), gate_cfg, {})
        if not gate_res.ok:
            logger.warning("  [testnet] %s 闸门拒绝新开: %s",
                           sym, "; ".join(gate_res.reasons))
            continue
        # 市价单开仓：同上，绕开测试网过期价格过滤器，实测可成交。
        resp = adapter.submit_market(sym, open_side, qty,
                                     signal_id=f"sig_{sym}")
        trades.append({"symbol": sym, "action": f"OPEN_{action}",
                       "side": open_side, "qty": qty,
                       "resp": resp.get("status", "ERROR"),
                       "ok": bool(resp.get("ok", False)),
                       "code": resp.get("code")})
        logger.info("  [testnet] %s 开仓 %s qty=%s", sym, action, qty)
        # 【P0】挂止损条件单（STOP_MARKET）
        dir_obj = Direction.LONG if action == "LONG" else Direction.SHORT
        _submit_sl_order(adapter, sym, open_side, price, dir_obj, qty)
        # 【P2】挂止盈条件单（TAKE_PROFIT_MARKET）
        _submit_tp_order(adapter, sym, open_side, price, qty)

    return trades


TTL_HOURS = 72  # 持仓超过此时间自动平仓（P1-5）


def _close_stale_positions(adapter: BinanceTestnetAdapter,
                            prices: Dict[str, float],
                            kill_switch: KillSwitch,
                            trades: List[dict]) -> None:
    """根据 fill_history 检查每笔持仓的入场时间，超时自动平仓。

    从 adapter.query_positions() 获取真实持仓，从 fill_history 反推入场时间。
    超过 TTL_HOURS 的仓位强制平仓，不受信号方向影响。
    """
    now = time.time()
    history = _load_fill_history()
    # 按 symbol 找最近一次 BUY 入场（假设最近一次 BUY 是当前持仓的开仓）
    entry_ts: Dict[str, float] = {}
    for f in history:
        if f["side"] == "BUY":
            entry_ts[f["symbol"]] = max(entry_ts.get(f["symbol"], 0), f["ts"])
    cutoff = now - TTL_HOURS * 3600
    for p in adapter.query_positions():
        sym = str(p.symbol)
        opened = entry_ts.get(sym, 0)
        if opened > 0 and opened < cutoff:
            logger.warning("  [TTL] %s 持仓超 %.0fh (入场=%.0fs), 自动平仓", sym, TTL_HOURS, opened)
            _close_position(adapter, sym, p, prices.get(sym, 0), trades)


def _write_outputs(adapter: BinanceTestnetAdapter, trades: List[dict],
                   signals: Dict[str, str],
                   prices: Optional[Dict[str, float]] = None,
                   kill_switch: Optional[KillSwitch] = None) -> None:
    """输出测试网状态 + 仪表盘。"""
    os.makedirs(PAPER_DIR, exist_ok=True)
    pnl = _calc_pnl(adapter, prices)
    out = {
        "updated": time.time(),
        "signals": signals,
        "positions": pnl["positions"],
        "trades": trades,
        "realized_pnl": pnl["realized_pnl"],
        "unrealized_pnl": pnl["unrealized_pnl"],
        "total_fees": pnl["total_fees"],
        "total_fills": pnl["total_fills"],
    }
    with open(os.path.join(PAPER_DIR, "testnet_state.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 仪表盘
    lines = [
        "# CryptoQuant 测试网模拟交易仪表盘",
        f"> 更新：{time.strftime('%Y-%m-%d %H:%M:%S')} · 测试网假钱 · 仅信号跟踪",
        "",
        "## 当前持仓",
        "| 币种 | 方向 | 数量 | 入场价 | 浮盈 |",
        "|------|------|------|--------|------|",
    ]
    for p in pnl["positions"]:
        lines.append(f"| {p['symbol']} | {p['direction']} | {p['qty']} "
                     f"| {p['entry_price']:.2f} | {p['unrealized_pnl']:.2f} |")
    if not pnl["positions"]:
        lines.append("| — | 无持仓 | — | — | — |")
    lines += [
        "",
        f"**已实现盈亏**: {pnl['realized_pnl']:.2f} USDT",
        f"**未实现盈亏(浮盈)**: {pnl['unrealized_pnl']:.2f} USDT",
        f"**累计手续费**: {pnl['total_fees']:.2f} USDT",
        f"**KillSwitch**: {kill_switch.level.name if kill_switch else 'N/A'}",
        f"**累计成交笔数**: {pnl['total_fills']}",
        "",
        "## 最近操作",
        "| 币种 | 动作 | 方向 | 数量 | 状态 |",
        "|------|------|------|------|------|",
    ]
    for t in trades[-20:]:
        lines.append(f"| {t['symbol']} | {t['action']} | {t.get('side','')} "
                     f"| {t.get('qty','')} | {t.get('resp','')} |")
    if not trades:
        lines.append("| — | — | — | — | — |")
    with open(os.path.join(PAPER_DIR, "testnet_dashboard.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("测试网仪表盘已更新（%d 笔成交）", pnl["total_fills"])


def main() -> int:
    from .util.logging_setup import setup_logging
    setup_logging()

    # 读取密钥
    key = os.getenv("CRYPTOQUANT_TESTNET_KEY")
    sec = os.getenv("CRYPTOQUANT_TESTNET_SECRET")
    if not key or not sec:
        logger.error("缺少 CRYPTOQUANT_TESTNET_KEY/SECRET（设于 .testnet_secret）")
        return 1

    # 读取信号
    signals, paper_detail = _load_paper_signals()
    if not signals:
        logger.info("paper_state.json 无信号，跳过本轮")
        return 0
    logger.info("测试网执行器启动 | 信号数=%d", len(signals))

    # 拉最新价格
    symbols = list(signals.keys())
    prices = _fetch_prices(symbols)
    logger.info("  价格获取 %d/%d 个", len(prices), len(symbols))

    # 构建适配器
    constitution = TradingConstitution(live_capital=False)
    adapter = BinanceTestnetAdapter(key, sec, constitution=constitution)

    # 【P0-1】风控接线：gate(四闸单币/总仓) + KillSwitch(日亏熔断)
    gate_cfg = GateConfig(equity=EQUITY_USDT, enforce_gate_b=False)
    ks = KillSwitch()
    daily_rp = _daily_realized_pnl()
    daily_pnl = daily_rp / EQUITY_USDT if EQUITY_USDT > 0 else 0.0
    # 【P1-4】KillSwitch 需 peak_dd 才能使 L3 条件可达
    total_history = _load_fill_history()
    total_pnl = sum(f["price"] * f["qty"] * (1 if f["side"] == "SELL" else -1) for f in total_history)
    total_pnl -= sum(f["price"] * f["qty"] * FEE_RATE_TAKER for f in total_history)  # 扣手续费
    current_equity = EQUITY_USDT + total_pnl
    peak_dd = max(0.0, (EQUITY_USDT - max(current_equity, EQUITY_USDT)) / EQUITY_USDT)  # 0 = 无回撤
    ks.update(daily_pnl=daily_pnl, peak_dd=-peak_dd)
    logger.info("  [风控] equity=%.0f 当日=%.2f daily_pnl=%.2f%% peak_dd=%.2f%% KillSwitch=%s",
                EQUITY_USDT, daily_rp, daily_pnl * 100, peak_dd * 100, ks.level.name)
    # 【P0-3】KillSwitch 降级告警
    if ks.level.value >= 0.5:
        logger.error("⛔ KillSwitch=%s 降级中(日亏=%.2f%% peak_dd=%.2f%%)",
                     ks.level.name, daily_pnl * 100, peak_dd * 100)

    # 【P1-5】仓位 TTL 检查：超过 72h 的仓位强制平仓
    ttl_trades: List[dict] = []  # 收集 TTL 平仓记录
    _close_stale_positions(adapter, prices, ks, ttl_trades)
    logger.info("  [TTL] 超时平仓 %d 笔", len(ttl_trades))

    # 同步持仓
    trades = sync_positions(adapter, signals, prices, ks, gate_cfg, details=paper_detail)
    # 【P0-1 修复】refresh 失败 → 跳过写入脏数据，保留上轮健康状态
    if trades is None:
        logger.error("[testnet] 刷新失败，跳过本轮写入，保留上轮状态")
        return 1
    trades = (ttl_trades or []) + (trades or [])
    logger.info("  本轮操作 %d 笔", len(trades))

    # 输出
    _write_outputs(adapter, trades, signals, prices, ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
