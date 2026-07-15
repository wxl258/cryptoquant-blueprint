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
LEVERAGE = int(os.getenv("CRYPTOQUANT_LEVERAGE", "2"))           # 目标杠杆倍数（会通过 API 通知币安）
# 测试网账户权益（gate 单币/总仓比例 + KillSwitch 日亏比例的基准）。
# 默认 5000 → 固定 200USDT 单币 ≈ 4% 对齐 SINGLE_CAP_PCT，使闸门阈值有意义。
# 注意：POS_SIZE_USDT × LEVERAGE 决定名义敞口，实际保证金 = POS_SIZE_USDT。
EQUITY_USDT = float(os.getenv("CRYPTOQUANT_EQUITY_USDT", "5000"))

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


def _load_paper_signals() -> Dict[str, str]:
    """解析 paper_state.json 产出 {symbol: LONG/SHORT/HOLD}。"""
    path = os.path.join(PAPER_DIR, "paper_state.json")
    if not os.path.exists(path):
        logger.warning("paper_state.json 不存在，跳过本轮测试网同步")
        return {}
    with open(path) as f:
        data = json.load(f)
    signals: Dict[str, str] = {}
    for r in data.get("records", []):
        act = r.get("action", "HOLD")
        if act in ("LONG", "SHORT", "HOLD"):
            signals[r["symbol"]] = act
        else:
            signals[r["symbol"]] = "HOLD"
    return signals


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


def _save_fill_history(history: List[dict]) -> None:
    path = os.path.join(PAPER_DIR, "fill_history.json")
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False)


def _daily_realized_pnl() -> float:
    """当日（自然日 00:00 起）已实现盈亏（USDT）。供 KillSwitch 日亏熔断用。

    按自然日重置，避免把长期累计微亏误触 L2；与 ExecutionEngine._realized_day 同口径。
    """
    history = _load_fill_history()
    today_start = datetime.combine(datetime.now().date(), _dtime.min).timestamp()
    # 【P0 符号修复】已实现 = Σ(卖出台约 proceeds) − Σ(买入成本)：SELL 记正、BUY 记负，
    # 末位**不再**取负。此前多乘 -1 使符号整体反，会把盈利日误判为亏损、误触发 L1 日亏熔断。
    return sum(f["price"] * f["qty"] * (1 if f["side"] == "SELL" else -1)
               for f in history if float(f.get("ts", 0)) >= today_start)


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
                    gate_cfg: GateConfig) -> List[dict]:
    """根据最新信号同步测试网持仓（跨 run 持久化真实持仓/挂单）。

    修复：先拉测试网真实持仓与挂单，否则每轮从空开始会重复开单、无力平仓历史单；
    持仓/挂单按真实状态决策，冲突挂单先撤。
    """
    trades: List[dict] = []
    # 刷新失败（网络/非200/json异常）→ 返回 None；此时内存持仓为空，
    # 若继续会盲目重复开单，故跳过本轮，等下次 cron 重试。
    if adapter.refresh_positions() is None or adapter.refresh_open_orders() is None:
        logger.error("[testnet] 持仓/挂单刷新失败, 跳过本轮, 避免盲目重复开单")
        return trades
    positions_map = {p.symbol: p for p in adapter.query_positions()}
    open_map = {o.symbol: o for o in adapter.query_open()}

    for sym, action in signals.items():
        if sym not in prices or not prices[sym]:
            continue
        price = prices[sym]
        # 首次交易该币前设杠杆（确保币安实际杠杆与配置一致，否则保证金显示 0）
        adapter._ensure_leverage(sym, LEVERAGE)
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
            continue

        # 【P0-1】KillSwitch 熔断：L1+ 暂停新开仓；HOLD 平仓（上面已处理）不受影响
        if not kill_switch.allows_new():
            logger.warning("  [testnet] KillSwitch=%s 暂停新开, 跳过 %s 开仓",
                           kill_switch.level.name, sym)
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
        qty = _fmt_qty(sym, POS_SIZE_USDT * LEVERAGE / price)
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

    return trades


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
    signals = _load_paper_signals()
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
    ks.update(daily_pnl=daily_pnl)
    logger.info("  [风控] equity=%.0f 当日已实现=%.2f daily_pnl=%.2f%% KillSwitch=%s",
                EQUITY_USDT, daily_rp, daily_pnl * 100, ks.level.name)

    # 同步持仓
    trades = sync_positions(adapter, signals, prices, ks, gate_cfg)
    logger.info("  本轮操作 %d 笔", len(trades))

    # 输出
    _write_outputs(adapter, trades, signals, prices, ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
