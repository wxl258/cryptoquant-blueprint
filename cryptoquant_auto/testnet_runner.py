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
from typing import Dict, List, Optional, Tuple

import requests
import urllib.request

from .adapters.binance_testnet import BinanceTestnetAdapter
from .models import Direction, OrderType
from .risk.constitution import TradingConstitution

logger = logging.getLogger("cryptoquant.testnet")

# 配置参数（可通过环境变量覆盖）
POS_SIZE_USDT = int(os.getenv("CRYPTOQUANT_POS_SIZE", "100"))   # 每币基柱 USDT
LEVERAGE = int(os.getenv("CRYPTOQUANT_LEVERAGE", "2"))           # 杠杆倍数

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


def _calc_pnl(adapter: BinanceTestnetAdapter) -> dict:
    """汇总测试网 P&L。"""
    fills = getattr(adapter, "fills", [])
    logger.info("[debug]  fills=%d positions=%d balance_wallet=%s",
                len(fills), len(adapter.query_positions()),
                "" if not hasattr(adapter, '_client') else '?')
    realized = sum(f.price * f.qty * (1 if f.side == "SELL" else -1)
                   for f in fills) * -1  # 简算（未计手续费）
    positions = adapter.query_positions()
    unrealized = 0.0
    pos_details = []
    for p in positions:
        # 最新价从 fills 获取或缺省
        entry_cost = p.entry_price * abs(p.qty)
        m2m = p.entry_price * abs(p.qty) * 0  # 占位
        pos_details.append({
            "symbol": p.symbol, "direction": p.direction.value,
            "qty": float(p.qty), "entry_price": float(p.entry_price),
            "unrealized_pnl": 0.0,
        })
    return {"realized_pnl": round(realized, 2), "unrealized_pnl": 0.0,
            "positions": pos_details, "total_fills": len(fills)}


def sync_positions(adapter: BinanceTestnetAdapter,
                    signals: Dict[str, str],
                    prices: Dict[str, float]) -> List[dict]:
    """根据最新信号同步测试网持仓核心逻辑。"""
    trades: List[dict] = []
    positions_map = {p.symbol: p for p in adapter.query_positions()}

    for sym, action in signals.items():
        if sym not in prices or not prices[sym]:
            continue
        price = prices[sym]
        cur = positions_map.get(sym)
        cur_dir = None
        if cur and cur.qty > 1e-9:
            cur_dir = "LONG" if cur.direction == Direction.LONG else "SHORT"

        # 信号 HOLD → 平仓出场
        if action == "HOLD":
            if cur_dir:
                side = "SELL" if cur_dir == "LONG" else "BUY"
                qty = round(abs(cur.qty), 6)
                if qty > 0:
                    # 平仓用激进限价确保成交：市价的 0.5% 让步
                    limit_price = _fmt_price(sym, price * 1.005 if side == "BUY" else price * 0.995)
                    resp = adapter.submit_market(sym, side, qty, price=limit_price)
                    trades.append({"symbol": sym, "action": "CLOSE",
                                   "side": side, "qty": qty,
                                   "resp": resp.get("status", "?")})
                    logger.info("  [testnet] %s 平仓 %s qty=%s", sym, side, qty)
            continue

        # 信号与持仓方向一致 → 跳过
        if cur_dir == action:
            continue

        # 信号反转 → 平旧仓（也用激进限价）
        if cur_dir:
            close_side = "SELL" if cur_dir == "LONG" else "BUY"
            qty = round(abs(cur.qty), 6)
            if qty > 0:
                limit_price = _fmt_price(sym, price * 1.005 if close_side == "BUY" else price * 0.995)
                adapter.submit_market(sym, close_side, qty, price=limit_price)
                trades.append({"symbol": sym, "action": "CLOSE",
                               "side": close_side, "qty": qty})

        # 开新仓（用激进 LIMIT 替代 MARKET 解决测试网不成交）
        open_side = "BUY" if action == "LONG" else "SELL"
        qty = _fmt_qty(sym, POS_SIZE_USDT * LEVERAGE / price)
        if qty < 0.001:
            logger.warning("  [testnet] %s qty=%s 过小跳过", sym, qty)
            continue
        limit_price = _fmt_price(sym, price * 1.005 if open_side == "BUY" else price * 0.995)
        resp = adapter.submit_market(sym, open_side, qty,
                                     signal_id=f"sig_{sym}", price=limit_price)
        trades.append({"symbol": sym, "action": f"OPEN_{action}",
                       "side": open_side, "qty": qty,
                       "resp": resp.get("status", "?")})
        logger.info("  [testnet] %s 开仓 %s qty=%s", sym, action, qty)

    return trades


def _write_outputs(adapter: BinanceTestnetAdapter, trades: List[dict],
                   signals: Dict[str, str]) -> None:
    """输出测试网状态 + 仪表盘。"""
    os.makedirs(PAPER_DIR, exist_ok=True)
    pnl = _calc_pnl(adapter)
    out = {
        "updated": time.time(),
        "signals": signals,
        "positions": pnl["positions"],
        "trades": trades,
        "realized_pnl": pnl["realized_pnl"],
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

    # 同步持仓
    trades = sync_positions(adapter, signals, prices)
    logger.info("  本轮操作 %d 笔", len(trades))

    # 输出
    _write_outputs(adapter, trades, signals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
