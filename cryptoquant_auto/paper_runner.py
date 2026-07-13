"""纸面模拟运行器（解放原型 · 只模拟 · 绝不下单）。

把原型从「验证/演示」形态解放为持续运行的纸面系统：

  行情(只读) → 9 维特征 → 四角色议会(自动 mock/真LLM) → 宪法硬锁校验 → 纸面决策日志

安全边界（fail-closed，架构级不可绕过）：
  - LIVE_CAPITAL 必须为 False（宪法 R0）。一旦被置 True，启动即退出，绝不碰真钱。
  - 数据源只用「历史回放」或「交易所只读公开 REST」，不接任何可下单的实盘适配器。
  - 全程不调用 submit/cancel 等下单接口；产出只落 paper/ 目录（日志 + 仪表盘）。

运行：
  python3 -m cryptoquant_auto.paper_runner --once          # 单次（沙箱/调试）
  python3 -m cryptoquant_auto.paper_runner --loop --interval 300   # 持续（服务器 cron/守护）
  python3 -m cryptoquant_auto.paper_runner --source gateio  # 接只读实时行情（服务器）

依赖：stage2_features / meta.agents / risk.constitution / adapters.real_llm（均已就位）。
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ---- 硬锁：纸面原型的唯一合法值（置 True 即启动即退）----
LIVE_CAPITAL = False

from .stage2_features import (
    assemble_feature, FEATURE_NAMES, _load_history, _load_deriv,
)
from .meta.agents import FourRoleCouncil, LONG, SHORT, HOLD
from .meta.memory import FinMemMemory
from .adapters.real_llm import get_llm, RealLLM
from .risk.constitution import TradingConstitution
from .risk.regime import detect_regime

ACTION_TO_INT = {"LONG": 0, "SHORT": 1, "HOLD": 2}

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")


# ---------------------------------------------------------------------------
# 数据源（只读，均不可下单）
# ---------------------------------------------------------------------------
class DataSource:
    def symbols(self) -> List[str]:
        raise NotImplementedError

    def snapshot(self) -> Dict[str, dict]:
        """返回 {symbol: {feat: np.ndarray(9), regime: str, price: float, ts: int}}。"""
        raise NotImplementedError


def _regime_of(closes: List[float]) -> str:
    try:
        return detect_regime(closes).regime
    except Exception:
        return "RANGE"


class HistoryDataSource(DataSource):
    """回放 history_cache.json（已有真实 1h K 线）+ deriv_data.json（fr/oi）。

    取每个币最近一段窗口，算出「当前」9 维特征。沙箱即可跑，零网络。
    """

    def __init__(self, warmup: int = 240, horizon: int = 12):
        self.warmup = warmup
        self.horizon = horizon
        self._hist = _load_history()
        self._deriv = _load_deriv()

    def symbols(self) -> List[str]:
        return list(self._hist.keys())

    def snapshot(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for sym, entry in self._hist.items():
            k1h = entry.get("1h", [])
            if len(k1h) < self.warmup + self.horizon + 1:
                continue
            i = len(k1h) - 1
            w = k1h[:i + 1]
            closes = [c["c"] for c in w]
            # fr/oi 取最近值（历史回放近似）
            fr = fr_delta = oi_pct = 0.0
            d = self._deriv.get(sym)
            if d:
                fr_list = sorted(d.get("fr", []), key=lambda x: x[0])
                if fr_list:
                    fr = float(fr_list[-1][1])
                oi_list = sorted(d.get("oi", []), key=lambda x: x[0])
                if len(oi_list) >= 2 and oi_list[-2][1]:
                    oi_pct = (oi_list[-1][1] - oi_list[-2][1]) / oi_list[-2][1]
            fng = entry.get("fng", {})
            day = (k1h[i]["t"] // 86400) * 86400
            fg = fng.get(day, 50)
            feat = assemble_feature(closes, w, fr, fr_delta, oi_pct, fg, i)
            out[sym] = {
                "feat": np.array(feat, dtype=float),
                "regime": _regime_of(closes),
                "price": float(k1h[i]["c"]),
                "ts": int(k1h[i]["t"]),
            }
        return out


class GateioPublicDataSource(DataSource):
    """只读公开 REST（无需 API 密钥）拉实时行情。仅生产服务器用，沙箱默认不启。

    端点：Gate.io USDT 永续 candlesticks / contracts + alternative.me 恐慌贪婪。
    全部只读、不签名、不下单。任一币拉取失败则跳过该币，不影响整体。
    """

    BASE = "https://api.gateio.ws/api/v4/futures/usdt"
    FNG = "https://api.alternative.me/fng/?limit=1"
    SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX",
               "DOGE", "ADA", "AVAX", "LINK", "TON", "SUI"]

    def __init__(self, warmup: int = 240, limit: int = 500):
        self.warmup = warmup
        self.limit = limit
        self._fng = 50.0

    @staticmethod
    def _get_json(url: str, timeout: float = 10.0):
        req = urllib.request.Request(url, headers={"User-Agent": "cryptoquant-paper/0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _refresh_fng(self):
        try:
            self._fng = float(self._get_json(self.FNG).get("data", [{}])[0].get("value", 50))
        except Exception:
            pass

    def symbols(self) -> List[str]:
        return self.SYMBOLS

    def snapshot(self) -> Dict[str, dict]:
        self._refresh_fng()
        out: Dict[str, dict] = {}
        for s in self.SYMBOLS:
            try:
                sym = f"{s}_USDT"
                now_ts = int(__import__("time").time())
                url = (f"{self.BASE}/candlesticks?contract={sym}&interval=1h"
                       f"&from={now_ts - 86400 * 30}&limit={self.limit}")
                raw = self._get_json(url)
                # Gate.io 返回字典或数组格式，兼容两者
                k1h = []
                for r in raw:
                    if isinstance(r, dict):
                        k1h.append({"t": int(r["t"]), "o": float(r["o"]),
                                    "h": float(r["h"]), "l": float(r["l"]),
                                    "c": float(r["c"]), "v": float(r.get("v", 0))})
                    else:
                        k1h.append({"t": int(r[0]), "o": float(r[5]),
                                    "h": float(r[3]), "l": float(r[4]),
                                    "c": float(r[2]), "v": float(r[1])})
                if len(k1h) < self.warmup + 2:
                    continue
                i = len(k1h) - 1
                w = k1h[:i + 1]
                closes = [c["c"] for c in w]
                fr = 0.0
                try:
                    info = self._get_json(f"{self.BASE}/contracts/{sym}")
                    fr = float(info.get("funding_rate", 0.0) or 0.0)
                except Exception:
                    pass
                fg = self._fng
                feat = assemble_feature(closes, w, fr, 0.0, 0.0, fg, i)
                out[s] = {
                    "feat": np.array(feat, dtype=float),
                    "regime": _regime_of(closes),
                    "price": float(k1h[i]["c"]),
                    "ts": int(k1h[i]["t"]),
                }
            except Exception as e:
                print(f"  [warn] {s} 拉取失败: {e}")
                continue
        return out


# ---------------------------------------------------------------------------
# 纸面决策（喂给宪法做合规校验的轻量载体）
# ---------------------------------------------------------------------------
class _PaperDecision:
    def __init__(self, action_int: int, rationale: str, confidence: float):
        self.action = action_int
        self.rationale = rationale
        self.confidence = confidence
        self.proposed_exposure = None


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    pass


def run_once(source: DataSource, council: FourRoleCouncil,
             constitution: TradingConstitution, memory: FinMemMemory) -> List[dict]:
    snap = source.snapshot()
    llm_is_real = isinstance(council.llm, RealLLM)
    records: List[dict] = []
    for sym, d in snap.items():
        verdict = council.decide(
            sym, d["feat"], d["regime"], spi_surprise=0.0, record=True)
        action_int = ACTION_TO_INT.get(verdict.action, 2)
        const = constitution.check(
            _PaperDecision(action_int, " ".join(verdict.rationale), verdict.confidence))
        rec = {
            "symbol": sym,
            "regime": d["regime"],
            "market_state": verdict.market_state,
            "action": verdict.action,
            "confidence": round(verdict.confidence, 4),
            "rationale": verdict.rationale,
            "vetoes": verdict.vetoes,
            "constitution_compliant": const.compliant,
            "violations": const.violations,
            "llm": "real" if llm_is_real else "mock",
            "price": round(d["price"], 4),
            "ts": d["ts"],
        }
        records.append(rec)
    _write_outputs(records)
    return records


def _write_outputs(records: List[dict]) -> None:
    os.makedirs(PAPER_DIR, exist_ok=True)
    # 1) 追加日志
    with open(os.path.join(PAPER_DIR, "paper_journal.jsonl"), "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 2) 最新状态（仪表盘数据源）
    with open(os.path.join(PAPER_DIR, "paper_state.json"), "w", encoding="utf-8") as f:
        json.dump({"updated": time.time(), "records": records}, f, ensure_ascii=False, indent=2)
    # 3) 人类可读仪表盘（Markdown，移动端友好）
    _write_dashboard(records)


def _write_dashboard(records: List[dict]) -> None:
    lines = ["# CryptoQuant 纸面模拟仪表盘", "",
             f"> 更新：{time.strftime('%Y-%m-%d %H:%M:%S')} · 仅模拟，绝不下单",
             "", "| 币种 | 状态 | 动作 | 置信 | regime | 否决 | 合规 |",
             "|------|------|------|------|--------|------|------|"]
    for r in records:
        lines.append(
            f"| {r['symbol']} | {r['market_state']} | {r['action']} | "
            f"{r['confidence']:.2f} | {r['regime']} | "
            f"{';'.join(r['vetoes']) or '—'} | "
            f"{'✅' if r['constitution_compliant'] else '❌'} |")
    lines += ["", "## 最近依据", ""]
    for r in records:
        if r["action"] != "HOLD":
            lines.append(f"- **{r['symbol']} → {r['action']}** ({r['confidence']:.2f}): "
                         + "；".join(r["rationale"][:2]))
    with open(os.path.join(PAPER_DIR, "paper_dashboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    # ---- 架构级硬锁：纸面原型绝不允许 live_capital=True ----
    if LIVE_CAPITAL:
        print("🛑 LIVE_CAPITAL=True 被检测到：纸面原型禁止触碰实盘，启动即退出。")
        return 2

    ap = argparse.ArgumentParser(description="CryptoQuant 纸面模拟运行器（只模拟不下单）")
    ap.add_argument("--once", action="store_true", help="单次运行（默认）")
    ap.add_argument("--loop", action="store_true", help="持续运行（守护/cron）")
    ap.add_argument("--interval", type=int, default=300, help="循环间隔秒（默认 300）")
    ap.add_argument("--source", choices=["history", "gateio"], default="history",
                    help="数据源：history 回放（默认） / gateio 只读实时")
    args = ap.parse_args()

    source = GateioPublicDataSource() if args.source == "gateio" else HistoryDataSource()
    memory = FinMemMemory()
    council = FourRoleCouncil(memory, llm=get_llm())
    constitution = TradingConstitution(live_capital=LIVE_CAPITAL)

    print(f"🧪 纸面模拟启动 | 数据源={args.source} | LLM={'真' if isinstance(council.llm, RealLLM) else 'mock'}"
          f" | 硬锁 live_capital={LIVE_CAPITAL}（仅模拟）")
    print(f"   覆盖币种：{', '.join(source.symbols())}")

    def tick():
        recs = run_once(source, council, constitution, memory)
        print(f"   本轮 {len(recs)} 个决策 → paper/paper_dashboard.md 已更新")
        for r in recs:
            if r["action"] != "HOLD":
                print(f"     · {r['symbol']} {r['action']} conf={r['confidence']:.2f} "
                      f"合规={'✅' if r['constitution_compliant'] else '❌'}")

    if args.loop:
        try:
            while True:
                tick()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 已停止（纸面模拟，无未平仓/无下单）")
    else:
        tick()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
