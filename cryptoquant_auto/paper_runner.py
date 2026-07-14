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
  python3 -m cryptoquant_auto.paper_runner --source binance # 币安只读实时（服务器主用）
  python3 -m cryptoquant_auto.paper_runner --source gateio  # Gate.io 只读实时（备选）

依赖：stage2_features / meta.agents / risk.constitution / adapters.real_llm（均已就位）。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .util.logging_setup import setup_logging

logger = logging.getLogger("cryptoquant")

# ---- 硬锁：纸面原型的唯一合法值（置 True 即启动即退）----
LIVE_CAPITAL = False

from .stage2_features import (
    assemble_feature, FEATURE_NAMES, _load_history, _load_deriv,
)
from .causal_discovery import get_causal_features
from .signals.tsfm import make_tsfm
from .history import get_symbols
from .meta.agents import FourRoleCouncil, LONG, SHORT, HOLD
from .meta.memory import FinMemMemory
from .adapters.real_llm import get_llm, RealLLM
from .risk.constitution import TradingConstitution
from .risk.conformal import SequentialConformalPredictor
from .risk.regime import detect_regime
from .risk.cvar_optimizer import CvarPositionOptimizer, cvar as _cvar_of, HAS_SCIPY

ACTION_TO_INT = {"LONG": 0, "SHORT": 1, "HOLD": 2}

PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")

# 【P2 修复】审计日志有界化：paper_journal.jsonl 原为 append-only、无轮转无截断，
# 5min cron 长期运行会无限增长（~1MB/天、350MB/年）最终撑满磁盘。设阈值，超限时
# 仅保留最近 JOURNAL_KEEP 行（约几天审计窗口），与短/长期记忆容量上限同思路。
JOURNAL_MAX_BYTES = 6 * 1024 * 1024   # 6MB：超过才触发回收，避免小文件反复整读
JOURNAL_KEEP = 20000                  # 保留行数（5min 节奏≈6天审计轨迹）

# 【P1-29 修复】cron/守护并发锁：paper 输出（dashboard/journal/state）为全局文件，
# 若 cron 间隔小于单次耗时，多进程并发写入会互相覆盖/截断。用 fcntl 全局排他锁，
# 已有实例运行时后续实例直接退出，保证单写者（fail-closed 而非竞态损坏）。
_PAPER_LOCK = os.path.join(PAPER_DIR, ".paper_runner.lock")

# ---- 路线图 B：CVaR 约束仓位优化的回看/样本参数（均可用环境变量覆盖）----
CVaR_LOOKBACK = int(os.environ.get("CRYPTOQUANT_CVAR_LOOKBACK", "120"))  # 1h 收益回看（5 天）
CVaR_MIN_SAMPLES = 24        # 收益样本下限（不足→该币不参与优化，权重 0）


def _symbol_returns(symbol: str, n: int = CVaR_LOOKBACK) -> Optional[np.ndarray]:
    """取该币最近 n 根 1h 对数收益数组（与因果发现/预报同源），失败返回 None。

    供 TSFM 预报与 CVaR 优化复用，避免重复读盘与重复计算。
    """
    try:
        hist = _load_history()
        entry = hist.get(symbol)
        if not entry:
            return None
        k1h = entry.get("1h", [])
        if len(k1h) < n + 1:
            return None
        closes = np.array([c["c"] for c in k1h], dtype=float)
        rets = np.diff(np.log(closes))
        return rets[-n:]
    except Exception:
        return None


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
    def __init__(self, warmup: int = 240, limit: int = 500):
        # 币种集统一走 get_symbols()（默认 6 币，CRYPTOQUANT_SYMBOLS 可扩到 12）
        self.SYMBOLS = get_symbols()
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
                logger.warning("[gateio] %s 拉取失败: %s", s, e)
                continue
        return out


class BinancePublicDataSource(DataSource):
    """只读公开 REST（无需 API 密钥）拉实时行情。币安主用，Gate.io 备选。

    端点（币安 U 本位合约 fapi，公开只读，不签名、不下单）：
      - fapi/v1/klines                1h K 线
      - fapi/v1/fundingRate          资金费率（fr）
      - futures/data/openInterestHist 持仓量（计算 oi_pct）
    全部只读、不签名、不下单。任一币拉取失败则跳过该币，不影响整体。

    关于「API 密钥」：公开行情无需密钥；fail-closed 架构也禁止接任何可下单的
    实盘/测试网适配器（见 adapters/binance_testnet.py，仅在非 paper 路径使用）。
    若日后需更高限速的「已鉴权公开」端点，密钥须经服务器环境变量注入，勿明文进代码/聊天。
    """

    BASE = "https://fapi.binance.com/fapi/v1"
    OI_BASE = "https://fapi.binance.com/futures/data"
    FNG = "https://api.alternative.me/fng/?limit=1"
    def __init__(self, warmup: int = 240, limit: int = 500):
        # 币种集统一走 get_symbols()（默认 6 币，CRYPTOQUANT_SYMBOLS 可扩到 12）
        self.SYMBOLS = get_symbols()
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
                sym = f"{s}USDT"
                now_ms = int(time.time() * 1000)
                url = (f"{self.BASE}/klines?symbol={sym}&interval=1h"
                       f"&startTime={now_ms - 86400 * 30 * 1000}&limit={self.limit}")
                raw = self._get_json(url)
                # 币安 klines 为数组：
                # [openTime, open, high, low, close, volume, closeTime, ...]（毫秒时间戳）
                k1h = []
                for r in raw:
                    k1h.append({"t": int(r[0]) // 1000, "o": float(r[1]),
                                "h": float(r[2]), "l": float(r[3]),
                                "c": float(r[4]), "v": float(r[5])})
                if len(k1h) < self.warmup + 2:
                    continue
                i = len(k1h) - 1
                w = k1h[:i + 1]
                closes = [c["c"] for c in w]
                # 资金费率
                fr = 0.0
                try:
                    fr_raw = self._get_json(f"{self.BASE}/fundingRate?symbol={sym}&limit=1")
                    if fr_raw:
                        fr = float(fr_raw[-1].get("fundingRate", 0.0) or 0.0)
                except Exception:
                    pass
                # 持仓量变化（oi_pct）：取最近两期 5m 快照
                oi_pct = 0.0
                try:
                    oi_raw = self._get_json(
                        f"{self.OI_BASE}/openInterestHist?symbol={sym}&period=5m&limit=2")
                    if (len(oi_raw) >= 2
                            and float(oi_raw[-2].get("sumOpenInterest", 0) or 0)):
                        oi_pct = (float(oi_raw[-1]["sumOpenInterest"])
                                  - float(oi_raw[-2]["sumOpenInterest"])) \
                            / float(oi_raw[-2]["sumOpenInterest"])
                except Exception:
                    pass
                fg = self._fng
                feat = assemble_feature(closes, w, fr, 0.0, oi_pct, fg, i)
                out[s] = {
                    "feat": np.array(feat, dtype=float),
                    "regime": _regime_of(closes),
                    "price": float(k1h[i]["c"]),
                    "ts": int(k1h[i]["t"]),
                }
            except Exception as e:
                logger.warning("[binance] %s 拉取失败: %s", s, e)
                continue
        return out


# ---------------------------------------------------------------------------
# 纸面决策（喂给宪法做合规校验的轻量载体）
# ---------------------------------------------------------------------------
class _PaperDecision:
    def __init__(self, action_int: int, rationale: str, confidence: float,
                 proposed_exposure: Optional[float] = None):
        self.action = action_int
        self.rationale = rationale
        self.confidence = confidence
        # 路线图 B：CVaR 优化器产出的仓位权重（None=未优化，退化为旧行为）
        self.proposed_exposure = proposed_exposure


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    pass


def _build_forecaster():
    """按环境构建实时预报器（路线图 A.TSFM）。

    默认 distilled（纯 numpy 岭回归，零依赖，立即可用）；
    设 CRYPTOQUANT_TSFM_ONNX 指向权重文件 → 启用 ONNX Runtime 后端（Time-MoE small 等）。
    任何失败 → 返回 None（议会退化为无预报，决策行为不变）。
    """
    try:
        backend = os.environ.get("CRYPTOQUANT_TSFM_BACKEND", "distilled")
        kw = {}
        if backend == "onnx" and os.environ.get("CRYPTOQUANT_TSFM_ONNX"):
            kw["model_path"] = os.environ["CRYPTOQUANT_TSFM_ONNX"]
        return make_tsfm(backend, **kw)
    except Exception as e:
        logger.warning("预报器构建失败（退回无预报）：%s", e)
        return None


def _forecast_point(forecaster, symbol: str, lookback: int = 24):
    """取该币最近 lookback 根 1h 对数收益，拟合并单步预报，返回点预测（log return）或 None。

    注：骨架阶段用 history_cache.json 的真实 1h 收益作预报输入（与因果发现同源）；
    实时精度可后续改为喂入 live klines。fit+forecast 成对调用，跨币不串扰。
    """
    if forecaster is None:
        return None
    try:
        rets = _symbol_returns(symbol, lookback + 1)
        if rets is None or len(rets) < lookback + 1:
            return None
        recent = rets                                  # 长度 = lookback + 1
        fitted = forecaster.fit(recent)                # DistilledTSFM 拟合并存系数；ONNX/pretrained 为 no-op
        pt, _, _ = fitted.forecast(recent[-lookback:], horizon=1)
        return float(pt[0])
    except Exception as e:
        logger.warning("预报 %s 失败（退回无预报）：%s", symbol, e)
        return None


def _make_cvar_optimizer() -> Optional[CvarPositionOptimizer]:
    """按环境构建 CVaR 约束仓位优化器（路线图 B）。

    参数均可用环境变量覆盖；任何失败 → 返回 None（议会退化为无仓位权重，行为不变）。
    """
    try:
        alpha = float(os.environ.get("CRYPTOQUANT_CVAR_ALPHA", "0.05"))
        budget = float(os.environ.get("CRYPTOQUANT_CVAR_BUDGET", "-0.02"))
        cap = float(os.environ.get("CRYPTOQUANT_CVAR_CAP", "0.12"))
        max_pos = float(os.environ.get("CRYPTOQUANT_CVAR_MAXPOS", "0.05"))
        return CvarPositionOptimizer(alpha=alpha, cvar_budget=budget,
                                     total_cap=cap, max_pos=max_pos)
    except Exception as e:
        logger.warning("CVaR 优化器构建失败（退化为无仓位权重）：%s", e)
        return None


def _optimize_positions(opt, mu: Dict[str, float], rets: Dict[str, Optional[np.ndarray]],
                        verdicts: Dict[str, tuple]) -> Dict[str, float]:
    """跨资产评估 CVaR 约束仓位权重（路线图 B）。

    mu: {sym: direction_int×confidence}（带符号的方向视图）；
    rets: {sym: 1h 对数收益数组或 None}；
    verdicts: {sym: (verdict, fc, d)}。
    返回 {sym: weight≥0}，未优化的币置 0.0。
    """
    if opt is None:
        return {s: 0.0 for s in mu}
    # 仅纳入「有方向 + 收益样本充足」的币（其余权重自然为 0，即 HOLD）
    syms = [s for s in mu
            if abs(mu[s]) > 1e-9 and rets.get(s) is not None
            and len(rets[s]) >= CVaR_MIN_SAMPLES]
    if not syms:
        return {s: 0.0 for s in mu}
    minlen = min(len(rets[s]) for s in syms)
    di = {s: verdicts[s][0].direction_int() for s in syms}
    try:
        # 方向化收益矩阵：LONG 取正、SHORT 取负（方向并入矩阵，权重非负）
        R = np.column_stack([di[s] * rets[s][-minlen:] for s in syms])  # (minlen, n)
    except Exception:
        return {s: 0.0 for s in mu}
    conviction = np.array([abs(mu[s]) for s in syms], float)
    wdict = opt.solve(conviction, R, syms)
    # 补齐未参与优化的币 → 0.0（保持与 mu 同键集）
    return {s: wdict.get(s, 0.0) for s in mu}


def run_once(source: DataSource, council: FourRoleCouncil,
             constitution: TradingConstitution, memory: FinMemMemory,
             conformal=None, forecaster=None, cvar_optimizer=None) -> List[dict]:
    snap = source.snapshot()
    llm_is_real = isinstance(council.llm, RealLLM)
    # 【P1-10 修复】conformal 由 main() 实例化并跨 tick 持有；decide() 内部完成
    # 惊喜度计算与风控门软降级（详见 agents.FourRoleCouncil.decide）。
    # ---- Pass 1：议会裁决 + 收集方向视图/收益（供跨资产 CVaR 优化）----
    verdicts: Dict[str, tuple] = {}
    mu: Dict[str, float] = {}
    rets: Dict[str, Optional[np.ndarray]] = {}
    for sym, d in snap.items():
        # 因果发现：仅让通过 Granger 稳定性筛选的特征对议会生效；
        # 未选特征经 _fget 中性化（贡献归零），降低噪声、聚焦因果信号。
        active_feats = get_causal_features(sym)
        logger.info("   · %s 因果生效特征(%d/%d): %s",
                    sym, len(active_feats), len(FEATURE_NAMES),
                    ",".join(active_feats))
        # TSFM 预报（路线图 A.TSFM）：取该币近期 1h 对数收益做单步点预测，
        # 作为第 10 路方向信号并入议会；任何失败 → None（议会退化为无预报，行为不变）。
        fc = _forecast_point(forecaster, sym)
        verdict = council.decide(
            sym, d["feat"], d["regime"], record=True, conformal=conformal,
            feature_names=active_feats, forecast=fc)
        verdicts[sym] = (verdict, fc, d)
        # 方向×置信 → 卷积代理（带符号）；CVaR 优化器取绝对值作奖励，方向并入收益矩阵
        mu[sym] = verdict.direction_int() * verdict.confidence
        rets[sym] = _symbol_returns(sym, CVaR_LOOKBACK)
    # ---- 跨资产 CVaR 仓位优化（路线图 B：替换线性仓位公式）----
    weights = _optimize_positions(cvar_optimizer, mu, rets, verdicts)
    # ---- Pass 2：宪法校验 + 产出（携带 proposed_exposure / cvar_pct）----
    records: List[dict] = []
    for sym, (verdict, fc, d) in verdicts.items():
        w = weights.get(sym, 0.0)
        # 每币「方向化」尾部 CVaR（仪表盘展示：负数=最差尾均值）
        signed_rets = rets.get(sym)
        di = verdict.direction_int()
        cvar_pct = (_cvar_of(di * np.asarray(signed_rets, float), 0.05)
                    if signed_rets is not None and len(signed_rets) >= CVaR_MIN_SAMPLES
                    else 0.0)
        action_int = ACTION_TO_INT.get(verdict.action, 2)
        const = constitution.check(
            _PaperDecision(action_int, " ".join(verdict.rationale),
                           verdict.confidence, proposed_exposure=w))
        final_action = verdict.action
        # 【P1-11 修复】宪法否决（含 R0 硬锁）→ 以 safe_action 覆盖，不得照常输出违规动作
        if not const.compliant:
            final_action = "HOLD"   # 宪法 safe_action 恒为"观望"
        rec = {
            "symbol": sym,
            "regime": d["regime"],
            "market_state": verdict.market_state,
            "action": final_action,
            "confidence": round(verdict.confidence, 4),
            "proposed_exposure": round(w, 4),
            "cvar_pct": round(float(cvar_pct), 4),
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
    # 1) 追加日志（审计轨迹）
    journal_path = os.path.join(PAPER_DIR, "paper_journal.jsonl")
    with open(journal_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 【P2 修复】有界化回收：仅当文件超过阈值才整读并截留最近 JOURNAL_KEEP 行，
    # 避免小文件每 tick 重读；稳态文件体积被钳在 ~6MB，永不上限增长。
    try:
        if os.path.getsize(journal_path) > JOURNAL_MAX_BYTES:
            with open(journal_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > JOURNAL_KEEP:
                with open(journal_path, "w", encoding="utf-8") as f:
                    f.writelines(lines[-JOURNAL_KEEP:])
    except OSError:
        pass
    # 2) 最新状态（仪表盘数据源）
    with open(os.path.join(PAPER_DIR, "paper_state.json"), "w", encoding="utf-8") as f:
        json.dump({"updated": time.time(), "records": records}, f, ensure_ascii=False, indent=2)
    # 3) 人类可读仪表盘（Markdown，移动端友好）
    _write_dashboard(records)


def _write_dashboard(records: List[dict]) -> None:
    lines = ["# CryptoQuant 纸面模拟仪表盘", "",
             f"> 更新：{time.strftime('%Y-%m-%d %H:%M:%S')} · 仅模拟，绝不下单",
             "", "| 币种 | 状态 | 动作 | 置信 | 仓位% | 尾部CVaR | regime | 否决 | 合规 |",
             "|------|------|------|------|-------|----------|--------|------|------|"]
    for r in records:
        exp = r.get("proposed_exposure", 0.0)
        cv = r.get("cvar_pct", 0.0)
        lines.append(
            f"| {r['symbol']} | {r['market_state']} | {r['action']} | "
            f"{r['confidence']:.2f} | {exp*100:.2f}% | {cv:+.2%} | {r['regime']} | "
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
    # ---- 解析参数（须先于 logging 配置，以便 --log-file 生效）----
    ap = argparse.ArgumentParser(description="CryptoQuant 纸面模拟运行器（只模拟不下单）")
    ap.add_argument("--once", action="store_true", help="单次运行（默认）")
    ap.add_argument("--loop", action="store_true", help="持续运行（守护/cron）")
    ap.add_argument("--interval", type=int, default=300, help="循环间隔秒（默认 300）")
    ap.add_argument("--source", choices=["history", "binance", "gateio"], default="history",
                    help="数据源：history 回放（默认，沙箱安全）/ binance 只读实时（主用）/ gateio 只读实时（备选）")
    ap.add_argument("--log-file", default=None,
                    help="日志落盘路径（守护/生产建议指定，启用 RotatingFileHandler 轮转）")
    args = ap.parse_args()

    # 【P2-B】结构化日志：所有诊断走 logging，避免散落 print 到 stderr；
    # 传 --log-file 即落盘（轮转），便于生产服务器持久化与接 CLS。
    # setup_logging 幂等，重复调用不叠加。
    setup_logging(log_file=args.log_file)

    # ---- 架构级硬锁：纸面原型绝不允许 live_capital=True ----
    if LIVE_CAPITAL:
        logger.error("🛑 LIVE_CAPITAL=True 被检测到：纸面原型禁止触碰实盘，启动即退出。")
        return 2

    # ---- 【P1-29】cron 并发排他锁 ----
    os.makedirs(PAPER_DIR, exist_ok=True)
    lock_fd = open(_PAPER_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.warning("⏳ 已有 paper_runner 实例运行（cron 排他锁占用），本次退出避免并发写冲突。")
        lock_fd.close()
        return 0

    # 数据源选择：binance 主用（U 本位合约只读）/ gateio 备选 / history 沙箱回放
    if args.source == "binance":
        source = BinancePublicDataSource()
    elif args.source == "gateio":
        source = GateioPublicDataSource()
    else:
        source = HistoryDataSource()
    memory = FinMemMemory()
    council = FourRoleCouncil(memory, llm=get_llm())
    constitution = TradingConstitution(live_capital=LIVE_CAPITAL)
    # 【P1-10】SPCI 在线预测器：跨 tick 持有，喂入历史置信分布，产出惊喜度。
    conformal = SequentialConformalPredictor(alpha=0.10)
    # 路线图 A.TSFM：实时预报器（默认 distilled/numpy，设 CRYPTOQUANT_TSFM_ONNX 启用 ONNX）
    forecaster = _build_forecaster()
    # 路线图 B：CVaR 约束仓位优化器（scipy.optimize；缺失/失败→None 退化为无权重）
    cvar_opt = _make_cvar_optimizer()

    logger.info("🧪 纸面模拟启动 | 数据源=%s | LLM=%s | 硬锁 live_capital=%s（仅模拟）",
                args.source, "真" if isinstance(council.llm, RealLLM) else "mock", LIVE_CAPITAL)
    logger.info("   覆盖币种：%s", ", ".join(source.symbols()))
    logger.info("   预报后端：%s", forecaster.name if forecaster else "无（未引入 TSFM 信号）")
    logger.info("   CVaR 优化器：%s（scipy=%s, budget=%.2f%%, cap=%.1f%%）",
                "启用" if cvar_opt else "未启用",
                "有" if HAS_SCIPY else "无",
                (cvar_opt.cvar_budget * 100 if cvar_opt else -2.0),
                (cvar_opt.total_cap * 100 if cvar_opt else 12.0))

    def tick():
        recs = run_once(source, council, constitution, memory,
                        conformal=conformal, forecaster=forecaster,
                        cvar_optimizer=cvar_opt)
        logger.info("   本轮 %d 个决策 → paper/paper_dashboard.md 已更新", len(recs))
        for r in recs:
            if r["action"] != "HOLD":
                logger.info("     · %s %s conf=%.2f 仓位=%.2f%% 合规=%s",
                            r["symbol"], r["action"], r["confidence"],
                            r.get("proposed_exposure", 0.0) * 100,
                            "✅" if r["constitution_compliant"] else "❌")

    if args.loop:
        try:
            while True:
                tick()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("\n🛑 已停止（纸面模拟，无未平仓/无下单）")
    else:
        tick()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
