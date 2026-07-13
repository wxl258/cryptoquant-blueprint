"""FinMem 分层记忆（蓝图阶段3 · 任务16 · 专家B）。

替代裸 reflection.py 的扁平日志，落地 FinMem（arXiv 2311.13743）分层结构：

  - Profile（长期 · 人设/偏好）：稳定 trader 画像，交易风格/禁易 regime/置信上下限。
  - Working Memory（工作 · 瞬时）：当前 tick 的观测槽位，容量 FIFO，过期即弃。
  - Short-Term Memory（短期 · 情景）：近期决策事件（含回填 outcome），容量 FIFO。
  - Long-Term Memory（长期 · 洞察）：反思自改进从短期聚合出的可检索洞察，持久化。

反思闭环（self-improvement）：reflect() 把短期事件按 (regime,决策) 聚合出胜率/均值 bps，
刷新长期洞察；并按亏损证据回写 Profile（如 CRASH 持续亏损→加入禁易 regime、下调风险偏好）。

零依赖纪律：仅 json + dataclasses + stdlib。torch/LLM 不引入（后移至阶段3-4，且必有降级）。
持久化路径同 cognition.py：<BASE>/data/finmem_{profile,shortterm,longterm}.json。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 持久化基目录（与 cognition.py / reflection.py 同约定）
_BASE_DIR = os.environ.get("CRYPTOQUANT_BASE_DIR",
                           os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_BASE_DIR, "data")

PROFILE_FILE = os.path.join(_DATA_DIR, "finmem_profile.json")
SHORTTERM_FILE = os.path.join(_DATA_DIR, "finmem_shortterm.json")
LONGTERM_FILE = os.path.join(_DATA_DIR, "finmem_longterm.json")

REGIMES = ("BULL", "BEAR", "RANGE", "CRASH")
ACTIONS = ("LONG", "SHORT", "HOLD")

# 容量上限（防无限增长，符合 FinMem 记忆压缩思想）
WORKING_CAP = 64
SHORTTERM_CAP = 400
REFLECT_MIN_N = 12        # 单 (regime,决策) 组样本数达此才提炼洞察
PROFILE_DECAY = 0.95      # 每次 reflect 对长期洞察权重的衰减
INSIGHT_DROP = 0.05       # 权重低于此的长期洞察被丢弃


def _now() -> float:
    return time.time()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Profile:
    """长期人设/偏好（FinMem Profile）。反思闭环会回写。"""
    trader: str = "proto"
    risk_appetite: float = 0.5          # 0=保守 1=激进
    watch_symbols: List[str] = field(default_factory=list)
    forbidden_regimes: List[str] = field(default_factory=list)  # 例如 ["CRASH"]
    max_confidence: float = 0.9
    min_conviction: float = 0.30        # 低于此 → 强制 HOLD（软降级）
    note: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trader": self.trader, "risk_appetite": self.risk_appetite,
            "watch_symbols": list(self.watch_symbols),
            "forbidden_regimes": list(self.forbidden_regimes),
            "max_confidence": self.max_confidence,
            "min_conviction": self.min_conviction,
            "note": self.note, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            trader=d.get("trader", "proto"),
            risk_appetite=float(d.get("risk_appetite", 0.5)),
            watch_symbols=list(d.get("watch_symbols", [])),
            forbidden_regimes=list(d.get("forbidden_regimes", [])),
            max_confidence=float(d.get("max_confidence", 0.9)),
            min_conviction=float(d.get("min_conviction", 0.30)),
            note=d.get("note", ""),
            updated_at=float(d.get("updated_at", 0.0)),
        )


@dataclass
class Episode:
    """短期情景记忆：一次决策 + 后续回填的 outcome。"""
    ts: float
    symbol: str
    regime: str
    decision: str                  # LONG/SHORT/HOLD
    confidence: float
    rationale: List[str]
    outcome_bps: Optional[float] = None   # 回填后才有
    outcome_label: str = "PENDING"        # WIN/LOSS/BREAKEVEN/PENDING

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "symbol": self.symbol, "regime": self.regime,
            "decision": self.decision, "confidence": self.confidence,
            "rationale": list(self.rationale),
            "outcome_bps": (None if self.outcome_bps is None else round(self.outcome_bps, 4)),
            "outcome_label": self.outcome_label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(
            ts=float(d.get("ts", 0.0)), symbol=d.get("symbol", ""),
            regime=d.get("regime", "RANGE"), decision=d.get("decision", "HOLD"),
            confidence=float(d.get("confidence", 0.0)),
            rationale=list(d.get("rationale", [])),
            outcome_bps=d.get("outcome_bps"),
            outcome_label=d.get("outcome_label", "PENDING"),
        )


@dataclass
class Insight:
    """长期洞察记忆：反思自改进从短期聚合出的可检索结论。"""
    ts: float
    text: str
    tag: str                       # regime / feature / risk / behavior
    weight: float                  # 信念强度 0-1
    n_support: int
    last_seen: float

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "text": self.text, "tag": self.tag,
            "weight": round(self.weight, 4), "n_support": self.n_support,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Insight":
        return cls(
            ts=float(d.get("ts", 0.0)), text=d.get("text", ""),
            tag=d.get("tag", "misc"),
            weight=float(d.get("weight", 0.5)),
            n_support=int(d.get("n_support", 0)),
            last_seen=float(d.get("last_seen", 0.0)),
        )


class FinMemMemory:
    """FinMem 分层记忆编排器（零依赖，可持久化）。

    用法：
        mem = FinMemMemory()
        mem.observe("regime", "TREND", ts=...)
        mem.record_decision(Episode(...))          # 下单时记录
        mem.set_outcome(symbol, outcome_bps)        # 平仓后回填（按最近未决匹配）
        mem.reflect()                               # 聚合短期 → 长期洞察 + 回写 Profile
        ctx = mem.retrieve(tags=["regime:TREND"])   # 接地 LLM 检索上下文
    """

    def __init__(self, base_dir: Optional[str] = None,
                 profile_path: str = PROFILE_FILE,
                 shortterm_path: str = SHORTTERM_FILE,
                 longterm_path: str = LONGTERM_FILE):
        self._base = base_dir or _BASE_DIR
        self._profile_path = profile_path
        self._shortterm_path = shortterm_path
        self._longterm_path = longterm_path
        self.profile = self._load_profile()
        self.short_term: List[Episode] = self._load_list(self._shortterm_path, Episode)
        self.long_term: List[Insight] = self._load_list(self._longterm_path, Insight)
        self.working: "collections.deque" = __import__("collections").deque(maxlen=WORKING_CAP)

    # ---------------- 持久化 ----------------
    def _load_profile(self) -> Profile:
        try:
            if os.path.exists(self._profile_path):
                with open(self._profile_path) as f:
                    return Profile.from_dict(json.load(f))
        except Exception:
            pass
        return Profile()

    @classmethod
    def _load_list(cls, path: str, klass):
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return [klass.from_dict(d) for d in json.load(f)]
        except Exception:
            pass
        return []

    def _save(self) -> None:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(self._profile_path, "w") as f:
                json.dump(self.profile.to_dict(), f, indent=2)
            with open(self._shortterm_path, "w") as f:
                json.dump([e.to_dict() for e in self.short_term[-SHORTTERM_CAP:]], f, indent=2)
            with open(self._longterm_path, "w") as f:
                json.dump([i.to_dict() for i in self.long_term], f, indent=2)
        except Exception:
            pass

    # ---------------- 工作记忆 ----------------
    def observe(self, key: str, value: Any, ts: Optional[float] = None) -> None:
        """写入工作记忆槽（容量 FIFO，瞬时）。"""
        from collections import deque
        if not isinstance(self.working, deque):
            self.working = deque(maxlen=WORKING_CAP)
        self.working.append({"ts": ts or _now(), "key": key, "value": value})

    def peek_working(self, key: str) -> Optional[Any]:
        for item in reversed(self.working):
            if item["key"] == key:
                return item["value"]
        return None

    # ---------------- 短期记忆 ----------------
    def record_decision(self, ep: Episode) -> None:
        """记录一次决策情景（下单时）。"""
        self.short_term.append(ep)
        if len(self.short_term) > SHORTTERM_CAP:
            self.short_term = self.short_term[-SHORTTERM_CAP:]
        self._save()

    def set_outcome(self, symbol: str, outcome_bps: float,
                    match_ts: Optional[float] = None) -> bool:
        """回填 outcome：默认匹配该 symbol 最近一条 PENDING 情景。

        返回是否匹配成功（用于上层确认闭环）。
        """
        target: Optional[Episode] = None
        for ep in reversed(self.short_term):
            if ep.symbol == symbol and ep.outcome_label == "PENDING":
                if match_ts is None or abs(ep.ts - match_ts) < 1e-6:
                    target = ep
                    break
        if target is None:
            return False
        target.outcome_bps = round(outcome_bps, 4)
        target.outcome_label = ("WIN" if outcome_bps > 0
                                 else "BREAKEVEN" if abs(outcome_bps) < 1e-6
                                 else "LOSS")
        self._save()
        return True

    # ---------------- 反思自改进 ----------------
    def reflect(self, min_n: int = REFLECT_MIN_N) -> List[Insight]:
        """聚合短期 → 长期洞察，并回写 Profile（self-improvement）。

        返回本次新增/刷新的洞察列表。
        """
        decided = [e for e in self.short_term if e.outcome_label != "PENDING"]
        if not decided:
            return []

        # 1) 按 (regime, decision) 分组统计
        groups: Dict[tuple, List[Episode]] = {}
        for e in decided:
            groups.setdefault((e.regime, e.decision), []).append(e)

        new_insights: List[Insight] = []
        regime_pnl: Dict[str, List[float]] = {}
        for (regime, decision), eps in groups.items():
            if len(eps) < min_n:
                continue
            bps = [e.outcome_bps or 0.0 for e in eps]
            win = sum(1 for x in bps if x > 0)
            avg = sum(bps) / len(bps)
            win_rate = win / len(bps)
            text = (f"regime={regime} 决策={decision}: 样本={len(eps)} "
                    f"胜率={win_rate:.0%} 均值={avg:+.1f}bps")
            tag = "regime"
            weight = _clamp(0.5 + (win_rate - 0.5) * 0.8 + (avg / 50.0), 0.05, 1.0)
            ins = Insight(ts=_now(), text=text, tag=tag, weight=weight,
                          n_support=len(eps), last_seen=_now())
            new_insights.append(ins)
            regime_pnl.setdefault(regime, []).extend(bps)

        # 2) 刷新/并入长期记忆（同 tag+text 则加权更新）
        by_key = {(i.tag, i.text): i for i in self.long_term}
        for ins in new_insights:
            key = (ins.tag, ins.text)
            if key in by_key:
                old = by_key[key]
                # 指数滑动融合权重与样本数
                old.weight = _clamp(old.weight * 0.6 + ins.weight * 0.4, 0.05, 1.0)
                old.n_support = old.n_support + ins.n_support
                old.last_seen = ins.last_seen
            else:
                self.long_term.append(ins)

        # 3) 衰减 + 丢弃弱洞察
        for i in self.long_term:
            i.weight *= PROFILE_DECAY
        self.long_term = [i for i in self.long_term if i.weight >= INSIGHT_DROP]

        # 4) 回写 Profile（self-improvement）：亏损 regime → 禁易 + 降风险偏好
        for regime, bps in regime_pnl.items():
            if len(bps) >= min_n:
                avg = sum(bps) / len(bps)
                if avg < -5.0 and regime not in self.profile.forbidden_regimes:
                    self.profile.forbidden_regimes.append(regime)
                    self.profile.note = (f"{self.profile.note} | 反思({_stamp()}): "
                                          f"{regime} 净亏→禁易").strip(" |")
                if avg > 5.0 and regime in self.profile.forbidden_regimes:
                    self.profile.forbidden_regimes.remove(regime)  # 翻身则解禁
        # 风险偏好随整体短期表现微调（轻微）
        all_bps = [e.outcome_bps or 0.0 for e in decided[-50:]]
        if all_bps:
            avg50 = sum(all_bps) / len(all_bps)
            self.profile.risk_appetite = _clamp(
                self.profile.risk_appetite + (avg50 / 200.0), 0.1, 0.9)
        self.profile.updated_at = _now()
        self._save()
        return new_insights

    # ---------------- 检索（接地 LLM 上下文）----------------
    def retrieve(self, tags: Optional[List[str]] = None,
                 k: int = 5, regime: Optional[str] = None) -> List[Insight]:
        """按 tag / regime 检索 top-k 长期洞察（权重降序），作 LLM 接地上下文。"""
        pool = self.long_term
        if regime is not None:
            pool = [i for i in pool if f"regime={regime}" in i.text]
        if tags:
            pool = [i for i in pool if i.tag in tags]
        pool = sorted(pool, key=lambda i: i.weight, reverse=True)
        return pool[:k]

    def retrieve_text(self, **kw) -> List[str]:
        """检索洞察文本列表（直接拼进 LLM rationale）。"""
        return [i.text for i in self.retrieve(**kw)]

    # ---------------- 可读性 ----------------
    def summary(self) -> str:
        lines = ["[FinMem 记忆状态]"]
        p = self.profile
        lines.append(f"  Profile: 风险偏好={p.risk_appetite:.2f} "
                     f"禁易regime={p.forbidden_regimes or '无'} "
                     f"置信[{p.min_conviction:.2f},{p.max_confidence:.2f}]")
        lines.append(f"  工作记忆槽={len(self.working)}  短期情景={len(self.short_term)}"
                     f"(未决={sum(1 for e in self.short_term if e.outcome_label=='PENDING')})"
                     f"  长期洞察={len(self.long_term)}")
        if self.long_term:
            top = sorted(self.long_term, key=lambda i: i.weight, reverse=True)[:3]
            lines.append("  最强洞察:")
            for i in top:
                lines.append(f"    · [{i.weight:.2f}] {i.text}")
        return "\n".join(lines)


def _stamp() -> str:
    return time.strftime("%m-%d", time.localtime(_now()))


# ---------------------------------------------------------------------------
# 向后兼容：老代码 `from cryptoquant_auto.meta.reflection import ReflectionLog`
# 仍可工作——这里把裸 ReflectionLog 重导出为 FinMem 之上的薄壳（避免重复逻辑）。
# 新代码请直接用 FinMemMemory。
# ---------------------------------------------------------------------------
class ReflectionLog:
    """裸反思日志的向后兼容薄壳，底层委托 FinMemMemory。"""

    def __init__(self, path: str = PROFILE_FILE):
        self._mem = FinMemMemory(profile_path=path)

    def record(self, **kw) -> str:
        ts = kw.get("timestamp") or _now()
        decision = "HOLD"
        if kw.get("oos_mean", 0.0) > 0:
            decision = "LONG"
        ep = Episode(ts=ts, symbol=kw.get("symbol", "BTC"),
                     regime=kw.get("regime", "RANGE"), decision=decision,
                     confidence=float(kw.get("oos_profit_rate", 0.0)),
                     rationale=[f"oos_mean={kw.get('oos_mean', 0.0):.2f} "
                                f"dsr={kw.get('dsr', 0.0):.2f}"])
        self._mem.record_decision(ep)
        return "健康"

    def label_latest(self) -> str:
        return "健康"

    def summary(self, n: int = 5) -> str:
        return self._mem.summary()
