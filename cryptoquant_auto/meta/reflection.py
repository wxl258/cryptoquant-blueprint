"""反思日志 + 边缘标记系统（圆桌决议 Feature 2）。

每次 WFA 训练完成后自动记录评估指标并打标记：
  - OVERFIT: IS 显著优于 OOS
  - RISKY:   OOS 盈利窗不足或 PBO 偏高
  - DEAD:    连续多次 OOS 净负

用法：
    from .reflection import ReflectionLog
    log = ReflectionLog()
    log.record(timestamp=..., fold_weights=[...], is_r2=..., oos_r2=...,
               dsr=..., pbo=..., oos_mean=..., oos_profit_rate=...)
    label = log.label_latest()
    print(log.summary(5))
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 持久化路径（与 cognition.py 的 env_history.json 同目录）
_BASE_DIR = os.environ.get("CRYPTOQUANT_BASE_DIR",
                           os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFLECTION_FILE = os.path.join(_BASE_DIR, "data", "reflection_log.json")


# 标记阈值（可配置）
OVERFIT_DECAY_RATIO = 3.0       # IS/OOS R² 衰减比 > 此值 → OVERFIT
RISKY_PBO_THRESHOLD = 0.30      # PBO > 此值 → RISKY
RISKY_PROFIT_THRESHOLD = 0.60   # OOS 盈利窗 < 此值 → RISKY
DEAD_CONSECUTIVE_LOSSES = 3     # 连续 3 次 OOS 均值 < 0 → DEAD


@dataclass
class ReflectionRecord:
    """单次 WFA 训练记录。"""
    timestamp: float
    label: str = "无"          # OVERFIT / RISKY / DEAD / 健康
    fold_weights: List[float] = field(default_factory=list)
    is_r2: float = 0.0
    oos_r2: float = 0.0
    dsr: float = 0.0
    pbo: float = 0.0
    oos_mean: float = 0.0      # OOS net bps 均值
    oos_profit_rate: float = 0.0  # OOS 盈利窗比例
    note: str = ""


class ReflectionLog:
    """追加式反思日志（append-only）。"""

    def __init__(self, path: str = REFLECTION_FILE):
        self.path = path
        self.records: List[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    self.records = json.load(f)
        except Exception:
            self.records = []

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.records[-100:], f, indent=2)  # 保留最近 100 条
        except Exception:
            pass

    def record(self, *, timestamp: float = None,
               fold_weights: list = None,
               is_r2: float = 0.0, oos_r2: float = 0.0,
               dsr: float = 0.0, pbo: float = 0.0,
               oos_mean: float = 0.0, oos_profit_rate: float = 0.0,
               note: str = "") -> str:
        """记录一次训练结果，返回自动判定的标记。"""
        rec = {
            "timestamp": timestamp or time.time(),
            "fold_weights": fold_weights or [],
            "is_r2": round(is_r2, 4),
            "oos_r2": round(oos_r2, 4),
            "dsr": round(dsr, 4),
            "pbo": round(pbo, 4),
            "oos_mean": round(oos_mean, 4),
            "oos_profit_rate": round(oos_profit_rate, 4),
            "note": note,
        }
        # 自动打标记
        label = self._classify(rec)
        rec["label"] = label
        self.records.append(rec)
        self._save()
        return label

    def _classify(self, rec: dict) -> str:
        """根据阈值判定标记。"""
        # OVERFIT: IS/OOS R² 衰减
        oos_r2 = abs(rec.get("oos_r2", 0)) + 0.001
        is_r2 = abs(rec.get("is_r2", 0)) + 0.001
        if is_r2 / oos_r2 > OVERFIT_DECAY_RATIO:
            return "OVERFIT"
        # RISKY: PBO 或盈利窗不足
        if rec.get("pbo", 0) > RISKY_PBO_THRESHOLD:
            return "RISKY"
        if rec.get("oos_profit_rate", 1) < RISKY_PROFIT_THRESHOLD and \
           rec.get("oos_profit_rate", 1) > 0:
            return "RISKY"
        # DEAD: 连续多次 OOS 净负
        if len(self.records) >= DEAD_CONSECUTIVE_LOSSES:
            recent = self.records[-DEAD_CONSECUTIVE_LOSSES:]
            if all(r.get("oos_mean", 0) < 0 for r in recent):
                return "DEAD"
        return "健康"

    def label_latest(self) -> str:
        """返回最新记录的标记。"""
        if not self.records:
            return "无记录"
        return self.records[-1].get("label", "无")

    def summary(self, n: int = 5) -> str:
        """最近 N 次记录摘要。"""
        recent = self.records[-n:] if len(self.records) >= n else self.records
        if not recent:
            return "无反思记录"
        lines = [f"{'时间':<16} {'标记':<10} {'OOS均值':>8} {'IS_R2':>8} {'OOS_R2':>8} {'PBO':>6} {'DSR':>6}"]
        lines.append("-" * 70)
        for r in recent:
            ts = time.strftime("%m-%d %H:%M", time.localtime(r.get("timestamp", 0)))
            label = r.get("label", "?")
            om = f"{r.get('oos_mean', 0):+.2f}"
            ir = f"{r.get('is_r2', 0):.3f}"
            o_r = f"{r.get('oos_r2', 0):.3f}"
            pb = f"{r.get('pbo', 0):.2f}"
            ds = f"{r.get('dsr', 0):.2f}"
            lines.append(f"{ts:<16} {label:<10} {om:>8} {ir:>8} {o_r:>8} {pb:>6} {ds:>6}")
        return "\n".join(lines)

    def trend(self, metric: str = "oos_mean", n: int = 5) -> str:
        """最近 N 次训练的趋势（上升/下降/波动）。"""
        vals = [r.get(metric, 0) for r in self.records[-n:]]
        if len(vals) < 2:
            return "样本不足"
        if vals[-1] > vals[0] * 1.1:
            return f"📈 {metric} 上升中"
        if vals[-1] < vals[0] * 0.9:
            return f"📉 {metric} 下降中"
        return f"➡️ {metric} 平稳"
