"""模块完整性自检（移植服务器 health_check 优点）。

解决"进程在跑=健康"假绿：启动 daemon 前逐个 import 关键模块、校验关键组件，
任一 FAIL 即视为系统不健康，禁止自动启动交易（Fail-closed）。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, List


# 关键模块（必须以 package 相对导入成功，否则视为假绿）
CRITICAL_MODULES = [
    "cryptoquant_auto.models",
    "cryptoquant_auto.adapters.mock",
    "cryptoquant_auto.adapters.binance_testnet",
    "cryptoquant_auto.risk.gate",
    "cryptoquant_auto.risk.kill_switch",
    "cryptoquant_auto.risk.circuit_breaker",
    "cryptoquant_auto.risk.signal_filter",
    "cryptoquant_auto.risk.liquidation_guard",
    "cryptoquant_auto.core.engine",
    "cryptoquant_auto.core.reconcile",
    "cryptoquant_auto.sim.backtest",
]


@dataclass
class HealthReport:
    modules: Dict[str, str] = field(default_factory=dict)
    healthy: bool = True

    @property
    def failures(self) -> List[str]:
        return [f"{k}: {v}" for k, v in self.modules.items() if not v.startswith("OK")]


def check_module_integrity(modules: List[str] = None) -> HealthReport:
    mods = modules or CRITICAL_MODULES
    rep = HealthReport()
    for name in mods:
        try:
            importlib.import_module(name)
            rep.modules[name] = "OK"
        except Exception as e:  # noqa
            rep.modules[name] = f"FAIL:{type(e).__name__}:{e}"
            rep.healthy = False
    return rep
