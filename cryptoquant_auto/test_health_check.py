"""health_check 冒烟测试（P2-5 覆盖率门禁配套）。

health_check 是 cron 的 fail-closed 自检（任一关键模块 import 失败即禁止自动交易）。
P2-5 覆盖率测量发现它 0% 覆盖——补冒烟测试拉满，确保这道「假绿」防线本身被测到。
"""
import importlib

import pytest

from cryptoquant_auto.core import health_check


def test_check_module_integrity_all_ok():
    """关键模块均可导入 → healthy=True，且每项状态以 OK 开头。"""
    rep = health_check.check_module_integrity()
    assert rep.healthy is True
    assert rep.modules, "应至少检查 CRITICAL_MODULES 列表"
    assert all(v.startswith("OK") for v in rep.modules.values())


def test_check_module_integrity_reports_failure(monkeypatch):
    """任一关键模块 import 失败 → healthy=False 且 failures 非空（fail-closed）。"""
    real_import = importlib.import_module

    def _boom(name, *a, **k):
        if name == "cryptoquant_auto.risk.gate":
            raise ImportError("injected failure")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _boom)
    rep = health_check.check_module_integrity()
    assert rep.healthy is False
    assert rep.failures, "应有失败项进入 failures"
    assert any("cryptoquant_auto.risk.gate" in f for f in rep.failures)


def test_main_returns_zero_when_healthy():
    """健康时 main() 返回 0（cron 据此放行本轮）。"""
    assert health_check.main() == 0
