"""P1-17 回归：ReflectionLog 桥接真实分类（非假 '健康'）。"""
import os
import tempfile

import pytest

from cryptoquant_auto.meta import ReflectionLog
from cryptoquant_auto.meta.reflection import ReflectionLog as RealReflectionLog


def test_meta_reflectionlog_is_real_class():
    # 公开 API（cryptoquant_auto.meta.ReflectionLog）必须是真实分类实现
    assert ReflectionLog is not None
    tmp = tempfile.mkdtemp()
    rl = ReflectionLog(path=os.path.join(tmp, "reflection_log.json"))
    assert isinstance(rl._real, RealReflectionLog)


def test_record_overfit_returns_overfit_label():
    tmp = tempfile.mkdtemp()
    rl = ReflectionLog(path=os.path.join(tmp, "reflection_log.json"))
    lbl = rl.record(is_r2=0.95, oos_r2=0.01, dsr=0.1, pbo=0.05,
                    oos_mean=-10.0, oos_profit_rate=0.2, note="t")
    assert lbl == "OVERFIT"
    assert rl.label_latest() == "OVERFIT"


def test_record_risky_returns_risky_label():
    tmp = tempfile.mkdtemp()
    rl = ReflectionLog(path=os.path.join(tmp, "reflection_log.json"))
    lbl = rl.record(is_r2=0.5, oos_r2=0.4, dsr=0.2, pbo=0.5,
                    oos_mean=5.0, oos_profit_rate=0.3, note="t")
    assert lbl == "RISKY"


def test_never_fakes_health_label():
    tmp = tempfile.mkdtemp()
    rl = ReflectionLog(path=os.path.join(tmp, "reflection_log.json"))
    # 注入明显过拟合 → 必须暴露 OVERFIT，绝不返回 '健康'
    lbl = rl.record(is_r2=0.99, oos_r2=0.001, dsr=0.0, pbo=0.9,
                    oos_mean=-50.0, oos_profit_rate=0.1, note="t")
    assert lbl != "健康"
