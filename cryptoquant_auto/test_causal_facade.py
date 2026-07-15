"""P2-1 · 因果发现统一门面回归测试。

锁定：
  1) 门面默认 method=granger，且确实派发到 causal_discovery.get_causal_features。
  2) method=pcmci 可达 signals.causal.CausalDiscovery 并产出白名单。
  3) 未知 method 抛 ValueError（Fail-closed，不静默回落）。
"""
import numpy as np
import pytest

from cryptoquant_auto import causal
from cryptoquant_auto.stage2_features import FEATURE_NAMES


def test_default_method_is_granger(monkeypatch):
    """门面默认走生产 Granger 后端（与 paper_runner 既有调用一致）。"""
    sentinel = ["__GRANGER_SENTINEL__"]
    monkeypatch.setattr(causal, "_granger_features",
                        lambda **kw: list(sentinel))
    out = causal.get_causal_features()  # 不传 method → 默认 granger
    assert out == sentinel


def test_pcmci_backend_returns_whitelist():
    """method=pcmci 可达 CausalDiscovery 并产出特征名子集。"""
    p = len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(400, p))
    # 给 y 一个与第 0 列相关的结构，确保白名单非空（至少选中该特征）
    y = 0.5 * X[:, 0] + rng.normal(0.0, 1.0, size=400)
    wl = causal.get_causal_features(method="pcmci", X=X, y=y)
    assert isinstance(wl, list)
    assert all(isinstance(n, str) for n in wl)
    assert set(wl).issubset(set(FEATURE_NAMES))
    assert len(wl) >= 1


def test_unknown_method_raises(monkeypatch):
    """未知 method 必须硬拒（Fail-closed，杜绝静默回落到某个后端）。"""
    monkeypatch.setattr(causal, "_granger_features",
                        lambda **kw: ["__SHOULD_NOT_BE_CALLED__"])
    with pytest.raises(ValueError):
        causal.get_causal_features(method="bogus")
