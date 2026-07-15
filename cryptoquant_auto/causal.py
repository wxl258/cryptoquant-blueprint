"""统一因果发现入口（P2-1 · 薄门面，非删除）。

背景：代码库里有两套因果发现，职责不同、不可互相替代：
  - Granger（causal_discovery.get_causal_features）：生产路径使用（paper_runner 直接调），
    产出「信号级」因果特征白名单，喂给信号生成。
  - PCMCI（signals.causal.CausalDiscovery）：研究路径使用（run_validation_stage2），
    做 PCMCI 类 PC + 稳定选择 + 不变因果结构，产出稳健白名单。

roundtable P2-1 要求「统一因果发现入口」，但两者不是冗余竞品——统一方式是
**薄门面 + method 派发**，而非删掉其中一个。本模块提供单一调用点
`get_causal_features(method=...)` 同时透出两个后端，默认走生产 Granger。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .causal_discovery import get_causal_features as _granger_features
from .signals.causal import CausalDiscovery

__all__ = ["get_causal_features", "CausalDiscovery", "DEFAULT_METHOD"]

# 生产默认后端 = Granger（与 paper_runner 既有调用一致）
DEFAULT_METHOD = "granger"


def get_causal_features(method: str = DEFAULT_METHOD, symbol: str = "BTC",
                        X: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None,
                        **kwargs) -> List[str]:
    """统一因果特征入口。

    method="granger"（默认）：生产信号级特征，调 causal_discovery.get_causal_features，
        透传 symbol / force 等参数。
    method="pcmci"：研究稳健白名单，需要 X(特征矩阵) 与 y(目标收益)，
        返回 CausalDiscovery.fit(...).whitelist。

    返回：特征名列表（List[str]）。
    """
    method = str(method).lower()
    if method == "granger":
        return _granger_features(symbol=symbol, **kwargs)
    if method == "pcmci":
        if X is None or y is None:
            raise ValueError("PCMCI 后端需要 X(特征矩阵) 与 y(目标收益) 两个参数")
        report = CausalDiscovery().fit(np.asarray(X, float), np.asarray(y, float))
        return report.whitelist
    raise ValueError(f"未知因果发现方法: {method!r}（可选 {DEFAULT_METHOD!r}/'pcmci'）")
