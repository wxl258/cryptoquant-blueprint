"""风险画像与相关性过滤（吸收生产系统 RISK_PROFILE + apply_correlation_filter 优点 P0-E）。

- RISK_PROFILE 三档已在 position_sizing 定义；此处提供选取与校验。
- apply_correlation_filter：同向仓位上限 + 高相关币种去重（避免同涨同跌集中）。
"""
from __future__ import annotations

from typing import Dict, List

from .position_sizing import RISK_PROFILES, ACTIVE_PROFILE

# 币对相关性分组（粗粒度，调试版用；生产可由历史收益计算）
CORR_GROUPS = {
    "BTC": ["BTC"],
    "ETH": ["ETH"],
    "ALT_L1": ["BNB", "SOL"],
    "ALT_L2": ["XRP"],
}


def get_profile(name: str = ACTIVE_PROFILE) -> Dict:
    return RISK_PROFILES.get(name, RISK_PROFILES[ACTIVE_PROFILE])


def apply_correlation_filter(active: List[dict], profile: str = ACTIVE_PROFILE) -> List[dict]:
    """相关性/同向过滤（吸收生产 apply_correlation_filter）。

    active: [{symbol, direction, position_pct, score}]
    规则：
      1) 同向(做多/做空)仓位数 ≤ profile['mdir']
      2) 同相关组内最多保留 1 个（取 score 最高），降低同涨同跌集中
    返回过滤后列表。
    """
    prof = get_profile(profile)
    max_dir = prof["mdir"]

    # 1) 同向计数上限
    per_dir: Dict[str, int] = {"LONG": 0, "SHORT": 0}
    kept_dir: List[dict] = []
    for s in sorted(active, key=lambda x: -x.get("score", 0)):
        d = s.get("direction", "LONG")
        if per_dir.get(d, 0) >= max_dir:
            continue
        per_dir[d] += 1
        kept_dir.append(s)

    # 2) 相关组内去重（取 score 最高）
    best_per_group: Dict[str, dict] = {}
    for s in kept_dir:
        grp = "OTHER"
        for g, members in CORR_GROUPS.items():
            if s["symbol"] in members:
                grp = g; break
        cur = best_per_group.get(grp)
        if cur is None or s.get("score", 0) > cur.get("score", 0):
            best_per_group[grp] = s
    return list(best_per_group.values())
