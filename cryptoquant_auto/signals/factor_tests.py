"""因子 IC 测试 + 分组收益 + 阈值扫描（Phase 1 量价因子）。

用法：
    python -m cryptoquant_auto.signals.factor_tests
    或在本模块中调用 run_factor_tests() 获得测试报告。

依赖：numpy (已安装), scipy 非必须（用纯 Python 实现 Spearman）
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, List, Tuple

from .indicators import (
    calc_obv_adx, calc_vol_price_divergence,
    calc_vol_confirmation, calc_vwap_deviation,
)
from ..history import SYMBOLS

_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_HIST_PATH = os.path.join(_PKG_DIR, "history_cache.json")


def _spearman_rank(xs: List[float], ys: List[float]) -> float:
    """Spearman 秩相关系数（纯 Python，无需 scipy）。"""
    n = len(xs)
    if n < 3:
        return 0.0
    # 秩
    rx = [sorted(xs).index(v) for v in xs]
    ry = [sorted(ys).index(v) for v in ys]
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    d1 = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    d2 = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)


def factor_ic(candles: List[dict], factor_func, horizon: int = 12,
              step: int = 3, warmup: int = 100, **fkwargs) -> Tuple[float, float, int]:
    """计算单因子的 Spearman IC 及 IC_IR。

    返回：(ic, ic_ir, n_samples)
      ic: Spearman 秩相关系数
      ic_ir: IC 均值 / IC 标准差（信息比率）
    """
    f_vals: List[float] = []
    rets: List[float] = []
    n = len(candles)
    for i in range(warmup, n - horizon, step):
        fv = factor_func(candles[:i + 1], **fkwargs)
        if isinstance(fv, tuple):
            fv = fv[0]  # 取第一个数值作为因子值
        if fv == 0 or not isinstance(fv, (int, float)):
            continue
        ret = candles[i + horizon]["c"] / candles[i]["c"] - 1
        f_vals.append(fv)
        rets.append(ret)
    if len(f_vals) < 10:
        return 0.0, 0.0, 0
    ic = _spearman_rank(f_vals, rets)
    # IC_IR
    ics = []
    for fold in range(min(5, len(f_vals) // 20)):
        start = fold * (len(f_vals) // 5)
        end = (fold + 1) * (len(f_vals) // 5)
        if start < end:
            ics.append(_spearman_rank(f_vals[start:end], rets[start:end]))
    ic_ir = (sum(ics) / len(ics)) / (math.sqrt(sum((x - sum(ics) / len(ics)) ** 2
                                                  for x in ics) / len(ics)) + 1e-9) if ics else 0.0
    return round(ic, 4), round(ic_ir, 3), len(f_vals)


def factor_quantile(candles: List[dict], factor_func, n_groups: int = 5,
                    horizon: int = 12, step: int = 3, warmup: int = 100,
                    **fkwargs) -> List[Tuple[int, float, int]]:
    """按因子值分 n_groups 组，计算每组平均未来收益。

    返回 [(group_index, mean_return, count), ...]
    """
    f_vals: List[float] = []
    rets: List[float] = []
    n = len(candles)
    for i in range(warmup, n - horizon, step):
        fv = factor_func(candles[:i + 1], **fkwargs)
        if isinstance(fv, tuple):
            fv = fv[0]
        if fv == 0 or not isinstance(fv, (int, float)):
            continue
        ret = candles[i + horizon]["c"] / candles[i]["c"] - 1
        f_vals.append(fv)
        rets.append(ret)
    if len(f_vals) < n_groups * 2:
        return [(g, 0.0, 0) for g in range(n_groups)]
    sorted_pairs = sorted(zip(f_vals, rets), key=lambda x: x[0])
    group_size = len(sorted_pairs) // n_groups
    results = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else len(sorted_pairs)
        group_rets = [sorted_pairs[i][1] for i in range(start, end)]
        mean_r = sum(group_rets) / len(group_rets) if group_rets else 0.0
        results.append((g, round(mean_r * 1e4, 2), len(group_rets)))
    return results


# ---- 因子测试报告 ----
def run_factor_tests(hist: dict = None):
    """跑所有量价因子的 IC 测试，输出报告。"""
    if hist is None:
        with open(_HIST_PATH) as f:
            hist = json.load(f)

    factors: List[Tuple[str, Callable, Dict]] = [
        ("F1_OBV_ADX", lambda c, p=14: calc_obv_adx(c, p)[0], {}),
        ("F1_OBV_DIR", lambda c, p=14: 1 if calc_obv_adx(c, p)[1] > calc_obv_adx(c, p)[2] else -1, {}),
        ("F2_VolPriceDiv", calc_vol_price_divergence, {"n": 24}),
        ("F2_VolPriceDiv_12", lambda c: calc_vol_price_divergence(c, n=12), {}),
        ("F3_VolConf_Signal", lambda c, p=20, t=1.5: calc_vol_confirmation(c, p, t)[0], {}),
        ("F3_VolConf_Ratio", lambda c, p=20, t=1.5: calc_vol_confirmation(c, p, t)[1], {}),
        ("F4_VWAP_z", lambda c, lb=24: calc_vwap_deviation(c, lb)[0], {}),
        ("F4_VWAP_atr", lambda c, lb=24: calc_vwap_deviation(c, lb)[1], {}),
    ]

    print(f"{'因子':<20} {'IC':>8} {'IC_IR':>8} {'n':>6}  per_coin_ic")
    print("-" * 70)

    all_ic: Dict[str, float] = {}
    for name, func, fw in factors:
        per_coin = {}
        for s in SYMBOLS:
            k1h = hist.get(s, {}).get("1h", [])
            if len(k1h) < 200:
                continue
            ic_val, _, n = factor_ic(k1h, func, horizon=12, step=6, warmup=120, **fw)
            per_coin[s] = ic_val
        if not per_coin:
            continue
        mean_ic = sum(per_coin.values()) / len(per_coin)
        # 跨币方向一致性
        pos = sum(1 for v in per_coin.values() if v > 0.01)
        neg = sum(1 for v in per_coin.values() if v < -0.01)
        # 用 IC_IR（均值 IC / 标准差）
        ics_list = list(per_coin.values())
        ic_std = math.sqrt(sum((x - mean_ic) ** 2 for x in ics_list) / len(ics_list)) or 1e-9
        ic_ir = round(mean_ic / ic_std, 2)
        ic_str = f"{mean_ic * 100:.2f}%"
        coin_str = " ".join(f"{s}={v*100:.1f}%" for s, v in sorted(per_coin.items(), key=lambda x: -abs(x[1])))
        print(f"{name:<20} {ic_str:>8} {ic_ir:>8} {len(per_coin):>6}  {coin_str}")
        all_ic[name] = mean_ic

    # 总结
    print("\n=== Phase 1 量价因子 IC 测试结论 ===")
    positives = [(n, ic) for n, ic in all_ic.items() if abs(ic) > 0.005]
    if positives:
        print(f"|IC|>0.005 的因子: {len(positives)} 个")
        for n, ic in sorted(positives, key=lambda x: -abs(x[1])):
            print(f"  {n:<20} IC={ic*100:.3f}%")
    else:
        print("⚠️ 所有因子 |IC|<0.005，因子预测能力很弱。")
        print("   建议：调整参数范围或考虑技术指标类因子。")


if __name__ == "__main__":
    run_factor_tests()
