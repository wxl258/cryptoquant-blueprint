"""Walk-Forward 过拟合护栏（移植服务器 WF + DSR/PBO/CSCV 优点，满足 Gate C）。

把 PaperBacktest 当黑盒，滚动窗口：IS 段估凯利输入(p,b)与调参，OOS 段验证。
输出 DSR（Deflated Sharpe）、PBO（过拟合概率）、OOS 盈利窗比例（Gate C 要求 ≥60%）。
依赖仅 numpy/math，无 scipy。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .backtest import PaperBacktest, BacktestConfig, make_random_signals
from ..risk.kelly import KellyConfig, fractional_kelly, kelly_nominal

# 【P1-27 修复】回测每步 = 1h bar，年化周期数应为 365*24=8760，而非日线的 252。
# 旧版用 sqrt(252) 会把 WF 的 DSR 与 metrics.deflated_sharpe（默认 8760）置于不同
# 年化尺度，导致同口径指标不可比、且 DSR 被系统性低估。统一为 1h 口径。
PERIODS_PER_YEAR_1H = 8760


@dataclass
class WFReport:
    n_windows: int = 0
    oos_profit_wins: int = 0
    oos_win_rate: float = 0.0       # Gate C 要求 >= 0.60
    dsr: float = 0.0                # Gate C 要求 > 0
    pbo: float = 0.0                # Gate C 要求 <= 0.20
    oos_sharpes: List[float] = field(default_factory=list)
    gate_c_pass: bool = False
    # 阶段0.5：Purged+Embargo 实际剪掉的 bar 数（仪表化，证明隔离层真的在干活）
    n_purged_bars: int = 0
    n_embargoed_bars: int = 0

    @property
    def summary(self) -> str:
        return (f"窗口={self.n_windows} OOS盈利窗={self.oos_win_rate:.0%} "
                f"DSR={self.dsr:.2f} PBO={self.pbo:.2f} "
                f"GateC={'✅' if self.gate_c_pass else '❌'}")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _deflated_sharpe(returns: np.ndarray, sr0: float = 0.0) -> float:
    """简化 DSR（Bailey & Lopez de Prado），无 scipy。"""
    T = len(returns)
    if T < 5:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    sr = mean / std * math.sqrt(PERIODS_PER_YEAR_1H)
    # 偏度/峰度
    skew = float(np.mean(((returns - mean) / std) ** 3))
    kurt = float(np.mean(((returns - mean) / std) ** 4))
    z = 1.645  # 95% 单侧
    denom = math.sqrt(max(1e-9, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    dsr_stat = (sr * math.sqrt(T) - z * (sr0 * math.sqrt(T) if sr0 else z)) / denom
    return _norm_cdf(dsr_stat)


def _cscv_pbo(perf: List[float], n_boot: int = 200, seed: int = 1) -> float:
    """CSCV 近似：随机二分，统计 min(后半) < min(前半) 的比例。"""
    if len(perf) < 2:
        return 0.0
    rnd = random.Random(seed)
    cnt = 0
    arr = np.array(perf, dtype=float)
    for _ in range(n_boot):
        idx = list(range(len(arr)))
        rnd.shuffle(idx)
        h = len(idx) // 2
        s1 = arr[idx[:h]]
        s2 = arr[idx[h:]]
        if s1.size and s2.size and s2.min() < s1.min():
            cnt += 1
    return cnt / n_boot


def estimate_pb(trades: list) -> Tuple[float, float]:
    """从成交记录估 (胜率 p, 盈亏比 b)。"""
    if not trades:
        return 0.5, 2.0
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    p = len(wins) / len(trades)
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = -sum(losses) / len(losses) if losses else 1.0
    b = (avg_w / avg_l) if avg_l > 0 else 2.0
    return p, b


def walk_forward(signals: list, windows: int = 6, is_ratio: float = 0.6,
                 seed: int = 7, embargo: float = 0.0, purge: float = 0.0,
                 min_iso_bars: int = 1) -> WFReport:
    """滚动 Walk-Forward（IS/OOS）过拟合护栏。

    阶段0.5 新增 embargo / purge 防泄漏间隔（Lopez de Prado）：
      - purge  : 丢弃训练段末尾 `purge*IS长度` 根 K（与测试段时间相邻，
                 其滚动特征已泄漏测试信息）—— 净化训练集。
      - embargo: 丢弃测试段开头 `embargo*OOS长度` 根 K（与训练段相邻，
                 标签/特征跨窗重叠）—— 净化测试集。
    默认 embargo=purge=0 时退化为原朴素 WF（向后兼容）。

    【P1-15 修复】隔离条数按 `int(frac*窗口长)` 计算，小窗口下会整除下取整为 0，
    导致 embargo/purge>0 却「无隔离」（泄漏护栏悄悄失效）。新增 min_iso_bars：
    只要 frac>0，隔离条数至少取 min_iso_bars，确保防泄漏层真正生效。
    """
    rep = WFReport()
    n = len(signals)
    if n < windows:
        return rep
    fold = n // windows
    rep.n_windows = windows
    oos_returns = []
    for w in range(windows):
        start = w * fold
        is_end = start + int(fold * is_ratio)
        te_start = is_end
        te_end = start + fold
        if te_end <= te_start:
            continue
        # ---- 阶段0.5：Purged + Embargo 间隔（防 IS/OOS 信息泄漏）----
        # 【P1-15】frac>0 时按 min_iso_bars 兜底，避免整除下取整为 0 使隔离失效。
        purge_bars = int(purge * max(0, is_end - start)) or (min_iso_bars if purge > 0 else 0)
        embargo_bars = int(embargo * max(0, te_end - te_start)) or (min_iso_bars if embargo > 0 else 0)
        train = signals[start: max(start, is_end - purge_bars)]   # 训练段去尾
        test = signals[te_start + embargo_bars: te_end]            # 测试段去头
        rep.n_purged_bars += purge_bars
        rep.n_embargoed_bars += embargo_bars
        if not train or not test:
            # 间隔过大导致某 fold 为空 → 跳过该窗（不污染统计）
            continue
        # IS：估 p,b（用真实训练段成交反推胜率与赔率）
        bt_is = PaperBacktest(BacktestConfig(equity=100_000, seed=seed + w))
        bt_is.run_batch(train)
        p, b = estimate_pb(bt_is.trades)
        # 【C6 修复 · 2026-07-12】OOS：用 IS 估的 (p,b) 构造 calibrated KellyConfig 注入，
        # 使 quarter-Kelly 名义上限用 IS 实测胜率，而非默认保守缺省。
        # 若 IS 样本不足（p 退化到默认），calibrated=False 仍走保守路径，不超配。
        from ..risk.kelly import KellyConfig
        kc_oos = KellyConfig(win_rate_est=p, payoff_ratio_est=b, calibrated=bool(bt_is.trades))
        cfg = BacktestConfig(equity=100_000, seed=seed + w + 100)
        bt_oos = PaperBacktest(cfg)
        bt_oos.run_batch(test, kc=kc_oos)
        if bt_oos.stats().net_pnl_pct > 0:
            rep.oos_profit_wins += 1
        # OOS 收益序列（用于 DSR）
        eq = np.array(bt_oos.equity_curve, dtype=float)
        if len(eq) > 2:
            rets = eq[1:] / eq[:-1] - 1.0
            oos_returns.append(rets)
            rep.oos_sharpes.append(float(np.mean(rets) / np.std(rets, ddof=1) * math.sqrt(PERIODS_PER_YEAR_1H)) if np.std(rets, ddof=1) > 0 else 0.0)
        rep.oos_win_rate = rep.oos_profit_wins / max(1, windows)
    # 【C7 修复 · 2026-07-12】DSR/PBO 统计前提修正
    # 原实现把各 fold 的 equity_curve 直接 concat 算 DSR —— fold 间已归零/不连续，
    # 拼接跳变污染 DSR；PBO 用 oos_sharpes（每 fold 一个值）做 CSCV 近似，统计学前提不满足。
    # 修复：
    #   - DSR：对各 fold 独立 returns 序列分别算 deflated sharpe，再取均值（等权），
    #     避免跨 fold 不连续拼接。若仅 1 fold 则退化为该 fold 自身 DSR。
    #   - PBO：基于各 fold OOS 净收益序列（oos_returns 为逐 bar returns）做 CSCV 近似，
    #     用 nn 随机二分比较后半最小值 < 前半最小值的比例，而非用 sharpe 值列表。
    if oos_returns:
        per_fold_dsr = [_deflated_sharpe(r) for r in oos_returns if len(r) >= 5]
        rep.dsr = float(np.mean(per_fold_dsr)) if per_fold_dsr else 0.0
        # 合并所有 fold 的逐 bar returns 作为 PBO 的输入（nn 拆分），比 sharpe 列表更合理
        all_rets = np.concatenate(oos_returns)
        rep.pbo = _cscv_pbo(list(all_rets))
    else:
        rep.dsr = 0.0
        rep.pbo = 0.0
    rep.gate_c_pass = (rep.oos_win_rate >= 0.60) and (rep.dsr > 0) and (rep.pbo <= 0.20)
    return rep


def purged_embargo_cv(signals: list, windows: int = 6, is_ratio: float = 0.6,
                      embargo: float = 0.01, purge: float = 0.01, seed: int = 7) -> dict:
    """阶段0.5 受控比较：朴素 WF（embargo=purge=0）vs Purged+Embargo WF。

    返回两侧 WFReport 摘要 + 泄漏差（leakage_delta_dsr = 朴素DSR − 净化DSR）。
    若朴素 DSR 显著虚高（泄漏导致乐观），则 leakage_delta_dsr > 0，验证了
    阶段0.5 防泄漏层的必要性；净化后的 DSR 才是可信的 edge 验收值。
    """
    naive = walk_forward(signals, windows=windows, is_ratio=is_ratio,
                         seed=seed, embargo=0.0, purge=0.0)
    clean = walk_forward(signals, windows=windows, is_ratio=is_ratio,
                         seed=seed, embargo=embargo, purge=purge)
    return {
        "naive": naive,
        "clean": clean,
        "leakage_delta_dsr": naive.dsr - clean.dsr,
        "leakage_delta_pbo": clean.pbo - naive.pbo,   # 净化后 PBO 应更真实（通常↑）
        "gate_naive": naive.gate_c_pass,
        "gate_clean": clean.gate_c_pass,
    }
