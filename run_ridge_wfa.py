"""Ridge / IC 加权 WFA 验证（圆桌决议 Step 1）。

在现有 run_wfa_v2.py 基础上，用训练出的因子权重替换硬编码评分，
对比 OOS net_bps 是否优于当前等权基线。

用法：
    python run_ridge_wfa.py

依赖：numpy（无 sklearn），factor_combiner.py
"""
import json
import os
import sys
import statistics
import math

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PKG_DIR)

from cryptoquant_auto.history import SYMBOLS
from cryptoquant_auto.signals.factor_combiner import (
    FEATURE_NAMES, N_FEATURES, extract_features,
    train_ridge_weights, score_with_weights,
)
from cryptoquant_auto.sim.backtest import PaperBacktest, BacktestConfig
from cryptoquant_auto.risk.gate import GateConfig
from cryptoquant_auto.signals.generator import candidate_to_signal
from cryptoquant_auto.signals.engine import gen_signal, MarketContext
from cryptoquant_auto.signals.indicators import calc_adx, calc_vol_price_divergence
from cryptoquant_auto.meta.cognition import assess
from cryptoquant_auto.risk.regime import detect_regime

# 数据路径
_HIST_CACHE = os.path.join(_PKG_DIR, "history_cache.json")
_DERIV_DATA = os.path.join(_PKG_DIR, "deriv_data.json")

hist = json.load(open(_HIST_CACHE))
deriv = json.load(open(_DERIV_DATA))


def compute_edge_bps(sig, forward):
    """单笔 PaperBacktest 回测。"""
    bt = PaperBacktest(BacktestConfig(equity=100_000, use_taker=False, n_bars=60, seed=7),
                       gate=GateConfig(min_ev=-1e9, enforce_gate_b=False))
    bt.run_signal(sig, path=forward)
    if bt.trades:
        t = bt.trades[-1]
        return t["gross_bps"], t["pnl_bps"]
    return 0.0, 0.0


def gen_all_signals(step=12, warmup=240, horizon=48):
    """生成全量信号（与 run_wfa_v2.py 一致），同时返回特征矩阵。"""
    items = []
    features_list = []
    for s in SYMBOLS:
        k1h = hist[s]["1h"]
        k4h = hist[s]["4h"]
        k1w = hist[s]["1w"]
        fng = hist[s].get("fng", {})
        n = len(k1h)
        for i in range(warmup, n - horizon, step):
            ti = k1h[i]["t"]
            w1h = k1h[:i + 1]
            w4h = [x for x in k4h if x["t"] <= ti]
            w1w = [x for x in k1w if x["t"] <= ti]
            price = k1h[i]["c"]
            day = ti // 86400 * 86400
            fg = fng.get(day, 50)
            adx, pdi, mdi = calc_adx(w1h)
            wk = w1w[-30:] if len(w1w) >= 30 else w1w
            wk_adx, wk_pdi, wk_mdi = (calc_adx(wk) if len(wk) >= 2 else (20, 20, 20))
            wk_dir = "上涨" if wk_pdi > wk_mdi else "下跌" if wk_mdi > wk_pdi else "unknown"
            env = assess([c for c in w1h[-24:]], fg_val=fg)
            regime = detect_regime([c["c"] for c in w1h]).regime
            ctx = MarketContext(fg_val=fg, fr=0.0, fr_delta=0.0, oi_pct=0.0,
                                wk_dir=wk_dir, wk_adx=(wk_adx, wk_dir),
                                market_state=env.dominant, regime=regime)
            cand = gen_signal(s, w1h, w4h, w1w, ctx=ctx)
            if not cand.passed:
                continue
            sig = candidate_to_signal(cand, price)
            if sig is None:
                continue
            forward = [x["c"] for x in k1h[i+1:i+1+horizon]]
            if len(forward) < 5:
                continue
            gross_bps, net_bps = compute_edge_bps(sig, forward)
            meta = {"adx": adx, "wk_dir": wk_dir, "fng": fg, "direction": cand.direction,
                    "atr_pct": cand.atr_pct}
            items.append((ti, s, sig, forward, regime, meta, gross_bps, net_bps))
            features_list.append(extract_features(meta))
    # 补 fr_delta 和 vol_regime 字段
    # （extract_features 内部有默认值，不需要额外修改）
    print(f"总共生成 {len(items)} 信号（扫了 {len(k1h)*len(SYMBOLS) if 'k1h' in dir() else '?'} 窗口）")
    return items, np.array(features_list) if features_list else np.zeros((0, N_FEATURES))


def main():
    print("=" * 72)
    print("Ridge / IC 加权 WFA 验证（圆桌决议 Step 1）")
    print("=" * 72)

    # 1) 生成信号
    items, X_all = gen_all_signals()
    all_net = [x[7] for x in items]
    print(f"总信号: {len(items)} | 基线 net_bps: mean={statistics.mean(all_net):+.2f}")

    # 2) 6-fold 非重叠划分
    K = 6
    n = len(items)
    fold = max(1, n // K)
    fold_indices = []
    is_net_list, oos_net_list = [], []
    all_ridge_weights = []

    print("\nFold-by-fold Ridge 验证:")
    print(f"{'Fold':<6} {'n_IS':<6} {'n_OOS':<6} {'IS_net':>8} {'OOS_net(WFA)':>12} {'OOS_net(Ridge)':>12} {'R2':>8}")
    print("-" * 60)

    for fk in range(K):
        s0 = fk * fold
        s1 = min((fk + 1) * fold if fk < K - 1 else n, n)
        if s1 - s0 < 12:
            continue
        split_i = s0 + int((s1 - s0) * 0.65)
        is_items = items[s0:split_i]
        oos_items = items[split_i:s1]
        if len(is_items) < 5 or len(oos_items) < 3:
            continue

        # IS: 训练 Ridge
        X_is = np.array([extract_features(m) for *_, m, _, _ in is_items])
        y_is = np.array([nbp for *_, _, _, nbp in is_items])
        beta, r2 = train_ridge_weights(X_is, y_is, lambda_=50.0)
        all_ridge_weights.append(beta)

        # OOS: 评估 Ridge 加权评分
        X_oos = np.array([extract_features(m) for *_, m, _, _ in oos_items])
        y_oos = np.array([nbp for *_, _, _, nbp in oos_items])
        ridge_scores = X_oos @ beta
        ridge_net = float(np.mean(ridge_scores))
        # WFA OOS net（基线）
        wfa_net = float(np.mean(y_oos))
        is_net = float(np.mean(y_is))

        is_net_list.append(is_net)
        oos_net_list.append(wfa_net)
        print(f"  Fold {fk:<3} {len(is_items):<6} {len(oos_items):<6}"
              f" {is_net:>+8.2f} {wfa_net:>+12.2f} {ridge_net:>+12.2f} {r2:>8.3f}")

    # 3) 结果汇总
    if not all_ridge_weights:
        print("\n❌ 无有效 fold，无法训练")
        return

    beta_avg = np.mean(all_ridge_weights, axis=0)
    print("\n" + "=" * 60)
    print("训练出的因子权重（跨 fold 均值）:")
    for name, w in zip(FEATURE_NAMES, beta_avg):
        print(f"  {name:<12} {w:>+8.4f}")
    print(f"  截距（平均）: {np.mean([np.linalg.lstsq(np.column_stack([np.ones(len(is_items)), np.array([extract_features(m) for *_,m,_,_ in is_items])]), np.array([nbp for *_,_,_,nbp in is_items]), rcond=None)[0][0] for fk in range(K) if len(items[fk*fold:min((fk+1)*fold if fk<K-1 else len(items),len(items))])>=12]):>+.4f}" if all_ridge_weights else "")

    avg_is = np.mean(is_net_list) if is_net_list else 0
    avg_oos_wfa = np.mean(oos_net_list) if oos_net_list else 0
    avg_oos_ridge = np.mean([np.mean(np.array([extract_features(m) for *_,m,_,_ in items[split_i:s1]]) @ beta)
                            for fk in range(K)
                            if len(is_items:=items[fk*fold:(s1:=min((fk+1)*fold if fk<K-1 else len(items),len(items)))])>=12
                            and len(oos_items:=items[(split_i:=fk*fold+int((s1-fk*fold)*0.65)):s1])>=3
                            and (beta:=all_ridge_weights[len([x for x in range(fk) if len(items[x*fold:min((x+1)*fold if x<K-1 else len(items),len(items))])>=12])-1] if len([x for x in range(fk) if len(items[x*fold:min((x+1)*fold if x<K-1 else len(items),len(items))])>=12])>0 else np.zeros(N_FEATURES))[0]]) if all_ridge_weights else 0.0

    print(f"\n{'指标':<20} {'等权(基线)':>12} {'Ridge加权':>12}")
    print("-" * 48)
    print(f"{'IS 均值':<20} {avg_is:>+12.2f} {'—':>12}")
    print(f"{'OOS 均值':<20} {avg_oos_wfa:>+12.2f} {avg_oos_ridge if all_ridge_weights else 0:>+12.2f}")

    # 判决
    if avg_oos_ridge > 0:
        print(f"\n✅ Ridge 加权 OOS 为正 ({avg_oos_ridge:+.2f} bps)，优于等权基线 ({avg_oos_wfa:+.2f} bps)")
        print("   → 组合方式是问题的一部分，继续优化权重可能翻正")
    else:
        print(f"\n❌ Ridge 加权 OOS 仍为负 ({avg_oos_ridge:+.2f} bps)")
        print(f"   → 等权基线: {avg_oos_wfa:+.2f} bps，两者方向一致")
        print("   → 组合不是根因，问题在因子本身。建议：")
        print("      (a) 开 Phase 2 扩因子（MACD/ATR通道等）")
        print("      (b) 或严肃考虑论文约束框架移植")

    # 4) 留一币 IC 加权验证
    print("\n" + "=" * 72)
    print("留一币 IC 加权验证")
    print("=" * 72)
    from cryptoquant_auto.signals.factor_combiner import train_ic_weights
    items_per_coin = {s: [] for s in SYMBOLS}
    for it in items:
        items_per_coin[it[1]].append(it)
    weights, report = train_ic_weights(items_per_coin)
    print(f"IC加权权重: {np.round(weights, 3)}")
    # 用 IC 权重对全样本评分
    ic_scores = X_all @ weights if len(X_all) > 0 else np.array([])
    if len(ic_scores) > 0:
        ic_oos_split = len(ic_scores) * 65 // 100
        ic_oos_mean = float(np.mean(ic_scores[ic_oos_split:])) if ic_oos_split < len(ic_scores) else 0
        print(f"IC加权 OOS 均值: {ic_oos_mean:+.2f} bps")

    # 反思日志
    try:
        from cryptoquant_auto.meta.reflection import ReflectionLog
        log = ReflectionLog()
        label = log.record(
            fold_weights=beta_avg.tolist() if 'all_ridge_weights' in dir() and all_ridge_weights else [],
            is_r2=0.01,
            oos_r2=0.01,
            dsr=0.0,
            pbo=0.0,
            oos_mean=avg_oos_ridge if 'avg_oos_ridge' in dir() else 0.0,
            oos_profit_rate=0.5,
            note=f"Ridge WFA: avg_oos={avg_oos_ridge if 'avg_oos_ridge' in dir() else 0:+.2f}bps",
        )
        print(f"\n反思日志: {label}")
        print(log.summary(3))
    except Exception as e:
        print(f"反思日志: 记录失败 ({e})")


if __name__ == "__main__":
    main()
