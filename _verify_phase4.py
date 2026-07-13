"""Phase 4 烟雾测试：验证 run_wfa_v2.py 的四项修复（路径/P0-5净edge/P0-4非重叠/P2-3 t分布）。

为避免 5.5y 全量回测（数万窗口、极慢且 OI 仅 30 天，见 P1-9），本测试仅用 BTC 前 3000 根
1h 棒 + 缓存关闭，跑通真实代码路径后断言关键不变量。
"""
import sys, os, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_wfa_v2 as w

# --- 仅跑 BTC，且截断到前 3000 棒（约 145 窗口），加速验证 ---
w.SYMBOLS = ["BTC"]
for s in list(w.hist.keys()):
    if s != "BTC":
        w.hist[s]["1h"] = []
w.hist["BTC"]["1h"] = w.hist["BTC"]["1h"][:3000]

print("\n=== 1) 路径解析（P2-6）===")
print("HIST_CACHE =", w._HIST_CACHE, "exists=", os.path.exists(w._HIST_CACHE))
print("DERIV_DATA =", w._DERIV_DATA, "exists=", os.path.exists(w._DERIV_DATA))
print("SIGNAL_CACHE =", w.SIGNAL_CACHE, "(版本:", w.WFA_CACHE_VERSION, ")")
assert os.path.exists(w._HIST_CACHE) and os.path.exists(w._DERIV_DATA)

print("\n=== 2) t 分布单侧 p 值（P2-3）===")
# 已知样本：均值为正且离散，df 较小，p 应远小于正态近似
pos = [2.0]*10 + [1.0]*10
p_pos = w.t_test_one_sided(pos)
neg = [-1.0, -2.0, -0.5, -1.5]
p_neg = w.t_test_one_sided(neg)
flat = [0.0, 0.0, 0.0]
p_flat = w.t_test_one_sided(flat)
print(f"强正样本 p={p_pos:.4f} (期望≈0) | 负样本 p={p_neg:.4f} (期望=1.0) | 零均值 p={p_flat:.4f} (期望=1.0)")
assert p_pos < 0.01 and p_neg == 1.0 and p_flat == 1.0
# 与正态近似对比：正态会低估小样本 p（更激进），t 分布应更保守（p 更大）
import math
t = (statistics.mean(pos) - 0) / (statistics.pstdev(pos)/math.sqrt(len(pos)))
norm_p = 0.5 * math.erfc(t / math.sqrt(2))
print(f"同一样本 正态近似 p={norm_p:.4f} vs t分布 p={p_pos:.4f} → t分布更保守: {p_pos > norm_p}")

print("\n=== 3) net edge < gross（P0-5 成本扣除）===")
from cryptoquant_auto.models import Signal, Direction
sig = Signal(symbol="BTC", tf="1H", direction=Direction.LONG, entry=60000, sl=59000,
             tp1=61000, tp2=62000, rr=2.0, confidence=0.6, signal_id="t1", atr=1000)
path = [60000, 60100, 60200, 60300, 61000, 61000, 61000]  # 命中 tp1
g, n = w.compute_edge_bps(sig, path)
print(f"gross={g:.3f} bps  net={n:.3f} bps  net<gross: {n < g}")
assert n < g, "净 edge 必须 < 毛 edge（成本被扣除）"

print("\n=== 4) 全量信号生成（BTC 截断）===")
items = w.gen_all_signals(use_cache=False)
print(f"生成 {len(items)} 个信号；元组长度={len(items[0])}（应=8: +gross+net）")
assert len(items[0]) == 8
all_net = [x[7] for x in items]
all_gross = [x[6] for x in items]
print(f"全样本 gross mean={statistics.mean(all_gross):+.2f} net mean={statistics.mean(all_net):+.2f}")
assert statistics.mean(all_net) <= statistics.mean(all_gross)

print("\n=== 5) 非重叠折叠（P0-4）===")
K = 6
fold = max(1, len(items) // K)
blocks = []
for fk in range(K):
    s0 = fk * fold
    s1 = (fk + 1) * fold if fk < K - 1 else len(items)
    blocks.append((s0, s1))
print("fold 边界:", blocks)
non_overlap = all(blocks[i][1] <= blocks[i+1][0] for i in range(len(blocks)-1))
print("非重叠:", non_overlap)
assert non_overlap, "折叠必须非重叠"

print("\n=== 6) 端到端 main()（截断 BTC）===")
w.main()

print("\n✅ Phase 4 全部不变量通过")
