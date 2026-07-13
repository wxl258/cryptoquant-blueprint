"""蓝图阶段1 验证：概率化贝叶斯脊柱 + 交易宪法。

跑在原型沙盒（纯信号融合校验，不触发任何资金操作）。覆盖：
  案例1 三专家一致看多 → 融合做多、低不确定、过宪法
  案例2 专家分歧 → 高不确定 → 软降级观望
  案例3 live_capital=True → 宪法硬锁、否决一切
  案例4 方向中立对称性校验（上行/下行镜像后验对称）
"""
from __future__ import annotations

import numpy as np

from .signals.engine import gen_signal, MarketContext
from .core.metacontroller import (BayesianMetacontroller, opinion_from_candidate,
                                  LONG, SHORT, HOLD)
from .risk.constitution import TradingConstitution


def _ctx(fg, fr, regime, wk_dir="上涨"):
    return MarketContext(fg_val=fg, fr=fr, regime=regime, wk_dir=wk_dir)


def _candle(n, drift):
    # 强趋势合成 1h K 线（OHLC 齐全，足够长以抬升 ADX ≥ 25，触发方向判定）
    price = 100.0
    out = []
    for i in range(n):
        price += drift                       # 每根 K 线性推进
        c = price
        span = abs(drift) * 0.5 + 0.5        # 高低波动幅，保证 TR>0
        out.append({"c": c, "h": c + span, "l": c - span, "v": 1000.0})
    return out


def _show(d):
    print(f"  标的={d.symbol} 决策={d.action_name} 置信={d.confidence:.2f} "
          f"不确定={d.uncertainty:.2f} 降级={d.degraded}")
    flag = "✅通过" if d.constitution_ok else "❌否决: " + "; ".join(d.violations)
    print(f"  宪法={flag}")
    print(f"  依据={d.rationale[:140]}")


def run():
    mc = BayesianMetacontroller(uncertainty_thresh=0.55)
    const = TradingConstitution(live_capital=False)

    print("=" * 64)
    print("案例1：三专家一致看多 → 融合做多，低不确定，过宪法")
    up = _candle(40, 2.0)
    opinions = [opinion_from_candidate(gen_signal("BTC", up, ctx=_ctx(fg, 0.0001, "TREND")), src)
                for src, fg in [("trend", 60), ("momentum", 65), ("sentiment", 55)]]
    d = mc.decide(opinions, "BTC")
    v = const.check(d)
    d.constitution_ok, d.violations = v.compliant, v.violations
    _show(d)

    print("=" * 64)
    print("案例2：专家分歧（多/空/观望）→ 高不确定 → 软降级观望")
    c_l = gen_signal("BTC", up, ctx=_ctx(70, 0.0001, "TREND", "上涨"))
    c_s = gen_signal("BTC", _candle(40, -2.0), ctx=_ctx(30, -0.0002, "TREND", "下跌"))
    c_h = gen_signal("ETH", up, ctx=_ctx(50, 0.0001, "RANGE", "未知"))
    opinions2 = [opinion_from_candidate(c_l, "trend"),
                 opinion_from_candidate(c_s, "sentiment"),
                 opinion_from_candidate(c_h, "range")]
    d2 = mc.decide(opinions2, "BTC")
    v2 = const.check(d2)
    d2.constitution_ok, d2.violations = v2.compliant, v2.violations
    _show(d2)

    print("=" * 64)
    print("案例3：live_capital=True → 宪法硬锁，否决一切")
    const_live = TradingConstitution(live_capital=True)
    d3 = mc.decide(opinions, "BTC")
    v3 = const_live.check(d3)
    d3.constitution_ok, d3.violations = v3.compliant, v3.violations
    _show(d3)

    print("=" * 64)
    print("案例4：方向中立对称性校验（镜像上下文）")
    # 方向中立要求：上行场景与其镜像（价格反向 + wk_dir 翻转 + fr 取反）
    # 下，LONG/SHORT 后验应互换且数值对称。原测试下行复用上行上下文
    # （wk_dir 同为"上涨"）导致三条件交做空降权，制造虚假不对称——已修正。
    def _mirror(ctx: MarketContext) -> MarketContext:
        wk_map = {"上涨": "下跌", "下跌": "上涨", "偏多": "偏空",
                  "偏空": "偏多", "unknown": "unknown"}
        return MarketContext(fg_val=ctx.fg_val, fr=-ctx.fr, regime=ctx.regime,
                             wk_dir=wk_map.get(ctx.wk_dir, ctx.wk_dir))
    ctx_up = _ctx(50, 0.0001, "TREND")          # wk_dir=上涨
    ctx_dn = _mirror(ctx_up)                     # wk_dir=下跌, fr=-0.0001
    o_up = opinion_from_candidate(gen_signal("BTC", _candle(40, 2.0), ctx=ctx_up), "t")
    o_dn = opinion_from_candidate(gen_signal("BTC", _candle(40, -2.0), ctx=ctx_dn), "t")
    sym = np.allclose(o_up.probs[[LONG, SHORT]], o_dn.probs[[SHORT, LONG]], atol=1e-6)
    print(f"  上行后验={np.round(o_up.probs,3)}")
    print(f"  下行后验={np.round(o_dn.probs,3)}")
    print(f"  方向中立对称性: {'✅通过' if sym else '❌失败'}")


if __name__ == "__main__":
    run()
