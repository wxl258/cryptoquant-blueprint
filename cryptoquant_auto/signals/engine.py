"""多因子评分信号引擎（吸收生产系统 signals/engine.py 优点 P0-A）。

设计要点（来自生产系统实测）：
- 10 维条件化加减分：ADX 分级 / DI 差 / RSI+恐慌贪婪极值 / 周线顺逆 / 资金费率 / 波动率扩张
- 动态门槛自适应 min_score_adj：恐慌+ADX 双重调整、弱趋势不放水、极端波动抬门槛
- ADX 分级闸门：≥25 全量 / 20-24 半量(上限4分) / <20 观望（均值回归友好）
- 牛市禁空 / 熊市禁多 / 震荡弱趋势保留双向（均值回归）

自包含：输入 K 线 + 辅助标量，输出 SignalCandidate（含 score / direction / 动态门槛）。
不接真钱，仅基于历史/Mock 数据驱动原型执行管线。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .indicators import calc_adx, calc_rsi, calc_atr, volatility_regime

# 评分映射参数（对齐生产系统 RC）
RC = {
    "min_adx": 25,        # 最低 ADX 门槛
    "min_score": 5,       # 默认最低评分门槛
    "ext_fr": 0.001,      # 极端费率阈值（0.1%）
    "rsi_long": 35,       # RSI 超卖（做多触发）
    "rsi_short": 65,      # RSI 超买（做空触发）
    "fg_ext": 20,         # 极端恐慌阈值
    "fg_ext_g": 80,       # 极端贪婪阈值
}


@dataclass
class SignalCandidate:
    symbol: str
    direction: str = "观望"      # 做多 / 做空 / 观望
    score: float = 0.0
    min_score_adj: float = 5.0   # 动态门槛
    conds: List[str] = field(default_factory=list)
    atr: float = 0.0
    atr_pct: float = 0.0
    adx: float = 20.0
    rr: float = 2.0

    @property
    def passed(self) -> bool:
        return self.direction != "观望" and self.score >= self.min_score_adj


@dataclass
class MarketContext:
    """评分所需的辅助标量（由 K 线/环境聚合得到，非真钱依赖）。"""
    fg_val: float = 50.0                 # 恐慌贪婪指数 0-100
    fr: float = 0.0                      # 当前资金费率（如 0.0003）
    fr_delta: float = 0.0                # 费率变化
    oi_pct: float = 0.0                  # 持仓量变化%
    wk_adx: Optional[tuple] = None       # 周线 (adx, dir_str, pdi, mdi) 或 (adx, dir_str)
    wk_dir: str = "unknown"              # 周线方向 上涨/下跌/偏多/偏空/unknown
    market_state: Optional[str] = None   # BULL/BEAR/RANGE（来自元认知，辅助）
    regime: Optional[str] = None         # TREND / RANGE / CRASH（来自 detect_regime，主路由）
    is_weekend: bool = False
    sr_risk: Optional[str] = None        # 支撑阻力风险描述


def gen_signal(symbol: str, candles_1h: List[dict], candles_4h: List[dict] = None,
               candles_1w: List[dict] = None, ctx: MarketContext = None) -> SignalCandidate:
    """生成单币种信号候选（吸收生产 gen_sig 核心逻辑，自包含重写）。"""
    ctx = ctx or MarketContext()
    c = SignalCandidate(symbol=symbol)
    adx, pdi, mdi = calc_adx(candles_1h)
    rsi = calc_rsi([x["c"] for x in candles_1h])
    atr, atr_pct = calc_atr(candles_1h)
    vol_regime, _ = volatility_regime(candles_1h)
    c.atr, c.atr_pct, c.adx = atr, atr_pct, adx

    adx4, pdi4, mdi4 = (20, 20, 20)
    if candles_4h:
        adx4, pdi4, mdi4 = calc_adx(candles_4h)
    tf4 = ("看涨" if adx4 >= RC["min_adx"] and pdi4 > mdi4
           else "看跌" if adx4 >= RC["min_adx"] and mdi4 > pdi4 else "震荡")

    # --- 方向初判（ADX 分级闸门）---
    if adx < 20:
        c.direction = "观望"
    elif adx < RC["min_adx"]:
        c.direction = "观望" if tf4 == "震荡" else ("做多" if tf4 == "看涨" else "做空")
        c.conds.append(f"ADX={adx} 弱趋势(20-24)半量评估")
    else:
        c.direction = "做多" if pdi > mdi else "做空"
        c.conds.append(f"ADX={adx} 趋势确立, DI差={abs(pdi-mdi):.0f}")

    # 周线方向
    wk_dir = ctx.wk_dir
    wk_adx_val, wk_adx_dir = 20.0, "unknown"
    if ctx.wk_adx:
        wk_adx_val = ctx.wk_adx[0]
        if len(ctx.wk_adx) >= 2:
            wk_adx_dir = ctx.wk_adx[1]

    # --- 动态门槛自适应 min_score_adj ---
    msa = float(RC["min_score"])
    # 恐慌指数 + ADX 中位数双重调整
    if ctx.fg_val <= RC["fg_ext"]:
        msa = max(4, msa - 2)          # 极端恐慌：降门槛抓反弹
        c.conds.append(f"极端恐慌(fg={ctx.fg_val})门槛-2")
    elif ctx.fg_val >= 55 and ctx.fg_val <= 70:
        msa = max(4, msa - 1)
    elif ctx.fg_val >= RC["fg_ext_g"]:
        msa = min(8, msa + 2)          # 极端贪婪：抬门槛防追高
        c.conds.append(f"极端贪婪(fg={ctx.fg_val})门槛+2")
    # 弱趋势不放水
    if adx < 25:
        msa = max(3, msa)
    # 方向敏感
    if c.direction == "做多" and ctx.fg_val >= RC["fg_ext_g"]:
        msa += 1
    if c.direction == "做空" and ctx.fg_val <= RC["fg_ext"]:
        msa += 1
    # 极端波动抬门槛（atr_pct=ATR占价百分比，calc_atr 返回，常态1-4%；
    #   <0.4% 极致低波动→假突破风险抬门槛；>5% 极端高波动→抬门槛+降回报比）
    if atr_pct < 0.4:
        msa = max(msa, 7)
        c.conds.append(f"极低波动(atr_pct={atr_pct:.2f}%)假突破风险抬门槛")
    elif atr_pct > 5.0:
        msa = max(msa, 6)
        c.rr = 1.5
        c.conds.append(f"极端高波动(atr_pct={atr_pct:.2f}%)抬门槛降回报比")
    msa = max(3, min(msa, 9))
    # 弱趋势(20-24)半量评估：门槛与封顶对齐，使半量路径真正可放行（修复 P1-1 死代码）
    if 20 <= adx < RC["min_adx"]:
        msa = min(msa, 5)
    c.min_score_adj = msa

    # --- 市场状态闸门（来自元认知，辅助）---
    if ctx.market_state == "BULL" and c.direction == "做空":
        c.direction = "观望"; c.conds.append("🤖状态=牛市 禁止做空")
    elif ctx.market_state == "BEAR" and c.direction == "做多":
        c.direction = "观望"; c.conds.append("🤖状态=熊市 禁止做多")
    elif ctx.market_state == "RANGE" and 20 <= adx < RC["min_adx"]:
        c.conds.append("🤖状态=震荡 弱趋势保留双向(均值回归)")

    # --- P0a：资金费率门控（修复"极负费率做多陷阱"）---
    # 极负费率做多/极正费率做空 = 历史回测陷阱（fr<-0.001 做多 4信号全亏；fr>+0.001 做空同理），予以 BLOCK。
    # 做多 fr<-ext_fr → BLOCK；做空 fr>+ext_fr → BLOCK（替换原 fr<0→做多+1~+2）。
    if c.direction == "做多" and ctx.fr < -RC["ext_fr"]:
        c.direction = "观望"
        c.conds.append(f"🔴fr={ctx.fr*100:.3f}% 极负费率做多陷阱→BLOCK")
    elif c.direction == "做空" and ctx.fr > RC["ext_fr"]:
        c.direction = "观望"
        c.conds.append(f"🔴fr={ctx.fr*100:.3f}% 极正费率做空陷阱→BLOCK")

    # --- P0b：regime 方向敏感路由（TREND/RANGE/CRASH 主路由）---
    # TREND: 双向全开；CRASH: 做空禁开 / 做多半量(降权)；RANGE: 双向禁开。
    rg = ctx.regime
    if rg == "RANGE":
        if c.direction != "观望":
            c.direction = "观望"
            c.conds.append("regime=RANGE 双向禁开(均值回归走MR路径)")
    elif rg == "CRASH":
        if c.direction == "做空":
            c.direction = "观望"
            c.conds.append("regime=CRASH 做空禁开")
        elif c.direction == "做多":
            c.score = max(0.0, c.score - 2)
            c.conds.append("regime=CRASH 做多半量(降权)")

    # --- 维度1: ADX 分级 ---
    if adx >= 45:
        c.score += 3; c.conds.append(f"ADX={adx} 强趋势")
    elif adx >= 35:
        c.score += 2; c.conds.append(f"ADX={adx} 明确趋势")
    elif adx >= 25:
        c.score += 1; c.conds.append(f"ADX={adx} 趋势确立")
    else:
        c.conds.append(f"ADX={adx} 趋势不足")

    # --- 维度2: DI 差（方向明确度）---
    did = abs(pdi - mdi)
    if did >= 25:
        c.score += 2; c.conds.append(f"DI差={did:.0f} 方向极明确")
    elif did >= 15:
        c.score += 1; c.conds.append(f"DI差={did:.0f} 方向可辨")
    else:
        c.conds.append(f"DI差={did:.0f} 方向模糊")

    # --- 维度3: RSI + 恐慌贪婪极值 ---
    if c.direction == "做多":
        if rsi < RC["rsi_long"] and ctx.fg_val <= RC["fg_ext"] and adx >= 25:
            c.score += 2; c.conds.append(f"RSI={rsi}+恐慌={ctx.fg_val} 双重超卖极值+趋势确认 +2")
        elif rsi < RC["rsi_long"]:
            c.score += 1; c.conds.append(f"RSI={rsi} 超卖 +1")
        elif RC["rsi_long"] <= rsi <= RC["rsi_short"]:
            pass  # 中性不贡献分
        elif rsi > RC["rsi_short"]:
            if adx >= 35:
                c.score += 1; c.conds.append(f"RSI={rsi} 高位但强趋势支撑")
            else:
                c.conds.append(f"RSI={rsi} 高位无趋势支撑")
    else:  # 做空
        if rsi > RC["rsi_short"] and ctx.fg_val >= RC["fg_ext_g"] and adx >= 25:
            c.score += 2; c.conds.append(f"RSI={rsi}+贪婪={ctx.fg_val} 双重超买极值+趋势确认 +2")
        elif rsi > RC["rsi_short"]:
            c.score += 1; c.conds.append(f"RSI={rsi} 超买 +1")
        elif RC["rsi_long"] <= rsi <= RC["rsi_short"]:
            pass
        elif rsi < RC["rsi_long"]:
            c.score += 1; c.conds.append(f"RSI={rsi} 超卖加速 空头延续")

    # --- 维度4: 周线顺逆大势 ---
    if c.direction == "做多" and wk_dir in ("上涨",):
        c.score += 1; c.conds.append("周线=上涨 顺大势 +1")
    elif c.direction == "做空" and wk_dir in ("下跌",):
        c.score += 1; c.conds.append("周线=下跌 顺大势 +1")
    if c.direction != "观望":
        if c.direction == "做多" and wk_adx_dir == "看跌":
            c.score -= 1; c.conds.append(f"周线ADX={wk_adx_val:.0f}看跌 逆大趋势 -1")
        elif c.direction == "做空" and wk_adx_dir == "看涨":
            c.score -= 1; c.conds.append(f"周线ADX={wk_adx_val:.0f}看涨 逆大趋势 -1")

    # --- P1-2: 三条件交（做空 + TREND + 周线顺向）强化为网关 ---
    # 做空+TREND+周线下跌 三条件交：历史 edge 最强，升级为强制网关而非软 +1。
    # 满足 → 额外 +1；做空但不满足(TREND 且 周线下跌) → 强降权 -2（实质只放行该组合）。
    if c.direction == "做空":
        three_cond = (ctx.regime == "TREND" and wk_dir == "下跌")
        if three_cond:
            c.score += 1; c.conds.append("三条件交(做空+TREND+周线下跌)强网关 +1")
        else:
            c.score -= 2; c.conds.append("做空未满足三条件交(需TREND+周线下跌) 降权 -2")
    # --- 方向中立对称（R3）：做多 + TREND + 周线上涨 三条件交镜像做空网关 ---
    # 原实现仅对做空设强网关，做多无对称项 → 结构性方向偏置，违反宪法 R3。
    # 镜像后：做多+TREND+周线上涨 同样 +1；TREND 但周线非上涨 → 对称降权 -2。
    # （CRASH 做多已在上方单独降权，regime!=TREND 不重复施加，避免叠加惩罚。）
    if c.direction == "做多" and ctx.regime == "TREND":
        three_cond_long = (wk_dir == "上涨")
        if three_cond_long:
            c.score += 1; c.conds.append("三条件交(做多+TREND+周线上涨)强网关 +1")
        else:
            c.score -= 2; c.conds.append("做多未满足三条件交(需TREND+周线上涨) 降权 -2")

    # --- 维度5/6: 资金费率（仅 Δ费率动量；绝对费率门控已由 P0a 处理）---
    # 注意：不再奖励"绝对负费率做多 / 绝对正费率做空"——那正是极负费率做多陷阱的源头。
    if c.direction == "做多":
        if ctx.fr_delta > 0.0005:
            c.score -= 1; c.conds.append(f"Δ费率={ctx.fr_delta*100:.4f}% 多头过热 -1")
        elif ctx.fr_delta < -0.0005:
            c.score += 1; c.conds.append(f"Δ费率={ctx.fr_delta*100:.4f}% 空头挤压潜力 +1")
    else:  # 做空
        if ctx.fr_delta < -0.0005:
            c.score -= 1; c.conds.append(f"Δ费率={ctx.fr_delta*100:.4f}% 空头过热 -1")
        elif ctx.fr_delta > 0.0005:
            c.score += 1; c.conds.append(f"Δ费率={ctx.fr_delta*100:.4f}% 多头挤压潜力 +1")

    # --- 维度7: 波动率扩张 ---
    if vol_regime == "expanding":
        c.score += 1; c.conds.append("波动率扩张 趋势延续 +1")
    elif vol_regime == "contracting":
        c.conds.append("波动率收敛 趋势减弱")

    # --- 弱趋势(20-24)评分上限5分（半量，与 msa<=5 对齐，路径已激活）---
    if 20 <= adx < RC["min_adx"] and c.score > 5:
        c.score = 5; c.conds.append("弱趋势评分封顶5分(半量)")

    c.conds.append(f"动态门槛={c.min_score_adj:.0f} ADX={adx} RSI={rsi} ATR%={atr_pct}")
    return c
