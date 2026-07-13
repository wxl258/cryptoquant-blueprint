"""执行成本模型（来自执行成本专家方案，单位 bps）。

用于回测/测试网真实建模：maker/taker 费率、滑点、资金费、beta。
effective_edge_bps() 直接回应「edge<成本」疑虑：taker 全负，仅 maker 为正。
"""
from __future__ import annotations

from typing import Dict

# 各币成本参数（bps），取 VIP0 公开量级；fund 取最不利正值
COST: Dict[str, Dict[str, float]] = {
    "BTC": {"maker": 2.0, "taker": 5.0,   "slip_taker": 1.0, "fund": 2.0, "beta": 1.00},
    "ETH": {"maker": 2.0, "taker": 5.0,   "slip_taker": 2.0, "fund": 2.0, "beta": 0.85},
    "BNB": {"maker": 2.0, "taker": 5.5,   "slip_taker": 3.0, "fund": 2.0, "beta": 0.70},
    "SOL": {"maker": 2.0, "taker": 5.5,   "slip_taker": 4.0, "fund": 2.5, "beta": 0.80},
    "XRP": {"maker": 2.0, "taker": 5.5,   "slip_taker": 6.0, "fund": 2.5, "beta": 0.55},
}

# 【C5 修复 · 2026-07-12】统一 gross edge 单一真相源
# ---------------------------------------------------------------------------
# 原代码存在两套 gross edge 常量并存、口径矛盾（审计 C5 红队复核确认）：
#   - GROSS_EDGE_BPS = 0.0  → 被 effective_edge_bps() 用于报告/成本建模（暗示"无 edge"）
#   - GATE_B_LOCKED_GROSS_EDGE_BPS = -2.5 → 被 gate.py Gate B 用于硬锁全市场拒单
# 两者指向同一概念"系统毛 edge"却有两个真相源，任何 set_gross_edge_bps() 调用
# 都会让 effective_edge_bps 与 Gate B 脱钩，属状态管理混乱。
#
# 修复：合并为单一模块级锚点 ACTIVE_GROSS_EDGE_BPS，effective_edge_bps 与 Gate B
# 共用同一值；-2.5 明确标注为「待大样本复核的临时 fail-closed 锚点」，非永久锁。
#
# 解锁契约（显式、可审计）：
#   仅当「每 regime OOS ≥50 笔」的大样本复核通过，并经 calibrate_gross_edge_bps()
#   反推 + apply_locked_gate_b() 重新锚定版本号后，方可上调锚点放行。
#   任何随笔改值都视为临时 override，不入正式校准版本链。
ACTIVE_GROSS_EDGE_BPS = -2.5  # 默认 fail-closed 锚点（临时，待复核解锁）
GROSS_EDGE_BPS = ACTIVE_GROSS_EDGE_BPS  # 别名保持向后兼容，单真相源
GATE_B_LOCKED_GROSS_EDGE_BPS = ACTIVE_GROSS_EDGE_BPS  # 与 ACTIVE 同一对象，消除矛盾
GATE_B_CALIBRATION_VERSION = "2026-07-12.unified-single-source"
GATE_B_CALIBRATION_NOTE = (
    "C5修复：合并 GROSS_EDGE_BPS 与 GATE_B_LOCKED 为单一锚点 ACTIVE_GROSS_EDGE_BPS=-2.5；"
    "原注释称基于跨所(币安/OKX)5年OOS pooled(-2.19/+0.62)，但该数值代码从未计算，"
    "仅为保守缺省。解锁须经大样本复核+apply_locked_gate_b()升版，非临时override。"
)


def effective_edge_bps(coin: str, taker: bool = False, slip: bool = True,
                       fund: bool = True) -> float:
    """含成本后的净 edge（bps）。始终使用单一锚点 ACTIVE_GROSS_EDGE_BPS（C5 修复）。"""
    c = COST.get(coin, COST["ETH"])
    fee = c["taker"] if taker else c["maker"]
    s = c["slip_taker"] if (taker and slip) else 0.0
    f = c["fund"] if fund else 0.0
    return ACTIVE_GROSS_EDGE_BPS - fee - s - f


def worst_case_pnl(coin: str) -> float:
    """最不利成本组合（taker + max 滑点 + 正资金费）净 edge。对应 Gate B。"""
    return effective_edge_bps(coin, taker=True, slip=True, fund=True)


def beta(coin: str) -> float:
    return COST.get(coin, COST["ETH"])["beta"]


def funding_cost(coin: str, periods: float) -> float:
    """持仓 periods 个资金费结算期的成本（bps）。periods = 持仓时长/8h。"""
    return COST.get(coin, COST["ETH"])["fund"] * periods


def gate_b_ok() -> Dict[str, bool]:
    """Gate B 成本敏感度：最不利组合净 edge>0 方可通过。"""
    return {c: worst_case_pnl(c) > 0 for c in COST}


def set_gross_edge_bps(value: float) -> None:
    """用真实 OOS 实测结果覆盖默认毛 edge（替换硬编码 8bps 谎言）。

    【C5 修复】同时更新 ACTIVE_GROSS_EDGE_BPS / GATE_B_LOCKED 两个别名，
    确保 effective_edge_bps 与 Gate B 共用同一真相源，不再脱钩。
    注意：此为临时 override，不入正式校准版本链；正式解锁须走
    calibrate_gross_edge_bps() + apply_locked_gate_b()。
    """
    global GROSS_EDGE_BPS, ACTIVE_GROSS_EDGE_BPS, GATE_B_LOCKED_GROSS_EDGE_BPS
    v = float(value)
    GROSS_EDGE_BPS = v
    ACTIVE_GROSS_EDGE_BPS = v
    GATE_B_LOCKED_GROSS_EDGE_BPS = v


def calibrate_gross_edge_bps(per_coin_net_edge_bps: Dict[str, float],
                             haircut: float = 0.5) -> float:
    """用 OOS 实测净 edge 反推真实毛 edge，替换 8bps 谎言（共识#2 / Gate B 校准）。

    ⚠️ 复核通道（非默认）。本函数在 demo 中仅供 --recalibrate 显式复核打印用，
       其结果【不】写回 GROSS_EDGE_BPS 默认路径（避免校准值漂移）。要正式升级锚点，
       须经「每 regime OOS ≥50 笔」大样本复核后，手动调用 apply_locked_gate_b() 重锚。

    逻辑链：gross - cost = net  ⇒  gross = net + cost。
      - 用各币 OOS 净 edge 反推毛 edge = 净edge + 最不利成本(绝对值)；
      - 取最保守聚合（各币最小毛 edge）：不拿表现最好币乐观外推到其他币；
      - 再乘 haircut(walk-forward 安全垫，默认 0.5) 防过拟合乐观；
      - 返回校准值（供打印/人工审计），不自动覆盖全局默认 fail-closed 锚点。

    若所有币净 edge 为负、反推毛 edge 仍 ≤ 成本，则校准值非正 → Gate B 如实失败（no-go）：
    这正是诚实结论——原型尚无经数据验证的 edge，禁止在成本之上幻想盈利。
    """
    if not per_coin_net_edge_bps:
        return 0.0
    gross_per: Dict[str, float] = {}
    for c, net in per_coin_net_edge_bps.items():
        cost_abs = abs(worst_case_pnl(c))        # 最不利成本绝对值
        gross_per[c] = net + cost_abs
    cal = min(gross_per.values()) * haircut     # 最保守：最差币 × 安全垫
    return cal


def apply_locked_gate_b() -> float:
    """应用【冻结锚点】作为 Gate B 默认毛 edge（圆桌共识二 · 2026-07-11）。

    【C5 修复】重新锚定到 ACTIVE_GROSS_EDGE_BPS（单一真相源），并自增版本号，
    使 Gate B 在任何随机样本下都维持 fail-closed，杜绝漂移。
    返回当前生效的锚点值与版本号，供审计。
    """
    global GATE_B_CALIBRATION_VERSION
    set_gross_edge_bps(ACTIVE_GROSS_EDGE_BPS)
    GATE_B_CALIBRATION_VERSION = "2026-07-12.unified-" + str(abs(int(ACTIVE_GROSS_EDGE_BPS * 100)))
    return ACTIVE_GROSS_EDGE_BPS
