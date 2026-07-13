"""分级 KillSwitch（Fail-closed 核心）。

L1 暂停新开 / L2 降险 / L3 生存态（自动减仓不自动全平）。
自动只做安全动作；恢复必须人工 ACK（由 engine 的 manual_resume 处理）。
"""
from __future__ import annotations

from enum import Enum


class KillLevel(Enum):
    NORMAL = 0
    L1_PAUSE_NEW = 1     # 暂停新开，持有仓按原计划管理
    L2_REDUCE = 2        # 降杠杆、缩减新仓
    L3_SURVIVE = 3       # 自动减仓，不自动全平


# 生存态减仓目标：L3 触发后持仓等比缩减至该比例（圆桌共识#10：转稳定币 + 降仓50%）
SURVIVAL_REDUCE_TO = 0.5


class KillSwitch:
    def __init__(self):
        self.level = KillLevel.NORMAL
        self.daily_pnl = 0.0       # 当日权益收益（分数）
        self.peak_dd = 0.0         # 峰值回撤（负分数）
        self.loss_streak = 0       # 连亏笔数
        self.btc_vol_sigma = 0.0   # BTC 1h 波动（单位 σ）
        self.api_fail_rate = 0.0   # API 失败率 0~1
        self.margin_ratio = 99.0   # 保证金率（倍数，默认极高=安全）
        self.black_swan = False
        self.ack_required = False  # 是否需人工 ACK 才能恢复
        self.oi_spike_pct = 0.0    # 持仓量(OI)异动%（共识#10：事后触发器）
        self.fr_spike = 0.0        # 资金费率异动（绝对值）
        self.atr_sigma_spike = 0.0 # ATR 波动率异动（σ 倍数）

    def update(self, *, daily_pnl=0.0, peak_dd=0.0, loss_streak=0,
               btc_vol_sigma=0.0, api_fail_rate=0.0, margin_ratio=99.0,
               black_swan=False, oi_spike_pct=0.0, fr_spike=0.0,
               atr_sigma_spike=0.0) -> KillLevel:
        self.daily_pnl = daily_pnl
        self.peak_dd = peak_dd
        self.loss_streak = loss_streak
        self.btc_vol_sigma = btc_vol_sigma
        self.api_fail_rate = api_fail_rate
        self.margin_ratio = margin_ratio
        self.black_swan = black_swan
        self.oi_spike_pct = oi_spike_pct
        self.fr_spike = fr_spike
        self.atr_sigma_spike = atr_sigma_spike
        # 若此前已要求 ACK 且仍处高危，保持不自动恢复
        if self.ack_required and self.level != KillLevel.NORMAL:
            return self.level
        self._eval()
        return self.level

    def _eval(self) -> None:
        # L3 最高优先（生存态）
        if self.margin_ratio < 1.3 or self.black_swan:
            self.level = KillLevel.L3_SURVIVE
            return
        # L2
        if self.daily_pnl <= -0.05 or self.peak_dd <= -0.10 or self.loss_streak >= 5:
            self.level = KillLevel.L2_REDUCE
            return
        # L1
        if (self.daily_pnl <= -0.03 or self.peak_dd <= -0.05 or self.loss_streak >= 3
                or self.btc_vol_sigma > 2.5 or self.api_fail_rate > 0.10):
            self.level = KillLevel.L1_PAUSE_NEW
            return
        # 共识#10 事后触发器：OI/资金费/ATR 异动即暂停新开，不幻想事前逃顶
        if self.oi_spike_pct > 0.10 or self.fr_spike > 0.001 or self.atr_sigma_spike > 3.0:
            self.level = KillLevel.L1_PAUSE_NEW
            return
        self.level = KillLevel.NORMAL

    def allows_new(self) -> bool:
        """是否允许新开仓（L1 及以上即暂停新开）。"""
        return self.level is KillLevel.NORMAL

    def reduce_mode(self) -> bool:
        return self.level.value >= KillLevel.L2_REDUCE.value

    def survival_action(self) -> dict:
        """L3 生存态动作（圆桌共识#10：事后触发器 → 转稳定币 + 降仓50%）。

        返回减仓指令，由引擎在下个 step 对现有持仓执行等比减至 SURVIVAL_REDUCE_TO：
          - reduce_to: 持仓量缩减目标比例（0.5 = 减掉一半）
          - to_stablecoin: 是否标记「转出至稳定币」意图（实际转账由适配器实现，
            原型仅标记意图并减仓，符合 Fail-closed「只做安全动作」）
          - active: 是否处于需执行的生存态（仅 L3_SURVIVE 为 True）
        """
        active = self.level is KillLevel.L3_SURVIVE
        return {
            "active": active,
            "reduce_to": SURVIVAL_REDUCE_TO if active else 1.0,
            "to_stablecoin": active,
        }

    def manual_resume(self) -> None:
        """Fail-closed：恢复必须人工 ACK。"""
        self.ack_required = False
        self.level = KillLevel.NORMAL
        self.daily_pnl = self.peak_dd = 0.0
        self.loss_streak = 0
        self.btc_vol_sigma = 0.0
        self.api_fail_rate = 0.0
        self.margin_ratio = 99.0
        self.black_swan = False
