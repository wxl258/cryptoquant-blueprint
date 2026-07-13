"""硬熔断（移植服务器 circuit_breaker 优点，与 KillSwitch 分工）。

KillSwitch = 回撤分级降级（仍可部分交易，人工ACK恢复）。
CircuitBreaker = 连亏/异常硬停（Trip=整体切断电源，绝不自动恢复）。

Fail-closed：自动只做"停"动作；恢复必须 manual_reset() 人工 ACK。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CircuitBreakerConfig:
    loss_trip: int = 3            # 连亏 >= 此值熔断
    anomaly_pct: float = 0.08     # 单笔异常亏损 > 此比例熔断
    signal_stall_s: float = 300.0  # 信号源中断超此时长熔断


class CircuitBreaker:
    def __init__(self, cfg: CircuitBreakerConfig = None):
        self.cfg = cfg or CircuitBreakerConfig()
        self.tripped = False
        self.reason = ""
        self.loss_streak = 0
        self.last_signal_ts = 0.0
        self.trip_history: list = []

    def on_trade_close(self, pnl_pct: float) -> None:
        """每笔平仓后更新连亏与异常。"""
        if self.tripped:
            return
        self.loss_streak = self.loss_streak + 1 if pnl_pct < 0 else 0
        if self.loss_streak >= self.cfg.loss_trip:
            self._trip("loss_streak>=%d" % self.cfg.loss_trip)
        elif pnl_pct <= -self.cfg.anomaly_pct:
            self._trip("anomaly_loss>=%.1f%%" % (self.cfg.anomaly_pct * 100))

    def feed_signal(self, ts: float) -> None:
        self.last_signal_ts = ts

    def check_stall(self, now: float) -> None:
        if not self.tripped and self.last_signal_ts > 0:
            if now - self.last_signal_ts > self.cfg.signal_stall_s:
                self._trip("signal_stall>%.0fs" % self.cfg.signal_stall_s)

    def _trip(self, r: str) -> None:
        self.tripped = True
        self.reason = r
        self.trip_history.append(r)

    def manual_reset(self) -> None:
        """Fail-closed：恢复必须人工 ACK。"""
        self.tripped = False
        self.reason = ""
        self.loss_streak = 0


# ===== 生产系统优点吸收（P0-D）：动态熔断阈值 =====
def dynamic_threshold(atr_pct: float) -> float:
    """动态熔断阈值（atr_pct=ATR占价百分比，calc_atr 返回，常态1-4%）。

    - 低波动(<0.4%): 3% (收紧)
    - 正常波动(0.4-5%): 4%
    - 高波动(>5%): 6% (放宽)
    """
    if atr_pct < 0.4:
        return 0.03
    if atr_pct > 5.0:
        return 0.06
    return 0.04


def check_price_circuit(current_price: float, last_price: float,
                        atr_pct: float = 2.0) -> (bool, str):
    """BTC 暴跌熔断检查（吸收服务器 check_circuit_breaker 核心）。

    用动态阈值比对当前价相对上一价的跌幅。返回 (是否熔断, 原因)。
    """
    if last_price <= 0:
        return False, ""
    drop = (last_price - current_price) / last_price
    th = dynamic_threshold(atr_pct)
    if drop >= th:
        return True, f"BTC 暴跌 {drop*100:.2f}% (阈值{th*100:.0f}%, ${last_price:.0f}→${current_price:.0f})"
    return False, ""
