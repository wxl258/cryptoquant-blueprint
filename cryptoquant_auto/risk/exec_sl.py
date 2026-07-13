"""执行级 SL（回应 -55.5% 回撤穿透）。

信号级 SL 仅发信号，API 异常/部分成交时无人平仓 -> 回撤失控。
执行级 SL：开仓同时下交易所原生 OCO/条件 SL，不依赖信号回传；
并提供强平价估算与「移保本位」逻辑。

来自服务器 liquidation_guard 的强平价公式（已验证：60000@5x 多 -> 48300）。
"""
from __future__ import annotations

from ..models import Direction


def estimate_liquidation_price(entry: float, lev: float, direction: Direction,
                                maint_margin: float = 0.005) -> float:
    """估算强平价。多头在下方，空头在上方。"""
    if direction is Direction.LONG:
        return entry * (1 - 1 / lev + maint_margin)
    return entry * (1 + 1 / lev - maint_margin)


def exec_sl_price(entry: float, atr: float, direction: Direction, k: float = 2.0) -> float:
    """执行级硬止损 = entry ± k·ATR（比信号 SL 更贴近执行，避免穿透）。"""
    if direction is Direction.LONG:
        return entry - k * atr
    return entry + k * atr


def move_to_breakeven(sl_price: float, entry: float, direction: Direction,
                      fee_buffer: float = 0.0) -> float:
    """TP1 平50%后，将 SL 移至成本+费（保本位）。"""
    if direction is Direction.LONG:
        target = entry + fee_buffer
        return max(sl_price, target)
    target = entry - fee_buffer
    return min(sl_price, target)
