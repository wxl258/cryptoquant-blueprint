"""核心数据模型：信号、订单、成交、持仓。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class OrderType(str, Enum):
    ENTRY = "ENTRY"   # 限价入场
    TP = "TP"         # 止盈限价
    SL = "SL"         # 止损（条件/市价）
    REDUCE = "REDUCE" # 生存态减仓（L3：降仓不 Futian 全平）


@dataclass
class Signal:
    """与系统 signals.json 字段对齐的信号。"""
    symbol: str
    tf: str                 # 1H / 4H / 1D / W
    direction: Direction
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    confidence: float       # 0~1
    signal_id: str
    atr: float = 0.0


@dataclass
class Order:
    """一笔订单；coid 为幂等锚点（symbol_signalid_leg）。"""
    coid: str
    symbol: str
    side: str               # BUY / SELL
    otype: OrderType
    price: float
    qty: float
    signal_id: str
    leg: str = "entry"          # entry / tp1 / tp2 / sl
    parent_coid: Optional[str] = None
    post_only: bool = False      # maker 模式：仅挂单，不吃 taker
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_price: float = 0.0

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED


@dataclass
class Fill:
    coid: str
    symbol: str
    side: str
    price: float
    qty: float
    ts: float


@dataclass
class Position:
    symbol: str
    direction: Direction
    entry_price: float
    qty: float                # 剩余持仓
    initial_qty: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    entry_coid: str
    state: str = "OPEN"       # OPEN -> TP1_FILLED -> BREAKEVEN -> CLOSED
    liq_price: Optional[float] = None  # 强平价；≤1x 杠杆(约1x无借)为 None（无交易所强平，风险由 SL 控）
    leverage: float = 1.0     # 实际杠杆 = 名义/权益（P2-2：由订单名义推导，取代硬编码 lev=5.0）
    realized_pnl: float = 0.0
    signal_id: str = ""       # 关联信号 id（用于跨信号隔离）
    _last_realized: float = 0.0   # 引擎记账用：上次已计入的已实现盈亏
