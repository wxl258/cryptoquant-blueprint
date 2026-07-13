"""交易所适配层抽象基类。

所有写操作以 coid 做幂等锚点；真实实现（测试网/实盘）需在此之上加
签名、限频、断线重连、序列缺口检测。当前 MockAdapter 仅模拟，无网络。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import Order, Position


class ExchangeAdapter(ABC):
    @abstractmethod
    def submit(self, order: Order) -> Order:
        """提交订单；幂等：同 coid 重复提交返回既有订单，不重复下单。"""

    @abstractmethod
    def cancel(self, coid: str) -> bool:
        """撤单。"""

    @abstractmethod
    def query_open(self) -> List[Order]:
        """查询当前挂单。"""

    @abstractmethod
    def query_position(self, symbol: str) -> Optional[Position]:
        """查询某标的持仓。"""

    @abstractmethod
    def query_positions(self) -> List[Position]:
        """查询全部持仓。"""

    @abstractmethod
    def simulate_market(self, prices: dict) -> None:
        """[调试用] 推进_mock行情，触发限价/止损/止盈成交。真实适配器无此方法。"""
