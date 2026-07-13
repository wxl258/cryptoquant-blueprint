"""OKX / GateIO 测试网适配桩。

Binance 测试网已实现于 binance_testnet.py（真实 REST，接 Key 即跑假钱）。
OKX/GateIO 签名与端点不同，此处仅预留骨架，待接 Key 后填充。
关键差异：OKX 签名串为 timestamp+method+path+body 预哈希；GateIO 限频较紧(~10/s)。
"""
from __future__ import annotations

from ..models import Order, Position
from .base import ExchangeAdapter


class _TestnetStubBase(ExchangeAdapter):
    venue = "testnet"

    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = "",
                 testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet

    def submit(self, order: Order) -> Order:
        raise NotImplementedError(f"[{self.venue}] 测试网 submit 未实现（需 API Key + 网络）")

    def cancel(self, coid: str) -> bool:
        raise NotImplementedError(f"[{self.venue}] 测试网 cancel 未实现")

    def query_open(self):
        raise NotImplementedError(f"[{self.venue}] 测试网 query_open 未实现")

    def query_position(self, symbol: str) -> Position:
        raise NotImplementedError(f"[{self.venue}] 测试网 query_position 未实现")

    def query_positions(self):
        raise NotImplementedError(f"[{self.venue}] 测试网 query_positions 未实现")

    def simulate_market(self, prices: dict) -> None:
        raise NotImplementedError(f"[{self.venue}] 测试网无 simulate_market（由真实 WS 行情驱动）")


class OkxDemoAdapter(_TestnetStubBase):
    venue = "okx-demo"


class GateioTestnetAdapter(_TestnetStubBase):
    venue = "gateio-testnet"
