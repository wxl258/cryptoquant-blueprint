"""CryptoQuant 自动化执行原型（capital-free 调试版）。

仅用于 Shadow / Paper / 测试网 调试，不涉及任何实盘资金。
架构：cron 信号生成保持不动，新增常驻 ExecutionEngine 守护进程接管
「信号 -> 风险闸门 -> 下单 -> 成交 -> 对账 -> 状态回写」闭环。
"""
