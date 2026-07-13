# CryptoQuant 自动化执行原型（capital-free 调试版）

> **零密钥、零资金。** 仅用于 Shadow / Paper / 交易所测试网 调试。
> 不涉及任何实盘资金；实盘资金开关由 Go/No-Go Gate A–F 锁定（见 `cryptoquant_auto_plan_20260710.md`）。

## 定位

把系统从「信号生成 + 企微通知」升级为「交易所 API 自动下单/平仓」的**完整调试沙盒**。
当前用 `MockAdapter` 模拟交易所（可模拟成交、注入故障），待提供测试网 API Key 后，
把 `MockAdapter` 换成 `adapters/binance_testnet.py`（Binance 已完整实现）或 `testnet_stub.py`（OKX/GateIO）即可跑真实 API 假钱。

## 运行

```bash
cd /workspace
python -m cryptoquant_auto.demo      # 端到端验证（9 单元用例 + 回测 + 测试网就绪）
```

## 目录

```
cryptoquant_auto/
  models.py                信号/订单/成交/持仓 数据模型（含 post_only maker 标记）
  adapters/
    base.py                适配层抽象基类（写操作以 coid 幂等）
    mock.py                Mock 交易所（模拟成交 + 故障注入）
    binance_testnet.py     Binance 测试网真实 REST 适配器（HMAC 签名，接Key跑假钱）
    testnet_stub.py        OKX / GateIO 测试网桩（签名/端点待填）
  risk/
    gate.py                assert_pre_trade 四闸门（regime/beta/single/thermo）
    kill_switch.py         分级 KillSwitch（L1暂停新开/L2降险/L3生存态）
    exec_sl.py             执行级 SL + 强平价估算 + 移保本位
    exec_cost.py           执行成本模型（maker/taker/滑点/资金费 + 各币edge）
  core/
    order_builder.py       信号 -> 订单（入场单 + TP/SL 子单，确定性 coid，maker 模式）
    reconcile.py           期望持仓 vs 实际持仓 对账防超仓
    router.py              跨所 Fallback 路由（Binance→OKX→GateIO）
    engine.py              ExecutionEngine 编排闭环（按 signal_id 隔离，防跨信号碰撞）
  sim/
    market_path.py         各币行情路径生成（回测用，按波动校准）
    metrics.py             回测指标（胜率/净值/夏普/回撤/各币edge）
    backtest.py            Paper/回测验证器（驱动完整管线 + 成本建模，出胜率）
  demo.py                  端到端调试验证（9 单元用例 + 回测 + 测试网就绪）
```

## 已验证能力（见 demo 输出）

- 信号 → 四闸门 → 下单 → 成交 → 对账 整条管线（9/9 单元用例）
- **订单幂等**：同 `coid` 重发不重复下单（重启/网络抖动安全）
- **瞬时超时**：视为未知，查单/重发恢复，不盲重复
- **维护态断路**：提交被拒不崩溃
- **状态机**：入场 → TP1 平50% → 移 SL 保本位 → TP2 全平
- **执行级 SL 止血**：单币亏损限制在 ~0.1% 权益，回应 -55.5% 穿透
- **KillSwitch L1**：当日回撤触发后禁止新开仓
- **对账防超仓**：期望 vs 实际差异标记 OVER/UNDER
- **跨所 Fallback 路由**：主所维护时自动切备份所，不丢单
- **回测验证**：117 笔样本，胜率 57.3%，直接出净值/夏普/回撤/各币 edge，maker 优于 taker
- **测试网就绪**：Binance 测试网适配器已就绪（待填 Key 即跑真实 API 假钱）

## 如何验证「胜率之类」（你提的点）

- **回测**：`PaperBacktest` 把信号序列灌入完整执行管线（含成本），统计胜率/净值/夏普/
  回撤/各币 edge，并校验 Gate B 成本敏感度。运行 `python -m cryptoquant_auto.demo`
  看「回测验证」段落。真实回测只需把 `make_random_signals` 换成历史信号。
- **连接测试网测试**：把 `MockAdapter` 换成 `BinanceTestnetAdapter(api_key, api_secret)`
  （零真金白银），即可跑真实 API 验证成交与胜率。

## 已知边界（待接测试网/实盘时补）

- OKX / GateIO 测试网适配器的真实签名与端点（Binance 已完整）
- 真实 REST/WebSocket 限频、断线重连、序列缺口检测、user data stream 真实成交回写
- β 聚合与强平价防御的实时保证金率监控（逻辑已在 kill_switch / gate 就位）

## 安全责任

- **Fail-closed**：自动只做安全动作（减仓/暂停/拒单）；恢复必须 `engine.manual_resume()` 人工 ACK。
- 翻转实盘资金开关前，须过 Gate A–F 全 AND + 影子验证 ≥6 月 / ≥200 笔。
