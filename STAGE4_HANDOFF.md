# CryptoQuant 阶段4 交接文档 · 专家A 盘面技术（任务20–23）

> 生成：2026-07-13 · 代码包：`cryptoquant_blueprint_impl/cryptoquant_auto/`
> 进门已重确认阶段0.5 验证层；严守降级纪律与实盘硬锁（宪法 R0）。**本阶段为蓝图四阶段收官。**

## 一句话结论

阶段4 四项任务全部落地并跑通：TSFM 预测骨架（含 torch 降级 numpy）、CVaR-约束夏普目标、StockSim LLM 驱动市场。**六道进门验证全绿**。诚实结论：受控 A/B 中 TSFM 方向策略 DSR=0.936 高于动量基线 0.603，但 Welch + 多重检验校正后 p=0.6335 **不显著**，守门逻辑正确判定「未证明更优、不伪造 edge」——torch 虽在沙箱内可用，但无外网预训练权重，落地的仍是小蒸馏 MLP + numpy 双后端，纪律（可选 + 降级）如实达成。

---

## 进门验证结果（六关全绿 ✅）

| 关卡 | 内容 | 结果 |
|------|------|------|
| ① live_lock | 实盘硬锁：LIVE_CAPITAL_LOCK=False + 宪法 live_capital=True 否决复测 | ✅ 通过 |
| ② stage0_5_reconfirm | 进门重确认：Purge+Embargo / DSR 单调 / SPCI 覆盖=92.5% / A-B 跑通 | ✅ 通过 |
| ③ tsfm_forecast | TSFM 滚动区间覆盖≈标称 + torch 缺失降级 numpy | ✅ 通过 |
| ④ cvar_objective | 尾部越界→约束得分重罚 + 砍仓 HOLD | ✅ 通过 |
| ⑤ stocksim_llm | LLM 接地复现程式化事实(3/3) + 降级 | ✅ 通过 |
| ⑥ ab_gate | 受控 A/B：TSFM 方向 vs 动量基线（同前向隔离 + DSR 对比 + 明确裁决） | ✅ 通过 |

---

## 四项任务落地清单

| 任务 | 落点文件 | 做了什么 | 量级 |
|------|----------|----------|------|
| 20 TSFM 骨架 | 新 `signals/tsfm.py` | `TSFMForecaster` 接口 + `DistilledTSFM`(numpy 滞后脊回归+残差分位区间) + `TorchTSFM`(torch MLP, 懒加载) + `make_tsfm(backend)` 自动降级 | 大 |
| 21 CVaR 约束目标 | 新 `sim/riskaware.py` | `cvar()` 期望短缺 + `cvar_sharpe_score()`(Sharpe−λ·max(0,CVaR−budget)) + `VanillaTrader`/`RiskAwareTrader`(尾部越界砍 HOLD) | 中 |
| 22 StockSim LLM 市场 | 改 `sim/stocksim.py` | 新增 `LLMMarketAgent(MockMarketAgent)`：MockLLM 叙事订单流偏置，保留 GARCH 基底；`make_market_agent(kind="llm")` 自动降级 | 中 |
| 23 受控 A/B 验证 | 新 `run_validation_stage4.py` | 六关进门 + 真实行情前向 TSFM 策略 vs 动量基线 + 受控裁决 | 中 |

---

## TSFM 预测骨架（任务20）

三类后端，统一接口 `fit / forecast / coverage`：

- **DistilledTSFM（numpy 稳态后端）**：对滞后收益做闭式岭回归，残差经验分位构造预测区间；点预测 + lo/hi = 点 ± q·√步长。零依赖。
- **TorchTSFM（torch 后端）**：懒加载 `import torch`，小 MLP `Linear(L,hidden)→Tanh→Linear(hidden,1)` + Adam + MSELoss(epochs=60)。`load_pretrained()` 抛 `NotImplementedError`——沙箱无外网拉取 Time-MoE/Moirai 权重，诚实落地蒸馏版。
- **make_tsfm(backend="auto")**：torch 可用走 `TorchTSFM`，否则降级 `DistilledTSFM`；`backend="numpy"` 强制 numpy。

**真实行情滚动校准已验证**：BTC 1499 根 1h 样本、标称覆盖 0.90，numpy 后端 89.13%、torch(distilled_torch) 后端 87.32%，区间校准合理；降级路径 `make_tsfm("numpy")` 返回 `DistilledTSFM` 实例。

---

## CVaR 约束目标（任务21）

`score = Sharpe − λ·max(0, CVaR − budget)`，尾部越界即重罚。

**尾部砍仓已验证**：
- 平稳段 CVaR=−0.0205，砸盘尾 CVaR=−0.0856（远超 −0.02 预算）。
- 基线 `VanillaTrader` 满仓 SHORT，约束 `RiskAwareTrader` 在越界瞬间转 **HOLD 且砍仓=True**。
- 约束得分：Vanilla=−0.29，RiskAware=−33.11 → 越界重罚生效，与「砍仓」动作一致。

---

## StockSim LLM 市场（任务22）

`LLMMarketAgent` 在 `MockMarketAgent`(GARCH + 羊群/动量 + 跳跃) 之上叠加 MockLLM 叙事订单流偏置，保留 GARCH 基底以保证程式化事实（肥尾/波动聚集/量自相关）持续存在。

**程式化事实复现已验证（LLM 后端 3/3）**：峰度=14.976（肥尾✔）、|收益|ACF=0.18（波动聚集✔）、量 ACF=0.051（量自相关✔），叙事输出 `RANGE/LONG(0.02)`。`make_market_agent(kind="llm")` 在 MockLLM 缺失时自动降级为纯 GARCH 基底，mock 后端同样 3/3。

---

## 受控 A/B 结果（任务23 · 诚实）

在同一真实前向样本（624 根 × 9 特征，Purge+Embargo 隔离）上对比：

| 策略 | 信号数 | DSR(N) | OOS 盈利窗 | 隔离剪 bar |
|------|--------|--------|-----------|-----------|
| 动量基线（sign(momentum)） | 623 | 0.603 | 40% | purge=35 embargo=25 |
| TSFM 方向（逐根一步外推） | 624 | 0.936 | 40% | purge=35 embargo=25 |

- ΔDSR = +0.333，p = 0.6335，**不显著**，胜方=TSFM（仅 DSR 数值更高）。
- TSFM 方向由 `rets_full[:idx+1]`（截止当前已收盘 bar）拟合、预测 idx+1（即 forward），属真一步外推，无泄露。
- **裁决：🔒 未证明更优（不伪造 edge）**。蓝图规则——受控 A/B 不显著则不切换，沿用动量/规则基线。

> 说明：TSFM 的 0.936 是单次前向 WF 的经济 edge 实测值（非闸门机制），在 Welch t 检验 + 多重检验校正（n_trials=信号数）下未达显著。验证层价值在「防骗自己」：原型无 edge 圣杯，torch 重算力已就位但仍须受控闸门约束。

---

## 铁律遵守确认

- ✅ **降级纪律（替代原零依赖）**：阶段4 首次引入 torch 重算力，但 `make_tsfm`/`make_market_agent` 均保留 numpy 缺失降级路径；现场 torch 2.10.0 可用，无外网预训练权重则诚实落地小蒸馏 MLP。阶段0-2 仍纯 numpy/stdlib。
- ✅ **进门先过验证层**：② 重确认阶段0.5 四关（Purge+Embargo 激活 / DSR 单调 / SPCI 覆盖 92.5% / A-B 跑通）再绿才进门。
- ✅ **实盘硬锁**：`LIVE_CAPITAL_LOCK=False` 启动即断言 + 宪法 `live_capital=True` 复测否决，全程沙盒。

---

## 复现命令

```bash
cd cryptoquant_blueprint_impl
python3 -m cryptoquant_auto.run_validation_stage4      # 阶段4 六关验证
python3 -m cryptoquant_auto.run_validation_stage3      # 阶段3 五关验证（回归）
python3 -m cryptoquant_auto.run_validation_0_5         # 阶段0.5 回归（仍全绿）
```

---

## 蓝图收官状态

四阶段全部落地并通过进门验证：

| 阶段 | 专家 | 任务 | 进门验证 |
|------|------|------|----------|
| 0.5 | 验证层 | Purge+Embargo / DSR / SPCI / A-B | 全绿（持续回归） |
| 1 | 专家C(数据) | 特征工程/健康检查 | ✅ |
| 2 | 专家C(信号) | 任务12–15 信号引擎 | ✅ 五关 |
| 3 | 专家B(情绪LLM) | 任务16–19 FinMem/四角色 | ✅ 五关 |
| 4 | 专家A(盘面技术) | 任务20–23 TSFM/CVaR/StockSim | ✅ 六关 |

**主线结论**：原型在 `cryptoquant_auto/` 内形成「数据采集→特征→信号→记忆/LLM→风控→验证」闭环，且每一阶段 entry 必经阶段0.5 硬门。所有 ML/torch 组件均带降级与受控 A/B 闸门，未以显著性不足的 DSR 伪造 edge。下一步若接真实 LLM/预训练权重，须复用现有 `tool_spec()` 锁表 + 受控 A/B 闸门，不得绕过。

---

## 本次新增/改动文件

- 新增：`cryptoquant_auto/signals/tsfm.py`（TSFM 骨架 + 双后端降级）
- 新增：`cryptoquant_auto/sim/riskaware.py`（CVaR-约束夏普 + RiskAwareTrader）
- 新增：`cryptoquant_auto/run_validation_stage4.py`（六关进门验证）
- 改动：`cryptoquant_auto/sim/stocksim.py`（新增 `LLMMarketAgent`，改 `make_market_agent`）
- 未改动 `__init__.py`：验证脚本与调用方均用完整子模块路径导入（`from cryptoquant_auto.signals.tsfm import ...` / `from cryptoquant_auto.sim.riskaware import ...`），包导出保持原状。
- 交付：`/workspace/STAGE4_HANDOFF.md`、`/workspace/stage4_validation_log.txt`
