# CryptoQuant 原型 · 完美蓝图（再优化版 v0.3）

> 日期：2026-07-12
> 性质：在 v0.2 基础上，专家再次联网检索 6 个新方向的前沿论文（2024–2026），补全具体方法。
> 范围：仍仅限原型沙盒（MockAdapter / 测试网 / 纸面回测），不接入实盘资金。
> 主线：把"自训小模型"升级为"预训练基座 + 具体算法"，进一步降数据饥渴、提稳健。

---

## 一、本轮新检索（6 方向真实前沿）

| 方向 | 真实前沿 | 来源 |
|------|---------|------|
| 时序基础模型 | Time-MoE（ICLR 2025 Spotlight，MoE decoder-only，零/少样本预报，超 SOTA）；Chronos-2/Moirai-2/TimesFM2.5 同期 | arXiv 2409.16040；TSFM.ai |
| 因果发现 | Causality-Inspired Financial TS（2024）：用因果发现做稳定特征选择，不变因果结构抗 regime 漂移 | arXiv 2408.09960 |
| 风险约束 RL | RiskawareTrader（Springer 2025/26）：DRL 组合优化引入下行风险/CVaR 约束 | arXiv 2511.11481 |
| LLM 记忆/反思 | FinMem（AAAI）：分层记忆（Profile/Memory/Decision）+ 反思自改进；Agent Memory 统一分类（2026） | arXiv 2311.13743；arXiv 2603.07670 |
| 序列共形 | SPCI（Sequential Predictive Conformal Inference）：针对非交换时序，用残差时依赖自适应重估分位 | SPCI 论文 |
| LLM 市场仿真 | StockSim（2025）：订单级仿真平台评 LLM；LLM-MAS 金融市场（中国科学 2026）；LLM 人造市场复现程式化事实 | arXiv 2507.09255；Sci China 2026 |

---

## 二、专家圆桌 · 用新论文再优化

**专家A（盘面技术）· 自训 Transformer → 预训练时序基座**
- v0.2 我提"轻量跨资产注意力"，但 6 币样本小、自训易过拟合。改用 **Time-MoE / Moirai 等预训练时序基础模型**作预报骨架——零样本迁移、数据饥渴骤降，且原生输出**预测区间**（与 Conformal 天然互补）。DRL 目标从纯 Sharpe 改为 **CVaR-约束 + Sharpe**（RiskawareTrader），下行风险优先，更贴"稳健"授权。

**专家C（宏观进化）· 因果层有了具体算法**
- v0.2 的"因果门禁"一直没给算法。现用 **arXiv 2408.09960 的因果发现流程**（PCMCI/NOTEARS 类）做稳定特征选择，且选**不变因果结构**——天然抗 regime 漂移，与 D 的漂移 CP 呼应。GP 染色体只吃因果特征，伪相关从入口掐死。

**专家B（情绪LLM）· RAG-Mock → FinMem 分层记忆**
- v0.2 的"reflection_log"太糙。改用 **FinMem 分层记忆**（工作/短期/长期三层 + Profile 人设定制 + Decision 转化洞察）+ **反思自改进**（Memory survey 五机制之一）。RAG-Mock 升级为检索增强的**分层记忆**，LLM 从记忆而非原始日志推理，反思闭环变结构化的。

**专家D（逆向稳健）· 在线 CP → SPCI 序列共形**
- v0.2 的"在线 CP"对非交换时序只是近似。改用 **SPCI**：利用预测残差的时间依赖自适应重估条件分位，残差作实时反馈——对动作序列的非交换性原生适配，覆盖保证更稳。决策论 CP（2502.02561）仍作"集合→点动作"的映射。

**专家C（续）· 对手盘 → StockSim 订单级 + LLM 人造市场**
- v0.2 的 MARL 对手升级为 **StockSim 订单级仿真**，并引入 **LLM 人造市场智能体**（Sci China 2026）复现程式化事实（肥尾/波动聚集/成交量自相关）。原型策略在"会复现真实统计特征"的市场里练，平移更稳；且仿真本身可 LLM 驱动，闭合 B 与 C 的回路。

**主持收口**：6 项新论文全部落到具体方法。v0.3 相对 v0.2 的核心变化是"自训小模型 → 预训练基座 + 具名算法"。

---

## 三、v0.2 → v0.3 关键升级

| 模块 | v0.2 | v0.3 真实校准后 | 依据 |
|------|------|----------------|------|
| A 预报骨架 | 轻量跨资产注意力（自训） | **预训练时序基础模型**（Time-MoE/Moirai）+ 原生区间 | arXiv 2409.16040 |
| A 目标 | Sharpe + minTRL | **CVaR-约束 + Sharpe**（下行风险优先） | RiskawareTrader(2511.11481) |
| C 因果 | 因果门禁（无算法） | **因果发现稳定特征选择 + 不变因果结构** | arXiv 2408.09960 |
| B 记忆 | reflection_log（粗糙） | **FinMem 分层记忆 + 反思自改进** | arXiv 2311.13743 |
| D 共形 | 在线 CP | **SPCI 序列共形**（非交换适配） | SPCI 论文 |
| C 对手 | MARL（规则+RL） | **StockSim 订单级 + LLM 人造市场** | arXiv 2507.09255；Sci China 2026 |

---

## 四、更新架构（v0.3）

```
┌────────────────────────────────────────────────────────────┐
│              原型沙盒（零资金·离线优先）                       │
│                                                              │
│  ① 预报骨架：预训练 TSFM（Time-MoE/Moirai）零样本 + 原生区间   │
│     目标 = CVaR-约束 + Sharpe（RiskawareTrader）             │
│  ② 因果 + 进化：因果发现稳定特征 → GP/NSGA-II(Pareto)         │
│     对手 = StockSim 订单级 + LLM 人造市场（复现程式化事实）    │
│  ③ 接地 LLM：FinMem 分层记忆 + 反思自改进 + Pydantic schema   │
│  ④ 不确定性：SPCI 序列共形 + 组合层宪法                       │
│                                                              │
│  ── 验证层（不可逾越）──                                       │
│  Purged+Embargo K-Fold → DSR/PSR+BH-FDR(N) → 受控A/B显著     │
│  可观测性：SPCI 覆盖率 / 漂移 / 性能硬开关                     │
│  实盘契约：只读信号 + 人工 ACK（Gate A–F）                     │
└────────────────────────────────────────────────────────────┘
```

> 依赖提示：TSFM / LLM 人造市场需 torch 或 API，留到阶段3–4 接测试网/云再引；原型期用小蒸馏版 / RAG-Mock 验证概念，保持零依赖可跑。

---

## 五、更新落地路径

| 阶段 | 建什么 | 真实依据 | 护栏 |
|------|--------|---------|------|
| 阶段0 | 零成本复用 | v1.1 | quality_gate |
| 阶段0.5 | 验证基建（Purged+Embargo / DSR / SPCI） | de Prado; Bailey2014; SPCI | 强制 |
| 阶段1 | SPCI 序列共形 + 组合层宪法 | SPCI + 决策论(2502.02561) | CP 覆盖 |
| 阶段2 | 因果发现特征 + GP/NSGA-II | 2408.09960; MOO3 | 因果层 + DSR |
| 阶段3 | FinMem 分层记忆 + 接地 LLM 4 角色 | FinMem; Function Calling | 受控 A/B |
| 阶段4 | TSFM 骨架 + CVaR 目标 + StockSim 对手 | Time-MoE; RiskawareTrader; StockSim | DSR + 宪法 |

---

## 六、诚实边界 + 决议

- **没有 edge 圣杯**：v0.3 提升工程纪律与稳健性，不保证更高胜率。所有前沿框架均研究/原型定位。
- **v0.3 相对 v0.2 的最大升级**：从"自训小模型"转向"**预训练基座 + 具名算法**"——Time-MoE 解数据饥渴、RiskawareTrader 解目标脆弱、FinMem 解记忆粗糙、SPCI 解序列非交换、StockSim 解对手不真实、因果发现解特征伪相关。六项都有 2024–2026 实据。
- **依赖现实**：重算力项（TSFM/LLM 仿真）后移，原型期不破坏零依赖。
- 本蓝图为 v0.3，取代 v0.2 作当前最优基线；后续可迭代 v0.4（如需要补具体超参/数据集协议）。
