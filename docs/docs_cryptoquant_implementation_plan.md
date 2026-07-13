# CryptoQuant 原型 · 详细落地计划圆桌（实施版）

> 日期：2026-07-12
> 依据：蓝图 v0.3 + 实际代码盘点（`cryptoquant_auto/` 包，已通读）
> 目标：把蓝图逐阶段拆成任务级实施计划，明确"怎么实施 / 需要什么 / 补全什么"
> 范围：仅限原型沙盒，不接入实盘资金

---

## 一、当前原型家底盘点（已有可复用）

| 模块 | 现状 | 可直接复用点 |
|------|------|-------------|
| `signals/engine.py` | 规则引擎 10 维评分，P0a/P0b/P1-2 已落地 | `gen_signal` / `SignalCandidate.conds`（归因源） |
| `signals/factor_combiner.py` | ridge/IC 权重拟合 | 评估框架，GP 可在其上扩展 |
| `sim/walk_forward.py` | WFA 框架（148 行） | **需确认是否含 Purged+Embargo**（大概率缺） |
| `sim/backtest.py` `sim/metrics.py` | 回测 + 指标 | DSR/PSR 加到 metrics |
| `risk/signal_filter.py` | quality_gate（124 行） | 挂不确定性维度 |
| `risk/gate.py` `kill_switch.py` `circuit_breaker.py` | 四闸门/熔断 | 宪法接入点 |
| `meta/cognition.py` | 规则式市场状态（122 行） | 阶段3 升级为 FinMem |
| `meta/reflection.py` | 反思日志（155 行） | 阶段3 升级为分层记忆 |
| `adapters/mock.py` | Mock 交易所 | `mock_llm.py` 仿写 |
| `history.py` + `history_cache.json` + `deriv_data.json` | 5.5 年 K线+fr+OI | 因果发现/TSFM/验证 数据源 |
| `run_wfa_v2.py` `run_p0_backtest*.py` `run_ridge_wfa.py` | 回测脚本 | 验证基建母脚本 |
| `core/metacontroller.py` `risk/constitution.py` | **我新建未运行** | 阶段1 基础，待跑通+改 SPCI |
| `demo_blueprint_stage1.py` | 我新建未运行 | 阶段1 验证 demo |

---

## 二、缺口清单（蓝图各层缺什么）

| 蓝图层 | 缺什么 | 复用基础 | 补全量级 |
|--------|--------|---------|---------|
| 阶段0 维度分歧 | SignalCandidate 无结构化 uncertainty 字段 | conds 字符串可解析 | 小 |
| 阶段0 conds 归因 | 无跨交易聚合器 | backtest/reflection | 小 |
| 阶段0 RAG-Mock | `adapters/mock_llm.py` 不存在 | mock.py | 小 |
| 阶段0.5 Purged+Embargo | walk_forward 大概率无 | walk_forward.py | 中 |
| 阶段0.5 DSR/PSR | metrics 无 | metrics.py | 中 |
| 阶段0.5 SPCI | 完全缺失 | 新建 risk/conformal.py | 中 |
| 阶段0.5 受控 A/B | 无 harness | run_wfa_v2 改造 | 中 |
| 阶段1 宪法集成 | constitution 未接 engine | 我建的 constitution.py | 小（接线） |
| 阶段1 metacontroller→SPCI | 现用熵，需改 SPCI | 我建的 metacontroller.py | 中 |
| 阶段2 因果发现 | 完全缺失 | 新建 signals/causal.py | 大 |
| 阶段2 GP/NSGA-II | 仅 ridge，无进化 | factor_combiner 扩展 | 大 |
| 阶段2 StockSim 对手 | sim 无订单级 | sim 扩展 | 大 |
| 阶段3 FinMem 记忆 | 仅 reflection | meta/memory.py 新 | 大 |
| 阶段3 Pydantic schema | 无结构化接口 | mock_llm 改造 | 中 |
| 阶段3 4 角色智能体 | cognition 单规则 | meta 重构 | 大 |
| 阶段4 TSFM 骨架 | 无预训练 | 引 torch/pretrained（后移） | 极大 |
| 阶段4 CVaR 目标 | DRL 无 | sim 扩展 | 大 |
| 阶段4 StockSim LLM 市场 | 无 | 阶段2 对手升级 | 大 |

---

## 三、专家圆桌 · 逐阶段详细实施

### 专家D（逆向稳健）· 阶段0 + 0.5 + 1

**阶段0（零成本复用，1–2 周）**
1. 维度分歧不确定性：给 `SignalCandidate` 加 `uncertainty` 字段，由 `gen_signal` 的 10 维加减分分歧度算出；`signal_filter.QualityGate` 加"高分歧→降权"。落点：`signals/engine.py` + `risk/signal_filter.py`
2. conds 归因：新增 `meta/attribution.py`，聚合回测序列的 conds 得因子贡献榜。落点：新文件 + 接 `backtest.py`
3. RAG-Mock：仿 `mock.py` 写 `adapters/mock_llm.py`，按 regime 检索 `reflection_log.json` 生成 rationale。落点：新文件
4. 假设回放：写 `run_hypothesis_replay.py`，取亏损交易→翻参数→`run_wfa_v2` 回放。落点：新脚本

**阶段0.5（验证基建，2–3 周，建议与阶段1 并行甚至先行）**
5. Purged+Embargo K-Fold：改 `sim/walk_forward.py`，加 purge/embargo 间隔。落点：`sim/walk_forward.py`
6. DSR/PSR+BH-FDR(N)：加 `sim/metrics.py` 的 `deflated_sharpe()`，N=试验次数（GP 代数×种群）。落点：`sim/metrics.py`
7. SPCI 序列共形：新 `risk/conformal.py`，残差时依赖自适应分位。落点：新文件
8. 受控 A/B harness：改 `run_wfa_v2.py` 支持 LLM vs 规则同流对比 + DSR 显著。落点：`run_wfa_v2.py`

**阶段1（Conformal + 宪法脊柱，1–2 周）**
9. 跑通并接线：运行我建的 `constitution.py` + `metacontroller.py`（先验证能跑），把 metacontroller 的"熵"换成 SPCI（第7条）。
10. 集成进 `core/engine.py` 的 `ingest_signal`：quality_gate 之后接 metacontroller 融合 → constitution 校验 → 动作。落点：`core/engine.py`
11. 验证：用 `demo_blueprint_stage1.py` 扩成 4 案例 + 一致性测试。

### 专家C（宏观进化）· 阶段2

**阶段2（因果 + GP/NSGA-II + StockSim，4–6 周）**
12. 因果发现：新 `signals/causal.py`，用 PCMCI/NOTEARS 类做稳定特征选择 + 不变因果结构，输出"因果特征白名单"。落点：新文件
13. GP/NSGA-II：在 `factor_combiner.py` 上扩展，GP 进化规则树结构 + NSGA-II 三目标（收益/回撤/换手）Pareto。落点：`factor_combiner.py` 扩展
14. StockSim 订单级对手：在 `sim/` 加订单级撮合 + LLM 人造市场智能体，复现程式化事实（肥尾/波动聚集）。落点：`sim/` 扩展
15. 验证：因果白名单 + GP 解须过 阶段0.5（Purged+Embargo + DSR）。

### 专家B（情绪LLM）· 阶段3

**阶段3（FinMem + 接地 LLM + 4 角色，4–6 周，需测试网/API 时引 LLM）**
16. FinMem 分层记忆：新 `meta/memory.py`（工作/短期/长期三层 + Profile + 反思自改进），替代裸 reflection。落点：新文件
17. Pydantic schema + Function Calling：把 `mock_llm.py` 升级为严格 JSON schema（market_state/confidence/rationale[]/proposed_action），LLM 只填表不产文本。落点：`adapters/mock_llm.py`
18. 4 角色智能体：分析→研究辩论→决策→风控，复用 v26 的 3专家+逆向 架构。落点：`meta/` 重构
19. 验证：受控 A/B（第8条）证明 LLM 比规则显著，否则回退规则。

### 专家A（盘面技术）· 阶段4

**阶段4（TSFM + CVaR + StockSim，6–10 周，重算力后移）**
20. TSFM 骨架：引 torch + 预训练 Time-MoE/Moirai（零样本），原型期用小蒸馏版。落点：新 `signals/tsfm.py`
21. CVaR 目标：DRL 目标改 CVaR-约束 + Sharpe（RiskawareTrader）。落点：`sim/` 扩展
22. StockSim LLM 市场：阶段2 对手升级为 LLM 驱动。落点：`sim/` 扩展
23. 验证：过 DSR + 宪法 + SPCI 覆盖。

---

## 四、资源 / 依赖需求清单

| 资源 | 是否需要 | 阶段 | 备注 |
|------|---------|------|------|
| numpy / stdlib | ✅ 已有 | 全部 | 阶段0–1 零依赖 |
| `history_cache.json` / `deriv_data.json` | ✅ 已有 | 0.5/2/4 | 5.5 年数据够因果/验证 |
| torch（GPU） | ⏸ 后移 | 3/4 | TSFM/LLM 仿真才需；原型期用小蒸馏/API |
| LLM API Key | ⏸ 测试网期 | 3 | 原型期用 mock_llm 零成本 |
| 预训练 TSFM 权重 | ⏸ 后移 | 4 | Time-MoE/Moirai 开源权重 |
| 计算：CPU 够 | ✅ | 0–2 | 重算力仅阶段3–4 |
| 人工审阅工时 | ✅ 持续 | 每阶段 | 宪法/DSR 需人判门禁 |

> 关键：阶段0–2 全程 numpy/stdlib + 已有数据即可跑，不引任何重依赖，保持原型可移植。torch/LLM 留到阶段3–4 接测试网/云。

---

## 五、风险登记

| 风险 | 概率 | 缓解 |
|------|------|------|
| walk_forward 实际已含部分防泄漏，重复造 | 中 | 先读 walk_forward.py 确认，避免重写 |
| SPCI 实现复杂踩坑 | 中 | 先用朴素在线 CP 跑通，再换 SPCI |
| GP/NSGA-II 搜索爆炸 | 高 | 树深惩罚 + 固定种子 + 计算预算早停 |
| 阶段0.5 与阶段1 谁先：验证层没建就训 ML | 高 | 阶段0.5 必须先于或等于阶段1 进门 |
| 重依赖破坏零依赖定位 | 中 | 可选依赖 + 降级路径（torch 缺失回退 numpy） |
| LLM 成本/延迟不可行 | 中 | 原型全 mock_llm，测试网期再接真 LLM + 缓存 |
| 数据窥视（同一数据既训练又验证） | 高 | Purged+Embargo + DSR(N) 强制隔离 |

---

## 六、第一步行动（建议立即开工）

**就从阶段0.5 验证基建起步**（蓝图定为"不可逾越层"，且阶段0–4 都依赖它）：

1. 先读 `sim/walk_forward.py` 确认是否已有 Purged+Embargo（避免重复造）
2. 实现 `sim/metrics.py` 的 `deflated_sharpe(psr, n_trials)` —— 这是后续所有 edge 验收闸门
3. 跑通我建的 `constitution.py` + `metacontroller.py`（先确认能 import/运行，再改 SPCI）
4. 把 metacontroller 接进 `core/engine.py` 的 `ingest_signal`，跑 `demo_blueprint_stage1.py` 验证四案例

> 这 4 步不引任何新依赖、不碰实盘，纯 numpy，1–2 周可见原型增量，且给后面所有 ML 立起"防骗自己"的基准线。

---

## 七、圆桌决议（实施版）

- 实施顺序：阶段0（零成本）→ **阶段0.5（验证基建，先于一切 ML）** → 阶段1 → 阶段2 → 阶段3 → 阶段4。
- 每阶段进门强制过：Purged+Embargo CV + DSR/PSR + SPCI 覆盖 + 受控 A/B 显著。
- 依赖纪律：阶段0–2 零依赖；torch/LLM 后移，且必须带降级路径。
- 我建的 `constitution.py` / `metacontroller.py` / `demo_blueprint_stage1.py` 是当前唯一可立即跑的代码资产，第一步就是先让它跑通并改 SPCI。
- 本计划为 v1 实施版，随开发可迭代 v2。
