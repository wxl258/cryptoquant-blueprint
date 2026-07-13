# CryptoQuant 阶段3 交接文档 · 专家B 情绪LLM（任务16–19）

> 生成：2026-07-13 · 代码包：`cryptoquant_blueprint_impl/cryptoquant_auto/`
> 进门已重确认阶段0.5 验证层；严守零依赖纪律与实盘硬锁（宪法 R0）。

## 一句话结论

阶段3 四项任务全部落地并跑通：FinMem 分层记忆、Pydantic 风格严格 schema + Function Calling、四角色智能体、受控 A/B 闸门。**五道进门验证全绿**。诚实结论：在真实行情前向样本上，四角色+FinMem 并未显著优于朴素规则（p=0.8560），因此守门逻辑正确判定「回退规则」——这正是验证层「防骗自己」的价值，而非伪造 edge。

> ⚠️ 数值诚信：本表数值由 `python3 -m cryptoquant_auto.run_validation_stage3` 于提交 `726b7d4` 实跑生成。早期交接文档曾误记为 规则 0.603 / 四角色 0.490（PSR 误标为 DSR、且为旧代码态）。A/B 打印值实为 `deflated_sharpe(n_trials=1)` 即 **PSR**，引擎以 `periods_per_year=1`（逐bar、非年化）调用，与 `sim/metrics.py` 默认 8760 不同。数值随代码/数据演进，**以实跑为准**，勿手填。

---

## 进门验证结果（五关全绿 ✅）

| 关卡 | 内容 | 结果 |
|------|------|------|
| ① live_lock | 实盘硬锁：LIVE_CAPITAL_LOCK=False + 宪法 live_capital=True 否决复测 | ✅ 通过 |
| ② stage0_5_reconfirm | 进门重确认：Purge+Embargo / DSR 单调 / SPCI 覆盖 / A-B 跑通 | ✅ 通过 |
| ③ finmem_memory | 记忆闭环：observe→record→outcome→reflect→retrieve + 反思自改进 | ✅ 通过 |
| ④ four_roles | 四角色 schema 合法 + 风控对禁易 regime 否决生效 | ✅ 通过 |
| ⑤ ab_gate | 受控 A/B：规则 vs 四角色（同前向隔离 + DSR 对比 + 明确裁决） | ✅ 通过 |

---

## 四项任务落地清单

| 任务 | 落点文件 | 做了什么 | 量级 |
|------|----------|----------|------|
| 16 FinMem 分层记忆 | 新 `meta/memory.py` | 工作/短期/长期三层 + Profile 人设 + 反思自改进（短期聚合→长期洞察并回写 Profile） | 大 |
| 17 严格 schema + Function Calling | 新 `adapters/mock_llm.py` | LLMDecision 严格 schema（market_state/confidence/rationale[]/proposed_action）+ OpenAI 风格 tools 锁表；MockLLM 确定性接地填表并强制校验 | 中 |
| 18 四角色智能体 | 新 `meta/agents.py` | 分析→研究辩论→决策(LLM 填表)→风控；复用 v26 三专家+逆向架构 | 大 |
| 19 受控 A/B 验证 | 新 `run_validation_stage3.py` + `demo_blueprint_stage3.py` | 规则 vs 四角色同前向对比，不显著则回退规则 | 中 |

---

## FinMem 分层记忆（任务16）

四层结构，全部纯 json 持久化到 `data/finmem_*.json`：

- **Profile（长期人设）**：风险偏好、禁易 regime、置信上下限；反思闭环会回写。
- **Working（工作记忆）**：当前 tick 观测槽，容量 FIFO，瞬时。
- **Short-Term（短期情景）**：近期决策事件含回填 outcome，容量 FIFO。
- **Long-Term（长期洞察）**：反思从短期聚合出的可检索结论，带信念权重。

**反思自改进（self-improvement）已验证**：注入 15 笔 CRASH 持续亏损情景后，`reflect()` 自动把 `CRASH` 写入 `Profile.forbidden_regimes`，并在后续四角色决策中由风控层强制转 HOLD。这就是「记忆驱动行为改变」。

---

## 四角色智能体（任务17/18）

决策管线（接地 LLM，LLM 只填表不产文本）：

1. **Analyst（分析）**：读 9 维特征 + regime → 市场态 + 驱动 + 初步信念。
2. **Researcher（研究辩论）**：从 FinMem 检索该 regime 长期洞察，产支持/反对论据并调置信。
3. **DecisionMaker（决策）**：融合方向 + 上下文，调 MockLLM 填严格 schema 表（Function Calling 锁表）。
4. **RiskController（风控）**：宪法式硬锁——禁易 regime 转 HOLD、低置信软降级、SPCI 高惊喜度转 HOLD、置信钳制。

**schema 守门已验证**：越界字段（如 market_state="XXX"、confidence=2.0、proposed_action="FLY"）会被 `validate()` 直接拒，模拟 pydantic 校验失败。

---

## 受控 A/B 结果（任务19 · 诚实）

在同一真实前向样本（624 根 × 9 特征，含 18 根 CRASH / 142 TREND / 464 RANGE）上对比：

| 策略 | 信号数 | PSR(N)* | OOS 盈利窗 | 隔离剪 bar |
|------|--------|--------|-----------|-----------|
| 规则（动量逐根） | 623 | 0.968 | 40% | purge=35 embargo=25 |
| 四角色+FinMem | 419 | 0.913 | 50% | purge=20 embargo=15 |

> \* 口径：A/B 打印值由 `deflated_sharpe(n_trials=1)` 计算（即 PSR，非 DSR）；引擎以 `periods_per_year=1`（逐bar、非年化）调用，与 `sim/metrics.py` 默认 8760 不同。数值于提交 `726b7d4` 实跑生成，随代码/数据演进，以实跑为准。

- ΔPSR = −0.055，p = 0.8560，**不显著**，胜方=规则。
- 四角色风控/记忆过滤掉 205 笔（多为 RANGE 低质量），但本样本上未带来更优 edge。
- **裁决：🔒 回退规则**（A/B 未证明 LLM 显著更优，守门逻辑成立，未伪造 edge）。

> 说明：本阶段 A/B 的「规则基线」取朴素动量逐根，四角色在其上叠加记忆/风险门控。两者同根，差异来自门控质量。结论诚实——原型无 edge 圣杯，验证层价值在「防骗自己」。

---

## 铁律遵守确认

- ✅ **零依赖纪律**：阶段3 全部纯 numpy + stdlib；未引 torch/LLM/pydantic。MockLLM 为确定性接地 mock，真实 LLM 留待阶段3-4 测试网/云，且已留降级钩子（接真实 LLM 须复用 `tool_spec()` 锁表）。
- ✅ **进门先过验证层**：② 重确认阶段0.5 四关（Purge+Embargo / DSR / SPCI / A-B）再绿才进门。
- ✅ **实盘硬锁**：`LIVE_CAPITAL_LOCK=False` 启动即断言 + 宪法 `live_capital=True` 复测否决，全程沙盒。

---

## 复现命令

```bash
cd cryptoquant_blueprint_impl
python3 -m cryptoquant_auto.run_validation_stage3      # 阶段3 五关验证
python3 -m cryptoquant_auto.demo_blueprint_stage3      # 演示入口
python3 -m cryptoquant_auto.run_validation_0_5         # 阶段0.5 回归（仍全绿）
```

---

## 下一步（阶段4 · 专家A · 盘面技术）

| 任务 | 内容 | 落点 |
|------|------|------|
| 20 TSFM 骨架 | 引 torch + 预训练 Time-MoE/Moirai（零样本），小蒸馏版；须带 torch 缺失降级 numpy | 新 `signals/tsfm.py` |
| 21 CVaR 目标 | DRL 目标改 CVaR-约束 + Sharpe（RiskawareTrader） | `sim/` 扩展 |
| 22 StockSim LLM 市场 | 阶段2 对手升级为 LLM 驱动 | `sim/` 扩展 |
| 23 验证 | 过 DSR + 宪法 + SPCI 覆盖 | — |

阶段4 将首次真正引入重算力（torch），届时**必须**保留降级路径与实盘硬锁，进门仍须过阶段0.5。

---

## 本次新增/改动文件

- 新增：`cryptoquant_auto/meta/memory.py`（FinMem）
- 新增：`cryptoquant_auto/adapters/mock_llm.py`（严格 schema + Function Calling）
- 新增：`cryptoquant_auto/meta/agents.py`（四角色）
- 新增：`cryptoquant_auto/run_validation_stage3.py` + `demo_blueprint_stage3.py`
- 改动：`cryptoquant_auto/meta/__init__.py`、`cryptoquant_auto/adapters/__init__.py`（导出新模块）
- 向后兼容：`reflection.py` 的 `ReflectionLog` 由 `memory.py` 薄壳重导出，老调用方不受影响
- 交付：`/workspace/STAGE3_HANDOFF.md`、`/workspace/stage3_validation_log.txt`
