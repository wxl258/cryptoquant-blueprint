# CryptoQuant 蓝图实施 · 交付说明

> 阶段 0.5「防骗自己」验证层 ＋ 阶段 1 决策脊柱
> 交付日期：2026-07-13 · 状态：✅ 全链路跑通 · 安全级别：零资金 / 纯沙盒 / fail-closed

## 一句话结论

阶段 0.5 四道安检闸门（防泄漏 / 多重检验校正 / 自适应不确定度 / 受控 A/B）与阶段 1 决策脊柱（元控制器 → 宪法 → 执行引擎）已全部落地、接线并通过自检。原型严格 fail-closed，绝不碰真钱。

---

## 一、本次交付了什么

两层结构，互相独立可验证：

- **阶段 0.5 验证层（不可逾越层）**：给系统装上四道「防骗自己」的验收闸门，专门治理量化最常见的三类自欺——数据泄漏、把运气当能力（多重检验假阳性）、不确定度估错就莽。
- **阶段 1 决策脊柱**：把多专家意见融合 → 宪法硬约束校验 → 执行引擎落单，串成一条可运行的决策链，并内置「实盘硬锁」。

---

## 二、怎么验收（跑一遍就知道）

工作区在 `cryptoquant_blueprint_impl/`，零外部依赖（仅 numpy / 标准库），沙盒内直接可执行：

1. 全套自检：`python3.11 cryptoquant_auto/run_validation_0_5.py`
2. 四案例演示（**必须带 `-m` 模块方式**，否则相对导入报错）：`python3.11 -m cryptoquant_auto.demo_blueprint_stage1`

两者预期：五道关卡全绿、四案例全过。任何一项翻红即说明环境或代码被改动。

---

## 三、验证结果（本轮实测）

| 关卡 | 验证内容 | 实测结果 | 判定 |
|------|----------|----------|------|
| ① Purged+Embargo | 防 IS/OOS 数据泄漏 | 净化侧剪掉 bar：purge=12 / embargo=6，机制确认在干活；无泄漏玩具数据 ΔDSR=0（预期非假阳性） | ✅ |
| ② DSR(N)/PSR/BH-FDR | 多重检验校正 | 单 bar SR=0.15 时 DSR(N=1)=0.81 → DSR(N=50)=0.11，N 越大越保守 | ✅ |
| ③ SPCI 序列共形 | 时依赖自适应不确定度 | 同分布覆盖率 92.5%（命中目标≈90%）；异常惊喜度 3.29 >> 正常 0.00；naive 降级 88% | ✅ |
| ④ 受控 A/B | LLM vs 规则同流对比 | 规则 SR=0.19 / LLM SR=0.51，差异 p≈0 显著，winner=llm，建议放行 | ✅ |
| ⑤ 阶段1 脊柱 | metacontroller→宪法→引擎 | 融合建仓通过；SPCI 在线样本=6 已切换惊喜度口径；live_capital=True 硬锁否决一切 | ✅ |

> BH-FDR(α=0.10) 附带演示：10 个假设中 5 个显著，校正后 p 值列表已正确收紧。

---

## 四、本轮修掉的关键 bug（重点）

这些 bug 的共同特征是「静默」——代码不报错、演示看起来在跑，但验收闸门实际上形同虚设。不修就等于没有这套防线。

| # | 问题 | 后果 | 修法 |
|---|------|------|------|
| 1 | A/B 的 `periods_per_year` 没传给 DSR/PSR，偷偷用了默认 8760 倍年化 | 单 bar 收益被放大成 SR=14/42，DSR 饱和在 1.0，`winner` 因两边都=1.0 误判成「平局」 | `controlled_ab` 加 `periods_per_year` 参数并向下传递，统一 `periods_per_year=1` |
| 2 | DSR(N) 演示同样缺 `periods_per_year=1` | N=1 与 N=50 都显示 1.0，看不出「N 越大越保守」的核心价值 | 验证脚本显式传 `periods_per_year=1`，呈现 0.81→0.11 衰减 |
| 3 | SPCI 的 decay 权重写反（越旧权重越高），与文档「近因权重更高」相反 | 区间被早期随机样本带窄，且语义错误 | 改为按「距现在位置」算权重，最新样本权重最高 |
| 4 | ① 号隔离层是空操作：`embargo=0.02` 在 fold=30 时 `int()` 直接截成 0，却谎称「机制激活」 | purge/embargo 没剪任何 bar，防泄漏纯摆设 | 调大比例 + 新增「被剪 bar 数」仪表，用真实剪裁量证明隔离层生效；措辞改诚实 |
| 5 | SPCI 覆盖率虚标 90% 实际 74%：训练样本仅 60 个太少，经验分位偏窄 | 演示与声明不符，易被抓包 | 训练样本提到 200，收敛到 92.5%，与「目标≈90%」吻合 |

---

## 五、文件清单

| 文件 | 状态 | 职责 |
|------|------|------|
| `cryptoquant_auto/risk/conformal.py` | 🆕 新建 | SPCI 序列共形预测器（自适应不确定度 + naive 降级路径） |
| `cryptoquant_auto/risk/constitution.py` | 📋 复制自 handoff | 交易宪法：R0 实盘硬锁 / R1 fail-closed / R2 最大回撤 / R3 方向中立 |
| `cryptoquant_auto/core/metacontroller.py` | ✏️ 修改 | 贝叶斯元控制器：接入 SPCI 作不确定度来源，>5 样本切换惊喜度口径 |
| `cryptoquant_auto/core/engine.py` | ✏️ 修改 | 执行引擎：`ingest_meta` 把元决策落成 v8 Signal，过宪法后建仓 |
| `cryptoquant_auto/signals/engine.py` | ✏️ 修改 | 信号引擎：补齐对称「做多三条件网关」，消除结构偏多（R3 要求） |
| `cryptoquant_auto/sim/metrics.py` | ✏️ 修改 | PSR / DSR(N) / BH-FDR 多重检验校正，纯 numpy |
| `cryptoquant_auto/sim/walk_forward.py` | ✏️ 修改 | Purged+Embargo 滚动 WF + 被剪 bar 计数仪表 |
| `cryptoquant_auto/sim/ab_harness.py` | 🆕 新建 | 受控 A/B：Welch t 检验 + DSR 显著 + 放行建议 |
| `cryptoquant_auto/run_validation_0_5.py` | 🆕 新建 | 五关卡自检脚本（本交付的验收入口） |
| `cryptoquant_auto/demo_blueprint_stage1.py` | ✏️ 修改 | 四案例演示（OHLC 修复 + 案例4 镜像上下文修复） |

> 注：handoff 原始 README 称 v9 包内含上述 3 个新文件，实测仅在 `code/` 目录，已手动集成进包正确路径。

---

## 六、安全约束（红线，不可逾越）

- **R0 实盘硬锁**：`live_capital=True` 时宪法否决一切动作，原型仅运行于沙盒。
- **fail-closed**：缺少理由 / 缺少置信度 → 直接拒单；不确定度超过阈值 → 软降级为观望，绝不裸奔。
- **R3 方向中立**：多空后验在镜像市场上下文下严格对称，系统不会偷偷偏多或偏空。
- **零资金**：本工作区所有回测 / 验证均为合成或历史数据，无下单、无密钥、无真实账户接入。

---

## 七、下一步建议（按蓝图节奏）

| 阶段 | 内容 | 当前状态 |
|------|------|----------|
| 阶段 2 | 接入真实行情数据，替换合成收益流 | 待启动 |
| 阶段 3 | torch / LLM 融合（A/B 闸门已为其放行铺好） | 待启动 |
| 阶段 4 | 实盘前最终 CSCV PBO 过拟合概率复核 | 待启动 |

---

## 八、给接手人 / 友人的一句话

代码在工作区 `cryptoquant_blueprint_impl/cryptoquant_auto/`，跑 `run_validation_0_5.py` 五道全绿即代表这套「防骗自己」的防线是立着的。改任何验证逻辑前，先确保这五道关仍全绿——这是底线。

## 九、跨沙盒持久化（Gist，真·无缝接续）

代码包 + 本说明 + 蓝图原文 + 阶段2/3/4 规格已推送为私有 Gist，任何新对话 / 新沙箱免服务器密钥即可拉回继续：

- Gist 地址：https://gist.github.com/wxl258/bf434bb45cff1febd42210884f92f135
- 含文件：
  - `RESTORE.md`（恢复指南）
  - `cryptoquant_blueprint_impl.tar.gz.b64`（base64 代码包，阶段0.5+1 已落地）
  - `BLUEPRINT_HANDOFF.md`（蓝图入口）
  - `docs_cryptoquant_blueprint_v0.3.md` / `docs_cryptoquant_implementation_plan.md` / `docs_cryptoquant_server_spec.md`（蓝图原文 + 任务级计划）
  - `阶段2_3_4速查.md`（阶段2/3/4 衔接索引，下一对话起步必读）
- 一键恢复：`base64 -d cryptoquant_blueprint_impl.tar.gz.b64 > cq.tar.gz && tar -xzf cq.tar.gz && cd cryptoquant_blueprint_impl && python3 cryptoquant_auto/run_validation_0_5.py`
- 下一对话起步：先跑自检确认五关全绿，再按 `docs_cryptoquant_implementation_plan.md` 第三节推进阶段2（详见 `阶段2_3_4速查.md`）

> 与合约系统 v26 的 GitHub Gist 持久化模式一致；推完用的 PAT 可随时在 GitHub 后台吊销。
