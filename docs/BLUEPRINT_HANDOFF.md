# CryptoQuant 会话交接包 · 入口文档

> 生成：2026-07-12
> 用途：本会话上下文将满，此包供**下一个对话**下载引用，无需从零解释。
> 读者：下一个 AI 对话（技术向）；用户为小白，结论用白话，细节给 agent。

---

## 一、小白版一句话总结

今天做的事：从服务器下载了 CryptoQuant 交易原型代码 → 删掉一份过时的"交接卡" → 然后请多位"专家"讨论了**怎么把最新的人工智能量化技术（会聊天的 AI 多智能体 / 强化学习 / Transformer / 自动进化 / 可解释）装进这个原型里** → 产出了一整套从"蓝图"到"详细实施计划"的文档 → 还讨论了以后跑起来需要什么服务器、能不能调用 AI 接口。

所有讨论都遵守一条铁律：**只在"原型沙盒"里玩，绝不碰真钱**（真钱有独立开关锁着）。

---

## 二、当前状态（下一个对话必读）

| 项 | 状态 | 说明 |
|----|------|------|
| 原型代码下载 + 清理 | ✅ 完成 | 删除过时交接卡 + 注释死引用，重打包覆盖上传 |
| 前沿技术圆桌分析 | ✅ 完成 | `docs/cryptoquant_*` 系列 |
| 蓝图 v0 → v0.3 | ✅ 完成 | 每版都联网用真实论文校准 |
| 详细实施计划 | ✅ 完成 | `docs/cryptoquant_implementation_plan.md` |
| 服务器配置规格 | ✅ 完成 | `docs/cryptoquant_server_spec.md` |
| AI API 接入方案 | ✅ 讨论完成 | 分阶段：原型用 mock，测试网才引真 API |
| **3 个代码文件** | ⚠️ **写了但未运行** | `code/constitution.py` `code/metacontroller.py` `code/demo_blueprint_stage1.py` |
| 阶段 0–4 实际训练/集成 | ❌ 未做 | 仅蓝图 + 计划，无实际 ML 训练 |

> ⚠️ 最重要：本会话**只产出文档和 3 个未运行的代码文件**，没有真正训练或接入任何模型。下一个对话若要继续"写代码"，从实施计划第六节的第一步开始。

---

## 三、文件地图（本包内容）

| 文件 | 内容 | 给谁 |
|------|------|------|
| `README_HANDOFF.md` | 本文件（入口） | 下一个对话 |
| `docs/cryptoquant_frontier_roundtable.md` | 前沿技术落地初版分析 | 背景 |
| `docs/cryptoquant_roundtable_prototype_v1.md` | 原型改进圆桌 v1 | 背景 |
| `docs/cryptoquant_roundtable_prototype_v1.1.md` | v1 优化（零依赖优先） | 背景 |
| `docs/cryptoquant_blueprint_v0.md` | 完美蓝图初版 | 核心 |
| `docs/cryptoquant_blueprint_v0.1.md` | 联网校准选型 | 核心 |
| `docs/cryptoquant_blueprint_v0.2.md` | 补全验证/安全协议 | 核心 |
| `docs/cryptoquant_blueprint_v0.3.md` | 再优化（预训练基座+具名算法） | **最新蓝图** |
| `docs/cryptoquant_implementation_plan.md` | 详细落地计划（任务级） | **下一步必读** |
| `docs/cryptoquant_server_spec.md` | 服务器配置三档 + 实盘层 | 规划 |
| `code/constitution.py` | 交易宪法（硬约束） | 阶段1 |
| `code/metacontroller.py` | 概率化贝叶斯脊柱 | 阶段1 |
| `code/demo_blueprint_stage1.py` | 阶段1 验证 demo（未运行） | 阶段1 |

---

## 四、服务器上的位置速查

| 内容 | 服务器路径 | 说明 |
|------|-----------|------|
| 原型代码（已清理） | `/root/cryptoquant_v9_pkg.tar.gz` | 含本会话写的 3 个代码文件 |
| 原型代码（原版备份） | `/root/cryptoquant_v9_pkg.tar.gz.bak` | 删除前备份，可回滚 |
| **本交接包** | `/root/cryptoquant_blueprint_handoff.tar.gz` | 本会话全部文档 |

下载命令（下一个对话用）：
```
scp -i <key> root@8.217.35.251:/root/cryptoquant_blueprint_handoff.tar.gz ./
scp -i <key> root@8.217.35.251:/root/cryptoquant_v9_pkg.tar.gz ./
```

---

## 五、关键决策记录（避免下一个对话重蹈）

1. **过时交接卡已删除**：原 `HANDOVER_CARD_v8.md` 与 `_backup_reports/` 已删（代码其实比卡新，卡是谎言）。
2. **只做原型**：所有前沿技术仅限 MockAdapter / 测试网 / 纸面回测；实盘由 Go/No-Go Gate A–F 锁死。
3. **编号统一**：新代码注释用 P0/P1/P2 + 序号，消除历史 P0-2/C6 双轨混乱。
4. **LLM 分阶段**：原型期用 `mock_llm.py`（零 key 零成本），测试网期才引真 AI API，且密钥安全托管、超时回退规则、输出过四闸门。
5. **验证层不可逾越**：Purged+Embargo K-Fold + DSR/PSR + SPCI + 受控 A/B 是每阶段进门强制护栏。
6. **依赖纪律**：阶段 0–2 零新依赖（numpy/stdlib）；torch/LLM 后移且带降级路径。

---

## 六、给下一个对话的建议起点

若用户说"继续"或"按蓝图实施"：

1. 读 `docs/cryptoquant_implementation_plan.md` 第六节（第一步行动）。
2. 从服务器拉原型：`cryptoquant_v9_pkg.tar.gz`（已含 3 个代码文件）。
3. 先跑通 `code/` 里三个文件（阶段 0.5 验证基建 + 阶段1 脊柱），再逐阶段推进。
4. 严守：不碰实盘、原型沙盒、验证层强制。

若用户只想"看结论"：读 `docs/cryptoquant_blueprint_v0.3.md`（最新蓝图）+ 本文件即可。
