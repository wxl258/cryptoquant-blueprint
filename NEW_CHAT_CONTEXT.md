# 🔄 CryptoQuant 蓝图原型 · 新对话交接卡

> **生成时间**：2026-07-14 11:00 GMT+8
> **服务器**：🔥 `8.217.35.251`（香港 · 腾讯云轻量应用服务器）
> **仓库**：`https://github.com/wxl258/cryptoquant-blueprint.git`（公开仓库）
> **版本**：commit `0be10c4` — feat(causal): 稳定性阈值从70%调至50%

---

## 1. 🔴 紧急安全项（新对话必须优先提醒）

| 暴露的密钥 | 位置 | 处理 |
|-----------|------|------|
| GitHub PAT `ghp_UDhb...6aKc` | 旧对话中明文贴过 | **立即到 GitHub 撤销** |
| SSH 私钥（ed25519）| 旧对话中明文贴过两次 | **到腾讯云控制台重置 SSH 密钥** |
| DeepSeek API Key `sk-d8024334...` | 旧对话中明文贴过 | **到 platform.deepseek.com 撤销** |
| Binance 测试网 API Key + Secret | 旧对话中明文贴过 | **到 testnet.binancefuture.com 撤销** |

系统已删除沙箱内临时 SSH 密钥副本。**新对话不要再贴任何密钥到聊天。**

---

## 2. 服务器状态

| 项 | 值 |
|----|-----|
| IP | `8.217.35.251:22` (HK Tencent Cloud) |
| 用户 | `root` |
| 配置 | Intel Xeon 2核 / **4G 内存**（刚升级） / 40G 盘（32G 可用） |
| Python | 3.12.3 |
| 新系统路径 | `/root/cryptoquant_blueprint_impl`（git clone, 可 `git pull`） |
| crontab | `*/5 * * * * /root/cryptoquant_blueprint_impl/run_cron.sh` |
| 已装依赖 | numpy, openai, requests, scipy, pandas, statsmodels, deap(未装但备用) |

### 当前 cron 每 5 分钟执行流程

```
run_cron.sh
  ├── source .llm_secret          # DeepSeek 密钥
  ├── source .testnet_secret      # Binance 测试网密钥
  ├── python3 paper_runner        # 信号生成（Binance→DeepSeek→6币决策）
  └── python3 testnet_runner      # 测试网执行（读信号→市价单→跟踪持仓）
```

两个密钥文件位于 `/root/cryptoquant_blueprint_impl/`，`chmod 600` 保护。

---

## 3. 本次会话构建的功能（需在新对话续接）

### 3.1 币安主数据源 + Gate.io 备选

- **`BinancePublicDataSource`**（`paper_runner.py`）：只读币安 U 本位合约公开 REST，**无需密钥**
- `--source` 选项：`{history, binance, gateio}`，默认 `history`（沙箱安全）
- 服务器 cron 使用 `--source binance`（香港直连验证通过）
- 详见 `BinancePublicDataSource` vs `GateioPublicDataSource` 对称结构

### 3.2 DeepSeek V4-Flash 真 LLM

- `get_llm()` 工厂从环境变量读 `CRYPTOQUANT_LLM_BASE_URL` / `CRYPTOQUANT_LLM_MODEL`
- 思考模式（thinking）已自动禁用（`extra_body={"thinking": {"type": "disabled"}}`）
- content-JSON 兜底：若模型不吐 tool_calls 则解析 content 并仍走 schema 校验
- `.llm_secret` 文件存在于服务器，密钥已注入

### 3.3 币种统一为可配置

- 新增 `history.py:get_symbols()`，默认 6 币：BTC/ETH/SOL/BNB/XRP/TRX
- `CRYPTOQUANT_SYMBOLS` 环境变量可扩到 12 币
- `BinancePublicDataSource` / `GateioPublicDataSource` 都使用 `get_symbols()`

### 3.4 测试网真实模拟交易（`testnet_runner.py`）

- 读 `paper/paper_state.json` 信号 → `BinanceTestnetAdapter` 市价/激进限价单 → 测试网成交 + 持仓跟踪
- `submit_market()` 支持 LIMIT 激进限价（防测试网 MARKET 不成交）+ 1.5s 重查
- `_sign` 签名已改为字母序排序（修复 Binance fapi 签名要求）
- 步长/价格精度已修复（`_fmt_qty` / `_fmt_price` + 硬编码 LOT_SIZE 表）
- 仪表盘 `paper/testnet_dashboard.md`

### 3.5 日志有界化

- `paper_journal.jsonl` 超过 6MB 自动截留最近 20000 行
- 常量 `JOURNAL_MAX_BYTES` / `JOURNAL_KEEP`

### 3.6 因果发现特征筛选（新增）

- **`causal_discovery.py`**：Granger 因果 + 滚动窗口稳定性筛选
- 默认参数：lag=12, p<0.05, 稳定性≥50%
- 回退纪律：无 statsmodels / 数据不足 / 任何异常 → 全量特征
- BTC 1h 实测：`vol_regime` + `momentum` 2 个特征通过（筛掉 7 个）
- 缓存 `data/causal_features.json`，3 天自动重跑

### 3.7 服务器部署与清理

- 旧蓝图残留（备份目录+同步压缩包）已全部清理
- 保留 `cryptoquant_v9_pkg.tar.gz`（历史参考）
- 部署脚本 `run_cron.sh` + `.llm_secret.example` + `.testnet_secret` 仅存在于服务器

---

## 4. 蓝图 v0.3 完成度

| 模块 | 状态 | 说明 |
|------|------|------|
| ✅ 阶段0.5 验证基建 | **已完成** | Purged+Embargo/DSR/SPCI/受控A/B |
| ✅ 阶段2 特征引擎 | **已完成** | 9维手写特征 + regime检测 |
| ✅ 阶段3 FinMem+四角色LLM | **已完成** | DeepSeek V4-Flash 驱动 |
| ✅ 阶段4 TSFM骨架+CVaR+StockSim | **骨架已完成** | 验证脚本通过,但非完整实现 |
| ✅ 测试网执行 | **已完成** | Binance testnet 真单成交+持仓跟踪 |
| 🟡 **C.因果发现** | **✅ 首次完成** | 9→2维筛选, 已跑通 |
| ❌ **A.TSFM预报骨架** | **未开始** | Time-MoE ONNX 推理 |
| ❌ **B.CVaR仓位优化** | **未开始** | scipy.minimize |
| ❌ **D.进化优化** | **未开始** | DEAP NSGA-II |
| ❌ **E.StockSim+LLM市场** | **未开始** | Stage4仅骨架 |

---

## 5. 下一步推荐实施顺序

按专家D的分析，建议按此顺序：

```
第1周 → C.因果发现（已跑通，需集成到特征管线）
第2周 → A.TSFM（ONNX Runtime + Time-MoE small，4G 可跑）
第3周 → B.CVaR（scipy.optimize，替换仓位公式）
第4周 → D.进化（DEAP NSGA-II，参数帕累托优化）
第5周 → E.StockSim全量（订单簿仿真+LLM人造对手）
```

---

## 6. 关键代码位置索引

| 文件 | 内容 |
|------|------|
| `cryptoquant_auto/paper_runner.py` | 主运行器，含 Binance/Gateio 数据源 |
| `cryptoquant_auto/testnet_runner.py` | 测试网执行器 |
| `cryptoquant_auto/causal_discovery.py` | 因果发现模块 |
| `cryptoquant_auto/history.py` | `get_symbols()` 统一币种配置 |
| `cryptoquant_auto/adapters/real_llm.py` | DeepSeek V4 + thinking 禁用 |
| `cryptoquant_auto/adapters/binance_testnet.py` | 测试网适配器 + submit_market |
| `cryptoquant_auto/adapters/binance_testnet.py` | `_sign` 签名（已排序） |
| `tests/test_binance_datasource.py` | 币安数据源测试（含符号配置） |
| `tests/test_real_llm_deepseek.py` | DeepSeek LLM 测试 |
| `tests/test_journal_bounded.py` | 日志有界化测试 |
| `paper/paper_dashboard.md` | 实时决策仪表盘 |
| `paper/testnet_dashboard.md` | 测试网持仓仪表盘 |
| `data/causal_features.json` | 因果发现缓存 |
| `run_cron.sh` | cron 包装脚本 |
| `.llm_secret` | DeepSeek 密钥 |
| `.testnet_secret` | Binance 测试网密钥 |

---

## 7. 服务器运维命令

```bash
# 查看最新决策
cat /root/cryptoquant_blueprint_impl/paper/paper_dashboard.md

# 查看测试网持仓
cat /root/cryptoquant_blueprint_impl/paper/testnet_dashboard.md

# 手动跑一次
cd /root/cryptoquant_blueprint_impl && source .testnet_secret && source .llm_secret
python3 -m cryptoquant_auto.paper_runner --once --source binance --log-file paper/paper.log
python3 -m cryptoquant_auto.testnet_runner

# 更新代码
cd /root/cryptoquant_blueprint_impl && git pull

# 查日志
tail -f paper/paper.log
```

---

## 8. 需要新对话做的第一件事

1. **阅读本文档**（已完成）
2. **提醒用户轮换所有暴露的密钥**（GitHub PAT / SSH 密钥 / DeepSeek Key / Binance Key）
3. **确认系统正在运行**：SSH 到服务器检查 cron、仪表盘、LLM 状态
4. **继续实施**：因果发现集成→TSFM（按上面路线图）

**注意**：沙箱无法直连 `8.217.35.251`（网络限制待确认），建议通过 `git push` → 服务器 `git pull` 方式部署。

---



---



---

## 5. P0 修复状态（2026-07-15 追加，含符号 bug 修复）

> 6 项 P0 全部落地并验证：cryptoquant_auto 全量套件 48 passed / exit 0。

- P0-1 到 P0-6 已交付（详见 /workspace/P0_fix_report.md）。改动落盘服务器但未 git commit。
- realized PnL 符号反转 bug 已修复（用户确认后执行）：去掉 _calc_pnl 与 _daily_realized_pnl 末位取负，新增 2 个回归单测；实证 realized=+20、daily_realized_pnl>0。该 bug 原使 KillSwitch 日亏熔断方向反（盈利日误暂停）。
- 轮换 Binance 测试网 Key（testnet.binancefuture.com）仍待人工，SSH 无法代做。

---

## 9. P1 优化完成状态（2026-07-15）

> 圆桌 `roundtable_optimization.md` 的 P1-1~P1-7 全部落地并验证。提交 `3ad73e3`（本地，无远程不 push）。
> 全量 `cryptoquant_auto/` 套件：**124 passed / 1 skipped / 0 failed**。

| 项 | 结果 | 关键文件 |
|----|------|---------|
| P1-1 清理备份 | 🟢 9 `.bak_*`+`.safeguard_*` → `archive/`，`.gitignore` 忽略 | `archive/`, `.gitignore` |
| P1-2 单币上限统一4% | 🟢 `cvar.max_pos` 0.05→0.04 + 预算放宽日志 | `risk/cvar_optimizer.py` |
| P1-3 中段预警带 | 🟢 `WARN=0.5`；修边界 `>`→`>=`（1.8σ/5%含下界） | `risk/kill_switch.py`, `test_kill_switch.py` |
| P1-4 cron 健壮性 | 🟢 health_check 自检 + `timeout 280` + logrotate.d | `run_cron.sh`, `core/health_check.py`, `/etc/logrotate.d/cryptoquant` |
| P1-5 历史刷新 | 🟢 `limit` 1500→9000，`__main__` CLI + 每日 4:17 cron；实跑 6 币 1w=53 | `history.py` |
| P1-6 ab_harness 硬拒 | 🟢 新增 `GateRejected` + `assert_gate_passed` + e2e 测试 3 例 | `risk/gate.py`, `sim/ab_harness.py`, `test_ab_gate_e2e.py` |
| P1-7 死代码（纠正误判） | 🟢 `mean_reversion` 是活代码（generator.py:68/91 调用）保留；wf/PSI 降级 P2 | `signals/mean_reversion.py` |

**两项圆桌误判已纠正**：
1. `mean_reversion` 非死代码 → 保留 + LIVE 注释防误删；
2. `kill_switch` 中段带边界 `>` 应为 `>=`（恰好 1.8σ/5% 原不进 WARN）→ 已修，测试由 fail→pass。

**降级 P2**：合并 mock 层（P1-1）、在线 walk-forward/PSI 监控（P1-7）。
**当前 cron**：`*/5` 主信号 + `17 4 * * *` 历史刷新；logrotate 接管 `paper/*.log`。

---

## 10. P2 优化完成状态（2026-07-15）

**P2-1 因果统一门面** 🟢：新增 cryptoquant_auto/causal.py 薄门面，get_causal_features(method=granger/pcmci) 派发；Granger(生产 paper_runner 直调) 与 PCMCI(研究 stage2) 两后端均保留，未删除。+test_causal_facade(3)。

**P2-2 testnet 手续费** 🟢（此前已部署未提交，本次一并提交）：testnet_runner 计 buy/sell 双边 + taker 0.0004 手续费，dashboard 显累计手续费；+test_testnet_pnl(6)。

**P2-3 FNG UTC 对齐** 🟢（此前已部署未提交）：fetch_fng 已 //86400*86400 对齐 00:00 UTC，补回归锁 test_history_fng(2)。

**P2-4 run_validation 统一入口** 🟢：新增 cryptoquant_auto/run_validation.py，--stage 0.5/2/3/4 派发既有 main()（薄调度器，不改原模块）；CI 改走统一入口。+test_run_validation_dispatch(3)。已验证 stage0.5/4 RC=0。

**P2-5 年化口径 + 覆盖率门禁** 🟡（含一项圆桌误判纠正）：
- 审计结论：A/B 闸门(controlled_ab + run_validation_*) 刻意单 bar SR(periods_per_year=1)；metrics/evolution 已默认 8760(真实 1h 报告口径)。「A/B 也用 8760」是误判——年化 ×√8760 会把边缘 per-bar edge 放大成假显著，破坏 DSR 显著性。故**不改 A/B 默认 8760**，锁死 controlled_ab 默认=1(+test_ab_annualization_contract)。
- 覆盖率门禁：ci.yml 新增 Coverage gate 步(--cov=cryptoquant_auto, fail-under=40)，.coveragerc 排除 demo/run_validation/测试/研究重型模块；快子集 177 passed/64 deselected，引擎覆盖率 46%。
- 补 health_check 冒烟测试(3)，拉满 cron fail-closed 自检(原 0% 缺口)。

**验证汇总**：177 passed / 64 deselected(快子集，重型单测按设计由 validation stage 覆盖)；stage0.5/4 经统一调度器 RC=0；引擎覆盖率 46% > 门禁 40。
**提交**：77ede90（P2 全量）+ 后续 .gitignore 提交。本地未 push（无远程）。
**已知缺口(P3 建议)**：router/position_sizing/profile 等生产模块仍 0% 覆盖；覆盖率门禁阈值 40 为起点，后续随测试补全应上调。
