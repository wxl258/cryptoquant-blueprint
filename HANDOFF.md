# CryptoQuant 蓝图原型 · 会话交接包

> 用途:上下文超限换新对话时,新会话拉取本仓库后读此文件即可无缝续接。
> 生成时间:2026-07-14 | 对应提交:`4285c13`(本文件另见后续提交)

---

## 0. 一句话定位(务必先读)

当前所有工作在 **CryptoQuant 蓝图原型(cryptoquant-blueprint)**——一个**四阶段沙盒**交易/分析系统:**零密钥、零本金、MockAdapter 驱动**,聚焦"防骗自己"的验证层(SPCI 序列共形 / DSR 经济 edge 实测 / 受控 A/B / Welch 自相关修正 / walk-forward 净化)。

**V31 生产系统(8.217.35.251 上 ~1200 行)已弃用**,不要再为其投入重构。本仓库即未来主线。

---

## 1. 本会话完成清单

| 任务 | 产出 | 状态 |
|------|------|------|
| P2-A 测试覆盖 | `tests/` 0 → **63 项 pytest** | ✅ |
| P2-B 结构化日志 | `cryptoquant_auto/util/logging_setup.py`(RotatingFileHandler 落盘 + 幂等 `setup_logging`) | ✅ |
| P2-C 胖文件重构 | `demo.py` 808→435 行;分析块抽离至 `demo_sections.py`(408 行) | ✅ |
| P2-D 回测验证 | `test_backtest` 2→7 项(负期望/失败闭环/可复现/maker-taker) | ✅ |
| 根因修复 | `meta/memory.py` `reflect()` 解除禁令时清 `forbidden_at`,解除 stage3 死锁 | ✅ |
| 部署包(plan A) | `deploy/start.sh` + `deploy/systemd/cryptoquant.service` + `deploy/RUNBOOK.md` | ✅ 本地就绪 |
| CI 守门(plan B) | `.github/workflows/ci.yml` + `verify_p1_fixes.py` 入仓并修 sys.path | ✅ |
| 仓库卫生 | `.gitignore` 排除 `*.log/*.lock` 与 `paper/` 运行期产物 | ✅ |

---

## 2. 当前仓库状态

- **远程**:`https://github.com/wxl258/cryptoquant-blueprint.git`
- **默认分支**:`main`
- **最新提交**:`4285c13`(P2 收尾 + 部署 + CI 守门)
- **CI**:已随 `4285c13` 推送自动触发,去 **Actions** 看 `CI` 工作流
- **本地工作树**:干净(本文件除外,见末尾提交说明)
- **本地模拟 CI 全链**:`ALL GREEN`(pytest 63 + verify 12 + stage4/3/0.5 ✅)

---

## 3. 关键文件索引

| 路径 | 作用 |
|------|------|
| `.github/workflows/ci.yml` | CI 守门:pytest + verify_p1_fixes + 三验证阶段 |
| `cryptoquant_auto/util/logging_setup.py` | 日志落盘(RotatingFileHandler,幂等) |
| `cryptoquant_auto/demo_sections.py` | 从 demo.py 抽离的分析块(回归/回测/OOS/regime/testnet) |
| `cryptoquant_auto/demo.py` | 主入口(已瘦身为 435 行) |
| `deploy/start.sh` | 守护启动脚本(`--loop --interval 300 --source history --log-file`) |
| `deploy/systemd/cryptoquant.service` | systemd 单元(Restart=always + NoNewPrivileges) |
| `deploy/RUNBOOK.md` | 服务器部署 / 运维 / 排障手册 |
| `verify_p0_fixes.py` / `verify_p1_fixes.py` | P0/P1 修复验证(12 项/批) |
| `tests/` | 63 项 pytest |
| `cryptoquant_auto/run_validation_stage3.py` / `stage4.py` / `run_validation_0_5.py` | 阶段进门验证 |
| `cryptoquant_auto/meta/memory.py` | FinMem 分层记忆(含本次 forbidden_at 根因修复) |

---

## 4. 常用命令

```bash
# 本地模拟 CI 全链(与 GitHub Actions 一致)
python3.11 -m pytest tests/ -q && \
python3.11 verify_p1_fixes.py && \
python3.11 -m cryptoquant_auto.run_validation_stage4 && \
python3.11 -m cryptoquant_auto.run_validation_stage3 && \
python3.11 -m cryptoquant_auto.run_validation_0_5

# 跑单次阶段验证
python3.11 -m cryptoquant_auto.run_validation_stage4

# 本地跑 paper runner(日志落盘)
python3.11 -m cryptoquant_auto.paper_runner --once --source history --log-file /tmp/paper.log
```

---

## 5. 待办 / 下一步

- [ ] **去 Actions 确认 CI 全绿**(https://github.com/wxl258/cryptoquant-blueprint/actions)
- [ ] **吊销聊天里暴露的两段 `ghp_` PAT**(安全,见第 6 节)
- [ ] **服务器部署**:按 `deploy/RUNBOOK.md` —— `scp -r` 到 `/opt/cryptoquant/`、建 `cryptoquant` 用户、改 `cryptoquant.service` 占位路径、`systemctl enable --now cryptoquant`
- [ ] **后续增强(可选)**:回测防过拟合(参数变更人工审阅,勿自动替换)、验证层继续加固
- [ ] **V31 生产系统已弃用**,勿再投入

---

## 6. 约束与坑(给新会话)

- **沙箱可连 GitHub**:`curl github.com` 返回 200,可直接 `git push`(旧约束"网络阻断"已不通)。
- **推送含 workflow 文件的提交,令牌须带 `workflow` scope**,否则被远程拒绝(`refusing to allow a Personal Access Token to create or update workflow ... without workflow scope`)。普通 `repo` scope 不够。
- **令牌勿明文贴聊**;用完即去 https://github.com/settings/tokens 吊销。
- **蓝图 fail-closed**:`LIVE_CAPITAL=False` 硬锁,任何实盘动作被否决。无密钥、无本金。
- **运行期产物已被 `.gitignore` 排除**(`*.log`、`*.lock`、`paper/paper_*`);提交时勿 `git add -A` 误带。
- **`verify_p1_fixes.py` 的 sys.path 已从 repo 根修正**,CI 在仓库根目录运行无误。

---

## 7. 新会话续接步骤

1. `git clone` / `git pull` 本仓库,`git log` 看最新提交。
2. 读本文档 + `deploy/RUNBOOK.md`;看 Actions 里 CI 是否全绿。
3. 按第 5 节"待办"推进;优先确认 CI 绿 + 部署上线。
4. 任何改动走 `git commit` + `git push`,CI 自动守门。

---

*本交接包由上一会话在上下文超限前生成,覆盖 P2 工程化收尾、部署包、CI 守门与首次推送。*
