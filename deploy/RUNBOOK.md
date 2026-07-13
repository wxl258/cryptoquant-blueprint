# CryptoQuant 纸面模拟 · 部署与运维 Runbook

> 对象：`cryptoquant-blueprint`（零资金 / 零密钥 / MockAdapter，纸面模拟，绝不下单）
> 目标：在云服务器以 **systemd 守护进程** 7×24 运行 `paper_runner --loop`，结构化日志**落盘轮转**。

## 一、前置条件

- Linux 服务器（systemd 发行版：Ubuntu 20.04+ / Debian 11+ / CentOS 7+）
- Python 3.11+
- 项目已放到服务器，例如 `/opt/cryptoquant/cryptoquant-blueprint`
- 仅用 `history` 数据源时可**完全离线**跑（复用本地 `history_cache.json`）；`gateio` 数据源需服务器能访问 `api.gateio.ws`（只读、不下单）

## 二、部署步骤

### 1. 上传代码
```bash
scp -r cryptoquant-blueprint user@<你的服务器IP>:/opt/cryptoquant/
```

### 2. 建专用用户（最小权限，非 root 运行）
```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin cryptoquant
sudo chown -R cryptoquant:cryptoquant /opt/cryptoquant
```

### 3. 按服务器改占位符
编辑 `deploy/systemd/cryptoquant.service`：
- `User` / `Group` → 你的运行用户（默认 `cryptoquant`）
- `WorkingDirectory` / `ExecStart` / `Documentation` 路径 → 实际项目根
- 如需日志落到 `/var/log/cryptoquant`，在 `[Service]` 加：
  `Environment=CRYPTOQUANT_LOG_DIR=/var/log/cryptoquant`

### 4. 安装 systemd 单元
```bash
sudo cp deploy/systemd/cryptoquant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cryptoquant
```

## 三、运维检查清单

| 检查项 | 命令 | 期望 |
|--------|------|------|
| 服务状态 | `sudo systemctl status cryptoquant` | `active (running)` |
| 落盘日志 | `sudo tail -f /opt/cryptoquant/cryptoquant-blueprint/logs/paper.log` | 结构化日志 `时间戳 \| 级别 \| cryptoquant \| 消息` |
| 仪表盘 | `cat /opt/cryptoquant/cryptoquant-blueprint/paper/paper_dashboard.md` | 每 5 分钟刷新 |
| 并发锁 | 手动再启一个实例 | 第二个实例因 `fcntl` 排他锁**立即退出**（防并发写损坏）|
| 硬锁 | 临时设 `LIVE_CAPITAL=True` 启动 | 启动即退出，绝不碰实盘（宪法 R0）|

### 常用操作
```bash
sudo systemctl restart cryptoquant   # 重启
sudo systemctl stop cryptoquant      # 停止
journalctl -u cryptoquant -f         # 看 systemd 捕获的启动/崩溃
```

### 日志轮转
`--log-file` 启用 `RotatingFileHandler`：单文件 **5MB**、保留 **5 个备份**（约 30MB 上限），无需额外 `logrotate` 配置。备份文件命名 `paper.log.1` … `paper.log.5`。

### 单次/调试运行（不走守护）
```bash
cd /opt/cryptoquant/cryptoquant-blueprint
python3 -m cryptoquant_auto.paper_runner --once            # 单次，日志到 stderr
python3 -m cryptoquant_auto.paper_runner --once --log-file /tmp/cq.log   # 单次落盘
```

## 四、故障排查

- **启动即退，日志无内容**：多半是 `LIVE_CAPITAL=True`（硬锁）或 `paper/` 目录无写权限。看 `journalctl -u cryptoquant`。
- **日志落盘 PermissionError**：确认 `CRYPTOQUANT_LOG_DIR` 目录归属运行用户（单元已含 `ExecStartPre` 建 `/var/log/cryptoquant`，自定义路径需手动建）。
- **数据/记忆污染**：`FinMemMemory` 默认写 `<项目>/cryptoquant_auto/data/`。如需干净状态，停服后删除该目录下的 `finmem_*.json` 再启动。
- **进程卡死不刷新**：`fcntl` 锁在进程死亡时自动释放；`Restart=always` 会在崩溃后 10s 重启。

## 五、安全边界（fail-closed，架构级不可绕过）

1. `LIVE_CAPITAL=False` 为唯一合法值（宪法 R0），置 `True` 启动即退。
2. 数据源只用历史回放 / 只读公开 REST，**不接任何可下单适配器**。
3. 全程不调用 `submit/cancel`，产出只落 `paper/`（日志 + 仪表盘）。
4. 并发排他锁保证单写者，避免多实例互相截断仪表盘。
