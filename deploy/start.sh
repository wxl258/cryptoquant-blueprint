#!/usr/bin/env bash
# CryptoQuant 纸面模拟守护启动脚本（零资金 / 零密钥 / fail-closed）
#
# 由 systemd 单元或手动调用。作用：
#   - 定位项目根（脚本位于 <项目>/deploy/start.sh，上溯一级即项目根）
#   - 创建日志目录
#   - 以 --loop 持续运行 paper_runner，并把结构化日志落盘到 LOG_DIR/paper.log
#
# 环境变量（均可覆盖，便于不同服务器/场景）：
#   CRYPTOQUANT_LOG_DIR    日志目录（默认 <项目>/logs）
#   CRYPTOQUANT_INTERVAL  循环间隔秒（默认 300）
#   CRYPTOQUANT_SOURCE    数据源 history | gateio（默认 history）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="${CRYPTOQUANT_LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"

exec python3 -m cryptoquant_auto.paper_runner \
  --loop \
  --interval "${CRYPTOQUANT_INTERVAL:-300}" \
  --source "${CRYPTOQUANT_SOURCE:-history}" \
  --log-file "$LOG_DIR/paper.log"
