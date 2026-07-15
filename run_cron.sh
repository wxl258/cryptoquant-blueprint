#!/usr/bin/env bash
# CryptoQuant 蓝图原型 · 生产 cron 包装脚本
#
# 【P0-4 可观测性】相对 paper_runner 自身的 paper/paper.log（业务日志），本包装层
# 额外落地 paper/cron.log 记录 cron 执行轨迹与任何阶段异常/回溯；并支持失败告警
# webhook（CRYPTOQUANT_ALERT_WEBHOOK，未配置则静默 no-op，不影响正常流程）。
# 【P1-4 健壮性】启动前跑 health_check（模块完整性，fail-closed）；两阶段整体用
# timeout 280s 包裹，避免单轮卡死拖垮 5min 槽位（280<300 留余量）；paper.log 由
# /etc/logrotate.d/cryptoquant 按 size+daily 轮转压缩。
set -eo pipefail
cd /root/cryptoquant_blueprint_impl

# ---- cron 包装层日志（独立于 paper.log）----
CRON_LOG="paper/cron.log"
mkdir -p paper
exec > >(tee -a "$CRON_LOG") 2>&1

# ---- 失败告警（可选 webhook）----
# 配置后任一阶段非 0 退出即 POST 一条 JSON 告警；未配置静默，绝不阻断流程。
alert() {
  local msg="$1"
  local wh="${CRYPTOQUANT_ALERT_WEBHOOK:-}"
  if [ -n "$wh" ]; then
    curl -fsS -m 5 -X POST "$wh" \
      -H 'Content-Type: application/json' \
      -d "{\"text\":\"[CryptoQuant cron] $msg @ $(date -Is)\"}" >/dev/null 2>&1 || true
  fi
  echo "[alert] $(date -Is) $msg"
}

echo "[cron] $(date -Is) 启动"

# DeepSeek 环境变量(最新一代 V4)
export CRYPTOQUANT_LLM_BASE_URL="https://api.deepseek.com"
export CRYPTOQUANT_LLM_MODEL="deepseek-v4-flash"
if [ -f .llm_secret ]; then source .llm_secret; fi

# 测试网密钥(市价单用)
if [ -f .testnet_secret ]; then source .testnet_secret; fi

# ---- 启动前健康检查(Fail-closed：模块完整性不达标禁止本轮开仓) ----
if ! python3 -m cryptoquant_auto.core.health_check; then
  alert "health_check 未通过，禁止本轮交易(见上方 XX 项)"
  exit 1
fi

# 先出信号,再同步到测试网(整体 timeout 280s 包裹，避免 5min 槽位重叠卡死)
if ! timeout 280 python3 -m cryptoquant_auto.paper_runner --once --source binance --log-file paper/paper.log; then
  alert "paper_runner 阶段失败(exit=$?)"
  exit 1
fi
if ! timeout 280 python3 -m cryptoquant_auto.testnet_runner; then
  alert "testnet_runner 阶段失败(exit=$?)"
  exit 1
fi
echo "[cron] $(date -Is) 完成"
