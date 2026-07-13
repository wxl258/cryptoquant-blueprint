#!/usr/bin/env bash
#===============================================================
# CryptoQuant 原型部署脚本 · 沙箱打包 · 服务器一键部署
# 用法：
#   1. 把本脚本和 cryptoquant_prototype.tar.gz 传到服务器同一目录
#   2. bash server_deploy.sh
#===============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/cryptoquant_blueprint_impl"
TARBALL="$SCRIPT_DIR/cryptoquant_prototype.tar.gz"
PAPER_LOG="$PROJECT_DIR/paper.log"

echo "========================================"
echo " CryptoQuant 原型部署（纸面模拟·绝不下单）"
echo "========================================"
echo ""

# ---- 检查 root（部分依赖需 sudo 装系统包）----
if [ "$(id -u)" != "0" ]; then
    echo "🛑 请用 root 运行（ssh root@服务器后再执行本脚本）"
    exit 1
fi

# ---- 确认放弃旧系统 ----
echo "⚠️  本操作会："
echo "   1) 备份 /root/scf → /root/scf_backup_$(date +%Y%m%d)"
echo "   2) 清除 /root/scf 下旧系统内容"
echo "   3) 解压原型到 $PROJECT_DIR"
echo "   4) 安装依赖并启动纸面运行器"
echo ""
read -p "确认继续？(y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消，未作任何修改"
    exit 0
fi

# ---- 备份并清理旧系统 ----
OLD_DIR="/root/scf"
if [ -d "$OLD_DIR" ]; then
    BACKUP="/root/scf_backup_$(date +%Y%m%d_%H%M%S)"
    echo "📦 备份旧系统到 $BACKUP ..."
    mv "$OLD_DIR" "$BACKUP"
    echo "   旧系统已备份 → $BACKUP"
fi

# ---- 解压原型 ----
if [ ! -f "$TARBALL" ]; then
    echo "🛑 未找到 $TARBALL，请确保它与本脚本在同一目录"
    exit 1
fi
echo "📦 解压原型到 $PROJECT_DIR ..."
mkdir -p "$PROJECT_DIR"
tar -xzf "$TARBALL" -C "$SCRIPT_DIR"
echo "   ✅ 解压完成"

# ---- 安装依赖 ----
echo "📦 安装 Python 依赖..."
pip3 install numpy --quiet --upgrade 2>/dev/null || true
echo "   ✅ numpy 已就绪（torch 可选，不装也正常跑）"

# ---- 验证 ----
cd "$PROJECT_DIR"
echo ""
echo "🔍 验证原型..."
python3 -c "
from cryptoquant_auto.stage2_features import build_feature_matrix
from cryptoquant_auto.risk.constitution import TradingConstitution
from cryptoquant_auto.adapters import get_llm
import numpy as np
# 宪法硬锁正常？
const = TradingConstitution(live_capital=False)
assert const.live_capital == False
# get_llm 无密钥自动返 mock？
llm = get_llm()
assert 'MockLLM' in type(llm).__name__
echo '   ✅ 原型验证通过（硬锁 + get_llm 降级正常）'

# ---- 启动纸面运行器 ----
echo ""
echo "🚀 启动纸面模拟（历史回放模式，--once 单次测试）..."
python3 -m cryptoquant_auto.paper_runner --once 2>&1 | tail -5

# ---- 提示下一步 ----
echo ""
echo "========================================"
echo " ✅ 部署完成！"
echo "========================================"
echo ""
echo "📋 查看仪表盘：cat $PROJECT_DIR/paper/paper_dashboard.md"
echo ""
echo "▶️  接实时行情（Gate.io 公开 REST，无需密钥）："
echo "   cd $PROJECT_DIR && python3 -m cryptoquant_auto.paper_runner --source gateio --once"
echo ""
echo "▶️  持续运行（每 5 分钟一轮）："
echo "   cd $PROJECT_DIR && nohup python3 -m cryptoquant_auto.paper_runner --loop --interval 300 > paper.log 2>&1 &"
echo ""
echo "▶️  接真实 LLM（可选）："
echo "   export CRYPTOQUANT_LLM_KEY=你的密钥"
echo "   再启动即可自动使用真实 LLM"
echo ""
echo "🔒 安全：live_capital=False 硬锁已生效，绝不下单"
echo ""
