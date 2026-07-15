"""CryptoQuant Web 仪表盘（Flask 入口）。

路由：
  /         概览：仓位/PnL/KillSwitch/最近成交
  /signals  信号面板：各币方向/置信/regime
  /logs     系统日志：错误过滤+搜索

启动：python3 dashboard.py  (监听 0.0.0.0:5000)
安全：仅内网，单用户无认证。
"""
import json
import time
from pathlib import Path

from flask import Flask, render_template, request

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
PAPER = BASE / "paper"


def _fmt_ts(ts):
    """Unix 时间戳 → 本地时间字符串。"""
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(float(ts)))
    except Exception:
        return str(ts)[:16]


@app.template_filter("strftime")
def _jinja_strftime(ts):
    return _fmt_ts(ts)


def _load_json(name: str) -> dict:
    p = PAPER / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _load_file(name: str) -> str:
    p = PAPER / name
    if p.exists():
        try:
            return p.read_text()
        except Exception:
            return ""
    return ""


def _get_cron_last() -> str:
    """从 cron.log 取最后一次完成时间。"""
    p = PAPER / "cron.log"
    if p.exists():
        try:
            lines = p.read_text().strip().split("\n")
            for line in reversed(lines):
                if "[cron]" in line:
                    return line.split("[cron]")[-1].strip()
        except Exception:
            pass
    return "N/A"


def _get_log_errors(lines: int = 50) -> list:
    """取最近 N 行日志，标记 ERROR/WARNING。"""
    p = PAPER / "cron.log"
    if not p.exists():
        return []
    try:
        all_lines = p.read_text().strip().split("\n")
    except Exception:
        return []
    return all_lines[-lines:]


@app.route("/")
def index():
    state = _load_json("testnet_state.json")
    fill = _load_json("fill_history.json") or []
    # 最近 8 笔成交（取最晚的，fill_history 按时间升序）
    recent_fills = fill[-8:] if fill else []

    pos = state.get("positions", [])
    realized = state.get("realized_pnl", 0)
    unrealized = state.get("unrealized_pnl", 0)
    fees = state.get("total_fees", 0)
    total = realized + unrealized - fees
    ks = state.get("signals", {})  # actually signals dict, KS not stored here
    fills_count = state.get("total_fills", 0)

    # 从 paper 读 KillSwitch
    paper_state = _load_json("paper_state.json")

    return render_template("index.html",
                           positions=pos,
                           realized=realized,
                           unrealized=unrealized,
                           fees=fees,
                           total=total,
                           fills_count=fills_count,
                           recent_fills=recent_fills,
                           cron_last=_get_cron_last(),
                           fill_count=len(fill))


@app.route("/signals")
def signals():
    paper = _load_json("paper_state.json")
    records = paper.get("records", [])
    # 从 testnet_state 拿实际持仓方向
    testnet = _load_json("testnet_state.json")
    positions = {p["symbol"]: p for p in testnet.get("positions", [])}
    return render_template("signals.html",
                           records=records,
                           positions=positions)


@app.route("/logs")
def logs():
    search = request.args.get("q", "").lower()
    raw_lines = _get_log_errors(200)
    if search:
        raw_lines = [l for l in raw_lines if search in l.lower()]
    return render_template("logs.html", lines=raw_lines, search=request.args.get("q", ""))


if __name__ == "__main__":
    import logging
    import sys
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"🚀 CryptoQuant Dashboard → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
