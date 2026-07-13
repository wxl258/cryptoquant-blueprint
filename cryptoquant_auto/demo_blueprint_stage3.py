"""阶段3 演示入口（复用 run_validation_stage3.main）。

蓝图：专家B · 情绪LLM（任务16–19）。零依赖 / 零资金 / 沙盒可跑。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptoquant_auto.run_validation_stage3 import main

if __name__ == "__main__":
    raise SystemExit(main())
