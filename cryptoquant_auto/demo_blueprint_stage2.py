"""阶段2 演示入口（与阶段1 的 demo_blueprint_stage1.py 同名风格对齐）。

直接复用 run_validation_stage2.main() 跑完整进门验证 + 四任务演示。
用法：python3 -m cryptoquant_auto.demo_blueprint_stage2
"""
from __future__ import annotations

from .run_validation_stage2 import main


if __name__ == "__main__":
    raise SystemExit(main())
