"""统一回测验证入口（P2-4）。

将散落的 4 个验证脚本收敛为单一命令，避免「记不住该跑哪个 stage、各自维护
argparse」的熵增：

    python -m cryptoquant_auto.run_validation --stage 4
    python -m cryptoquant_auto.run_validation --stage 0.5

设计：薄调度器，直接复用各原模块的 main()（**不改动原模块**，低风险）。
CI 改走本入口即可。各 stage 含义：
  - 0.5  阶段0.5：LLM vs 规则 同流 A/B 自洽（合成收益 DSR 显著性）。
  - 2    阶段2：信号层（因果发现 + 因子）研究验证。
  - 3    阶段3：前向 walk-forward 真实分布 A/B。
  - 4    阶段4：端到端（含 CVaR/风险感知）验证。
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict

from . import (run_validation_0_5, run_validation_stage2,
               run_validation_stage3, run_validation_stage4)

# stage 键 → 原模块 main()（均为无参，返回 int 或 None）
STAGES: Dict[str, Callable[[], "object"]] = {
    "0.5": run_validation_0_5.main,
    "2": run_validation_stage2.main,
    "3": run_validation_stage3.main,
    "4": run_validation_stage4.main,
}


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cryptoquant_auto.run_validation",
        description="CryptoQuant 回测验证统一入口（--stage 选择阶段）",
    )
    parser.add_argument(
        "--stage", required=True, choices=sorted(STAGES.keys()),
        help="验证阶段：0.5 / 2 / 3 / 4",
    )
    args = parser.parse_args(argv)
    rc = STAGES[args.stage]()
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
