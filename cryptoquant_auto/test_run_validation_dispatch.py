"""P2-4 · 统一验证入口调度器回归测试。

锁定：
  1) STAGES 派发表含全部 4 个阶段（0.5 / 2 / 3 / 4），且均为可调用 main。
  2) 非法 --stage 触发 argparse 退出（SystemExit），不会误派发到某模块。
  3) 缺 --stage 必报错（required）。

注意：本测试**不执行**任何 stage 的 main()（避免联网/耗时）；只校验派发契约。
"""
import pytest

from cryptoquant_auto import run_validation


EXPECTED_STAGES = {"0.5", "2", "3", "4"}


def test_all_stages_present_and_callable():
    assert set(run_validation.STAGES.keys()) == EXPECTED_STAGES
    for fn in run_validation.STAGES.values():
        assert callable(fn)


def test_invalid_stage_exits():
    """非法 stage 必须触发 argparse 退出，绝不静默回落到某模块。"""
    with pytest.raises(SystemExit):
        run_validation.main(["--stage", "9"])


def test_missing_stage_required():
    """缺 --stage 必须报错（required）。"""
    with pytest.raises(SystemExit):
        run_validation.main([])
