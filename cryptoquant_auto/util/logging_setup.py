"""结构化日志引导（P2：替代裸 print 到 stderr）。

统一格式：``时间戳 | 级别 | 模块 | 消息``，便于 CLS/ELK 等日志系统解析。
调用方在入口（paper_runner / demo）最开头执行一次 ``setup_logging()`` 即可，
之后用 ``logging.getLogger(__name__)`` 取模块级 logger，不再裸 print。

守护/生产部署：传 ``log_file`` 启用 ``RotatingFileHandler`` 落盘（本地轮转，
接 CLS 前的过渡方案；轮转避免单文件无限膨胀撑爆磁盘）。
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# 落盘默认：单文件 5MB，保留 5 个备份（约 30MB 上限）
_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_BACKUP_COUNT = 5


def setup_logging(level: int = logging.INFO,
                  fmt: str = _DEFAULT_FMT,
                  stream=None,
                  log_file: str | None = None,
                  max_bytes: int = _DEFAULT_MAX_BYTES,
                  backup_count: int = _DEFAULT_BACKUP_COUNT) -> logging.Logger:
    """配置根 logger 并返回名为 'cryptoquant' 的 logger。

    幂等：重复调用不会叠加 handler。

    - 不传 ``log_file``：加 stderr stream handler（交互/调试）。
    - 传 ``log_file``：加 ``RotatingFileHandler`` 落盘（守护/生产部署）。
      同一文件路径不会重复添加。
    """
    root = logging.getLogger()

    if log_file:
        # 同文件路径的 file handler 已存在则跳过（幂等）
        abs_path = os.path.abspath(log_file)
        for h in root.handlers:
            if getattr(h, "_cryptoquant_file", False) and getattr(h, "baseFilename", None) == abs_path:
                return logging.getLogger("cryptoquant")
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        fh = RotatingFileHandler(abs_path, maxBytes=max_bytes,
                                 backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, _DATE_FMT))
        fh._cryptoquant = True            # type: ignore[attr-defined]
        fh._cryptoquant_file = True       # type: ignore[attr-defined]
        root.addHandler(fh)
        root.setLevel(level)
        for noisy in ("urllib3", "openai", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        return logging.getLogger("cryptoquant")

    if any(getattr(h, "_cryptoquant", False) for h in root.handlers):
        return logging.getLogger("cryptoquant")
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, _DATE_FMT))
    handler._cryptoquant = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    # 抑制第三方噪声（openai/urllib3 等）在原型运行时的刷屏
    for noisy in ("urllib3", "openai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("cryptoquant")
