"""
日志工具。

重构后每次训练 / 测试由 `RunContext` 决定 ``run.log`` 路径，调用时显式传入 ``log_file``。
若 ``log_file`` 省略，则回落到 ``experiments/runs/_loose/<name>_<YYYYMMDD_HHMM>.log``——
主要用于调试或非 run 场景，避免污染项目根目录。
"""
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.path_tool import get_abs_path

# 不再在 import 时创建固定的 logs/ 目录；fallback 路径延迟到实际使用时才 mkdir。
_FALLBACK_LOG_DIR = get_abs_path(os.path.join("experiments", "runs", "_loose"))

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def _resolve_log_file(name: str, log_file) -> str:
    """给定 name 与可选 log_file，返回最终落盘路径并确保父目录存在。"""
    if log_file:
        d = os.path.dirname(os.path.abspath(log_file))
        if d:
            os.makedirs(d, exist_ok=True)
        return log_file
    os.makedirs(_FALLBACK_LOG_DIR, exist_ok=True)
    return os.path.join(
        _FALLBACK_LOG_DIR,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.log",
    )


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None,
) -> logging.Logger:
    """返回一个 logger；避免重复加 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件 handler
    file_path = _resolve_log_file(name, log_file)
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


def get_file_only_logger(
        name: str,
        log_file: str = None,
        file_level: int = logging.INFO,
) -> logging.Logger:
    """仅写入文件、不输出到控制台的 logger，避免干扰 tqdm 进度条。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_path = _resolve_log_file(name, log_file)
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    logger = get_logger()
    logger.info("info")
    logger.error("error")
    logger.warning("warning")
    logger.debug("debug")
