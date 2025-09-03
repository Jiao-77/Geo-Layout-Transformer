# src/utils/logging.py
import logging
import sys

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """创建并配置一个日志记录器。

    Args:
        name: 日志记录器的名称。
        level: 日志记录级别。

    Returns:
        一个配置好的日志记录器实例。
    """
    # 获取指定名称的日志记录器
    logger = logging.getLogger(name)
    # 设置日志记录器的级别
    logger.setLevel(level)

    # 创建一个处理器，用于将日志记录输出到标准输出
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # 创建一个格式化器，并将其添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 将处理器添加到日志记录器（如果尚未添加）
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
