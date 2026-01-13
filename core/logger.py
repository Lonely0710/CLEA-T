"""
Cloud Server 日志配置模块

提供统一的日志配置，支持文件和控制台输出
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

# 日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的日志文件备份数量

    Returns:
        配置好的日志记录器
    """
    # 创建或获取日志记录器
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器
    if log_to_file:
        # 主日志文件
        log_file = LOG_DIR / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 错误日志文件
        error_log_file = LOG_DIR / f"{name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)


def init_cloud_logger(
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_to_console: Optional[bool] = None
) -> logging.Logger:
    """
    初始化云服务器主日志记录器

    Args:
        log_level: 日志级别（从环境变量LOG_LEVEL读取）
        log_to_file: 是否输出到文件（默认True）
        log_to_console: 是否输出到控制台（默认True）

    Returns:
        配置好的主日志记录器
    """
    # 从环境变量读取配置
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_to_file = log_to_file if log_to_file is not None else True
    log_to_console = log_to_console if log_to_console is not None else True

    # 设置主日志记录器
    logger = setup_logger(
        "cloud_server",
        log_level=log_level,
        log_to_file=log_to_file,
        log_to_console=log_to_console
    )

    # 记录初始化信息
    logger.info("=" * 50)
    logger.info("Cloud Server Logging Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log Directory: {LOG_DIR}")
    logger.info(f"Log to File: {log_to_file}")
    logger.info(f"Log to Console: {log_to_console}")
    logger.info("=" * 50)

    return logger


# 默认配置
def configure_logging():
    """
    配置根日志记录器（兼容旧代码）
    """
    # 使用环境变量配置日志级别
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # 配置根日志记录器（仅控制台输出，避免与文件日志冲突）
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=getattr(logging, log_level.upper(), logging.INFO)
    )


def log_with_category(
    logger: logging.Logger,
    level: str,
    category: str,
    message: str,
    data: Optional[Dict[str, Any]] = None
):
    """
    带分类和数据的结构化日志记录

    Args:
        logger: 日志记录器
        level: 日志级别 (info, warning, error, debug)
        category: 分类标签 (e.g. CONNECTION, PROTOCOL)
        message: 日志消息
        data: 可选的数据字典 (将格式化为JSON)
    """
    log_func = getattr(logger, level.lower(), logger.info)
    
    formatted_msg = f"[{category}] {message}"
    
    if data:
        try:
            # 尝试格式化JSON
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            formatted_msg += f"\n{json_str}"
        except Exception:
            # 格式化失败则直接附加字符串
            formatted_msg += f" {str(data)}"
            
    log_func(formatted_msg)