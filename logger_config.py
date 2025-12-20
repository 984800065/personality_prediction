"""
Logger配置模块
统一管理日志配置
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = None,
    log_dir: str = "logs",
    colorize: bool = True
):
    """
    配置loguru logger
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件名（可选），如果提供则保存到文件
        log_dir: 日志目录
        colorize: 是否启用彩色输出
    """
    # 移除默认handler
    logger.remove()
    
    # 控制台输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{file.path}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 文件输出格式（更详细，不包含颜色代码）
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{file.path}:{line} | "
        "{function} | "
        "{message}"
    )
    
    # 添加控制台handler
    logger.add(
        sys.stderr,
        format=console_format,
        colorize=colorize,
        level=log_level
    )
    
    # 如果指定了日志文件，则添加文件handler
    if log_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        log_file_path = log_dir_path / log_file
        
        logger.add(
            log_file_path,
            format=file_format,
            colorize=False,
            level=log_level,
            rotation="10 MB",  # 日志文件大小超过10MB时轮转
            retention="7 days",  # 保留7天的日志
            compression="zip"  # 压缩旧日志
        )
    
    return logger

