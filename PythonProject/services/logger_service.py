import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOG_DIR, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def setup_logger(name="Agent", log_level=logging.DEBUG):
    """配置并返回日志记录器"""
    # 创建日志目录
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 文件处理器（轮转）
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, LOG_FILE),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # 控制台处理器（开发时方便查看）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 统一格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建全局日志实例，供其他模块导入使用
logger = setup_logger()

def get_logger():
    """获取日志实例"""
    return logger