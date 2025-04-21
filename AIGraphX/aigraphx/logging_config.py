"""日志配置模块，提供详细的日志记录功能"""
import logging
import os
import time
from logging.config import dictConfig
from pathlib import Path
import datetime
from typing import Optional, Any

# 自定义格式化器，支持中国时区（UTC+8）
class ChinaTimeFormatter(logging.Formatter):
    """使用中国时区（UTC+8）的日志格式化器"""
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """重写时间格式化方法，使用CST时区"""
        # 获取UTC时间并添加8小时偏移量，得到中国标准时间
        utc_dt = datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
        china_tz = datetime.timezone(datetime.timedelta(hours=8))
        china_dt = utc_dt.astimezone(china_tz)
        
        if datefmt:
            return china_dt.strftime(datefmt)
        else:
            return china_dt.strftime("%Y-%m-%d %H:%M:%S")

# 确保日志目录存在
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# 带时间戳的日志文件名（使用中国时间）
utc_now = datetime.datetime.now(datetime.timezone.utc)
china_now = utc_now.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
timestamp = china_now.strftime("%Y%m%d-%H%M%S")
app_log_file = f"logs/aigraphx_{timestamp}.log"
error_log_file = f"logs/aigraphx_error_{timestamp}.log"
access_log_file = f"logs/access_{timestamp}.log"

# 日志配置字典
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": ChinaTimeFormatter,  # 使用自定义的中国时区格式化器
            "format": "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d — %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "()": ChinaTimeFormatter,  # 使用自定义的中国时区格式化器
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] [%(process)d:%(thread)d] — %(funcName)s — %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": ChinaTimeFormatter,  # 使用自定义的中国时区格式化器
            "format": "[%(asctime)s] [ACCESS] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "DEBUG",
        },
        "app_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": app_log_file,
            "formatter": "detailed",
            "level": "DEBUG",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": error_log_file,
            "formatter": "detailed",
            "level": "ERROR",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
        "access_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": access_log_file,
            "formatter": "access",
            "level": "DEBUG",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        # Uvicorn 相关日志
        "uvicorn": {"handlers": ["console", "app_file"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console", "error_file", "app_file"], "level": "DEBUG", "propagate": False},
        "uvicorn.access": {"handlers": ["console", "access_file", "app_file"], "level": "DEBUG", "propagate": False},
        
        # FastAPI 相关日志
        "fastapi": {"handlers": ["console", "app_file"], "level": "DEBUG", "propagate": False},
        
        # 应用相关日志
        "aigraphx": {"handlers": ["console", "app_file", "error_file"], "level": "DEBUG", "propagate": False},
        "aigraphx.api": {"handlers": ["console", "app_file", "error_file"], "level": "DEBUG", "propagate": False},
        "aigraphx.services": {"handlers": ["console", "app_file", "error_file"], "level": "DEBUG", "propagate": False},
        "aigraphx.repositories": {"handlers": ["console", "app_file", "error_file"], "level": "DEBUG", "propagate": False},
        
        # 第三方库日志
        "sqlalchemy.engine": {"handlers": ["console", "app_file"], "level": "WARNING", "propagate": False},
        "sqlalchemy.engine.base.Engine": {"handlers": ["app_file"], "level": "WARNING", "propagate": False},
    },
    # 根日志记录器用于捕获其他所有日志
    "root": {"handlers": ["console", "app_file", "error_file"], "level": "INFO"},
}

def setup_logging() -> logging.Logger:
    """设置日志配置"""
    dictConfig(LOGGING_CONFIG)
    # 使用中国时间显示初始化信息
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    china_now = utc_now.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    china_time_str = china_now.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"日志配置已初始化，中国标准时间: {china_time_str}，应用日志文件: {app_log_file}")
    return logging.getLogger("aigraphx") 