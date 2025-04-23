"""
日志配置模块 (Logging Configuration Module)

功能 (Function):
这个模块负责配置 Python 标准的 `logging` 模块，为整个 AIGraphX 应用提供统一、结构化、可配置的日志记录功能。主要包括：
1. 定义日志格式 (Formatters)，包括时间戳、日志级别、模块名、行号、消息等。
2. 定义日志处理器 (Handlers)，决定日志输出到哪里，例如控制台 (Console) 和不同级别的日志文件 (应用日志、错误日志、访问日志)。
3. 配置不同的日志记录器 (Loggers)，例如针对 uvicorn、fastapi、sqlalchemy 以及应用本身的不同模块 (`aigraphx`, `aigraphx.api` 等)，可以为它们设置不同的日志级别和处理器。
4. 提供一个 `setup_logging` 函数，用于在应用启动时应用这些配置。
5. 实现了一个自定义的日志格式化器 `ChinaTimeFormatter`，确保日志时间戳显示为中国标准时间 (UTC+8)。
6. 自动创建 `logs` 目录，并使用带时间戳的文件名来存储日志，便于归档和查找。

交互 (Interaction):
- 依赖 (Depends on):
    - `aigraphx.core.config`: (隐式) 日志级别可以通过环境变量 `LOG_LEVEL` 控制，但当前配置字典中硬编码了 DEBUG 或 INFO，未来可以改为从 config 加载。
- 被导入 (Imported by):
    - `aigraphx.main`: 在 FastAPI 应用启动时调用 `setup_logging()` 来初始化日志系统。
    - `scripts/*`: 独立的脚本也应该调用 `setup_logging()` 来配置它们的日志记录。

设计原则 (Design Principles):
- **集中配置 (Centralized Configuration):** 使用字典 (`LOGGING_CONFIG`) 统一管理所有日志设置。
- **标准化 (Standardization):** 基于 Python 内置的 `logging` 模块，易于理解和扩展。
- **灵活性 (Flexibility):** 可以方便地调整日志级别、格式和输出目标。
- **环境适应性 (Environment Adaptability):** 虽然当前硬编码较多，但设计上易于修改为根据不同环境 (`settings.environment`) 加载不同配置。
- **实用性 (Utility):** 区分不同级别的日志文件（通用、错误、访问），并使用滚动文件处理器 (`RotatingFileHandler`) 防止日志文件无限增大。
- **时区统一 (Timezone Consistency):** 确保所有日志时间戳使用中国标准时间。
"""

import logging
import os
import time
from logging.config import dictConfig
from pathlib import Path
import datetime
from typing import Optional, Any


# --- 自定义日志格式化器 (Custom Log Formatter) ---


# 定义一个继承自 logging.Formatter 的类
class ChinaTimeFormatter(logging.Formatter):
    """
    自定义日志格式化器，确保日志时间戳显示为中国标准时间 (UTC+8)。

    默认的 logging.Formatter 使用的是服务器本地时间或 UTC 时间，
    这个类重写了 `formatTime` 方法来实现时区转换。
    """

    # 重写父类的 formatTime 方法
    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """
        将日志记录的创建时间 (UTC) 转换为中国标准时间 (CST/UTC+8) 并格式化。

        Args:
            record (logging.LogRecord): 当前正在处理的日志记录对象，包含了时间戳 (record.created)。
            datefmt (Optional[str]): 用户指定的时间格式字符串 (例如 "%Y-%m-%d")，如果为 None，则使用默认格式。

        Returns:
            str: 格式化后的中国标准时间字符串。
        """
        # record.created 是日志记录创建时的 Unix 时间戳 (自 epoch 以来的秒数)
        # 1. 将 Unix 时间戳转换为带有时区的 UTC datetime 对象
        utc_dt = datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)

        # 2. 定义中国时区 (UTC+8)
        china_tz = datetime.timezone(datetime.timedelta(hours=8))

        # 3. 将 UTC 时间转换为中国时区时间
        china_dt = utc_dt.astimezone(china_tz)

        # 4. 根据用户是否提供了 datefmt，使用相应的格式进行格式化
        if datefmt:
            s = china_dt.strftime(datefmt)
        else:
            # 如果没有指定格式，使用默认的 "年-月-日 时:分:秒" 格式
            s = china_dt.strftime("%Y-%m-%d %H:%M:%S")
            # Pylint 可能会建议添加毫秒，可以根据需要添加：
            # s = china_dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] # 格式化到毫秒
        return s


# --- 日志目录和文件准备 ---

# 定义日志文件存放的目录路径
logs_dir = Path("logs")
# 创建 logs 目录，如果目录已存在则忽略 (exist_ok=True)
logs_dir.mkdir(exist_ok=True)

# 生成带时间戳的日志文件名，确保每次应用启动时生成新的日志文件
# 1. 获取当前的 UTC 时间
utc_now = datetime.datetime.now(datetime.timezone.utc)
# 2. 转换为中国标准时间
china_now = utc_now.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
# 3. 格式化时间戳为 "年月日-时分秒" 字符串
timestamp = china_now.strftime("%Y%m%d-%H%M%S")

# 定义不同类型日志文件的完整路径和名称
# 应用主日志文件
app_log_file = logs_dir / f"aigraphx_{timestamp}.log"
# 错误日志文件 (只记录 ERROR 及以上级别的日志)
error_log_file = logs_dir / f"aigraphx_error_{timestamp}.log"
# 访问日志文件 (主要记录 Uvicorn 的访问日志)
access_log_file = logs_dir / f"access_{timestamp}.log"


# --- 日志配置字典 (Logging Configuration Dictionary) ---

# 这是传递给 logging.config.dictConfig 的核心配置
LOGGING_CONFIG = {
    "version": 1,  # 配置模式的版本，目前固定为 1
    "disable_existing_loggers": False,  # 是否禁用在配置加载前已存在的 logger 实例。False 表示不禁用，允许它们继续工作并继承配置。
    # --- 格式化器定义 (Formatters Definition) ---
    "formatters": {
        # 'default': 一个通用的格式化器名称
        "default": {
            # '()': 指定用于创建格式化器实例的类，这里使用我们自定义的 ChinaTimeFormatter
            "()": ChinaTimeFormatter,
            # 'format': 定义日志输出的具体格式字符串
            # %(asctime)s: 日期时间 (由 formatTime 处理)
            # %(levelname)s: 日志级别 (e.g., INFO, WARNING)
            # %(name)s: Logger 的名称 (e.g., aigraphx.services)
            # %(lineno)d: 日志发生处的代码行号
            # %(message)s: 实际的日志消息内容
            "format": "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d — %(message)s",
            # 'datefmt': 传递给 formatTime 的时间格式字符串
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        # 'detailed': 一个更详细的格式化器，包含更多信息
        "detailed": {
            "()": ChinaTimeFormatter,
            # %(filename)s: 文件名
            # %(process)d: 进程 ID
            # %(thread)d: 线程 ID
            # %(funcName)s: 函数名
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] [%(process)d:%(thread)d] — %(funcName)s — %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        # 'access': 专门用于记录访问日志的格式化器
        "access": {
            "()": ChinaTimeFormatter,
            "format": "[%(asctime)s] [ACCESS] %(message)s",  # 格式相对简单
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    # --- 处理器定义 (Handlers Definition) ---
    "handlers": {
        # 'console': 输出到控制台 (标准输出/错误流)
        "console": {
            # 'class': 指定处理器的类
            "class": "logging.StreamHandler",
            # 'formatter': 使用哪个格式化器
            "formatter": "detailed",  # 控制台使用详细格式
            # 'level': 这个处理器处理的最低日志级别
            "level": "DEBUG",  # 控制台显示所有 DEBUG 及以上级别的日志
            # 'stream': 可以指定输出流，默认是 sys.stderr
            # "stream": "ext://sys.stdout", # 例如，显式指定标准输出
        },
        # 'app_file': 输出到应用主日志文件，并进行滚动
        "app_file": {
            # 'class': 使用滚动文件处理器
            "class": "logging.handlers.RotatingFileHandler",
            # 'filename': 指定日志文件路径
            "filename": str(app_log_file),  # Path 对象需要转为字符串
            "formatter": "detailed",  # 文件使用详细格式
            "level": "DEBUG",  # 文件记录所有 DEBUG 及以上级别的日志
            # 'maxBytes': 单个日志文件的最大大小 (字节)，这里是 10MB
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            # 'backupCount': 保留的旧日志文件数量
            "backupCount": 5,
            # 'encoding': 日志文件的编码
            "encoding": "utf-8",
        },
        # 'error_file': 输出到错误日志文件，只记录错误及以上级别
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(error_log_file),
            "formatter": "detailed",
            "level": "ERROR",  # 只处理 ERROR 和 CRITICAL 级别的日志
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        # 'access_file': 输出到访问日志文件
        "access_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(access_log_file),
            "formatter": "access",  # 使用 'access' 格式化器
            "level": "INFO",  # 通常访问日志是 INFO 级别 (uvicorn.access 默认为 INFO)
            # 如果希望记录 DEBUG 级别的访问信息（如果 uvicorn 配置了），可以设为 DEBUG
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    # --- 日志记录器配置 (Loggers Configuration) ---
    "loggers": {
        # 配置 uvicorn 相关的 logger
        "uvicorn": {
            # 'handlers': 这个 logger 使用哪些处理器
            "handlers": ["console", "app_file"],  # uvicorn 主日志输出到控制台和应用文件
            # 'level': 这个 logger 处理的最低级别
            "level": "INFO",
            # 'propagate': 是否将日志消息传递给父 logger (root logger)。False 表示不传递。
            "propagate": False,  # 阻止 uvicorn 日志被 root logger 重复处理
        },
        "uvicorn.error": {
            # uvicorn 错误日志输出到控制台、错误文件和应用文件
            "handlers": ["console", "error_file", "app_file"],
            "level": "INFO",  # 通常设为 INFO，如果需要详细调试 uvicorn 错误则设为 DEBUG
            "propagate": False,
        },
        "uvicorn.access": {
            # uvicorn 访问日志输出到控制台、访问文件和应用文件
            "handlers": ["console", "access_file", "app_file"],
            "level": "INFO",  # 对应 access_file handler 的 level
            "propagate": False,
        },
        # 配置 FastAPI 相关的 logger
        "fastapi": {
            "handlers": ["console", "app_file"],
            "level": "INFO",  # FastAPI 本身的日志通常不需要 DEBUG
            "propagate": False,
        },
        # 配置我们自己的应用 ('aigraphx' 包及其子模块) 的 logger
        "aigraphx": {  # 这是根应用 logger
            # 所有 aigraphx 下的日志都输出到控制台和应用文件，错误级别的也输出到错误文件
            "handlers": ["console", "app_file", "error_file"],
            # 应用日志级别可以根据需要调整，DEBUG 会非常详细
            "level": "DEBUG",
            "propagate": False,  # 不传递给 root，因为我们已经为它指定了处理器
            # 如果设为 True，并且 root logger 也配置了 handler，可能会重复记录
        },
        # 可以为特定子模块配置更细致的日志行为，但通常继承 'aigraphx' 的配置即可
        # "aigraphx.api": { ... },
        # "aigraphx.services": { ... },
        # "aigraphx.repositories": { ... },
        # 配置第三方库的 logger (示例)
        "httpx": {  # 如果使用了 httpx 库
            "handlers": ["console", "app_file"],
            "level": "WARNING",  # 通常第三方库设置为 WARNING 或 ERROR，避免过多无关日志
            "propagate": False,
        },
        "asyncio": {  # 异步 IO 库
            "handlers": ["console", "error_file"],
            "level": "WARNING",
            "propagate": False,
        },
        # Neo4j 驱动日志
        "neo4j": {
            "handlers": ["console", "app_file"],
            "level": "WARNING",  # 设置为 INFO 或 DEBUG 可以看到驱动的详细操作
            "propagate": False,
        },
        # Psycopg (PostgreSQL 驱动) 日志
        "psycopg": {
            "handlers": ["console", "app_file"],
            "level": "WARNING",  # 设置为 INFO 或 DEBUG 可以看到 SQL 执行和连接池信息
            "propagate": False,
        },
        "psycopg.pool": {  # 连接池日志可以单独控制
            "handlers": ["console", "app_file"],
            "level": "INFO",
            "propagate": False,
        },
        # Sentence Transformers 库日志
        "sentence_transformers": {
            "handlers": ["console", "app_file"],
            "level": "INFO",
            "propagate": False,
        },
        # 如果使用了 SQLAlchemy (当前项目未使用，保留作为示例)
        # "sqlalchemy.engine": {
        #     "handlers": ["console", "app_file"],
        #     "level": "WARNING", # 设置为 INFO 可以看到执行的 SQL 语句
        #     "propagate": False,
        # },
    },
    # 配置根 logger (Root Logger)
    # 它会捕获所有未被特定 logger 处理（且 propagate=True）的日志，以及直接使用 logging.info() 等产生的日志
    "root": {
        "handlers": ["console", "app_file", "error_file"],  # 根 logger 也输出到所有地方
        "level": "INFO",  # 根 logger 的级别设为 INFO，避免库的 DEBUG 日志淹没控制台
        # 注意: 如果 'aigraphx' level 是 DEBUG，且 propagate=True，那么 DEBUG 日志也会被 root 处理
        # 但由于 'aigraphx' 设置了 propagate=False，所以应用本身的 DEBUG 日志不会被 root 重复处理
    },
}

# --- 日志设置函数 ---


def setup_logging() -> None:
    """
    应用日志配置。

    在应用启动时调用此函数，以根据 LOGGING_CONFIG 字典配置整个日志系统。
    """
    # 使用 logging.config.dictConfig 应用上面定义的配置字典
    dictConfig(LOGGING_CONFIG)

    # 获取根 logger 或特定 logger 记录一条初始化信息
    logger_instance = logging.getLogger("aigraphx")  # 获取我们应用的主 logger

    # 使用中国时间显示初始化信息
    utc_now_init = datetime.datetime.now(datetime.timezone.utc)
    china_now_init = utc_now_init.astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    )
    china_time_str_init = china_now_init.strftime("%Y-%m-%d %H:%M:%S")

    # 记录一条信息表明日志已成功初始化，并指明日志文件的位置
    logger_instance.info(
        f"Logging system initialized successfully using dictConfig. Current China Standard Time: {china_time_str_init}"
    )
    logger_instance.info(f"Application logs will be written to: {app_log_file}")
    logger_instance.info(
        f"Error logs (ERROR level and above) will be written to: {error_log_file}"
    )
    logger_instance.info(f"Access logs will be written to: {access_log_file}")
    # 这个函数不需要返回 logger 实例，因为之后可以通过 logging.getLogger("aigraphx") 获取
    # return logger_instance # 通常不需要返回
