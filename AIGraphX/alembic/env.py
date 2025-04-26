# -*- coding: utf-8 -*-
"""
Alembic 环境配置文件 (`env.py`)。

此文件在运行 `alembic` 命令时被执行，负责设置 Alembic 的运行环境，
主要包括：
1. 配置日志记录。
2. 获取数据库连接 URL (关键步骤)。
3. 设置 SQLAlchemy 的 MetaData 对象（如果使用模型自动生成迁移，本项目未使用）。
4. 提供运行迁移的两种模式：在线 (online) 和离线 (offline)。
"""

import os  # 导入 os 模块，用于路径操作和环境变量访问
import sys  # 导入 sys 模块，用于修改 Python 解释器的模块搜索路径
from logging.config import (
    fileConfig,
)  # 从 logging.config 导入 fileConfig，用于加载日志配置
from typing import Optional, Union, cast  # 导入类型提示
from dotenv import load_dotenv  # 导入 python-dotenv 库，用于从 .env 文件加载环境变量

from sqlalchemy import (
    engine_from_config,
)  # 从 SQLAlchemy 导入，用于根据配置创建数据库引擎
from sqlalchemy import pool  # 从 SQLAlchemy 导入连接池选项 (这里使用 NullPool)

from alembic import context  # 导入 Alembic 的上下文对象，用于配置和执行迁移

# --- 将项目根目录添加到 sys.path ---
# 确保 Alembic 运行时能找到项目自身的包（例如 aigraphx），以便后续可能导入模型或配置。
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # 计算项目根目录路径
if project_root not in sys.path:  # 如果根目录不在搜索路径中
    sys.path.insert(0, project_root)  # 将其添加到搜索路径列表的开头

# --- 显式加载 .env 文件 ---
# 优先尝试从项目根目录的 .env 文件加载环境变量。
dotenv_path = os.path.join(project_root, ".env")  # 构建 .env 文件路径
if os.path.exists(dotenv_path):  # 检查文件是否存在
    print(f"正在从以下路径加载环境变量: {dotenv_path}")
    # 加载 .env 文件，override=False 表示不覆盖已存在的环境变量
    load_dotenv(dotenv_path=dotenv_path, override=False)
else:
    print(f"警告: 在 {dotenv_path} 未找到 .env 文件")

# --- 获取数据库连接 URL (DATABASE_URL) ---
# 优先级顺序：环境变量 > 应用配置 (config.py)
DB_URL: Optional[str] = os.getenv("DATABASE_URL")  # 首先尝试从环境变量获取
USING_ENV_URL = False  # 标记是否使用了环境变量中的 URL

if DB_URL:  # 如果环境变量中设置了 DATABASE_URL
    print("正在使用环境变量中的 DATABASE_URL。")
    USING_ENV_URL = True
else:  # 如果环境变量中没有设置
    # 尝试从应用的配置文件 (aigraphx.core.config) 加载
    print("未找到 DATABASE_URL 环境变量，正在从配置文件加载...")
    try:
        # 在添加了项目根目录到 sys.path 后，导入应用配置模块
        from aigraphx.core import config

        # 检查配置模块的结构，以兼容不同的配置方式
        # 方式一：使用 Pydantic Settings (config.settings.DATABASE_URL)
        if hasattr(config, "settings") and hasattr(config.settings, "DATABASE_URL"):
            # 确保获取到的 URL 是字符串类型
            DB_URL = str(config.settings.DATABASE_URL)
            print("已从 config.settings 加载 DB_URL。")
        # 方式二：直接作为模块属性 (config.DATABASE_URL)
        elif hasattr(config, "DATABASE_URL"):
            DB_URL = str(
                getattr(config, "DATABASE_URL", "")
            )  # 获取属性值，确保是字符串
            print("已从 config 模块属性加载 DB_URL。")
        else:
            # 如果两种结构都找不到
            DB_URL = None
            print("在预期的配置结构中未找到 DATABASE_URL。")

        if not DB_URL:  # 如果从配置加载后 DB_URL 仍然为空
            print("警告：从配置加载的 DATABASE_URL 为空或未找到。")

    except (ImportError, AttributeError) as e:  # 处理导入错误或属性错误
        print(f"警告：无法从配置加载 DATABASE_URL: {e}")
        DB_URL = None  # 加载失败，确保 DB_URL 为 None

# --- 最终检查和 URL 调整 ---
if not DB_URL:  # 如果经过环境变量和配置检查后，DB_URL 仍然无效
    # 抛出 ImportError，因为没有数据库 URL 无法进行迁移
    raise ImportError("DATABASE_URL 未在环境变量中设置，也无法从配置中加载。")

# 确保 DB_URL 是字符串类型
DB_URL = str(DB_URL)

# --- 调整数据库 URL scheme 以适配 psycopg (v3) ---
# SQLAlchemy 推荐为 PostgreSQL + psycopg v3 使用 'postgresql+psycopg' 方案
if DB_URL.startswith("postgresql://"):  # 如果是旧的 scheme
    adjusted_url = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    if adjusted_url != DB_URL:  # 仅在实际调整时打印
        print(f"已将 DB_URL scheme 调整为: {adjusted_url}")
    DB_URL = adjusted_url
elif DB_URL.startswith("postgresql+psycopg://"):  # 如果已经是正确的 scheme
    pass  # 不需要调整
else:  # 如果是其他 scheme (例如 sqlite 等)
    print(
        f"警告：DB_URL scheme ('{DB_URL.split('://')[0]}://') 可能与 psycopg v3 不兼容。"
    )


# --- Alembic 配置 ---
# context.config 是 Alembic 的 Config 对象，提供对 alembic.ini 文件中配置的访问
alembic_config = context.config

# --- 以编程方式设置数据库 URL ---
# 使用从环境或配置中获取并调整后的 DB_URL，覆盖 alembic.ini 中的 sqlalchemy.url 设置
alembic_config.set_main_option("sqlalchemy.url", DB_URL)
print(f"Alembic 配置使用数据库 URL: {DB_URL}")

# --- 配置日志记录 ---
# 根据 alembic.ini 文件中的 [loggers], [handlers], [formatters] 部分配置 Python 日志
if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# --- 配置 target_metadata (用于模型自动生成迁移) ---
# 如果使用 SQLAlchemy 模型并希望 Alembic 自动检测模型变化生成迁移脚本，
# 需要将模型的 MetaData 对象赋值给 target_metadata。
# 例如: from myapp.models import Base; target_metadata = Base.metadata
# 本项目目前使用原生 SQL 编写迁移脚本，所以 target_metadata 设为 None。
target_metadata = None  # type: ignore # 明确设为 None，忽略类型检查器的可能警告


# 其他配置值可以从 config 对象获取，例如：
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


# --- 定义离线迁移模式 ---
def run_migrations_offline() -> None:
    """
    在“离线”模式下运行迁移。

    这种模式不直接连接数据库，而是将生成的 SQL DDL 语句打印到标准输出
    或写入到一个 SQL 文件中。这对于无法直接访问生产数据库但需要生成
    升级脚本的场景很有用。

    context.configure() 使用数据库 URL 配置上下文，但不创建引擎。
    literal_binds=True 表示将参数直接嵌入 SQL 语句中。
    context.execute() 在这种模式下会将 SQL 字符串输出。
    """
    print("正在以离线模式运行迁移...")
    context.configure(
        url=DB_URL,  # 直接使用数据库 URL 字符串
        target_metadata=target_metadata,  # 传递 MetaData (虽然是 None)
        literal_binds=True,  # 将绑定参数直接写入 SQL 字符串
        dialect_opts={"paramstyle": "named"},  # 方言选项
    )

    # 开始一个“事务”（在离线模式下主要是组织输出）
    with context.begin_transaction():
        # 执行迁移脚本中的 SQL 命令（将打印到输出）
        context.run_migrations()
    print("离线模式迁移完成。")


# --- 定义在线迁移模式 ---
def run_migrations_online() -> None:
    """
    在“在线”模式下运行迁移。

    这种模式会创建 SQLAlchemy 引擎，建立到数据库的实际连接，
    并在数据库事务中执行迁移命令。这是最常用的模式。
    """
    print("正在以在线模式运行迁移...")
    # 从配置字典创建 SQLAlchemy 引擎
    # 使用 NullPool 避免 Alembic 持有连接池
    connectable = engine_from_config(
        {"sqlalchemy.url": DB_URL},  # 直接传递包含 URL 的配置字典
        prefix="sqlalchemy.",  # 指定配置前缀 (虽然这里只用了 url)
        poolclass=pool.NullPool,  # 不使用连接池
    )

    # 建立数据库连接
    with connectable.connect() as connection:
        print("数据库连接已建立。")
        # 配置 Alembic 上下文，使用实际的数据库连接
        context.configure(connection=connection, target_metadata=target_metadata)

        # 开始数据库事务
        print("开始数据库事务...")
        with context.begin_transaction():
            print("正在执行迁移...")
            # 执行迁移脚本中的数据库操作
            context.run_migrations()
            print("迁移执行完毕。")
        print("数据库事务已提交。")
    print("在线模式迁移完成。")


# --- 根据 Alembic 的运行模式选择执行函数 ---
if context.is_offline_mode():  # 检查当前是否处于离线模式
    run_migrations_offline()  # 执行离线模式函数
else:  # 否则执行在线模式函数
    run_migrations_online()
