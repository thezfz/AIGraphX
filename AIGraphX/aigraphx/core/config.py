# aigraphx/core/config.py

"""
全局配置模块 (Global Configuration Module)

功能 (Function):
这个文件是整个 AIGraphX 后端应用的配置中心。它负责：
1. 定义所有配置项，例如数据库连接信息、API 密钥、Faiss 索引路径、日志级别等。
2. 使用 Pydantic Settings 库从环境变量和 `.env` 文件中加载配置值。
3. 对加载的配置进行类型检查和基本的验证。
4. 提供一个全局可访问的 `settings` 对象，供应用的其他模块导入和使用。

交互 (Interaction):
- 读取 (Reads): `.env` 文件 (如果存在) 和系统环境变量。
- 被导入 (Imported by):
    - `aigraphx.main`: 用于获取项目名称、API 前缀、日志级别等基本应用配置。
    - `aigraphx.core.db`: 用于获取数据库连接 URL、用户名、密码、连接池大小等，以便初始化数据库连接。
    - `aigraphx.core.logging_config`: 用于获取日志级别，配置日志记录器。
    - `aigraphx.repositories.*`: 可能会读取特定的仓库配置，例如 Faiss 索引路径。
    - `aigraphx.vectorization.embedder`: 用于获取 Sentence Transformer 模型名称和设备配置。
    - `aigraphx.services.*`: 可能会读取服务所需的配置项。
    - `aigraphx.api.v1.dependencies`: 用于在创建 Repository 或 Service 实例时可能需要获取配置。
    - `scripts/*`: 独立脚本需要导入此模块来加载运行所需的配置。
    - `tests/*`: 测试代码需要导入此模块，并经常使用 `monkeypatch` 来模拟或覆盖特定的测试配置。

设计原则 (Design Principles):
- **集中管理 (Centralized Management):** 所有配置项集中在此定义和加载，避免分散在代码各处。
- **环境变量优先 (Environment Variable First):** 优先从环境变量加载，方便在不同环境（开发、测试、生产）中覆盖配置。
- **类型安全 (Type Safety):** 使用 Pydantic 进行类型提示和验证，减少因配置错误导致的运行时问题。
- **敏感信息处理 (Sensitive Information Handling):** API 密钥等敏感信息通过环境变量或 `.env` 文件加载，`.env` 文件应被 `.gitignore` 排除，防止泄露。
- **明确的测试配置 (Explicit Test Configuration):** 为测试环境定义了专门的 `TEST_*` 配置项，方便在测试时使用独立的数据库和资源。
"""

# 导入标准库
import os  # 用于访问环境变量和操作系统相关功能，例如检查文件是否存在、获取路径
from pathlib import Path  # 提供面向对象的路径操作 (虽然在此版本中直接使用 os.path)
from typing import (
    Optional,
    List,
    Literal,
)  # 用于类型提示，Optional表示可选，List表示列表，Literal限制变量只能是指定的几个值之一

# 导入第三方库
# from dotenv import load_dotenv # 不再需要手动调用 load_dotenv，Pydantic Settings 会处理
from loguru import logger  # 用于记录日志信息，比标准 logging 库更方便易用

# 从 Pydantic 导入用于数据验证和类型提示的工具
# Field 用于为模型字段添加元数据（如默认值、别名）
# field_validator 用于定义自定义验证逻辑
# ValidationInfo 包含验证过程中的上下文信息
from pydantic import Field, field_validator, ValidationInfo

# 从 pydantic-settings 导入核心类
# BaseSettings 是所有配置类的基类，提供了从环境变量和 .env 文件加载配置的功能
# SettingsConfigDict 用于配置 BaseSettings 的行为，例如指定 .env 文件路径
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- 环境变量检查与路径计算 ---

# 检查是否在 pytest 测试环境中运行
# Pytest 在运行时通常会设置 'PYTEST_RUNNING' 环境变量
IS_PYTEST = os.getenv("PYTEST_RUNNING") == "1"
logger.info(f"Is running under pytest? {IS_PYTEST}")

# 计算项目根目录的绝对路径
# __file__ 指的是当前文件(config.py)的路径
# os.path.dirname(__file__) 获取 config.py 所在的目录 (core)
# os.path.join(..., "..", "..") 向上两级目录，到达项目根目录 AIGraphX/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 计算 .env 文件的绝对路径
# 默认情况下，.env 文件应该放在项目根目录下
dotenv_path = os.path.join(project_root, ".env")

# 记录计算出的 .env 文件路径和文件是否存在的信息，用于调试
logger.info(f"Calculated project root path: {project_root}")
logger.info(f"Calculated .env path for Pydantic Settings: {dotenv_path}")
logger.info(f"Does .env file exist at calculated path? {os.path.exists(dotenv_path)}")

# --- 配置模型定义 ---


# 定义一个继承自 BaseSettings 的 Settings 类
# Pydantic 会自动读取环境变量和 .env 文件，并填充到类的字段中
# 字段名与环境变量名默认是大小写不敏感匹配的
class Settings(BaseSettings):
    """
    应用配置模型 (Application Settings Model)

    使用 Pydantic BaseSettings 定义和加载所有配置项。
    字段的类型提示确保了配置值的正确性。
    `Field` 用于指定默认值和环境变量别名 (`alias`)。
    `model_config` 用于配置 Pydantic Settings 的行为。
    """

    # 使用 model_config 显式配置 Pydantic Settings 的行为
    model_config = SettingsConfigDict(
        # 指定要加载的 .env 文件路径
        env_file=dotenv_path
        if os.path.exists(dotenv_path)
        else None,  # 仅当文件存在时才加载
        # 指定 .env 文件的编码
        env_file_encoding="utf-8",
        # 如何处理额外的环境变量或 .env 文件中的字段
        # 'ignore': 忽略未在模型中定义的额外字段
        # 'allow': 允许额外字段，但不会被加载到模型实例中
        # 'forbid': 如果存在额外字段，则抛出验证错误
        extra="ignore",
    )

    # --- 常规配置 (General Settings) ---
    project_name: str = Field(default="AIGraphX", alias="PROJECT_NAME")  # 项目名称
    api_v1_str: str = Field(
        default="/api/v1", alias="API_V1_STR"
    )  # API 版本 1 的路径前缀
    # 日志级别，使用 Literal 限制为指定的几个值
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    # 应用运行环境，例如 'development', 'staging', 'production', 'test'
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # --- 主数据库连接配置 (Primary Database URLs) ---
    # 使用 Optional[str] 表示这些配置可以是 None (如果没有在环境或 .env 中设置)
    # PostgreSQL 数据库连接 URL (遵循 SQLAlchemy 标准格式)
    # 例如: postgresql+asyncpg://user:password@host:port/database
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    # Neo4j 图数据库连接 URI
    # 例如: neo4j://host:port 或 bolt://host:port 或 neo4j+s://host:port (加密连接)
    neo4j_uri: Optional[str] = Field(default=None, alias="NEO4J_URI")
    # Neo4j 用户名
    neo4j_username: Optional[str] = Field(default="neo4j", alias="NEO4J_USER")
    # Neo4j 密码
    neo4j_password: Optional[str] = Field(default=None, alias="NEO4J_PASSWORD")
    # Neo4j 逻辑数据库名称 (社区版通常是 'neo4j')
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # --- 数据库连接池大小 (Primary Database Pool Sizes) ---
    # PostgreSQL 连接池的最小连接数
    pg_pool_min_size: int = Field(default=1, alias="PG_POOL_MIN_SIZE")
    # PostgreSQL 连接池的最大连接数
    pg_pool_max_size: int = Field(default=10, alias="PG_POOL_MAX_SIZE")

    # --- 文本嵌入模型配置 (Embedding Settings) ---
    # Sentence Transformer 模型的名称或路径 (Hugging Face Hub 模型名称或本地路径)
    sentence_transformer_model: str = Field(
        default="intfloat/multilingual-e5-large", alias="SENTENCE_TRANSFORMER_MODEL"
    )
    # 运行嵌入模型的设备 ('cpu', 'cuda', 'mps' 等)
    embedder_device: str = Field(default="cpu", alias="EMBEDDER_DEVICE")

    # --- Faiss 索引配置 (Faiss Settings) ---
    # Faiss 向量索引文件的存储路径 (存储论文向量)
    faiss_index_path: str = Field(
        default=str(Path(project_root) / "data" / "faiss_index.bin"),
        alias="FAISS_INDEX_PATH",
    )
    # Faiss 论文 ID 到其在索引中位置的映射文件的存储路径
    faiss_mapping_path: str = Field(
        default=str(Path(project_root) / "data" / "papers_faiss_ids.json"),
        alias="FAISS_MAPPING_PATH",
    )
    # Faiss 向量索引文件的存储路径 (存储模型向量)
    models_faiss_index_path: str = Field(
        default=str(Path(project_root) / "data" / "models_faiss.index"),
        alias="MODELS_FAISS_INDEX_PATH",
    )
    # Faiss 模型 ID 到其在索引中位置的映射文件的存储路径
    models_faiss_mapping_path: str = Field(
        default=str(Path(project_root) / "data" / "models_faiss_ids.json"),
        alias="MODELS_FAISS_MAPPING_PATH",
    )
    # 构建 Faiss 索引时，每次处理的数据批次大小
    build_faiss_batch_size: int = Field(default=128, alias="BUILD_FAISS_BATCH_SIZE")

    # --- API 密钥 (API Keys) ---
    # 外部服务 API 密钥，默认值为 None，应通过环境变量或 .env 文件安全提供
    # Hugging Face API 密钥
    hf_api_key: Optional[str] = Field(default=None, alias="HF_API_KEY")
    # GitHub API 密钥
    github_api_key: Optional[str] = Field(default=None, alias="GITHUB_API_KEY")
    # Papers with Code API 密钥
    pwc_api_key: Optional[str] = Field(default=None, alias="PWC_API_KEY")

    # --- 测试环境特定配置覆盖 (Test Specific Overrides) ---
    # 这些字段用于在测试环境中加载以 'TEST_' 开头的环境变量
    # Pydantic Settings 会首先加载常规配置，然后如果存在 TEST_* 变量，会用它们的值覆盖对应非 TEST_ 前缀的字段
    # 例如，如果在测试环境中设置了 TEST_DATABASE_URL，那么 settings.database_url 将会被覆盖
    # 注意：这种覆盖机制是 Pydantic Settings 的内置行为，我们只需定义这些字段即可。
    # 测试环境的 PostgreSQL 数据库 URL
    test_database_url: Optional[str] = Field(default=None, alias="TEST_DATABASE_URL")
    # 测试环境的 Neo4j 连接 URI
    test_neo4j_uri: Optional[str] = Field(default=None, alias="TEST_NEO4J_URI")
    # 测试环境的 Neo4j 密码
    test_neo4j_password: Optional[str] = Field(
        default=None, alias="TEST_NEO4J_PASSWORD"
    )
    # 测试环境的 Neo4j 数据库名称 (注意社区版限制，通常仍为 'neo4j')
    test_neo4j_database: Optional[str] = Field(
        default="neo4j", alias="TEST_NEO4J_DATABASE"
    )
    # 测试环境的 Faiss 论文索引路径
    test_faiss_paper_index_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_PAPER_INDEX_PATH"
    )
    # 测试环境的 Faiss 论文映射路径
    test_faiss_paper_mapping_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_PAPER_MAPPING_PATH"
    )
    # 测试环境的 Faiss 模型索引路径
    test_faiss_model_index_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_MODEL_INDEX_PATH"
    )
    # 测试环境的 Faiss 模型映射路径
    test_faiss_model_mapping_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_MODEL_MAPPING_PATH"
    )

    # --- 自定义验证器 (Validation) ---
    # 使用 @field_validator 装饰器定义验证函数
    # "database_url", "neo4j_uri": 指定这个验证器应用于哪些字段
    # mode="before": 指定验证器在 Pydantic 进行类型转换和默认值处理 *之前* 运行
    @field_validator("database_url", "neo4j_uri", mode="before")
    @classmethod  # 验证器方法需要是类方法
    def check_not_empty(
        cls, value: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """
        确保必要的 URL 字段如果被设置了，其值不是空字符串。
        如果环境变量被设置为空字符串 (e.g., DATABASE_URL=""),
        Pydantic 默认会将其视为空字符串，而不是 None。
        这个验证器将空字符串转换回 None，以符合 Optional[str] 的预期语义。

        Args:
            value (Optional[str]): 待验证的字段值。
            info (ValidationInfo): 包含字段名称等上下文信息。

        Returns:
            Optional[str]: 如果值是空字符串则返回 None，否则返回原值。
        """
        # info.field_name 包含了当前被验证的字段名 ("database_url" 或 "neo4j_uri")
        if value == "":
            logger.warning(
                f"Configuration field '{info.field_name}' was set to an empty string. "
                f"Treating as None (not set)."
            )
            return None  # 将空字符串转换成 None
        return value  # 如果不是空字符串，保持原样


# --- 实例化配置对象 ---
# 创建 Settings 类的一个实例。
# 在实例化时，Pydantic Settings 会自动执行加载和验证逻辑。
# 这个 `settings` 对象将作为全局配置，被应用的其他部分导入和使用。
try:
    settings = Settings()
    logger.info("Settings loaded successfully.")
    # 可以在这里添加更详细的日志，例如打印部分非敏感配置
    logger.debug(f"Project Name: {settings.project_name}")
    logger.debug(f"Environment: {settings.environment}")
    logger.debug(f"Log Level: {settings.log_level}")
    logger.debug(f"Embedder Model: {settings.sentence_transformer_model}")

    # 如果在测试环境中，记录测试数据库配置的使用情况
    if IS_PYTEST:
        logger.info(
            "Running in pytest environment. Applying test configurations if set."
        )
        # 检查 database_url 是否为字符串再进行切片
        db_url_log = "not set or empty"
        if isinstance(settings.database_url, str) and settings.database_url:
            db_url_log = (
                f"{settings.database_url[:15]}..."  # 只记录部分 URL 防止敏感信息泄露
            )
        logger.info(
            f"Test override applied: Using DATABASE_URL from TEST_DATABASE_URL: {db_url_log}"
        )

        if settings.test_neo4j_uri:
            logger.info(
                f"Test override applied: Using NEO4J_URI from TEST_NEO4J_URI: {settings.neo4j_uri}"
            )
            if settings.test_neo4j_password:
                # Log the actual used neo4j_password if test_neo4j_password was set
                logger.info(
                    f"Test override applied: Using NEO4J_PASSWORD from TEST_NEO4J_PASSWORD (set). Password itself is not logged."
                )
            # Log the actual used neo4j_database if test_neo4j_database was set (or its default)
            logger.info(
                f"Test override applied: Using NEO4J_DATABASE from TEST_NEO4J_DATABASE: {settings.neo4j_database}"
            )
        if settings.test_faiss_paper_index_path:
            # Log the actual used path if test_faiss_paper_index_path was set
            logger.info(
                f"Test override applied: Using FAISS_INDEX_PATH from TEST_FAISS_PAPER_INDEX_PATH: {settings.faiss_index_path}"
            )
        if settings.test_faiss_paper_mapping_path:
            logger.info(
                f"Test override applied: Using FAISS_MAPPING_PATH from TEST_FAISS_PAPER_MAPPING_PATH: {settings.faiss_mapping_path}"
            )
        if settings.test_faiss_model_index_path:
            logger.info(
                f"Test override applied: Using MODELS_FAISS_INDEX_PATH from TEST_FAISS_MODEL_INDEX_PATH: {settings.models_faiss_index_path}"
            )
        if settings.test_faiss_model_mapping_path:
            logger.info(
                f"Test override applied: Using MODELS_FAISS_MAPPING_PATH from TEST_FAISS_MODEL_MAPPING_PATH: {settings.models_faiss_mapping_path}"
            )


except Exception as e:
    # 如果在加载或验证配置时发生错误，记录严重错误并可能需要退出
    logger.critical(f"Failed to load or validate settings: {e}", exc_info=True)
    # 在某些情况下，可能希望在这里引发异常或退出程序
    # raise RuntimeError(f"Failed to initialize settings: {e}") from e


# --- 记录敏感信息加载状态 ---
# 检查关键的 API 密钥是否已配置，如果未配置则发出警告日志
# 这有助于在启动时快速发现配置问题，但不会泄露密钥本身
if not settings.hf_api_key:
    logger.warning("HF_API_KEY is not set in environment variables or .env file.")
# else: # 可以选择性地添加成功加载的日志，但不建议在生产环境打印过多信息
#     logger.info("HF_API_KEY found.")

if not settings.github_api_key:
    logger.warning("GITHUB_API_KEY is not set in environment variables or .env file.")
# else:
#     logger.info("GITHUB_API_KEY found.")

if not settings.pwc_api_key:
    logger.warning("PWC_API_KEY is not set in environment variables or .env file.")
# else:
#      logger.info("PWC_API_KEY found.")

# --- 清理 ---
# 确保在这个文件末尾没有定义任何额外的变量，
# 所有配置都应该通过 settings 对象访问。
