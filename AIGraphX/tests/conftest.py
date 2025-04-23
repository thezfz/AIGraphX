# tests/conftest.py

"""
文件目的：定义 Pytest Fixtures 和测试配置

概述：
这个文件是 Pytest 的核心配置文件之一，名为 `conftest.py`。
Pytest 会自动发现并使用这个文件中定义的 Fixtures (测试装置/夹具)。
Fixtures 是一种强大的机制，用于提供测试所需的数据、资源、环境或模拟对象，
并且可以控制它们的生命周期（例如，每个测试函数创建一个新的，或者整个测试会话只创建一个）。

本文件定义了 AIGraphX 后端测试所需的各种 Fixtures，包括：
- **测试配置**: 加载测试环境特定的配置 (如数据库连接 URL、Faiss 文件路径)。
- **数据库连接与管理**:
    - 创建连接到**测试** PostgreSQL 数据库的连接池 (`db_pool`)。
    - 创建连接到**测试** Neo4j 数据库的驱动程序 (`neo4j_driver`)。
    - 在测试运行前应用 Alembic 数据库迁移 (`apply_migrations`)。
    - 提供清理数据库的机制（例如，在每个测试前清空数据）。
- **仓库实例**: 提供连接到测试数据库的 `PostgresRepository` (`repository`) 和 `Neo4jRepository` (`neo4j_repo_fixture`) 实例。
- **模拟对象**:
    - 提供模拟的 `GraphService` (`mock_graph_service_fixture`)，用于 API 层的单元测试。
    - 提供模拟的 `FaissRepository` (`mock_faiss_repository`)，用于需要模拟 Faiss 交互的测试。
    - 提供模拟的 `TextEmbedder` (`mock_embedder`)，用于模拟文本嵌入。
- **测试应用实例**: 创建一个配置了测试设置和依赖覆盖的 FastAPI 应用实例 (`test_app`)，用于 API 集成测试。
- **HTTP 测试客户端**: 提供一个异步 HTTP 客户端 (`client`)，用于向测试应用发送请求。
- **辅助 Fixtures**: 如会话范围的锁 (`db_cleanup_lock`)、临时 Faiss 文件 (`temp_faiss_files`)、预加载的文本嵌入器 (`loaded_text_embedder`) 等。

测试策略影响：
此文件是实现 "测试奖杯" 策略的关键，特别是对于**集成测试**：
- 通过 `db_pool`, `neo4j_driver`, `repository`, `neo4j_repo_fixture` 支持了仓库层与真实测试数据库的交互。
- 通过 `test_app` 和 `client` 支持了 API 层的集成测试（通常 Mock 服务层）。
- 通过 `apply_migrations` 确保了测试数据库模式与代码一致。
- 通过数据库清理逻辑保证了测试之间的隔离性。

使用方法：
测试函数可以通过将 Fixture 名称作为参数来请求使用它们。Pytest 会自动查找并注入相应的 Fixture 实例。
例如: `async def test_something(client: AsyncClient, repository: PostgresRepository): ...`

注意事项：
- 运行依赖数据库的测试需要正确配置测试数据库的连接信息（通常在 `.env` 或 `.env.test` 文件中，并通过 `test_settings` fixture 加载）。
- 确保测试数据库服务（PostgreSQL, Neo4j）正在运行。
- 理解 Fixture 的 `scope` (作用域) 很重要，它决定了 Fixture 的生命周期和共享范围（`function`, `class`, `module`, `session`）。
"""

# 导入测试框架和标准库
import pytest  # Pytest 核心框架
import pytest_asyncio  # Pytest 异步支持插件
from fastapi import FastAPI  # FastAPI 应用类
from functools import partial  # 用于创建偏函数，固定函数的部分参数
import httpx  # HTTP 客户端库，用于 API 测试
from httpx import AsyncClient, ASGITransport  # 异步 HTTP 客户端和 ASGI 传输适配器
from unittest.mock import AsyncMock, MagicMock  # 单元测试模拟工具 (AsyncMock 用于异步，MagicMock 通用)
from typing import (  # 类型提示工具
    AsyncGenerator,  # 异步生成器类型 (用于异步 Fixture)
    Generator,  # 同步生成器类型 (用于同步 Fixture)
    Dict,
    Any,
    Optional,
    Tuple,
    Union,
)
import os  # 操作系统交互功能 (路径操作, 环境变量)
import json  # JSON 数据处理
import datetime  # 日期时间处理
import asyncio  # 异步 I/O 库 (例如，用于锁)
import subprocess  # 用于运行子进程 (例如，执行 Alembic 命令)
from psycopg_pool import AsyncConnectionPool  # PostgreSQL 异步连接池
from psycopg.rows import dict_row  # psycopg 行工厂，使结果返回为字典
from dotenv import load_dotenv  # 用于从 .env 文件加载环境变量
import logging  # 日志记录库
from pytest_mock import MockerFixture  # pytest-mock 插件提供的类型提示
import sys  # 系统相关功能 (例如，平台检查)
import faiss  # Faiss 库，用于向量相似性搜索
import numpy as np  # NumPy 库，用于处理向量数据
from pathlib import Path  # 面向对象的路径操作
from unittest.mock import patch  # 模拟工具，用于替换对象或方法
from starlette.datastructures import State  # Starlette 的状态对象，FastAPI 应用使用它存储共享状态
from contextlib import asynccontextmanager  # 用于创建异步上下文管理器
import inspect  # 用于获取运行时信息，如此处的调用者函数名

# 导入 Neo4j 驱动相关类
from neo4j import AsyncGraphDatabase, AsyncDriver, basic_auth  # 异步驱动、基础认证

# --- 应用程序组件导入 ---
# 不直接导入主应用的 app 实例，以避免过早加载配置或初始化
# from aigraphx.main import app as main_app

# 导入构建测试应用所需的组件
from aigraphx.core.db import lifespan  # 应用生命周期管理函数 (启动/关闭资源)
from aigraphx.core.config import Settings  # 配置模型类
from aigraphx.api.v1.api import api_router as api_v1_router  # API v1 路由

# 导入服务类 (用于模拟时的类型规范 spec)
from aigraphx.services.graph_service import GraphService
from aigraphx.services.search_service import SearchService # 添加 SearchService 导入

# 导入仓库类 (用于集成测试 Fixture)
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# 导入原始的 FastAPI 依赖项获取函数 (用于覆盖或测试)
# from aigraphx.api.v1 import deps # 路径不正确
from aigraphx.api.v1.dependencies import get_graph_service, get_search_service, get_postgres_repository, get_neo4j_repository, get_faiss_repository_papers, get_faiss_repository_models, get_embedder, get_app_state # 正确路径

# 导入被测试的 lifespan 函数
from aigraphx.core.db import lifespan

# 导入需要模拟的类 (用于 mock fixture)
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.vectorization.embedder import TextEmbedder

# 导入 asgi-lifespan 用于手动管理 FastAPI 生命周期 (主要在 client fixture 中)
from asgi_lifespan import LifespanManager

# --- 日志设置 ---
# 获取一个名为 'conftest' 的日志记录器实例，用于记录 conftest.py 文件自身的日志
logger = logging.getLogger(__name__)

# 添加额外的日志记录器，用于在模块加载时记录关键的原始环境变量值，帮助调试配置问题
logger_conftest_top = logging.getLogger("conftest_top")
raw_test_neo4j_db_env = os.getenv("TEST_NEO4J_DATABASE")
logger_conftest_top.critical(
    f"[CONTEST TOP] 模块加载时原始 os.getenv('TEST_NEO4J_DATABASE'): '{raw_test_neo4j_db_env}'"
)

# --- 尽早加载 .env 文件 ---
# 目的是为了能够读取测试数据库的 URL 等信息，供后续 Fixture 使用。
# 定位 .env 文件路径 (通常在项目根目录，即 conftest.py 的上两级目录)
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
# 加载 .env 文件中的环境变量，如果存在的话。不会覆盖已存在的环境变量。
load_dotenv(dotenv_path=dotenv_path)
# 从环境变量中读取测试数据库的 URL 和 Neo4j 连接信息
TEST_DB_URL_FROM_ENV = os.getenv("TEST_DATABASE_URL")
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI")
TEST_NEO4J_USER = os.getenv("TEST_NEO4J_USER", "neo4j") # 如果未设置，默认为 "neo4j"
TEST_NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD")

# --- 会话范围的异步锁 ---
# `scope="session"` 表示这个 Fixture 在整个测试会话期间只创建一次实例。
# 用于同步对共享资源的访问，如此处的数据库清理操作，防止并发问题。
@pytest.fixture(scope="session")
def db_cleanup_lock() -> asyncio.Lock:
    """提供一个会话范围的 asyncio.Lock，用于序列化数据库清理操作。"""
    logger.info("[db_cleanup_lock fixture] 正在创建会话范围的锁。")
    return asyncio.Lock()


# --- 会话范围的临时 Faiss 文件 Fixture ---
# `scope="session"` 保证了临时文件在整个测试会话中只创建一次，提高效率。
@pytest.fixture(scope="session")
def temp_faiss_files(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, str]:
    """
    创建临时的、有效的 Faiss 索引文件 (.index) 和 ID 映射文件 (.json) 用于测试。
    这些文件包含少量随机生成的向量和示例 ID 映射。

    Args:
        tmp_path_factory: Pytest 内置的用于创建临时目录的工厂 fixture。

    Returns:
        Dict[str, str]: 一个字典，包含指向创建的论文和模型 Faiss 文件路径的字符串。
                        键为 "paper_index", "paper_map", "model_index", "model_map"。
    """
    # 使用会话范围的临时目录工厂创建基础目录
    base_path = tmp_path_factory.mktemp("faiss_test_data")
    logger.info(f"[temp_faiss_files] 正在临时 Faiss 文件目录: {base_path}")

    dim = 384  # 示例嵌入向量维度，应与测试中使用的 embedder 匹配
    num_vectors = 5  # 创建少量示例向量

    # --- 创建论文 Faiss 文件 ---
    paper_index_path = base_path / "test_papers.index"
    paper_map_path = base_path / "test_papers_map.json"

    # 创建一个简单的 Faiss L2 索引
    try:
        paper_index = faiss.IndexFlatL2(dim)
        # 生成随机向量并添加到索引
        vectors = np.random.rand(num_vectors, dim).astype("float32")
        paper_index.add(vectors)
        # 将索引写入文件
        faiss.write_index(paper_index, str(paper_index_path))
        logger.info(f"[temp_faiss_files] 已创建临时论文索引: {paper_index_path}")
    except Exception as e:
        logger.error(f"创建临时论文 Faiss 索引失败: {e}")
        pytest.fail(f"创建临时论文 Faiss 索引失败: {e}")

    # 创建一个简单的 ID 映射 (Faiss 内部索引 -> 论文 paper_id (int))
    # 示例: Faiss 索引 0 对应 paper_id 100, 1 对应 101, ...
    paper_id_map = {
        i: i + 100 for i in range(num_vectors)
    }
    try:
        # 将映射写入 JSON 文件
        with open(paper_map_path, "w") as f:
            json.dump(paper_id_map, f)
        logger.info(f"[temp_faiss_files] 已创建临时论文映射: {paper_map_path}")
    except Exception as e:
        logger.error(f"创建临时论文 Faiss 映射失败: {e}")
        pytest.fail(f"创建临时论文 Faiss 映射失败: {e}")

    # --- 创建模型 Faiss 文件 ---
    model_index_path = base_path / "test_models.index"
    model_map_path = base_path / "test_models_map.json"

    # 创建一个简单的模型 Faiss 索引
    try:
        model_index = faiss.IndexFlatL2(dim)
        model_vectors = np.random.rand(num_vectors, dim).astype("float32")
        model_index.add(model_vectors)
        faiss.write_index(model_index, str(model_index_path))
        logger.info(f"[temp_faiss_files] 已创建临时模型索引: {model_index_path}")
    except Exception as e:
        logger.error(f"创建临时模型 Faiss 索引失败: {e}")
        pytest.fail(f"创建临时模型 Faiss 索引失败: {e}")

    # 创建一个简单的 ID 映射 (Faiss 内部索引 -> 模型 model_id (str))
    # 示例: Faiss 索引 0 对应 model_id "model_0", 1 对应 "model_1", ...
    model_id_map = {i: f"model_{i}" for i in range(num_vectors)}
    try:
        with open(model_map_path, "w") as f:
            json.dump(model_id_map, f)
        logger.info(f"[temp_faiss_files] 已创建临时模型映射: {model_map_path}")
    except Exception as e:
        logger.error(f"创建临时模型 Faiss 映射失败: {e}")
        pytest.fail(f"创建临时模型 Faiss 映射失败: {e}")

    # 返回包含所有文件路径的字典
    return {
        "paper_index": str(paper_index_path),
        "paper_map": str(paper_map_path),
        "model_index": str(model_index_path),
        "model_map": str(model_map_path),
    }


# --- 测试配置 Fixture (会话范围) ---
# `scope="session"` 保证整个测试会话使用同一份测试配置。
@pytest.fixture(scope="session")
def test_settings(temp_faiss_files: Dict[str, str]) -> Settings:
    """
    创建并返回一个 `Settings` 配置对象实例，该实例专门用于测试环境。
    它首先尝试从环境变量（可能由 .env 文件加载）中加载设置，
    然后使用临时的 Faiss 文件路径覆盖相关的配置项，并将环境标记为 "test"。
    同时，它会将主数据库 URL/URI 等配置指向测试数据库。

    Args:
        temp_faiss_files (Dict[str, str]): 由 `temp_faiss_files` fixture 提供的
                                            包含临时 Faiss 文件路径的字典。

    Returns:
        Settings: 配置好的用于测试的 Settings 对象。
    """
    # 1. 加载基础设置：Settings 类（基于 Pydantic BaseSettings）会自动从环境变量读取
    settings_from_env = Settings()
    # 记录加载后的初始值，用于调试配置覆盖逻辑
    logger.info(
        f"[test_settings fixture] Settings() 加载的初始值: "
        f"test_db_url='{settings_from_env.test_database_url}', "
        f"db_url='{settings_from_env.database_url}', "
        f"test_neo4j_uri='{settings_from_env.test_neo4j_uri}', "
        f"neo4j_uri='{settings_from_env.neo4j_uri}', "
        f"test_neo4j_pwd set={'***' if settings_from_env.test_neo4j_password else 'No'}, " # 不记录密码本身
        f"neo4j_pwd set={'***' if settings_from_env.neo4j_password else 'No'}, "
        f"test_neo4j_db='{settings_from_env.test_neo4j_database}', "
        f"neo4j_db='{settings_from_env.neo4j_database}'"
    )

    # 2. 覆盖数据库连接信息，优先使用测试专用环境变量
    # 覆盖 PostgreSQL URL
    original_db_url = settings_from_env.database_url
    # 如果 test_database_url 有值，则用它覆盖 database_url
    settings_from_env.database_url = (
        settings_from_env.test_database_url or settings_from_env.database_url
    )
    logger.info(
        f"[test_settings fixture] PostgreSQL URL 覆盖: "
        f"test='{settings_from_env.test_database_url}', "
        f"original='{original_db_url}', final='{settings_from_env.database_url}'"
    )

    # 覆盖 Neo4j URI
    original_neo4j_uri = settings_from_env.neo4j_uri
    settings_from_env.neo4j_uri = (
        settings_from_env.test_neo4j_uri or settings_from_env.neo4j_uri
    )
    logger.info(
        f"[test_settings fixture] Neo4j URI 覆盖: "
        f"test='{settings_from_env.test_neo4j_uri}', "
        f"original='{original_neo4j_uri}', final='{settings_from_env.neo4j_uri}'"
    )

    # 覆盖 Neo4j 密码
    original_neo4j_pwd = settings_from_env.neo4j_password
    settings_from_env.neo4j_password = (
        settings_from_env.test_neo4j_password or settings_from_env.neo4j_password
    )
    pwd_overridden = (
        original_neo4j_pwd != settings_from_env.neo4j_password
        and settings_from_env.test_neo4j_password is not None
    )
    logger.info(
        f"[test_settings fixture] Neo4j 密码覆盖: "
        f"test_pwd_set={settings_from_env.test_neo4j_password is not None}, "
        f"overridden={pwd_overridden}"
    )

    # 覆盖 Neo4j 数据库名称
    # 注意：Neo4j 社区版限制，测试通常只能使用默认的 'neo4j' 数据库。
    # 但这里保留覆盖逻辑，以防使用企业版或未来版本。
    original_neo4j_db = settings_from_env.neo4j_database
    if settings_from_env.test_neo4j_database:
        logger.info(
            f"[test_settings fixture] 正在使用 test_neo4j_database ('{settings_from_env.test_neo4j_database}') 覆盖 neo4j_database"
        )
        settings_from_env.neo4j_database = settings_from_env.test_neo4j_database
    else:
        # 如果 TEST_NEO4J_DATABASE 未设置，则保持 Settings() 加载的默认值（通常是 'neo4j'）
        logger.info(
            f"[test_settings fixture] test_neo4j_database 未设置或为空，保持原始 neo4j_database ('{original_neo4j_db}')。"
        )
        # 确保最终使用的 neo4j_database 不为 None 或空字符串，如果需要默认为 'neo4j'
        if not settings_from_env.neo4j_database:
             settings_from_env.neo4j_database = "neo4j"
             logger.info(f"[test_settings fixture] 由于 neo4j_database 为空, 设置为默认值 'neo4j'.")


    # 3. 覆盖 Faiss 文件路径
    settings_from_env.faiss_index_path = temp_faiss_files["paper_index"]
    settings_from_env.faiss_mapping_path = temp_faiss_files["paper_map"]
    settings_from_env.models_faiss_index_path = temp_faiss_files["model_index"]
    settings_from_env.models_faiss_mapping_path = temp_faiss_files["model_map"]

    # 4. 设置环境标记
    settings_from_env.environment = "test"

    # 5. （可选）验证必要的 Neo4j 配置是否存在
    if settings_from_env.neo4j_uri and not settings_from_env.neo4j_password:
        logger.warning("Neo4j URI 已设置，但密码未设置。")
        # 这里可以决定是跳过相关测试还是依赖默认密码等

    # 记录最终生效的 Neo4j 数据库名称
    logger.info(
        f"[test_settings fixture] 生效的 Neo4j 数据库名称: '{settings_from_env.neo4j_database}'"
    )

    return settings_from_env


# --- 会话范围的文本嵌入器 Fixture ---
# `scope="session"` 避免在每个测试中重复加载耗时的模型。
@pytest.fixture(scope="session")
def loaded_text_embedder(
    test_settings: Settings,  # 依赖测试配置
) -> Generator[TextEmbedder, None, None]:
    """
    在每个测试会话中初始化并加载一次 TextEmbedder 模型。
    使用 `test_settings` 中的配置。
    此 Fixture 可被需要嵌入器的测试直接使用，或通过 FastAPI 依赖覆盖注入到测试应用中。

    Args:
        test_settings (Settings): 测试配置对象。

    Yields:
        TextEmbedder: 加载好模型的 TextEmbedder 实例。
    """
    logger.info(
        "[loaded_text_embedder fixture] 正在为测试会话初始化 TextEmbedder..."
    )
    # 确保模型名称配置存在
    if not test_settings.sentence_transformer_model:
        logger.error(
            "[loaded_text_embedder fixture] test_settings 中未设置 sentence_transformer_model。"
        )
        pytest.fail(
            "测试设置中需要 sentence_transformer_model 但未找到。"
        )

    # 创建 TextEmbedder 实例，传入配置
    embedder = TextEmbedder(
        model_name=test_settings.sentence_transformer_model,
        # cache_dir=test_settings.embedding_model_cache_dir, # Settings 中无此属性，移除
        device=test_settings.embedder_device, # 使用配置的设备 (cpu, cuda)
    )
    try:
        # TextEmbedder 的 __init__ 方法会自动调用 _load_model() 加载模型
        # 无需显式调用 embedder.load_model()

        # 检查模型是否已成功加载 (即 self.model 是否被赋值)
        if embedder.model is not None:
            logger.info(
                f"[loaded_text_embedder fixture] TextEmbedder 模型 '{test_settings.sentence_transformer_model}' 已成功为会话初始化。"
            )
            # 使用 yield 将加载好的 embedder 提供给测试
            yield embedder
        else:
            # 如果模型仍为 None，说明 _load_model() 内部失败了（应该已经记录了错误）
            logger.error(
                "[loaded_text_embedder fixture] 初始化后 embedder.model 仍为 None。请检查之前的日志以获取加载错误信息。"
            )
            pytest.fail(
                "在会话范围的 Fixture 中初始化 TextEmbedder 模型失败 (model 属性为 None)。"
            )

    except Exception as e:
        # 捕获 TextEmbedder 初始化过程中可能发生的其他异常
        logger.exception(
            f"[loaded_text_embedder fixture] TextEmbedder 初始化过程中出错: {e}"
        )
        pytest.fail(
            f"在会话范围的 Fixture 中 TextEmbedder 初始化失败: {e}"
        )


# --- 会话范围的测试 FastAPI 应用实例 Fixture ---
# `scope="session"` 确保所有测试使用同一个配置好的 FastAPI 应用实例。
@pytest.fixture(scope="session")
def test_app(
    test_settings: Settings,  # 依赖测试配置
    loaded_text_embedder: TextEmbedder,  # 依赖预加载的嵌入器
) -> FastAPI:
    """
    为整个测试会话创建一个 FastAPI 应用实例。
    - 使用 `test_settings` 进行配置。
    - 使用 `lifespan` 管理应用的启动和关闭事件（如数据库连接）。
    - 使用依赖覆盖 (dependency_overrides) 将 `get_embedder` 依赖项替换为
      会话范围预加载的 `loaded_text_embedder` 实例，避免重复加载模型。

    Args:
        test_settings (Settings): 测试配置对象。
        loaded_text_embedder (TextEmbedder): 预加载的 TextEmbedder 实例。

    Returns:
        FastAPI: 配置好的用于测试的 FastAPI 应用实例。
    """
    logger.info("[test_app fixture] 正在创建会话范围的 FastAPI 测试应用实例...")
    # 使用偏函数将 test_settings 绑定到 lifespan 函数
    # 这假设 lifespan 函数可以接受一个 settings 参数，或者能通过导入 config 模块访问到
    configured_lifespan = partial(lifespan, settings=test_settings)
    logger.info(
        f"[test_app fixture] 已准备 lifespan，绑定的 test_settings ID: {id(test_settings)}"
    )

    # 创建 FastAPI 应用实例，传入配置好的 lifespan
    # 可以选择性地初始化 state
    initial_state = State()
    app = FastAPI(
        title="Test AIGraphX",
        lifespan=configured_lifespan,  # 直接使用偏函数
        state=initial_state,
    )
    logger.info(f"[test_app fixture] FastAPI 实例已创建，ID: {id(app)}")

    # --- 设置依赖覆盖 ---
    # 确保应用内部任何地方通过 Depends(get_embedder) 请求嵌入器时，
    # 都会得到我们预加载的 loaded_text_embedder 实例。
    app.dependency_overrides[get_embedder] = lambda: loaded_text_embedder
    logger.info(f"[test_app fixture] 已应用 get_embedder 的依赖覆盖。")

    # 检查 lifespan 是否成功附加到应用的路由上
    lifespan_attached = getattr(app.router, "lifespan_context", None) is not None
    logger.info(
        f"[test_app fixture] Lifespan 上下文是否已附加到 app.router: {lifespan_attached}"
    )

    # 包含 API 路由
    app.include_router(api_v1_router, prefix="/api/v1") # 确保使用正确的前缀
    logger.info(f"[test_app fixture] 已包含 API v1 路由。")

    logger.info(f"[test_app fixture] 返回测试应用实例。")
    return app


# --- 函数范围的异步 HTTP 客户端 Fixture ---
# `scope="function"` 意味着每个测试函数都会获得一个新的客户端实例，并在测试结束后清理。
# 这有助于隔离测试状态，特别是涉及应用生命周期管理时。
@pytest_asyncio.fixture(scope="function")
async def client(
    test_app: FastAPI,  # 依赖会话范围的测试应用实例
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    为每个测试函数创建一个 `httpx.AsyncClient` 实例，用于向 `test_app` 发送 API 请求。
    使用 `asgi-lifespan` 的 `LifespanManager` 来确保在客户端的上下文中正确执行
    FastAPI 应用的启动 (startup) 和关闭 (shutdown) 事件（由 `lifespan` 函数定义）。

    Args:
        test_app (FastAPI): 会话范围的测试 FastAPI 应用实例。

    Yields:
        httpx.AsyncClient: 可用于发送请求的异步 HTTP 客户端。
    """
    logger.debug(
        "\n--- [client fixture - function scope] 开始执行 ---"
    )
    # 使用 LifespanManager 包装测试应用，它会在进入上下文时触发 startup 事件，退出时触发 shutdown 事件
    async with LifespanManager(test_app) as manager:
        logger.debug(
            "--- [client fixture] LifespanManager 已启动应用 ---"
        )
        # 确保应用状态已由 lifespan 初始化 (如果 lifespan 逻辑正确的话)
        logger.debug(f"--- [client fixture] manager.app.state: {getattr(manager.app, 'state', 'N/A')}")

        # 创建 ASGI 传输层，使用已启动的应用
        transport = ASGITransport(app=manager.app)
        # 创建异步 HTTP 客户端，使用该传输层
        async with AsyncClient(
            transport=transport, base_url="http://test" # base_url 用于相对路径请求
        ) as async_client:
            logger.debug(
                "--- [client fixture] AsyncClient 已创建，Lifespan 已管理 ---"
            )
            # 使用 yield 将客户端提供给测试函数
            yield async_client
        logger.debug("--- [client fixture] AsyncClient 上下文结束 ---")
    # LifespanManager 在退出上下文时会自动处理应用的 shutdown 事件
    logger.debug(
        "--- [client fixture] LifespanManager 已关闭应用，Fixture 结束 ---"
    )


# --- 模拟 GraphService 的 Fixture ---
# 这个 Fixture 用于需要 Mock GraphService 的单元测试或 API 测试。
@pytest_asyncio.fixture
async def mock_graph_service_fixture(
    test_app: FastAPI,  # 依赖测试应用实例，以便在其上应用依赖覆盖
) -> AsyncGenerator[AsyncMock, None]:
    """
    提供一个模拟的 `GraphService` 实例，并在 `test_app` 上设置依赖覆盖，
    使得通过 `Depends(get_graph_service)` 获取依赖时得到的是这个模拟实例。
    测试结束后会自动恢复原始的依赖关系。

    Args:
        test_app (FastAPI): 测试 FastAPI 应用实例。

    Yields:
        AsyncMock: 一个配置了 `spec=GraphService` 的异步模拟对象。
    """
    logger.debug("[mock_graph_service_fixture] 正在创建模拟 GraphService...")
    # 创建一个异步模拟对象，spec=GraphService 确保模拟对象具有 GraphService 的接口
    mock_service = AsyncMock(spec=GraphService)

    # --- 应用依赖覆盖 ---
    # 保存原始的依赖覆盖字典副本，以便测试后恢复
    original_overrides = test_app.dependency_overrides.copy()
    # 获取原始的依赖项获取函数 (get_graph_service) 作为覆盖的键
    override_key = get_graph_service
    # 设置覆盖：当请求 get_graph_service 依赖时，返回我们的 mock_service
    test_app.dependency_overrides[override_key] = lambda: mock_service
    logger.debug("[mock_graph_service_fixture] 已在 test_app 上应用 GraphService 依赖覆盖。")

    # 使用 yield 将模拟对象提供给测试函数
    yield mock_service

    # --- 测试结束后清理 ---
    # 恢复原始的依赖覆盖设置
    test_app.dependency_overrides = original_overrides
    logger.debug("[mock_graph_service_fixture] 已恢复原始的依赖覆盖。")

# --- 应用 Alembic 数据库迁移的 Fixture ---
# `scope="module"` 表示这个 Fixture 在每个测试模块（文件）执行前只运行一次。
# `autouse=True` 表示它会自动应用于所有模块，无需在测试函数中显式请求。
@pytest.fixture(scope="module", autouse=True)
def apply_migrations() -> None:
    """
    确保在每个测试模块运行前，将 Alembic 数据库迁移应用到 **测试** 数据库。
    这是保证测试数据库模式与代码期望一致的关键步骤。
    它通过运行 `alembic upgrade head` 命令实现。
    """
    logger.info("--- [apply_migrations fixture - module scope] 开始执行 ---")
    # 使用在模块加载时从环境变量读取的测试数据库 URL
    logger.info(
        f"[apply_migrations] 使用全局读取的 TEST_DATABASE_URL: {TEST_DB_URL_FROM_ENV}"
    )

    # 如果未设置测试数据库 URL，则跳过迁移
    if not TEST_DB_URL_FROM_ENV:
        logger.warning("全局 TEST_DATABASE_URL 未设置，跳过数据库迁移。")
        return

    logger.info(
        f"正在将 Alembic 迁移应用到由 TEST_DATABASE_URL 定义的测试数据库。"
    )
    # Alembic 需要的数据库 URL 格式可能与 psycopg 不同 (需要 'postgresql+psycopg://')
    db_url_for_alembic = TEST_DB_URL_FROM_ENV
    if db_url_for_alembic and db_url_for_alembic.startswith("postgresql://"):
        db_url_for_alembic = db_url_for_alembic.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    logger.info(f"[apply_migrations] 用于 Alembic 的 URL: {db_url_for_alembic}")

    # 创建一个包含数据库 URL 的环境变量副本，传递给 Alembic 进程
    alembic_env = os.environ.copy()
    alembic_env["DATABASE_URL"] = db_url_for_alembic

    # --- 记录传递给子进程的 PATH 环境变量 (用于调试 Conda 环境问题) ---
    conda_prefix = os.environ.get("CONDA_PREFIX") # 直接获取 Conda 环境前缀
    expected_bin_path = ""
    if conda_prefix:
        expected_bin_path = os.path.join(conda_prefix, "bin") # Conda 环境的 bin 目录
    else:
        # 如果不在 Conda 环境中，尝试基于 Python 可执行文件猜测
        # 这不太可靠，但作为后备
        conda_base_path = os.path.dirname(os.path.dirname(sys.executable))
        expected_bin_path = os.path.join(conda_base_path, "bin")
        logger.warning(
            f"[apply_migrations] CONDA_PREFIX 未设置，猜测的 bin 路径: {expected_bin_path}"
        )

    logger.info(f"[apply_migrations] 检查 Conda bin 路径: {expected_bin_path}")
    current_path = alembic_env.get("PATH", "PATH 键未找到!") # 获取将传递给子进程的 PATH
    if expected_bin_path and expected_bin_path not in current_path:
        logger.warning(
            f"[apply_migrations] 预期的 Conda bin 路径 '{expected_bin_path}' 未在 PATH 中找到! 当前 PATH: {current_path}"
        )
    elif expected_bin_path:
        logger.info(
            f"[apply_migrations] 预期的 Conda bin 路径 '{expected_bin_path}' 已在 PATH 中找到。"
        )
    # --- PATH 日志记录结束 ---

    try:
        logger.info("正在运行: alembic upgrade head")
        # 使用 subprocess.run 执行 alembic 命令
        result = subprocess.run(
            ["alembic", "upgrade", "head"], # 命令和参数
            env=alembic_env, # 传递包含数据库 URL 的环境变量
            check=True,      # 如果命令返回非零退出码，则抛出 CalledProcessError
            capture_output=True, # 捕获标准输出和标准错误
            text=True,       # 以文本模式处理输出
            timeout=60,      # 设置超时时间 (秒)
        )
        logger.info("Alembic upgrade 命令成功完成 (退出码 0)。")
        logger.info("Alembic upgrade head 标准输出:\n%s", result.stdout)
        # 即使成功，也记录标准错误，可能包含警告信息
        if result.stderr:
            logger.warning(
                "Alembic upgrade head 标准错误 (命令成功):\n%s", result.stderr
            )
        logger.info("Alembic 迁移成功应用 (模块范围)。")
    except FileNotFoundError:
        # 如果系统找不到 'alembic' 命令
        logger.error("在模块设置期间未找到 'alembic' 命令。")
        pytest.fail("未找到 'alembic' 命令。请确保已安装并在 PATH 上。")
    except subprocess.TimeoutExpired as e:
        # 如果命令执行超时
        logger.error(f"在模块设置期间 Alembic upgrade 超时: {e.timeout}s。")
        if e.stdout: logger.error("超时前的 Alembic 标准输出:\n%s", e.stdout)
        if e.stderr: logger.error("超时前的 Alembic 标准错误:\n%s", e.stderr)
        pytest.fail("在模块设置期间 Alembic upgrade 超时。")
    except subprocess.CalledProcessError as e:
        # 如果命令返回非零退出码
        logger.error(
            f"在模块设置期间 Alembic upgrade 失败 (退出码 {e.returncode})!"
        )
        logger.error("Alembic 标准输出:\n%s", e.stdout)
        logger.error("Alembic 标准错误:\n%s", e.stderr)
        pytest.fail(f"在模块设置期间 Alembic upgrade head 失败: {e}")
    except Exception as e:
        # 捕获其他意外错误
        logger.exception(
            "在 Alembic 迁移 (模块设置) 过程中发生意外错误。"
        )
        pytest.fail(f"在 Alembic 迁移 (模块设置) 过程中发生意外错误: {e}")
    logger.info("--- [apply_migrations fixture - module scope] 执行结束 ---")


# --- 函数范围的 PostgreSQL 连接池 Fixture ---
# `scope="function"` 确保每个测试函数都使用独立的连接池实例（如果需要严格隔离）
# 或者至少确保连接池在每个测试后正确关闭。
@pytest_asyncio.fixture(scope="function")
async def db_pool(
    test_settings: Settings, # 依赖测试配置以获取数据库 URL
) -> AsyncGenerator[AsyncConnectionPool, None]:
    """
    为每个测试函数管理一个异步 PostgreSQL 连接池。
    使用 `test_settings` 中的数据库 URL 连接到 **测试** 数据库。
    假设数据库迁移已由模块范围的 `apply_migrations` fixture 应用。

    Args:
        test_settings (Settings): 测试配置对象。

    Yields:
        AsyncConnectionPool: 可用的异步连接池实例。
    """
    # 从测试配置中获取测试数据库 URL
    test_db_url = test_settings.database_url
    logger.info(
        f"[db_pool fixture - function scope] 正在为 test_settings URL 创建连接池: {test_db_url}"
    )

    # 如果 URL 未配置，则跳过此 Fixture
    if not test_db_url:
        pytest.skip("在 test_settings 中未找到数据库 URL，跳过连接池创建。")

    pool: Optional[AsyncConnectionPool] = None
    try:
        # 创建连接池实例
        pool = AsyncConnectionPool(
            conninfo=test_db_url, # 使用测试数据库 URL
            min_size=test_settings.pg_pool_min_size, # 使用配置的池大小
            max_size=test_settings.pg_pool_max_size,
            open=False, # 先不打开，下面手动打开
            timeout=60, # 连接超时时间
        )
        logger.info("[db_pool fixture] 正在打开连接池...")
        # 异步打开连接池并等待其准备就绪
        await pool.open(wait=True)
        logger.info(
            "[db_pool fixture] 测试数据库连接池成功打开。"
        )

        # 使用 yield 将连接池提供给测试函数
        yield pool

    except Exception as e:
        # 捕获创建或打开连接池过程中的错误
        logger.exception(
            f"[db_pool fixture] 为 {test_db_url} 创建/打开测试数据库连接池时出错: {e}"
        )
        pytest.fail(f"创建/打开测试数据库连接池失败: {e}")

    finally:
        # 测试函数执行结束后，无论成功与否，都关闭连接池
        if pool:
            logger.info("[db_pool fixture] 正在关闭连接池...")
            await pool.close()
            logger.info("[db_pool fixture] 测试数据库连接池已关闭。")


# --- 函数范围的 PostgresRepository Fixture ---
# `scope="function"` 确保每个测试函数获得一个新的仓库实例，并执行测试前清理。
@pytest_asyncio.fixture()
async def repository(
    db_pool: AsyncConnectionPool, # 依赖上面定义的函数范围连接池
    db_cleanup_lock: asyncio.Lock, # 依赖会话范围的清理锁
) -> AsyncGenerator[PostgresRepository, None]:
    """
    创建 `PostgresRepository` 实例，使用测试数据库连接池。
    在将仓库实例提供给测试函数**之前**，使用锁来同步并清空相关的数据库表，
    以确保测试隔离性。

    Args:
        db_pool (AsyncConnectionPool): 测试数据库连接池。
        db_cleanup_lock (asyncio.Lock): 用于同步数据库清理操作的锁。

    Yields:
        PostgresRepository: 连接到测试数据库的仓库实例。
    """
    # 如果连接池不可用，则跳过
    if not db_pool:
        pytest.skip("测试数据库连接池不可用。")

    # 获取调用此 fixture 的测试函数的名称，用于日志记录
    frame = inspect.currentframe()
    func_name = "unknown_test" # 默认名称
    if frame and frame.f_back:
        caller_frame = frame.f_back
        if caller_frame:
            func_name = caller_frame.f_code.co_name

    # --- 测试前清理 (使用锁确保串行执行) ---
    logger.info(
        f"[repository fixture for {func_name}] 测试前: 尝试获取数据库清理锁..."
    )
    try:
        async with db_cleanup_lock: # 获取锁
            logger.info(
                f"[repository fixture for {func_name}] 测试前: 数据库清理锁已获取。正在清理表..."
            )
            try:
                # 从连接池获取一个连接
                async with db_pool.connection() as conn:
                    logger.info(
                        f"[repository fixture for {func_name}] 测试前: 已获取 PG 连接用于 TRUNCATE。"
                    )
                    # 创建游标
                    async with conn.cursor() as cur:
                        logger.info(
                            f"[repository fixture for {func_name}] 测试前: 正在执行 TRUNCATE..."
                        )
                        # 执行 TRUNCATE 命令清空所有相关表
                        # RESTART IDENTITY 重置自增序列，CASCADE 级联删除相关外键约束引用的数据 (如果适用)
                        await cur.execute(
                            """
                            TRUNCATE TABLE model_paper_links, hf_models, papers,
                                         pwc_tasks, pwc_datasets, pwc_repositories
                                         RESTART IDENTITY CASCADE;
                            """
                        )
                        logger.info(
                            f"[repository fixture for {func_name}] 测试前: TRUNCATE 命令已执行。"
                        )
                    # 提交事务以使清理生效
                    await conn.commit()
                    logger.info(
                        f"[repository fixture for {func_name}] 测试前: TRUNCATE 后已执行 COMMIT。"
                    )
                logger.info(
                    f"[repository fixture for {func_name}] 测试前: 表格已成功截断并提交。"
                )
            except Exception as e:
                # 如果清理过程中发生错误
                logger.error(
                    f"[repository fixture for {func_name}] 测试前: 截断表格时出错 (持有锁): {e}",
                    exc_info=True,
                )
                pytest.fail(
                    f"[repository fixture for {func_name}] 测试前: 清理测试数据库失败: {e}"
                )
    finally:
        # 释放锁
        logger.info(
            f"[repository fixture for {func_name}] 测试前: 数据库清理锁已释放。"
        )

    # --- 创建并提供仓库实例 ---
    # 使用清理后的连接池创建仓库实例
    repo = PostgresRepository(pool=db_pool)
    logger.info(f"[repository fixture for {func_name}] 正在提供仓库实例。")
    yield repo # 将仓库实例提供给测试函数

    # --- 测试后清理 (已移除) ---
    # logger.info(f"[repository fixture for {func_name}] 测试结束 (无测试后清理)。")


# --- 函数范围的 Neo4j 驱动程序 Fixture ---
# `scope="function"` 对于需要管理外部连接和潜在事件循环交互的 Fixture 通常更安全。
@pytest_asyncio.fixture(scope="function")
async def neo4j_driver(test_settings: Settings) -> AsyncGenerator[AsyncDriver, None]:
    """
    为单个测试函数提供一个异步 Neo4j 驱动程序 (`AsyncDriver`) 实例。
    在测试结束后确保驱动程序被关闭。
    使用函数作用域有助于避免与事件循环相关的潜在问题。

    Args:
        test_settings (Settings): 测试配置对象，用于获取 Neo4j 连接信息。

    Yields:
        AsyncDriver: 连接到测试 Neo4j 数据库的驱动程序实例。
    """
    # --- 获取 Neo4j 连接信息 ---
    # 注意：Neo4j 社区版限制，测试通常连接到测试容器内的默认 'neo4j' 数据库。
    # 测试隔离依赖于在每次测试前清理数据库。
    neo4j_uri = test_settings.neo4j_uri
    neo4j_user = test_settings.neo4j_username # Settings 中已包含默认 'neo4j'
    neo4j_pwd = test_settings.neo4j_password
    neo4j_db = test_settings.neo4j_database # 从 test_settings 获取最终生效的数据库名

    # 检查必要的连接信息是否存在
    if not neo4j_uri or not neo4j_user or not neo4j_pwd:
        logger.error(
            "测试设置中未配置 Neo4j URI、用户名或密码。跳过需要驱动程序的 Neo4j 测试。"
        )
        pytest.skip("未配置用于测试的 Neo4j URI、用户名或密码。")
        # 下面的 yield None 不会执行，但满足类型检查器
        yield None # type: ignore
        return

    # 使用 neo4j.basic_auth 创建认证对象
    neo4j_auth_obj = basic_auth(neo4j_user, neo4j_pwd)

    logger.debug(
        f"[neo4j_driver fixture] 正在为数据库 '{neo4j_db}' (URI: {neo4j_uri}) 创建 Neo4j 驱动程序"
    )
    driver: Optional[AsyncDriver] = None
    try:
        # 创建异步驱动程序实例
        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=neo4j_auth_obj)
        # (可选但推荐) 验证驱动程序是否能成功连接到数据库
        try:
            await driver.verify_connectivity()
            logger.debug(
                f"[neo4j_driver fixture] Neo4j 驱动程序连接性已验证 (数据库: '{neo4j_db}')"
            )
        except Exception as e:
            logger.error(
                f"[neo4j_driver fixture] Neo4j 驱动程序连接性检查失败 (数据库: '{neo4j_db}'): {e}",
                exc_info=True,
            )
            pytest.fail(f"Neo4j 驱动程序连接性检查失败: {e}")

        # 使用 yield 将驱动程序提供给测试函数
        yield driver
    except Exception as e:
        # 捕获创建驱动程序过程中的错误
        logger.error(
            f"[neo4j_driver fixture] 创建 Neo4j 驱动程序时出错: {e}", exc_info=True
        )
        pytest.fail(f"创建 Neo4j 驱动程序失败: {e}")
        yield None # 不会执行，满足类型检查器
    finally:
        # 测试函数结束后，无论成功与否，都尝试关闭驱动程序
        if driver:
            logger.debug(
                f"[neo4j_driver fixture] 正在关闭 Neo4j 驱动程序 (数据库: '{neo4j_db}')。"
            )
            try:
                # 异步关闭驱动程序
                await driver.close()
                logger.debug(
                    f"[neo4j_driver fixture] Neo4j 驱动程序已关闭 (数据库: '{neo4j_db}')。"
                )
            except RuntimeError as e:
                # 捕获并记录关闭驱动程序时可能发生的 RuntimeError (例如事件循环已关闭)
                # 这可能是 pytest-asyncio 和某些驱动程序关闭逻辑之间的已知问题
                logger.error(
                    f"[neo4j_driver fixture] 关闭 Neo4j 驱动程序时发生运行时错误 (数据库: '{neo4j_db}'): {e}",
                    exc_info=True,
                )
                # 这里选择不让测试失败，因为这可能是环境问题而不是代码问题
            except Exception as e:
                # 捕获其他关闭驱动程序时的意外错误
                logger.error(
                    f"[neo4j_driver fixture] 关闭 Neo4j 驱动程序时发生意外错误 (数据库: '{neo4j_db}'): {e}",
                    exc_info=True,
                )
                # 可以选择性地让测试失败
                # pytest.fail(f"关闭 Neo4j 驱动程序时发生意外错误: {e}")


# --- 函数范围的 Neo4jRepository Fixture ---
# `scope="function"` 确保每个测试获得新的仓库实例，并在此 Fixture 内执行测试前清理。
@pytest_asyncio.fixture(scope="function")
async def neo4j_repo_fixture(
    neo4j_driver: AsyncDriver, # 依赖上面定义的函数范围驱动程序
) -> AsyncGenerator[Neo4jRepository, None]:
    """
    为每个测试函数提供一个 `Neo4jRepository` 实例。
    在提供实例之前，会执行数据库清理操作 (`MATCH (n) DETACH DELETE n`)
    以确保测试环境的纯净。

    Args:
        neo4j_driver (AsyncDriver): 连接到测试 Neo4j 数据库的驱动程序。

    Yields:
        Neo4jRepository: 连接到测试数据库的 Neo4j 仓库实例。
    """
    logger.info(
        "[neo4j_repo_fixture] 正在创建 Neo4jRepository 并清理数据库 (函数范围)"
    )
    # --- 测试前清理 ---
    try:
        # 使用驱动程序创建会话并执行清理查询
        async with neo4j_driver.session() as session: # 默认连接到驱动配置的数据库
            logger.info(
                "[neo4j_repo_fixture] 测试前正在运行: MATCH (n) DETACH DELETE n..."
            )
            # 注意：这里没有使用 execute_write，因为清理操作通常不需要事务保证，
            # 且 run() 对于简单查询更直接。如果需要事务性清理，应使用 execute_write。
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info(
                "[neo4j_repo_fixture] 测试前完成: MATCH (n) DETACH DELETE n。"
            )
    except Exception as e:
        logger.exception(
            f"[neo4j_repo_fixture] 测试前清理 Neo4j 数据库失败: {e}"
        )
        pytest.fail(f"测试前清理 Neo4j 数据库失败: {e}")

    # --- 创建并提供仓库实例 ---
    repo = Neo4jRepository(driver=neo4j_driver)
    try:
        yield repo # 将仓库实例提供给测试函数
    finally:
        # 测试结束后可以添加清理逻辑，但通常清理在测试前进行
        logger.info("[neo4j_repo_fixture] Fixture 结束 (函数范围)")


# --- 模拟 FaissRepository 的 Fixture ---
# 这个 Fixture 使用 patch 来模拟 FaissRepository 类本身。
@pytest.fixture
def mock_faiss_repository() -> Generator[
    Tuple[MagicMock, MagicMock, MagicMock], None, None
]:
    """
    使用 `unittest.mock.patch` 模拟 `aigraphx.core.db.FaissRepository` 类。
    提供模拟的类对象本身，以及两个模拟的实例（一个用于论文，一个用于模型）。
    这允许测试代码控制 FaissRepository 的行为，而无需实际加载索引文件。

    Yields:
        Tuple[MagicMock, MagicMock, MagicMock]: 一个元组，包含：
            - mock_repo_class: 模拟的 FaissRepository 类对象。
            - mock_instance_papers: 模拟的用于论文的 FaissRepository 实例。
            - mock_instance_models: 模拟的用于模型的 FaissRepository 实例。
    """
    # 指定要 patch 的目标路径，这里是 lifespan 函数 (aigraphx.core.db) 中导入和使用的 FaissRepository
    with patch(
        "aigraphx.core.db.FaissRepository",
        new_callable=MagicMock, # 使用 MagicMock 作为替代品
    ) as mock_repo_class:
        # 创建模拟的论文仓库实例
        mock_instance_papers = MagicMock(spec=FaissRepository) # spec 确保接口匹配
        mock_instance_papers.index = MagicMock() # 模拟内部的 index 属性
        mock_instance_papers.id_map = {} # 模拟 ID 映射
        mock_instance_papers.is_ready.return_value = True # 默认配置为已就绪
        mock_instance_papers.id_type = "int" # 设置论文 ID 类型

        # 创建模拟的模型仓库实例
        mock_instance_models = MagicMock(spec=FaissRepository)
        mock_instance_models.index = MagicMock()
        mock_instance_models.id_map = {}
        mock_instance_models.is_ready.return_value = True
        mock_instance_models.id_type = "str" # 设置模型 ID 类型

        # 配置模拟类对象的 side_effect，使其在被调用（实例化）时，
        # 根据传入的 id_type 参数返回对应的模拟实例。
        # 这模拟了 lifespan 中根据 id_type 创建不同 FaissRepository 实例的行为。
        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            id_type = kwargs.get("id_type")
            if id_type == "int":
                return mock_instance_papers
            elif id_type == "str":
                return mock_instance_models
            else:
                # 如果传入了意外的 id_type，则抛出错误
                raise ValueError(
                    f"在 FaissRepository 模拟中遇到意外的 id_type: {id_type}"
                )

        mock_repo_class.side_effect = side_effect
        # 使用 yield 将模拟的类和实例提供给测试函数
        yield mock_repo_class, mock_instance_papers, mock_instance_models


# --- 模拟 FastAPI 应用状态的 Fixture ---
# 这个 fixture 主要用于需要直接测试依赖 app.state 的组件（例如依赖项获取函数）的场景。
@pytest.fixture
def mock_app_state() -> State:
    """创建一个模拟的 FastAPI/Starlette State 对象。"""
    return State()


# --- 模拟 FastAPI 应用的 Fixture (主要用于测试 lifespan) ---
# 这个 fixture 创建一个简单的 FastAPI 应用实例，主要用于测试 lifespan 函数本身，
# 而不是用于完整的 API 测试 (那个使用 test_app fixture)。
@pytest.fixture
def mock_app() -> FastAPI:
    """创建一个基本的 FastAPI 应用实例，带有初始化的 state 属性，用于测试。"""
    app = FastAPI()
    app.state = State()  # 初始化 state 对象
    return app


# --- 模拟 TextEmbedder 的 Fixture ---
# 用于需要模拟 TextEmbedder 行为的测试。
@pytest.fixture
def mock_embedder() -> MagicMock:
    """提供一个模拟的 TextEmbedder 实例。"""
    embedder = MagicMock(spec=TextEmbedder)
    # 配置 embed 方法返回一个 numpy 数组，模拟嵌入向量
    embedder.embed.return_value = np.random.rand(
        384
    ).astype(np.float32) # 维度应匹配实际模型
    return embedder

# --- 模拟 PostgresRepository 的 Fixture ---
@pytest.fixture
def mock_pg_repo() -> MagicMock:
    """提供一个模拟的 PostgresRepository 实例。"""
    repo = MagicMock(spec=PostgresRepository)
    # 可以根据需要配置模拟方法的返回值或行为
    repo.get_papers_details_by_ids = AsyncMock(return_value=[])
    repo.get_hf_models_by_ids = AsyncMock(return_value=[])
    repo.search_papers_by_keyword = AsyncMock(return_value=([], 0))
    repo.search_models_by_keyword = AsyncMock(return_value=([], 0))
    # 添加默认的 paper_details_map 属性，以防测试中访问它
    repo.paper_details_map = {}
    return repo

# --- 模拟 Neo4jRepository 的 Fixture ---
@pytest.fixture
def mock_neo4j_repo() -> Optional[MagicMock]:
    """
    提供一个模拟的 Neo4jRepository 实例。
    如果 Neo4j 连接信息未配置，则返回 None。
    """
    # 检查是否配置了 Neo4j 连接，如果未配置，则此模拟可能不需要或无法有效模拟
    if not TEST_NEO4J_URI or not TEST_NEO4J_PASSWORD:
         logger.warning("Neo4j 未配置，mock_neo4j_repo 返回 None。")
         return None

    repo = MagicMock(spec=Neo4jRepository)
    # 根据需要配置模拟方法的返回值
    repo.get_neighbors = AsyncMock(return_value=[])
    repo.get_related_nodes = AsyncMock(return_value=[])
    # ... 其他需要模拟的方法
    return repo

# --- 模拟 FaissRepository 实例的 Fixture ---
# 提供单独的模拟实例，方便在测试中直接使用。
@pytest.fixture
def mock_faiss_paper_repo() -> MagicMock:
    """提供一个模拟的用于论文的 FaissRepository 实例。"""
    repo = MagicMock(spec=FaissRepository)
    repo.is_ready.return_value = True
    repo.id_type = "int"
    repo.index = MagicMock()
    repo.id_map = {}
    repo.search_similar = AsyncMock(return_value=[]) # 默认返回空
    return repo

@pytest.fixture
def mock_faiss_model_repo() -> MagicMock:
    """提供一个模拟的用于模型的 FaissRepository 实例。"""
    repo = MagicMock(spec=FaissRepository)
    repo.is_ready.return_value = True
    repo.id_type = "str"
    repo.index = MagicMock()
    repo.id_map = {}
    repo.search_similar = AsyncMock(return_value=[]) # 默认返回空
    return repo

# --- 模拟 SearchService 的 Fixture ---
# 这个 fixture 创建一个 SearchService 实例，但其依赖项（仓库、嵌入器）都被替换为模拟对象。
@pytest.fixture
def search_service(
    mock_embedder: Optional[TextEmbedder], # 可以是模拟对象或 None
    mock_faiss_paper_repo: FaissRepository,
    mock_faiss_model_repo: FaissRepository,
    mock_pg_repo: PostgresRepository,
    mock_neo4j_repo: Optional[Neo4jRepository], # 可以是模拟对象或 None
) -> SearchService:
    """
    提供一个 SearchService 实例，其所有依赖项都被替换为模拟对象。
    用于单元测试 SearchService 自身的逻辑。

    Args:
        mock_embedder: 模拟的 TextEmbedder。
        mock_faiss_paper_repo: 模拟的 Faiss 论文仓库。
        mock_faiss_model_repo: 模拟的 Faiss 模型仓库。
        mock_pg_repo: 模拟的 PG 仓库。
        mock_neo4j_repo: 模拟的 Neo4j 仓库。

    Returns:
        SearchService: 使用模拟依赖项初始化的 SearchService 实例。
    """
    return SearchService(
        embedder=mock_embedder,
        faiss_repo_papers=mock_faiss_paper_repo,
        faiss_repo_models=mock_faiss_model_repo,
        pg_repo=mock_pg_repo,
        neo4j_repo=mock_neo4j_repo,
    )


# --- (可选) 事件循环策略 Fixture (通常不需要全局设置) ---
# 用于解决特定平台（如 Windows）上 asyncio 事件循环与某些库（如 testcontainers）的兼容性问题。
# 在 Linux/WSL 上通常不需要。
# @pytest.fixture(scope="session")
# def event_loop_policy(request):
#     """为测试会话设置 asyncio 事件循环策略。"""
#     # 在 Windows 上，默认的 ProactorEventLoop 可能与 testcontainers 启动的子进程不兼容。
#     # 可以切换到 SelectorEventLoop。
#     # if sys.platform == "win32":
#     #     logger.debug("为 Windows 设置 asyncio 事件循环策略为 SelectorEventLoop。")
#     #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     # 对于其他平台（如 Linux/WSL），使用默认策略即可。
#     # pytest-asyncio 需要返回策略对象。
#     return asyncio.get_event_loop_policy() # 获取当前平台的默认策略
