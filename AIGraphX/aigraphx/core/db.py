"""
数据库与资源生命周期管理模块 (Database & Resource Lifecycle Management Module)

功能 (Function):
这个模块的核心是 `lifespan` 异步上下文管理器。它的主要职责是：
1. 在 FastAPI 应用启动时 (startup phase)，初始化并建立与外部资源的连接或加载所需实例，例如：
    - PostgreSQL 数据库连接池 (`psycopg_pool.AsyncConnectionPool`)。
    - Neo4j 图数据库驱动 (`neo4j.AsyncDriver`)。
    - Faiss 向量索引仓库实例 (`FaissRepository`)，区分论文和模型。
    - 文本嵌入模型实例 (`TextEmbedder`)。
2. 对初始化的资源进行健康检查（例如，检查 Faiss 索引文件是否存在且可用）。如果关键资源初始化失败或未就绪，则抛出异常，阻止应用启动。
3. 将成功初始化并准备就绪的资源实例存储在 FastAPI 应用的 `app.state` 对象中，使其在整个应用生命周期内可用，并能被依赖注入系统访问。
4. 在 FastAPI 应用关闭时 (shutdown phase)，优雅地释放和清理这些资源，例如：
    - 关闭 PostgreSQL 连接池。
    - 关闭 Neo4j 驱动。
    - (如果需要) 清理 Faiss 或 Embedder 相关的资源。

交互 (Interaction):
- 依赖 (Depends on):
    - `aigraphx.core.config`: 需要 `Settings` 对象来获取数据库 URL、用户名/密码、连接池大小、Faiss 路径、嵌入模型名称等配置信息。`lifespan` 函数现在显式接收 `settings` 对象作为参数。
    - `psycopg_pool`: 用于创建 PostgreSQL 异步连接池。
    - `neo4j`: 用于创建 Neo4j 异步驱动。
    - `aigraphx.repositories.*`: 导入 `FaissRepository` 类用于实例化。
    - `aigraphx.vectorization.embedder`: 导入 `TextEmbedder` 类用于实例化。
    - `fastapi`: 需要 `FastAPI` 应用实例 (`app`) 来访问 `app.state`。
- 被使用 (Used by):
    - `aigraphx.main`: 将 `lifespan` 函数（通常通过 `functools.partial` 包装以传入 `settings`）传递给 `FastAPI` 应用实例的 `lifespan` 参数。FastAPI 框架会自动在启动和关闭时调用它。
    - `aigraphx.api.v1.dependencies`: 依赖注入函数（如 `get_postgres_pool`, `get_neo4j_driver`, `get_app_state`）会从 `request.app.state` 中获取由 `lifespan` 初始化的资源实例。

设计原则 (Design Principles):
- **资源集中管理 (Centralized Resource Management):** 所有需要在应用生命周期内共享的、需要显式初始化和清理的资源都在 `lifespan` 中统一处理。
- **状态管理规范 (State Management Standard):** 严格使用 `app.state` 存储共享资源，禁止使用全局变量。
- **启动时检查 (Startup Checks):** 在应用启动时验证关键资源的可用性，实现快速失败 (Fail Fast)，避免应用在无法正常工作的情况下启动。
- **优雅关闭 (Graceful Shutdown):** 确保在应用关闭时释放资源，避免资源泄露。
- **可测试性 (Testability):** 通过将 `settings` 对象作为参数传入 `lifespan`，使得在测试中可以方便地注入不同的配置（例如连接到测试数据库）。
- **明确的错误处理 (Explicit Error Handling):** 对资源初始化和检查过程中的错误进行捕获、记录，并抛出明确的异常，提供清晰的失败信息。
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict
from functools import partial  # Added for potential use later if needed

from dotenv import load_dotenv
from fastapi import FastAPI
from neo4j import AsyncGraphDatabase, AsyncDriver
import psycopg_pool

# Import config variables - Keep the main import for type hinting if needed elsewhere
# but the lifespan function will now rely on the passed settings object.
from aigraphx.core.config import Settings, settings as global_settings

# Import repository and embedder classes
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.vectorization.embedder import TextEmbedder

logger = logging.getLogger(__name__)


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(
    app: FastAPI, settings: Settings
) -> AsyncGenerator[Dict, None]:  # Added settings parameter
    """
    Handles application startup and shutdown events.
    Initializes database connections, models, etc., and cleans them up.
    Uses app.state to store resources.
    Receives settings explicitly to allow for easier testing overrides.
    """
    # Removed: from aigraphx.core.config import settings (no longer needed inside)
    print("--- DEBUG: Lifespan STARTING ---")
    logger.info("Application lifespan startup: Initializing resources...")

    # Initialize resources directly on app.state
    app.state.pg_pool = None
    app.state.neo4j_driver = None
    app.state.faiss_repo_papers = None
    app.state.faiss_repo_models = None
    app.state.embedder = None
    logger.debug("Initialized app.state attributes to None.")

    # Use the passed settings object
    logger.debug(f"[Lifespan Startup] Settings object ID used: {id(settings)}")
    logger.info(
        f"[Lifespan Startup] DATABASE_URL from settings: {settings.database_url}"
    )

    # --- Log Faiss paths for debugging ---
    logger.info(
        f"[Lifespan Startup] Paper Faiss Index Path: {settings.faiss_index_path}"
    )
    logger.info(
        f"[Lifespan Startup] Paper Faiss Map Path: {settings.faiss_mapping_path}"
    )
    logger.info(
        f"[Lifespan Startup] Model Faiss Index Path: {settings.models_faiss_index_path}"
    )
    logger.info(
        f"[Lifespan Startup] Model Faiss Map Path: {settings.models_faiss_mapping_path}"
    )
    # --- End Log ---

    # Validate Database URL before initializing pool
    db_url = settings.database_url
    if not db_url:
        logger.error(
            "CRITICAL: DATABASE_URL is not configured in settings. Cannot initialize PostgreSQL pool."
        )
        # Option 1: Raise immediately (prevents app startup without DB)
        raise RuntimeError("Database URL is not configured, cannot start application.")
        # Option 2: Set pool to None and proceed (app might start but DB features fail)
        # app.state.pg_pool = None
        # logger.warning("PostgreSQL pool set to None due to missing DATABASE_URL.")
    else:
        # Initialize PostgreSQL Pool using passed settings
        try:
            logger.info(f"Attempting to initialize PostgreSQL pool for {db_url}...")
            app.state.pg_pool = psycopg_pool.AsyncConnectionPool(
                conninfo=db_url,  # Use validated db_url
                min_size=settings.pg_pool_min_size,
                max_size=settings.pg_pool_max_size,
            )
            logger.info(
                f"PostgreSQL pool successfully assigned to app.state.pg_pool. Type: {type(app.state.pg_pool)}"
            )
        except Exception as e:
            logger.exception(f"Failed to initialize PostgreSQL pool: {e}")
            logger.error(
                "CRITICAL: PostgreSQL pool initialization failed. Raising RuntimeError."
            )
            raise RuntimeError("PostgreSQL pool initialization failed") from e

    # Initialize Neo4j Driver using passed settings
    if settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password:
        try:
            logger.info(f"Initializing Neo4j driver for {settings.neo4j_uri}...")
            app.state.neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            logger.info("Neo4j driver initialized.")
        except Exception as e:
            logger.exception(f"Failed to initialize Neo4j driver: {e}")
            raise RuntimeError("Neo4j driver initialization failed") from e
    else:
        logger.warning(
            "Neo4j connection details not fully provided in settings. Neo4j features will be unavailable."
        )

    # Initialize Faiss Repository for Papers (Simplified Error Handling)
    logger.info("Initializing Faiss Repository for Papers...")
    faiss_repo_papers_instance = None  # Temporary variable
    try:
        # Only try to instantiate the repository here
        faiss_repo_papers_instance = FaissRepository(
            index_path=settings.faiss_index_path,
            id_map_path=settings.faiss_mapping_path,
            id_type="int",
        )
        app.state.faiss_repo_papers = (
            faiss_repo_papers_instance  # Assign if instantiation succeeded
        )
        logger.info("FaissRepository for Papers instantiated.")

    except Exception as e:
        # Catch only instantiation errors
        logger.exception(
            f"Failed to initialize Papers Faiss Repository (Instantiation Error): {e}"
        )
        app.state.faiss_repo_papers = (
            None  # Ensure state is None on instantiation exception
        )
        logger.error(
            "CRITICAL: Papers Faiss Repository instantiation failed. Raising RuntimeError."
        )
        raise RuntimeError("Papers Faiss Repository initialization failed") from e

    # Now, check readiness *outside* the instantiation try...except block
    # Only proceed if instantiation was successful (faiss_repo_papers_instance is not None)
    if faiss_repo_papers_instance is not None:
        logger.info("Checking readiness of Papers Faiss Repository...")
        if not faiss_repo_papers_instance.is_ready():
            index_exists = os.path.exists(settings.faiss_index_path)
            ntotal = (
                faiss_repo_papers_instance.index.ntotal
                if faiss_repo_papers_instance.index
                else None
            )
            map_exists_and_not_empty = bool(faiss_repo_papers_instance.id_map)
            logger.error(
                f"[Lifespan Check Failed] Papers Faiss State: index_exists={index_exists}, ntotal={ntotal}, map_not_empty={map_exists_and_not_empty}"
            )
            # Raise the specific error for not being ready
            logger.error(
                "CRITICAL: Papers Faiss Repository is not ready. Raising RuntimeError."
            )
            # Set state to None *before* raising, as the repo is unusable
            app.state.faiss_repo_papers = None
            raise RuntimeError(
                "Papers Faiss Repository is not ready after initialization."
            )
        else:
            logger.info("Papers Faiss Repository initialized and ready.")
    else:
        # This case is technically handled by the raise in the except block,
        # but adding an info log might be useful for clarity.
        logger.info(
            "Skipping readiness check for Papers Faiss Repository due to instantiation failure."
        )

    # Initialize Faiss Repository for Models (Simplified Error Handling)
    logger.info("Initializing Faiss Repository for Models...")
    faiss_repo_models_instance = None  # Temporary variable
    try:
        # Only try to instantiate the repository here
        faiss_repo_models_instance = FaissRepository(
            index_path=settings.models_faiss_index_path,
            id_map_path=settings.models_faiss_mapping_path,
            id_type="str",
        )
        app.state.faiss_repo_models = (
            faiss_repo_models_instance  # Assign if instantiation succeeded
        )
        logger.info("FaissRepository for Models instantiated.")

    except Exception as e:
        # Catch only instantiation errors
        logger.exception(
            f"Failed to initialize Models Faiss Repository (Instantiation Error): {e}"
        )
        app.state.faiss_repo_models = (
            None  # Ensure state is None on instantiation exception
        )
        logger.error(
            "CRITICAL: Models Faiss Repository instantiation failed. Raising RuntimeError."
        )
        raise RuntimeError("Models Faiss Repository initialization failed") from e

    # Now, check readiness *outside* the instantiation try...except block
    # Only proceed if instantiation was successful
    if faiss_repo_models_instance is not None:
        logger.info("Checking readiness of Models Faiss Repository...")
        if not faiss_repo_models_instance.is_ready():
            index_exists_m = os.path.exists(settings.models_faiss_index_path)
            ntotal_m = (
                faiss_repo_models_instance.index.ntotal
                if faiss_repo_models_instance.index
                else None
            )
            map_exists_and_not_empty_m = bool(faiss_repo_models_instance.id_map)
            logger.error(
                f"[Lifespan Check Failed] Models Faiss State: index_exists={index_exists_m}, ntotal={ntotal_m}, map_not_empty={map_exists_and_not_empty_m}"
            )
            # Raise the specific error for not being ready
            logger.error(
                "CRITICAL: Models Faiss Repository is not ready. Raising RuntimeError."
            )
            # Set state to None *before* raising
            app.state.faiss_repo_models = None
            raise RuntimeError(
                "Models Faiss Repository is not ready after initialization."
            )
        else:
            logger.info("Models Faiss Repository initialized and ready.")
    else:
        logger.info(
            "Skipping readiness check for Models Faiss Repository due to instantiation failure."
        )

    # Initialize Text Embedder (New)
    logger.info("Initializing Text Embedder...")
    try:
        if settings.sentence_transformer_model:
            logger.info(
                f"Loading sentence transformer model: {settings.sentence_transformer_model} on device: {settings.embedder_device}"
            )
            app.state.embedder = TextEmbedder(
                model_name=settings.sentence_transformer_model,
                device=settings.embedder_device,
            )
            logger.info(
                f"Text Embedder initialized successfully with model. Model loaded: {app.state.embedder.model is not None}"
            )
        else:
            logger.warning(
                "SENTENCE_TRANSFORMER_MODEL not specified in settings. Embedder not initialized."
            )
            app.state.embedder = None
    except Exception as e:
        logger.exception(f"Failed to initialize Text Embedder: {e}")
        # Depending on criticality, you might want to raise an error here
        logger.error(
            "CRITICAL: Text Embedder initialization failed. Semantic search will be unavailable."
        )
        app.state.embedder = None  # Ensure it's None if init fails
        # Optionally raise RuntimeError if embedder is absolutely required
        # raise RuntimeError("Text Embedder initialization failed") from e

    logger.info("Resource initialization process completed.")
    print("--- DEBUG: Lifespan YIELDING (Resources should be initialized) ---")

    # --- Shutdown ---
    print("--- DEBUG: Lifespan SHUTDOWN starting ---")
    logger.info("Application lifespan shutdown: Cleaning up resources...")
    # Use passed settings object here too, although it likely won't change between startup and shutdown
    logger.info(
        f"[Lifespan Shutdown] DATABASE_URL from settings: {settings.database_url}"
    )
    pg_pool = getattr(app.state, "pg_pool", None)  # Get from app.state
    if pg_pool:
        try:
            # Log the conninfo actually used by the pool at shutdown for verification
            logger.info(
                f"Closing PostgreSQL pool (pool conninfo: {pg_pool.conninfo})..."
            )
            await pg_pool.close()
            logger.info("PostgreSQL pool closed.")
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL pool: {e}")

    neo4j_driver = getattr(app.state, "neo4j_driver", None)  # Get from app.state
    if neo4j_driver:
        try:
            logger.info("Closing Neo4j driver...")
            await neo4j_driver.close()
            logger.info("Neo4j driver closed.")
        except Exception as e:
            logger.warning(f"Error closing Neo4j driver: {e}")

    logger.info("Resource cleanup finished.")
    print("--- DEBUG: Lifespan FINISHED ---")

    # --- Startup Complete ---
    logger.info("Application lifespan startup phase completed successfully.")
    print("--- DEBUG: Lifespan STARTUP YIELDING ---")

    # 使用 yield 将控制权交还给 FastAPI，应用开始处理请求
    yield {
        "pg_pool": pg_pool,
        "neo4j_driver": neo4j_driver,
        "faiss_repo_papers": app.state.faiss_repo_papers,
        "faiss_repo_models": app.state.faiss_repo_models,
        "embedder": app.state.embedder,
    }

    # --- Shutdown Phase ---
    # 当 FastAPI 应用关闭时，yield 之后的代码会被执行
    print("--- DEBUG: Lifespan SHUTDOWN STARTING ---")
    logger.info("Application lifespan shutdown: Cleaning up resources...")

    # 关闭 PostgreSQL 连接池
    pg_pool_to_close = getattr(app.state, "pg_pool", None)
    if not pg_pool_to_close:
        pg_pool_to_close = getattr(app.state, "pg_pool", None)

    if pg_pool_to_close:
        logger.info("Closing PostgreSQL connection pool...")
        try:
            await pg_pool_to_close.close()
            logger.info("PostgreSQL connection pool closed.")
        except Exception as e:
            logger.exception(f"Error closing PostgreSQL pool: {e}")

    # 关闭 Neo4j 驱动
    neo4j_driver_to_close = getattr(app.state, "neo4j_driver", None)
    if not neo4j_driver_to_close:
        neo4j_driver_to_close = getattr(app.state, "neo4j_driver", None)

    if neo4j_driver_to_close:
        logger.info("Closing Neo4j driver...")
        try:
            await neo4j_driver_to_close.close()
            logger.info("Neo4j driver closed.")
        except Exception as e:
            logger.exception(f"Error closing Neo4j driver: {e}")

    # (可选) 清理 Faiss 相关资源
    # FaissRepository 在 Python 层面通常不需要显式关闭，索引由文件系统管理
    # 如果有内存映射等特殊情况可能需要处理
    logger.debug("Faiss repositories do not require explicit closing in this setup.")

    # (可选) 清理 Text Embedder 相关资源
    # SentenceTransformer 模型通常由 Python 垃圾回收处理
    # 如果使用了特定设备（如 GPU）且需要释放显存，可能需要额外操作
    logger.debug("Text embedder does not require explicit closing in this setup.")

    logger.info("Application lifespan shutdown phase completed.")
    print("--- DEBUG: Lifespan SHUTDOWN COMPLETE ---")


# --- Removed Dependency Injection Functions ---
# These are now handled in aigraphx/api/dependencies.py

# async def get_postgres_repo() -> PostgresRepository:
#     ...

# async def get_neo4j_repo() -> Optional[Neo4jRepository]:
#     ...

# async def get_faiss_repo() -> FaissRepository:
#     ...

# async def get_embedder() -> TextEmbedder:
#     ...
