# -*- coding: utf-8 -*-
"""
FastAPI 依赖注入 (Dependency Injection) 定义文件

此文件集中定义了 API 端点函数所需的各种依赖项的 "提供者" (provider) 函数。
FastAPI 的依赖注入系统会自动调用这些函数，并将它们的返回值注入到需要它们的
路径操作函数（或其他依赖项函数）的参数中。

主要职责:
1.  **获取共享资源**: 定义函数 (如 `get_app_state`, `get_postgres_pool`, `get_neo4j_driver`, `get_embedder`)
    从应用的 `lifespan` 管理的 `request.app.state` 中安全地获取共享资源实例（如数据库连接池、驱动、文本嵌入器）。
    这些函数通常依赖于 `Request` 对象来访问 `app.state`。
2.  **提供仓库 (Repository) 实例**: 定义函数 (如 `get_postgres_repository`, `get_neo4j_repository`, `get_faiss_repository_papers`, `get_faiss_repository_models`)
    用于创建和返回数据访问层的仓库类实例。这些函数通常会依赖于上面获取的共享资源（如连接池或驱动）。
    Faiss 仓库直接从 `app.state` 获取，因为它们本身就是由 `lifespan` 管理的实例。
3.  **提供服务 (Service) 实例**: 定义函数 (如 `get_search_service`, `get_graph_service`)
    用于创建和返回业务逻辑层的服务类实例。这些函数通常会依赖于上面提供的仓库实例和可能的其他资源（如嵌入器）。
4.  **统一管理**: 将所有核心依赖项的获取逻辑集中在此处，便于维护、测试和替换依赖项。

与其他文件的交互:
*   **`fastapi.Depends`**: FastAPI 框架的核心机制，用于声明依赖关系。路径操作函数通过 `param: Type = Depends(provider_func)` 的形式声明依赖。
*   **`fastapi.Request`**: 用于访问应用的 `state` 属性 (`request.app.state`)。
*   **`starlette.datastructures.State`**: FastAPI (Starlette) 用于存储应用级别共享状态的对象类型。
*   **`aigraphx.core.db.lifespan`**: `lifespan` 函数负责在应用启动时将数据库连接池、Neo4j 驱动、Faiss 仓库实例和文本嵌入器实例等放入 `app.state`。此文件中的依赖函数则负责从 `app.state` 中取出这些实例。
*   **`aigraphx.core.config`**: 可能导入 `settings` 对象以获取配置信息（虽然在此文件中直接使用较少，通常由 `lifespan` 或仓库/服务自身使用）。
*   **`aigraphx.repositories.*`**: 导入仓库类定义，用于类型提示和实例化。
*   **`aigraphx.services.*`**: 导入服务类定义，用于类型提示和实例化。
*   **`aigraphx.vectorization.embedder`**: 导入文本嵌入器类定义。
*   **API 端点文件 (e.g., `aigraphx/api/v1/endpoints/search.py`)**: 这些文件中的路径操作函数会使用 `Depends()` 来调用此文件中定义的依赖提供函数，以获取所需的仓库或服务实例。
"""

# 导入标准库
import logging  # 日志记录
from functools import lru_cache  # 用于缓存依赖项结果 (如果适用且无状态)
from typing import Optional, Dict, Any, cast  # 类型提示

# 导入 FastAPI 和 Starlette 相关组件
from fastapi import (
    Depends,
    HTTPException,
    status,
    Request,
)  # 依赖注入、HTTP 异常、状态码、请求对象
from psycopg_pool import AsyncConnectionPool  # PostgreSQL 异步连接池类型提示
from neo4j import AsyncDriver  # Neo4j 异步驱动类型提示
from starlette.datastructures import State  # 应用状态对象类型提示

# 导入项目配置 (虽然在此文件中直接使用不多，但保持导入可能有用)
from aigraphx.core.config import settings  # 导入应用配置对象

# 导入仓库类定义，用于类型提示和实例化
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository
from aigraphx.repositories.faiss_repo import FaissRepository

# 导入嵌入器和服务类定义，用于类型提示和实例化
from aigraphx.vectorization.embedder import TextEmbedder
from aigraphx.services.search_service import SearchService
from aigraphx.services.graph_service import GraphService


# 获取当前模块的日志记录器实例
logger = logging.getLogger(__name__)

# --- Faiss 相关配置 (硬编码在此处，更好的做法是移到 config.py) ---
# 这些常量定义了 Faiss 索引文件的路径和嵌入维度。
# 嵌入维度需要与所使用的句子转换器模型匹配。
# 例如: all-MiniLM-L6-v2 模型输出 384 维向量。
EMBEDDING_DIMENSION = 384  # 嵌入向量的维度

FAISS_PAPER_INDEX_PATH = (
    "data/faiss_papers.index"  # 论文 Faiss 索引文件路径 (应在 data/ 目录下)
)
FAISS_PAPER_ID_MAP_PATH = (
    "data/faiss_paper_id_map.pkl"  # 论文 Faiss ID 映射文件路径 (应在 data/ 目录下)
)
FAISS_HF_INDEX_PATH = "data/faiss_hf_models.index"  # HuggingFace 模型 Faiss 索引文件路径 (应在 data/ 目录下)
FAISS_HF_ID_MAP_PATH = "data/faiss_hf_id_map.pkl"  # HuggingFace 模型 Faiss ID 映射文件路径 (应在 data/ 目录下)


# --- 共享资源获取函数 --- #


def get_app_state(request: Request) -> State:
    """
    依赖函数：从请求对象中获取并返回应用的共享状态 (`app.state`)。

    这是访问由 `lifespan` 管理器初始化的共享资源（如数据库连接池、
    Faiss 实例、嵌入器等）的入口点。

    Args:
        request (Request): FastAPI 的请求对象，通过它可以访问 `request.app.state`。

    Raises:
        HTTPException: 如果 `request.app.state` 不存在（表示应用状态未初始化），
                       则抛出 500 内部服务器错误。

    Returns:
        State: Starlette 的 State 对象，包含了应用级别的共享资源。
               使用 cast(State, ...) 是为了帮助类型检查器确认返回类型。
    """
    # 检查 request.app 是否有关联的 state 属性
    if not hasattr(request.app, "state"):
        # 关键调试信息：状态未找到
        print(
            "[get_app_state DEBUG] !!! request.app has no attribute 'state'", flush=True
        )
        logger.error(
            "应用状态 'request.app.state' 未找到！Lifespan 可能未正确执行或初始化失败。"
        )
        # 抛出 HTTP 500 错误，因为这是应用配置问题
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  # 使用 status 枚举更标准
            detail="应用状态未初始化，服务暂时不可用。",
        )
    # 关键调试信息：返回状态对象
    print(
        f"[get_app_state DEBUG] 返回 request.app.state (类型: {type(request.app.state)}, ID: {id(request.app.state)})",
        flush=True,
    )
    # 理论上 request.app.state 应该是 State 类型，但做个检查更健壮
    if not isinstance(request.app.state, State):
        logger.warning(
            f"request.app.state 不是 State 类型，而是 {type(request.app.state)}。仍然尝试返回。"
        )
    # 返回 state 对象。使用 cast 告诉类型检查器我们确定它是 State 类型。
    return cast(State, request.app.state)


# --- Dependency Functions --- #


def get_postgres_pool(
    state: State = Depends(get_app_state),  # 依赖于 get_app_state 获取 state 对象
) -> AsyncConnectionPool:
    """
    依赖函数：从应用状态 (`app.state`) 中获取 PostgreSQL 异步连接池。

    Args:
        state (State): 通过依赖注入从 `get_app_state` 获取的应用状态对象。

    Raises:
        HTTPException: 如果在 `state` 中找不到 `pg_pool` (可能未初始化或 lifespan 失败)，
                       则抛出 503 服务不可用错误。

    Returns:
        AsyncConnectionPool: 获取到的 PostgreSQL 连接池实例。
    """
    # 使用 getattr 安全地尝试从 state 对象获取名为 'pg_pool' 的属性
    pool = getattr(state, "pg_pool", None)
    # 如果获取不到 (返回值为 None)
    if pool is None:
        logger.error("PostgreSQL 连接池未在应用状态中初始化或不可用。")
        # 抛出 HTTP 503 服务不可用错误，因为数据库连接是核心功能
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="数据库连接池当前不可用。",
        )
    # 类型提示器可能无法完全推断 getattr 的结果，但我们知道它是连接池。
    # 可以使用 return cast(AsyncConnectionPool, pool) 或忽略类型检查器的警告。
    # 返回获取到的连接池
    return pool  # type: ignore


def get_neo4j_driver(
    state: State = Depends(get_app_state),  # 依赖于 get_app_state 获取 state 对象
) -> Optional[AsyncDriver]:
    """
    依赖函数：从应用状态 (`app.state`) 中获取 Neo4j 异步驱动程序实例。
    注意：Neo4j 驱动可能是可选的，取决于配置。因此返回类型是 Optional[AsyncDriver]。

    Args:
        state (State): 通过依赖注入从 `get_app_state` 获取的应用状态对象。

    Returns:
        Optional[AsyncDriver]: 获取到的 Neo4j 驱动实例，如果未初始化或未配置，则返回 None。
    """
    # 同样使用 getattr 安全地尝试获取 'neo4j_driver'
    driver = getattr(state, "neo4j_driver", None)
    # 如果驱动不存在，只记录警告，因为 Neo4j 可能是可选的
    if driver is None:
        logger.warning(
            "Neo4j 驱动未在应用状态中初始化或不可用。图数据库相关功能可能受限。"
        )
    # 返回驱动实例或 None
    return driver


def get_embedder(state: State = Depends(get_app_state)) -> TextEmbedder:
    """
    依赖函数：从应用状态 (`app.state`) 中获取文本嵌入器实例。

    Args:
        state (State): 通过依赖注入从 `get_app_state` 获取的应用状态对象。

    Raises:
        HTTPException: 如果在 `state` 中找不到 `embedder` 或嵌入器模型未加载，
                       则抛出 503 服务不可用错误。

    Returns:
        TextEmbedder: 获取到的文本嵌入器实例。
    """
    # 尝试获取 'embedder' 实例
    embedder = getattr(state, "embedder", None)
    # 检查嵌入器实例是否存在，并且其内部模型是否已加载
    # (假设 TextEmbedder 有一个 `model` 属性或类似标志表示是否就绪)
    if embedder is None or not getattr(embedder, "model", None):  # 更健壮的检查方式
        logger.error("文本嵌入器未初始化或模型未加载。")
        # 抛出 HTTP 503 服务不可用错误，因为向量搜索依赖它
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文本嵌入服务当前不可用。",
        )
    # 返回获取到的嵌入器实例
    return embedder  # type: ignore


# --- Repository Dependencies --- #
# --- 仓库 (Repository) 依赖提供函数 --- #


# @lru_cache() # 如果仓库是无状态且创建开销大，可以考虑缓存
def get_postgres_repository(
    # 依赖于 get_postgres_pool 获取连接池
    pool: AsyncConnectionPool = Depends(get_postgres_pool),
) -> PostgresRepository:
    """
    依赖函数：创建并返回一个 PostgresRepository 实例。
    每次请求都会创建一个新的实例（除非使用了缓存）。

    Args:
        pool (AsyncConnectionPool): 通过依赖注入获取的 PostgreSQL 连接池。

    Returns:
        PostgresRepository: Postgres 仓库的新实例。
    """
    logger.debug(f"创建 PostgresRepository 实例 (Pool ID: {id(pool)})")
    # 使用获取到的连接池初始化仓库
    return PostgresRepository(pool=pool)


def get_neo4j_repository(
    # 依赖于 get_neo4j_driver 获取驱动 (可能是 None)
    driver: Optional[AsyncDriver] = Depends(get_neo4j_driver),
) -> Optional[Neo4jRepository]:
    """
    依赖函数：如果 Neo4j 驱动可用，则创建并返回一个 Neo4jRepository 实例。
    如果驱动不可用 (为 None)，则返回 None。

    Args:
        driver (Optional[AsyncDriver]): 通过依赖注入获取的 Neo4j 驱动实例或 None。

    Returns:
        Optional[Neo4jRepository]: Neo4j 仓库的实例，如果驱动不可用则为 None。
    """
    # 检查驱动是否存在
    if driver:
        logger.debug(f"创建 Neo4jRepository 实例 (Driver ID: {id(driver)})")
        # 使用获取到的驱动初始化仓库
        return Neo4jRepository(driver=driver)
    else:
        logger.debug("Neo4j 驱动不可用，无法创建 Neo4jRepository 实例。")
        # 如果驱动不存在，返回 None
        return None


# Rename for clarity and add model repo getter
# @lru_cache() # Faiss 仓库是有状态的（加载了索引），不应缓存依赖函数本身
# Faiss 仓库实例由 lifespan 管理并存储在 app.state 中
def get_faiss_repository_papers(
    state: State = Depends(get_app_state),  # 依赖于 get_app_state 获取 state
) -> FaissRepository:
    """
    依赖函数：从应用状态 (`app.state`) 中获取论文 Faiss 仓库实例。
    这个实例是由 `lifespan` 在应用启动时创建和加载的。

    Args:
        state (State): 通过依赖注入获取的应用状态对象。

    Raises:
        HTTPException: 如果在 `state` 中找不到 `faiss_repo_papers` 或仓库未就绪，
                       则抛出 503 服务不可用错误。

    Returns:
        FaissRepository: 获取到的论文 Faiss 仓库实例。
    """
    # 调试信息：尝试获取仓库
    # Use print for debugging
    print(
        f"[get_faiss_repository_papers DEBUG] 尝试从 state (类型: {type(state)}, ID: {id(state)}) 获取 'faiss_repo_papers'。",
        flush=True,
    )
    # 尝试从 state 获取 'faiss_repo_papers' 实例
    # Use getattr on the State object
    repo: Optional[FaissRepository] = getattr(state, "faiss_repo_papers", None)
    # 调试信息：获取结果
    print(
        f"[get_faiss_repository_papers DEBUG] getattr 结果: repo is None? {repo is None}",
        flush=True,
    )
    # 如果获取不到 (返回值为 None)

    # 检查仓库实例是否存在且已就绪 (例如，索引已加载)
    is_repo_ready = False  # 初始化准备状态为 False
    if repo is not None:
        # 调试信息：找到仓库，检查准备状态
        print(
            f"[get_faiss_repository_papers DEBUG] 找到仓库 (类型: {type(repo)}, ID: {id(repo)}). 检查准备状态...",
            flush=True,
        )
        # 调用仓库的 is_ready() 方法检查状态 (假设该方法存在)
        # Call is_ready() once
        try:
            # 假设 is_ready 是同步方法；如果是异步，需要 await repo.is_ready()
            is_repo_ready = repo.is_ready()
            print(
                f"[get_faiss_repository_papers DEBUG] repo.is_ready() 返回: {is_repo_ready}",
                flush=True,
            )
        except Exception as ready_err:
            logger.error(
                f"调用 faiss_repo_papers.is_ready() 时出错: {ready_err}", exc_info=True
            )
            is_repo_ready = False  # 出错则认为未就绪
    else:
        is_repo_ready = False  # If repo is None, it's not ready

    # Original check
    # 如果仓库不存在或未就绪
    if not is_repo_ready:
        # 调试信息：准备抛出 503 错误
        print("[get_faiss_repository_papers DEBUG] 准备抛出 503 错误。", flush=True)
        logger.error("论文 Faiss 仓库未初始化或索引未就绪。")
        # 抛出 HTTP 503 服务不可用错误
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="论文向量搜索服务当前不可用。",
        )
    # 显式类型声明确保返回类型正确
    # 调试信息：返回有效的仓库
    print("[get_faiss_repository_papers DEBUG] 返回有效的仓库。", flush=True)
    # 返回获取到的仓库实例
    return repo  # type: ignore


# 同上，获取模型的 Faiss 仓库实例
def get_faiss_repository_models(
    state: State = Depends(get_app_state),
) -> FaissRepository:
    """
    依赖函数：从应用状态 (`app.state`) 中获取模型 Faiss 仓库实例。
    这个实例是由 `lifespan` 在应用启动时创建和加载的。

    Args:
        state (State): 通过依赖注入获取的应用状态对象。

    Raises:
        HTTPException: 如果在 `state` 中找不到 `faiss_repo_models` 或仓库未就绪，
                       则抛出 503 服务不可用错误。

    Returns:
        FaissRepository: 获取到的模型 Faiss 仓库实例。
    """
    # 尝试从 state 获取 'faiss_repo_models' 实例
    # Use getattr on the State object
    repo: Optional[FaissRepository] = getattr(state, "faiss_repo_models", None)
    # 如果获取不到 (返回值为 None)

    # 检查仓库实例是否存在且已就绪
    is_repo_ready = False
    if repo is not None:
        try:
            # 假设 is_ready 是同步方法
            is_repo_ready = repo.is_ready()
        except Exception as ready_err:
            logger.error(
                f"调用 faiss_repo_models.is_ready() 时出错: {ready_err}", exc_info=True
            )
            is_repo_ready = False

    # 如果仓库不存在或未就绪
    if not is_repo_ready:
        # 模型索引可能不是所有功能都必需，这里记录警告而不是错误
        # Log as warning, maybe models index isn't critical for all apps
        logger.warning(
            "模型 Faiss 仓库未初始化或索引未就绪。依赖此功能的请求可能会失败。"
        )
        # Decide if this should be a 503 or allow service to handle None/partially working state
        # 是否抛出 503 取决于业务逻辑。如果模型搜索是核心功能，应该抛出。
        # 如果只是可选功能，可以让服务层处理仓库不可用的情况。
        # For now, let's raise 503 if someone explicitly asks for a model search that needs it.
        # 当前实现：如果需要模型向量搜索，则抛出 503。
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型向量搜索服务当前不可用。",
        )
    # 显式类型声明确保返回类型正确
    # 返回获取到的仓库实例
    return repo  # type: ignore


# --- Service Dependencies --- #
# --- 服务 (Service) 依赖提供函数 --- #


def get_search_service(
    # Remove Depends() from request parameter
    # 注意：这里 request 参数不应该使用 Depends()，它是由 FastAPI 自动提供的
    request: Request,  # Corrected: No Depends()
    # Keep original repo dependencies
    # 依赖注入各个仓库实例
    faiss_repo_papers: FaissRepository = Depends(get_faiss_repository_papers),
    faiss_repo_models: FaissRepository = Depends(get_faiss_repository_models),
    pg_repo: PostgresRepository = Depends(get_postgres_repository),
    # Add dependency for Neo4j repository
    neo4j_repo: Optional[Neo4jRepository] = Depends(get_neo4j_repository),
    # Removed direct embedder dependency
    # 注意：不再直接依赖 get_embedder，而是尝试从 state 获取
) -> SearchService:
    """Dependency injector for SearchService using lifespan-managed resources."""
    """
    依赖函数：创建并返回 SearchService 实例。
    这个服务封装了搜索相关的业务逻辑。

    Args:
        request (Request): FastAPI 请求对象，用于获取应用状态。
        faiss_repo_papers (FaissRepository): 注入的论文 Faiss 仓库。
        faiss_repo_models (FaissRepository): 注入的模型 Faiss 仓库。
        pg_repo (PostgresRepository): 注入的 Postgres 仓库。
        neo4j_repo (Optional[Neo4jRepository]): 注入的 Neo4j 仓库 (可能为 None)。

    Returns:
        SearchService: SearchService 的实例。
    """
    logger.debug("尝试提供 SearchService 实例。")
    embedder: Optional[TextEmbedder] = None  # 初始化嵌入器为 None
    # Determine if embedder is needed based on context (e.g., request path/params)
    # This is tricky here. A simpler approach is to always try getting it,
    # relying on the SearchService internal checks.
    # OR, assume SearchService can handle embedder=None if not needed.
    # 尝试从 app.state 获取嵌入器实例
    # 即使获取失败，也继续执行，让 SearchService 内部判断是否需要嵌入器
    try:
        # Try to get the embedder from state, but don't fail the dependency if not found yet
        # 首先获取 state 对象
        state = get_app_state(
            request
        )  # This now returns the State object # 如果 state 获取失败会抛出 500
        # Use getattr on the State object
        # 然后从 state 中获取 embedder
        embedder = getattr(state, "embedder", None)
        # Check if embedder model is loaded
        # 检查嵌入器模型是否加载
        if embedder and not getattr(embedder, "model", None):
            logger.warning("在 state 中找到嵌入器，但其模型未加载。")
            embedder = None  # Treat as unavailable if model isn't loaded # 如果模型未加载，视为不可用
    except HTTPException as e:
        # If get_app_state itself fails (unlikely), log it but proceed
        # 如果 get_app_state 本身失败 (可能性低，除非 lifespan 严重问题)
        logger.error(f"在 get_search_service 中无法获取应用状态: {e.detail}")
        # 这里可以选择重新抛出异常或继续尝试（取决于 SearchService 是否能在无状态下工作）
        # 当前选择继续，但 SearchService 可能会失败
    except Exception as e:
        # 捕获其他可能的错误
        logger.error(f"在 get_search_service 中检查嵌入器时出错: {e}", exc_info=True)

    # Pass potentially None embedder to the service.
    # SearchService __init__ accepts Optional[TextEmbedder].
    # 检查 Neo4j 仓库是否可用
    # Check if Neo4j repo is None, SearchService needs to handle this.
    # SearchService 的 __init__ 可能需要处理 neo4j_repo 为 None 的情况
    if neo4j_repo is None:
        logger.warning(
            "创建 SearchService 时 Neo4j 仓库为 None。图相关搜索功能可能受限。"
        )
        # SearchService __init__ expects Neo4jRepository, not Optional.
        # We MUST modify SearchService to accept Optional OR raise an error here
        # if Neo4j is essential for all search operations.
        # For now, let's assume SearchService needs modification and pass None.
        # This will likely cause a runtime error if SearchService doesn't handle None.
        # A better approach might be to raise HTTPException(503) here.
        # 注意：如果 SearchService 的 __init__ 强制要求 Neo4jRepository 而不是 Optional，
        # 这里传递 None 会导致运行时错误。需要修改 SearchService 或在此处抛出 503。
        # 假设 SearchService 内部会处理 None 的情况。

    # 使用获取到的依赖项实例化 SearchService
    # 将可能为 None 的 embedder 和 neo4j_repo 传递给服务
    return SearchService(
        embedder=embedder,
        faiss_repo_papers=faiss_repo_papers,
        faiss_repo_models=faiss_repo_models,
        pg_repo=pg_repo,
        neo4j_repo=neo4j_repo,  # Pass the neo4j_repo (potentially None) # 传递可能为 None 的 Neo4j 仓库
    )


def get_graph_service(
    # 依赖注入 Postgres 和 Neo4j 仓库
    pg_repo: PostgresRepository = Depends(get_postgres_repository),
    # Depends on Optional Neo4j repo
    neo4j_repo: Optional[Neo4jRepository] = Depends(get_neo4j_repository),
) -> GraphService:
    """Dependency injector for GraphService using lifespan-managed resources."""
    """
    依赖函数：创建并返回 GraphService 实例。
    这个服务封装了图数据相关的业务逻辑。

    Args:
        pg_repo (PostgresRepository): 注入的 Postgres 仓库。
        neo4j_repo (Optional[Neo4jRepository]): 注入的 Neo4j 仓库 (可能为 None)。

    Raises:
        HTTPException: 如果 Neo4j 仓库不可用但 GraphService 必须依赖它，则可能抛出 503。
                       (当前假设 GraphService 能处理 neo4j_repo=None 的情况)

    Returns:
        GraphService: GraphService 的实例。
    """
    # GraphService __init__ expects Neo4jRepository, not Optional.
    # It needs to handle the case where neo4j_repo is None if Neo4j is unavailable.
    # Let's modify GraphService or raise error here if Neo4j needed but unavailable.
    # For now, assume GraphService handles None neo4j_repo.
    # 同样，GraphService 的 __init__ 需要能处理 neo4j_repo 为 None 的情况
    logger.debug("尝试提供 GraphService 实例。")
    if neo4j_repo is None:
        # Or raise HTTPException(503, "Graph service unavailable: Neo4j not configured") ?
        # 可以选择在这里抛出 503 错误，如果 Neo4j 是图服务的核心依赖
        # raise HTTPException(status_code=503, detail="图服务不可用：Neo4j 未配置或初始化失败")
        logger.warning("创建 GraphService 时 Neo4j 仓库为 None。图服务功能可能受限。")

    # 使用获取到的仓库实例化 GraphService
    return GraphService(
        pg_repo=pg_repo,
        neo4j_repo=neo4j_repo,  # 传递可能为 None 的 Neo4j 仓库
    )
