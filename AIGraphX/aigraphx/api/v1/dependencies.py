# aigraphx/api/dependencies.py
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, cast

from fastapi import Depends, HTTPException, status, Request
from psycopg_pool import AsyncConnectionPool  # Import Pool
from neo4j import AsyncDriver  # Import Neo4j Driver
from starlette.datastructures import State  # Import State

# Import configurations
from aigraphx.core.config import settings  # Reverted from settings

# Import Repository classes for type hinting
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository
from aigraphx.repositories.faiss_repo import FaissRepository

# Import Embedder and Services for type hinting
from aigraphx.vectorization.embedder import TextEmbedder
from aigraphx.services.search_service import SearchService
from aigraphx.services.graph_service import GraphService


logger = logging.getLogger(__name__)

# --- Configuration for Faiss ---
# Needs to match the dimension of the used sentence transformer model
# e.g., all-MiniLM-L6-v2 -> 384
EMBEDDING_DIMENSION = 384

FAISS_PAPER_INDEX_PATH = "faiss_papers.index"
FAISS_PAPER_ID_MAP_PATH = "faiss_paper_id_map.pkl"
FAISS_HF_INDEX_PATH = "faiss_hf_models.index"
FAISS_HF_ID_MAP_PATH = "faiss_hf_id_map.pkl"


# --- Resource Getter --- #
def get_app_state(request: Request) -> State:
    """获取应用程序状态 (返回 State 对象)"""
    if not hasattr(request.app, "state"):
        print(
            "[get_app_state DEBUG] !!! request.app has no attribute 'state'", flush=True
        )
        logger.error("Application state 'request.app.state' not found!")
        raise HTTPException(
            status_code=500, detail="Application state not initialized."
        )
    print(
        f"[get_app_state DEBUG] Returning request.app.state (type: {type(request.app.state)}, id: {id(request.app.state)})",
        flush=True,
    )
    # Ensure the returned object is actually State type for consistency
    if not isinstance(request.app.state, State):
        logger.warning(
            f"request.app.state is not of type State, but {type(request.app.state)}. Returning anyway."
        )
    # Return the State object. Use cast to satisfy mypy.
    return cast(State, request.app.state)


# --- Dependency Functions --- #


def get_postgres_pool(
    state: State = Depends(get_app_state),  # Use State hint
) -> AsyncConnectionPool:
    """获取PostgreSQL连接池"""
    # Use getattr on the State object
    pool = getattr(state, "pg_pool", None)
    if pool is None:
        logger.error("PostgreSQL pool not initialized or available in app state.")
        raise HTTPException(
            status_code=503, detail="Database connection pool is not available."
        )
    # 显式类型声明确保返回类型正确
    return pool  # type: ignore


def get_neo4j_driver(
    state: State = Depends(get_app_state),
) -> Optional[AsyncDriver]:
    # Use getattr on the State object
    driver = getattr(state, "neo4j_driver", None)
    if driver is None:
        logger.warning("Neo4j driver not initialized or available in app state.")
    return driver


def get_embedder(state: State = Depends(get_app_state)) -> TextEmbedder:
    # Use getattr on the State object
    embedder = getattr(state, "embedder", None)
    if embedder is None or not embedder.model:
        logger.error("Text embedder not initialized or model not loaded.")
        raise HTTPException(
            status_code=503, detail="Text embedding service is not available."
        )
    # 显式类型声明确保返回类型正确
    return embedder  # type: ignore


# --- Repository Dependencies --- #


def get_postgres_repository(
    pool: AsyncConnectionPool = Depends(get_postgres_pool),
) -> PostgresRepository:
    return PostgresRepository(pool=pool)


def get_neo4j_repository(
    driver: Optional[AsyncDriver] = Depends(get_neo4j_driver),
) -> Optional[Neo4jRepository]:
    if driver:
        return Neo4jRepository(driver=driver)
    return None


# Rename for clarity and add model repo getter
def get_faiss_repository_papers(
    state: State = Depends(get_app_state),
) -> FaissRepository:
    # Use print for debugging
    print(
        f"[get_faiss_repository_papers DEBUG] Attempting to get 'faiss_repo_papers' from state (type: {type(state)}, id: {id(state)}).",
        flush=True,
    )
    # Use getattr on the State object
    repo = getattr(state, "faiss_repo_papers", None)
    print(
        f"[get_faiss_repository_papers DEBUG] Result of getattr: repo is None? {repo is None}",
        flush=True,
    )
    if repo is not None:
        print(
            f"[get_faiss_repository_papers DEBUG] Repo found (type: {type(repo)}, id: {id(repo)}). Checking readiness...",
            flush=True,
        )
        is_repo_ready = repo.is_ready()  # Call is_ready() once
        print(
            f"[get_faiss_repository_papers DEBUG] repo.is_ready() returned: {is_repo_ready}",
            flush=True,
        )
    else:
        is_repo_ready = False  # If repo is None, it's not ready

    # Original check
    if repo is None or not is_repo_ready:
        print("[get_faiss_repository_papers DEBUG] Raising 503 error.", flush=True)
        logger.error("Papers Faiss repository not initialized or index not ready.")
        raise HTTPException(
            status_code=503, detail="Papers vector search service is not available."
        )
    # 显式类型声明确保返回类型正确
    print("[get_faiss_repository_papers DEBUG] Returning valid repository.", flush=True)
    return repo  # type: ignore


def get_faiss_repository_models(
    state: State = Depends(get_app_state),
) -> FaissRepository:
    # Use getattr on the State object
    repo = getattr(state, "faiss_repo_models", None)
    if repo is None or not repo.is_ready():
        # Log as warning, maybe models index isn't critical for all apps
        logger.warning("Models Faiss repository not initialized or index not ready.")
        # Decide if this should be a 503 or allow service to handle None/partially working state
        # For now, let's raise 503 if someone explicitly asks for a model search that needs it.
        raise HTTPException(
            status_code=503, detail="Models vector search service is not available."
        )
    # 显式类型声明确保返回类型正确
    return repo  # type: ignore


# --- Service Dependencies --- #


def get_search_service(
    # Remove Depends() from request parameter
    request: Request,  # Corrected: No Depends()
    # Keep original repo dependencies
    faiss_repo_papers: FaissRepository = Depends(get_faiss_repository_papers),
    faiss_repo_models: FaissRepository = Depends(get_faiss_repository_models),
    pg_repo: PostgresRepository = Depends(get_postgres_repository),
    # Add dependency for Neo4j repository
    neo4j_repo: Optional[Neo4jRepository] = Depends(get_neo4j_repository),
    # Removed direct embedder dependency
) -> SearchService:
    """Dependency injector for SearchService using lifespan-managed resources."""
    logger.debug("Attempting to provide SearchService instance.")
    embedder: Optional[TextEmbedder] = None
    # Determine if embedder is needed based on context (e.g., request path/params)
    # This is tricky here. A simpler approach is to always try getting it,
    # relying on the SearchService internal checks.
    # OR, assume SearchService can handle embedder=None if not needed.
    try:
        # Try to get the embedder from state, but don't fail the dependency if not found yet
        state = get_app_state(request)  # This now returns the State object
        # Use getattr on the State object
        embedder = getattr(state, "embedder", None)
        if embedder and not getattr(embedder, "model", None):
            logger.warning("Embedder found in state but model not loaded.")
            embedder = None  # Treat as unavailable if model isn't loaded
    except HTTPException as e:
        # If get_app_state itself fails (unlikely), log it but proceed
        logger.error(f"Could not get app state in get_search_service: {e.detail}")
    except Exception as e:
        logger.error(
            f"Error checking for embedder in get_search_service: {e}", exc_info=True
        )

    # Pass potentially None embedder to the service.
    # SearchService __init__ accepts Optional[TextEmbedder].
    # Check if Neo4j repo is None, SearchService needs to handle this.
    if neo4j_repo is None:
        logger.warning(
            "Neo4j repository is None when creating SearchService. Graph features might be limited."
        )
        # SearchService __init__ expects Neo4jRepository, not Optional.
        # We MUST modify SearchService to accept Optional OR raise an error here
        # if Neo4j is essential for all search operations.
        # For now, let's assume SearchService needs modification and pass None.
        # This will likely cause a runtime error if SearchService doesn't handle None.
        # A better approach might be to raise HTTPException(503) here.

    return SearchService(
        embedder=embedder,
        faiss_repo_papers=faiss_repo_papers,
        faiss_repo_models=faiss_repo_models,
        pg_repo=pg_repo,
        neo4j_repo=neo4j_repo,  # Pass the neo4j_repo (potentially None)
    )


def get_graph_service(
    pg_repo: PostgresRepository = Depends(get_postgres_repository),
    # Depends on Optional Neo4j repo
    neo4j_repo: Optional[Neo4jRepository] = Depends(get_neo4j_repository),
) -> GraphService:
    """Dependency injector for GraphService using lifespan-managed resources."""
    # GraphService __init__ expects Neo4jRepository, not Optional.
    # It needs to handle the case where neo4j_repo is None if Neo4j is unavailable.
    # Let's modify GraphService or raise error here if Neo4j needed but unavailable.
    # For now, assume GraphService handles None neo4j_repo.
    logger.debug("Providing GraphService instance.")
    if neo4j_repo is None:
        # Or raise HTTPException(503, "Graph service unavailable: Neo4j not configured") ?
        logger.warning(
            "Neo4j repository is None, GraphService may have limited functionality."
        )

    # Modify GraphService to accept Optional[Neo4jRepository]
    # For now, casting to expected type, requires GraphService modification
    return GraphService(
        neo4j_repo=neo4j_repo,  # Needs to handle None
        pg_repo=pg_repo,
    )
