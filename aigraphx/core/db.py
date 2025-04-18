import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
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
) -> AsyncGenerator[None, None]:  # Added settings parameter
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
    try:
        app.state.faiss_repo_papers = FaissRepository(
            index_path=settings.faiss_index_path,
            id_map_path=settings.faiss_mapping_path,
            id_type="int",
        )
        # Check readiness immediately after instantiation
        if not app.state.faiss_repo_papers.is_ready():
            logger.error("Papers Faiss Repository is not ready after initialization.")
            # Log the state from is_ready for more info
            index_exists = app.state.faiss_repo_papers.index is not None
            ntotal = (
                app.state.faiss_repo_papers.index.ntotal
                if app.state.faiss_repo_papers.index
                else -1
            )
            map_exists_and_not_empty = bool(app.state.faiss_repo_papers.id_map)
            logger.error(
                f"[Lifespan Check Failed] Papers Faiss State: index_exists={index_exists}, ntotal={ntotal}, map_not_empty={map_exists_and_not_empty}"
            )
            # Set state to None if not ready
            app.state.faiss_repo_papers = None
            # Do not raise RuntimeError here to allow tests to check state
            logger.warning(
                "Papers Faiss Repository set to None due to initialization failure."
            )
        else:
            logger.info("Papers Faiss Repository initialized and ready.")
    except Exception as e:
        logger.exception(f"Failed to initialize Papers Faiss Repository: {e}")
        app.state.faiss_repo_papers = None  # Ensure state is None on exception

    # Initialize Faiss Repository for Models (Simplified Error Handling)
    logger.info("Initializing Faiss Repository for Models...")
    try:
        app.state.faiss_repo_models = FaissRepository(
            index_path=settings.models_faiss_index_path,
            id_map_path=settings.models_faiss_mapping_path,
            id_type="str",
        )
        # Check readiness immediately after instantiation
        if not app.state.faiss_repo_models.is_ready():
            logger.error("Models Faiss Repository is not ready after initialization.")
            # Log the state from is_ready for more info
            index_exists_m = app.state.faiss_repo_models.index is not None
            ntotal_m = (
                app.state.faiss_repo_models.index.ntotal
                if app.state.faiss_repo_models.index
                else -1
            )
            map_exists_and_not_empty_m = bool(app.state.faiss_repo_models.id_map)
            logger.error(
                f"[Lifespan Check Failed] Models Faiss State: index_exists={index_exists_m}, ntotal={ntotal_m}, map_not_empty={map_exists_and_not_empty_m}"
            )
            # Set state to None if not ready
            app.state.faiss_repo_models = None
            # Do not raise RuntimeError here to allow tests to check state
            logger.warning(
                "Models Faiss Repository set to None due to initialization failure."
            )
        else:
            logger.info("Models Faiss Repository initialized and ready.")
    except Exception as e:
        logger.exception(f"Failed to initialize Models Faiss Repository: {e}")
        app.state.faiss_repo_models = None  # Ensure state is None on exception

    logger.info("Resource initialization process completed.")
    print("--- DEBUG: Lifespan YIELDING (Resources should be initialized) ---")
    yield  # Application runs here
    print("--- DEBUG: Lifespan RESUMING AFTER YIELD ---")

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
