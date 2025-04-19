# tests/conftest.py
import pytest
import pytest_asyncio  # Import pytest_asyncio
from fastapi import FastAPI
from functools import partial  # Import partial
import httpx  # Ensure httpx is imported
from httpx import AsyncClient, ASGITransport  # Restore async client import
from unittest.mock import AsyncMock, MagicMock  # Import MagicMock for Neo4j
from typing import (
    AsyncGenerator,
    Generator,
    Dict,
    Any,
    Optional,
    Tuple,
    Union,
)  # 添加Generator和数据库相关类型
import os
import json
import datetime
import asyncio
import subprocess  # Import subprocess
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv
import logging
from pytest_mock import MockerFixture  # Import MockerFixture type
import sys
import faiss  # Add imports for Faiss
import numpy as np
from pathlib import Path  # Add Path import
from unittest.mock import patch
from starlette.datastructures import State  # Import State for mock_app
from contextlib import asynccontextmanager
import inspect  # Import inspect module

# Import Neo4j Driver
from neo4j import AsyncGraphDatabase, AsyncDriver, basic_auth  # Import basic_auth

# Import the app instance directly
# from aigraphx.main import app as main_app # Don't import app directly to avoid early settings load

# Import components needed to build the app for tests
from aigraphx.core.db import lifespan  # Import lifespan
from aigraphx.core.config import Settings  # Import Settings class
from aigraphx.api.v1.api import api_router as api_v1_router

# Import the GraphService for the mock fixture spec
from aigraphx.services.graph_service import GraphService

# Import the PostgresRepository and Neo4jRepository for integration test fixture
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository  # Import Neo4j repo

# Import the original dependency getter functions
# from aigraphx.api.v1 import deps # Incorrect path
from aigraphx.api.v1.dependencies import get_graph_service  # Correct path
from aigraphx.vectorization.embedder import TextEmbedder  # Import TextEmbedder

# Import the dependency getter function to override
from aigraphx.api.v1.dependencies import get_embedder

# Import the lifespan function to test
from aigraphx.core.db import lifespan

# Import classes being mocked
# from aigraphx.vectorization.embedder import TextEmbedder # No longer needed to mock here for lifespan tests
from aigraphx.repositories.faiss_repo import FaissRepository  # Added this import

# Import asgi-lifespan
from asgi_lifespan import LifespanManager

# --- Logging Setup (Optional but helpful) ---
logger = logging.getLogger(__name__)  # Logger for conftest itself

# ADDED: Debug log for raw environment variable at module load time
logger_conftest_top = logging.getLogger("conftest_top")
raw_test_neo4j_db_env = os.getenv("TEST_NEO4J_DATABASE")
logger_conftest_top.critical(
    f"[CONTEST TOP] Raw os.getenv('TEST_NEO4J_DATABASE'): '{raw_test_neo4j_db_env}'"
)

# --- Load .env early - For getting TEST URLs ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)
TEST_DB_URL_FROM_ENV = os.getenv("TEST_DATABASE_URL")
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI")
TEST_NEO4J_USER = os.getenv("TEST_NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD")


# --- Session-Scoped Lock for Database Cleanup ---
@pytest.fixture(scope="session")
def db_cleanup_lock() -> asyncio.Lock:
    """Provides a session-scoped asyncio Lock to serialize DB cleanup operations."""
    logger.info("[db_cleanup_lock fixture] Creating session-scoped lock.")
    return asyncio.Lock()


# --- Fixture for Temporary Faiss Files (Session scope for efficiency) ---
@pytest.fixture(scope="session")
def temp_faiss_files(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, str]:
    """Creates temporary, valid Faiss index and ID map files for testing."""
    # Use session-scoped temp path
    base_path = tmp_path_factory.mktemp("faiss_test_data")
    logger.info(f"[temp_faiss_files] Creating temp Faiss files in: {base_path}")

    dim = 384  # Example dimension, should match embedder
    num_vectors = 5

    # --- Create Papers Faiss Files ---
    paper_index_path = base_path / "test_papers.index"
    paper_map_path = base_path / "test_papers_map.json"

    # Create a simple Faiss index
    try:
        paper_index = faiss.IndexFlatL2(dim)
        vectors = np.random.rand(num_vectors, dim).astype("float32")
        paper_index.add(vectors)
        faiss.write_index(paper_index, str(paper_index_path))
        logger.info(f"[temp_faiss_files] Created temp paper index: {paper_index_path}")
    except Exception as e:
        logger.error(f"Failed to create temp paper Faiss index: {e}")
        pytest.fail(f"Failed to create temp paper Faiss index: {e}")

    # Create a simple ID map (Faiss index to paper ID)
    paper_id_map = {
        i: i + 100 for i in range(num_vectors)
    }  # Example: 0 -> 100, 1 -> 101, ...
    try:
        with open(paper_map_path, "w") as f:
            json.dump(paper_id_map, f)
        logger.info(f"[temp_faiss_files] Created temp paper map: {paper_map_path}")
    except Exception as e:
        logger.error(f"Failed to create temp paper Faiss map: {e}")
        pytest.fail(f"Failed to create temp paper Faiss map: {e}")

    # --- Create Models Faiss Files ---
    model_index_path = base_path / "test_models.index"
    model_map_path = base_path / "test_models_map.json"

    # Create a simple Faiss index for models
    try:
        model_index = faiss.IndexFlatL2(dim)
        model_vectors = np.random.rand(num_vectors, dim).astype("float32")
        model_index.add(model_vectors)
        faiss.write_index(model_index, str(model_index_path))
        logger.info(f"[temp_faiss_files] Created temp model index: {model_index_path}")
    except Exception as e:
        logger.error(f"Failed to create temp model Faiss index: {e}")
        pytest.fail(f"Failed to create temp model Faiss index: {e}")

    # Create a simple ID map (Faiss index to model ID string)
    model_id_map = {i: f"model_{i}" for i in range(num_vectors)}
    try:
        with open(model_map_path, "w") as f:
            json.dump(model_id_map, f)
        logger.info(f"[temp_faiss_files] Created temp model map: {model_map_path}")
    except Exception as e:
        logger.error(f"Failed to create temp model Faiss map: {e}")
        pytest.fail(f"Failed to create temp model Faiss map: {e}")

    return {
        "paper_index": str(paper_index_path),
        "paper_map": str(paper_map_path),
        "model_index": str(model_index_path),
        "model_map": str(model_map_path),
    }


# --- Fixture for Test Settings (Updated to use temp Faiss files) ---
@pytest.fixture(scope="session")
def test_settings(temp_faiss_files: Dict[str, str]) -> Settings:
    """Creates a Settings instance with test-specific overrides,
    including paths to temporary Faiss files.
    Relies on BaseSettings loading from environment first, then overrides paths.
    """
    settings_from_env = Settings()
    # ADDED: Log values immediately after Settings() instantiation
    logger.info(
        f"[test_settings fixture] Settings() loaded initial values: test_db_url='{settings_from_env.test_database_url}', db_url='{settings_from_env.database_url}', test_neo4j_uri='{settings_from_env.test_neo4j_uri}', neo4j_uri='{settings_from_env.neo4j_uri}', test_neo4j_pwd='{settings_from_env.test_neo4j_password}', neo4j_pwd='{settings_from_env.neo4j_password}', test_neo4j_db='{settings_from_env.test_neo4j_database}', neo4j_db='{settings_from_env.neo4j_database}'"
    )

    # Override only the necessary paths and environment marker
    # Use test_database_url loaded by Settings if available
    original_db_url = settings_from_env.database_url
    settings_from_env.database_url = (
        settings_from_env.test_database_url or settings_from_env.database_url
    )
    logger.info(
        f"[test_settings fixture] DB URL override: test='{settings_from_env.test_database_url}', original='{original_db_url}', final='{settings_from_env.database_url}'"
    )

    # Use test_neo4j_uri loaded by Settings if available
    original_neo4j_uri = settings_from_env.neo4j_uri
    settings_from_env.neo4j_uri = (
        settings_from_env.test_neo4j_uri or settings_from_env.neo4j_uri
    )
    logger.info(
        f"[test_settings fixture] Neo4j URI override: test='{settings_from_env.test_neo4j_uri}', original='{original_neo4j_uri}', final='{settings_from_env.neo4j_uri}'"
    )

    # Use TEST_NEO4J_USER from top-level load (assuming it's correct or default works)
    # This still uses the top-level loaded var, maybe risky?
    # Let's trust Settings() load for username too for consistency
    # settings_from_env.neo4j_username = TEST_NEO4J_USER # Already has default
    # logger.info(f"[test_settings fixture] Neo4j User kept from top-level/default: '{settings_from_env.neo4j_username}'")
    # No override needed usually for user, Settings() default or .env should suffice.

    # Use test_neo4j_password loaded by Settings if available
    original_neo4j_pwd = settings_from_env.neo4j_password
    settings_from_env.neo4j_password = (
        settings_from_env.test_neo4j_password or settings_from_env.neo4j_password
    )
    # Avoid logging actual password, just log if override happened
    pwd_overridden = (
        original_neo4j_pwd != settings_from_env.neo4j_password
        and settings_from_env.test_neo4j_password is not None
    )
    logger.info(
        f"[test_settings fixture] Neo4j Pwd override: test_pwd_set={settings_from_env.test_neo4j_password is not None}, overridden={pwd_overridden}"
    )

    # Use test_neo4j_database loaded by Settings if available
    original_neo4j_db = settings_from_env.neo4j_database
    if settings_from_env.test_neo4j_database:
        logger.info(
            f"[test_settings fixture] Overriding neo4j_database with test_neo4j_database ('{settings_from_env.test_neo4j_database}')"
        )
        settings_from_env.neo4j_database = settings_from_env.test_neo4j_database
    else:
        logger.info(
            f"[test_settings fixture] Keeping original neo4j_database ('{original_neo4j_db}') as test_neo4j_database is None/empty."
        )

    settings_from_env.faiss_index_path = temp_faiss_files["paper_index"]
    settings_from_env.faiss_mapping_path = temp_faiss_files["paper_map"]
    settings_from_env.models_faiss_index_path = temp_faiss_files["model_index"]
    settings_from_env.models_faiss_mapping_path = temp_faiss_files["model_map"]
    settings_from_env.environment = "test"

    # Validate required Neo4j settings if URI is present
    if settings_from_env.neo4j_uri and not settings_from_env.neo4j_password:
        logger.warning("Neo4j URI is set but password is not.")
        # Decide behavior: skip Neo4j tests or use default password?
        # For now, assume a default password might exist or is handled elsewhere.

    # Log the final effective Neo4j DB name for debugging
    logger.info(
        f"[test_settings fixture] Effective Neo4j DB name: '{settings_from_env.neo4j_database}'"
    )

    return settings_from_env


# --- Fixture for a Session-Scoped Loaded Text Embedder ---
@pytest.fixture(scope="session")
def loaded_text_embedder(
    test_settings: Settings,
) -> Generator[TextEmbedder, None, None]:
    """
    Initializes and loads the TextEmbedder model once per session.
    Uses configuration from test_settings.
    Used by both API tests (via dependency injection override)
    and non-API tests directly.
    """
    logger.info(
        "[loaded_text_embedder fixture] Initializing TextEmbedder for session..."
    )
    # Ensure relevant settings are available
    # Use the correct setting name from Settings model
    if not test_settings.sentence_transformer_model:
        logger.error(
            "[loaded_text_embedder fixture] sentence_transformer_model not set in test_settings."
        )
        pytest.fail(
            "sentence_transformer_model is required but not set in test settings."
        )

    embedder = TextEmbedder(
        model_name=test_settings.sentence_transformer_model,
        # cache_dir=test_settings.embedding_model_cache_dir, # Removed: Attribute does not exist on Settings
        # Pass other relevant settings if needed, e.g., device
        device=test_settings.embedder_device,
    )
    try:
        # Model is loaded automatically in __init__ via _load_model()
        # logger.info(f"[loaded_text_embedder fixture] Loading model: {test_settings.sentence_transformer_model}...")
        # embedder.load_model() # No need to call load_model explicitly

        # Check if the model attribute is populated after initialization
        if embedder.model is not None:
            logger.info(
                f"[loaded_text_embedder fixture] TextEmbedder model '{test_settings.sentence_transformer_model}' initialized successfully for session."
            )
            yield embedder
        else:
            # If model is None, _load_model must have failed and logged an error
            logger.error(
                "[loaded_text_embedder fixture] embedder.model is None after initialization. Check previous logs for loading errors."
            )
            pytest.fail(
                "Failed to initialize TextEmbedder model in session-scoped fixture (model attribute is None)."
            )

    except Exception as e:
        # Catch potential errors during __init__ itself, though _load_model handles its own exceptions
        logger.exception(
            f"[loaded_text_embedder fixture] Error during TextEmbedder initialization: {e}"
        )
        pytest.fail(
            f"Failed during TextEmbedder initialization in session-scoped fixture: {e}"
        )


# --- Fixture for a Session-Scoped Test Application ---
@pytest.fixture(scope="session")
def test_app(
    test_settings: Settings,
    loaded_text_embedder: TextEmbedder,  # Depend on the loaded embedder
) -> FastAPI:
    """Creates a FastAPI application instance for the test session.

    - Uses test_settings.
    - Overrides TextEmbedder dependency with the session-loaded one.
    - Attaches the lifespan context manager.
    """
    # Use a partial function to pass test_settings to the main app factory/lifespan
    # This assumes lifespan can accept settings or they are globally accessible via config module
    configured_lifespan = partial(lifespan, settings=test_settings)

    # Explicitly create state if needed by lifespan or dependencies
    initial_state = State()
    # Example: if lifespan expects embedder in state, set it (though override is preferred)
    # initial_state.embedder = loaded_text_embedder

    print("[test_app fixture DEBUG] Preparing lifespan...")
    # REMOVED: bound_lifespan = asynccontextmanager(configured_lifespan)
    # Log the ID of the settings object bound to the lifespan
    print(
        f"[test_app fixture DEBUG] Prepared lifespan with bound test_settings ID: {id(test_settings)}"
    )

    app = FastAPI(
        title="Test AIGraphX",
        lifespan=configured_lifespan,  # Use the partial function directly
        state=initial_state,  # Pass initial state if created
    )
    print(f"[test_app fixture DEBUG] FastAPI instance created with ID: {id(app)}")

    # --- Dependency Override for TextEmbedder ---
    # Ensure any part of the test app requesting the embedder gets the pre-loaded one
    app.dependency_overrides[get_embedder] = lambda: loaded_text_embedder
    print(f"[test_app fixture DEBUG] Applied dependency override for get_embedder.")

    # Check if lifespan got attached correctly
    lifespan_attached = getattr(app.router, "lifespan_context", None) is not None
    print(
        f"[test_app fixture DEBUG] Lifespan context attached to app.router: {lifespan_attached}"
    )

    app.include_router(api_v1_router, prefix="/api/v1")  # CHANGED: Correct prefix
    print(f"[test_app fixture DEBUG] API v1 router included.")

    print(f"[test_app fixture DEBUG] Returning test app instance.")
    return app


# --- Fixture for Async HTTP Client ---
@pytest_asyncio.fixture(scope="function")  # CHANGED: Reverted to function scope
async def client(
    test_app: FastAPI,  # Depends on session-scoped test_app
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Creates an httpx.AsyncClient for testing the API.
    Uses asgi-lifespan to manage the application's lifespan within the client's context.
    """
    # The lifespan is managed by the session-scoped test_app fixture. # <-- This assumption is incorrect (Keeping old comment for context)
    # This function-scoped client simply uses the already configured test_app. # <-- Incorrect assumption
    print(
        "\\n--- DEBUG: Running client fixture (function scope) ---"
    )  # Updated debug print
    # Use LifespanManager to wrap the app and handle startup/shutdown
    async with LifespanManager(test_app):
        # Transport uses the (now lifespan-managed) app
        transport = ASGITransport(app=test_app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as async_client:
            print(
                "--- DEBUG: AsyncClient created (function scope, lifespan managed) ---"
            )  # Updated debug print
            yield async_client
    # No explicit close needed for ASGITransport client normally
    print(
        "--- DEBUG: AsyncClient closed (function scope, lifespan ended) ---"
    )  # Updated debug print


# --- Mock Fixture for GraphService ---
@pytest_asyncio.fixture
async def mock_graph_service_fixture(
    test_app: FastAPI,
) -> AsyncGenerator[AsyncMock, None]:
    """Provides a mock GraphService and handles dependency overrides on the test_app."""
    mock_service = AsyncMock(spec=GraphService)

    # --- Store original overrides and apply mock ---
    original_overrides = test_app.dependency_overrides.copy()
    override_key = get_graph_service
    test_app.dependency_overrides[override_key] = lambda: mock_service

    yield mock_service  # Provide the mock to the test

    # --- Restore original overrides after test ---
    test_app.dependency_overrides = original_overrides


# --- Fixture to Apply Alembic Migrations (Module Scope) ---
@pytest.fixture(scope="module", autouse=True)
def apply_migrations() -> None:
    """Ensures Alembic migrations are applied once per module to the TEST database."""
    logger.info("--- Running apply_migrations fixture (module scope) ---")
    # Use the globally read TEST_DB_URL_FROM_ENV
    logger.info(
        f"[apply_migrations] Using globally read TEST_DATABASE_URL: {TEST_DB_URL_FROM_ENV}"
    )

    if not TEST_DB_URL_FROM_ENV:
        logger.warning("TEST_DATABASE_URL not set (globally), skipping migrations.")
        return  # Skip if URL not set

    logger.info(
        f"Applying Alembic migrations to TEST database defined by TEST_DATABASE_URL."
    )
    db_url_for_alembic = TEST_DB_URL_FROM_ENV  # Use global value
    if db_url_for_alembic and db_url_for_alembic.startswith("postgresql://"):
        db_url_for_alembic = db_url_for_alembic.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    logger.info(f"[apply_migrations] URL used for Alembic: {db_url_for_alembic}")

    alembic_env = os.environ.copy()
    alembic_env["DATABASE_URL"] = db_url_for_alembic

    # --- Log the PATH passed to subprocess ---
    conda_prefix = os.environ.get("CONDA_PREFIX")  # Get conda prefix directly
    if conda_prefix:
        # The bin path is directly under the prefix
        expected_bin_path = os.path.join(conda_prefix, "bin")
    else:  # Fallback if CONDA_PREFIX is not set
        # Try finding based on sys.executable (less reliable)
        conda_base_path = os.path.dirname(os.path.dirname(sys.executable))
        expected_bin_path = os.path.join(conda_base_path, "bin")
        logger.warning(
            f"[apply_migrations] CONDA_PREFIX not set, guessing bin path: {expected_bin_path}"
        )

    logger.info(f"[apply_migrations] Checking for Conda bin path: {expected_bin_path}")
    current_path = alembic_env.get(
        "PATH", "PATH key not found!"
    )  # Get PATH from the env passed to subprocess
    if expected_bin_path not in current_path:
        logger.warning(
            f"[apply_migrations] Expected Conda bin path '{expected_bin_path}' NOT FOUND in PATH! PATH was: {current_path}"
        )
    else:
        logger.info(
            f"[apply_migrations] Expected Conda bin path '{expected_bin_path}' FOUND in PATH."
        )
    # --- End PATH logging ---

    try:
        logger.info("Running: alembic upgrade head")
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            env=alembic_env,
            check=True,  # Keep check=True to fail on non-zero exit code
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info("Alembic upgrade command finished successfully (exit code 0).")
        logger.info("Alembic upgrade head stdout:\n%s", result.stdout)
        # Log stderr even on success, as it might contain warnings
        if result.stderr:
            logger.warning(
                "Alembic upgrade head stderr (command succeeded):\n%s", result.stderr
            )
        logger.info("Alembic migrations applied successfully (module scope).")
    except FileNotFoundError:
        logger.error("'alembic' command not found during module setup.")
        pytest.fail("'alembic' command not found. Ensure it's installed and on PATH.")
    except subprocess.TimeoutExpired as e:
        logger.error(f"Alembic upgrade timed out during module setup: {e.timeout}s.")
        # Log output before failing
        if e.stdout:
            logger.error("Alembic stdout before timeout:\n%s", e.stdout)
        if e.stderr:
            logger.error("Alembic stderr before timeout:\n%s", e.stderr)
        pytest.fail("Alembic upgrade timed out during module setup.")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Alembic upgrade failed during module setup (code {e.returncode})!"
        )
        logger.error("Alembic stdout:\n%s", e.stdout)
        logger.error("Alembic stderr:\n%s", e.stderr)
        pytest.fail(f"Alembic upgrade head failed during module setup: {e}")
    except Exception as e:
        logger.exception(
            "An unexpected error occurred during Alembic migration (module setup)."
        )
        pytest.fail(f"Unexpected error during Alembic migration (module setup): {e}")
    logger.info("--- Finished apply_migrations fixture (module scope) ---")


@pytest_asyncio.fixture(scope="function")
async def db_pool(
    test_settings: Settings,
) -> AsyncGenerator[AsyncConnectionPool, None]:  # Depends on test_settings
    """Manages an async connection pool for the TEST PostgreSQL database for each test function.
    Uses the database URL from test_settings.
    Assumes migrations have been applied by the module-scoped 'apply_migrations' fixture.
    """
    # Use the database URL from test_settings
    test_db_url = test_settings.database_url
    logger.info(
        f"[db_pool fixture - function scope] Creating pool for test_settings URL: {test_db_url}"
    )

    if not test_db_url:
        pytest.skip("Database URL not found in test_settings, skipping pool creation.")

    pool: Optional[AsyncConnectionPool] = None
    try:
        pool = AsyncConnectionPool(
            conninfo=test_db_url,  # Use URL from test_settings
            min_size=test_settings.pg_pool_min_size,  # Use pool size from settings
            max_size=test_settings.pg_pool_max_size,
            open=False,
            timeout=60,  # Consider making this configurable via settings too
        )
        logger.info("[db_pool fixture] Opening connection pool...")
        await pool.open(wait=True)
        logger.info(
            "[db_pool fixture] TEST database connection pool opened successfully."
        )

        yield pool  # Provide the pool to the test function

    except Exception as e:
        logger.exception(
            f"[db_pool fixture] Error creating/opening TEST DB pool for {test_db_url}: {e}"
        )
        pytest.fail(f"Failed to create/open TEST DB pool: {e}")

    finally:
        if pool:
            logger.info("[db_pool fixture] Closing connection pool...")
            await pool.close()
            logger.info("[db_pool fixture] TEST database connection pool closed.")


@pytest_asyncio.fixture()
async def repository(
    db_pool: AsyncConnectionPool,
    db_cleanup_lock: asyncio.Lock,
) -> AsyncGenerator[PostgresRepository, None]:
    """Creates a PostgresRepository instance using the test database pool.

    Cleans relevant tables BEFORE yielding the repository for test isolation,
    using a lock to prevent concurrent cleanup operations.
    (Post-test cleanup REMOVED for simplification during debugging).
    """
    if not db_pool:
        pytest.skip("Test database pool is not available.")

    # Safely get calling test name
    frame = inspect.currentframe()
    func_name = "unknown_test" # Default value
    if frame and frame.f_back:
        caller_frame = frame.f_back
        if caller_frame:
             func_name = caller_frame.f_code.co_name

    # --- Pre-test Cleanup --- (Keep pre-test cleanup)
    logger.info(f"[repository fixture for {func_name}] PRE-TEST: Attempting to acquire DB cleanup lock...")
    try:
        async with db_cleanup_lock: # Acquire lock
            logger.info(f"[repository fixture for {func_name}] PRE-TEST: DB cleanup lock ACQUIRED. Cleaning tables...")
            try:
                async with db_pool.connection() as conn:
                    logger.info(f"[repository fixture for {func_name}] PRE-TEST: Acquired PG connection for TRUNCATE.")
                    async with conn.cursor() as cur:
                        logger.info(f"[repository fixture for {func_name}] PRE-TEST: Executing TRUNCATE...")
                        await cur.execute(
                            """
                            TRUNCATE TABLE model_paper_links, hf_models, papers,
                                         pwc_tasks, pwc_datasets, pwc_repositories
                                         RESTART IDENTITY CASCADE;
                            """
                        )
                        logger.info(f"[repository fixture for {func_name}] PRE-TEST: TRUNCATE command executed.")
                    await conn.commit()
                    logger.info(f"[repository fixture for {func_name}] PRE-TEST: COMMIT executed after TRUNCATE.")
                logger.info(f"[repository fixture for {func_name}] PRE-TEST: Tables truncated and committed successfully.")
            except Exception as e:
                logger.error(f"[repository fixture for {func_name}] PRE-TEST: Error truncating tables (while holding lock): {e}", exc_info=True)
                pytest.fail(f"[repository fixture for {func_name}] PRE-TEST: Failed to clean test database: {e}")
    finally:
         logger.info(f"[repository fixture for {func_name}] PRE-TEST: DB cleanup lock released.")

    # Yield the repository
    repo = PostgresRepository(pool=db_pool)
    logger.info(f"[repository fixture for {func_name}] Yielding repository instance.")
    yield repo

    logger.info(f"[repository fixture for {func_name}] Teardown complete (NO post-test cleanup).")


# --- Fixture for Neo4j Driver (Function Scope) ---
@pytest_asyncio.fixture(scope="function")  # CHANGED: Back to function scope
async def neo4j_driver(test_settings: Settings) -> AsyncGenerator[AsyncDriver, None]:
    """
    Provides an async Neo4j driver instance for a single test function.
    Ensures the driver is closed after the test.
    Uses function scope for better test isolation regarding event loops.
    """
    # --- Neo4j Connection Details ---
    # NOTE: Due to Neo4j Community Edition limitations, tests run against the
    # default 'neo4j' database within the test container. We cannot programmatically
    # create separate logical databases for testing.
    # Isolation is achieved by clearing the database before each test runs
    # (e.g., using MATCH (n) DETACH DELETE n in relevant test setups or fixtures).
    neo4j_uri = test_settings.neo4j_uri
    # Ensure username and password are not None before creating auth tuple
    neo4j_user = test_settings.neo4j_username
    neo4j_pwd = test_settings.neo4j_password
    neo4j_db = (
        test_settings.neo4j_database
    )  # Uses the configured test DB (defaults to 'neo4j')

    if (
        not neo4j_uri or not neo4j_user or not neo4j_pwd
    ):  # Check if URI, user or password is None/empty
        logger.error(
            "Neo4j URI, username or password not configured in test settings. Skipping Neo4j tests that need a driver."
        )
        pytest.skip("Neo4j URI, username or password not configured for tests.")
        yield None  # Should not be reached due to skip, but satisfies type checker
        return  # Explicit return

    # Use basic_auth helper for clarity and type safety
    neo4j_auth_obj = basic_auth(neo4j_user, neo4j_pwd)

    logger.debug(
        f"[neo4j_driver fixture] Creating Neo4j driver for DB '{neo4j_db}' at {neo4j_uri}"
    )
    driver: Optional[AsyncDriver] = None
    try:
        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=neo4j_auth_obj)
        # Optional: Verify connectivity once per driver instance
        try:
            await driver.verify_connectivity()
            logger.debug(
                f"[neo4j_driver fixture] Neo4j driver connectivity verified for DB '{neo4j_db}'"
            )
        except Exception as e:
            logger.error(
                f"[neo4j_driver fixture] Neo4j driver connectivity check failed for DB '{neo4j_db}': {e}",
                exc_info=True,
            )
            pytest.fail(f"Neo4j driver connectivity check failed: {e}")

        yield driver
    except Exception as e:
        logger.error(
            f"[neo4j_driver fixture] Error creating Neo4j driver: {e}", exc_info=True
        )
        pytest.fail(f"Failed to create Neo4j driver: {e}")
        yield None  # Should not be reached
    finally:
        if driver:
            logger.debug(
                f"[neo4j_driver fixture] Closing Neo4j driver for DB '{neo4j_db}'."
            )
            try:
                # --- REVERTED: Back to simple await ---
                await driver.close()
                # --------------------------------------
                logger.debug(
                    f"[neo4j_driver fixture] Neo4j driver closed for DB '{neo4j_db}'."
                )
            except RuntimeError as e:
                # Log potential runtime errors during close (e.g., loop already closed/different loop)
                logger.error(
                    f"[neo4j_driver fixture] Runtime error closing Neo4j driver for DB '{neo4j_db}': {e}",
                    exc_info=True,
                )
                # Do not fail the test here, as this might be the known issue we are temporarily ignoring
            except Exception as e:
                logger.error(
                    f"[neo4j_driver fixture] Unexpected error closing Neo4j driver for DB '{neo4j_db}': {e}",
                    exc_info=True,
                )
                # Potentially fail the test run here as well, depending on strictness
                # pytest.fail(f"Unexpected error closing Neo4j driver: {e}")


@pytest_asyncio.fixture(scope="function")  # CHANGED: Back to function scope
async def neo4j_repo_fixture(
    neo4j_driver: AsyncDriver,
) -> AsyncGenerator[Neo4jRepository, None]:  # Return type fixed for yield
    """Function-scoped fixture for Neo4jRepository instance with pre-test cleanup."""
    logger.info(
        "[neo4j_repo_fixture] Creating Neo4jRepository and cleaning DB (function scope)"
    )
    # --- Pre-test Cleanup ---
    try:
        async with neo4j_driver.session() as session:
            logger.info(
                "[neo4j_repo_fixture] Running: MATCH (n) DETACH DELETE n before test..."
            )
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info(
                "[neo4j_repo_fixture] Finished: MATCH (n) DETACH DELETE n before test."
            )
    except Exception as e:
        logger.exception(
            f"[neo4j_repo_fixture] FAILED to clean Neo4j DB before test: {e}"
        )
        pytest.fail(f"Failed to clean Neo4j DB before test: {e}")

    # Create repository instance
    repo = Neo4jRepository(driver=neo4j_driver)
    try:
        yield repo
    finally:
        # Teardown logic (if any) can go here, but cleanup is done before yield
        logger.info("[neo4j_repo_fixture] Fixture teardown (function scope)")


@pytest.fixture
def mock_faiss_repository() -> Generator[
    Tuple[MagicMock, MagicMock, MagicMock], None, None
]:
    """Mocks the FaissRepository class and provides mock instances for papers and models."""
    with patch(
        "aigraphx.core.db.FaissRepository",  # Correct target for lifespan's usage
        new_callable=MagicMock,
    ) as mock_repo_class:
        mock_instance_papers = MagicMock(spec=FaissRepository)
        mock_instance_papers.index = MagicMock()
        mock_instance_papers.id_map = {}
        mock_instance_papers.is_ready.return_value = True

        mock_instance_models = MagicMock(spec=FaissRepository)
        mock_instance_models.index = MagicMock()
        mock_instance_models.id_map = {}
        mock_instance_models.is_ready.return_value = True

        # Type hint for side_effect function
        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            id_type = kwargs.get("id_type")
            if id_type == "int":
                return mock_instance_papers
            elif id_type == "str":
                return mock_instance_models
            else:
                raise ValueError(
                    f"Unexpected id_type in FaissRepository mock: {id_type}"
                )

        mock_repo_class.side_effect = side_effect
        yield mock_repo_class, mock_instance_papers, mock_instance_models


# Helper to create a mock app
@pytest.fixture
def mock_app() -> FastAPI:  # Removed unused dependencies
    app = FastAPI()
    app.state = State()  # Initialize state
    return app


# Helper fixture to create mock settings with patched values
# ... rest of the file ...

# --- Fixture for Event Loop Policy (If needed globally, usually not) ---
# @pytest.fixture(scope="session")
# def event_loop_policy(request):
#     """Set the asyncio event loop policy for the session."""
#     # On Windows, the default ProactorEventLoop is not compatible with subprocesses
#     # started by testcontainers. Use SelectorEventLoop instead.
#     # if sys.platform == "win32":
#     #     logger.debug("Setting asyncio event loop policy to SelectorEventLoop for Windows.")
#     #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#     # For other platforms (like Linux/WSL), use the default policy
#     # Need to return the policy object expected by pytest-asyncio
#     # Getting the default policy for the platform.
#     return asyncio.get_event_loop_policy()
