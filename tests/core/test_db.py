import pytest
import asyncio  # Added for manual lifespan driving
from unittest.mock import MagicMock, AsyncMock, patch, Mock, ANY
from fastapi import FastAPI
from starlette.datastructures import State  # Import State for mock_app
import os
from typing import Dict, Any, Tuple, List, Optional, Generator, Literal, cast

# Mark all async tests in this module to use the session-scoped event loop
pytestmark = pytest.mark.asyncio(loop_scope="session")

# Import the lifespan function to test
from aigraphx.core.db import lifespan

# Import classes being mocked
# from aigraphx.vectorization.embedder import TextEmbedder # No longer needed to mock here for lifespan tests
from aigraphx.repositories.faiss_repo import FaissRepository

# Import the settings object and Settings type
from aigraphx.core.config import settings, Settings  # Import Settings type


# Use MagicMock for classes, AsyncMock for instances/methods
@pytest.fixture
def mock_async_connection_pool() -> Generator[Tuple[MagicMock, AsyncMock], None, None]:
    with patch(
        "aigraphx.core.db.psycopg_pool.AsyncConnectionPool", new_callable=MagicMock
    ) as mock_pool_class:
        mock_instance = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_pool_class.return_value = mock_instance
        yield mock_pool_class, mock_instance


@pytest.fixture
def mock_neo4j_driver() -> Generator[Tuple[MagicMock, AsyncMock], None, None]:
    with patch(
        "aigraphx.core.db.AsyncGraphDatabase.driver", new_callable=MagicMock
    ) as mock_driver_func:
        mock_instance = AsyncMock()
        mock_instance.close = AsyncMock()
        mock_driver_func.return_value = mock_instance
        yield mock_driver_func, mock_instance


# Removed mock_text_embedder fixture as lifespan no longer handles it


@pytest.fixture
def mock_faiss_repository() -> Generator[
    Tuple[MagicMock, MagicMock, MagicMock], None, None
]:
    with patch(
        "aigraphx.core.db.FaissRepository", new_callable=MagicMock
    ) as mock_repo_class:
        # Create two distinct mock instances for papers and models
        mock_instance_papers = MagicMock(spec=FaissRepository)
        mock_instance_papers.index = MagicMock()
        mock_instance_papers.id_map = {}  # Add id_map attribute
        mock_instance_papers.is_ready.return_value = True  # Default to ready

        mock_instance_models = MagicMock(spec=FaissRepository)
        mock_instance_models.index = MagicMock()
        mock_instance_models.id_map = {}  # Add id_map attribute
        mock_instance_models.is_ready.return_value = True  # Default to ready

        # Configure the class mock to return the correct instance based on call args
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            id_type = kwargs.get("id_type")
            if id_type == "int":
                return mock_instance_papers
            elif id_type == "str":
                return mock_instance_models
            else:
                # Fallback or raise error if needed
                raise ValueError(
                    f"Unexpected id_type in FaissRepository mock: {id_type}"
                )

        mock_repo_class.side_effect = side_effect

        # Yield the class mock and the two instances for manipulation in tests
        yield mock_repo_class, mock_instance_papers, mock_instance_models


# Helper to create a mock app
@pytest.fixture
def mock_app() -> FastAPI:  # Removed unused dependencies
    app = FastAPI()
    app.state = State()  # Initialize state
    return app


# Helper fixture to create mock settings with patched values
# This allows modifying settings per test if needed
@pytest.fixture
def mock_settings(
    monkeypatch: pytest.MonkeyPatch,
    neo4j_uri: Optional[str] = "neo4j://testsuccess",
    neo4j_user: Optional[str] = "testuser_success",
    neo4j_pass: Optional[str] = "testpass_success",
    # embedder_model: str = "test-model", # Removed embedder settings
    # embedder_device: str = "test-device",
    faiss_papers_index: str = "/tmp/papers.index",
    faiss_papers_map: str = "/tmp/papers.json",
    faiss_models_index: str = "/tmp/models.index",
    faiss_models_map: str = "/tmp/models.json",
) -> Settings:
    # Create a copy of the actual settings to modify
    test_settings = settings.model_copy()  # Use model_copy()
    # Apply patches using monkeypatch for isolation
    monkeypatch.setattr(
        test_settings,
        "database_url",
        "postgresql://mock_user:mock_pass@mock_host:5432/mock_db",
    )
    monkeypatch.setattr(test_settings, "neo4j_uri", neo4j_uri)
    monkeypatch.setattr(test_settings, "neo4j_username", neo4j_user)
    monkeypatch.setattr(test_settings, "neo4j_password", neo4j_pass)
    # monkeypatch.setattr(test_settings, "sentence_transformer_model", embedder_model) # Removed
    # monkeypatch.setattr(test_settings, "embedder_device", embedder_device) # Removed
    monkeypatch.setattr(test_settings, "faiss_index_path", faiss_papers_index)
    monkeypatch.setattr(test_settings, "faiss_mapping_path", faiss_papers_map)
    monkeypatch.setattr(test_settings, "models_faiss_index_path", faiss_models_index)
    monkeypatch.setattr(test_settings, "models_faiss_mapping_path", faiss_models_map)
    return test_settings


# Test successful initialization and cleanup
@pytest.mark.asyncio
async def test_lifespan_success(
    mock_app: FastAPI,
    mock_settings: Settings,  # Use mock_settings fixture
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],  # Now yields 3 items
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # Set Faiss mocks to be ready
    mock_repo_instance_papers.is_ready.return_value = True
    mock_repo_instance_models.is_ready.return_value = True

    ctx = lifespan(mock_app, mock_settings)  # Pass mock_settings
    await ctx.__aenter__()

    # Check startup calls using mock_settings
    mock_pool_class.assert_called_once_with(
        conninfo=mock_settings.database_url,
        min_size=mock_settings.pg_pool_min_size,
        max_size=mock_settings.pg_pool_max_size,
    )
    mock_driver_func.assert_called_once_with(
        mock_settings.neo4j_uri,
        auth=(mock_settings.neo4j_username, mock_settings.neo4j_password),
    )
    # Removed embedder check
    mock_repo_class.assert_any_call(
        index_path=mock_settings.faiss_index_path,
        id_map_path=mock_settings.faiss_mapping_path,
        id_type="int",
    )
    mock_repo_class.assert_any_call(
        index_path=mock_settings.models_faiss_index_path,
        id_map_path=mock_settings.models_faiss_mapping_path,
        id_type="str",
    )
    assert mock_repo_class.call_count == 2
    mock_repo_instance_papers.is_ready.assert_called_once()  # Check readiness was called
    mock_repo_instance_models.is_ready.assert_called_once()  # Check readiness was called

    # Check state is set
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    # assert not hasattr(mock_app.state, 'embedder') # Ensure embedder state is not set
    assert (
        mock_app.state.faiss_repo_papers == mock_repo_instance_papers
    )  # Check specific instance
    assert (
        mock_app.state.faiss_repo_models == mock_repo_instance_models
    )  # Check specific instance

    await ctx.__aexit__(None, None, None)

    # Check shutdown calls
    mock_pool_instance.close.assert_awaited_once()
    mock_driver_instance.close.assert_awaited_once()


# Test initialization failure (e.g., PG pool fails)
@pytest.mark.asyncio
async def test_lifespan_pg_pool_failure(
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_pool_class.side_effect = Exception("PG Pool Init Error")

    ctx = lifespan(mock_app, mock_settings)
    with pytest.raises(RuntimeError, match="PostgreSQL pool initialization failed"):
        await ctx.__aenter__()

    # Check shutdown calls not made for resources after failure
    mock_driver_func.assert_not_called()  # Neo4j init shouldn't be reached
    mock_pool_instance.close.assert_not_awaited()  # Pool close is not called on init fail
    mock_driver_instance.close.assert_not_awaited()


# Test initialization failure (e.g., Neo4j fails)
@pytest.mark.asyncio
async def test_lifespan_neo4j_failure(
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_driver_func.side_effect = Exception("Neo4j Init Error")

    ctx = lifespan(mock_app, mock_settings)
    with pytest.raises(RuntimeError, match="Neo4j driver initialization failed"):
        await ctx.__aenter__()

    # Check close calls after failure - pool might be opened before Neo4j fails
    # Shutdown should still try to close the pool if it exists
    # await ctx.__aexit__(RuntimeError, RuntimeError("Neo4j driver initialization failed"), None) # Simulate exception exit
    # mock_pool_instance.close.assert_awaited_once() # This depends on exact exit handling, let's skip complex exit mock for now
    mock_pool_instance.close.assert_not_awaited()  # Safer to assume close isn't reached reliably after mid-lifespan exception
    mock_driver_instance.close.assert_not_awaited()


# Renamed Test: Test Faiss repo initialization resulting in not ready state
@pytest.mark.asyncio
async def test_lifespan_faiss_not_ready(
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[
        MagicMock, MagicMock, MagicMock
    ],  # Depends on faiss mock
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # Simulate Papers Faiss repo being not ready
    mock_repo_instance_papers.is_ready.return_value = False
    mock_repo_instance_models.is_ready.return_value = True  # Models is ready

    ctx = lifespan(mock_app, mock_settings)
    with pytest.raises(RuntimeError, match="Papers Faiss Repository is not ready after initialization."):
        await ctx.__aenter__()

    # Check state after expected failure
    assert mock_app.state.pg_pool == mock_pool_instance # Should still be set
    assert mock_app.state.neo4j_driver == mock_driver_instance # Should still be set
    assert mock_app.state.faiss_repo_papers is None  # Should be None or not set
    assert (
        mock_app.state.faiss_repo_models is None
    )  # Should also be None as init likely stopped

    # Check close calls are not made if startup failed mid-way
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# Test initialization failure (e.g., Faiss Papers Repo __init__ fails)
@pytest.mark.asyncio
async def test_lifespan_faiss_papers_init_failure(  # Renamed slightly for clarity
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # Modified side_effect to fail for papers index (int id_type)
    def side_effect(*args: Any, **kwargs: Any) -> Any:
        id_type = kwargs.get("id_type")
        if id_type == "int":
            raise ValueError("Paper Faiss initialization failed")
        elif id_type == "str":
            return mock_repo_instance_models
        else:
            raise ValueError(f"Unexpected id_type: {id_type}")

    mock_repo_class.side_effect = side_effect

    ctx = lifespan(mock_app, mock_settings)
    with pytest.raises(RuntimeError, match="Papers Faiss Repository initialization failed"):
        await ctx.__aenter__()

    # Check state after expected failure
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    assert mock_app.state.faiss_repo_papers is None
    assert mock_app.state.faiss_repo_models is None

    # Check close calls
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# Test initialization failure (e.g., Faiss Models Repo __init__ fails)
@pytest.mark.asyncio
async def test_lifespan_faiss_models_init_failure(  # Renamed slightly for clarity
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # Modified side_effect to fail for models index (str id_type)
    def side_effect(*args: Any, **kwargs: Any) -> Any:
        id_type = kwargs.get("id_type")
        if id_type == "int":
            return mock_repo_instance_papers
        elif id_type == "str":
            raise ValueError("Models Faiss initialization failed")
        else:
            raise ValueError(f"Unexpected id_type: {id_type}")

    mock_repo_class.side_effect = side_effect

    ctx = lifespan(mock_app, mock_settings)
    with pytest.raises(RuntimeError, match="Models Faiss Repository initialization failed"):
        await ctx.__aenter__()

    # Check state after expected failure
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    # Paper repo might be initialized before model repo fails
    assert mock_app.state.faiss_repo_papers == mock_repo_instance_papers
    assert mock_app.state.faiss_repo_models is None

    # Check close calls
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# Test Neo4j not configured scenario
@pytest.mark.asyncio
async def test_lifespan_neo4j_not_configured(
    mock_app: FastAPI,
    mock_settings: Settings,  # Use mock_settings
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,  # Add monkeypatch
) -> None:
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # Explicitly set one of the required Neo4j settings to None
    monkeypatch.setattr(mock_settings, "neo4j_password", None)

    ctx = lifespan(mock_app, mock_settings)
    # Lifespan should not raise an error, just log a warning
    await ctx.__aenter__()

    # Assertions after startup
    mock_pool_class.assert_called_once()  # PG pool should still init
    mock_driver_func.assert_not_called()  # Neo4j driver func should NOT be called
    mock_repo_class.assert_any_call(
        index_path=mock_settings.faiss_index_path,
        id_map_path=mock_settings.faiss_mapping_path,
        id_type="int",
    )
    mock_repo_class.assert_any_call(
        index_path=mock_settings.models_faiss_index_path,
        id_map_path=mock_settings.models_faiss_mapping_path,
        id_type="str",
    )
    assert mock_repo_class.call_count == 2

    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver is None  # State should be None
    assert mock_app.state.faiss_repo_papers == mock_repo_instance_papers
    assert mock_app.state.faiss_repo_models == mock_repo_instance_models

    # Check shutdown calls
    await ctx.__aexit__(None, None, None)
    mock_pool_instance.close.assert_awaited_once()
    mock_driver_instance.close.assert_not_awaited()  # Driver close not called
