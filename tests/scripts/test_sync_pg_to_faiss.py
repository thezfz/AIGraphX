import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, mock_open, call, ANY, MagicMock
import asyncpg  # type: ignore[import-untyped]
import json
import os
import builtins
import numpy as np
import faiss  # type: ignore[import-untyped]
from pathlib import Path  # Import Path for tmp_path
from datetime import date
from typing import AsyncGenerator, Optional, List, Tuple, Any, cast, Dict, Callable

# Import the specific functions/classes we want to test/mock from the script
from scripts.sync_pg_to_faiss import (
    build_index as build_faiss_index,
    main as run_sync_faiss_main,
)

# Import necessary classes used within the script that we might mock
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.vectorization.embedder import TextEmbedder
from tests.conftest import (
    repository as postgres_repository_fixture,
    test_settings as test_settings_fixture,  # Import the fixture
)  # Import PG fixture
from aigraphx.core.config import Settings  # Import Settings for type hinting

# --- Test Data ---
TEST_PAPER_1_FAISS = {
    "pwc_id": "test-faiss-pwc-1",
    "title": "Faiss Paper 1",
    "summary": "This is the first paper summary for Faiss testing.",
    "published_date": date(2023, 6, 1),
    "area": "Vision",
}
TEST_PAPER_2_FAISS = {
    "pwc_id": "test-faiss-pwc-2",
    "title": "Faiss Paper 2",
    "summary": "Second paper abstract, slightly different content.",
    "published_date": date(2023, 6, 10),
    "area": "NLP",
}


# --- Helper to insert PG data ---
async def insert_faiss_pg_data(repo: PostgresRepository) -> List[int]:
    paper_ids = []
    async with repo.pool.connection() as conn:
        async with conn.cursor() as cur:
            for paper_data in [TEST_PAPER_1_FAISS, TEST_PAPER_2_FAISS]:
                await cur.execute(
                    "INSERT INTO papers (pwc_id, title, summary, published_date, area) VALUES (%s, %s, %s, %s, %s) RETURNING paper_id",
                    (
                        paper_data["pwc_id"],
                        paper_data["title"],
                        paper_data["summary"],
                        paper_data["published_date"],
                        paper_data["area"],
                    ),
                )
                result = await cur.fetchone()
                if result:
                    paper_id = result[0]  # 安全地访问索引，确保result不是None
                    paper_ids.append(paper_id)
        # Add explicit commit to ensure data is visible to the script
        await conn.commit()
    return paper_ids


# --- Test Case (Integration) ---


@pytest.mark.asyncio
# @pytest.mark.skip(reason="Skipping slow embedding test during refactor") # Optional: Skip if embedder init/run is slow
async def test_build_faiss_integration(
    postgres_repository_fixture: PostgresRepository,
    test_settings_fixture: Settings,  # Inject test_settings
    tmp_path: Path,
) -> None:
    """Integration test for building the Faiss index for papers."""
    pg_repo = postgres_repository_fixture
    test_settings = test_settings_fixture  # Use injected settings

    # 1. Setup: Insert data into test PG database
    inserted_paper_ids = await insert_faiss_pg_data(pg_repo)
    assert len(inserted_paper_ids) == 2

    # 2. Setup: Define temporary file paths using tmp_path AND test_settings
    # Get base names from test_settings which now hold the temp paths
    index_file_name = Path(test_settings.faiss_index_path).name
    map_file_name = Path(test_settings.faiss_mapping_path).name
    # Construct paths within tmp_path for this specific test run if needed,
    # OR directly use the paths provided by test_settings (which are already temporary)
    # Let's use the paths from test_settings directly as they are session-scoped temp files.
    temp_index_path = Path(test_settings.faiss_index_path)
    temp_map_path = Path(test_settings.faiss_mapping_path)

    # Ensure the directories for these paths exist (tmp_path_factory handles the base)
    temp_index_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Setup: Instantiate the real TextEmbedder
    # Assumes SENTENCE_TRANSFORMER_MODEL and DEVICE are set in environment or defaults are okay
    try:
        embedder = TextEmbedder()
        # Check dimension (optional)
        dimension = embedder.get_embedding_dimension()
        assert isinstance(dimension, int) and dimension > 0
    except Exception as e:
        pytest.fail(
            f"Failed to initialize TextEmbedder: {e}. Ensure model is downloaded or network is accessible."
        )

    # --- IMPORTANT: Script Dependency Assumption ---
    # Assume build_faiss_index uses the repository and embedder passed to it.

    # 4. Execute the build_index function using paths from test_settings
    await build_faiss_index(
        pg_repo=pg_repo,
        embedder=embedder,
        index_path=str(temp_index_path),  # Use the path from settings
        id_map_path=str(temp_map_path),  # Use the path from settings
        batch_size=32,  # Added batch_size argument
        reset_index=False,  # Test without reset first
    )

    # 5. Assert: Check if files were created in tmp_path
    assert temp_index_path.exists()
    assert temp_map_path.exists()

    # 6. Assert: Check content of the ID map
    with open(temp_map_path, "r") as f:
        loaded_id_map = json.load(f)
    # Convert keys back to int for comparison
    loaded_id_map = {int(k): v for k, v in loaded_id_map.items()}
    # The map should contain the IDs inserted earlier
    assert len(loaded_id_map) == len(inserted_paper_ids)
    assert set(loaded_id_map.values()) == set(inserted_paper_ids)
    # Keys should be 0 to n-1
    assert set(loaded_id_map.keys()) == set(range(len(inserted_paper_ids)))

    # 7. Assert: Check basic properties of the index (optional but recommended)
    try:
        loaded_index = faiss.read_index(str(temp_index_path))
        assert loaded_index.ntotal == len(inserted_paper_ids)
        assert loaded_index.d == dimension
    except Exception as e:
        pytest.fail(f"Failed to read or verify the created Faiss index: {e}")

    # TODO: Add test case for reset_index=True if needed, verifying os.remove is called
    # (might need patching os.remove specifically for that verification)


# Mock Embedder class
class MockTextEmbedder:
    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = MagicMock()  # Simulate a loaded model
        self._dimension = 384  # Example dimension
        print(f"MockTextEmbedder initialized with {model_name}, {device}")

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def embed_batch(self, texts: List[Optional[str]]) -> np.ndarray:
        print(f"MockTextEmbedder embedding batch of size {len(texts)}")
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        return np.random.rand(len(texts), self._dimension).astype(np.float32)


# Mock PostgresRepository
class MockPostgresRepository:
    def __init__(self, pool: AsyncMock) -> None:
        self.pool = pool
        self.paper_data = [(1, "Summary 1"), (2, "Summary 2"), (3, "Summary 3")]

    async def get_all_paper_ids_and_text(self) -> AsyncGenerator[Tuple[int, str], None]:
        for item in self.paper_data:
            yield item
        # Simulate ending - just return
        return

    async def close(self) -> None:
        pass  # Mock close method


# --- Test Fixtures ---


@pytest.fixture
def mock_faiss_index(mocker: Any) -> MagicMock:
    """Fixture for a mocked Faiss index."""
    mock_index = MagicMock(spec=faiss.Index)
    mock_index.ntotal = 0

    # Correct side effect: increment ntotal based on the size of the embeddings passed
    def mock_add(embeddings: Any) -> None:
        if isinstance(embeddings, np.ndarray):
            mock_index.ntotal += embeddings.shape[0]
        else:
            # Handle potential non-numpy input if necessary, though unlikely here
            pass  # Or raise error

    mock_index.add.side_effect = mock_add
    mocker.patch("faiss.IndexFlatL2", return_value=mock_index)
    mocker.patch("faiss.write_index")
    return mock_index


@pytest.fixture
def mock_pg_repo(mocker: Any) -> MockPostgresRepository:
    # Return an instance of our mock repo
    return MockPostgresRepository(pool=AsyncMock())  # Pool mock needed for init


@pytest.fixture
def mock_embedder(mocker: Any) -> MockTextEmbedder:
    # Return an instance of our mock embedder
    return MockTextEmbedder()


# --- Test Cases ---


@pytest.mark.asyncio
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
async def test_build_faiss_success_no_reset(
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder,
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """Tests successful Faiss index build without reset."""
    test_settings = test_settings_fixture

    # Call build_index with mocked dependencies and paths from test_settings
    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,  # Added batch_size argument
        reset_index=False,
    )

    # --- Assertions ---
    # 1. Papers fetched from DB (check if the async generator was iterated)
    # Hard to assert iteration directly, check downstream effects

    # 2. Faiss index created and embeddings added
    # Verify that IndexFlatL2 was called correctly
    faiss.IndexFlatL2.assert_called_once_with(mock_embedder.get_embedding_dimension())
    # Check that add was called (at least once, likely just once for a small batch)
    mock_faiss_index.add.assert_called()
    # Assert that the final ntotal matches the number of items yielded by the MOCK repo
    assert mock_faiss_index.ntotal == len(mock_pg_repo.paper_data)

    # 3. Files NOT deleted (no reset)
    mock_os_remove.assert_not_called()

    # 4. Index and ID map saved
    # Use the directory path as a string, consistent with os.path.dirname
    expected_dir = os.path.dirname(test_settings.faiss_index_path)
    mock_os_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

    faiss.write_index.assert_called_once()
    # Check args for write_index (first arg is the index object, second is path string)
    expected_id_map = {
        i: paper_id for i, (paper_id, _) in enumerate(mock_pg_repo.paper_data)
    }
    mock_file_open.assert_called_once_with(test_settings.faiss_mapping_path, "w")
    mock_json_dump.assert_called_once_with(expected_id_map, mock_file_open())


@pytest.mark.asyncio
@patch("os.path.exists", return_value=True)  # Assume files *do* exist
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
async def test_build_faiss_success_with_reset(
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder,
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """Tests successful Faiss index build *with* reset."""
    test_settings = test_settings_fixture

    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,  # Added batch_size argument
        reset_index=True,
    )

    # --- Assertions ---
    # 1. os.path.exists called for both files
    mock_os_exists.assert_any_call(test_settings.faiss_index_path)
    mock_os_exists.assert_any_call(test_settings.faiss_mapping_path)

    # 2. os.remove called for both files
    expected_remove_calls = [
        call(test_settings.faiss_index_path),
        call(test_settings.faiss_mapping_path),
    ]
    mock_os_remove.assert_has_calls(expected_remove_calls, any_order=True)
    assert mock_os_remove.call_count == 2

    # 3. Rest of the process runs (basic check)
    faiss.write_index.assert_called_once()
    mock_json_dump.assert_called_once()


@pytest.mark.asyncio
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
@patch("scripts.sync_pg_to_faiss.logger")  # Mock logger
async def test_build_faiss_embedding_error(
    mock_logger: MagicMock,
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder,
    mocker: Any,
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """Tests handling of errors during the embedding process."""
    test_settings = test_settings_fixture

    # Patch the embedder instance's method
    embed_error = Exception("Simulated embedding error")
    mocker.patch.object(mock_embedder, "embed_batch", side_effect=embed_error)

    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,  # Added batch_size argument
        reset_index=False,
    )

    # --- Assertions ---
    # 1. Embedder method was called (use the patched object)
    mock_embedder.embed_batch.assert_called()  # type: ignore[attr-defined]

    # 2. Error logged - Assert the specific error message was logged at least once
    assert mock_logger.error.call_count >= 1
    first_call_args, first_call_kwargs = mock_logger.error.call_args_list[0]
    assert len(first_call_args) == 1
    assert (
        first_call_args[0]
        == f"An unexpected error occurred during Faiss index build: {embed_error}"
    )

    # 3. File operations (saving) should not happen after error
    faiss.write_index.assert_not_called()
    mock_file_open.assert_not_called()
    mock_json_dump.assert_not_called()


@pytest.mark.asyncio
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
@patch("faiss.write_index", side_effect=IOError("Disk full"))  # Simulate save error
@patch("scripts.sync_pg_to_faiss.logger")  # Mock logger
async def test_build_faiss_save_error(
    mock_logger: MagicMock,
    mock_write_index: MagicMock,
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder,
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """Tests handling of errors during file saving."""
    test_settings = test_settings_fixture

    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,  # Added batch_size argument
        reset_index=False,
    )

    # --- Assertions ---
    # 1. write_index was called
    mock_write_index.assert_called_once_with(
        mock_faiss_index, test_settings.faiss_index_path
    )

    # 2. Error logged - Use .error() instead of .exception()
    mock_logger.error.assert_called_once_with(
        f"Error saving Faiss index or ID map: {mock_write_index.side_effect}"
    )

    # 3. ID map NOT saved
    mock_json_dump.assert_not_called()


# --- Consider adding tests for the main() function if needed --- #
# (e.g., argument parsing, resource initialization/cleanup calls)
