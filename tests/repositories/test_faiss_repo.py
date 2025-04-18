# tests/repositories/test_faiss_repo.py
import pytest
import os
import json
import numpy as np
import faiss  # type: ignore[import-untyped]
from unittest.mock import patch, MagicMock
from pathlib import Path  # Import Path for tmp_path
from typing import Dict, List, Tuple, Any, Optional, Generator

from aigraphx.repositories.faiss_repo import FaissRepository

# --- Constants for Test Data (Dynamic) ---
# These will match the data created within the fixture
TEST_DIMENSION = 8
TEST_VECTORS = np.array(
    [
        [0.1] * TEST_DIMENSION,
        [0.2] * TEST_DIMENSION,
        [0.9] * TEST_DIMENSION,
    ]
).astype(np.float32)
TEST_IDS = [101, 202, 999]  # Corresponding paper_ids
TEST_ID_MAP = {i: id_ for i, id_ in enumerate(TEST_IDS)}
TEST_EXPECTED_NUM_VECTORS = len(TEST_IDS)

# --- Fixtures ---


@pytest.fixture(scope="function")  # Use function scope for tmp_path isolation
def repository(tmp_path: Path) -> Generator[FaissRepository, None, None]:
    """Creates a temporary Faiss index and map using tmp_path,
    and provides an initialized FaissRepository instance."""
    # Define paths within the temporary directory
    index_path = tmp_path / "test.index"
    id_map_path = tmp_path / "test_ids.json"

    # 1. Build and save the test index
    index = faiss.IndexFlatL2(TEST_DIMENSION)
    index.add(TEST_VECTORS)
    faiss.write_index(index, str(index_path))

    # 2. Save the test ID map
    with open(id_map_path, "w") as f:
        json.dump(TEST_ID_MAP, f)

    # 3. Initialize the repository with temporary paths
    repo = FaissRepository(index_path=str(index_path), id_map_path=str(id_map_path))

    # 4. Check if ready (optional but good practice)
    if not repo.is_ready():
        pytest.fail("FaissRepository failed to initialize with temporary test data.")

    yield repo  # Provide the initialized repository

    # Cleanup is handled automatically by tmp_path fixture


# --- Test Cases (Adjusted for Dynamic Data) ---


def test_load_and_properties(repository: FaissRepository) -> None:
    """Test loading the temporary index/map and check basic properties."""
    assert repository.index is not None
    assert repository.index.d == TEST_DIMENSION
    assert repository.index.ntotal == TEST_EXPECTED_NUM_VECTORS
    # Compare loaded map content (keys should be int after loading)
    assert repository.id_map == TEST_ID_MAP
    assert repository.is_ready() is True


def test_get_index_size(repository: FaissRepository) -> None:
    """Test getting the index size."""
    assert repository.get_index_size() == TEST_EXPECTED_NUM_VECTORS


@pytest.mark.asyncio
async def test_search_similar_found(repository: FaissRepository) -> None:
    """Test searching for similar vectors finds the correct closest item."""
    # Query vector close to the first vector ([0.1]*8, mapped to paper_id 101)
    query_vec = np.array([[0.11] * TEST_DIMENSION]).astype(np.float32)
    k = 2  # Ask for 2 nearest

    results = await repository.search_similar(query_vec, k=k)

    assert len(results) == k
    # Expect paper_id 101 (index 0) to be the closest
    assert results[0][0] == 101
    assert isinstance(results[0][1], float)  # Distance
    # Expect paper_id 202 (index 1) to be the second closest
    assert results[1][0] == 202
    assert results[0][1] < results[1][1]  # Distances ordered


@pytest.mark.asyncio
async def test_search_similar_k_too_large(repository: FaissRepository) -> None:
    """Test searching when k is larger than the number of vectors."""
    query_vec = np.array([[0.91] * TEST_DIMENSION]).astype(np.float32)
    # Await the coroutine
    results = await repository.search_similar(
        query_vec, k=TEST_EXPECTED_NUM_VECTORS + 5
    )
    # Should return only the existing neighbors
    assert len(results) == TEST_EXPECTED_NUM_VECTORS
    # Check if the closest is paper_id 999 (vector [0.9]*8 at index 2)
    assert results[0][0] == 999


@pytest.mark.asyncio
async def test_search_similar_k_zero(repository: FaissRepository) -> None:
    """Test searching with k=0."""
    query_vec = np.random.rand(1, TEST_DIMENSION).astype(np.float32)
    results = await repository.search_similar(query_vec, k=0)
    assert results == []


@pytest.mark.asyncio
async def test_search_wrong_dimension(repository: FaissRepository, caplog: Any) -> None:
    """Test searching with a query vector of wrong dimension."""
    query_vec = np.random.rand(1, TEST_DIMENSION + 5).astype(np.float32)
    results = await repository.search_similar(query_vec, k=1)
    assert results == []
    assert "does not match index dimension" in caplog.text


@pytest.mark.asyncio
@patch.object(FaissRepository, "is_ready", return_value=False)
async def test_search_when_not_ready(
    mock_is_ready: MagicMock, repository: FaissRepository, caplog: Any
) -> None:
    """Test that search returns empty list if repository is not ready."""
    query_vec = np.random.rand(1, TEST_DIMENSION).astype(np.float32)
    results = await repository.search_similar(query_vec, k=1)
    assert results == []
    mock_is_ready.assert_called_once()
    assert "index or ID map not ready" in caplog.text
