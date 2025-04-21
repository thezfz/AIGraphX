import pytest
import numpy as np
from unittest.mock import (
    patch,
    AsyncMock,
    MagicMock,
    mock_open,
)  # Keep patch for potential future use
import json
import faiss  # type: ignore[import-untyped]
import os
from typing import List, Optional, Dict, Any, Tuple, cast
from pathlib import Path  # Import Path for tmp_path
from datetime import datetime

# Import the function to test
from scripts.sync_pg_to_models_faiss import build_index as build_models_faiss_index

# Import real components needed for the integration test
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.vectorization.embedder import TextEmbedder  # Import the real embedder
from tests.conftest import (
    repository as postgres_repository_fixture,
    test_settings as test_settings_fixture,  # Import the fixture
)  # Import PG fixture
from aigraphx.core.config import Settings  # Import Settings for type hinting

# --- Test Data ---
TEST_MODEL_1_FAISS = {
    "hf_model_id": "test-faiss-model-1",
    "hf_author": "faiss_author1",
    "hf_sha": "faiss_sha1",
    "hf_last_modified": datetime(2023, 7, 1, 10, 0, 0),
    "hf_tags": json.dumps(["faiss", "model"]),
    "hf_pipeline_tag": "feature-extraction",
    "hf_downloads": 300,
    "hf_likes": 30,
    "hf_library_name": "sentence-transformers",
    # Simulate text that would be used for indexing
    "_index_text": "Model 1 for faiss testing. Author: faiss_author1. Tags: faiss, model.",
}
TEST_MODEL_2_FAISS = {
    "hf_model_id": "test-faiss-model-2",
    "hf_author": "faiss_author2",
    "hf_sha": "faiss_sha2",
    "hf_last_modified": datetime(2023, 7, 5, 12, 0, 0),
    "hf_tags": json.dumps(["image", "faiss"]),
    "hf_pipeline_tag": "image-classification",
    "hf_downloads": 400,
    "hf_likes": 40,
    "hf_library_name": "timm",
    "_index_text": "Model 2 image classifier by faiss_author2.",
}


# --- Helper to insert PG model data ---
async def insert_faiss_model_pg_data(repo: PostgresRepository) -> List[str]:
    model_ids: List[str] = []
    async with repo.pool.connection() as conn:
        async with conn.cursor() as cur:
            for model_data in [TEST_MODEL_1_FAISS, TEST_MODEL_2_FAISS]:
                model_ids.append(str(model_data["hf_model_id"]))
                await cur.execute(
                    "INSERT INTO hf_models (hf_model_id, hf_author, hf_sha, hf_last_modified, hf_tags, hf_pipeline_tag, hf_downloads, hf_likes, hf_library_name) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (hf_model_id) DO UPDATE SET hf_author=EXCLUDED.hf_author",
                    (
                        model_data["hf_model_id"],
                        model_data["hf_author"],
                        model_data["hf_sha"],
                        model_data["hf_last_modified"],
                        model_data["hf_tags"],
                        model_data["hf_pipeline_tag"],
                        model_data["hf_downloads"],
                        model_data["hf_likes"],
                        model_data["hf_library_name"],
                    ),
                )
            await conn.commit()
    return model_ids


# --- Test Case (Integration) ---


@pytest.mark.asyncio
# @pytest.mark.skip(reason="Skipping slow embedding test during refactor") # Optional: Skip if embedder init/run is slow
async def test_build_models_faiss_integration(
    postgres_repository_fixture: PostgresRepository,
    test_settings_fixture: Settings,  # Inject test_settings
    tmp_path: Path,
) -> None:
    """Integration test for building the Faiss index for models."""
    pg_repo = postgres_repository_fixture
    test_settings = test_settings_fixture  # Use injected settings

    # 1. Setup: Insert data into test PG database
    inserted_model_ids = await insert_faiss_model_pg_data(pg_repo)
    assert len(inserted_model_ids) == 2

    # 2. Setup: Define temporary file paths using tmp_path AND test_settings
    index_file_name = Path(test_settings.models_faiss_index_path).name
    map_file_name = Path(test_settings.models_faiss_mapping_path).name
    # Use paths from test_settings directly as they point to temp files
    temp_index_path = Path(test_settings.models_faiss_index_path)
    temp_map_path = Path(test_settings.models_faiss_mapping_path)

    # Ensure directories exist
    temp_index_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Setup: Instantiate the real TextEmbedder
    try:
        embedder = TextEmbedder()
        dimension = embedder.get_embedding_dimension()
        assert isinstance(dimension, int) and dimension > 0
    except Exception as e:
        pytest.fail(
            f"Failed to initialize TextEmbedder: {e}. Ensure model is downloaded or network is accessible."
        )

    # --- IMPORTANT: Script Dependency Assumption ---
    # Assume build_models_faiss_index uses the repository and embedder passed to it.
    # Also assumes the script correctly calls `repo.get_all_models_for_indexing()`

    # 4. Execute the build_index function using paths from test_settings
    await build_models_faiss_index(
        pg_repo=pg_repo,
        embedder=embedder,
        index_path=str(temp_index_path),
        id_map_path=str(temp_map_path),
        reset_index=True,
    )

    # 5. Assert: Check if files were created
    assert temp_index_path.exists()
    assert temp_map_path.exists()

    # 6. Assert: Check content of the ID map
    with open(temp_map_path, "r") as f:
        loaded_id_map = json.load(f)
    loaded_id_map = {int(k): v for k, v in loaded_id_map.items()}
    assert len(loaded_id_map) == len(inserted_model_ids)
    assert set(loaded_id_map.values()) == set(inserted_model_ids)
    assert set(loaded_id_map.keys()) == set(range(len(inserted_model_ids)))

    # 7. Assert: Check basic properties of the index
    try:
        loaded_index = faiss.read_index(str(temp_index_path))
        assert loaded_index.ntotal == len(inserted_model_ids)
        assert loaded_index.d == dimension
    except Exception as e:
        pytest.fail(f"Failed to read or verify the created models Faiss index: {e}")


# Note: No mocked unit tests were present for the models index, only integration.
# If unit tests are added later, they should also be updated to use test_settings fixture.
