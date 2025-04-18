# tests/scripts/test_sync_pg_to_neo4j.py
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, call, MagicMock
from pytest_mock import MockerFixture
from datetime import datetime, date
from typing import Optional, Tuple
import sys
import os
import json
import pprint
import asyncio
import logging

# ADDED: Debug log for raw environment variable at module load time
logger_conftest_top = logging.getLogger("conftest_top")
raw_test_neo4j_db_env = os.getenv("TEST_NEO4J_DATABASE")
logger_conftest_top.critical(
    f"[CONTEST TOP] Raw os.getenv('TEST_NEO4J_DATABASE'): '{raw_test_neo4j_db_env}'"
)

# Mark all async tests in this module to use the session-scoped event loop
pytestmark = pytest.mark.asyncio(loop_scope="session")

# Add project root to path to allow importing from aigraphx and scripts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the function to test and the classes to mock
from scripts.sync_pg_to_neo4j import (
    run_sync,
)  # Import only what's needed for testing execution

# Import Repository classes AFTER path modification, needed for patching targets and type hints
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# Import Repository fixtures from conftest
from tests.conftest import repository as postgres_repository_fixture
from tests.conftest import neo4j_repo_fixture

# --- Constants for Test Data ---
# Define some constants for batch sizes used in the script
# These should ideally match the script, or be configurable/patched if necessary
PG_FETCH_BATCH_SIZE_TEST = 10  # Use smaller batches for testing
NEO4J_WRITE_BATCH_SIZE_TEST = 5

# Use distinct IDs/data for integration tests
TEST_HF_MODEL_1 = {
    "hf_model_id": "test-sync-hf-1",
    "hf_author": "sync_auth1",
    "hf_sha": "sync_sha1",
    "hf_last_modified": datetime(2023, 5, 1, 10, 0, 0),
    "hf_tags": json.dumps(["syncA", "syncB"]),
    "hf_pipeline_tag": "text-classification",
    "hf_downloads": 150,
    "hf_likes": 15,
    "hf_library_name": "transformers",
}
TEST_HF_MODEL_2 = {
    "hf_model_id": "test-sync-hf-2",
    "hf_author": "sync_auth2",
    "hf_sha": "sync_sha2",
    "hf_last_modified": datetime(2023, 5, 2, 11, 0, 0),
    "hf_tags": None,
    "hf_pipeline_tag": "image-generation",
    "hf_downloads": 250,
    "hf_likes": 25,
    "hf_library_name": "diffusers",
}
TEST_PAPER_1 = {
    # paper_id will be generated
    "pwc_id": "test-sync-pwc-1",
    "arxiv_id_base": "sync.1111",
    "arxiv_id_versioned": "sync.1111v1",
    "title": "Sync Paper 1",
    "authors": json.dumps(["Sync Auth 1", "Sync Auth 2"]),
    "summary": "Sync summary 1",
    "published_date": date(2023, 5, 1),
    "area": "ML",
    "pwc_url": "sync_url1",
    "pdf_url": "sync_pdf1",
    "doi": "sync_doi1",
    "primary_category": "cs.LG",
    "categories": json.dumps(["cs.LG", "cs.AI"]),
}
TEST_PAPER_2 = {
    # paper_id will be generated
    "pwc_id": "test-sync-pwc-2",
    "arxiv_id_base": "sync.2222",
    "arxiv_id_versioned": "sync.2222v1",
    "title": "Sync Paper 2",
    "authors": None,
    "summary": "Sync summary 2",
    "published_date": date(2023, 5, 10),
    "area": "Robotics",
    "pwc_url": "sync_url2",
    "pdf_url": "sync_pdf2",
    "doi": "sync_doi2",
    "primary_category": "cs.RO",
    "categories": json.dumps(["cs.RO"]),
}
TEST_LINK_1 = {"hf_model_id": "test-sync-hf-1", "pwc_id": "test-sync-pwc-1"}


# --- Helper function to insert PG data --- #
async def insert_pg_data(repo: PostgresRepository) -> Tuple[int, int]:
    """Inserts sample data into the TEST PG database for sync testing."""
    paper1_id = -1
    paper2_id = -1
    async with repo.pool.connection() as conn:
        async with conn.cursor() as cur:
            # Insert papers
            await cur.execute(
                "INSERT INTO papers (pwc_id, title, summary, published_date, area, authors) VALUES (%s, %s, %s, %s, %s, %s) RETURNING paper_id",
                (
                    TEST_PAPER_1["pwc_id"],
                    TEST_PAPER_1["title"],
                    TEST_PAPER_1["summary"],
                    TEST_PAPER_1["published_date"],
                    TEST_PAPER_1["area"],
                    json.dumps(TEST_PAPER_1["authors"]),
                ),
            )
            paper1_row = await cur.fetchone()
            if not paper1_row:
                raise ValueError("Failed to insert TEST_PAPER_1")
            paper1_id = paper1_row[0]

            await cur.execute(
                "INSERT INTO papers (pwc_id, title, summary, published_date, area, authors) VALUES (%s, %s, %s, %s, %s, %s) RETURNING paper_id",
                (
                    TEST_PAPER_2["pwc_id"],
                    TEST_PAPER_2["title"],
                    TEST_PAPER_2["summary"],
                    TEST_PAPER_2["published_date"],
                    TEST_PAPER_2["area"],
                    json.dumps(TEST_PAPER_2["authors"]),
                ),
            )
            paper2_row = await cur.fetchone()
            if not paper2_row:
                raise ValueError("Failed to insert TEST_PAPER_2")
            paper2_id = paper2_row[0]

            # Insert models (Include hf_library_name)
            await cur.execute(
                "INSERT INTO hf_models (hf_model_id, hf_author, hf_tags, hf_pipeline_tag, hf_library_name) VALUES (%s, %s, %s, %s, %s)",
                (
                    TEST_HF_MODEL_1["hf_model_id"],
                    TEST_HF_MODEL_1["hf_author"],
                    TEST_HF_MODEL_1["hf_tags"],
                    TEST_HF_MODEL_1["hf_pipeline_tag"],
                    TEST_HF_MODEL_1["hf_library_name"],
                ),
            )
            await cur.execute(
                "INSERT INTO hf_models (hf_model_id, hf_author, hf_tags, hf_pipeline_tag, hf_library_name) VALUES (%s, %s, %s, %s, %s)",
                (
                    TEST_HF_MODEL_2["hf_model_id"],
                    TEST_HF_MODEL_2["hf_author"],
                    TEST_HF_MODEL_2["hf_tags"],
                    TEST_HF_MODEL_2["hf_pipeline_tag"],
                    TEST_HF_MODEL_2["hf_library_name"],
                ),
            )

            # Insert links (Corrected: Remove pwc_id column)
            await cur.execute(
                "INSERT INTO model_paper_links (hf_model_id, paper_id) VALUES (%s, %s)",
                (TEST_HF_MODEL_1["hf_model_id"], paper1_id),
            )
            await cur.execute(
                "INSERT INTO model_paper_links (hf_model_id, paper_id) VALUES (%s, %s)",
                (TEST_HF_MODEL_2["hf_model_id"], paper2_id),
            )
            # Insert link between model 1 and paper 2 as well for relation check
            await cur.execute(
                "INSERT INTO model_paper_links (hf_model_id, paper_id) VALUES (%s, %s)",
                (TEST_HF_MODEL_1["hf_model_id"], paper2_id),
            )

        await conn.commit()  # Ensure data is committed
    return paper1_id, paper2_id


# --- Test Case --- #


@pytest.mark.asyncio
async def test_run_sync_integration(
    postgres_repository_fixture: PostgresRepository,  # Real PG repo
    neo4j_repo_fixture: Neo4jRepository,  # Real Neo4j repo
) -> None:
    """Integration test for the sync_pg_to_neo4j script."""
    pg_repo = postgres_repository_fixture
    neo4j_repo = neo4j_repo_fixture

    # 1. Setup: Insert data into the test PostgreSQL database
    paper1_id, paper2_id = await insert_pg_data(pg_repo)

    # --- IMPORTANT: Script Dependency Assumption ---
    # Assume run_sync() internally creates PostgresRepository and Neo4jRepository
    # instances configured using the TEST environment variables set by conftest.
    # If run_sync needs to be passed instances, this test needs adjustment.

    # 2. Execute the script's main logic
    await run_sync(pg_repo=pg_repo, neo4j_repo=neo4j_repo)

    # 3. Assert: Verify data in the test Neo4j database
    async with neo4j_repo.driver.session() as session:
        # Check HF Models
        result_hf1 = await session.run(
            "MATCH (m:HFModel {model_id: $id}) RETURN m.author as author, m.library_name as lib",
            id=TEST_HF_MODEL_1["hf_model_id"],
        )
        record_hf1 = await result_hf1.single()
        assert record_hf1 is not None
        assert record_hf1["author"] == TEST_HF_MODEL_1["hf_author"]
        assert record_hf1["lib"] == TEST_HF_MODEL_1["hf_library_name"]

        result_hf2 = await session.run(
            "MATCH (m:HFModel {model_id: $id}) RETURN m.author as author",
            id=TEST_HF_MODEL_2["hf_model_id"],
        )
        record_hf2 = await result_hf2.single()
        assert record_hf2 is not None
        assert record_hf2["author"] == TEST_HF_MODEL_2["hf_author"]

        # Check Papers
        result_p1 = await session.run(
            "MATCH (p:Paper {pwc_id: $id}) RETURN p.title as title, p.area as area",
            id=TEST_PAPER_1["pwc_id"],
        )
        record_p1 = await result_p1.single()
        assert record_p1 is not None
        assert record_p1["title"] == TEST_PAPER_1["title"]
        assert record_p1["area"] == TEST_PAPER_1["area"]

        result_p2 = await session.run(
            "MATCH (p:Paper {pwc_id: $id}) RETURN p.title as title",
            id=TEST_PAPER_2["pwc_id"],
        )
        record_p2 = await result_p2.single()
        assert record_p2 is not None
        assert record_p2["title"] == TEST_PAPER_2["title"]

        # Check Relationship (Change from USES_PAPER to MENTIONS)
        result_link = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_LINK_1["hf_model_id"],
            pwc_id=TEST_LINK_1["pwc_id"],
        )
        record_link = await result_link.single()
        assert record_link is not None
        assert record_link["c"] == 1

        # === START: Enhanced Relationship Assertions ===

        # Verify Model 1 -> Paper 2 relationship exists
        result_link_m1p2 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_1["hf_model_id"],
            pwc_id=TEST_PAPER_2["pwc_id"],
        )
        record_link_m1p2 = await result_link_m1p2.single()
        assert record_link_m1p2 is not None, (
            f"Relationship M1->P2 not found for {TEST_HF_MODEL_1['hf_model_id']} -> {TEST_PAPER_2['pwc_id']}"
        )
        assert record_link_m1p2["c"] == 1, (
            f"Expected 1 M1->P2 relationship, found {record_link_m1p2['c']}"
        )

        # Verify Model 2 -> Paper 2 relationship exists
        result_link_m2p2 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_2["hf_model_id"],
            pwc_id=TEST_PAPER_2["pwc_id"],
        )
        record_link_m2p2 = await result_link_m2p2.single()
        assert record_link_m2p2 is not None, (
            f"Relationship M2->P2 not found for {TEST_HF_MODEL_2['hf_model_id']} -> {TEST_PAPER_2['pwc_id']}"
        )
        assert record_link_m2p2["c"] == 1, (
            f"Expected 1 M2->P2 relationship, found {record_link_m2p2['c']}"
        )

        # --- Assertions for Tasks, Datasets, Repositories (requires data insertion first) ---

        # Insert some tasks, datasets, repos for Paper 1 in PG
        async with pg_repo.pool.connection() as pg_conn:
            async with pg_conn.cursor() as pg_cur:
                await pg_cur.execute(
                    "INSERT INTO pwc_tasks (paper_id, task_name) VALUES (%s, %s), (%s, %s)",
                    (paper1_id, "Text Summarization", paper1_id, "Question Answering"),
                )
                await pg_cur.execute(
                    "INSERT INTO pwc_datasets (paper_id, dataset_name) VALUES (%s, %s)",
                    (paper1_id, "SQuAD"),
                )
                await pg_cur.execute(
                    "INSERT INTO pwc_repositories (paper_id, url) VALUES (%s, %s)",
                    (paper1_id, "https://github.com/sync/test-repo"),
                )
            await pg_conn.commit()

        # Rerun the sync to pick up the new relations
        # NOTE: Ideally, sync would be idempotent, but for testing this ensures relations are processed.
        # If sync is slow, consider adding these relations before the first run_sync call.
        print("\n--- Running sync again to process relations ---")
        await run_sync(pg_repo=pg_repo, neo4j_repo=neo4j_repo)
        print("--- Second sync run complete ---")

        # Now verify the relations in Neo4j
        # Check Paper 1 -> Task relationship
        result_p1_tasks = await session.run(
            """MATCH (p:Paper {pwc_id: $pwc_id})-[:HAS_TASK]->(t:Task)
               RETURN count(t) as task_count, collect(t.name) as task_names""",
            pwc_id=TEST_PAPER_1["pwc_id"],
        )
        record_p1_tasks = await result_p1_tasks.single()
        assert record_p1_tasks is not None, "Paper 1 Task relationship check failed"
        assert record_p1_tasks["task_count"] == 2, (
            f"Expected 2 tasks for Paper 1, found {record_p1_tasks['task_count']}"
        )
        assert sorted(record_p1_tasks["task_names"]) == sorted(
            ["Text Summarization", "Question Answering"]
        )

        # Check Paper 1 -> Dataset relationship
        result_p1_datasets = await session.run(
            """MATCH (p:Paper {pwc_id: $pwc_id})-[:USES_DATASET]->(d:Dataset)
               RETURN count(d) as dataset_count, d.name as dataset_name""",
            pwc_id=TEST_PAPER_1["pwc_id"],
        )
        record_p1_datasets = await result_p1_datasets.single()
        assert record_p1_datasets is not None, (
            "Paper 1 Dataset relationship check failed"
        )
        assert record_p1_datasets["dataset_count"] == 1, (
            f"Expected 1 dataset for Paper 1, found {record_p1_datasets['dataset_count']}"
        )
        assert record_p1_datasets["dataset_name"] == "SQuAD"

        # Check Paper 1 -> Repository relationship
        result_p1_repos = await session.run(
            """MATCH (p:Paper {pwc_id: $pwc_id})-[:HAS_REPOSITORY]->(r:Repository)
               RETURN count(r) as repo_count, r.url as repo_url""",
            pwc_id=TEST_PAPER_1["pwc_id"],
        )
        record_p1_repos = await result_p1_repos.single()
        assert record_p1_repos is not None, (
            "Paper 1 Repository relationship check failed"
        )
        assert record_p1_repos["repo_count"] == 1, (
            f"Expected 1 repository for Paper 1, found {record_p1_repos['repo_count']}"
        )
        assert record_p1_repos["repo_url"] == "https://github.com/sync/test-repo"

        # === END: Enhanced Relationship Assertions ===

        # Optional: Add check for the second link (Model 1 -> Paper 2)
        result_link_2 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_1["hf_model_id"],
            pwc_id=TEST_PAPER_2["pwc_id"],
        )
        record_link_2 = await result_link_2.single()
        assert record_link_2 is not None
        assert record_link_2["c"] == 1

        # Optional: Add check for the third link (Model 2 -> Paper 2)
        result_link_3 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_2["hf_model_id"],
            pwc_id=TEST_PAPER_2["pwc_id"],
        )
        record_link_3 = await result_link_3.single()
        assert record_link_3 is not None
        assert record_link_3["c"] == 1

    # Note: neo4j_repo_fixture automatically cleans the DB and ensures constraints


# TODO: Add tests for error handling during sync if possible in integration context
# (e.g., simulate PG connection error during fetch - might require patching PG repo *within* script context)
