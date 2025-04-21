#!/usr/bin/env python
import asyncio
import logging
import os
from typing import Optional
import traceback

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver
import asyncpg  # type: ignore[import-untyped]
from psycopg_pool import AsyncConnectionPool  # Import pool

# Adjust import paths assuming script is run from project root
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository
from aigraphx.repositories.faiss_repo import FaissRepository

# Import embedder only to get dimension for Faiss initialization
from aigraphx.vectorization.embedder import TextEmbedder

# --- Configuration ---
# Load environment variables from .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Faiss configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/papers.faiss")
FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "./data/papers_mapping.pkl")

# Embedder model (only needed for dimension)
EMBEDDING_MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL")
EMBEDDER_DEVICE = os.getenv("EMBEDDER_DEVICE")

EXPECTED_COUNT = 2500  # The expected number based on the load script log

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ],
)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Starting data count verification across storage layers...")

    # 首先创建PostgreSQL连接池
    pg_pool = None
    pg_repo = None
    try:
        # 创建连接池
        pg_pool = await asyncpg.create_pool(dsn=DATABASE_URL)
        if pg_pool is None:
            logger.error("Failed to create PostgreSQL connection pool")
        else:
            # 使用连接池初始化仓库
            pg_repo = PostgresRepository(pool=pg_pool)
    except Exception as e:
        logger.error(f"Error creating PostgreSQL pool: {e}")

    neo4j_driver: Optional[AsyncDriver] = None
    neo4j_repo: Optional[Neo4jRepository] = None
    faiss_repo: Optional[FaissRepository] = None
    embedder: Optional[TextEmbedder] = None

    pg_count = -1
    neo4j_count = -1
    faiss_index_count = -1
    faiss_map_count = -1

    try:
        # 1. Count PostgreSQL
        logger.info("Checking PostgreSQL count...")
        if pg_repo:
            pg_count = await pg_repo.count_papers()
            if pg_count == -1:
                logger.error("Failed to get count from PostgreSQL.")
        else:
            logger.error("PostgreSQL repository not initialized.")

        # 2. Count Neo4j
        logger.info("Checking Neo4j count...")
        if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
            try:
                neo4j_driver = AsyncGraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                neo4j_repo = Neo4jRepository(driver=neo4j_driver)
                neo4j_count = await neo4j_repo.count_paper_nodes()
                if neo4j_count == -1:
                    logger.error("Failed to get count from Neo4j.")
            except Exception as e:
                logger.error(f"Failed to connect or count in Neo4j: {e}")
                neo4j_count = -1  # Ensure error state
        else:
            logger.warning("Neo4j connection details missing. Skipping Neo4j count.")
            neo4j_count = -2  # Indicate skipped

        # 3. Count Faiss
        logger.info("Checking Faiss count...")
        try:
            # Need embedder dimension to load/initialize Faiss repo
            embedder = TextEmbedder(
                model_name=EMBEDDING_MODEL_NAME, device=EMBEDDER_DEVICE
            )
            dimension = embedder.get_embedding_dimension()
            if dimension > 0:
                faiss_repo = FaissRepository(
                    index_path=FAISS_INDEX_PATH,
                    id_map_path=FAISS_MAPPING_PATH,
                )
                if faiss_repo.index:
                    faiss_index_count = faiss_repo.get_index_size()
                    faiss_map_count = len(faiss_repo.id_map)
                    logger.info(f"Faiss index contains {faiss_index_count} vectors.")
                    logger.info(f"Faiss mapping contains {faiss_map_count} entries.")
                else:
                    logger.error("Faiss index could not be loaded/initialized.")
            else:
                logger.error("Could not get embedding dimension, cannot check Faiss.")
        except FileNotFoundError:
            logger.error(
                f"Faiss index or mapping file not found at expected paths: {FAISS_INDEX_PATH}, {FAISS_MAPPING_PATH}"
            )
        except Exception as e:
            logger.exception(
                f"Failed to initialize Faiss repository or get counts: {e}"
            )

        # 4. Report Summary
        logger.info("--- Verification Summary ---")
        logger.info(f"Expected Count:      {EXPECTED_COUNT}")
        logger.info(f"PostgreSQL Count:    {pg_count if pg_count != -1 else 'ERROR'}")
        if neo4j_count == -2:
            logger.info("Neo4j Count:         SKIPPED")
        else:
            logger.info(
                f"Neo4j (:Paper) Count: {neo4j_count if neo4j_count != -1 else 'ERROR'}"
            )
        logger.info(
            f"Faiss Index Count:   {faiss_index_count if faiss_index_count != -1 else 'ERROR/NOT CHECKED'}"
        )
        logger.info(
            f"Faiss Mapping Count: {faiss_map_count if faiss_map_count != -1 else 'ERROR/NOT CHECKED'}"
        )

        # Check consistency
        counts = [
            c
            for c in [pg_count, neo4j_count, faiss_index_count, faiss_map_count]
            if c >= 0
        ]  # Get valid counts
        consistent = False
        if len(counts) > 1:
            consistent = all(c == counts[0] for c in counts)

        if consistent and counts and counts[0] == EXPECTED_COUNT:
            logger.info(
                "Result: All checked counts are consistent and match the expected count."
            )
        elif consistent and counts:
            logger.warning(
                f"Result: All checked counts are consistent ({counts[0]}) but DO NOT match the expected count ({EXPECTED_COUNT})."
            )
            if counts[0] == 2501:  # Specific message for the known issue
                logger.warning(
                    "This likely confirms the presence of one extra record in the databases compared to the loaded file (expected). Action may be needed depending on requirements."
                )
        elif counts:  # Some counts were checked, but they are inconsistent
            logger.error("Result: Counts are INCONSISTENT across storage layers!")
        else:  # No counts could be retrieved
            logger.error(
                "Result: Could not retrieve valid counts from any storage layer."
            )

    except Exception as e:
        logger.critical(f"An unexpected error occurred during verification: {e}")
        logger.critical(traceback.format_exc())
    finally:
        # Ensure connections are closed
        logger.info("Closing connections...")
        if pg_pool:
            await pg_pool.close()
        if neo4j_driver:
            await neo4j_driver.close()
        logger.info("Verification script finished.")


if __name__ == "__main__":
    # Ensure asyncio runs the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in script execution: {e}")
        logger.critical(traceback.format_exc())
