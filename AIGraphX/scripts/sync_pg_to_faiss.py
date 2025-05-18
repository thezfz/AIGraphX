#!/usr/bin/env python
import asyncio
import os
import json
import logging
import sys
import time
import traceback
from typing import List, Tuple, AsyncGenerator, Optional, cast
import argparse

# Third-party imports
import psycopg_pool  # Import psycopg_pool
import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

# --- Project Imports ---
# Adjust the path dynamically to import from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use config directly
# from aigraphx.core import config # Keep this if needed elsewhere, but import settings
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.vectorization.embedder import TextEmbedder

# Import the settings object from config.py
from aigraphx.core.config import settings  # Reverted from settings

# --- Configuration ---
# Removed dotenv_path and load_dotenv call as settings handles .env loading

# --- Constants ---
# Constants like BATCH_SIZE are now read from settings where applicable
# (e.g., settings.build_faiss_batch_size)

# --- Logging Setup ---
logging.basicConfig(
    level=settings.log_level.upper(),  # Use log level from settings
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
        # Optionally add: logging.FileHandler('build_faiss.log')
    ],
)
logger = logging.getLogger(__name__)


# --- Database Function (Moved to Repository) ---
# async def fetch_papers_for_indexing(...)
# This logic is now encapsulated within PostgresRepository.get_all_paper_ids_and_text


# --- Main Logic ---
async def build_index(
    pg_repo: PostgresRepository,
    embedder: TextEmbedder,
    index_path: str,
    id_map_path: str,
    batch_size: int,  # Pass batch size explicitly
    reset_index: bool = False,
) -> None:
    """Build and save a Faiss index with paper embeddings from PostgreSQL.

    Args:
        pg_repo: PostgreSQL repository instance
        embedder: Text embedder instance
        index_path: Path to save the Faiss index
        id_map_path: Path to save the ID mapping
        batch_size: Number of papers to process in each batch
        reset_index: Whether to delete existing index files first
    """
    logger.info("Starting Faiss index build process for papers...")
    start_time = time.time()

    # --- Reset Logic ---
    if reset_index:
        logger.warning(
            "Reset flag specified. Deleting existing Faiss index and mapping files..."
        )
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"Deleted existing index file: {index_path}")
            if os.path.exists(id_map_path):
                os.remove(id_map_path)
                logger.info(f"Deleted existing ID map file: {id_map_path}")
        except OSError as e:
            logger.error(f"Error deleting existing Faiss files: {e}")

    # --- Ensure Directory Exists ---
    index_dir = os.path.dirname(index_path)
    map_dir = os.path.dirname(id_map_path)
    try:
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        if map_dir and map_dir != index_dir:
            os.makedirs(map_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directories for Faiss files: {e}")
        return

    faiss_index = None
    paper_id_map = {}
    papers_processed = 0
    faiss_idx_counter = 0

    try:
        # --- Initialize Embedder ---
        embedding_dim = embedder.get_embedding_dimension()
        logger.info(f"Embedder ready. Embedding dimension: {embedding_dim}")

        # --- Create Faiss Index ---
        logger.info(f"Creating Faiss IndexFlatL2 with dimension {embedding_dim}...")
        faiss_index = faiss.IndexFlatL2(embedding_dim)

        # --- Process in Batches ---
        logger.info(
            f"Fetching papers and generating embeddings in batches of {batch_size}..."
        )
        batch_papers = []
        batch_ids = []
        total_papers = 0  # Track total yielded

        # Use the repository method directly
        async for paper_id, summary in pg_repo.get_all_paper_ids_and_text():
            total_papers += 1
            # Ensure summary is a string, handle None
            text_to_embed = summary if summary is not None else ""
            batch_papers.append(text_to_embed)
            batch_ids.append(paper_id)

            if len(batch_papers) >= batch_size:
                batch_start_time = time.time()
                logger.info(
                    f"Processing batch {papers_processed // batch_size + 1} (Papers {papers_processed + 1}-{papers_processed + len(batch_papers)})..."
                )
                # Pass list of strings to embedder
                embeddings = embedder.embed_batch(batch_papers)
                if embeddings is not None and embeddings.shape[0] > 0:
                    faiss_index.add(embeddings)
                    for p_id in batch_ids:
                        paper_id_map[faiss_idx_counter] = p_id
                        faiss_idx_counter += 1
                    logger.info(
                        f"Batch {papers_processed // batch_size + 1} added to index ({embeddings.shape[0]} vectors). Time: {time.time() - batch_start_time:.2f}s. Total vectors in index: {faiss_index.ntotal}"
                    )
                else:
                    logger.warning(
                        f"Batch {papers_processed // batch_size + 1} resulted in no embeddings."
                    )
                papers_processed += len(batch_papers)
                batch_papers = []
                batch_ids = []

        # Process the last batch if any papers remain
        if batch_papers:
            batch_start_time = time.time()
            logger.info(
                f"Processing final batch (Papers {papers_processed + 1}-{papers_processed + len(batch_papers)})..."
            )
            embeddings = embedder.embed_batch(batch_papers)
            if embeddings is not None and embeddings.shape[0] > 0:
                faiss_index.add(embeddings)
                for p_id in batch_ids:
                    paper_id_map[faiss_idx_counter] = p_id
                    faiss_idx_counter += 1
                logger.info(
                    f"Final batch added to index ({embeddings.shape[0]} vectors). Time: {time.time() - batch_start_time:.2f}s. Total vectors in index: {faiss_index.ntotal}"
                )
            else:
                logger.warning("Final batch resulted in no embeddings.")
            papers_processed += len(batch_papers)

        if total_papers == 0:
            logger.warning(
                "No papers with summaries found in the database. Faiss index will be empty."
            )

        # --- Save Index and ID Map ---
        if faiss_index is not None and faiss_index.ntotal > 0:
            logger.info("Saving Faiss index and ID map...")
            try:
                logger.info(f"Writing Faiss index to {index_path}")
                faiss.write_index(faiss_index, index_path)
                logger.info("Index written successfully.")

                logger.info(f"Writing ID mapping to {id_map_path}")
                with open(id_map_path, "w") as f:
                    json.dump(paper_id_map, f)
                logger.info("ID mapping written successfully.")
                logger.info("Faiss index and ID map saved successfully.")
            except Exception as save_e:
                logger.error(f"Error saving Faiss index or ID map: {save_e}")
        elif faiss_index is not None and faiss_index.ntotal == 0:
            logger.warning("Faiss index is empty, nothing to save.")
        else:
            logger.error("Faiss index object is None, cannot save.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during Faiss index build: {e}")
        logger.error(traceback.format_exc())

    finally:
        end_time = time.time()
        logger.info(
            f"--- Faiss index build finished in {end_time - start_time:.2f} seconds ---"
        )
        if faiss_index:
            logger.info(f"Final index contains {faiss_index.ntotal} vectors.")
        else:
            logger.warning("Faiss index object was not successfully created.")


# --- main function --- #
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize PostgreSQL paper summaries to Faiss index."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Faiss index and mapping before starting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.build_faiss_batch_size,  # Use default from settings
        help="Number of papers to process in each batch.",
    )
    args = parser.parse_args()

    pool = None
    try:
        # Validate database URL
        db_url = settings.database_url
        if not db_url:
            logger.error("DATABASE_URL is not configured in settings. Exiting.")
            return

        logger.info("Creating database connection pool...")
        # Use psycopg_pool and settings
        # Corrected exception types based on linter feedback
        pool = psycopg_pool.AsyncConnectionPool(
            conninfo=db_url,  # Use validated db_url
            min_size=settings.pg_pool_min_size,  # Use settings
            max_size=settings.pg_pool_max_size,  # Use settings
        )
        await pool.wait()  # Wait for pool to be ready
        logger.info("Database connection pool created.")
        # Pass the pool directly, repo will manage connections
        pg_repo = PostgresRepository(pool)

        logger.info("Initializing text embedder...")
        # Use settings for embedder config
        embedder = TextEmbedder(
            model_name=settings.sentence_transformer_model,
            device=settings.embedder_device,
        )
        if not embedder.model:  # Basic check if model loaded
            logger.error("Failed to load embedding model. Exiting.")
            return
        logger.info("Text embedder initialized.")

        # Use settings for index/map paths
        await build_index(
            pg_repo=pg_repo,
            embedder=embedder,
            index_path=settings.faiss_index_path,
            id_map_path=settings.faiss_mapping_path,
            batch_size=args.batch_size,  # Use batch size from args/settings
            reset_index=args.reset,
        )

    except psycopg_pool.PoolTimeout:  # Corrected: PoolTimeout
        logger.exception("Database connection pool timeout.")
    except psycopg_pool.PoolClosed:  # Corrected: PoolClosed
        logger.exception("Database connection pool is closed.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the main execution: {e}")
    finally:
        if pool:
            logger.info("Closing PostgreSQL connection pool...")
            await pool.close()
            logger.info("PostgreSQL connection pool closed.")


if __name__ == "__main__":
    asyncio.run(main())
