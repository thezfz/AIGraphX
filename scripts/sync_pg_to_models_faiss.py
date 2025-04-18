import logging
import argparse
import asyncio
import os
import json
import time
import numpy as np
import faiss  # type: ignore[import-untyped]
import psycopg_pool
from typing import List, Optional

# --- Project Imports ---
# Ensure paths are correct relative to the project structure
# Add project root to sys.path if necessary when running as script
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the settings object from config.py
from aigraphx.core.config import settings  # Reverted from settings
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.vectorization.embedder import TextEmbedder

# --- Logging Setup ---
# Use log level from settings
logging.basicConfig(
    level=settings.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Consider moving these to settings if they need to be configurable
EMBEDDING_BATCH_SIZE = settings.build_faiss_batch_size  # Use settings
PG_FETCH_BATCH_SIZE = (
    1000  # How many models to fetch from PG at once (used by repo method)
)


async def build_index(
    pg_repo: PostgresRepository,
    embedder: TextEmbedder,
    index_path: str,
    id_map_path: str,
    reset_index: bool = False,
) -> None:
    """Build and save a Faiss index with model embeddings from PostgreSQL.

    Args:
        pg_repo: PostgreSQL repository instance
        embedder: Text embedder instance
        index_path: Path to save the Faiss index
        id_map_path: Path to save the ID mapping
        reset_index: Whether to delete existing index files first
    """
    logger.info(f"Starting Faiss index build for models...")
    logger.info(f"Index path: {index_path}")
    logger.info(f"ID map path: {id_map_path}")
    logger.info(f"Reset existing index: {reset_index}")

    if reset_index:
        logger.info(f"Deleting existing index at {index_path}...")
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(id_map_path):
                os.remove(id_map_path)
            logger.info("Existing index files deleted.")
        except Exception as e:
            logger.warning(f"Error deleting existing index files: {e}")

    if os.path.exists(index_path) and not reset_index:
        logger.warning(
            f"Index file {index_path} already exists and reset=False. Skipping build."
        )
        # Optionally load existing index to check dimensions? For now, just skip.
        return

    start_time = time.time()
    all_model_ids = []
    all_embeddings_list = []
    total_models_processed = 0

    logger.info("Fetching model data and generating embeddings...")
    try:
        texts_batch: List[Optional[str]] = []
        ids_batch = []
        async for model_id, text in pg_repo.get_all_models_for_indexing():
            ids_batch.append(model_id)
            texts_batch.append(text)
            total_models_processed += 1

            if len(texts_batch) >= EMBEDDING_BATCH_SIZE:
                logger.info(f"Processing batch of {len(texts_batch)} models...")
                embeddings = embedder.embed_batch(texts_batch)
                if embeddings is not None and embeddings.shape[0] > 0:
                    all_embeddings_list.append(embeddings)
                    all_model_ids.extend(ids_batch)  # Add corresponding IDs
                else:
                    logger.warning(
                        f"Embedder returned None or empty for batch starting with ID {ids_batch[0]}. Skipping batch."
                    )

                # Clear batch
                texts_batch = []
                ids_batch = []
                logger.info(f"Total models processed so far: {total_models_processed}")

        # Process any remaining texts in the last batch
        if texts_batch:
            logger.info(f"Processing final batch of {len(texts_batch)} models...")
            embeddings = embedder.embed_batch(texts_batch)
            if embeddings is not None and embeddings.shape[0] > 0:
                all_embeddings_list.append(embeddings)
                all_model_ids.extend(ids_batch)
            else:
                logger.warning(
                    f"Embedder returned None or empty for final batch starting with ID {ids_batch[0]}. Skipping batch."
                )

    except Exception as e:
        logger.exception(f"Error fetching models or generating embeddings: {e}")
        return  # Stop the process

    if not all_embeddings_list:
        logger.error("No embeddings were generated. Cannot build index.")
        return

    logger.info(f"Concatenating {len(all_embeddings_list)} embedding batches...")
    try:
        embeddings_np = np.concatenate(all_embeddings_list, axis=0).astype("float32")
        logger.info(f"Total embeddings shape: {embeddings_np.shape}")
        logger.info(f"Total model IDs collected: {len(all_model_ids)}")

        # Verify consistency
        if embeddings_np.shape[0] != len(all_model_ids):
            logger.error(
                f"Mismatch between number of embeddings ({embeddings_np.shape[0]}) and IDs ({len(all_model_ids)}). Aborting index build."
            )
            return

    except ValueError as e:
        logger.error(
            f"Error concatenating embeddings: {e}. Check if batches have consistent dimensions."
        )
        return

    # --- Build Faiss Index ---
    embedding_dim = embedder.get_embedding_dimension()
    if embedding_dim <= 0:
        logger.error(
            "Could not determine embedding dimension from embedder. Cannot build index."
        )
        return

    logger.info(f"Building Faiss index (IndexFlatL2) with dimension {embedding_dim}...")
    # Using IndexFlatL2, which is simple and doesn't require training.
    # Suitable for exact search, performance depends on dataset size.
    # Consider other index types (e.g., IndexIVFFlat) for larger datasets.
    index = faiss.IndexFlatL2(embedding_dim)

    logger.info(f"Adding {embeddings_np.shape[0]} embeddings to the index...")
    try:
        index.add(embeddings_np)
        logger.info(f"Embeddings added successfully. Index total: {index.ntotal}")
    except Exception as e:
        logger.exception(f"Error adding embeddings to Faiss index: {e}")
        return

    # --- Create ID Map ---
    # Maps the sequential index position (0, 1, 2...) to the original model_id (string)
    logger.info("Creating ID map...")
    id_map = {i: model_id for i, model_id in enumerate(all_model_ids)}

    # --- Save Index and ID Map ---
    logger.info(f"Saving Faiss index to {index_path}...")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        logger.info("Index saved successfully.")
    except Exception as e:
        logger.exception(f"Error saving Faiss index: {e}")
        return  # Don't save map if index saving fails

    logger.info(f"Saving ID map to {id_map_path}...")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(id_map_path), exist_ok=True)
        with open(id_map_path, "w") as f:
            json.dump(id_map, f)
        logger.info("ID map saved successfully.")
    except Exception as e:
        logger.exception(f"Error saving ID map: {e}")

    end_time = time.time()
    logger.info(
        f"Faiss index build for models completed in {end_time - start_time:.2f} seconds."
    )
    logger.info(f"Total models indexed: {index.ntotal}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Faiss index for AI models from PostgreSQL data."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Faiss index and ID map before building.",
    )
    args = parser.parse_args()

    pool = None
    try:
        # Validate database URL
        db_url = settings.database_url
        if not db_url:
            logger.error("DATABASE_URL is not configured in settings. Exiting.")
            return

        # Initialize PG Pool using settings
        logger.info("Initializing PostgreSQL connection pool...")
        pool = psycopg_pool.AsyncConnectionPool(
            conninfo=db_url,  # Use validated db_url
            min_size=settings.pg_pool_min_size,
            max_size=settings.pg_pool_max_size,
        )
        await pool.wait()  # Wait for pool to be ready
        pg_repo = PostgresRepository(pool=pool)
        logger.info("PostgreSQL connection pool initialized.")

        # Initialize Embedder using settings
        logger.info("Initializing Text Embedder...")
        embedder = TextEmbedder(
            model_name=settings.sentence_transformer_model,
            device=settings.embedder_device,
        )
        if not embedder.model:
            logger.error("Failed to load embedding model. Exiting.")
            return
        logger.info("Text Embedder initialized.")

        # Build Index using settings for paths
        await build_index(
            pg_repo=pg_repo,
            embedder=embedder,
            index_path=settings.models_faiss_index_path,
            id_map_path=settings.models_faiss_mapping_path,
            reset_index=args.reset,
        )

    except psycopg_pool.PoolTimeout:  # Corrected: PoolTimeout
        logger.exception("Database connection pool timeout.")
    except psycopg_pool.PoolClosed:  # Corrected: PoolClosed
        logger.exception("Database connection pool is closed.")
    except Exception as e:
        logger.exception(f"An error occurred during the main execution: {e}")
    finally:
        if pool:
            logger.info("Closing PostgreSQL connection pool...")
            await pool.close()
            logger.info("PostgreSQL connection pool closed.")


if __name__ == "__main__":
    logger.info("Starting script: sync_pg_to_models_faiss.py")
    asyncio.run(main())
    logger.info("Script finished: sync_pg_to_models_faiss.py")
