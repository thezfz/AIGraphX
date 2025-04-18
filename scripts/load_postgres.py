#!/usr/bin/env python
import argparse
import asyncio
import os
import json
import logging
import sys
import traceback
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, Tuple, List

import psycopg  # Import psycopg
from psycopg_pool import AsyncConnectionPool  # Import the pool
from dotenv import load_dotenv

# --- Configuration ---
# Load .env specifically for the script
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# --- Constants ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print(
        "Critical Error: DATABASE_URL environment variable not set or empty.",
        file=sys.stderr,
    )
    sys.exit(1)

# Use Pool configuration from environment variables or defaults
PG_POOL_MIN_SIZE = int(os.getenv("PG_POOL_MIN_SIZE", "1"))
PG_POOL_MAX_SIZE = int(os.getenv("PG_POOL_MAX_SIZE", "10"))

DEFAULT_INPUT_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"  # Default value
CHECKPOINT_FILE = "data/pg_load_checkpoint.txt"
CHECKPOINT_INTERVAL = 100  # How often to save checkpoint (in lines)
BATCH_SIZE = 50  # How many records to process in one DB transaction

# Define tables to be truncated on reset
TRUNCATE_TABLES = [
    "model_paper_links",
    # "pwc_methods", # Assuming methods table exists if needed
    "pwc_repositories",
    "pwc_datasets",
    "pwc_tasks",
    "hf_models",
    "papers",
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.StreamHandler()
        # Optionally add: logging.FileHandler('load_postgres.log')
    ],
)
logger = logging.getLogger(__name__)

# --- Area Mapping ---
AREA_MAP = {
    "cs.CV": "CV",
    "cs.CL": "NLP",
    "cs.LG": "ML",
    "cs.AI": "AI",
    "cs.IR": "IR",
    "cs.RO": "Robotics",
    "stat.ML": "ML",
    # Add other mappings as needed
}


def get_area_from_category(primary_category: Optional[str]) -> Optional[str]:
    """Derives the area from the primary ArXiv category."""
    if not primary_category:
        return None
    # Handle potential subcategories like cs.CV.Computation
    main_category = (
        primary_category.split(".")[0] + "." + primary_category.split(".")[1]
        if "." in primary_category
        else primary_category
    )
    return AREA_MAP.get(main_category, "Other")  # Default to Other


# --- Helper Functions ---
def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Safely parse date strings (YYYY-MM-DD) into date objects."""
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str.split("T")[0])
    except (ValueError, TypeError):
        logger.warning(f"Could not parse date: {date_str}")
        return None


def parse_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """Safely parse ISO datetime strings into timezone-aware UTC datetime objects."""
    if not datetime_str:
        return None
    try:
        dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        # Ensure timezone-aware (convert to UTC if naive, although ISO should include offset)
        if dt.tzinfo is None:
            # This case should ideally not happen with fromisoformat on compliant strings,
            # but handle defensively by assuming UTC if no timezone info.
            logger.warning(f"Parsed datetime {datetime_str} was naive, assuming UTC.")
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        logger.warning(f"Could not parse datetime: {datetime_str}")
        return None


# --- Checkpointing ---
def _save_checkpoint(line_count: int) -> None:
    """Saves the number of lines successfully processed."""
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(line_count))
        logger.debug(f"Checkpoint saved: {line_count} lines processed.")
    except IOError as e:
        logger.error(f"Failed to save checkpoint to {CHECKPOINT_FILE}: {e}")


def _load_checkpoint(reset_checkpoint: bool = False) -> int:
    """Loads the number of lines already processed. If reset is True, ignores existing file."""
    if reset_checkpoint and os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            logger.info(
                f"Reset flag specified. Removed existing checkpoint file: {CHECKPOINT_FILE}"
            )
        except OSError as e:
            logger.error(
                f"Failed to remove checkpoint file {CHECKPOINT_FILE} during reset: {e}"
            )
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            processed_count = int(f.read().strip())
            logger.info(
                f"Checkpoint loaded: Resuming after {processed_count} processed lines."
            )
            return processed_count
    except (IOError, ValueError) as e:
        logger.error(
            f"Failed to load or parse checkpoint {CHECKPOINT_FILE}: {e}. Starting from 0."
        )
        return 0


# --- Database Operations (using psycopg) ---


async def insert_hf_model(
    conn: psycopg.AsyncConnection, model_data: Dict[str, Any]
) -> None:
    """Inserts or updates a model in the hf_models table using psycopg."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO hf_models (
                hf_model_id, hf_author, hf_sha, hf_last_modified, hf_downloads,
                hf_likes, hf_tags, hf_pipeline_tag, hf_library_name
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hf_model_id) DO UPDATE SET
                hf_author = EXCLUDED.hf_author,
                hf_sha = EXCLUDED.hf_sha,
                hf_last_modified = EXCLUDED.hf_last_modified,
                hf_downloads = EXCLUDED.hf_downloads,
                hf_likes = EXCLUDED.hf_likes,
                hf_tags = EXCLUDED.hf_tags,
                hf_pipeline_tag = EXCLUDED.hf_pipeline_tag,
                hf_library_name = EXCLUDED.hf_library_name,
                updated_at = NOW()
        """,
            (
                model_data.get("hf_model_id"),
                model_data.get("hf_author"),
                model_data.get("hf_sha"),
                parse_datetime(model_data.get("hf_last_modified")),
                model_data.get("hf_downloads"),
                model_data.get("hf_likes"),
                json.dumps(model_data.get("hf_tags"))
                if model_data.get("hf_tags")
                else None,
                model_data.get("hf_pipeline_tag"),
                model_data.get("hf_library_name"),
            ),
        )


async def get_or_insert_paper(
    conn: psycopg.AsyncConnection, paper_data: Dict[str, Any]
) -> Optional[int]:
    """
    Finds an existing paper by pwc_id or arxiv_id_base, or inserts a new one using psycopg.
    Returns the integer paper_id.
    """
    pwc_entry = paper_data.get("pwc_entry") or {}
    arxiv_meta = paper_data.get("arxiv_metadata") or {}
    pwc_id = pwc_entry.get("pwc_id")
    arxiv_id_base = paper_data.get("arxiv_id_base")
    arxiv_id_versioned = arxiv_meta.get("arxiv_id_versioned")
    primary_category = arxiv_meta.get("primary_category")
    area = get_area_from_category(primary_category)

    if not pwc_id and not arxiv_id_base:
        logger.warning(
            f"Skipping paper insert: Neither pwc_id nor arxiv_id_base found in data: {paper_data}"
        )
        return None

    async with conn.cursor() as cur:
        existing_id: Optional[int] = None
        if pwc_id:
            await cur.execute(
                "SELECT paper_id FROM papers WHERE pwc_id = %s", (pwc_id,)
            )
            row = await cur.fetchone()
            if row:
                existing_id = row[0]

        if not existing_id and arxiv_id_base:
            await cur.execute(
                "SELECT paper_id FROM papers WHERE arxiv_id_base = %s", (arxiv_id_base,)
            )
            row = await cur.fetchone()
            if row:
                existing_id = row[0]

        if existing_id:
            logger.debug(
                f"Found existing paper ID {existing_id} for pwc_id={pwc_id}, arxiv_id={arxiv_id_base}"
            )
            await cur.execute(
                "UPDATE papers SET area = %s, updated_at = NOW() WHERE paper_id = %s AND area IS NULL",
                (area, existing_id),
            )
            return existing_id
        else:
            logger.debug(
                f"Inserting new paper for pwc_id={pwc_id}, arxiv_id={arxiv_id_base}"
            )
            try:
                await cur.execute(
                    """
                    INSERT INTO papers (
                        pwc_id, arxiv_id_base, arxiv_id_versioned, title, authors, summary,
                        published_date, updated_date, pdf_url, doi, primary_category,
                        categories, pwc_title, pwc_url, area
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING paper_id
                """,
                    (
                        pwc_id,
                        arxiv_id_base,
                        arxiv_id_versioned,
                        arxiv_meta.get("title"),
                        json.dumps(arxiv_meta.get("authors"))
                        if arxiv_meta.get("authors")
                        else None,
                        arxiv_meta.get("summary"),
                        parse_date(arxiv_meta.get("published_date")),
                        parse_date(arxiv_meta.get("updated_date")),
                        arxiv_meta.get("pdf_url"),
                        arxiv_meta.get("doi"),
                        primary_category,
                        json.dumps(arxiv_meta.get("categories"))
                        if arxiv_meta.get("categories")
                        else None,
                        pwc_entry.get("title"),
                        pwc_entry.get(
                            "url_pdf"
                        ),  # Assuming this is the intended PWC URL
                        area,
                    ),
                )
                row = await cur.fetchone()
                if row and isinstance(row[0], int):
                    return row[0]
                else:
                    logger.error(
                        f"Failed to retrieve valid paper_id (int) after insertion for pwc_id={pwc_id}, arxiv_id={arxiv_id_base}. Got: {row}"
                    )
                    return None
            except Exception as e:
                logger.error(
                    f"Error inserting paper (pwc_id={pwc_id}, arxiv={arxiv_id_base}): {e}"
                )
                logger.error(traceback.format_exc())
                return None


async def insert_pwc_relation(
    conn: psycopg.AsyncConnection,
    paper_id: int,
    relation_type: str,
    items: Optional[List[str]],
) -> None:
    """Inserts related items (tasks, datasets) for a paper."""
    if not items:
        return
    table_name = f"pwc_{relation_type}"  # e.g., pwc_tasks, pwc_datasets
    column_name = f"{relation_type[:-1]}_name"  # e.g., task_name, dataset_name

    # Prepare data tuples: (paper_id, item_name)
    data_tuples = [(paper_id, item) for item in items]

    # Use executemany for batch insertion
    sql = f"INSERT INTO {table_name} (paper_id, {column_name}) VALUES (%s, %s) ON CONFLICT DO NOTHING"
    async with conn.cursor() as cur:
        await cur.executemany(sql, data_tuples)
    logger.debug(f"Inserted {len(data_tuples)} {relation_type} for paper_id {paper_id}")


async def insert_pwc_repositories(
    conn: psycopg.AsyncConnection, paper_id: int, repos: Optional[List[Dict[str, Any]]]
) -> None:
    """Inserts repository information for a paper."""
    if not repos:
        return
    # Prepare data tuples: (paper_id, url, stars, is_official, framework)
    data_tuples = [
        (
            paper_id,
            repo.get("url"),
            repo.get("stars"),
            repo.get("is_official"),
            repo.get("framework"),
        )
        for repo in repos
        if repo.get("url")  # Only insert if URL exists
    ]

    if not data_tuples:
        return

    sql = """
        INSERT INTO pwc_repositories (paper_id, url, stars, is_official, framework)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (paper_id, url) DO UPDATE SET
            stars = EXCLUDED.stars,
            is_official = EXCLUDED.is_official,
            framework = EXCLUDED.framework
    """
    async with conn.cursor() as cur:
        await cur.executemany(sql, data_tuples)
    logger.debug(
        f"Inserted/Updated {len(data_tuples)} repositories for paper_id {paper_id}"
    )


async def insert_model_paper_link(
    conn: psycopg.AsyncConnection, hf_model_id: str, paper_id: Optional[int]
) -> None:
    """Inserts a link between a Hugging Face model and a paper."""
    if not paper_id:
        logger.warning(f"Cannot link model {hf_model_id}: paper_id is missing.")
        return

    # Removed pwc_id and arxiv_id_base from INSERT statement and parameters
    # Assuming model_paper_links only has hf_model_id and paper_id as core columns
    # Check your schema (migration file) to confirm the correct columns.
    sql = """
        INSERT INTO model_paper_links (hf_model_id, paper_id)
        VALUES (%s, %s)
        ON CONFLICT (hf_model_id, paper_id) DO NOTHING
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (hf_model_id, paper_id))
    logger.debug(f"Linked model {hf_model_id} to paper_id {paper_id}")


# --- Main Processing Logic ---


async def process_batch(
    conn: psycopg.AsyncConnection, batch: List[Tuple[int, Dict[str, Any]]]
) -> int:
    """Processes a batch of model data within a single transaction."""
    processed_in_batch = 0
    # Start a transaction for the batch
    async with conn.transaction():
        for line_num, model_record in batch:
            hf_model_id = model_record.get("hf_model_id")
            if not hf_model_id:
                logger.warning(f"Skipping line {line_num}: Missing 'hf_model_id'.")
                continue

            try:
                # 1. Insert/Update HF Model
                await insert_hf_model(conn, model_record)

                # 2. Process linked papers
                linked_papers = model_record.get("linked_papers", [])
                if not isinstance(linked_papers, list):
                    logger.warning(
                        f"Skipping papers for model {hf_model_id} on line {line_num}: 'linked_papers' is not a list."
                    )
                    linked_papers = []

                for paper_data in linked_papers:
                    if not isinstance(paper_data, dict):
                        logger.warning(
                            f"Skipping invalid paper entry for model {hf_model_id} on line {line_num}: not a dictionary."
                        )
                        continue

                    # 3. Get or Insert Paper
                    paper_id = await get_or_insert_paper(conn, paper_data)

                    if paper_id:
                        # 4. Link Model to Paper (Call updated function)
                        await insert_model_paper_link(
                            conn,
                            hf_model_id,
                            paper_id,
                        )

                        # 5. Insert PWC Relations (Tasks, Datasets, Repos)
                        pwc_entry = paper_data.get("pwc_entry") or {}
                        await insert_pwc_relation(
                            conn, paper_id, "tasks", pwc_entry.get("tasks")
                        )
                        await insert_pwc_relation(
                            conn, paper_id, "datasets", pwc_entry.get("datasets")
                        )
                        # await insert_pwc_relation(conn, paper_id, "methods", pwc_entry.get("methods")) # Uncomment if methods are added
                        await insert_pwc_repositories(
                            conn, paper_id, pwc_entry.get("repositories")
                        )

                processed_in_batch += 1
            except Exception as e:
                logger.error(
                    f"Error processing record for model {hf_model_id} on line {line_num}: {e}"
                )
                logger.error(traceback.format_exc())
                # Decide whether to continue batch or raise error to rollback transaction
                # For robustness, log error and continue processing other records in batch

    return processed_in_batch


async def main(input_file_path: str, reset_checkpoint: bool) -> None:
    """Main function to orchestrate the data loading process."""
    processed_count = _load_checkpoint(reset_checkpoint)
    start_line = processed_count
    total_lines_processed = processed_count
    batch: List[Tuple[int, Dict[str, Any]]] = []
    lines_since_last_checkpoint = 0

    try:
        if DATABASE_URL is None:
            logger.error("DATABASE_URL is not set. Check your environment variables.")
            sys.exit(1)

        logger.info(f"Attempting to connect to database: {DATABASE_URL.split('@')[-1]}")
        async with AsyncConnectionPool(
            conninfo=DATABASE_URL, min_size=PG_POOL_MIN_SIZE, max_size=PG_POOL_MAX_SIZE
        ) as pool:
            logger.info("DB pool created via async context manager.")

            if reset_checkpoint:
                logger.warning("Reset flag. TRUNCATING tables...")
                async with pool.connection() as conn, conn.transaction():
                    for table in TRUNCATE_TABLES:
                        try:
                            await conn.execute(
                                f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"
                            )
                            logger.info(f"Truncated table: {table}")
                        except Exception as e:
                            logger.error(f"Error truncating table {table}: {e}")
                            raise
                logger.info("Tables truncated.")

            logger.info(
                f"Starting processing from line {start_line + 1} of {input_file_path}"
            )
            records_in_current_batch = 0
            try:
                with open(input_file_path, "r", encoding="utf-8") as infile:
                    for line_num, line in enumerate(infile, 1):
                        if line_num <= start_line:
                            continue

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            model_record = json.loads(line)
                            batch.append((line_num, model_record))
                            records_in_current_batch += 1

                            if records_in_current_batch >= BATCH_SIZE:
                                async with pool.connection() as conn:
                                    processed_in_batch = await process_batch(
                                        conn, batch
                                    )
                                    total_lines_processed += processed_in_batch
                                    lines_since_last_checkpoint += len(batch)
                                    logger.info(
                                        f"Processed batch ending line {line_num}. Total processed: {total_lines_processed}"
                                    )
                                batch = []
                                records_in_current_batch = 0

                                if lines_since_last_checkpoint >= CHECKPOINT_INTERVAL:
                                    _save_checkpoint(total_lines_processed)
                                    lines_since_last_checkpoint = 0

                        except json.JSONDecodeError:
                            logger.error(
                                f"Failed JSON parse line {line_num}: {line[:100]}... Skipping."
                            )
                        # Catch specific DB errors during batch processing? or rely on process_batch?

                    if batch:  # Process final batch
                        async with pool.connection() as conn:
                            processed_in_batch = await process_batch(conn, batch)
                            total_lines_processed += processed_in_batch
                            logger.info(
                                f"Processed final batch of {len(batch)}. Total processed: {total_lines_processed}"
                            )

                logger.info(
                    f"Finished file. Total lines processed across runs: {total_lines_processed}"
                )
                _save_checkpoint(total_lines_processed)

            except FileNotFoundError:
                logger.error(f"Input file not found: {input_file_path}")
                sys.exit(1)
            # Let other file IO errors propagate up to the outer try/except

    except psycopg.Error as db_err:
        logger.exception(f"Database connection or operational error: {db_err}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")
        sys.exit(1)
    # Pool is closed automatically by async with


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load Hugging Face model data into PostgreSQL."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=DEFAULT_INPUT_JSONL_FILE,
        help=f"Path to the input JSONL file (default: {DEFAULT_INPUT_JSONL_FILE})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and truncate tables before loading.",
    )
    args = parser.parse_args()

    logger.info(f"Starting data load from: {args.file}")
    if args.reset:
        logger.warning(
            "RESET flag is active. Checkpoint will be ignored and tables truncated."
        )

    asyncio.run(main(input_file_path=args.file, reset_checkpoint=args.reset))

    logger.info("Data loading process finished.")
