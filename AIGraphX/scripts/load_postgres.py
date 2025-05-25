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
import yaml  # 新增 PyYAML 导入
from yaml.scanner import ScannerError # 新增 ScannerError 导入
import re # 保留 re 以备其他可能的用途，或在此处移除（如果不再需要）

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

# --- Known Language Codes for Filtering ---
KNOWN_LANGUAGE_CODES = frozenset([
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az", 
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs", 
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy", 
    "da", "de", "dv", "dz", 
    "ee", "el", "en", "eo", "es", "et", "eu", 
    "fa", "ff", "fi", "fj", "fo", "fr", "fy", 
    "ga", "gd", "gl", "gn", "gu", "gv", 
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz", 
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu", 
    "ja", "jv", "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky", 
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv", 
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", 
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny", 
    "oc", "oj", "om", "or", "os", 
    "pa", "pi", "pl", "ps", "pt", 
    "qu", "rm", "rn", "ro", "ru", "rw", 
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw", 
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty", 
    "ug", "uk", "ur", "uz", 
    "ve", "vi", "vo", 
    "wa", "wo", 
    "xh", "yi", "yo", 
    "za", "zh", "zu"
])
# --- End Known Language Codes ---


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


# 新增的辅助函数，用于从 README frontmatter 提取 tasks using PyYAML
def _extract_tasks_from_readme_yaml(readme_content: Optional[str]) -> Optional[List[str]]:
    if not readme_content:
        return None
    
    frontmatter_text: Optional[str] = None
    try:
        # 使用正则表达式从 README 文本中分离出 YAML frontmatter 部分
        # (--- 包裹的部分)
        match = re.search(r"^\s*---\s*\n(.*?)^\s*---\s*\n", readme_content, re.DOTALL | re.MULTILINE)
        if not match:
            # logger.debug("No YAML frontmatter found in README.") # Optional: log if no frontmatter
            return None 
        frontmatter_text = match.group(1)
    except Exception as e:
        logger.warning(f"Regex error extracting frontmatter from README: {e}")
        return None

    if frontmatter_text is None: # Should be caught by `if not match` but defensive
        return None

    try:
        # 解析 YAML frontmatter
        parsed_yaml = yaml.safe_load(frontmatter_text)
        
        if not isinstance(parsed_yaml, dict):
            # logger.debug("Parsed YAML frontmatter is not a dictionary.") # Optional
            return None

        tasks_data = parsed_yaml.get("tasks")

        if tasks_data is None:
            # logger.debug("No 'tasks' key found in YAML frontmatter.") # Optional
            return None
        
        raw_tasks_list: Optional[List[Any]] = None
        if isinstance(tasks_data, str):
            # logger.info(f"Found 'tasks' in YAML as a single string: '{tasks_data}'. Wrapping it in a list.")
            raw_tasks_list = [tasks_data]
        elif isinstance(tasks_data, list):
            raw_tasks_list = tasks_data
        else:
            logger.warning(f"Found 'tasks' in YAML, but it's not a list or a single string: {type(tasks_data)}. Content: {tasks_data}")
            return None

        if not raw_tasks_list: # Should not happen if tasks_data was string or list, but defensive
             return None

        # Filter out non-string items and potential language codes
        final_tasks: List[str] = []
        for task_item in raw_tasks_list:
            if not isinstance(task_item, str):
                logger.warning(f"Skipping non-string item in 'tasks' list: {task_item} (type: {type(task_item)})")
                continue
            
            if task_item.lower() in KNOWN_LANGUAGE_CODES:
                logger.info(f"Filtering out potential language code '{task_item}' from 'tasks' list.")
            else:
                final_tasks.append(task_item)
        
        return final_tasks if final_tasks else None # Return list if not empty, else None

    except ScannerError as e:
        logger.warning(f"PyYAML ScannerError parsing frontmatter: {e}. Frontmatter text (first 500 chars): \\n{frontmatter_text[:500]}...")
        return None
    except Exception as e:
        logger.warning(f"General error parsing YAML frontmatter: {e}")
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
    
    # Extract and process base_model from hf_tags
    base_model_values = []
    hf_tags = model_data.get("hf_tags")
    if isinstance(hf_tags, list):
        for tag in hf_tags:
            if isinstance(tag, str) and tag.startswith("base_model:"):
                base_model_id = tag.split(":", 1)[1].strip()
                if base_model_id: # Ensure it's not an empty string after stripping
                    base_model_values.append(base_model_id)
    
    processed_base_models_json = None
    if base_model_values:
        if len(base_model_values) == 1:
            # If only one base model, store as a JSON string directly
            processed_base_models_json = json.dumps(base_model_values[0])
        else:
            # If multiple base models, store as a JSON array of strings
            processed_base_models_json = json.dumps(base_model_values)
            
    # --- 新增：从 README 提取 tasks using PyYAML ---
    readme_tasks_list = _extract_tasks_from_readme_yaml(model_data.get("hf_readme_content"))
    hf_readme_tasks_json = json.dumps(readme_tasks_list) if readme_tasks_list else None
    # --- 结束新增 ---
            
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO hf_models (
                hf_model_id, hf_author, hf_sha, hf_last_modified, hf_downloads,
                hf_likes, hf_tags, hf_pipeline_tag, hf_library_name,
                hf_readme_content, hf_dataset_links, hf_base_models,
                hf_readme_tasks 
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hf_model_id) DO UPDATE SET
                hf_author = EXCLUDED.hf_author,
                hf_sha = EXCLUDED.hf_sha,
                hf_last_modified = EXCLUDED.hf_last_modified,
                hf_downloads = EXCLUDED.hf_downloads,
                hf_likes = EXCLUDED.hf_likes,
                hf_tags = EXCLUDED.hf_tags,
                hf_pipeline_tag = EXCLUDED.hf_pipeline_tag,
                hf_library_name = EXCLUDED.hf_library_name,
                hf_readme_content = EXCLUDED.hf_readme_content,
                hf_dataset_links = EXCLUDED.hf_dataset_links,
                hf_base_models = EXCLUDED.hf_base_models,
                hf_readme_tasks = EXCLUDED.hf_readme_tasks, 
                updated_at = NOW()
        """,
            (
                model_data.get("hf_model_id"),
                model_data.get("hf_author"),
                model_data.get("hf_sha"),
                parse_datetime(model_data.get("hf_last_modified")),
                model_data.get("hf_downloads"),
                model_data.get("hf_likes"),
                json.dumps(hf_tags) # Store all original tags as before
                if hf_tags 
                else None,
                model_data.get("hf_pipeline_tag"),
                model_data.get("hf_library_name"),
                model_data.get("hf_readme_content"),
                json.dumps(model_data.get("hf_dataset_links"))
                if model_data.get("hf_dataset_links")
                else None,
                processed_base_models_json, # Use the newly processed base models
                hf_readme_tasks_json, # 新增列的数据
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
    conference = pwc_entry.get("conference")

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
            update_clauses = []
            update_params = []
            if area:
                update_clauses.append("area = %s")
                update_params.append(area)
            if conference:
                update_clauses.append("conference = %s")
                update_params.append(conference)
            
            if update_clauses:
                conditions_str = ' OR '.join([f"{col.split(' = ')[0]} IS NULL" for col in update_clauses])
                update_query = f"UPDATE papers SET {', '.join(update_clauses)}, updated_at = NOW() WHERE paper_id = %s AND ({conditions_str})"
                final_update_params = tuple(update_params + [existing_id])
                await cur.execute(update_query, final_update_params)
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
                        published_date, updated_at, pdf_url, doi, primary_category,
                        categories, pwc_title, pwc_url, area, conference
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        parse_datetime(arxiv_meta.get("updated_date")),
                        arxiv_meta.get("pdf_url"),
                        arxiv_meta.get("doi"),
                        primary_category,
                        json.dumps(arxiv_meta.get("categories"))
                        if arxiv_meta.get("categories")
                        else None,
                        pwc_entry.get("title"),
                        pwc_entry.get("pwc_url"),
                        area,
                        conference,
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
    # Correctly generate table and column names
    table_name = f"pwc_{relation_type}s"  # e.g., pwc_tasks, pwc_datasets
    column_name = f"{relation_type}_name"  # e.g., task_name, dataset_name

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
    # Prepare data tuples: (paper_id, url, stars, is_official, framework, license, language)
    data_tuples = [
        (
            paper_id,
            repo.get("url"),
            repo.get("stars"),
            repo.get("is_official"),
            repo.get("framework"),
            repo.get("license"),
            repo.get("language"),
        )
        for repo in repos
        if repo.get("url")  # Only insert if URL exists
    ]

    if not data_tuples:
        return

    sql = """
        INSERT INTO pwc_repositories (paper_id, url, stars, is_official, framework, license, language)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_id, url) DO UPDATE SET
            stars = EXCLUDED.stars,
            is_official = EXCLUDED.is_official,
            framework = EXCLUDED.framework,
            license = EXCLUDED.license,
            language = EXCLUDED.language
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
    """Processes a batch of records within a single transaction."""
    processed_in_batch = 0
    successful_lines_in_batch = 0  # Track successful lines within the batch
    async with conn.transaction():
        logger.debug(f"Starting transaction for batch of {len(batch)} records.")
        for line_num, record in batch:
            try:
                # Assume record processing starts successfully unless exception occurs
                record_processed_successfully = True

                # --- Process Hugging Face Model ---
                hf_model_id = record.get("hf_model_id")
                if hf_model_id:
                    await insert_hf_model(conn, record)
                else:
                    logger.warning(f"Record on line {line_num} missing hf_model_id.")
                    record_processed_successfully = (
                        False  # Mark as unsuccessful if critical ID missing
                    )
                    # continue # Optionally skip further processing for this record

                # --- Process Linked Papers (Iterate through the list) ---
                linked_papers = record.get("linked_papers", [])
                if not isinstance(linked_papers, list):
                    logger.warning(
                        f"Record on line {line_num}: 'linked_papers' is not a list. Skipping paper processing."
                    )
                    linked_papers = []  # Treat as empty list

                # Use a flag to track if *any* paper was successfully processed for this model record
                at_least_one_paper_processed = False

                for paper_data in linked_papers:
                    if not isinstance(paper_data, dict):
                        logger.warning(
                            f"Skipping invalid paper entry for model {hf_model_id} on line {line_num}: not a dictionary."
                        )
                        continue  # Skip this invalid paper entry

                    # --- Process Single Paper (Get or Insert) ---
                    # Pass the individual paper_data dictionary here
                    paper_id = await get_or_insert_paper(conn, paper_data)

                    # --- Link Model and Paper (only if both exist) ---
                    if hf_model_id and paper_id:
                        await insert_model_paper_link(conn, hf_model_id, paper_id)
                        at_least_one_paper_processed = True  # Mark success if linked
                    # Log cases where linking didn't happen (optional)
                    # elif paper_id and not hf_model_id:
                    #     logger.debug(f"Paper on line {line_num} processed but no HF model ID.")
                    # elif hf_model_id and not paper_id:
                    #     logger.debug(f"HF model {hf_model_id} on line {line_num} processed but paper insertion failed.")

                    # --- Process PWC Relations and Repositories (only if paper was successfully inserted/found) ---
                    if paper_id:
                        # IMPORTANT: Get pwc_entry from paper_data, not the top-level record
                        pwc_entry = paper_data.get("pwc_entry") or {}
                        await insert_pwc_relation(
                            conn, paper_id, "task", pwc_entry.get("tasks")
                        )
                        await insert_pwc_relation(
                            conn, paper_id, "method", pwc_entry.get("methods")
                        )
                        await insert_pwc_relation(
                            conn,
                            paper_id,
                            "dataset",
                            pwc_entry.get("datasets_used"),  # Assuming field name
                        )
                        await insert_pwc_repositories(
                            conn, paper_id, pwc_entry.get("repositories")
                        )
                        at_least_one_paper_processed = True  # Also mark success here
                    else:
                        # If get_or_insert_paper returned None, the paper processing failed for this entry
                        logger.warning(
                            f"Failed to get or insert paper for entry in linked_papers on line {line_num}. Paper data: {paper_data}"
                        )
                        # Consider if this failure should mark the whole model record as failed
                        # record_processed_successfully = False

                # Increment the main counter only if the model record itself was deemed successful
                # (e.g., hf_model_id was present, and potentially if at least one paper linked if required)
                if (
                    record_processed_successfully
                ):  # Adjust this condition based on requirements
                    processed_in_batch += 1
                    successful_lines_in_batch += 1  # Increment success counter
                    logger.debug(
                        f"Successfully processed record from line {line_num} (including linked papers if any)."
                    )
                else:
                    logger.warning(
                        f"Marked record from line {line_num} as processed with errors/skips."
                    )
                    # Even if marked as error, we might count it towards the total lines *attempted* in the batch
                    processed_in_batch += 1

            except Exception as e:
                # Log detailed error including traceback and the problematic record line number
                tb_str = traceback.format_exc()
                logger.error(
                    f"Error processing record from line {line_num}: {e}\\nRecord: {record}\\nTraceback:\\n{tb_str}"
                )
                # Re-raise the exception to trigger the automatic rollback of conn.transaction()
                raise  # This will rollback the *entire* batch

    # If the 'with conn.transaction()' block completes without exceptions, commit is automatic.
    # If an exception occurs, rollback is automatic.
    logger.debug(
        f"Transaction for batch completed (Commit or Rollback occurred). Successfully processed {successful_lines_in_batch} lines in this attempt."
    )
    # Return the count of lines successfully processed within the transaction
    # If an exception caused rollback, this will be 0 from the perspective of the DB, but we return the count *attempted* before failure.
    # Let's return successful_lines_in_batch to be more accurate about what potentially committed.
    return successful_lines_in_batch


async def main(input_file_path: str, reset_db: bool, reset_checkpoint: bool) -> None:
    """Main function to load data from JSONL into PostgreSQL."""
    logger.info(f"Starting data load from: {input_file_path}")
    logger.info(f"Reset Checkpoint: {reset_checkpoint}")
    logger.info(f"Reset Database (ignored by script, handled by tests): {reset_db}")

    pool: Optional[AsyncConnectionPool] = None # Add type hint
    processed_count = 0
    batch_count = 0
    error_count = 0
    current_batch: List[Tuple[int, Dict[str, Any]]] = []
    start_line = _load_checkpoint(reset_checkpoint)
    # Initialize i here, it's used in the finally block for checkpointing
    i = start_line -1 # if start_line is 0, i=-1, so first line is 0+1=1. If start_line is N, means N lines processed, so next line is N.

    # reset_db flag is now only informational for logging
    # Database truncation/cleanup is expected to be handled externally (e.g., by test fixtures)

    try:
        # Initialize Connection Pool
        logger.info(f"Initializing database pool for {DATABASE_URL}...")
        if not DATABASE_URL:
            logger.critical("DATABASE_URL not configured. Exiting.")
            return

        pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=PG_POOL_MIN_SIZE,
            max_size=PG_POOL_MAX_SIZE,
            open=True,  # Open pool on creation
            # Increase timeout if needed for long transactions or slow DB
            # timeout=60.0,
        )
        logger.info("Database pool initialized.")

        # Process Input File
        with open(input_file_path, "r") as infile:
            for i, line in enumerate(infile):
                line_num = i + 1
                if line_num <= start_line:
                    continue  # Skip already processed lines

                try:
                    data = json.loads(line)
                    if not isinstance(data, dict) or not data.get("hf_model_id"):
                        logger.warning(
                            f"Skipping line {line_num}: Invalid format or missing 'hf_model_id'. Content: {line.strip()}"
                        )
                        error_count += 1
                        continue

                    current_batch.append((line_num, data))

                    if len(current_batch) >= BATCH_SIZE:
                        batch_start_line = current_batch[0][0]
                        logger.info(
                            f"Processing batch starting at line {batch_start_line} (size: {len(current_batch)})..."
                        )
                        async with (
                            pool.connection() as conn
                        ):  # Get connection for batch
                            lines_processed_in_batch = await process_batch(
                                conn, current_batch
                            )
                        processed_count += lines_processed_in_batch
                        batch_count += 1
                        current_batch = []
                        if batch_count % (CHECKPOINT_INTERVAL // BATCH_SIZE) == 0:
                            _save_checkpoint(line_num)

                except json.JSONDecodeError:
                    logger.error(
                        f"Skipping line {line_num}: Invalid JSON. Content: {line.strip()}"
                    )
                    error_count += 1
                except Exception as e:
                    logger.error(
                        f"Error processing line {line_num}: {e}. Content: {line.strip()}",
                        exc_info=True,
                    )
                    error_count += 1
                    # Optionally break or implement retry logic here

            # Process the last partial batch
            if current_batch:
                batch_start_line = current_batch[0][0]
                logger.info(
                    f"Processing final batch starting at line {batch_start_line} (size: {len(current_batch)})..."
                )
                async with pool.connection() as conn:
                    lines_processed_in_batch = await process_batch(conn, current_batch)
                processed_count += lines_processed_in_batch

            # Final checkpoint save
            # Corrected condition: Save checkpoint if we processed any new lines
            final_line_num = i + 1  # Use the actual last line number read
            if final_line_num > start_line:
                _save_checkpoint(final_line_num)

    except FileNotFoundError:
        logger.critical(f"Input file not found: {input_file_path}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if pool:
            await pool.close()
            logger.info("Database pool closed.")
        logger.info(
            f"Data loading finished. Total lines processed successfully: {processed_count}. Total errors: {error_count}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load AI Graph X data from JSONL file into PostgreSQL."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_JSONL_FILE,
        help=f"Path to the input JSONL file (default: {DEFAULT_INPUT_JSONL_FILE})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing checkpoint and start loading from the beginning of the input file.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="DEPRECATED/IGNORED: Database reset is handled externally (e.g., tests). This flag is ignored.",
    )

    args = parser.parse_args()

    asyncio.run(main(args.input_file, args.reset_db, args.reset))
