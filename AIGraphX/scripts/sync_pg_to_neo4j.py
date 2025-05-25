#!/usr/bin/env python
import asyncio
import logging
import os
import traceback  # Import traceback
from typing import Optional, Dict, Any, List, Union, Set  # Import List, Union, and Set
import json
import sys  # Import sys for path manipulation
import argparse  # Import argparse
import re  # Import re for regex operations

# Third-party imports
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver
from psycopg_pool import AsyncConnectionPool  # Import PG Pool

# We need asyncpg for the pool creation and exception handling
import asyncpg  # type: ignore[import-untyped]

# Adjust import paths assuming script is run from project root OR handle fallback
try:
    # Try direct imports first (if running as module or PYTHONPATH is set)
    from aigraphx.repositories.postgres_repo import PostgresRepository
    from aigraphx.repositories.neo4j_repo import Neo4jRepository
except ImportError:
    # Fallback if running as a standalone script
    print("INFO: Failed initial import, attempting path modification...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        print(f"INFO: Adding project root to sys.path: {project_root}")
        sys.path.insert(0, project_root)

    # Retry imports after modifying path
    # --- Start of Corrected Block ---
    try:
        # Correctly indented imports UNDER the second try
        from aigraphx.repositories.postgres_repo import PostgresRepository
        from aigraphx.repositories.neo4j_repo import Neo4jRepository

        # Correctly indented print UNDER the second try
        print("INFO: Successfully imported modules after path modification.")
    except ImportError as e:  # Correctly aligned except with the second try
        # Correctly indented print statements UNDER the second except
        print(
            f"CRITICAL: Failed to import required modules even after path modification: {e}"
        )
        print(
            "CRITICAL: Ensure the script is run from the project root or the PYTHONPATH is set correctly."
        )
        # Correctly indented sys.exit UNDER the second except
        sys.exit(1)  # Exit if core components cannot be imported
    # --- End of Corrected Block ---


# --- Configuration ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# PostgreSQL Connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set.", file=sys.stderr)
    sys.exit(1)

# Neo4j Connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    print(
        "Error: Neo4j connection details (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) not fully set.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- Constants ---
PG_FETCH_BATCH_SIZE = 500  # How many records to fetch from PG at a time
NEO4J_WRITE_BATCH_SIZE = 200  # How many records to write to Neo4j in one batch

# --- Task Filtering Constants ---
KNOWN_LANGUAGE_CODES = frozenset([
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az", "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy", "da", "de", "dv", "dz", "ee", "el", "en", "eo", "es", "et", "eu", "fa", "ff",
    "fi", "fj", "fo", "fr", "fy", "ga", "gd", "gl", "gn", "gu", "gv", "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz", "ia", "id",
    "ie", "ig", "ii", "ik", "io", "is", "it", "iu", "ja", "jv", "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku",
    "kv", "kw", "ky", "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv", "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny", "oc", "oj", "om", "or", "os", "pa", "pi", "pl", "ps", "pt", "qu",
    "rm", "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv",
    "sw", "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty", "ug", "uk", "ur", "uz", "ve", "vi", "vo",
    "wa", "wo", "xh", "yi", "yo", "za", "zh", "zu"
])

KNOWN_LIBRARY_NAMES = frozenset([
    "diffusers", "safetensors", "transformers", "pytorch", "tensorflow", "keras", "timm", "accelerate", 
    "peft", "bitsandbytes", "sentence-transformers", "spacy", "stanza", "allennlp", "fairseq", 
    "speechbrain", "espnet", "asteroid", "pyannote-audio", "openvino", "onnx", "tensorrt", 
    "coremltools", "mlx", "jax", "unsloth", "tensorboard", "tf-keras", "transformers.js",
    "mergekit", "trl", "vllm", "axolotl", "alignment-handbook", "ai-toolkit", "core-ml" # core-ml is a repeat but ok
])

KNOWN_FILE_TYPES_FORMATS = frozenset([
    "gguf", "ggml", "bin", "pt", "h5", "json", "yaml", "zip", "tar.gz"
])

KNOWN_LICENSES = frozenset([
    "mit", "apache-2.0", "gpl-3.0", "cc-by-sa-4.0", "openrail", "bigscience-openrail-m", "bsd", "cc0" # added bsd, cc0 from previous
])

KNOWN_ORGANIZATIONS_SET = frozenset([
    "meta", "facebook", "google", "openai", "microsoft", "nvidia", "intel", "ibm", "amazon", "apple", 
    "huggingface", "stabilityai", "deepmind", "anthropic", "mistralai", "cohere", "baai", "salesforce", 
    "stanford", "berkeley", "cmu", "mit", "eleutherai", "togethercomputer", "tencent", "alibaba", "bytedance",
    "thudm" # Added from your list
])

TRAINING_EVAL_DEPLOY_KEYWORDS = frozenset([
    "training", "evaluation", "inference", "deployment", "optimization", "benchmark", "metrics", "loss", 
    "optimizer", "checkpoint", "lora", "qlora", "adapter", "prompt-tuning", "few-shot", "zero-shot", 
    "fine-tuning", "pre-training", "distilled", "pruned", "quantized", "converged"
])

HUGGINGFACE_HUB_NON_TASK_TAGS = frozenset([
    "endpoints-compatible", "autotrain-compatible", "custom-code", "model-index", "generated-from-trainer", 
    "pytorch-model-hub-mixin", "model-hub-mixin", "space-compatible", "gradio", "streamlit", "docker", 
    "paperswithcode", "config", "tokenizer", "modelcard", "readme", "license", "gguf-my-repo" # Added from your list
])

HARDWARE_KEYWORDS = frozenset([
    "gpu", "cpu", "tpu", "nvidia-a100", "cuda"
])

# More general non-task words, including broad/meta categories from before
GENERAL_NON_TASK_WORDS = frozenset([
    "vision", "audio", "speech", "chat", "code", "vision-language", "language-model", "multi-modal", "agent", 
    "reinforcement-learning", "large language model", "function calling", "function-calling", "tool-use",
    "multilingual", "uncensored", "experimental", "deprecated", "lite", "small", "base", "large", "tiny", 
    "dev", "core", "full", "mini", "edition", "preview", "alpha", "beta", "official", "community", "research",
    "example", "demo", "tool", "utility", "pipeline", "pytorch_model.bin", "tf_model.h5"
])

# Combine all keyword-based exclusion sets (after lowercasing them)
EXCLUDED_KEYWORDS_RAW = (
    KNOWN_LANGUAGE_CODES |
    KNOWN_LIBRARY_NAMES |
    KNOWN_FILE_TYPES_FORMATS |
    KNOWN_LICENSES |
    KNOWN_ORGANIZATIONS_SET |
    TRAINING_EVAL_DEPLOY_KEYWORDS |
    HUGGINGFACE_HUB_NON_TASK_TAGS |
    HARDWARE_KEYWORDS |
    GENERAL_NON_TASK_WORDS
)
# Normalize by replacing spaces, underscores, and colons with hyphens for keyword matching
NORMALIZED_EXCLUDED_KEYWORDS = frozenset([kw.lower().replace(" ", "-").replace("_", "-").replace(":", "-") for kw in EXCLUDED_KEYWORDS_RAW])

# Regex patterns for exclusion
ARXIV_PATTERN = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)
QUANT_BIT_PATTERN = re.compile(r"^\d+-?bit$", re.IGNORECASE)
QUANT_FP_INT_PATTERN = re.compile(r"^(fp|int)(4|8|16|32)$")
QUANT_GGUF_PATTERN = re.compile(r"^q\\d+_[0-9a-z_]+$", re.IGNORECASE) # GGUF quant types e.g. q4_0, q8_k_m
# More explicit pattern for base_model variations
BASE_MODEL_PREFIX_PATTERN = re.compile(r"^base_model:", re.IGNORECASE) # Matches underscore version
DIFFUSER_PREFIX_PATTERN = re.compile(r"^diffuser(s|:).*:?", re.IGNORECASE) # New pattern for diffuser/diffusers/diffuser: prefix
VERSION_PATTERN = re.compile(r"^v\\d+(\\.\\d+)*[a-zA-Z]*$", re.IGNORECASE)
NUMERIC_LIKE_PATTERN = re.compile(r"^\d+(\.\d+){0,2}$")
# Pattern for license: or region: like tags (more flexible than just keyword list for these)
PREFIX_COLON_PATTERN = re.compile(r"^(license|region|template|library|dataset):", re.IGNORECASE)


# Function to check if a tag should be excluded (renamed and enhanced)
def is_likely_not_a_task(tag_str: Union[str, None]) -> bool:
    # --- START BYTE REPRESENTATION LOG ---
    if isinstance(tag_str, str):
        try:
            bytes_repr = tag_str.encode('utf-8', 'surrogateescape')
            hex_repr = bytes_repr.hex(' ')
            logger.debug(f"BYTE_DEBUG: tag_str='{tag_str[:50]}', bytes={bytes_repr!r}, hex='{hex_repr}'")
        except Exception as e:
            logger.debug(f"BYTE_DEBUG: Error encoding tag_str='{tag_str[:50]}': {e}")
    # --- END BYTE REPRESENTATION LOG ---

    if not isinstance(tag_str, str) or not tag_str.strip():
        return True # Exclude empty or non-string tags
    
    original_tag_for_logging = tag_str # For logging
    normalized_tag_for_keywords = tag_str.lower().strip().replace(" ", "-").replace("_", "-").replace(":", "-")
    tag_lower_original_form = tag_str.lower().strip() 

    # Specific logging for base-model related tags and other problematic ones
    is_problematic_tag_for_logging = (
        "autotrain" in tag_lower_original_form or
        "endpoints" in tag_lower_original_form or
        tag_lower_original_form.startswith("base_model") # Covers base_model: and base_model_finetune: etc.
    )

    if is_problematic_tag_for_logging:
        logger.debug(
            f"DEBUG_FOCUS: Checking specific tag: Original='{original_tag_for_logging}', "
            f"Normalized for Keywords='{normalized_tag_for_keywords}', "
            f"Lower Original Form='{tag_lower_original_form}'"
        )

    if normalized_tag_for_keywords in NORMALIZED_EXCLUDED_KEYWORDS:
        logger.debug(f"Tag '{original_tag_for_logging}' (normalized: '{normalized_tag_for_keywords}') found in NORMALIZED_EXCLUDED_KEYWORDS. Excluding.")
        return True
    
    # Regex checks 
    if ARXIV_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched ARXIV_PATTERN. Excluding.")
        return True
    if QUANT_BIT_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched QUANT_BIT_PATTERN. Excluding.")
        return True
    if QUANT_FP_INT_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched QUANT_FP_INT_PATTERN. Excluding.")
        return True
    if QUANT_GGUF_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched QUANT_GGUF_PATTERN. Excluding.")
        return True

    # Enhanced logging for BASE_MODEL_PREFIX_PATTERN
    # ---- START DETAILED CHAR LOGGING ----
    if tag_lower_original_form.startswith("base_model"): # Condition uses underscore
        char_details = [(char, ord(char)) for char in tag_lower_original_form[:20]] # Log first 20 chars
        logger.debug(f"CHAR_DEBUG for '{tag_lower_original_form}': {char_details}")
    # ---- END DETAILED CHAR LOGGING ----
    match_result = BASE_MODEL_PREFIX_PATTERN.match(tag_lower_original_form)
    logger.debug(f"DEBUG_REGEX_CHECK: Pattern='{BASE_MODEL_PREFIX_PATTERN.pattern}', String='{tag_lower_original_form}', MatchResult={bool(match_result)}")

    if is_problematic_tag_for_logging: # This log is mostly for focus
        logger.debug(f"DEBUG_FOCUS:    Attempting match with BASE_MODEL_PREFIX_PATTERN ('{BASE_MODEL_PREFIX_PATTERN.pattern}') on '{tag_lower_original_form}'. Match object: {match_result}")
    
    if match_result: # Use the stored match_result
        if is_problematic_tag_for_logging:
            logger.debug(f"DEBUG_FOCUS:    SUCCESS: Tag '{original_tag_for_logging}' matched BASE_MODEL_PREFIX_PATTERN. Excluding.")
        else:
            logger.debug(f"Tag '{original_tag_for_logging}' matched BASE_MODEL_PREFIX_PATTERN. Excluding.")
        return True # Exclude if pattern matches
    elif is_problematic_tag_for_logging: # Explicitly log if it's a problematic tag but didn't match
        logger.debug(f"DEBUG_FOCUS:    FAILURE: Tag '{original_tag_for_logging}' (processed as '{tag_lower_original_form}') did NOT match BASE_MODEL_PREFIX_PATTERN ('{BASE_MODEL_PREFIX_PATTERN.pattern}'). Not excluding based on this rule.")

    if DIFFUSER_PREFIX_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched DIFFUSER_PREFIX_PATTERN. Excluding.")
        return True
    if VERSION_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched VERSION_PATTERN. Excluding.")
        return True
    if PREFIX_COLON_PATTERN.match(tag_lower_original_form):
        logger.debug(f"Tag '{original_tag_for_logging}' matched PREFIX_COLON_PATTERN. Excluding.")
        return True

    # Numeric check with length consideration
    if NUMERIC_LIKE_PATTERN.match(tag_lower_original_form) and len(tag_lower_original_form) < 6:
        logger.debug(f"Tag '{original_tag_for_logging}' matched NUMERIC_LIKE_PATTERN and length < 6. Excluding.")
        return True

    if '/' in tag_str and not any(kw in tag_lower_original_form for kw in ["to-", "-to-", "2", "classification", "detection", "generation", "answering"]):
        parts = tag_str.split('/')
        if len(parts) == 2:
            author_part_normalized = parts[0].lower().strip().replace(" ", "-").replace("_", "-").replace(":", "-")
            if author_part_normalized in NORMALIZED_EXCLUDED_KEYWORDS:
                 logger.debug(f"Excluding '{original_tag_for_logging}' due to author part '{parts[0]}' (normalized: '{author_part_normalized}') being in excluded keywords.")
                 return True
    
    logger.debug(f"Tag '{original_tag_for_logging}' not excluded by any rule.")
    return False


# --- Synchronization Logic ---


async def sync_hf_models(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """Synchronizes HF models from PostgreSQL to Neo4j."""
    logger.info("Starting HFModel synchronization...")
    models_synced = 0
    query = "SELECT hf_model_id, hf_author, hf_sha, hf_last_modified, hf_downloads, hf_likes, hf_tags, hf_pipeline_tag, hf_library_name, hf_readme_content, hf_dataset_links, hf_readme_tasks FROM hf_models ORDER BY hf_model_id"
    models_to_process: List[Dict[str, Any]] = []

    try:
        # Corrected call: Use keyword argument for batch_size
        async for model_record in pg_repo.fetch_data_cursor(
            query, batch_size=batch_size
        ):
            model_data = dict(model_record)
            model_id_for_logs = model_data.get('hf_model_id') # For logging

            all_potential_task_strings: Set[str] = set()

            # 1. Process hf_tags
            raw_hf_tags = model_data.get("hf_tags")
            if isinstance(raw_hf_tags, str):
                try:
                    raw_hf_tags = json.loads(raw_hf_tags)
                except json.JSONDecodeError:
                    logger.warning(f"Model {model_id_for_logs}: hf_tags is a string but not valid JSON: '{str(raw_hf_tags)[:200]}'.")
                    if not (raw_hf_tags.startswith('[') and raw_hf_tags.endswith(']')):
                        raw_hf_tags = [raw_hf_tags] 
                    else:
                        raw_hf_tags = [] 
            
            if isinstance(raw_hf_tags, list):
                for item_from_raw_list in raw_hf_tags:
                    if isinstance(item_from_raw_list, str) and item_from_raw_list.startswith('[') and item_from_raw_list.endswith(']'):
                        try:
                            inner_list = json.loads(item_from_raw_list)
                            if isinstance(inner_list, list):
                                for tag_in_inner_list in inner_list:
                                    if isinstance(tag_in_inner_list, str):
                                        all_potential_task_strings.add(tag_in_inner_list)
                                    else:
                                        logger.debug(f"Model {model_id_for_logs}: Non-string item '{str(tag_in_inner_list)[:100]}' in nested hf_tags list skipped.")
                            else: # Parsed, but not a list
                                if isinstance(item_from_raw_list, str): all_potential_task_strings.add(item_from_raw_list)
                        except json.JSONDecodeError:
                            if isinstance(item_from_raw_list, str): all_potential_task_strings.add(item_from_raw_list) 
                    elif isinstance(item_from_raw_list, str):
                        all_potential_task_strings.add(item_from_raw_list)
                    else:
                        logger.debug(f"Model {model_id_for_logs}: Non-string item '{str(item_from_raw_list)[:100]}' in hf_tags list skipped.")
            
            # 2. Process hf_readme_tasks
            raw_readme_tasks = model_data.get("hf_readme_tasks")
            parsed_readme_tasks_list: List[str] = []
            if isinstance(raw_readme_tasks, str):
                try:
                    content = json.loads(raw_readme_tasks)
                    if isinstance(content, list):
                        parsed_readme_tasks_list = [str(item) for item in content if isinstance(item, str)]
                except json.JSONDecodeError:
                    logger.warning(f"Model {model_id_for_logs}: Could not decode hf_readme_tasks JSON: {raw_readme_tasks[:200]}")
            elif isinstance(raw_readme_tasks, list):
                parsed_readme_tasks_list = [str(item) for item in raw_readme_tasks if isinstance(item, str)]
            
            for task_name in parsed_readme_tasks_list:
                all_potential_task_strings.add(task_name)

            # 3. Process hf_pipeline_tag
            pipeline_tag_value = model_data.get("hf_pipeline_tag")
            if isinstance(pipeline_tag_value, str) and pipeline_tag_value.strip():
                all_potential_task_strings.add(pipeline_tag_value)
            
            logger.debug(f"Model {model_id_for_logs}: All potential task strings collected before filtering: {all_potential_task_strings}")

            # 4. Unified Filtering
            final_task_names_for_neo4j: Set[str] = set()
            for task_candidate in all_potential_task_strings:
                logger.debug(f"Model {model_id_for_logs}: PRE-CHECK (unified filter) for task_candidate: Type={type(task_candidate)}, Value='{str(task_candidate)[:200]}'")
                if not is_likely_not_a_task(task_candidate):
                    final_task_names_for_neo4j.add(task_candidate)
                else:
                    logger.debug(f"Model {model_id_for_logs}: Filtered out (unified filter) '{task_candidate}' by is_likely_not_a_task_POST_CHECK.")
            
            logger.debug(f"Model {model_id_for_logs}: Final task names for Neo4j after unified filtering: {final_task_names_for_neo4j}")

            # Prepare model_data for Neo4j, using original and newly filtered task data as appropriate
            model_data_for_neo4j = {
                "model_id": model_data.get("hf_model_id"),
                "author": model_data.get("hf_author"),
                "sha": model_data.get("hf_sha"),
                "last_modified": model_data.get("hf_last_modified"),
                "tags": model_data.get("hf_tags") or [], # Store original hf_tags, not the filtered ones here. Filtering is for relationships.
                "pipeline_tag": model_data.get("hf_pipeline_tag"), # Store original pipeline_tag
                "downloads": model_data.get("hf_downloads"),
                "likes": model_data.get("hf_likes"),
                "library_name": model_data.get("hf_library_name"),
                "hf_readme_content": model_data.get("hf_readme_content"),
                "hf_dataset_links": json.loads(model_data["hf_dataset_links"])
                                    if isinstance(model_data.get("hf_dataset_links"), str)
                                    else (model_data.get("hf_dataset_links") or []),
                "hf_readme_tasks_for_relation": list(final_task_names_for_neo4j), # THIS IS THE KEY CHANGE FOR RELATIONSHIPS
                "task_names_for_neo4j": list(final_task_names_for_neo4j) # This is what save_hf_models_batch expects
            }

            model_data_for_neo4j_cleaned = {k: v for k, v in model_data_for_neo4j.items() if v is not None}
            models_to_process.append(model_data_for_neo4j_cleaned)

            if len(models_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.save_hf_models_batch(models_to_process)
                    models_synced += len(models_to_process)
                    logger.info(f"Synced {models_synced} HF models so far...")
                except Exception as e:
                    logger.error(f"Error saving HF model batch to Neo4j: {e}")
                    # No need to import traceback here if already imported globally
                    logger.error(traceback.format_exc())
                finally:
                    models_to_process = []
    except Exception as e:
        logger.error(f"Error fetching HF models from Postgres: {e}")
        # No need to import traceback here
        logger.error(traceback.format_exc())

    # Sync any remaining models
    if models_to_process:
        try:
            await neo4j_repo.save_hf_models_batch(models_to_process)
            models_synced += len(models_to_process)
        except Exception as e:
            logger.error(f"Error saving final HF model batch to Neo4j: {e}")
            # No need to import traceback here
            logger.error(traceback.format_exc())

    logger.info(
        f"HFModel synchronization finished. Total models synced: {models_synced}"
    )


async def sync_papers_and_relations(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> int:
    """Fetches papers and their relations from PG and syncs them to Neo4j.

    Returns:
        int: 同步的论文总数
    """
    logger.info("Starting Paper and relations synchronization...")
    papers_synced_arxiv = 0
    papers_synced_pwc = 0

    logger.info("Fetching and syncing Paper nodes...")
    paper_query = """
        SELECT
            p.paper_id, p.pwc_id, p.arxiv_id_base, p.arxiv_id_versioned, p.title,
            p.authors, p.summary, p.published_date, p.area, p.pwc_url,
            p.pdf_url, p.doi, p.primary_category, p.categories,
            p.conference
        FROM papers p
    """
    papers_to_process: List[Dict[str, Any]] = []
    arxiv_only_papers: List[
        Dict[str, Any]
    ] = []  # For papers only identified by arxiv_id
    try:
        # Corrected call: Use keyword argument for batch_size
        async for paper_record in pg_repo.fetch_data_cursor(
            paper_query, batch_size=batch_size
        ):
            paper_data = dict(paper_record)

            # Convert JSON string fields back to lists if necessary (depends on PG repo handling)
            if isinstance(paper_data.get("authors"), str):
                try:
                    paper_data["authors"] = json.loads(paper_data["authors"])
                except json.JSONDecodeError:
                    paper_data["authors"] = []
            if isinstance(paper_data.get("categories"), str):
                try:
                    paper_data["categories"] = json.loads(paper_data["categories"])
                except json.JSONDecodeError:
                    paper_data["categories"] = []
            if paper_data.get("published_date"):
                paper_data["published_date"] = paper_data["published_date"].isoformat()
            
            # Ensure conference is included (it's fetched by the query now)
            # paper_data["conference"] will be set from dict(paper_record)

            # Initialize relation keys - enrichment happens later for pwc_id papers
            paper_data["tasks"] = []
            paper_data["datasets"] = []
            paper_data["repositories"] = []

            # Sorting papers by identifier availability
            if paper_data.get("pwc_id"):
                # If paper has a PWC ID, enrich with additional relations
                # and save using the pwc_id-based Neo4j methods
                papers_to_process.append(paper_data)
            elif paper_data.get("arxiv_id_base"):
                # If paper has only an ArXiv ID, we'll save using a different method
                arxiv_only_papers.append(paper_data)
            else:
                logger.warning(
                    f"Paper id={paper_data.get('paper_id')} has neither pwc_id nor arxiv_id. Skipping."
                )
                continue

            # Batch processing - PWC ID papers
            if len(papers_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    enriched_papers = await enrich_papers_with_relations(
                        pg_repo, papers_to_process
                    )
                    await neo4j_repo.save_papers_batch(enriched_papers)
                    papers_synced_pwc += len(enriched_papers)
                    logger.info(
                        f"Synced {papers_synced_pwc} PWC papers and {papers_synced_arxiv} ArXiv-only papers..."
                    )
                except Exception as e:
                    logger.error(f"Error saving paper batch to Neo4j: {e}")
                    # No need to import traceback here
                    logger.error(traceback.format_exc())
                finally:
                    papers_to_process = []

            # Batch processing - ArXiv-only papers
            if len(arxiv_only_papers) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.save_papers_by_arxiv_batch(arxiv_only_papers)
                    papers_synced_arxiv += len(arxiv_only_papers)
                    logger.info(
                        f"Synced {papers_synced_pwc} PWC papers and {papers_synced_arxiv} ArXiv-only papers..."
                    )
                except Exception as e:
                    logger.error(f"Error saving arxiv paper batch to Neo4j: {e}")
                    # No need to import traceback here
                    logger.error(traceback.format_exc())
                finally:
                    arxiv_only_papers = []

    except Exception as e:
        logger.error(f"Error fetching papers from Postgres: {e}")
        # No need to import traceback here
        logger.error(traceback.format_exc())

    # Process any remaining papers
    if papers_to_process:
        try:
            enriched_papers = await enrich_papers_with_relations(
                pg_repo, papers_to_process
            )
            await neo4j_repo.save_papers_batch(enriched_papers)
            papers_synced_pwc += len(enriched_papers)
        except Exception as e:
            logger.error(f"Error saving final paper batch to Neo4j: {e}")
            # No need to import traceback here
            logger.error(traceback.format_exc())

    if arxiv_only_papers:
        try:
            await neo4j_repo.save_papers_by_arxiv_batch(arxiv_only_papers)
            papers_synced_arxiv += len(arxiv_only_papers)
        except Exception as e:
            logger.error(f"Error saving final arxiv paper batch to Neo4j: {e}")
            # No need to import traceback here
            logger.error(traceback.format_exc())

    total_papers = papers_synced_pwc + papers_synced_arxiv
    logger.info(
        f"Paper synchronization finished. PWC papers: {papers_synced_pwc}, ArXiv-only papers: {papers_synced_arxiv}, Total: {total_papers}"
    )

    return total_papers


# --- START: CORRECTED enrich_papers_with_relations --- #
async def enrich_papers_with_relations(
    pg_repo: PostgresRepository, paper_batch: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fetches related Tasks, Datasets, Methods, and Repositories for a batch of papers
    and adds them to the corresponding paper dictionaries.
    """
    if not paper_batch:
        return []

    paper_map = {
        int(p["paper_id"]): p for p in paper_batch if p.get("paper_id") is not None
    }
    paper_ids = list(paper_map.keys())

    if not paper_ids:
        logger.warning("Enrichment called with batch containing no valid paper_ids.")
        return paper_batch

    # Ensure relation keys exist (even if enrichment fails)
    for paper_data in paper_map.values():
        paper_data.setdefault("tasks", [])
        paper_data.setdefault("datasets", [])
        paper_data.setdefault("methods", [])  # Add default for methods
        paper_data.setdefault("repositories", [])

    # --- Fetch Tasks using the CORRECT repository method --- #
    try:
        tasks_map = await pg_repo.get_tasks_for_papers(paper_ids)
        for paper_id, tasks_list in tasks_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["tasks"] = tasks_list
    except Exception as e:
        logger.error(
            f"Error fetching tasks relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    # --- Fetch Datasets using the CORRECT repository method --- #
    try:
        datasets_map = await pg_repo.get_datasets_for_papers(paper_ids)
        for paper_id, datasets_list in datasets_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["datasets"] = datasets_list
    except Exception as e:
        logger.error(
            f"Error fetching datasets relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    # --- Fetch Methods using the CORRECT repository method --- #
    # Assuming a method pg_repo.get_methods_for_papers exists
    try:
        methods_map = await pg_repo.get_methods_for_papers(paper_ids)  # Call the method
        for paper_id, methods_list in methods_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["methods"] = methods_list  # Assign fetched methods
    except AttributeError:
        logger.error(
            f"PostgresRepository does not have a 'get_methods_for_papers' method. Methods cannot be enriched."
        )
    except Exception as e:
        logger.error(
            f"Error fetching methods relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    # --- Fetch Repositories using the CORRECT repository method --- #
    try:
        repos_map = await pg_repo.get_repositories_for_papers(paper_ids)
        for paper_id, repo_list_of_dicts in repos_map.items(): # Expecting a list of dicts
            if paper_id in paper_map:
                # Directly assign the list of repository dictionaries
                # Each dict should contain: url, stars, is_official, framework, license, language
                paper_map[paper_id]["repositories"] = repo_list_of_dicts
    except Exception as e:
        logger.error(
            f"Error fetching repository relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    if paper_map:
        first_paper_key = next(iter(paper_map))
        logger.debug(
            f"[Enrich] Returning enriched batch. Example paper ID {first_paper_key} tasks: {paper_map[first_paper_key].get('tasks')}, methods: {paper_map[first_paper_key].get('methods')}"  # Log methods too
        )
    else:
        logger.debug("[Enrich] Returning empty batch.")

    return list(paper_map.values())


# --- End of CORRECTED enrich_papers_with_relations --- #


async def sync_model_paper_links(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """同步HFModel和Paper之间的关系"""
    logger.info("开始同步HF模型和论文之间的关系...")

    # 从model_paper_links表获取关系
    link_query = """
    SELECT mpl.hf_model_id, p.pwc_id, mpl.paper_id
    FROM model_paper_links mpl
    JOIN papers p ON mpl.paper_id = p.paper_id
    WHERE p.pwc_id IS NOT NULL
    """

    try:
        # 获取数据
        links_to_process = []
        link_count = 0

        async for link_record in pg_repo.fetch_data_cursor(
            link_query, batch_size=batch_size
        ):
            # 只处理有效记录
            if not link_record.get("hf_model_id") or not link_record.get("pwc_id"):
                continue

            # 转换格式
            link_data = {
                "model_id": link_record["hf_model_id"],
                "pwc_id": link_record["pwc_id"],
                "confidence": 1.0,  # 默认置信度
            }
            links_to_process.append(link_data)

            # 批处理
            if len(links_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    # 特殊调试输出
                    logger.info(
                        f"正在处理 {len(links_to_process)} 条HFModel-Paper关系，第一条: {links_to_process[0]}"
                    )

                    # 创建MENTIONS关系
                    await neo4j_repo.link_model_to_paper_batch(links_to_process)
                    link_count += len(links_to_process)
                    logger.info(f"已同步 {link_count} 条模型-论文关系...")
                except Exception as e:
                    logger.error(f"保存模型-论文关系批次时出错: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    links_to_process = []

        # 处理剩余的关系
        if links_to_process:
            try:
                # 特殊调试输出
                logger.info(f"正在处理最后 {len(links_to_process)} 条HFModel-Paper关系")

                # 创建MENTIONS关系
                await neo4j_repo.link_model_to_paper_batch(links_to_process)
                link_count += len(links_to_process)
            except Exception as e:
                logger.error(f"保存最后一批模型-论文关系时出错: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"模型-论文关系同步完成，总计: {link_count} 条关系")

    except Exception as e:
        logger.error(f"获取模型-论文关系时出错: {e}")
        logger.error(traceback.format_exc())


async def sync_model_derivations(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """同步HFModel之间的DERIVED_FROM关系"""
    logger.info("开始同步模型之间的派生关系 (DERIVED_FROM)...")

    # 从hf_models表获取hf_model_id和hf_base_models
    # 确保在 load_postgres.py 中使用的列名是 hf_base_models
    query = """
    SELECT hf_model_id, hf_base_models 
    FROM hf_models 
    WHERE hf_base_models IS NOT NULL AND hf_base_models <> 'null' AND hf_base_models <> '[]'
    """

    try:
        links_to_process: List[Dict[str, str]] = []
        link_count = 0

        # Corrected call: Use keyword argument for batch_size
        async for record in pg_repo.fetch_data_cursor(
            query, batch_size=batch_size
        ):
            current_model_id = record.get("hf_model_id")
            base_models_data = record.get("hf_base_models") # This is likely a JSON string or already parsed by psycopg

            if not current_model_id or not base_models_data:
                continue

            parsed_base_models: Optional[Union[str, List[str]]] = None
            if isinstance(base_models_data, str): # If it's a JSON string from DB
                try:
                    parsed_base_models = json.loads(base_models_data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        f"无法解析模型 {current_model_id} 的 hf_base_models JSON: {base_models_data}"
                    )
                    continue # Skip this record if parsing fails
            elif isinstance(base_models_data, (list, dict)): # If psycopg already parsed it
                 # Assuming if it's a dict, it's an error or unexpected format for base_models list/str
                 if isinstance(base_models_data, dict):
                    logger.warning(f"模型 {current_model_id} 的 hf_base_models 格式错误 (应为列表或字符串): {base_models_data}")
                    continue
                 parsed_base_models = base_models_data
            else:
                logger.warning(f"模型 {current_model_id} 的 hf_base_models 类型未知: {type(base_models_data)}")
                continue

            if not parsed_base_models:
                continue

            if isinstance(parsed_base_models, str):
                # Single base model ID
                links_to_process.append(
                    {"source_model_id": current_model_id, "base_model_id": parsed_base_models}
                )
            elif isinstance(parsed_base_models, list):
                # List of base model IDs
                for base_id in parsed_base_models:
                    if isinstance(base_id, str) and base_id.strip(): # Ensure it's a non-empty string
                        links_to_process.append(
                            {"source_model_id": current_model_id, "base_model_id": base_id.strip()}
                        )
                    else:
                        logger.debug(f"模型 {current_model_id} 的 hf_base_models 列表中包含无效条目: {base_id}")
            else:
                logger.warning(f"模型 {current_model_id} 解析后的 hf_base_models 类型无效: {type(parsed_base_models)}")

            # Batch processing
            if len(links_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.link_models_derived_from_batch(links_to_process)
                    link_count += len(links_to_process)
                    logger.info(f"已同步 {link_count} 条模型派生关系...")
                except Exception as e:
                    logger.error(f"保存模型派生关系批次时出错: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    links_to_process = []

        # Process any remaining links
        if links_to_process:
            try:
                await neo4j_repo.link_models_derived_from_batch(links_to_process)
                link_count += len(links_to_process)
            except Exception as e:
                logger.error(f"保存最后一批模型派生关系时出错: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"模型派生关系同步完成，总计: {link_count} 条关系")

    except Exception as e:
        logger.error(f"获取模型派生关系时出错: {e}")
        logger.error(traceback.format_exc())


# --- Synchronization Runner ---
async def run_sync(
    pg_repo: Optional[PostgresRepository] = None,
    neo4j_repo: Optional[Neo4jRepository] = None,
) -> int:
    """Runs the full synchronization process.

    Returns:
        int: 同步的论文数量，用于测试断言
    """
    logger.info("Starting full PG to Neo4j synchronization...")

    # Create repositories if not provided
    created_pg_repo = False
    created_neo4j_repo = False

    if pg_repo is None:
        try:
            # Create PostgreSQL pool
            if DATABASE_URL is None:
                logger.error("DATABASE_URL is None")
                return 0
            pg_pool = AsyncConnectionPool(conninfo=DATABASE_URL, open=True)
            pg_repo = PostgresRepository(pool=pg_pool)
            created_pg_repo = True
        except Exception as e:
            logger.error(f"Failed to initialize Postgres repository: {e}")
            logger.error(traceback.format_exc())
            return 0

    if neo4j_repo is None:
        try:
            # Create Neo4j driver
            if NEO4J_URI is None or NEO4J_USER is None or NEO4J_PASSWORD is None:
                logger.error("Neo4j connection parameters are None")
                return 0
            neo4j_driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            neo4j_repo = Neo4jRepository(driver=neo4j_driver)
            created_neo4j_repo = True
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j repository: {e}")
            logger.error(traceback.format_exc())
            # Clean up Postgres connection if we created it
            if created_pg_repo and pg_repo is not None:
                await pg_repo.close()
            return 0

    try:
        # 确保repositories不为None
        if pg_repo is None or neo4j_repo is None:
            logger.error("Repository objects are None after initialization")
            return 0

        # 1. Make sure Neo4j constraints exist - Changed to create_constraints_and_indexes
        await neo4j_repo.create_constraints_and_indexes()

        # 2. Sync HF Models
        await sync_hf_models(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        # 3. Sync Papers (with their relations)
        papers_count = await sync_papers_and_relations(
            pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE
        )

        # 4. Sync Model<->Paper links
        await sync_model_paper_links(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        # 5. Sync Model<->Model (DERIVED_FROM) links
        await sync_model_derivations(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        logger.info("Full synchronization completed successfully.")

        # 6. Count papers in Neo4j (可选的验证步骤)
        neo4j_papers = await neo4j_repo.count_paper_nodes()
        logger.info(f"Current Neo4j paper count: {neo4j_papers}")

        return papers_count  # 返回实际同步的论文数量
    except Exception as e:
        logger.error(f"Synchronization failed with error: {e}")
        logger.error(traceback.format_exc())
        return 0
    finally:
        # Clean up resources if we created them
        if created_pg_repo and pg_repo is not None:
            await pg_repo.close()
        if created_neo4j_repo and neo4j_repo is not None:
            # Neo4j driver cleaned up through repository close method
            pass


# --- Main Execution ---
async def main(reset_neo4j: bool) -> None:
    """Main function to run the synchronization process."""
    pg_pool = None
    neo4j_driver = None
    pg_repo = None  # Initialize repo variable
    neo4j_repo = None  # Initialize repo variable
    total_papers_synced = 0

    try:
        # --- Initialize Connections ---
        logger.info(f"Initializing PostgreSQL pool for {DATABASE_URL}...")
        # Add assertion for DATABASE_URL
        assert DATABASE_URL is not None, "DATABASE_URL environment variable must be set"
        # Explicitly create the pool instance
        pg_pool = AsyncConnectionPool(conninfo=DATABASE_URL, open=True)
        logger.info("PostgreSQL pool initialized.")

        logger.info(f"Initializing Neo4j driver for {NEO4J_URI}...")
        # Add assertions for Neo4j connection details
        assert NEO4J_URI is not None, "NEO4J_URI environment variable must be set"
        assert NEO4J_USER is not None, "NEO4J_USER environment variable must be set"
        assert NEO4J_PASSWORD is not None, (
            "NEO4J_PASSWORD environment variable must be set"
        )
        # Explicitly create the driver instance
        neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        logger.info("Neo4j driver initialized.")

        # --- Create Repository Instances with initialized pool/driver ---
        # Pass the initialized pool and driver instances
        pg_repo = PostgresRepository(pool=pg_pool)
        neo4j_repo = Neo4jRepository(driver=neo4j_driver)
        logger.info("Repositories initialized.")

        # --- Optional: Reset Neo4j if requested ---
        if reset_neo4j:
            logger.warning("Resetting Neo4j database...")
            await neo4j_repo.reset_database()  # Assuming this method exists
            logger.info("Neo4j database reset complete.")
        else:
            logger.info("Skipping Neo4j database reset.")

        # --- Run Synchronization ---
        logger.info("--- Starting synchronization from PostgreSQL to Neo4j ---")
        total_papers_synced = await run_sync(pg_repo=pg_repo, neo4j_repo=neo4j_repo)

    except asyncpg.exceptions.CannotConnectNowError as pg_conn_err:
        logger.critical(
            f"FATAL: Could not connect to PostgreSQL at {DATABASE_URL}. Check if DB is running and accessible. Error: {pg_conn_err}"
        )
    except ConnectionRefusedError as neo4j_conn_err:
        logger.critical(
            f"FATAL: Could not connect to Neo4j at {NEO4J_URI}. Check if DB is running and accessible. Error: {neo4j_conn_err}"
        )
    except Exception as e:
        logger.critical(f"An unexpected error occurred during synchronization: {e}")
        # No need to import traceback here
        logger.critical(traceback.format_exc())
    finally:
        # --- Close Connections ---
        if pg_pool:
            logger.info("Closing PostgreSQL pool...")
            await pg_pool.close()
        if neo4j_driver:
            logger.info("Closing Neo4j driver...")
            await neo4j_driver.close()
        logger.info("Connections closed (or closing attempted).")
        # Log final paper count if sync was attempted
        if total_papers_synced > 0:
            logger.info(
                f"Final count of papers synced in this run: {total_papers_synced}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize data from PostgreSQL to Neo4j for AIGraphX."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the entire Neo4j database before starting synchronization.",
    )
    args = parser.parse_args()

    # Run the main asynchronous function
    try:
        asyncio.run(main(reset_neo4j=args.reset))
        logger.info("Script finished successfully.")
    except Exception as e:
        # Catch errors happening during asyncio.run() itself if any
        logger.critical(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)
