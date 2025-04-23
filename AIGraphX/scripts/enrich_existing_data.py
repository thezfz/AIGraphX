#!/usr/bin/env python
import asyncio
import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set, TypedDict, Union
from datetime import datetime, timezone
import sys
import re
import argparse
from functools import partial  # For asyncio.to_thread with kwargs

# --- Library Imports (Copy relevant ones from collect_data.py) ---
from dotenv import load_dotenv
import httpx
import tenacity
from aiolimiter import AsyncLimiter

# Import necessary HF Hub functions and errors
from huggingface_hub import HfApi, hf_hub_download, ModelInfo
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# Adjust import path for helper functions if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration (Similar to collect_data.py) ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")

DEFAULT_INPUT_JSONL = "data/aigraphx_knowledge_data_v1.jsonl"
DEFAULT_OUTPUT_JSONL = "data/aigraphx_knowledge_data_enriched.jsonl"
CHECKPOINT_FILE_ENRICH = "data/enrich_checkpoint.txt"  # Use a separate checkpoint file
CHECKPOINT_INTERVAL_ENRICH = 100

LOG_FILE_ENRICH = "logs/enrich_data.log"
LOG_DIR_ENRICH = os.path.dirname(LOG_FILE_ENRICH)

# API Endpoints
PWC_BASE_URL = "https://paperswithcode.com/api/v1/"
GITHUB_API_BASE_URL = "https://api.github.com/"

# Concurrency and Rate Limiting (Adjust if needed)
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent enrichment operations per record
hf_limiter = AsyncLimiter(5, 1.0)
pwc_limiter = AsyncLimiter(2, 1.0)
github_limiter = AsyncLimiter(1, 1.0)

# --- Logging Setup (Similar to collect_data.py, use different log file) ---
os.makedirs(LOG_DIR_ENRICH, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger("enrich_script")
# Remove default handlers if any
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
stream_handler_enrich = logging.StreamHandler()
stream_handler_enrich.setLevel(logging.INFO)
stream_formatter_enrich = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
stream_handler_enrich.setFormatter(stream_formatter_enrich)
logger.addHandler(stream_handler_enrich)
try:
    file_handler_enrich = logging.FileHandler(
        LOG_FILE_ENRICH, mode="a", encoding="utf-8"
    )
    file_handler_enrich.setLevel(logging.DEBUG)
    file_formatter_enrich = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] [%(funcName)s] %(message)s"
    )
    file_handler_enrich.setFormatter(file_formatter_enrich)
    logger.addHandler(file_handler_enrich)
except Exception as e:
    logger.error(f"Failed to set up file logging to {LOG_FILE_ENRICH}: {e}")

logger.info("Enrichment script logging configured.")

# --- Tenacity Retry Config (Copy from collect_data.py) ---
RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
retry_config_http = dict(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20)
    + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# --- API Client Initialization (Shared Clients) ---
# Use a shared httpx.AsyncClient for PWC and GitHub
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(15.0, read=60.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    http2=True,
)
# Shared GitHub headers
github_headers = {}
if GITHUB_TOKEN:
    github_headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
else:
    github_headers = {"Accept": "application/vnd.github.v3+json"}


# --- Helper Functions (Reuse/Adapt from collect_data.py) ---


# Reuse fetch_pwc_relation_list (slightly adapted logging maybe)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20)
    + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_pwc_relation_list_enrich(
    pwc_paper_id: str, relation: str
) -> List[Dict[str, Any]]:
    # Function body copied and adapted from collect_data.py's fetch_pwc_relation_list
    # Ensure logging uses the enrich_script logger
    all_results = []
    current_url: Optional[str] = f"{PWC_BASE_URL}papers/{pwc_paper_id}/{relation}/"
    page_num = 1
    while current_url:
        async with pwc_limiter:
            logger.debug(
                f"[Enrich] Fetching PWC {relation} for {pwc_paper_id} from {current_url}"
            )
            try:
                response = await http_client.get(current_url)
                response.raise_for_status()
                data = response.json()
                results = []
                next_url: Optional[str] = None
                if isinstance(data, list):
                    results = data
                elif (
                    isinstance(data, dict)
                    and "results" in data
                    and isinstance(data["results"], list)
                ):
                    results = data["results"]
                    next_url = data.get("next")
                    if next_url == current_url:
                        next_url = None  # Prevent loop
                else:
                    logger.warning(
                        f"[Enrich] Unexpected data format for PWC {relation} for {pwc_paper_id}. Got: {type(data)}"
                    )
                    break
                all_results.extend(results)
                current_url = next_url
                if current_url:
                    page_num += 1
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"[Enrich] HTTP error fetching PWC {relation} for {pwc_paper_id}: {e.response.status_code}"
                )
                raise e
            except Exception as e:
                logger.warning(
                    f"[Enrich] Error fetching/parsing PWC {relation} for {pwc_paper_id}: {e}"
                )
                raise e
    logger.debug(
        f"[Enrich] Finished fetching PWC {relation} for {pwc_paper_id}. Found {len(all_results)} results."
    )
    return all_results


# Reuse fetch_github_details (ensure logging uses enrich_script logger)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20)
    + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_github_details_enrich(
    repo_url: str, follow_redirects: bool = True, max_redirects: int = 3
) -> Optional[Dict[str, Any]]:
    # Function body copied and adapted from collect_data.py's fetch_github_details
    # Ensure logging uses the enrich_script logger
    if not GITHUB_TOKEN:
        return None
    if not repo_url:
        return None
    current_api_url_to_fetch = None
    original_url = repo_url
    redirect_count = 0
    # URL Parsing logic... (same as in collect_data.py)
    try:
        # ... (URL parsing logic from collect_data.py) ...
        if "github.com" in repo_url.lower():
            clean_url = (
                repo_url.replace("https://", "").replace("http://", "").strip("/")
            )
            if clean_url.endswith(".git"):
                clean_url = clean_url[:-4]
            parts = clean_url.split("/")
            if len(parts) >= 3 and parts[0].lower() == "github.com":
                owner, repo_name = parts[1], parts[2]
                if owner and repo_name:
                    current_api_url_to_fetch = (
                        f"{GITHUB_API_BASE_URL}repos/{owner}/{repo_name}"
                    )
                else:
                    return None
            else:
                return None  # Not github.com/owner/repo
        elif "api.github.com/repositories/" in repo_url.lower():
            current_api_url_to_fetch = repo_url
        else:
            return None  # Not a github URL
    except Exception as parse_error:
        logger.warning(
            f"[Enrich] Error parsing initial GitHub URL {repo_url}: {parse_error}"
        )
        return None

    while current_api_url_to_fetch and redirect_count <= max_redirects:
        async with github_limiter:
            logger.debug(
                f"[Enrich] Fetching GitHub details from {current_api_url_to_fetch}"
            )
            try:
                response = await http_client.get(
                    current_api_url_to_fetch,
                    headers=github_headers,
                    follow_redirects=False,
                )
                if response.status_code == 200:
                    data = response.json()
                    stars = data.get("stargazers_count")
                    language = data.get("language")
                    license_info = data.get("license")
                    license_name = None
                    if license_info and isinstance(license_info, dict):
                        license_name = license_info.get("spdx_id")
                    logger.debug(
                        f"[Enrich] Fetched details for {data.get('full_name')}: S={stars}, L={language}, Lic={license_name}"
                    )
                    return {
                        "stars": stars,
                        "language": language,
                        "license": license_name,
                    }
                # Redirect and error handling... (same as collect_data.py, use logger)
                elif (
                    follow_redirects
                    and response.status_code in (301, 302, 307, 308)
                    and "location" in response.headers
                ):
                    redirect_count += 1
                    current_api_url_to_fetch = response.headers["location"]
                    continue
                elif response.status_code == 404:
                    logger.info(
                        f"[Enrich] GitHub API 404 for {current_api_url_to_fetch}"
                    )
                    return None
                elif response.status_code == 403:
                    logger.error(
                        f"[Enrich] GitHub API 403 for {current_api_url_to_fetch}. Rate limit/perms? Check token."
                    )
                    return None
                elif response.status_code == 401:
                    logger.error(
                        f"[Enrich] GitHub API 401 for {current_api_url_to_fetch}. Check GITHUB_API_KEY."
                    )
                    return None
                else:
                    response.raise_for_status()  # Let tenacity handle other retryable errors
            except Exception as e:
                logger.error(
                    f"[Enrich] Error fetching/parsing GitHub details from {current_api_url_to_fetch}: {e}"
                )
                raise e
    return None


# Reuse fetch_arxiv_metadata if needed for conference (or adapt PWC fetch)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20)
    + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def find_pwc_entry_by_arxiv_id_enrich(arxiv_id: str) -> Optional[Dict[str, Any]]:
    # Function body copied and adapted from collect_data.py's find_pwc_entry_by_arxiv_id
    # Ensure logging uses the enrich_script logger
    arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)
    async with pwc_limiter:
        url = f"{PWC_BASE_URL}papers/"
        params = {"arxiv_id": arxiv_id_base}
        logger.debug(f"[Enrich] Querying PWC for ArXiv ID {arxiv_id_base}")
        try:
            response = await http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            count = data.get("count")
            if isinstance(count, int) and count == 1 and data.get("results"):
                result_entry = data["results"][0]
                if isinstance(result_entry, dict):
                    return result_entry  # Return the summary entry which might contain conference
                else:
                    return None
            else:
                return None  # Not found or multiple entries
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None  # Not found is expected sometimes
            logger.warning(
                f"[Enrich] HTTP error querying PWC for {arxiv_id_base}: {e.response.status_code}"
            )
            raise e
        except Exception as e:
            logger.error(f"[Enrich] Error querying PWC for {arxiv_id_base}: {e}")
            raise e


# Function to fetch README content
async def fetch_readme_content_enrich(model_id: str) -> Optional[str]:
    logger.debug(f"[Enrich] Attempting to fetch README for {model_id}")
    try:
        readme_path = await asyncio.to_thread(
            partial(  # Use partial to pass kwargs to hf_hub_download in thread
                hf_hub_download,
                repo_id=model_id,
                filename="README.md",
                token=HF_API_TOKEN,
                repo_type="model",
                cache_dir=None,  # Consider using cache
            )
        )
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"[Enrich] Successfully fetched README for {model_id}")
        return content
    except HfHubHTTPError as e:
        status_code = e.response.status_code if hasattr(e, "response") else 0
        if status_code == 404:
            logger.info(f"[Enrich] README.md not found for {model_id}.")
        else:
            logger.warning(
                f"[Enrich] HTTP error {status_code} fetching README for {model_id}: {e}"
            )
        return None
    except Exception as e:
        logger.warning(f"[Enrich] Error fetching/reading README for {model_id}: {e}")
        return None


# --- Checkpointing for Enrichment ---
def _save_checkpoint_enrich(line_count: int) -> None:
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_ENRICH), exist_ok=True)
        with open(CHECKPOINT_FILE_ENRICH, "w") as f:
            f.write(str(line_count))
        logger.debug(f"[Enrich] Checkpoint saved: {line_count} lines processed.")
    except IOError as e:
        logger.error(f"Failed to save enrichment checkpoint: {e}")


def _load_checkpoint_enrich(reset_checkpoint: bool = False) -> int:
    if reset_checkpoint and os.path.exists(CHECKPOINT_FILE_ENRICH):
        try:
            os.remove(CHECKPOINT_FILE_ENRICH)
            logger.info("[Enrich] Removed existing checkpoint.")
        except OSError as e:
            logger.error(f"Failed to remove checkpoint: {e}")
    if not os.path.exists(CHECKPOINT_FILE_ENRICH):
        return 0
    try:
        with open(CHECKPOINT_FILE_ENRICH, "r") as f:
            processed_count = int(f.read().strip())
        logger.info(
            f"[Enrich] Checkpoint loaded: Resuming after {processed_count} lines."
        )
        return processed_count
    except (IOError, ValueError) as e:
        logger.error(f"Failed to load enrichment checkpoint: {e}. Starting from 0.")
        return 0


# --- Main Enrichment Logic ---
async def enrich_record(record: Dict[str, Any]) -> bool:
    """Performs enrichment operations on a single record dictionary. Returns True if modified."""
    modified = False
    model_id = record.get("hf_model_id")
    if not model_id:
        logger.warning("Record missing hf_model_id, cannot enrich.")
        return False

    # --- Initialize missing fields ---
    if "hf_dataset_links" not in record:
        record["hf_dataset_links"] = []
    if "linked_papers" not in record:
        record["linked_papers"] = []

    # --- Extract Dataset links from hf_tags ---
    hf_tags = record.get("hf_tags")
    if isinstance(hf_tags, list):
        initial_links_count = len(record["hf_dataset_links"])
        # Store full URLs in the set and list now
        existing_links_set = set(record["hf_dataset_links"])  # Use set for faster check

        for tag in hf_tags:
            if isinstance(tag, str) and tag.startswith("dataset:"):
                dataset_name = tag.split(":", 1)[1].strip()
                # Ensure the extracted name is not empty
                if dataset_name:
                    # Construct the full URL
                    dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
                    # Check if the URL is already present
                    if dataset_url not in existing_links_set:
                        record["hf_dataset_links"].append(dataset_url)
                        existing_links_set.add(
                            dataset_url
                        )  # Keep set updated with the URL

        if len(record["hf_dataset_links"]) > initial_links_count:
            modified = True
            logger.debug(
                f"[Enrich] Extracted dataset links from hf_tags for {model_id}: {record['hf_dataset_links']}"
            )

    # --- Stage 1: Gather top-level enrichment tasks (README) ---
    hf_enrich_tasks = []
    if record.get("hf_readme_content") is None:
        hf_enrich_tasks.append(fetch_readme_content_enrich(model_id))
    else:
        # Placeholder: If readme exists, add a coroutine returning existing content
        hf_enrich_tasks.append(
            asyncio.sleep(0, result=record.get("hf_readme_content"))
        )  # Pass existing content

    # --- Stage 2: Prepare enrichment tasks for each paper --- (Gather info needed)
    paper_enrichment_plan = []  # List of tuples: (paper_dict, task_dict)

    for paper in record.get("linked_papers", []):
        if not isinstance(paper, dict):
            continue
        pwc_entry = paper.get("pwc_entry")
        if not pwc_entry or not isinstance(pwc_entry, dict):
            continue

        pwc_id = pwc_entry.get("pwc_id")
        arxiv_id_base = paper.get("arxiv_id_base")

        tasks_for_this_paper: Dict[str, Any] = {
            "methods": None,  # Coroutine or existing data
            "conference_entry": None,  # Coroutine or None
            "repo_tasks": [],  # List of Coroutines for repo details
            "repos_to_update": [],  # List of repo dicts to update
        }
        needs_paper_gather = False

        # 3a. PWC Methods
        if pwc_id and ("methods" not in pwc_entry or not pwc_entry["methods"]):
            tasks_for_this_paper["methods"] = fetch_pwc_relation_list_enrich(
                pwc_id, "methods"
            )
            needs_paper_gather = True
        else:
            tasks_for_this_paper["methods"] = pwc_entry.get(
                "methods", []
            )  # Store existing data

        # 3b. Conference
        if pwc_id and "conference" not in pwc_entry:
            if arxiv_id_base:
                # This returns a coroutine or None
                tasks_for_this_paper["conference_entry"] = (
                    find_pwc_entry_by_arxiv_id_enrich(arxiv_id_base)
                )  # type: ignore[assignment]
                needs_paper_gather = True
            else:
                logger.warning(
                    f"[Enrich] Cannot fetch conference for PWC ID {pwc_id} as ArXiv ID is missing."
                )
                tasks_for_this_paper["conference_entry"] = None  # Store None
        else:
            tasks_for_this_paper["conference_entry"] = None  # Store None

        # 3c. Repositories
        repo_tasks = []
        repos_to_update = []
        for repo in pwc_entry.get("repositories", []):
            if not isinstance(repo, dict):
                continue
            repo_url = repo.get("url")
            needs_enrich = (
                "license" not in repo
                or repo["license"] is None
                or "language" not in repo
                or repo["language"] is None
            )

            if repo_url and "github.com" in repo_url.lower() and needs_enrich:
                repo_tasks.append(fetch_github_details_enrich(repo_url))
                repos_to_update.append(repo)
                needs_paper_gather = True
            # No need for placeholders in repo_tasks list

        tasks_for_this_paper["repo_tasks"] = repo_tasks
        tasks_for_this_paper["repos_to_update"] = repos_to_update

        if needs_paper_gather:
            paper_enrichment_plan.append((paper, tasks_for_this_paper))

    # --- Stage 3: Run all gathered tasks --- #
    # Run HF tasks
    hf_results = await asyncio.gather(*hf_enrich_tasks, return_exceptions=True)

    # Run paper enrichment tasks (grouped by paper)
    paper_results_list = await asyncio.gather(
        *[
            asyncio.gather(
                plan[1]["methods"]
                if asyncio.iscoroutine(plan[1]["methods"])
                else asyncio.sleep(0, result=plan[1]["methods"]),  # type: ignore
                plan[1]["conference_entry"]
                if asyncio.iscoroutine(plan[1]["conference_entry"])
                else asyncio.sleep(0, result=None),  # type: ignore
                asyncio.gather(
                    *plan[1]["repo_tasks"], return_exceptions=True
                ),  # Gather repo tasks for this paper
            )
            for plan in paper_enrichment_plan
        ],
        return_exceptions=True,
    )

    # --- Stage 4: Process results and update record --- #

    # Process HF results
    if hf_results:
        readme_result = hf_results[0]
        # Check if we *attempted* to fetch the readme because it was missing/None initially
        readme_was_missing_or_none = record.get("hf_readme_content") is None

        if isinstance(readme_result, str) and readme_was_missing_or_none:
            # Successfully fetched README content when it was missing
            record["hf_readme_content"] = readme_result
            modified = True
            logger.info(f"[Enrich] Successfully updated README for {model_id}")
        elif readme_was_missing_or_none and not isinstance(readme_result, str):
            # Attempted to fetch missing README, but failed (returned None or Exception)
            # Explicitly set to None to indicate it was checked but not found/error.
            if isinstance(readme_result, Exception):
                logger.warning(
                    f"[Enrich] Failed HF README fetch for {model_id}. Setting to None. Error: {readme_result}"
                )
            else:
                # This covers the case where fetch_readme_content_enrich returned None (e.g., 404)
                logger.info(
                    f"[Enrich] README not found or fetch returned None for {model_id}. Setting hf_readme_content to None."
                )
            record["hf_readme_content"] = None
            modified = True  # Mark as modified because we added/set the key to None
        elif not readme_was_missing_or_none and record["hf_readme_content"] is None:
            # This case handles if the record *initially* had "hf_readme_content": null
            # and the fetch placeholder returned None. No change needed, not modified.
            pass
        # If readme_result is str but readme_was_missing_or_none is False, it means
        # the readme existed already, placeholder returned it, no change needed.

    # Process Paper results
    if len(paper_results_list) != len(paper_enrichment_plan):
        logger.error(
            "[Enrich] Mismatch between paper enrichment plan and results length!"
        )
    else:
        for i, paper_overall_result in enumerate(paper_results_list):
            original_paper_dict, task_plan = paper_enrichment_plan[i]
            pwc_entry_ref = original_paper_dict.get("pwc_entry")
            if not pwc_entry_ref:
                continue  # Should not happen based on plan creation

            if isinstance(paper_overall_result, Exception):
                logger.warning(
                    f"[Enrich] Error gathering results for paper (PWC ID: {pwc_entry_ref.get('pwc_id')}): {paper_overall_result}"
                )
                continue

            # paper_overall_result is a tuple: (methods_result, conference_result, repo_gather_result)
            if (
                not isinstance(paper_overall_result, (list, tuple))
                or len(paper_overall_result) != 3
            ):
                logger.error(
                    f"[Enrich] Unexpected inner result structure: {paper_overall_result}"
                )
                continue

            methods_res, conf_res, repo_res_list = paper_overall_result

            # Update Methods
            if asyncio.iscoroutine(
                task_plan["methods"]
            ):  # Check if we actually fetched
                if isinstance(methods_res, list):
                    extracted_methods = [
                        str(m.get("name")) for m in methods_res if m.get("name")
                    ]
                    pwc_entry_ref["methods"] = extracted_methods
                    if extracted_methods:
                        modified = True
                    logger.debug(
                        f"[Enrich] Updated methods for PWC ID {pwc_entry_ref.get('pwc_id')}"
                    )
                elif isinstance(methods_res, Exception):
                    logger.warning(
                        f"[Enrich] Failed methods fetch for PWC ID {pwc_entry_ref.get('pwc_id')}: {methods_res}"
                    )

            # Update Conference
            if asyncio.iscoroutine(
                task_plan["conference_entry"]
            ):  # Check if we actually fetched
                if isinstance(conf_res, dict):
                    conf_name = conf_res.get("conference")
                    if conf_name:
                        # Check again before writing, although the outer check should prevent it
                        if "conference" not in pwc_entry_ref:
                            pwc_entry_ref["conference"] = conf_name
                            modified = True
                            logger.debug(
                                f"[Enrich] Updated conference for PWC ID {pwc_entry_ref.get('pwc_id')}"
                            )
                elif isinstance(conf_res, Exception):
                    logger.warning(
                        f"[Enrich] Failed conference fetch for PWC ID {pwc_entry_ref.get('pwc_id')}: {conf_res}"
                    )

            # Update Repositories
            repos_to_update_list = task_plan["repos_to_update"]  # type: ignore[assignment]
            if isinstance(repo_res_list, list):  # Check if repo gather succeeded
                if len(repo_res_list) == len(repos_to_update_list):  # type: ignore[arg-type]
                    for k, repo_detail_result in enumerate(repo_res_list):
                        repo_dict_ref = repos_to_update_list[k]  # type: ignore[index]
                        if isinstance(repo_detail_result, dict):
                            if (
                                repo_dict_ref.get("license") is None
                                and repo_detail_result.get("license") is not None
                            ):
                                repo_dict_ref["license"] = repo_detail_result["license"]
                            if (
                                repo_dict_ref.get("language") is None
                                and repo_detail_result.get("language") is not None
                            ):
                                repo_dict_ref["language"] = repo_detail_result[
                                    "language"
                                ]
                            # No separate log here, main modified flag covers it
                        elif isinstance(repo_detail_result, Exception):
                            logger.warning(
                                f"[Enrich] Failed GitHub details fetch for {repo_dict_ref.get('url')}: {repo_detail_result}"
                            )
                else:
                    logger.error(
                        "[Enrich] Mismatch between repo tasks and results length!"
                    )
            elif isinstance(repo_res_list, Exception):
                logger.warning(
                    f"[Enrich] Error gathering repo details for PWC ID {pwc_entry_ref.get('pwc_id')}: {repo_res_list}"
                )

    return modified


async def main_enrich(
    input_file: str, output_file: str, reset_checkpoint: bool
) -> None:
    logger.info(f"Starting enrichment process.")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Reset checkpoint: {reset_checkpoint}")

    start_line = _load_checkpoint_enrich(reset_checkpoint)
    processed_count = 0
    enriched_count = 0
    error_count = 0
    last_line_num = (
        start_line  # Keep track of the last line number successfully processed
    )

    try:
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(
                output_file, "w" if start_line == 0 else "a", encoding="utf-8"
            ) as outfile,
        ):
            # If appending, we need to ensure the output file starts correctly
            if start_line > 0:
                logger.info(f"Appending to existing output file: {output_file}")
                # Ensure we are positioned correctly if outfile was 'a' mode (though 'a' handles this)
                # outfile.seek(0, os.SEEK_END) # Generally not needed for append mode

            current_line = 0
            for line in infile:
                current_line += 1
                if current_line <= start_line:
                    continue  # Skip already processed lines

                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        logger.warning(
                            f"Skipping line {current_line}: Invalid format (not a dictionary)."
                        )
                        error_count += 1
                        continue

                    logger.debug(
                        f"Processing record from line {current_line} (HF ID: {record.get('hf_model_id')})"
                    )
                    was_modified = await enrich_record(record)

                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_count += 1
                    last_line_num = (
                        current_line  # Update last successfully processed line
                    )

                    if was_modified:
                        enriched_count += 1

                    # Save checkpoint periodically
                    if processed_count % CHECKPOINT_INTERVAL_ENRICH == 0:
                        _save_checkpoint_enrich(last_line_num)

                except json.JSONDecodeError:
                    logger.error(f"Skipping line {current_line}: Invalid JSON.")
                    error_count += 1
                except Exception as e:
                    logger.error(
                        f"Error processing record from line {current_line}: {e}",
                        exc_info=True,
                    )
                    error_count += 1
                    # Decide whether to stop or continue on error

            # Final checkpoint save
            if last_line_num > start_line:  # Save only if new lines were processed
                _save_checkpoint_enrich(last_line_num)

    except FileNotFoundError:
        logger.critical(f"Input file not found: {input_file}")
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during enrichment: {e}", exc_info=True
        )
    finally:
        # Close shared HTTP client
        if http_client and not http_client.is_closed:
            await http_client.aclose()
            logger.info("HTTP client closed.")

        logger.info("--- Enrichment process finished ---")
        logger.info(f"Lines read (after checkpoint): {processed_count}")
        logger.info(f"Records enriched/modified: {enriched_count}")
        logger.info(f"Errors encountered: {error_count}")
        logger.info(f"Output written to: {output_file}")
        logger.info(
            f"Last processed line number (for next checkpoint): {last_line_num}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich existing AIGraphX data with missing fields."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_JSONL,
        help=f"Path to the input JSONL file (default: {DEFAULT_INPUT_JSONL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Path to the output enriched JSONL file (default: {DEFAULT_OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing enrichment checkpoint and start processing from the beginning of the input file.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main_enrich(args.input, args.output, args.reset))
    except KeyboardInterrupt:
        logger.info("Enrichment interrupted by user.")
    except Exception as e:
        logger.critical(f"Script execution failed: {e}", exc_info=True)
        # Attempt to close client even on failure
        try:
            if http_client and not http_client.is_closed:
                asyncio.run(http_client.aclose())
        except Exception:
            pass  # Ignore errors during final cleanup attempt
        sys.exit(1)
