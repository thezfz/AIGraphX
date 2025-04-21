import asyncio
import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set, TypedDict, Union
from datetime import datetime, timezone, timedelta
import sys  # For exiting early on critical errors
import re  # For extracting ArXiv IDs

# --- Library Imports ---
from dotenv import load_dotenv
import httpx  # Using httpx for async requests
import tenacity
from aiolimiter import AsyncLimiter
import arxiv  # type: ignore[import-untyped]
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
)
import aiohttp
import bs4  # type: ignore[import-untyped] # Use bs4 directly
import requests  # type: ignore[import-untyped]

# --- Configuration ---
# Load .env specifically for the script
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# API Keys (CRITICAL!)
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")

if not GITHUB_TOKEN:
    logging.warning(
        "GITHUB_API_KEY not found in environment variables. GitHub star fetching will be disabled."
    )

# Collection Parameters
MAX_MODELS_TO_PROCESS = 15000
MODELS_SORT_BY = "downloads"
OUTPUT_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"
# CHECKPOINT_FILE = "data/collection_checkpoint.txt"
# CHECKPOINT_INTERVAL = 50
# --- ADD ID Tracking File ---
PROCESSED_IDS_FILE = (
    "data/processed_hf_model_ids.txt"  # Stores successfully processed model IDs
)
SAVE_PROCESSED_IDS_INTERVAL = 100  # Save the set of processed IDs every N *new* models

LOG_FILE = "logs/collect_data.log"  # Log file path
LOG_DIR = os.path.dirname(LOG_FILE)

# API Endpoints
PWC_BASE_URL = "https://paperswithcode.com/api/v1/"
GITHUB_API_BASE_URL = "https://api.github.com/"

# Concurrency and Rate Limiting
MAX_CONCURRENT_MODELS = 5
hf_limiter = AsyncLimiter(5, 1.0)
arxiv_limiter = AsyncLimiter(1, 3.0)
pwc_limiter = AsyncLimiter(2, 1.0)
github_limiter = AsyncLimiter(1, 1.0)  # Keep the reduced rate

# --- Logging Setup (WITH FILE LOGGING) ---
# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Basic config first to set the root logger level (DEBUG captures everything)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
logger = logging.getLogger(__name__)  # Get the specific logger for this module

# Remove default handlers attached by basicConfig IF they exist (might not)
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# Configure Stream Handler (Console Output - INFO level)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)  # Add handler to our specific logger

# Configure File Handler (File Output - DEBUG level)
try:
    file_handler = logging.FileHandler(
        LOG_FILE, mode="a", encoding="utf-8"
    )  # Append mode
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] [%(funcName)s] %(message)s"
    )  # More detail in file
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)  # Add handler to our specific logger
except Exception as e:
    logger.error(f"Failed to set up file logging to {LOG_FILE}: {e}")

# Prevent logs from propagating to the root logger if handlers were added there
# logger.propagate = False # Optional: uncomment if double logging occurs

logger.info("Logging configured. Console level: INFO, File level: DEBUG")

# --- Tenacity Retry Configuration ---
# Define exceptions that might indicate temporary issues
RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,  # Includes ConnectError, ReadError etc.
    httpx.RemoteProtocolError,
)
# Define HTTP status codes worth retrying
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Define a retry strategy for network/server issues
# Note: This won't retry 404 Not Found or Auth errors (401/403) by default
# It also won't retry ArXiv library specific errors unless they inherit from common exceptions
retry_config_http = dict(
    stop=tenacity.stop_after_attempt(4),  # Max 4 attempts (1 initial + 3 retries)
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20)
    + tenacity.wait_random(0, 2),  # Exponential backoff + jitter
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(
        logger, logging.WARNING
    ),  # Log before retrying
    reraise=True,  # Reraise the exception after all retries fail
)
# Simpler retry for potentially blocking sync calls run in threads
retry_config_sync = dict(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),  # Simpler wait for thread issues
    retry=tenacity.retry_if_exception_type(
        Exception
    ),  # Retry any exception from the thread call
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# --- Define TypedDict for paper_data ---
class PwcEntryData(TypedDict, total=False):
    pwc_id: str
    pwc_url: str
    title: Optional[str]
    tasks: List[str]
    datasets: List[str]
    repositories: List[Dict[str, Any]]
    error: str  # For error case


class PaperData(TypedDict, total=False):
    arxiv_id_base: str
    arxiv_id_versioned: Optional[str]
    arxiv_metadata: Optional[Dict[str, Any]]
    pwc_entry: Optional[Union[PwcEntryData, Dict[str, Any]]]


class ModelOutputData(TypedDict, total=False):
    hf_model_id: str
    hf_author: Optional[str]
    hf_sha: Optional[str]
    hf_last_modified: Optional[str]
    hf_downloads: Optional[int]
    hf_likes: Optional[int]
    hf_tags: Optional[List[str]]
    hf_pipeline_tag: Optional[str]
    hf_library_name: Optional[str]
    processing_timestamp_utc: str
    linked_papers: List[PaperData]


# --- ADD ID Based Tracking ---
def _load_processed_ids() -> Set[str]:
    """Loads the set of successfully processed HF model IDs from the tracking file."""
    processed_ids: Set[str] = set()
    if not os.path.exists(PROCESSED_IDS_FILE):
        logger.info(
            f"Processed IDs file not found ('{PROCESSED_IDS_FILE}'). Starting fresh."
        )
        return processed_ids
    try:
        with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                model_id = line.strip()
                if model_id:  # Avoid adding empty lines
                    processed_ids.add(model_id)
        logger.info(
            f"Loaded {len(processed_ids)} processed model IDs from '{PROCESSED_IDS_FILE}'."
        )
    except IOError as e:
        logger.error(
            f"Failed to load processed IDs from {PROCESSED_IDS_FILE}: {e}. Starting fresh."
        )
        return set()  # Return empty set on error
    return processed_ids


def _save_processed_ids(ids_set: Set[str]) -> None:
    """Saves the current set of processed HF model IDs, overwriting the file."""
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(PROCESSED_IDS_FILE), exist_ok=True)
        with open(PROCESSED_IDS_FILE, "w", encoding="utf-8") as f:
            for model_id in sorted(list(ids_set)):  # Sort for consistency
                f.write(model_id + "\n")
        logger.debug(
            f"Saved {len(ids_set)} processed model IDs to '{PROCESSED_IDS_FILE}'."
        )
    except IOError as e:
        logger.error(f"Failed to save processed IDs to {PROCESSED_IDS_FILE}: {e}")


# --- API Client Initialization (Module Level) ---
hf_api = HfApi(token=HF_API_TOKEN)
arxiv_client = arxiv.Client()
github_headers = {}
if GITHUB_TOKEN:
    github_headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
else:
    # Still set Accept header for anonymous requests
    github_headers = {"Accept": "application/vnd.github.v3+json"}

# Use a shared httpx.AsyncClient
# Consider adjusting limits based on expected load and server behavior
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(15.0, read=60.0),  # Connect timeout 15s, read timeout 60s
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    http2=True,  # Enable HTTP/2 if servers support it
)

# --- Helper Functions (Async) ---


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_fixed(2),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_hf_model_details(model_id: str) -> Optional[ModelInfo]:
    """Fetches detailed ModelInfo using hf_api.model_info() in a thread."""
    async with hf_limiter:
        try:
            logger.debug(f"Fetching HF details for {model_id}")
            # Run synchronous model_info in a separate thread
            model_info = await asyncio.to_thread(
                hf_api.model_info, repo_id=model_id, token=HF_API_TOKEN
            )
            return model_info
        except RepositoryNotFoundError:
            logger.warning(f"HF model repository not found: {model_id}")
            return None
        except HfHubHTTPError as e:
            # --- UPDATED HfHubHTTPError Handling ---
            status_code = (
                e.response.status_code
                if hasattr(e, "response") and hasattr(e.response, "status_code")
                else None
            )
            logger.error(
                f"HF API HTTP error fetching details for {model_id}: Status {status_code} - {e}"
            )

            if status_code == 401:  # Unauthorized
                logger.critical(
                    f"HF API Authentication Error (401). Check HUGGINGFACE_API_KEY."
                )
                return None  # Don't retry auth errors

            # Define retryable HF-specific status codes that retry_config_sync should handle
            retryable_hf_status = {429, 500, 502, 503, 504}
            if status_code in retryable_hf_status:
                logger.warning(
                    f"Retryable HF API error encountered (Status {status_code}). Re-raising for tenacity."
                )
                raise e  # Re-raise retryable HF errors for tenacity to handle

            # For other non-retryable HF HTTP errors (e.g., 400 Bad Request, 403 Forbidden (permission issue))
            logger.warning(
                f"Non-retryable HF API HTTP error (Status {status_code}). Skipping model detail fetch."
            )
            return None  # Return None for non-retryable HF errors (like 400, 403)
            # --- END UPDATED ---
        except Exception as e:
            logger.error(f"Unexpected error fetching HF details for {model_id}: {e}")
            logger.debug(traceback.format_exc())
            raise e  # Re-raise other exceptions to allow tenacity retry


def extract_arxiv_ids(tags: Optional[List[str]]) -> List[str]:
    """Extracts valid ArXiv IDs (format: arxiv:YYMM.NNNNN or arxiv:arch-ive/YYMMNNN) from tags."""
    if not tags:
        return []
    arxiv_ids = set()  # Use a set to avoid duplicates
    # Regex for new format (e.g., arxiv:2303.08774) and old format (e.g., arxiv:hep-th/0207021)
    # Allowing vN at the end
    pattern = re.compile(r"arxiv:([\w.-]+(?:/\d{7})?(?:v\d+)?)$", re.IGNORECASE)
    for tag in tags:
        match = pattern.match(tag)
        if match:
            arxiv_id = match.group(1)
            # Simple check: contains digits and potentially a dot/slash
            if "." in arxiv_id or "/" in arxiv_id:
                # Remove potential version numbers (e.g., v1, v2) for searching
                arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)
                arxiv_ids.add(arxiv_id_base)
            else:
                logger.debug(
                    f"Skipping potentially invalid ArXiv ID format in tag: {tag}"
                )

    return sorted(list(arxiv_ids))


# Use tenacity with explicit parameters
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
async def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetches metadata from ArXiv API using arxiv.py Client.results() in a thread."""
    async with arxiv_limiter:
        try:
            logger.debug(f"Fetching ArXiv metadata for {arxiv_id} using Client.results")
            # Define the search query
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)
            # Use the globally initialized client (arxiv_client) to run the search in a thread
            # Pass the search object to client.results()
            # Convert the results generator to a list within the thread
            results = await asyncio.to_thread(
                list, arxiv_client.results(search)
            )  # Use client.results(search)
            if results:
                paper = results[0]
                # Extract relevant fields
                return {
                    "arxiv_id_versioned": paper.entry_id.split("/")[
                        -1
                    ],  # Get ID with version
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "summary": paper.summary.replace(
                        "\n", " "
                    ),  # Replace newlines in summary
                    "published_date": paper.published.isoformat()
                    if paper.published
                    else None,
                    "updated_date": paper.updated.isoformat()
                    if paper.updated
                    else None,
                    "pdf_url": paper.pdf_url,
                    "doi": paper.doi,
                    "primary_category": paper.primary_category,
                    "categories": paper.categories,
                }
            else:
                logger.warning(
                    f"ArXiv ID {arxiv_id} not found via arxiv.py API (Client.results)."
                )
                return None
        except Exception as e:
            # Catch exceptions from arxiv.py or asyncio.to_thread
            logger.error(
                f"Error fetching ArXiv metadata for {arxiv_id} (Client.results): {e}"
            )
            logger.debug(traceback.format_exc())
            # Do not re-raise here, let tenacity handle retries based on the exception type
            # If tenacity gives up, it will re-raise, caught in process_single_model
            return None  # Return None on error after retries or non-retryable error


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
async def find_pwc_entry_by_arxiv_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Queries PWC API using ?arxiv_id= parameter via shared httpx client."""
    # Remove potential version number from ArXiv ID for PWC query
    arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)
    async with pwc_limiter:
        url = f"{PWC_BASE_URL}papers/"
        params = {"arxiv_id": arxiv_id_base}
        logger.debug(f"Querying PWC for ArXiv ID {arxiv_id_base} (from {arxiv_id})")
        try:
            response = await http_client.get(url, params=params)
            response.raise_for_status()  # Check for 4xx/5xx errors
            data = response.json()
            count = data.get("count")

            if isinstance(count, int) and count == 1 and data.get("results"):
                logger.info(f"Found 1 PWC entry for ArXiv ID {arxiv_id_base}")
                result_entry = data["results"][0]
                # Explicitly check if the result is a dictionary before returning
                if isinstance(result_entry, dict):
                    return result_entry
                else:
                    logger.warning(
                        f"PWC entry for ArXiv ID {arxiv_id_base} is not a dictionary: {type(result_entry)}"
                    )
                    return None
            elif isinstance(count, int) and count == 0:
                logger.info(f"No PWC entry found for ArXiv ID {arxiv_id_base}")
                return None
            else:
                # Includes count > 1 or unexpected response structure
                logger.warning(
                    f"Found {count} PWC entries or unexpected response for ArXiv ID {arxiv_id_base}. Skipping."
                )
                return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # PWC might return 404 if the ArXiv ID format is wrong or not indexed
                logger.info(f"PWC API returned 404 for ArXiv ID query: {arxiv_id_base}")
                return None
            # Other HTTP errors will be retried by tenacity based on retry_config_http
            logger.warning(
                f"HTTP error querying PWC for {arxiv_id_base}: {e.response.status_code} - {e}"
            )
            raise  # Reraise for tenacity
        except (json.JSONDecodeError, Exception) as e:
            logger.error(
                f"Error querying PWC or parsing response for {arxiv_id_base}: {e}"
            )
            logger.debug(traceback.format_exc())
            raise  # Reraise for tenacity


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
async def fetch_pwc_relation_list(
    pwc_paper_id: str, relation: str
) -> List[Dict[str, Any]]:
    """
    Fetches a list of related items (repos, tasks, datasets) for a PWC paper ID.
    Handles cases where the API returns a paginated dictionary instead of a direct list.
    """
    all_results = []
    current_url: Optional[str] = f"{PWC_BASE_URL}papers/{pwc_paper_id}/{relation}/"
    page_num = 1  # Start with page 1 for potential future pagination handling

    while (
        current_url
    ):  # Loop as long as there is a 'next' URL (or for the first request)
        async with pwc_limiter:
            logger.debug(
                f"Fetching PWC {relation} for {pwc_paper_id} (Page: {page_num if page_num > 1 else 'initial'}) from {current_url}"
            )
            try:
                response = await http_client.get(current_url)
                response.raise_for_status()
                data = response.json()

                # --- NEW: Handle both list and dictionary responses ---
                results = []
                next_url: Optional[str] = None

                if isinstance(data, list):
                    # Case 1: API returns a direct list (original expected behavior)
                    logger.debug(
                        f"PWC {relation} for {pwc_paper_id} returned a direct list."
                    )
                    results = data
                    # No more pages if it's a direct list
                elif (
                    isinstance(data, dict)
                    and "results" in data
                    and isinstance(data["results"], list)
                ):
                    # Case 2: API returns a paginated dictionary (observed behavior)
                    logger.debug(
                        f"PWC {relation} for {pwc_paper_id} returned a paginated dictionary."
                    )
                    results = data["results"]
                    # Check for pagination (though in examples 'next' was null)
                    next_url = data.get("next")
                    # Basic protection against infinite loops if 'next' URL is same as current
                    if next_url == current_url:
                        logger.warning(
                            f"PWC {relation} 'next' URL is same as current, breaking loop: {current_url}"
                        )
                        next_url = None
                else:
                    # Case 3: Unexpected format
                    logger.warning(
                        f"Unexpected data format for PWC {relation} for {pwc_paper_id}. Expected list or dict with 'results', got: {type(data)}"
                    )
                    # Stop processing this relation if format is wrong
                    break

                all_results.extend(results)

                # Prepare for the next iteration (or break if no next page)
                current_url = next_url  # Set URL for next page, or None to exit loop
                if current_url:
                    page_num += 1

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"HTTP error fetching PWC {relation} page for {pwc_paper_id}: {e.response.status_code}"
                )
                # Let tenacity handle retries based on status code
                raise e  # Reraise for tenacity to catch
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"Error fetching or parsing PWC {relation} page for {pwc_paper_id}: {e}"
                )
                logger.debug(traceback.format_exc())
                # Let tenacity handle retries based on exception type
                raise e  # Reraise for tenacity

    # After loop (or if initial fetch failed after retries), return all collected results
    if (
        not all_results and current_url is None
    ):  # Check if loop finished normally without results
        logger.info(
            f"Finished fetching PWC {relation} for {pwc_paper_id}. Found 0 results."
        )
    elif current_url:  # Loop broke due to error after retries
        logger.error(
            f"Failed to fetch all pages for PWC {relation} for {pwc_paper_id} after retries."
        )

    return all_results


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
async def fetch_github_stars(
    repo_url: str, follow_redirects: bool = True, max_redirects: int = 3
) -> Optional[int]:
    """
    Fetches star count from GitHub API for a given repo URL via shared httpx client.
    Optionally follows 301/302 redirects, handling both owner/repo and repositories/ID URLs.
    """
    if not GITHUB_TOKEN:
        return None
    if not repo_url:  # Simplified initial check
        return None

    current_api_url_to_fetch = None  # The API URL we will actually GET
    original_url = repo_url  # Keep original for logging
    redirect_count = 0

    # --- Initial URL Parsing to get the *first* API URL ---
    try:
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
                    logger.debug(
                        f"Could not parse owner/repo from initial URL: {repo_url}"
                    )
                    return None
            else:
                logger.debug(
                    f"Initial URL format not recognized as github.com/owner/repo: {repo_url}"
                )
                # It *might* already be an api.github.com/repositories/ID URL, handle below
        # Allow direct api.github.com/repositories/ID URLs as input too
        elif "api.github.com/repositories/" in repo_url.lower():
            current_api_url_to_fetch = (
                repo_url  # Assume it's already the correct API URL
            )
        else:
            logger.debug(
                f"URL does not appear to be a standard GitHub repo URL: {repo_url}"
            )
            return None
    except Exception as parse_error:
        logger.warning(f"Error parsing initial GitHub URL {repo_url}: {parse_error}")
        return None

    # --- Loop for fetching and handling redirects ---
    while current_api_url_to_fetch and redirect_count <= max_redirects:
        async with github_limiter:
            logger.debug(
                f"Fetching GitHub data from {current_api_url_to_fetch} (Attempt: {redirect_count + 1})"
            )
            try:
                response = await http_client.get(
                    current_api_url_to_fetch,
                    headers=github_headers,
                    follow_redirects=False,
                )  # Disable auto redirects

                if response.status_code == 200:
                    data = response.json()
                    stars = data.get("stargazers_count")
                    owner_repo_name = data.get(
                        "full_name", "[unknown repo]"
                    )  # Get name from response
                    if isinstance(stars, int):
                        log_prefix = (
                            f"Successfully fetched stars for {owner_repo_name}: {stars}"
                        )
                        log_suffix = (
                            f" (Original URL: {original_url})"
                            if redirect_count > 0
                            else ""
                        )
                        logger.info(log_prefix + log_suffix)
                        return stars
                    else:
                        logger.warning(
                            f"Could not parse stars for {owner_repo_name}. Value: {stars}"
                        )
                        return None

                # --- MANUAL REDIRECT HANDLING ---
                elif (
                    follow_redirects
                    and response.status_code in (301, 302, 307, 308)
                    and "location" in response.headers
                ):
                    redirect_count += 1
                    redirect_url = response.headers["location"]
                    logger.info(
                        f"GitHub request to {current_api_url_to_fetch} redirected ({response.status_code}) to: {redirect_url}. Following redirect ({redirect_count}/{max_redirects})."
                    )
                    # The redirect location IS the new API URL to fetch
                    current_api_url_to_fetch = redirect_url
                    continue  # Go to the next iteration with the new URL

                elif response.status_code == 404:
                    logger.info(
                        f"GitHub API returned 404 for URL: {current_api_url_to_fetch}"
                    )
                    return None  # Stop, resource not found
                elif response.status_code == 403:
                    logger.error(
                        f"GitHub API forbidden (403) for URL: {current_api_url_to_fetch}. Check token/rate limits. Stopping retries for this repo."
                    )
                    return None
                elif response.status_code == 401:
                    logger.error(
                        f"GitHub API Unauthorized (401) for URL: {current_api_url_to_fetch}. Check GITHUB_API_KEY."
                    )
                    return None
                else:
                    logger.warning(
                        f"HTTP error {response.status_code} fetching GitHub data from {current_api_url_to_fetch}."
                    )
                    response.raise_for_status()  # Raise exception for tenacity

            except (json.JSONDecodeError, Exception) as e:
                logger.error(
                    f"Error fetching/parsing GitHub data from {current_api_url_to_fetch}: {e}"
                )
                logger.debug(traceback.format_exc())
                raise e  # Reraise for tenacity

    # If loop finishes due to max_redirects reached
    if redirect_count > max_redirects:
        logger.warning(
            f"Exceeded maximum redirects ({max_redirects}) for original URL: {original_url}"
        )

    return None  # Return None if stars couldn't be fetched


async def process_single_model(model_id: str) -> Optional[ModelOutputData]:
    """Orchestrates fetching and combining all data for a single HF model."""
    logger.info(f"--- Start processing model: {model_id} ---")
    processing_start_time = datetime.now(timezone.utc)
    output_data: Optional[ModelOutputData] = None  # Initialize

    try:
        # 1. Get HF Details
        hf_model_info = await fetch_hf_model_details(model_id)
        if not hf_model_info:
            logger.error(
                f"Failed to get required HF details for {model_id}. Skipping model."
            )
            return None  # Cannot proceed without basic HF info

        # Initialize output structure with HF data
        output_data = {
            "hf_model_id": hf_model_info.id,
            "hf_author": hf_model_info.author,
            "hf_sha": hf_model_info.sha,
            "hf_last_modified": hf_model_info.lastModified.isoformat()
            if hf_model_info.lastModified
            else None,
            "hf_downloads": hf_model_info.downloads,
            "hf_likes": hf_model_info.likes,
            "hf_tags": hf_model_info.tags,
            "hf_pipeline_tag": hf_model_info.pipeline_tag,
            "hf_library_name": hf_model_info.library_name,
            "processing_timestamp_utc": processing_start_time.isoformat(),
            "linked_papers": [],  # Initialize list for linked papers
        }

        # 2. Extract ArXiv IDs from HF tags
        arxiv_ids = extract_arxiv_ids(hf_model_info.tags)
        if not arxiv_ids:
            logger.info(f"No valid ArXiv IDs found in tags for {model_id}.")
            return output_data  # Return HF data even if no papers linked

        logger.info(f"Found {len(arxiv_ids)} ArXiv ID(s) for {model_id}: {arxiv_ids}")

        # 3. Process each found ArXiv ID
        for arxiv_id in arxiv_ids:
            paper_data: PaperData = {
                "arxiv_id_base": re.sub(r"v\d+$", "", arxiv_id),
                "arxiv_metadata": None,
                "pwc_entry": None,
            }

            try:
                # 3a. Get ArXiv Metadata
                arxiv_meta = await fetch_arxiv_metadata(arxiv_id)
                if arxiv_meta:
                    paper_data["arxiv_metadata"] = arxiv_meta
                    # Use the versioned ArXiv ID found by the API if available
                    paper_data["arxiv_id_versioned"] = arxiv_meta.get(
                        "arxiv_id_versioned", paper_data["arxiv_id_base"]
                    )
                else:
                    # Logged in fetch function, still add paper entry with base ID
                    output_data["linked_papers"].append(paper_data)
                    continue  # Move to next ArXiv ID if metadata fails

                # 3b. Find PWC Entry using the base ArXiv ID
                pwc_entry_summary = await find_pwc_entry_by_arxiv_id(
                    paper_data["arxiv_id_base"]
                )
                if not pwc_entry_summary:
                    # Logged in find function
                    output_data["linked_papers"].append(
                        paper_data
                    )  # Add entry with ArXiv data
                    continue  # Move to next ArXiv ID if PWC entry not found

                pwc_paper_id = pwc_entry_summary.get("id")
                if not pwc_paper_id:
                    logger.warning(
                        f"PWC entry found for {paper_data['arxiv_id_base']} but missing 'id'."
                    )
                    output_data["linked_papers"].append(paper_data)
                    continue

                # 3c. Get PWC Details (Repositories, Tasks, Datasets) concurrently
                pwc_details_tasks = {
                    "repositories": fetch_pwc_relation_list(
                        pwc_paper_id, "repositories"
                    ),
                    "tasks": fetch_pwc_relation_list(pwc_paper_id, "tasks"),
                    "datasets": fetch_pwc_relation_list(pwc_paper_id, "datasets"),
                }
                pwc_details_results = await asyncio.gather(
                    *pwc_details_tasks.values(), return_exceptions=True
                )
                pwc_details = dict(zip(pwc_details_tasks.keys(), pwc_details_results))

                # Check for exceptions during gather
                fetched_repos = []
                fetched_tasks = []
                fetched_datasets = []
                if isinstance(pwc_details["repositories"], list):
                    fetched_repos = pwc_details["repositories"]
                else:
                    logger.warning(
                        f"Failed to fetch PWC repositories for {pwc_paper_id}: {pwc_details['repositories']}"
                    )
                if isinstance(pwc_details["tasks"], list):
                    fetched_tasks = pwc_details["tasks"]
                else:
                    logger.warning(
                        f"Failed to fetch PWC tasks for {pwc_paper_id}: {pwc_details['tasks']}"
                    )
                if isinstance(pwc_details["datasets"], list):
                    fetched_datasets = pwc_details["datasets"]
                else:
                    logger.warning(
                        f"Failed to fetch PWC datasets for {pwc_paper_id}: {pwc_details['datasets']}"
                    )

                # 3d. Get GitHub Stars for PWC Repositories concurrently
                processed_repos = []
                if fetched_repos:
                    star_fetch_tasks = []
                    valid_repo_data = []  # Keep track of repos we attempt to fetch stars for
                    for repo in fetched_repos:
                        repo_url = repo.get("url")
                        if repo_url and "github.com" in repo_url.lower():
                            star_fetch_tasks.append(fetch_github_stars(repo_url))
                            valid_repo_data.append(repo)
                        else:
                            # If not a GitHub URL or no URL, add repo without stars
                            processed_repos.append(
                                {
                                    "url": repo_url,
                                    "stars": None,  # Mark as not applicable or unknown
                                    "is_official": repo.get("is_official"),
                                    "framework": repo.get("framework"),
                                }
                            )

                    if star_fetch_tasks:
                        repo_stars_results = await asyncio.gather(
                            *star_fetch_tasks, return_exceptions=True
                        )

                        for repo_meta, stars_result in zip(
                            valid_repo_data, repo_stars_results
                        ):
                            stars = None
                            if isinstance(stars_result, int):
                                stars = stars_result
                            elif isinstance(stars_result, Exception):
                                logger.warning(
                                    f"Failed to fetch GitHub stars for {repo_meta.get('url')}: {stars_result}"
                                )
                            # Else: stars_result is None (e.g., non-GitHub URL handled earlier or fetch returned None)

                            processed_repos.append(
                                {
                                    "url": repo_meta.get("url"),
                                    "stars": stars,
                                    "is_official": repo_meta.get("is_official"),
                                    "framework": repo_meta.get("framework"),
                                }
                            )
                else:
                    logger.info(
                        f"No repositories listed in PWC entry for {pwc_paper_id}"
                    )

                # Assemble PWC entry data
                paper_data["pwc_entry"] = {
                    "pwc_id": pwc_paper_id,
                    "pwc_url": f"https://paperswithcode.com/paper/{pwc_paper_id}",
                    "title": pwc_entry_summary.get(
                        "title"
                    ),  # Title from PWC summary search
                    "tasks": [
                        str(task.get("name"))
                        for task in fetched_tasks
                        if task.get("name") is not None
                    ],
                    "datasets": [
                        str(dataset.get("name"))
                        for dataset in fetched_datasets
                        if dataset.get("name") is not None
                    ],
                    "repositories": processed_repos,  # Includes stars where available
                }

                # Add the fully populated paper data to the list
                output_data["linked_papers"].append(paper_data)

            except Exception as e:
                # Catch unexpected errors during the processing of a single ArXiv ID
                logger.error(
                    f"Unexpected error processing ArXiv ID {arxiv_id} for model {model_id}: {e}"
                )
                logger.error(traceback.format_exc())
                # Add partially filled paper_data to indicate an attempt was made
                if "pwc_entry" not in paper_data:
                    paper_data["pwc_entry"] = {"error": str(e)}
                if output_data:  # Ensure output_data was initialized
                    output_data["linked_papers"].append(paper_data)
                # Continue to the next ArXiv ID

        logger.info(f"--- Finished processing model: {model_id} ---")
        return output_data

    except Exception as e:
        # Catch unexpected errors at the top level of processing a model
        logger.critical(f"CRITICAL error during processing of model {model_id}: {e}")
        logger.critical(traceback.format_exc())
        return output_data  # Return potentially partial data collected before the critical error


# --- New function to fetch target model IDs (Reusable) ---
async def fetch_target_model_ids(limit: int, sort_by: str) -> List[str]:
    """Fetches a list of model IDs from Hugging Face Hub, sorted and limited."""
    logger.info(f"Fetching top {limit} model IDs from HF Hub, sorted by {sort_by}...")
    target_ids: List[str] = []
    try:
        # Use tenacity for retrying the list_models call
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
            before_sleep=tenacity.before_sleep_log(
                logger, logging.WARNING
            ),
            reraise=True,
        )
        async def get_models_list() -> List[ModelInfo]:
            # Run the synchronous call in a thread to avoid blocking asyncio loop
            loop = asyncio.get_running_loop()
            models_iterator = await loop.run_in_executor(
                None,  # Use default executor
                lambda: hf_api.list_models(
                    full=False,  # Only need modelId
                    sort=sort_by,
                    direction=-1,  # Descending order (most downloads/likes first)
                    limit=limit,  # Apply the limit directly
                    # Removed fetch_config=False - list_models doesn't have this param
                ),
            )
            return list(models_iterator)

        models = await get_models_list()
        target_ids = [model.id for model in models if model.id]
        logger.info(f"Successfully fetched {len(target_ids)} model IDs from HF Hub.")

    except tenacity.RetryError as e:
        logger.error(
            f"Failed to fetch model list from HF Hub after multiple retries: {e}. Returning empty list."
        )
    except (HfHubHTTPError, Exception) as e:
        logger.error(
            f"An unexpected error occurred while fetching model list from HF Hub: {e}. Returning empty list."
        )
        logger.debug(traceback.format_exc())  # Log full traceback for debugging

    return target_ids


# --- Main Orchestration ---
async def main() -> None:
    # Load already processed IDs instead of count
    processed_ids_set = _load_processed_ids()  # This set will be updated directly
    logger.info(
        f"Starting data collection run. Will attempt to process up to {MAX_MODELS_TO_PROCESS} target models."
    )
    logger.info(f"Loaded {len(processed_ids_set)} previously processed model IDs.")

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_JSONL_FILE)
    try:
        os.makedirs(
            output_dir, exist_ok=True
        )  # Ensure data directory exists before writing
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.critical(
            f"Failed to create output directory {output_dir}: {e}. Exiting."
        )
        sys.exit(1)

    # Determine write mode (always append with ID tracking)
    # Optionally, you could still have a flag to force overwrite/clear processed IDs
    write_mode = "a"
    logger.info(f"Output file '{OUTPUT_JSONL_FILE}' will be appended to.")
    # Ensure data dir for processed IDs also exists (handled in _save_processed_ids)

    target_model_ids = await fetch_target_model_ids(
        MAX_MODELS_TO_PROCESS, MODELS_SORT_BY
    )
    if not target_model_ids:
        logger.warning("Could not fetch target model IDs. Exiting.")
        await close_http_client_safely()  # Ensure client is closed
        return

    models_to_run = [mid for mid in target_model_ids if mid not in processed_ids_set]
    logger.info(f"Target models from HF: {len(target_model_ids)}")
    logger.info(f"Already processed models: {len(processed_ids_set)}")
    logger.info(f"Models to process in this run: {len(models_to_run)}")

    if not models_to_run:
        logger.info(
            "No new models to process based on the current target list and processed IDs file."
        )
        await close_http_client_safely()  # Ensure client is closed
        return

    attempted_this_run = 0
    successful_saves_this_run = 0  # New counter for successful saves

    process_semaphore = asyncio.Semaphore(MAX_CONCURRENT_MODELS)
    tasks = []
    for model_id in models_to_run:

        async def process_with_semaphore(mid: str) -> Optional[ModelOutputData]:
            nonlocal attempted_this_run  # Make sure to modify the outer scope variable
            async with process_semaphore:
                attempted_this_run += 1
                return await process_single_model(mid)  # Call the main processing logic

        tasks.append(process_with_semaphore(model_id))

    # Process tasks and write results
    try:
        with open(OUTPUT_JSONL_FILE, "a", encoding="utf-8") as f:
            # Process tasks as they complete using asyncio.as_completed
            for i, future in enumerate(asyncio.as_completed(tasks), 1):
                # Removed: current_total_processed = processed_count + i # Track attempts in this run
                try:
                    result_data = await future  # Wait for the next task to finish

                    if (
                        result_data
                        and isinstance(result_data, dict)
                        and "hf_model_id" in result_data
                    ):
                        hf_model_id = result_data["hf_model_id"]

                        # --- IMMEDIATE SAVE LOGIC ---
                        # 1. Write to JSONL file
                        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                        f.flush()  # Flush immediately after write

                        # 2. Add to the main processed set
                        processed_ids_set.add(hf_model_id)

                        # 3. Immediately save the updated set
                        _save_processed_ids(processed_ids_set)
                        # --- END IMMEDIATE SAVE LOGIC ---

                        successful_saves_this_run += 1
                        logger.info(
                            f"Saved: {hf_model_id} ({successful_saves_this_run} saved this run / {attempted_this_run} attempted / {len(processed_ids_set)} total unique)"
                        )

                    elif result_data is None:
                        # Handle cases where process_single_model explicitly returned None
                        logger.warning(
                            f"Skipped saving for model attempt {attempted_this_run} (returned None). Total unique processed so far: {len(processed_ids_set)}"
                        )
                    else:
                        logger.error(
                            f"Unexpected result type from process_single_model (attempt {attempted_this_run}): {type(result_data)}. Total unique processed so far: {len(processed_ids_set)}"
                        )

                except Exception as e:
                    # Catch unexpected errors from await future or within process_with_semaphore
                    # Note: attempted_this_run was already incremented
                    logger.error(
                        f"Error awaiting or processing model future (attempt {attempted_this_run}): {e}"
                    )
                    logger.error(traceback.format_exc())
                    # Do not add to processed_ids_set or save if processing failed

    except IOError as e:
        logger.critical(
            f"CRITICAL: Failed to open or write to output file {OUTPUT_JSONL_FILE}: {e}"
        )
    except Exception as e:
        logger.critical(
            f"CRITICAL: An unexpected error occurred during the main processing loop: {e}"
        )
        logger.critical(traceback.format_exc())
    finally:
        # --- Final Saving and Cleanup ---
        # Final save is still good practice as a backup, although immediate saves reduce its necessity
        logger.info(
            "Saving final set of processed model IDs (redundant if no errors occurred)..."
        )
        _save_processed_ids(processed_ids_set)  # Save the most current set

        # Close the shared httpx client gracefully
        await close_http_client_safely()

        # Final summary logs
        logger.info("--- Data collection run finished ---")
        logger.info(
            f"Attempted to process {attempted_this_run} new models in this run."
        )
        logger.info(
            f"Successfully collected and saved data for {successful_saves_this_run} new models."
        )  # Use new counter
        logger.info(
            f"Total unique models processed across all runs (in {PROCESSED_IDS_FILE}): {len(processed_ids_set)}"
        )  # Reflect final set size


async def close_http_client_safely() -> None:
    """Close the shared client if it exists and is not closed."""
    logger.info("Closing HTTP client...")
    # Check if client exists and is closable
    if (
        "http_client" in globals()
        and isinstance(http_client, httpx.AsyncClient)
        and not http_client.is_closed
    ):
        try:
            await http_client.aclose()
            logger.info("HTTP client closed.")
        except Exception as close_e:
            logger.error(f"Error closing HTTP client: {close_e}")
    else:
        logger.info("HTTP client already closed or not initialized.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(
            "Collection interrupted by user. Attempting to save final state and close client."
        )
        # Attempt graceful shutdown on KeyboardInterrupt
        # Note: state saving happens in main's finally block now
        asyncio.run(close_http_client_safely())  # Just ensure client is closed
    except Exception as e:
        logger.critical(f"Unhandled critical exception during script execution: {e}")
        logger.critical(traceback.format_exc())
        # Attempt to close client even on critical error outside main loop
        asyncio.run(close_http_client_safely())
        sys.exit(1)
