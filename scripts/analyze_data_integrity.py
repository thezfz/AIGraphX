import json
import os
import logging
from collections import Counter
from typing import Dict, Set, List, Any, Optional
import traceback
from datetime import datetime

# --- Configuration ---
INPUT_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"
REPORT_FILE = "logs/data_integrity_report.txt"
LOG_DIR = os.path.dirname(REPORT_FILE)

# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [DataAnalysis] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "logs/data_analysis.log", mode="w", encoding="utf-8"
        ),  # Overwrite log each time
    ],
)
logger = logging.getLogger(__name__)


def analyze_data_integrity(jsonl_filepath: str) -> Optional[Dict[str, Any]]:
    """
    Reads the JSONL data file, performs integrity checks, and calculates statistics.

    Args:
        jsonl_filepath (str): Path to the input JSONL file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the analysis results, or None if file not found.
    """
    stats: Dict[str, Any] = {
        "total_lines_read": 0,
        "json_parse_errors": 0,
        "missing_hf_id": 0,
        "unique_hf_model_ids": set(),
        "models_with_arxiv_link": set(),
        "unique_arxiv_ids": set(),
        "total_arxiv_links_processed": 0,
        "arxiv_links_with_pwc_entry": 0,
        "total_github_repos_found": 0,
        "github_repos_with_none_stars": 0,
        "missing_hf_author": 0,
        "missing_hf_sha": 0,
        "missing_hf_last_modified": 0,
    }

    logger.info(f"Starting data integrity analysis for: {jsonl_filepath}")

    try:
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats["total_lines_read"] += 1
                if line_num % 500 == 0:
                    logger.info(f"Processed {line_num} lines...")

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_num}: Failed to parse JSON.")
                    stats["json_parse_errors"] += 1
                    continue
                except Exception as e:
                    logger.error(f"Line {line_num}: Unexpected error parsing JSON: {e}")
                    stats["json_parse_errors"] += 1
                    continue

                # --- HF Model Checks ---
                hf_model_id = data.get("hf_model_id")
                if not hf_model_id:
                    logger.warning(f"Line {line_num}: Record missing 'hf_model_id'.")
                    stats["missing_hf_id"] += 1
                    continue  # Skip further processing if no primary ID

                stats["unique_hf_model_ids"].add(hf_model_id)

                # Optional: Check other HF fields
                if not data.get("hf_author"):
                    stats["missing_hf_author"] += 1
                if not data.get("hf_sha"):
                    stats["missing_hf_sha"] += 1
                if not data.get("hf_last_modified"):
                    stats["missing_hf_last_modified"] += 1

                # --- Linked Papers Analysis ---
                linked_papers = data.get("linked_papers", [])
                if isinstance(linked_papers, list) and linked_papers:
                    stats["models_with_arxiv_link"].add(hf_model_id)

                    for paper in linked_papers:
                        if not isinstance(paper, dict):
                            continue  # Skip malformed paper entries

                        arxiv_id_base = paper.get("arxiv_id_base")
                        if arxiv_id_base:
                            stats["unique_arxiv_ids"].add(arxiv_id_base)
                            stats["total_arxiv_links_processed"] += 1

                            pwc_entry = paper.get("pwc_entry")
                            if (
                                isinstance(pwc_entry, dict) and pwc_entry
                            ):  # Check if pwc_entry exists and is not empty
                                stats["arxiv_links_with_pwc_entry"] += 1

                                # --- GitHub Repo Analysis ---
                                repositories = pwc_entry.get("repositories", [])
                                if isinstance(repositories, list):
                                    for repo in repositories:
                                        if not isinstance(repo, dict):
                                            continue  # Skip malformed repo entries

                                        url = repo.get("url")
                                        if (
                                            url
                                            and isinstance(url, str)
                                            and "github.com" in url.lower()
                                        ):
                                            stats["total_github_repos_found"] += 1
                                            if repo.get("stars") is None:
                                                # Note: This counts cases where stars were explicitly None
                                                # It doesn't capture repos that were filtered out *before* star fetching (e.g., non-github)
                                                stats[
                                                    "github_repos_with_none_stars"
                                                ] += 1
                        else:
                            logger.debug(
                                f"Line {line_num}: Paper entry missing 'arxiv_id_base' in linked_papers for model {hf_model_id}."
                            )

    except FileNotFoundError:
        logger.error(f"Input file not found: {jsonl_filepath}")
        return None
    except IOError as e:
        logger.error(f"I/O error reading file {jsonl_filepath}: {e}")
        return None
    except Exception as e:
        logger.critical(f"Unexpected critical error during analysis: {e}")
        logger.critical(traceback.format_exc())
        return None

    logger.info(f"Finished analyzing {stats['total_lines_read']} lines.")
    return stats


def format_and_log_results(
    stats: Optional[Dict[str, Any]], report_filepath: str
) -> None:  # Allow stats to be None
    """Formats the statistics and logs them, optionally writing to a file."""
    if not stats:
        logger.error("Analysis failed or file not found, no statistics to report.")
        return

    # --- Calculate derived statistics ---
    total_unique_models = len(stats["unique_hf_model_ids"])
    total_models_with_arxiv = len(stats["models_with_arxiv_link"])
    total_unique_arxiv = len(stats["unique_arxiv_ids"])

    pwc_link_success_rate = 0
    if stats["total_arxiv_links_processed"] > 0:
        pwc_link_success_rate = (
            stats["arxiv_links_with_pwc_entry"] / stats["total_arxiv_links_processed"]
        ) * 100

    github_star_missing_rate = 0
    if stats["total_github_repos_found"] > 0:
        github_star_missing_rate = (
            stats["github_repos_with_none_stars"] / stats["total_github_repos_found"]
        ) * 100

    # --- Prepare Report ---
    report_lines = [
        "--- Data Integrity and Statistics Report ---",
        f"Analyzed File: {INPUT_JSONL_FILE}",
        f"Timestamp: {datetime.now().isoformat()}",
        "=" * 40,
        f"File Reading:",
        f"  - Total lines read: {stats['total_lines_read']}",
        f"  - JSON parsing errors: {stats['json_parse_errors']}",
        f"  - Records missing 'hf_model_id': {stats['missing_hf_id']}",
        "=" * 40,
        f"Hugging Face Model Statistics:",
        f"  - Total unique models found: {total_unique_models}",
        f"  - Models with at least one ArXiv link: {total_models_with_arxiv}",
        f"  - Models missing 'hf_author': {stats['missing_hf_author']}",
        f"  - Models missing 'hf_sha': {stats['missing_hf_sha']}",
        f"  - Models missing 'hf_last_modified': {stats['missing_hf_last_modified']}",
        "=" * 40,
        f"ArXiv & PWC Linkage:",
        f"  - Total unique ArXiv IDs found in links: {total_unique_arxiv}",
        f"  - Total ArXiv links processed: {stats['total_arxiv_links_processed']}",
        f"  - ArXiv links successfully linked to a PWC entry: {stats['arxiv_links_with_pwc_entry']}",
        f"  - ArXiv -> PWC Link Success Rate: {pwc_link_success_rate:.2f}%",
        "=" * 40,
        f"GitHub Repository Statistics:",
        f"  - Total GitHub repositories found in PWC entries: {stats['total_github_repos_found']}",
        f"  - GitHub repositories where 'stars' is None: {stats['github_repos_with_none_stars']}",
        f"  - GitHub Star Data Missing Rate: {github_star_missing_rate:.2f}%",
        "=" * 40,
    ]

    # --- Log Results ---
    logger.info("--- Analysis Summary ---")
    for line in report_lines:
        logger.info(line)
    logger.info("--- End of Summary ---")

    # --- Write Report File ---
    try:
        with open(report_filepath, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")
        logger.info(f"Full report written to: {report_filepath}")
    except IOError as e:
        logger.error(f"Failed to write report file to {report_filepath}: {e}")


if __name__ == "__main__":
    logger.info("Starting data integrity analysis script...")
    analysis_results = analyze_data_integrity(INPUT_JSONL_FILE)
    format_and_log_results(
        analysis_results, REPORT_FILE
    )  # Pass None if analysis failed
    logger.info("Data integrity analysis script finished.")
