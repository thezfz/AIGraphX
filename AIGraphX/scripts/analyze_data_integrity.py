#!/usr/bin/env python
import json
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Set, List, Any, Optional
import traceback
from datetime import datetime
import argparse  # Use argparse for flexibility

# --- Configuration ---
DEFAULT_INPUT_JSONL = (
    "data/aigraphx_knowledge_data.jsonl"  # Assume enriched file is input
)
DEFAULT_REPORT_FILE = "logs/data_validation_report.txt"
DEFAULT_LOG_FILE = "logs/data_validation.log"


# --- Logging Setup ---
# Setup moved to a function for clarity
def setup_logging(log_filepath: str) -> logging.Logger:
    """Configures logging for the script."""
    log_dir = os.path.dirname(log_filepath)
    os.makedirs(log_dir, exist_ok=True)

    # Remove existing handlers to avoid duplicates if script is run multiple times
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [DataValidation] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_filepath, mode="w", encoding="utf-8"
            ),  # Overwrite log
        ],
    )
    return logging.getLogger(__name__)


# --- Main Analysis Function ---
def analyze_data_integrity_v2(jsonl_filepath: str) -> Optional[Dict[str, Any]]:
    """
    Reads the JSONL data file, performs comprehensive integrity checks,
    and calculates statistics including new fields.

    Args:
        jsonl_filepath (str): Path to the input JSONL file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the analysis results,
                                  or None if file not found or critical error occurs.
    """
    stats: Dict[str, Any] = {
        # Overall File Stats
        "input_filepath": jsonl_filepath,
        "total_lines_read": 0,
        "json_parse_errors": 0,
        "records_valid_json": 0,
        # HF Model Level Stats
        "missing_hf_id": 0,
        "unique_hf_model_ids": set(),
        "missing_hf_author": 0,
        "missing_hf_sha": 0,
        "missing_hf_last_modified": 0,
        "missing_hf_tags": 0,  # Added
        "missing_hf_downloads": 0,  # Added
        "missing_hf_likes": 0,  # Added
        # README Stats
        "missing_readme_key": 0,  # Check if the key exists at all
        "readme_is_null": 0,  # Key exists, value is null/None
        "readme_is_present": 0,  # Key exists, value is a non-empty string
        # Dataset Links Stats
        "missing_dataset_links_key": 0,
        "dataset_links_is_empty": 0,  # Key exists, value is []
        "dataset_links_is_present": 0,  # Key exists, value is non-empty list
        "total_dataset_links_found": 0,
        "unique_dataset_links": set(),
        # Linked Papers & PWC Stats
        "models_with_linked_papers": set(),
        "total_papers_processed": 0,
        "papers_missing_arxiv_id": 0,
        "unique_arxiv_ids": set(),
        "papers_missing_pwc_entry": 0,  # Includes cases where pwc_entry key is missing or null/empty dict
        "pwc_entries_processed": 0,
        "pwc_missing_conference": 0,  # Added
        "pwc_missing_tasks": 0,  # Added (check if list is empty)
        "pwc_missing_datasets": 0,  # Added (check if list is empty)
        "pwc_missing_methods": 0,  # Added (check if list is empty)
        "pwc_missing_repositories": 0,  # Added (check if list is empty)
        # GitHub Repo Stats (within PWC entries)
        "total_repos_processed": 0,
        "repos_missing_url": 0,
        "repos_not_github": 0,
        "github_repos_processed": 0,
        "github_repos_missing_stars": 0,  # Value is None
        "github_repos_missing_license": 0,  # Value is None (Added)
        "github_repos_missing_language": 0,  # Value is None (Added)
    }

    logger = logging.getLogger(__name__)  # Get logger configured in main
    logger.info(f"Starting data integrity analysis for: {jsonl_filepath}")

    try:
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats["total_lines_read"] += 1
                if line_num % 1000 == 0:  # Log progress every 1000 lines
                    logger.info(f"Processed {line_num} lines...")

                # 1. Check JSON Parsing
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise TypeError("Line did not parse into a dictionary")
                    stats["records_valid_json"] += 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Line {line_num}: Failed to parse JSON or not a dict. Error: {e}"
                    )
                    stats["json_parse_errors"] += 1
                    continue  # Skip to next line if basic structure is wrong

                # 2. Check Core HF Model ID
                hf_model_id = data.get("hf_model_id")
                if not hf_model_id or not isinstance(hf_model_id, str):
                    logger.warning(
                        f"Line {line_num}: Record missing or invalid 'hf_model_id'. Value: {hf_model_id}"
                    )
                    stats["missing_hf_id"] += 1
                    continue  # Cannot proceed without a valid ID

                stats["unique_hf_model_ids"].add(hf_model_id)

                # 3. Check Other HF Model Fields
                if data.get("hf_author") is None:
                    stats["missing_hf_author"] += 1
                if data.get("hf_sha") is None:
                    stats["missing_hf_sha"] += 1
                if data.get("hf_last_modified") is None:
                    stats["missing_hf_last_modified"] += 1
                if data.get("hf_tags") is None:
                    stats["missing_hf_tags"] += 1  # Check presence
                if data.get("hf_downloads") is None:
                    stats["missing_hf_downloads"] += 1
                if data.get("hf_likes") is None:
                    stats["missing_hf_likes"] += 1

                # 4. Check README Content
                readme_value = data.get(
                    "hf_readme_content", "KEY_MISSING"
                )  # Default to distinguish missing key
                if readme_value == "KEY_MISSING":
                    stats["missing_readme_key"] += 1
                elif readme_value is None:  # Handles JSON null
                    stats["readme_is_null"] += 1
                elif (
                    isinstance(readme_value, str) and readme_value
                ):  # Check non-empty string
                    stats["readme_is_present"] += 1
                else:  # Covers empty string or unexpected types
                    stats["readme_is_null"] += (
                        1  # Treat empty string same as null for this check
                    )
                    logger.debug(
                        f"Line {line_num}, ID {hf_model_id}: README content is present but empty or invalid type."
                    )

                # 5. Check Dataset Links
                links_value = data.get("hf_dataset_links", "KEY_MISSING")
                if links_value == "KEY_MISSING":
                    stats["missing_dataset_links_key"] += 1
                elif isinstance(links_value, list):
                    if not links_value:  # Empty list
                        stats["dataset_links_is_empty"] += 1
                    else:  # Non-empty list
                        stats["dataset_links_is_present"] += 1
                        stats["total_dataset_links_found"] += len(links_value)
                        for link in links_value:
                            if isinstance(link, str) and link.startswith("http"):
                                stats["unique_dataset_links"].add(link)
                            else:
                                logger.warning(
                                    f"Line {line_num}, ID {hf_model_id}: Invalid dataset link found: {link}"
                                )
                else:  # Not a list or None/null (should ideally be list)
                    logger.warning(
                        f"Line {line_num}, ID {hf_model_id}: 'hf_dataset_links' field has unexpected type: {type(links_value)}"
                    )
                    stats["dataset_links_is_empty"] += 1  # Count as empty/invalid

                # 6. Analyze Linked Papers
                linked_papers = data.get("linked_papers", [])
                if isinstance(linked_papers, list) and linked_papers:
                    stats["models_with_linked_papers"].add(hf_model_id)

                    for paper_num, paper in enumerate(linked_papers, 1):
                        if not isinstance(paper, dict):
                            logger.warning(
                                f"Line {line_num}, ID {hf_model_id}: Invalid item in linked_papers list (item {paper_num})."
                            )
                            continue  # Skip malformed paper entries

                        stats["total_papers_processed"] += 1
                        arxiv_id_base = paper.get("arxiv_id_base")
                        if not arxiv_id_base or not isinstance(arxiv_id_base, str):
                            stats["papers_missing_arxiv_id"] += 1
                        else:
                            stats["unique_arxiv_ids"].add(arxiv_id_base)

                        # Analyze PWC Entry within the paper
                        pwc_entry = paper.get("pwc_entry")
                        if not pwc_entry or not isinstance(
                            pwc_entry, dict
                        ):  # Check key exists and value is non-empty dict
                            stats["papers_missing_pwc_entry"] += 1
                        else:
                            stats["pwc_entries_processed"] += 1
                            if pwc_entry.get("conference") is None:
                                stats["pwc_missing_conference"] += 1
                            if not pwc_entry.get("tasks"):
                                stats["pwc_missing_tasks"] += (
                                    1  # Check if list is missing or empty
                                )
                            if not pwc_entry.get("datasets"):
                                stats["pwc_missing_datasets"] += 1
                            if not pwc_entry.get("methods"):
                                stats["pwc_missing_methods"] += 1

                            # Analyze Repositories within PWC Entry
                            repositories = pwc_entry.get("repositories", [])
                            if not repositories:  # Check if list is missing or empty
                                stats["pwc_missing_repositories"] += 1
                            elif isinstance(repositories, list):
                                for repo_num, repo in enumerate(repositories, 1):
                                    if not isinstance(repo, dict):
                                        logger.warning(
                                            f"Line {line_num}, ID {hf_model_id}, Paper {paper_num}: Invalid item in repositories list (item {repo_num})."
                                        )
                                        continue  # Skip malformed repo entries

                                    stats["total_repos_processed"] += 1
                                    url = repo.get("url")
                                    if not url or not isinstance(url, str):
                                        stats["repos_missing_url"] += 1
                                        continue  # Can't check further without URL

                                    if "github.com" in url.lower():
                                        stats["github_repos_processed"] += 1
                                        if repo.get("stars") is None:
                                            stats["github_repos_missing_stars"] += 1
                                        if repo.get("license") is None:
                                            stats["github_repos_missing_license"] += 1
                                        if repo.get("language") is None:
                                            stats["github_repos_missing_language"] += 1
                                    else:
                                        stats["repos_not_github"] += 1
                            else:
                                logger.warning(
                                    f"Line {line_num}, ID {hf_model_id}, Paper {paper_num}: 'repositories' field has unexpected type: {type(repositories)}"
                                )
                                stats["pwc_missing_repositories"] += 1

    except FileNotFoundError:
        logger.error(f"Input file not found: {jsonl_filepath}")
        return None
    except IOError as e:
        logger.error(f"I/O error reading file {jsonl_filepath}: {e}")
        return None
    except Exception as e:
        logger.critical(f"Unexpected critical error during analysis: {e}")
        logger.critical(traceback.format_exc())
        return None  # Return None on critical errors

    logger.info(f"Finished analyzing {stats['total_lines_read']} lines.")
    return stats


# --- Reporting Function ---
def format_and_log_results_v2(
    stats: Optional[Dict[str, Any]], report_filepath: str
) -> None:
    """Formats the comprehensive statistics and logs them, writing to a report file."""
    logger = logging.getLogger(__name__)  # Get logger configured in main

    if not stats:
        logger.error("Analysis failed or file not found, no statistics to report.")
        return

    # --- Calculate Derived Statistics ---
    total_valid_records = stats["records_valid_json"]
    total_unique_models = len(stats["unique_hf_model_ids"])
    total_models_with_papers = len(stats["models_with_linked_papers"])
    total_unique_arxiv = len(stats["unique_arxiv_ids"])
    total_unique_datasets = len(stats["unique_dataset_links"])

    # Percentages (handle division by zero)
    perc = lambda count, total: (count / total * 100) if total > 0 else 0

    perc_readme_null = perc(stats["readme_is_null"], total_valid_records)
    perc_readme_present = perc(stats["readme_is_present"], total_valid_records)
    perc_readme_key_missing = perc(stats["missing_readme_key"], total_valid_records)

    perc_dslinks_present = perc(stats["dataset_links_is_present"], total_valid_records)
    perc_dslinks_key_missing = perc(
        stats["missing_dataset_links_key"], total_valid_records
    )

    perc_papers_missing_pwc = perc(
        stats["papers_missing_pwc_entry"], stats["total_papers_processed"]
    )
    perc_pwc_missing_conf = perc(
        stats["pwc_missing_conference"], stats["pwc_entries_processed"]
    )
    perc_pwc_missing_methods = perc(
        stats["pwc_missing_methods"], stats["pwc_entries_processed"]
    )
    # ... add percentages for other PWC fields if desired ...

    perc_gh_missing_stars = perc(
        stats["github_repos_missing_stars"], stats["github_repos_processed"]
    )
    perc_gh_missing_license = perc(
        stats["github_repos_missing_license"], stats["github_repos_processed"]
    )
    perc_gh_missing_language = perc(
        stats["github_repos_missing_language"], stats["github_repos_processed"]
    )

    # --- Prepare Report ---
    report_lines = [
        "--- Data Validation Report ---",
        f"Analyzed File: {stats['input_filepath']}",
        f"Timestamp: {datetime.now().isoformat()}",
        "=" * 40,
        f"File Summary:",
        f"  - Total lines read: {stats['total_lines_read']}",
        f"  - Lines with JSON parsing errors: {stats['json_parse_errors']}",
        f"  - Records successfully parsed as JSON: {total_valid_records}",
        "=" * 40,
        f"Hugging Face Model Records:",
        f"  - Total unique models found (based on hf_model_id): {total_unique_models}",
        f"  - Records missing 'hf_model_id': {stats['missing_hf_id']}",
        f"  - Records missing 'hf_author': {stats['missing_hf_author']} ({perc(stats['missing_hf_author'], total_valid_records):.1f}%)",
        f"  - Records missing 'hf_sha': {stats['missing_hf_sha']} ({perc(stats['missing_hf_sha'], total_valid_records):.1f}%)",
        f"  - Records missing 'hf_last_modified': {stats['missing_hf_last_modified']} ({perc(stats['missing_hf_last_modified'], total_valid_records):.1f}%)",
        f"  - Records missing 'hf_tags': {stats['missing_hf_tags']} ({perc(stats['missing_hf_tags'], total_valid_records):.1f}%)",
        f"  - Records missing 'hf_downloads': {stats['missing_hf_downloads']} ({perc(stats['missing_hf_downloads'], total_valid_records):.1f}%)",
        f"  - Records missing 'hf_likes': {stats['missing_hf_likes']} ({perc(stats['missing_hf_likes'], total_valid_records):.1f}%)",
        "=" * 40,
        f"README Content ('hf_readme_content'):",
        f"  - Records missing the key entirely: {stats['missing_readme_key']} ({perc_readme_key_missing:.1f}%)",
        f"  - Records where value is null (no README found/fetch error): {stats['readme_is_null']} ({perc_readme_null:.1f}%)",
        f"  - Records with README content present: {stats['readme_is_present']} ({perc_readme_present:.1f}%)",
        "=" * 40,
        f"Dataset Links ('hf_dataset_links'):",
        f"  - Records missing the key entirely: {stats['missing_dataset_links_key']} ({perc_dslinks_key_missing:.1f}%)",
        f"  - Records with link list present but empty: {stats['dataset_links_is_empty']}",
        f"  - Records with dataset links present: {stats['dataset_links_is_present']} ({perc_dslinks_present:.1f}%)",
        f"  - Total dataset links found across all records: {stats['total_dataset_links_found']}",
        f"  - Total unique dataset links found: {total_unique_datasets}",
        "=" * 40,
        f"Linked Papers & PWC Entries:",
        f"  - Models with at least one linked paper: {total_models_with_papers}",
        f"  - Total paper entries processed: {stats['total_papers_processed']}",
        f"  - Paper entries missing 'arxiv_id_base': {stats['papers_missing_arxiv_id']}",
        f"  - Total unique ArXiv IDs found: {total_unique_arxiv}",
        f"  - Paper entries missing 'pwc_entry' (or entry is null/empty): {stats['papers_missing_pwc_entry']} ({perc_papers_missing_pwc:.1f}%)",
        f"  - Total PWC entries processed: {stats['pwc_entries_processed']}",
        f"  - PWC entries missing 'conference': {stats['pwc_missing_conference']} ({perc_pwc_missing_conf:.1f}%)",
        f"  - PWC entries missing 'methods' (or empty list): {stats['pwc_missing_methods']} ({perc_pwc_missing_methods:.1f}%)",
        f"  - PWC entries missing 'tasks' (or empty list): {stats['pwc_missing_tasks']}",
        f"  - PWC entries missing 'datasets' (or empty list): {stats['pwc_missing_datasets']}",
        f"  - PWC entries missing 'repositories' (or empty list): {stats['pwc_missing_repositories']}",
        "=" * 40,
        f"GitHub Repositories (within PWC entries):",
        f"  - Total repository entries processed: {stats['total_repos_processed']}",
        f"  - Repository entries missing 'url': {stats['repos_missing_url']}",
        f"  - Repository entries with non-GitHub URLs: {stats['repos_not_github']}",
        f"  - Total GitHub repository entries processed: {stats['github_repos_processed']}",
        f"  - GitHub repos where 'stars' is null: {stats['github_repos_missing_stars']} ({perc_gh_missing_stars:.1f}%)",
        f"  - GitHub repos where 'license' is null: {stats['github_repos_missing_license']} ({perc_gh_missing_license:.1f}%)",
        f"  - GitHub repos where 'language' is null: {stats['github_repos_missing_language']} ({perc_gh_missing_language:.1f}%)",
        "=" * 40,
    ]

    # --- Log Results ---
    logger.info("--- Analysis Summary ---")
    for line in report_lines:
        logger.info(line)
    logger.info("--- End of Summary ---")

    # --- Write Report File ---
    try:
        report_dir = os.path.dirname(report_filepath)
        os.makedirs(report_dir, exist_ok=True)
        with open(report_filepath, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")
        logger.info(f"Full report written to: {report_filepath}")
    except IOError as e:
        logger.error(f"Failed to write report file to {report_filepath}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the integrity and completeness of the collected AIGraphX data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_JSONL,
        help=f"Path to the input JSONL data file (default: {DEFAULT_INPUT_JSONL})",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_FILE,
        help=f"Path to the output validation report file (default: {DEFAULT_REPORT_FILE})",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=DEFAULT_LOG_FILE,
        help=f"Path to the detailed log file (default: {DEFAULT_LOG_FILE})",
    )
    args = parser.parse_args()

    # Setup logging using the specified log file path
    logger = setup_logging(args.log)

    logger.info("Starting data validation script...")
    analysis_results = analyze_data_integrity_v2(args.input)
    format_and_log_results_v2(analysis_results, args.report)
    logger.info("Data validation script finished.")
