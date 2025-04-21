# 检查数据收集脚本收集的数据文件中是否存在重复的hf_model_id，并删除重复的行
import json
import os
import traceback
from collections import Counter
import logging
from typing import Dict, Set, Tuple  # <--- Added Tuple
import shutil  # For safer file replacement

# --- Configuration ---
OUTPUT_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"
TEMP_OUTPUT_FILE = "data/aigraphx_knowledge_data.jsonl.tmp"
# DUPLICATE_REPORT_FILE = "logs/duplicate_models_report.txt" # Reporting duplicates separately is less relevant now
LOG_DIR = "logs"

# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [DeduplicateScript] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "logs/deduplicate_script.log", mode="w", encoding="utf-8"
        ),  # Overwrite log each time
    ],
)
logger = logging.getLogger(__name__)


def deduplicate_jsonl_by_key(
    input_filepath: str, output_filepath: str, key_field: str
) -> Tuple[bool, int, int, int]:
    """
    Reads a JSONL file, keeps only the first occurrence of each unique value
    for the specified key_field, and writes the unique lines to an output file.

    Args:
        input_filepath (str): Path to the input JSONL file.
        output_filepath (str): Path to the temporary output file for unique lines.
        key_field (str): The JSON key field to use for deduplication (e.g., 'hf_model_id').

    Returns:
        Tuple[bool, int, int, int]: A tuple containing:
            - bool: True if deduplication completed successfully, False otherwise.
            - int: Total lines read from the input file.
            - int: Total unique lines written to the output file.
            - int: Total duplicate lines skipped.
    """
    seen_keys: Set[str] = set()
    lines_read = 0
    lines_written = 0
    duplicates_skipped = 0
    parse_errors = 0

    logger.info(f"Starting deduplication for key '{key_field}' in: {input_filepath}")
    logger.info(f"Writing unique lines to temporary file: {output_filepath}")

    try:
        with (
            open(input_filepath, "r", encoding="utf-8") as infile,
            open(output_filepath, "w", encoding="utf-8") as outfile,
        ):
            for line_num, line in enumerate(infile, 1):
                lines_read += 1
                try:
                    data = json.loads(line)
                    key_value = data.get(key_field)

                    if key_value:
                        if key_value not in seen_keys:
                            seen_keys.add(key_value)
                            outfile.write(line)  # Write the original line
                            lines_written += 1
                        else:
                            duplicates_skipped += 1
                            logger.debug(
                                f"Line {line_num}: Skipping duplicate key '{key_value}'."
                            )
                    else:
                        logger.warning(
                            f"Line {line_num}: Record missing or has empty key '{key_field}'. Skipping line."
                        )
                        # Decide if you want to write lines with missing keys? No, skipping.
                        parse_errors += 1  # Treat missing key as an issue to report

                except json.JSONDecodeError:
                    logger.warning(
                        f"Line {line_num}: Failed to parse JSON. Skipping line."
                    )
                    parse_errors += 1
                except Exception as e:
                    logger.error(
                        f"Line {line_num}: Unexpected error processing line: {e}. Skipping line."
                    )
                    parse_errors += 1

        logger.info(
            f"Finished processing input file. Read: {lines_read}, Written (Unique): {lines_written}, Skipped (Duplicates): {duplicates_skipped}"
        )
        if parse_errors > 0:
            logger.warning(
                f"Encountered {parse_errors} lines with parsing errors or missing keys."
            )
        return True, lines_read, lines_written, duplicates_skipped

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filepath}")
        # Clean up temp file if it was created
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        return False, lines_read, lines_written, duplicates_skipped
    except IOError as e:
        logger.error(f"I/O error during deduplication: {e}")
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        return False, lines_read, lines_written, duplicates_skipped
    except Exception as e:
        logger.critical(f"Unexpected critical error during deduplication: {e}")
        logger.critical(traceback.format_exc())
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        return False, lines_read, lines_written, duplicates_skipped


if __name__ == "__main__":
    logger.info("Starting deduplication script...")

    success, lines_read, lines_written, duplicates_skipped = deduplicate_jsonl_by_key(
        OUTPUT_JSONL_FILE,
        TEMP_OUTPUT_FILE,
        "hf_model_id",  # Specify the key for deduplication
    )

    if success:
        logger.info("Deduplication process completed successfully.")
        logger.info(
            f"Read: {lines_read}, Unique Written: {lines_written}, Duplicates Removed: {duplicates_skipped}"
        )

        # Replace original file with the temporary deduplicated file
        try:
            logger.info(
                f"Attempting to replace original file '{OUTPUT_JSONL_FILE}' with '{TEMP_OUTPUT_FILE}'"
            )
            # Use shutil.move for atomic replace where possible, otherwise delete and rename
            shutil.move(TEMP_OUTPUT_FILE, OUTPUT_JSONL_FILE)
            logger.info("Original file replaced successfully.")
        except OSError as e:
            logger.critical(f"CRITICAL: Failed to replace original file: {e}")
            logger.critical(
                f"The deduplicated data is safe in '{TEMP_OUTPUT_FILE}'. Please manually replace the original file."
            )
        except Exception as e:
            logger.critical(f"CRITICAL: Unexpected error during file replacement: {e}")
            logger.critical(traceback.format_exc())
            logger.critical(
                f"The deduplicated data is safe in '{TEMP_OUTPUT_FILE}'. Please manually replace the original file."
            )

    else:
        logger.error("Deduplication process failed. Original file was not modified.")
        # Ensure temp file is cleaned up if it exists from a failed run
        if os.path.exists(TEMP_OUTPUT_FILE):
            try:
                os.remove(TEMP_OUTPUT_FILE)
                logger.info(
                    f"Removed temporary file '{TEMP_OUTPUT_FILE}' after failed run."
                )
            except OSError as e:
                logger.error(
                    f"Failed to remove temporary file '{TEMP_OUTPUT_FILE}' after failed run: {e}"
                )

    logger.info("Deduplication script finished.")
