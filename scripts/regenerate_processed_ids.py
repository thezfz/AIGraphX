import json
import os
import logging
from typing import Set, Tuple
import traceback  # <--- Import traceback here

# --- Configuration ---
SOURCE_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"  # 指向你的 5044 条数据文件
PROCESSED_IDS_FILE = "data/processed_hf_model_ids.txt"  # 要重新生成的文件
LOG_DIR = "logs"

# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [RegenIDs] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/regenerate_ids.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def generate_id_file_from_jsonl(
    jsonl_filepath: str, output_id_filepath: str
) -> Tuple[bool, int, int]:
    """
    Reads a JSONL file, extracts unique 'hf_model_id's, and writes them to a new ID file.
    """
    processed_ids: Set[str] = set()
    lines_read = 0
    errors = 0

    logger.info(f"Reading model IDs from: {jsonl_filepath}")

    try:
        with open(jsonl_filepath, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                lines_read += 1
                try:
                    data = json.loads(line)
                    model_id = data.get("hf_model_id")
                    if model_id:
                        processed_ids.add(model_id)
                    else:
                        logger.warning(
                            f"Line {line_num}: Record missing 'hf_model_id'."
                        )
                        errors += 1
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_num}: Failed to parse JSON.")
                    errors += 1
                except Exception as e:
                    logger.error(
                        f"Line {line_num}: Unexpected error processing line: {e}"
                    )
                    errors += 1

        logger.info(
            f"Finished reading {lines_read} lines. Found {len(processed_ids)} unique model IDs. Encountered {errors} errors."
        )

        # Write the extracted IDs to the output file
        logger.info(f"Writing {len(processed_ids)} unique IDs to: {output_id_filepath}")
        os.makedirs(os.path.dirname(output_id_filepath), exist_ok=True)
        with open(output_id_filepath, "w", encoding="utf-8") as outfile:
            for model_id in sorted(list(processed_ids)):
                outfile.write(model_id + "\n")
        logger.info("ID tracking file regenerated successfully.")
        return True, lines_read, len(processed_ids)

    except FileNotFoundError:
        logger.error(f"Source file not found: {jsonl_filepath}")
        return False, lines_read, len(processed_ids)
    except IOError as e:
        logger.error(f"I/O error: {e}")
        return False, lines_read, len(processed_ids)
    except Exception as e:
        logger.critical(f"Unexpected critical error during ID regeneration: {e}")
        logger.critical(traceback.format_exc())
        return False, lines_read, len(processed_ids)


if __name__ == "__main__":
    logger.info("Starting script to regenerate processed IDs file...")

    success, total_lines, unique_ids = generate_id_file_from_jsonl(
        SOURCE_JSONL_FILE, PROCESSED_IDS_FILE
    )

    if success:
        logger.info(
            f"Successfully regenerated '{PROCESSED_IDS_FILE}' with {unique_ids} unique IDs from {total_lines} lines in '{SOURCE_JSONL_FILE}'."
        )
    else:
        logger.error("Failed to regenerate the processed IDs file.")

    logger.info("Regeneration script finished.")
