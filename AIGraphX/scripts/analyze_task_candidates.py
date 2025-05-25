import asyncio
import json
import os
import re
from collections import Counter
from typing import List, Set, Any, Optional

import asyncpg
from dotenv import load_dotenv

# --- 从 neo4j_repo.py 复制过来的常量 ---
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

KNOWN_LIBRARY_NAMES = frozenset([
    "safetensors",
    "diffusers",
    "transformers", # 添加常见的库,
    "pytorch",
    "tensorflow",
    "jax",
    "flax",
    "onnx",
    "coreml",
    "keras"
    # 可以根据需要添加更多已知的库名
])

# 扩展的停用词和模式，用于更精确地过滤非任务标签
ADDITIONAL_STOP_WORDS = frozenset([
    "model", "models", "dataset", "datasets", "library", "libraries",
    "tool", "tools", "framework", "frameworks", "community",
    "research", "paper", "papers", "arxiv", "huggingface", "hf",
    "evaluation", "benchmark", "leaderboard",
    "llm", "nlp", "cv", "deep-learning", "machine-learning",
    "multimodal", # 经常作为类别而非具体任务,
    "license", # 确保 license 及其变体被过滤
    "generic", "other", "example", "demo", "tutorial",
    "experimental", "alpha", "beta",
    "local-files-only", "allow-patterns", "ignore-patterns",
    "not-for-all-audiences",
    "template", "structured-data" # 结构化数据通常指数据类型
])

# 模式用于识别和排除非任务标签，例如 license:mit, region:us
# 增加了对版本号 (e.g., v1.0, 2.3.1), 尺寸 (e.g., 7b, 13b, 70b), 格式 (e.g. gguf, gptq)
# 以及其他常见的元数据模式的过滤
REGEX_FILTERS = [
    re.compile(r"^\d+$"),  # 纯数字
    re.compile(r"^\d+k$"), # e.g., 2k
    re.compile(r"^\d+m$"), # e.g., 2m
    re.compile(r"^\d+b$"), # e.g., 7b, 70b (模型参数量)
    re.compile(r"^v\d+(\.\d+)*(-\w+)?$"), # 版本号, e.g., v1.0, v2.0.1, v0.1.0-alpha
    re.compile(r"^[a-zA-Z]{2,3}-\d{2,}$"), # e.g., xx-00
    re.compile(r"^[a-zA-Z]{2,3}-[a-zA-Z]{2,3}$"), # e.g., en-es
    re.compile(r"^(mit|apache|gpl|bsd|cc0)(-\d+(\.\d+)?)?$", re.IGNORECASE), # 许可证
    re.compile(r"^(license|region|language|lang|dataset|author|model_name|pipeline|tag|framework|library|type|format|arch|architecture|size|version|sha|commit):", re.IGNORECASE),
    re.compile(r"^(gguf|gptq|awq|ggml|exl2|fp16|bf16|int8|4bit|8bit)$", re.IGNORECASE), # 模型格式/量化
    re.compile(r"^[a-f0-9]{7,}$", re.IGNORECASE), # 可能是 commit hash 或短ID
    re.compile(r".*eval.*", re.IGNORECASE), # 包含 eval 的词
    re.compile(r".*benchmark.*", re.IGNORECASE), # 包含 benchmark 的词
    re.compile(r"^(peft|lora|qlora)$", re.IGNORECASE), # 训练技术
    re.compile(r"^(sagemaker|autotrain|inference-endpoints|text-generation-inference)$", re.IGNORECASE), # AWS 或 HF 特定服务/工具
    re.compile(r"^generated_from_trainer", re.IGNORECASE), # 标记
    re.compile(r"^text-generation$", re.IGNORECASE), # 有时 pipeline_tag 就是 "text-generation",需要更细化
    re.compile(r"^stable-diffusion-xl.*", re.IGNORECASE), # sd xl variations
    re.compile(r"^sdxl-.*", re.IGNORECASE), # sdxl variations
    re.compile(r".*\d+d$", re.IGNORECASE), # e.g. 2d, 3d - could be task related, but often not.
    re.compile(r"^tpu$", re.IGNORECASE),
    re.compile(r"^gpu$", re.IGNORECASE),
    re.compile(r"^coreml$", re.IGNORECASE), # 库名已在KNOWN_LIBRARY_NAMES，但也加到这里
    re.compile(r"^onnx$", re.IGNORECASE),
    re.compile(r"^(llama|mistral|gemma|phi|bert|gpt|t5|bart|roberta|xlnet|albert|distilbert|opt|bloom|falcon|mpt)-?(\d*b)?(-?\w+)*$", re.IGNORECASE) # 常见模型家族名称
]
# --- End Constants ---

def is_potential_task(tag: str) -> bool:
    """
    Checks if a tag is a potential task, after basic normalization.
    Filters out known language codes, library names, additional stop words, and regex patterns.
    """
    if not tag or not isinstance(tag, str):
        return False

    normalized_tag = tag.lower().strip().replace("_", "-") # 统一为小写，去除首尾空格，下划线转横杠

    if not normalized_tag: # 如果处理后为空字符串
        return False

    if normalized_tag in KNOWN_LANGUAGE_CODES:
        return False
    if normalized_tag in KNOWN_LIBRARY_NAMES:
        return False
    if normalized_tag in ADDITIONAL_STOP_WORDS:
        return False

    for pattern in REGEX_FILTERS:
        if pattern.match(normalized_tag):
            return False
    
    # 避免非常短的标签 (通常不是任务) 或过长的标签
    if len(normalized_tag) < 3 or len(normalized_tag) > 50: # 调整长度限制
        return False

    return True

async def get_db_connection() -> Optional[asyncpg.Connection]:
    """Establishes and returns an asyncpg database connection."""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    try:
        host = os.getenv("POSTGRES_HOST")
        port_str = os.getenv("POSTGRES_PORT") # Corrected environment variable name

        if host == "postgres":
            print(f"Script detected POSTGRES_HOST='postgres'. Assuming local run and overriding to '127.0.0.1' for this script.")
            host = "127.0.0.1"

        if port_str is None:
            print("Error: POSTGRES_PORT environment variable not set or is empty.")
            return None
        
        try:
            port = int(port_str)
        except ValueError:
            print(f"Error: POSTGRES_PORT '{port_str}' is not a valid integer.")
            return None

        conn = await asyncpg.connect(
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB"),
            host=host, # Use potentially overridden host
            port=port  # Use corrected and converted port
        )
        print("Successfully connected to PostgreSQL.")
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

async def fetch_task_related_data(conn: asyncpg.Connection) -> List[asyncpg.Record]:
    """Fetches hf_tags, hf_pipeline_tag, and hf_readme_tasks from hf_models table."""
    if not conn:
        return []
    try:
        query = """
        SELECT hf_model_id, hf_tags, hf_pipeline_tag, hf_readme_tasks
        FROM hf_models;
        """
        # WHERE hf_readme_tasks IS NOT NULL AND hf_readme_tasks != \'[]\'; -- 可以只分析有readme task的模型
        records: List[asyncpg.Record] = await conn.fetch(query)
        print(f"Fetched {len(records)} records from hf_models.")
        return records
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def extract_tasks_from_record(record: asyncpg.Record) -> Set[str]:
    """Extracts and normalizes potential task strings from a single record."""
    potential_tasks = set()

    # 1. From hf_pipeline_tag
    pipeline_tag = record.get("hf_pipeline_tag")
    if pipeline_tag and isinstance(pipeline_tag, str):
        # pipeline_tag 很多时候已经是规范的任务了, 但还是需要清洗
        normalized_pipeline_tag = pipeline_tag.lower().strip().replace("_", "-")
        potential_tasks.add(normalized_pipeline_tag)

    # 2. From hf_tags (JSONB array of strings)
    hf_tags_json = record.get("hf_tags")
    if hf_tags_json:
        try:
            # hf_tags 可能是字符串形式的JSON,也可能是None
            hf_tags_list = json.loads(hf_tags_json) if isinstance(hf_tags_json, str) else hf_tags_json
            if isinstance(hf_tags_list, list):
                for tag_item in hf_tags_list: # Renamed tag to tag_item to avoid conflict
                    if isinstance(tag_item, str):
                        potential_tasks.add(tag_item.lower().strip().replace("_", "-"))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse hf_tags JSON for model {record.get('hf_model_id')}: {hf_tags_json}")

    # 3. From hf_readme_tasks (JSONB array of strings, already extracted from YAML)
    hf_readme_tasks_json = record.get("hf_readme_tasks")
    if hf_readme_tasks_json:
        try:
            # hf_readme_tasks 可能是字符串形式的JSON,也可能是None
            hf_readme_tasks_list = json.loads(hf_readme_tasks_json) if isinstance(hf_readme_tasks_json, str) else hf_readme_tasks_json
            if isinstance(hf_readme_tasks_list, list):
                for task_item in hf_readme_tasks_list:
                    if isinstance(task_item, str):
                        potential_tasks.add(task_item.lower().strip().replace("_", "-"))
                    # 有时可能是 {"type": "task_name"} 的结构, 但我们当前存的是字符串列表
        except json.JSONDecodeError:
             print(f"Warning: Could not parse hf_readme_tasks JSON for model {record.get('hf_model_id')}: {hf_readme_tasks_json}")
    
    return potential_tasks

async def main() -> None:
    conn = await get_db_connection()
    if not conn:
        return

    records = await fetch_task_related_data(conn)
    if not records:
        print("No data fetched, exiting.")
        await conn.close()
        return

    all_candidate_tasks: Counter[str] = Counter()
    filtered_task_counts: Counter[str] = Counter()
    
    total_models_processed = 0

    for record in records:
        total_models_processed += 1
        raw_tasks_from_record = extract_tasks_from_record(record)
        
        for task_str in raw_tasks_from_record:
            all_candidate_tasks[task_str] +=1 # 统计所有提取出来的原始字符串
            if is_potential_task(task_str):
                # 对通过过滤器的任务字符串进行最终的规范化和计数
                # (is_potential_task内部已做 .lower().strip().replace("_", "-"))
                filtered_task_counts[task_str.lower().strip().replace("_", "-")] += 1
    
    await conn.close()
    print("\\n--- Task Analysis Results ---")
    print(f"Total models processed: {total_models_processed}")
    
    # print("\\n--- All Extracted Strings (Before Filtering) ---")
    # for task, count in all_candidate_tasks.most_common(100): # 显示最常见的100个原始提取项
    #     print(f"{task}: {count}")

    print("\\n--- Potential Task Strings (After Filtering) ---")
    if not filtered_task_counts:
        print("No potential task strings found after filtering.")
    else:
        # 打印出现频率大于1的,或者总数少于50时打印全部
        min_freq_to_display = 2 if len(filtered_task_counts) > 50 else 1
        
        print(f"Displaying tasks with frequency >= {min_freq_to_display} (Total unique: {len(filtered_task_counts)})")
        
        sorted_filtered_tasks = filtered_task_counts.most_common()
        
        count_displayed = 0
        for task, count in sorted_filtered_tasks:
            if count >= min_freq_to_display:
                print(f"{task}: {count}")
                count_displayed+=1
        
        if count_displayed == 0 and len(sorted_filtered_tasks) > 0:
             print(f"No tasks met frequency >= {min_freq_to_display}. Top tasks are:")
             for task, count in sorted_filtered_tasks[:20]: # Display top 20 if nothing meets criteria
                 print(f"{task}: {count}")

    print("\\nScript finished.")

if __name__ == "__main__":
    asyncio.run(main())
