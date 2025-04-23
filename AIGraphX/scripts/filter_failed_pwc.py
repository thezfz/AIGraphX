import json
import logging
import os
import argparse
from typing import Dict, Any, Optional

# --- 日志记录设置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def is_pwc_entry_problematic(pwc_entry: Optional[Dict[str, Any]]) -> bool:
    """
    检查 PWC 条目是否存在问题（空、None 或包含错误标记）。

    Args:
        pwc_entry: 从 JSON 数据中获取的 pwc_entry 字典或 None。

    Returns:
        如果 pwc_entry 表示获取失败或未获取，则返回 True，否则返回 False。
    """
    if pwc_entry is None:
        return True  # 未找到 PWC 条目
    if not pwc_entry:  # 处理空字典 {} 的情况
        return True
    if isinstance(pwc_entry, dict) and "error" in pwc_entry:
        return True  # 明确标记了错误
    return False


def filter_records(input_path: str, output_path: str) -> None:
    """
    读取输入的 JSONL 文件，过滤掉包含有问题 PWC 条目的记录，
    并将保留的记录写入输出文件。

    Args:
        input_path: 输入的 JSONL 文件路径。
        output_path: 输出的过滤后的 JSONL 文件路径。
    """
    total_lines_read = 0
    records_kept = 0
    records_filtered = 0

    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
        ):
            logger.info(f"开始处理文件: {input_path}")
            for line in infile:
                total_lines_read += 1
                if total_lines_read % 1000 == 0:
                    logger.info(f"已处理 {total_lines_read} 行...")

                try:
                    model_record = json.loads(line.strip())
                except json.JSONDecodeError:
                    logger.warning(
                        f"跳过第 {total_lines_read} 行：无法解析 JSON - {line.strip()[:100]}..."
                    )
                    continue

                keep_record = True  # 默认保留记录
                linked_papers = model_record.get("linked_papers")

                if linked_papers and isinstance(linked_papers, list):
                    for paper in linked_papers:
                        if isinstance(paper, dict) and paper.get("arxiv_id_base"):
                            # 检查存在 arxiv_id 的论文，其 pwc_entry 是否有问题
                            pwc_entry = paper.get("pwc_entry")
                            if is_pwc_entry_problematic(pwc_entry):
                                keep_record = False  # 发现问题，标记整个记录为过滤对象
                                logger.debug(
                                    f"过滤记录 (模型 ID: {model_record.get('hf_model_id', '未知')})，原因：论文 {paper.get('arxiv_id_base')} 的 PWC 条目有问题。"
                                )
                                break  # 无需再检查此记录的其他论文

                if keep_record:
                    outfile.write(line)  # 保留的记录写入新文件
                    records_kept += 1
                else:
                    records_filtered += 1

        logger.info("文件处理完成。")
        logger.info(f"总共读取行数: {total_lines_read}")
        logger.info(f"保留的记录数: {records_kept}")
        logger.info(f"过滤掉的记录数: {records_filtered}")

    except FileNotFoundError:
        logger.error(f"错误：输入文件未找到 - {input_path}")
    except IOError as e:
        logger.error(f"读写文件时发生错误: {e}")
    except Exception as e:
        logger.error(f"处理过程中发生意外错误: {e}", exc_info=True)


if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="过滤 aigraphx_knowledge_data.jsonl 文件，移除包含获取失败的 PWC 条目的模型记录。"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/aigraphx_knowledge_data.jsonl",
        help="输入的原始 JSONL 文件路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/aigraphx_knowledge_data_filtered.jsonl",
        help="输出的过滤后的 JSONL 文件路径。",
    )
    args = parser.parse_args()

    filter_records(args.input, args.output)
