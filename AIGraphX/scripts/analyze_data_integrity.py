#!/usr/bin/env python
# -*- coding: utf-8 -*- # 指定编码为 UTF-8

# 文件作用说明：
# 该脚本用于分析和验证数据完整性，主要针对由 `enrich_data.py` 脚本生成的
# "enriched" JSONL 数据文件 (默认为 data/aigraphx_knowledge_data_enriched.jsonl)。
# 它会读取文件中的每一条记录（代表一个 AI 模型及其关联信息），检查关键字段是否存在、
# 格式是否基本正确，并统计各种指标，例如：
# - 文件整体统计（读取行数、JSON 解析错误数）
# - Hugging Face 模型字段的缺失情况（ID, 作者, SHA, 修改日期, 标签, 下载量, 点赞数）
# - README 内容的存在情况
# - 数据集链接的存在和数量
# - 关联论文和 PWC 条目的存在情况
# - PWC 条目内部字段（会议、任务、数据集、方法、代码库）的缺失情况
# - 代码库（特别是 GitHub 库）字段的缺失情况（URL, 星标数, 许可证, 语言）
#
# 脚本最终会生成一份文本格式的验证报告，并输出详细的日志信息。
# 这有助于评估数据采集和丰富流程的质量，发现潜在的数据问题。
#
# 交互对象：
# - 输入：JSONL 数据文件 (默认为 data/aigraphx_knowledge_data_enriched.jsonl)。
# - 输出：
#   - 文本格式的验证报告文件 (默认为 logs/data_validation_report.txt)。
#   - 详细的日志文件 (默认为 logs/data_validation.log)。
#   - 日志信息也会输出到控制台。
# - 配置：可以通过命令行参数指定输入、报告和日志文件的路径。

# 导入标准库
import json  # 用于解析 JSON 数据
import os  # 用于文件和目录操作 (路径处理, 创建目录)
import logging  # 用于日志记录
from collections import (
    Counter,
    defaultdict,
)  # Counter 用于计数，defaultdict 用于创建默认值的字典 (此处未使用，但常用)
from typing import Dict, Set, List, Any, Optional  # 类型提示
import traceback  # 用于获取详细的错误堆栈信息
from datetime import datetime  # 用于在报告中添加时间戳
import argparse  # 用于解析命令行参数，使脚本更灵活

# --- 配置常量 ---
# 默认的输入 JSONL 文件路径（假设是 enrich 脚本处理后的文件）
DEFAULT_INPUT_JSONL = "data/aigraphx_knowledge_data.jsonl"
# 默认的输出报告文件路径
DEFAULT_REPORT_FILE = "logs/data_validation_report.txt"
# 默认的日志文件路径
DEFAULT_LOG_FILE = "logs/data_validation.log"


# --- 日志设置函数 ---
def setup_logging(log_filepath: str) -> logging.Logger:
    """
    配置脚本的日志记录。

    Args:
        log_filepath (str): 日志文件的完整路径。

    Returns:
        logging.Logger: 配置好的日志记录器实例。
    """
    # 获取日志文件所在的目录
    log_dir = os.path.dirname(log_filepath)
    # 创建日志目录 (如果不存在)
    os.makedirs(log_dir, exist_ok=True)

    # 获取根日志记录器
    root_logger = logging.getLogger()
    # 移除所有已存在的处理器，防止在脚本重复运行时（例如在交互式环境中）重复添加处理器导致日志重复输出
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 配置日志记录的基本设置
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO (忽略 DEBUG)
        format="%(asctime)s - %(levelname)s - [DataValidation] %(message)s",  # 日志格式
        handlers=[  # 日志处理器列表
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(
                log_filepath, mode="w", encoding="utf-8"
            ),  # 输出到文件，模式为 'w' (覆盖)，编码为 utf-8
        ],
    )
    # 返回配置好的、名为当前模块 (__name__) 的日志记录器
    return logging.getLogger(__name__)


# --- 主要分析函数 ---
def analyze_data_integrity_v2(jsonl_filepath: str) -> Optional[Dict[str, Any]]:
    """
    读取 JSONL 数据文件，执行全面的完整性检查，并计算统计信息（包括新字段）。

    Args:
        jsonl_filepath (str): 输入的 JSONL 文件路径。

    Returns:
        Optional[Dict[str, Any]]: 包含分析结果的字典。如果文件未找到或发生严重错误，则返回 None。
    """
    # 初始化用于存储统计结果的字典
    stats: Dict[str, Any] = {
        # 文件整体统计
        "input_filepath": jsonl_filepath,  # 输入文件路径
        "total_lines_read": 0,  # 读取的总行数
        "json_parse_errors": 0,  # JSON 解析失败的行数
        "records_valid_json": 0,  # 成功解析为 JSON 字典的记录数
        # Hugging Face 模型层面统计
        "missing_hf_id": 0,  # 缺少 hf_model_id 的记录数
        "unique_hf_model_ids": set(),  # 存储唯一的 hf_model_id (用于计数)
        "missing_hf_author": 0,  # 缺少 hf_author 的记录数
        "missing_hf_sha": 0,  # 缺少 hf_sha 的记录数
        "missing_hf_last_modified": 0,  # 缺少 hf_last_modified 的记录数
        "missing_hf_tags": 0,  # 缺少 hf_tags 的记录数 (检查键是否存在)
        "missing_hf_downloads": 0,  # 缺少 hf_downloads 的记录数
        "missing_hf_likes": 0,  # 缺少 hf_likes 的记录数
        # README 内容统计
        "missing_readme_key": 0,  # 记录中完全缺少 'hf_readme_content' 键的数量
        "readme_is_null": 0,  # 'hf_readme_content' 键存在，但值为 null/None 或空字符串
        "readme_is_present": 0,  # 'hf_readme_content' 键存在，且值为非空字符串
        # 数据集链接统计
        "missing_dataset_links_key": 0,  # 记录中完全缺少 'hf_dataset_links' 键的数量
        "dataset_links_is_empty": 0,  # 'hf_dataset_links' 键存在，但值为空列表或无效类型
        "dataset_links_is_present": 0,  # 'hf_dataset_links' 键存在，且值为非空列表
        "total_dataset_links_found": 0,  # 所有记录中找到的数据集链接总数
        "unique_dataset_links": set(),  # 存储唯一的数据集链接 URL (用于计数)
        # 关联论文和 PWC 统计
        "models_with_linked_papers": set(),  # 存储至少有一个有效关联论文的 hf_model_id
        "total_papers_processed": 0,  # 处理的关联论文条目总数 (嵌套在模型记录中)
        "papers_missing_arxiv_id": 0,  # 关联论文条目中缺少 'arxiv_id_base' 的数量
        "unique_arxiv_ids": set(),  # 存储唯一的 arxiv_id_base (用于计数)
        "papers_missing_pwc_entry": 0,  # 关联论文条目中缺少 'pwc_entry' 或其值无效 (非字典) 的数量
        "pwc_entries_processed": 0,  # 成功处理的有效 'pwc_entry' 字典数量
        "pwc_missing_conference": 0,  # PWC 条目中缺少 'conference' 键的数量
        "pwc_missing_tasks": 0,  # PWC 条目中缺少 'tasks' 或其值为空列表的数量
        "pwc_missing_datasets": 0,  # PWC 条目中缺少 'datasets' 或其值为空列表的数量
        "pwc_missing_methods": 0,  # PWC 条目中缺少 'methods' 或其值为空列表的数量
        "pwc_missing_repositories": 0,  # PWC 条目中缺少 'repositories' 或其值为空列表的数量
        # GitHub 代码库统计 (嵌套在 PWC 条目中)
        "total_repos_processed": 0,  # 处理的代码库条目总数
        "repos_missing_url": 0,  # 代码库条目缺少 'url' 的数量
        "repos_not_github": 0,  # 代码库 URL 不是 GitHub 链接的数量
        "github_repos_processed": 0,  # 处理的 GitHub 代码库条目数量
        "github_repos_missing_stars": 0,  # GitHub 代码库条目中 'stars' 值为 None 的数量
        "github_repos_missing_license": 0,  # GitHub 代码库条目中 'license' 值为 None 的数量
        "github_repos_missing_language": 0,  # GitHub 代码库条目中 'language' 值为 None 的数量
    }

    # 获取已配置的日志记录器
    logger = logging.getLogger(__name__)
    logger.info(f"开始对文件进行数据完整性分析: {jsonl_filepath}")

    try:
        # 以只读、UTF-8 编码打开 JSONL 文件
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            # 逐行读取文件，使用 enumerate 获取行号 (从 1 开始)
            for line_num, line in enumerate(f, 1):
                # 增加总行数计数器
                stats["total_lines_read"] += 1
                # 每处理 1000 行打印一次进度日志
                if line_num % 1000 == 0:
                    logger.info(f"已处理 {line_num} 行...")

                # 1. 检查 JSON 解析是否成功
                try:
                    # 解析当前行的 JSON 字符串为 Python 对象
                    data = json.loads(line)
                    # 确保解析结果是一个字典
                    if not isinstance(data, dict):
                        raise TypeError("当前行未能解析为字典类型")
                    # 增加有效 JSON 记录计数器
                    stats["records_valid_json"] += 1
                except (json.JSONDecodeError, TypeError) as e:
                    # 如果解析失败或结果不是字典
                    logger.warning(
                        f"第 {line_num} 行: JSON 解析失败或结果非字典。错误: {e}"
                    )
                    # 增加 JSON 解析错误计数器
                    stats["json_parse_errors"] += 1
                    continue  # 跳过此行，处理下一行

                # 2. 检查核心的 Hugging Face 模型 ID
                hf_model_id = data.get("hf_model_id")  # 使用 .get() 安全获取
                # 检查 ID 是否存在、非空且为字符串
                if not hf_model_id or not isinstance(hf_model_id, str):
                    logger.warning(
                        f"第 {line_num} 行: 记录缺少有效的 'hf_model_id'。值为: {hf_model_id}"
                    )
                    stats["missing_hf_id"] += 1
                    continue  # 没有有效 ID，无法继续处理此记录

                # 将有效的模型 ID 添加到集合中，用于后续计算唯一模型数量
                stats["unique_hf_model_ids"].add(hf_model_id)

                # 3. 检查其他 Hugging Face 模型字段是否存在 (值为 None 也算缺失)
                if data.get("hf_author") is None:
                    stats["missing_hf_author"] += 1
                if data.get("hf_sha") is None:
                    stats["missing_hf_sha"] += 1
                if data.get("hf_last_modified") is None:
                    stats["missing_hf_last_modified"] += 1
                if data.get("hf_tags") is None:  # 检查键是否存在 (或值为 None)
                    stats["missing_hf_tags"] += 1
                if data.get("hf_downloads") is None:
                    stats["missing_hf_downloads"] += 1
                if data.get("hf_likes") is None:
                    stats["missing_hf_likes"] += 1

                # 4. 检查 README 内容字段 ('hf_readme_content')
                # 使用特殊默认值 "KEY_MISSING" 来区分键不存在和值为 None 的情况
                readme_value = data.get("hf_readme_content", "KEY_MISSING")
                if readme_value == "KEY_MISSING":
                    # 如果键完全不存在
                    stats["missing_readme_key"] += 1
                elif readme_value is None:  # 如果键存在但值为 null
                    stats["readme_is_null"] += 1
                elif (
                    isinstance(readme_value, str) and readme_value
                ):  # 如果是字符串且非空
                    stats["readme_is_present"] += 1
                else:  # 包括空字符串 "" 或其他非预期类型
                    # 将空字符串视为与 null/None 等同（无有效内容）
                    stats["readme_is_null"] += 1
                    logger.debug(
                        f"第 {line_num} 行, ID {hf_model_id}: README 内容存在但为空或类型无效。"
                    )

                # 5. 检查数据集链接字段 ('hf_dataset_links')
                links_value = data.get("hf_dataset_links", "KEY_MISSING")
                if links_value == "KEY_MISSING":
                    stats["missing_dataset_links_key"] += 1
                elif isinstance(links_value, list):  # 检查是否为列表
                    if not links_value:  # 如果是空列表 []
                        stats["dataset_links_is_empty"] += 1
                    else:  # 如果是非空列表
                        stats["dataset_links_is_present"] += 1
                        # 累加找到的链接总数
                        stats["total_dataset_links_found"] += len(links_value)
                        # 遍历链接，添加到唯一链接集合中
                        for link in links_value:
                            # 基本验证：是字符串且以 http 开头
                            if isinstance(link, str) and link.startswith("http"):
                                stats["unique_dataset_links"].add(link)
                            else:
                                logger.warning(
                                    f"第 {line_num} 行, ID {hf_model_id}: 发现无效的数据集链接: {link}"
                                )
                else:  # 如果值不是列表 (可能是 None 或其他类型)
                    logger.warning(
                        f"第 {line_num} 行, ID {hf_model_id}: 'hf_dataset_links' 字段类型非预期: {type(links_value)}"
                    )
                    # 将非列表或 None 的情况视为无效/空
                    stats["dataset_links_is_empty"] += 1

                # 6. 分析关联的论文 ('linked_papers')
                linked_papers = data.get("linked_papers", [])  # 安全获取，默认为空列表
                # 检查是否为列表且非空
                if isinstance(linked_papers, list) and linked_papers:
                    # 如果包含有效的论文列表，将模型 ID 加入到 "有论文的模型" 集合中
                    stats["models_with_linked_papers"].add(hf_model_id)

                    # 遍历模型关联的每一篇论文
                    for paper_num, paper in enumerate(linked_papers, 1):
                        # 确保论文条目本身是字典
                        if not isinstance(paper, dict):
                            logger.warning(
                                f"第 {line_num} 行, ID {hf_model_id}: linked_papers 列表中的第 {paper_num} 项无效 (非字典)。"
                            )
                            continue  # 跳过格式错误的论文条目

                        # 增加处理的论文总数计数器
                        stats["total_papers_processed"] += 1
                        # 检查 ArXiv ID 是否存在且有效
                        arxiv_id_base = paper.get("arxiv_id_base")
                        if not arxiv_id_base or not isinstance(arxiv_id_base, str):
                            stats["papers_missing_arxiv_id"] += 1
                        else:
                            # 将有效的 ArXiv ID 添加到唯一 ID 集合
                            stats["unique_arxiv_ids"].add(arxiv_id_base)

                        # 分析论文关联的 PWC 条目 ('pwc_entry')
                        pwc_entry = paper.get("pwc_entry")
                        # 检查 pwc_entry 是否存在且是一个非空字典
                        if not pwc_entry or not isinstance(pwc_entry, dict):
                            stats["papers_missing_pwc_entry"] += 1
                        else:
                            # 如果 pwc_entry 有效，增加处理计数器
                            stats["pwc_entries_processed"] += 1
                            # 检查 PWC 条目内部字段是否存在 (值为 None 也算缺失)
                            if pwc_entry.get("conference") is None:
                                stats["pwc_missing_conference"] += 1
                            # 检查列表字段是否存在且非空
                            if not pwc_entry.get(
                                "tasks"
                            ):  # .get() 返回 None 或空列表 [] 都会评估为 False
                                stats["pwc_missing_tasks"] += 1
                            if not pwc_entry.get(
                                "datasets"
                            ):  # PWC 数据中的字段名可能是 datasets_used 或 datasets
                                stats["pwc_missing_datasets"] += 1
                            if not pwc_entry.get("methods"):
                                stats["pwc_missing_methods"] += 1

                            # 分析 PWC 条目中的代码库 ('repositories')
                            repositories = pwc_entry.get(
                                "repositories", []
                            )  # 安全获取，默认为空列表
                            if not repositories:  # 检查是否为 None 或空列表
                                stats["pwc_missing_repositories"] += 1
                            elif isinstance(repositories, list):  # 确保是列表
                                # 遍历代码库列表
                                for repo_num, repo in enumerate(repositories, 1):
                                    # 确保每个代码库条目是字典
                                    if not isinstance(repo, dict):
                                        logger.warning(
                                            f"第 {line_num} 行, ID {hf_model_id}, 论文 {paper_num}: repositories 列表中的第 {repo_num} 项无效 (非字典)。"
                                        )
                                        continue  # 跳过格式错误的代码库条目

                                    # 增加处理的代码库总数计数器
                                    stats["total_repos_processed"] += 1
                                    # 检查 URL 是否存在且有效
                                    url = repo.get("url")
                                    if not url or not isinstance(url, str):
                                        stats["repos_missing_url"] += 1
                                        continue  # 没有 URL 无法进一步检查

                                    # 检查是否为 GitHub 链接 (不区分大小写)
                                    if "github.com" in url.lower():
                                        # 增加 GitHub 代码库处理计数器
                                        stats["github_repos_processed"] += 1
                                        # 检查 GitHub 特定字段是否存在 (值为 None 算缺失)
                                        if repo.get("stars") is None:
                                            stats["github_repos_missing_stars"] += 1
                                        if repo.get("license") is None:
                                            stats["github_repos_missing_license"] += 1
                                        if repo.get("language") is None:
                                            stats["github_repos_missing_language"] += 1
                                    else:
                                        # 如果 URL 不是 GitHub 的，增加非 GitHub 计数器
                                        stats["repos_not_github"] += 1
                            else:
                                # 如果 repositories 字段不是列表
                                logger.warning(
                                    f"第 {line_num} 行, ID {hf_model_id}, 论文 {paper_num}: 'repositories' 字段类型非预期: {type(repositories)}"
                                )
                                stats["pwc_missing_repositories"] += 1  # 视为无效/缺失

    # --- 文件处理结束或发生错误 ---
    except FileNotFoundError:
        logger.error(f"错误：输入文件未找到: {jsonl_filepath}")
        return None  # 返回 None 表示失败
    except IOError as e:
        logger.error(f"读取文件 {jsonl_filepath} 时发生 I/O 错误: {e}")
        return None
    except Exception as e:
        # 捕获其他所有未预料的严重错误
        logger.critical(f"分析过程中发生意外严重错误: {e}")
        logger.critical(traceback.format_exc())  # 打印详细的堆栈跟踪
        return None  # 返回 None 表示失败

    # 分析完成，记录结束信息
    logger.info(f"完成分析 {stats['total_lines_read']} 行。")
    # 返回包含统计结果的字典
    return stats


# --- 报告生成函数 ---
def format_and_log_results_v2(
    stats: Optional[Dict[str, Any]], report_filepath: str
) -> None:
    """
    格式化分析统计结果，记录到日志，并写入到指定的报告文件中。

    Args:
        stats (Optional[Dict[str, Any]]): analyze_data_integrity_v2 返回的统计字典，或 None (如果分析失败)。
        report_filepath (str): 输出报告文件的路径。
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)

    # 如果统计字典为 None (表示分析失败)
    if not stats:
        logger.error("分析失败或文件未找到，无法生成报告。")
        return

    # --- 计算衍生统计数据 ---
    total_valid_records = stats["records_valid_json"]  # 有效 JSON 记录总数
    total_unique_models = len(stats["unique_hf_model_ids"])  # 唯一模型 ID 数量
    total_models_with_papers = len(
        stats["models_with_linked_papers"]
    )  # 至少有一篇关联论文的模型数量
    total_unique_arxiv = len(stats["unique_arxiv_ids"])  # 唯一 ArXiv ID 数量
    total_unique_datasets = len(stats["unique_dataset_links"])  # 唯一数据集链接数量

    # 定义一个计算百分比的辅助函数 (处理分母为零的情况)
    perc = lambda count, total: (count / total * 100) if total > 0 else 0

    # 计算各项缺失/存在的百分比
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
    perc_pwc_missing_tasks = perc(
        stats["pwc_missing_tasks"], stats["pwc_entries_processed"]
    )
    perc_pwc_missing_datasets = perc(
        stats["pwc_missing_datasets"], stats["pwc_entries_processed"]
    )
    perc_pwc_missing_repos = perc(
        stats["pwc_missing_repositories"], stats["pwc_entries_processed"]
    )

    perc_gh_missing_stars = perc(
        stats["github_repos_missing_stars"], stats["github_repos_processed"]
    )
    perc_gh_missing_license = perc(
        stats["github_repos_missing_license"], stats["github_repos_processed"]
    )
    perc_gh_missing_language = perc(
        stats["github_repos_missing_language"], stats["github_repos_processed"]
    )

    # --- 准备报告文本内容 ---
    # 使用一个列表存储报告的每一行
    report_lines = [
        "--- 数据完整性验证报告 ---",
        f"分析文件: {stats['input_filepath']}",
        f"生成时间: {datetime.now().isoformat()}",  # 添加当前时间戳
        "=" * 40,  # 分隔线
        f"文件摘要:",
        f"  - 总读取行数: {stats['total_lines_read']}",
        f"  - JSON 解析错误行数: {stats['json_parse_errors']}",
        f"  - 成功解析为 JSON 的记录数: {total_valid_records}",
        "=" * 40,
        f"Hugging Face 模型记录:",
        f"  - 唯一模型总数 (基于 hf_model_id): {total_unique_models}",
        f"  - 缺少 'hf_model_id' 的记录数: {stats['missing_hf_id']}",
        f"  - 缺少 'hf_author' 的记录数: {stats['missing_hf_author']} ({perc(stats['missing_hf_author'], total_valid_records):.1f}%)",
        f"  - 缺少 'hf_sha' 的记录数: {stats['missing_hf_sha']} ({perc(stats['missing_hf_sha'], total_valid_records):.1f}%)",
        f"  - 缺少 'hf_last_modified' 的记录数: {stats['missing_hf_last_modified']} ({perc(stats['missing_hf_last_modified'], total_valid_records):.1f}%)",
        f"  - 缺少 'hf_tags' 的记录数: {stats['missing_hf_tags']} ({perc(stats['missing_hf_tags'], total_valid_records):.1f}%)",
        f"  - 缺少 'hf_downloads' 的记录数: {stats['missing_hf_downloads']} ({perc(stats['missing_hf_downloads'], total_valid_records):.1f}%)",
        f"  - 缺少 'hf_likes' 的记录数: {stats['missing_hf_likes']} ({perc(stats['missing_hf_likes'], total_valid_records):.1f}%)",
        "=" * 40,
        f"README 内容 ('hf_readme_content'):",
        f"  - 完全缺少键的记录数: {stats['missing_readme_key']} ({perc_readme_key_missing:.1f}%)",
        f"  - 值为空 (null/None/空字符串) 的记录数: {stats['readme_is_null']} ({perc_readme_null:.1f}%)",
        f"  - 存在有效 README 内容的记录数: {stats['readme_is_present']} ({perc_readme_present:.1f}%)",
        "=" * 40,
        f"数据集链接 ('hf_dataset_links'):",
        f"  - 完全缺少键的记录数: {stats['missing_dataset_links_key']} ({perc_dslinks_key_missing:.1f}%)",
        f"  - 链接列表为空或无效的记录数: {stats['dataset_links_is_empty']}",
        f"  - 存在有效数据集链接的记录数: {stats['dataset_links_is_present']} ({perc_dslinks_present:.1f}%)",
        f"  - 所有记录中找到的链接总数: {stats['total_dataset_links_found']}",
        f"  - 唯一有效数据集链接总数: {total_unique_datasets}",
        "=" * 40,
        f"关联论文 & PWC 条目:",
        f"  - 至少有一篇关联论文的模型数: {total_models_with_papers}",
        f"  - 处理的关联论文条目总数: {stats['total_papers_processed']}",
        f"  - 论文条目缺少 'arxiv_id_base' 的数量: {stats['papers_missing_arxiv_id']}",
        f"  - 找到的唯一 ArXiv ID 总数: {total_unique_arxiv}",
        f"  - 论文条目缺少有效 'pwc_entry' 的数量: {stats['papers_missing_pwc_entry']} ({perc_papers_missing_pwc:.1f}%)",
        f"  - 处理的有效 PWC 条目总数: {stats['pwc_entries_processed']}",
        f"  - PWC 条目缺少 'conference' 的数量: {stats['pwc_missing_conference']} ({perc_pwc_missing_conf:.1f}%)",
        f"  - PWC 条目缺少 'methods' (或列表为空) 的数量: {stats['pwc_missing_methods']} ({perc_pwc_missing_methods:.1f}%)",
        f"  - PWC 条目缺少 'tasks' (或列表为空) 的数量: {stats['pwc_missing_tasks']} ({perc_pwc_missing_tasks:.1f}%)",  # 添加 tasks 百分比
        f"  - PWC 条目缺少 'datasets' (或列表为空) 的数量: {stats['pwc_missing_datasets']} ({perc_pwc_missing_datasets:.1f}%)",  # 添加 datasets 百分比
        f"  - PWC 条目缺少 'repositories' (或列表为空) 的数量: {stats['pwc_missing_repositories']} ({perc_pwc_missing_repos:.1f}%)",  # 添加 repos 百分比
        "=" * 40,
        f"GitHub 代码库 (在 PWC 条目内):",
        f"  - 处理的代码库条目总数: {stats['total_repos_processed']}",
        f"  - 代码库条目缺少 'url' 的数量: {stats['repos_missing_url']}",
        f"  - 代码库 URL 非 GitHub 的数量: {stats['repos_not_github']}",
        f"  - 处理的 GitHub 代码库条目总数: {stats['github_repos_processed']}",
        f"  - GitHub 代码库 'stars' 为 null 的数量: {stats['github_repos_missing_stars']} ({perc_gh_missing_stars:.1f}%)",
        f"  - GitHub 代码库 'license' 为 null 的数量: {stats['github_repos_missing_license']} ({perc_gh_missing_license:.1f}%)",
        f"  - GitHub 代码库 'language' 为 null 的数量: {stats['github_repos_missing_language']} ({perc_gh_missing_language:.1f}%)",
        "=" * 40,
    ]

    # --- 将结果记录到日志 ---
    logger.info("--- 数据完整性分析摘要 ---")
    # 遍历报告的每一行并使用 INFO 级别记录
    for line in report_lines:
        logger.info(line)
    logger.info("--- 摘要结束 ---")

    # --- 将报告写入文件 ---
    try:
        # 获取报告文件所在的目录
        report_dir = os.path.dirname(report_filepath)
        # 创建报告目录 (如果不存在)
        os.makedirs(report_dir, exist_ok=True)
        # 以写入模式、UTF-8 编码打开报告文件
        with open(report_filepath, "w", encoding="utf-8") as f:
            # 逐行写入报告内容，并在每行后添加换行符
            for line in report_lines:
                f.write(line + "\n")
        # 记录报告写入成功信息
        logger.info(f"完整报告已写入到: {report_filepath}")
    except IOError as e:
        # 如果写入文件时发生错误
        logger.error(f"无法将报告文件写入到 {report_filepath}: {e}")


# --- 脚本主执行入口 ---
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="验证收集到的 AIGraphX 数据的完整性和完备性。"
    )
    # 添加 --input 参数，用于指定输入文件路径
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_JSONL,  # 使用常量作为默认值
        help=f"输入的 JSONL 数据文件路径 (默认: {DEFAULT_INPUT_JSONL})",
    )
    # 添加 --report 参数，用于指定报告文件路径
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_FILE,  # 使用常量作为默认值
        help=f"输出的验证报告文件路径 (默认: {DEFAULT_REPORT_FILE})",
    )
    # 添加 --log 参数，用于指定日志文件路径
    parser.add_argument(
        "--log",
        type=str,
        default=DEFAULT_LOG_FILE,  # 使用常量作为默认值
        help=f"详细日志文件的路径 (默认: {DEFAULT_LOG_FILE})",
    )
    # 解析命令行传入的参数
    args = parser.parse_args()

    # 使用命令行指定的日志文件路径设置日志记录
    logger = setup_logging(args.log)

    # 开始执行主逻辑
    logger.info("启动数据验证脚本...")
    # 调用分析函数，传入输入文件路径
    analysis_results = analyze_data_integrity_v2(args.input)
    # 调用报告生成函数，传入分析结果和报告文件路径
    format_and_log_results_v2(analysis_results, args.report)
    logger.info("数据验证脚本执行完毕。")
