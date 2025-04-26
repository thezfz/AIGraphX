#!/usr/bin/env python
# -*- coding: utf-8 -*- # 指定编码为 UTF-8，确保中文字符正确处理

# 文件作用说明：
# 该脚本负责将从数据源（如 Hugging Face, ArXiv, Papers with Code）收集并处理后
# 存储在 JSONL 文件 (默认为 data/aigraphx_knowledge_data.jsonl) 中的数据，
# 加载到项目使用的 PostgreSQL 数据库中。
# 它处理的数据结构通常包含 Hugging Face 模型信息以及与之关联的论文信息（包括元数据、PWC 关系等）。
#
# 主要功能：
# 1. 加载配置：从环境变量或 .env 文件读取数据库连接信息、连接池大小等。
# 2. 日志设置：配置日志记录器，用于输出脚本运行过程中的信息和错误。
# 3. 数据解析与转换：定义辅助函数来解析日期、时间字符串，以及从 ArXiv 分类推导领域（Area）。
# 4. 检查点管理：实现检查点机制，允许脚本在中断后从上次成功处理的位置继续，避免重复加载。
# 5. 数据库操作：
#    - 使用 psycopg (v3) 和连接池 (`psycopg_pool`) 与 PostgreSQL 数据库进行异步交互。
#    - 实现插入或更新 Hugging Face 模型、论文、PWC 关系（任务、方法、数据集）、代码库以及模型-论文链接的函数。
#    - 使用 "ON CONFLICT DO UPDATE/NOTHING" 处理已存在记录的情况。
# 6. 批量处理：将 JSONL 文件中的记录分批处理，并在单个数据库事务中插入，以提高性能和保证原子性。
# 7. 主处理逻辑：
#    - 读取 JSONL 文件。
#    - 跳过已通过检查点处理的行。
#    - 逐行解析 JSON 数据。
#    - 将数据分批并调用 `process_batch` 函数处理。
#    - 定期保存检查点。
# 8. 命令行参数解析：允许通过命令行参数指定输入文件路径和是否重置检查点。
#
# 交互对象：
# - 输入：JSONL 数据文件 (e.g., data/aigraphx_knowledge_data.jsonl)。
# - 输出：将数据写入 PostgreSQL 数据库中的相应表 (hf_models, papers, pwc_tasks, model_paper_links, etc.)。
# - 配置：.env 文件 (用于数据库连接等敏感信息)。
# - 检查点：检查点文件 (data/pg_load_checkpoint.txt)，用于记录处理进度。
# - 日志：输出到控制台 (以及可选的文件)。
#
# 注意：此脚本是一个独立的批处理脚本，不直接依赖 FastAPI 应用实例。

# 导入标准库
import argparse  # 用于解析命令行参数
import asyncio  # 用于运行异步代码 (数据库操作是异步的)
import os  # 用于与操作系统交互，例如读取环境变量、处理文件路径
import json  # 用于解析 JSON 数据 (JSONL 文件每行是一个 JSON 对象)
import logging  # 用于日志记录
import sys  # 用于与 Python 解释器交互，例如退出脚本 (sys.exit) 或访问 stderr
import traceback  # 用于获取和格式化异常的堆栈跟踪信息
from datetime import datetime, timezone, date  # 用于处理日期和时间对象
from typing import Dict, Any, Optional, Tuple, List  # 用于类型提示

# 导入第三方库
import psycopg  # 导入 psycopg (v3) 库，用于与 PostgreSQL 交互 (注意不是 psycopg2)
from psycopg_pool import (
    AsyncConnectionPool,
)  # 从 psycopg 导入异步连接池，提高数据库连接效率
from dotenv import load_dotenv  # 用于从 .env 文件加载环境变量

# --- 配置加载 ---
# 定位 .env 文件的路径 (通常在项目的 Backend 目录下)
# os.path.dirname(__file__) 获取当前脚本文件所在的目录
# os.path.join 用于构建跨平台的路径
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
# 加载 .env 文件中的环境变量，如果 .env 文件不存在或变量已在系统环境中设置，则不会覆盖
load_dotenv(dotenv_path=dotenv_path)
logger = logging.getLogger(__name__)  # 在加载dotenv之后初始化，确保日志格式正确
logger.info(f"从 {dotenv_path} 加载环境变量 (如果存在)。")


# --- 常量定义 ---
# 从环境变量获取数据库连接 URL
DATABASE_URL = os.getenv("DATABASE_URL")
# 如果未设置 DATABASE_URL，打印错误信息到标准错误流并退出脚本
if not DATABASE_URL:
    print(
        "严重错误: 环境变量 DATABASE_URL 未设置或为空。",
        file=sys.stderr,
    )
    sys.exit(1)  # 非正常退出

# 从环境变量获取连接池配置，如果未设置则使用默认值
PG_POOL_MIN_SIZE = int(os.getenv("PG_POOL_MIN_SIZE", "1"))  # 最小连接数
PG_POOL_MAX_SIZE = int(os.getenv("PG_POOL_MAX_SIZE", "10"))  # 最大连接数

# 默认的输入 JSONL 文件路径
DEFAULT_INPUT_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"
# 检查点文件路径，用于保存已处理的行数
CHECKPOINT_FILE = "data/pg_load_checkpoint.txt"
# 保存检查点的频率（每处理多少行保存一次）
CHECKPOINT_INTERVAL = 100
# 批量处理的大小（每次数据库事务处理多少条记录）
BATCH_SIZE = 50

# --- 日志设置 ---
# 配置基本的日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO (忽略 DEBUG 级别的日志)
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",  # 设置日志格式
    handlers=[  # 设置日志处理器
        logging.StreamHandler()  # 输出到控制台
        # 可以取消注释下面这行来同时输出到文件
        # logging.FileHandler('load_postgres.log')
    ],
)
# 获取当前脚本的日志记录器实例
# logger = logging.getLogger(__name__) # 已移到 dotenv 加载之后

# --- 领域映射 ---
# 将 ArXiv 的主分类映射到更通用的领域名称
AREA_MAP = {
    "cs.CV": "CV",  # 计算机视觉
    "cs.CL": "NLP",  # 计算语言学 (自然语言处理)
    "cs.LG": "ML",  # 机器学习
    "cs.AI": "AI",  # 人工智能
    "cs.IR": "IR",  # 信息检索
    "cs.RO": "Robotics",  # 机器人学
    "stat.ML": "ML",  # 统计机器学习 (也归为 ML)
    # 可以根据需要添加更多映射
}


def get_area_from_category(primary_category: Optional[str]) -> Optional[str]:
    """
    根据主要的 ArXiv 分类推导出论文所属的领域 (Area)。

    Args:
        primary_category: ArXiv 返回的主要分类字符串，例如 "cs.CV", "stat.ML"。

    Returns:
        映射后的领域名称 (例如 "CV", "ML")，如果无法映射则返回 "Other"，
        如果输入为空则返回 None。
    """
    if not primary_category:
        return None
    # 有些分类可能包含更细的子类，例如 cs.CV.Computation
    # 我们通常只关心前两部分 (cs.CV) 来进行映射
    if "." in primary_category:
        parts = primary_category.split(".")
        # 确保至少有两部分
        if len(parts) >= 2:
            main_category = parts[0] + "." + parts[1]
        else:
            main_category = primary_category  # 如果格式不符合预期，使用原始分类
    else:
        main_category = primary_category  # 没有点号，直接使用

    # 从 AREA_MAP 中查找映射，如果找不到，则默认为 "Other"
    return AREA_MAP.get(main_category, "Other")


# --- 辅助函数 ---
def parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    安全地将日期字符串 (格式通常为 "YYYY-MM-DD" 或包含时间 "YYYY-MM-DDTHH:MM:SS")
    解析为 Python 的 date 对象。

    Args:
        date_str: 可能包含日期的字符串。

    Returns:
        解析成功则返回 date 对象，否则返回 None。
    """
    if not date_str:
        return None
    try:
        # 有些日期字符串可能带有时间信息 (如 "2023-01-15T...")，我们只取日期部分
        date_part = date_str.split("T")[0]
        # 使用 date.fromisoformat 解析 "YYYY-MM-DD" 格式
        return date.fromisoformat(date_part)
    except (ValueError, TypeError) as e:
        # 如果解析失败 (格式错误或类型错误)，记录警告并返回 None
        logger.warning(f"无法解析日期: {date_str}，错误: {e}")
        return None


def parse_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """
    安全地将 ISO 8601 格式的日期时间字符串解析为时区感知的 UTC datetime 对象。
    处理可能存在的 'Z' (表示 UTC)。

    Args:
        datetime_str: 可能包含日期时间的 ISO 格式字符串。

    Returns:
        解析成功则返回时区感知 (UTC) 的 datetime 对象，否则返回 None。
    """
    if not datetime_str:
        return None
    try:
        # datetime.fromisoformat 可以处理 ISO 8601 格式
        # 将末尾的 'Z' 替换为 '+00:00'，确保解析器能正确识别为 UTC
        dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

        # 检查解析后的 datetime 对象是否是时区感知的
        if dt.tzinfo is None:
            # 理论上，如果原始字符串符合 ISO 8601 且包含时区信息或 'Z'，
            # fromisoformat 应该返回时区感知的对象。
            # 但作为防御性措施，如果它返回了 naive 对象，我们假设它是 UTC。
            logger.warning(f"解析的日期时间 {datetime_str} 是 naive 的，假设为 UTC。")
            return dt.replace(tzinfo=timezone.utc)
        else:
            # 如果已经是时区感知的，确保它转换为 UTC 时区
            return dt.astimezone(timezone.utc)
    except (ValueError, TypeError) as e:
        # 如果解析失败，记录警告并返回 None
        logger.warning(f"无法解析日期时间: {datetime_str}，错误: {e}")
        return None


# --- 检查点管理 ---
def _save_checkpoint(line_count: int) -> None:
    """
    将成功处理的行数保存到检查点文件中。

    Args:
        line_count: 已成功处理并提交到数据库的最后一行行号。
    """
    try:
        # 确保检查点文件所在的目录存在
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        # 以写入模式打开检查点文件 (会覆盖旧内容)
        with open(CHECKPOINT_FILE, "w") as f:
            # 将行数转换为字符串写入文件
            f.write(str(line_count))
        # 记录调试信息
        logger.debug(f"检查点已保存: 处理了 {line_count} 行。")
    except IOError as e:
        # 如果保存文件时出错，记录错误
        logger.error(f"无法将检查点保存到 {CHECKPOINT_FILE}: {e}")


def _load_checkpoint(reset_checkpoint: bool = False) -> int:
    """
    从检查点文件中加载已处理的行数。

    Args:
        reset_checkpoint: 如果为 True，则忽略并删除现有的检查点文件，从头开始。

    Returns:
        已处理的行数。如果检查点文件不存在、无法读取或解析，或者 reset_checkpoint 为 True，则返回 0。
    """
    # 如果指定了重置标志，并且检查点文件存在
    if reset_checkpoint and os.path.exists(CHECKPOINT_FILE):
        try:
            # 删除现有的检查点文件
            os.remove(CHECKPOINT_FILE)
            logger.info(f"指定了重置标志。已删除现有的检查点文件: {CHECKPOINT_FILE}")
        except OSError as e:
            # 如果删除失败，记录错误，但不影响后续逻辑 (会返回 0)
            logger.error(f"重置时删除检查点文件 {CHECKPOINT_FILE} 失败: {e}")
        # 重置后，直接返回 0
        return 0

    # 如果检查点文件不存在 (可能是第一次运行，或已被重置删除)
    if not os.path.exists(CHECKPOINT_FILE):
        return 0

    # 如果文件存在，尝试读取内容
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            # 读取文件内容，去除首尾空白，并转换为整数
            processed_count = int(f.read().strip())
            logger.info(f"检查点已加载: 将从第 {processed_count + 1} 行恢复处理。")
            return processed_count
    except (IOError, ValueError) as e:
        # 如果读取文件或转换整数时出错
        logger.error(
            f"加载或解析检查点 {CHECKPOINT_FILE} 失败: {e}。将从头开始处理 (返回 0)。"
        )
        return 0


# --- 数据库操作 (使用 psycopg v3 异步接口) ---

# 注意：以下所有数据库操作函数都接收一个 `psycopg.AsyncConnection` 对象作为参数，
# 这个连接对象应该由调用者（通常是 `process_batch` 函数）从连接池中获取并管理。


async def insert_hf_model(
    conn: psycopg.AsyncConnection, model_data: Dict[str, Any]
) -> None:
    """
    使用 psycopg 异步接口，向 `hf_models` 表中插入或更新一条模型记录。
    使用 `ON CONFLICT DO UPDATE` 来处理模型 ID 已存在的情况。

    Args:
        conn: 一个活跃的异步数据库连接。
        model_data: 包含模型信息的字典，键应对应数据库列名（或稍作转换）。
    """
    # 使用连接创建一个异步游标
    async with conn.cursor() as cur:
        # 执行 SQL 插入/更新语句
        # 使用 %s 作为占位符，psycopg 会自动处理参数转义，防止 SQL 注入
        await cur.execute(
            """
            INSERT INTO hf_models (
                hf_model_id, hf_author, hf_sha, hf_last_modified, hf_downloads,
                hf_likes, hf_tags, hf_pipeline_tag, hf_library_name
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hf_model_id) DO UPDATE SET -- 如果 hf_model_id 已存在
                hf_author = EXCLUDED.hf_author,       -- 更新为新提供的值 (EXCLUDED 表示 VALUES 子句中的值)
                hf_sha = EXCLUDED.hf_sha,
                hf_last_modified = EXCLUDED.hf_last_modified,
                hf_downloads = EXCLUDED.hf_downloads,
                hf_likes = EXCLUDED.hf_likes,
                hf_tags = EXCLUDED.hf_tags,            -- 注意 tags 需要是 JSON 格式
                hf_pipeline_tag = EXCLUDED.hf_pipeline_tag,
                hf_library_name = EXCLUDED.hf_library_name,
                updated_at = NOW()                   -- 更新 updated_at 时间戳
        """,
            (  # 提供与占位符顺序对应的参数元组
                model_data.get("hf_model_id"),
                model_data.get("hf_author"),
                model_data.get("hf_sha"),
                # 调用辅助函数将字符串解析为 datetime 对象
                parse_datetime(model_data.get("hf_last_modified")),
                model_data.get("hf_downloads"),
                model_data.get("hf_likes"),
                # 将 Python 列表/字典转换为 JSON 字符串存储，如果值为 None 则存 NULL
                json.dumps(model_data.get("hf_tags"))
                if model_data.get("hf_tags")
                else None,
                model_data.get("hf_pipeline_tag"),
                model_data.get("hf_library_name"),
            ),
        )
        # 不需要显式调用 commit，因为这通常在 `process_batch` 的事务中完成


async def get_or_insert_paper(
    conn: psycopg.AsyncConnection, paper_data: Dict[str, Any]
) -> Optional[int]:
    """
    根据 PWC ID 或 ArXiv ID (无版本号) 查找现有的论文记录，如果不存在则插入新记录。
    使用 psycopg 异步接口。

    Args:
        conn: 一个活跃的异步数据库连接。
        paper_data: 包含单篇论文信息的字典，通常嵌套在模型数据中。
                    期望包含 'pwc_entry' 和 'arxiv_metadata' 子字典，以及 'arxiv_id_base'。

    Returns:
        论文在 `papers` 表中的主键 ID (整数)，如果查找或插入失败则返回 None。
    """
    # 安全地获取 PWC 和 ArXiv 的元数据字典
    pwc_entry = paper_data.get("pwc_entry") or {}
    arxiv_meta = paper_data.get("arxiv_metadata") or {}

    # 提取用于查找或插入的关键 ID
    pwc_id = pwc_entry.get("pwc_id")
    arxiv_id_base = paper_data.get("arxiv_id_base")  # 通常是无版本号的 ArXiv ID
    arxiv_id_versioned = arxiv_meta.get("arxiv_id_versioned")  # 带版本号的 ArXiv ID
    primary_category = arxiv_meta.get("primary_category")  # ArXiv 主分类
    # 推导领域
    area = get_area_from_category(primary_category)

    # 如果 PWC ID 和 ArXiv Base ID 都没有，无法查找或插入论文
    if not pwc_id and not arxiv_id_base:
        logger.warning(
            f"跳过论文插入/查找: 缺少 pwc_id 和 arxiv_id_base。数据: {paper_data}"
        )
        return None

    # 使用连接创建异步游标
    async with conn.cursor() as cur:
        existing_id: Optional[int] = None

        # 1. 尝试根据 PWC ID 查找
        if pwc_id:
            await cur.execute(
                "SELECT paper_id FROM papers WHERE pwc_id = %s", (pwc_id,)
            )
            row = await cur.fetchone()  # 获取一行结果
            if row:
                existing_id = row[0]  # 第一列是 paper_id

        # 2. 如果 PWC ID 未找到，且存在 ArXiv Base ID，则根据 ArXiv Base ID 查找
        if not existing_id and arxiv_id_base:
            await cur.execute(
                "SELECT paper_id FROM papers WHERE arxiv_id_base = %s", (arxiv_id_base,)
            )
            row = await cur.fetchone()
            if row:
                existing_id = row[0]

        # 3. 处理查找结果
        if existing_id:
            # 如果找到了现有论文
            logger.debug(
                f"找到现有论文 ID {existing_id} (pwc_id={pwc_id}, arxiv_id={arxiv_id_base})"
            )
            # 可以考虑在这里更新现有记录，例如如果 area 之前是 NULL，现在可以更新
            await cur.execute(
                """
                UPDATE papers
                SET area = COALESCE(%s, area), -- 如果新 area 非 NULL 则更新，否则保留旧值
                    primary_category = COALESCE(%s, primary_category),
                    categories = COALESCE(%s::jsonb, categories), -- 更新分类列表
                    updated_at = NOW()
                WHERE paper_id = %s
                """,
                (
                    area,
                    primary_category,
                    json.dumps(arxiv_meta.get("categories"))
                    if arxiv_meta.get("categories")
                    else None,
                    existing_id,
                ),
            )
            return existing_id
        else:
            # 如果没有找到现有论文，则插入新记录
            logger.debug(f"插入新论文 (pwc_id={pwc_id}, arxiv_id={arxiv_id_base})")
            try:
                await cur.execute(
                    """
                    INSERT INTO papers (
                        pwc_id, arxiv_id_base, arxiv_id_versioned, title, authors, summary,
                        published_date, updated_date, pdf_url, doi, primary_category,
                        categories, pwc_title, pwc_url, area
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING paper_id -- 返回新插入行的 paper_id
                """,
                    (  # 提供参数元组
                        pwc_id,
                        arxiv_id_base,
                        arxiv_id_versioned,
                        arxiv_meta.get("title"),
                        # 作者和分类列表需要转换为 JSON 字符串存储
                        json.dumps(arxiv_meta.get("authors"))
                        if arxiv_meta.get("authors")
                        else None,
                        arxiv_meta.get("summary"),
                        parse_date(arxiv_meta.get("published_date")),  # 解析日期
                        parse_date(arxiv_meta.get("updated_date")),  # 解析日期
                        arxiv_meta.get("pdf_url"),
                        arxiv_meta.get("doi"),
                        primary_category,
                        json.dumps(arxiv_meta.get("categories"))
                        if arxiv_meta.get("categories")
                        else None,
                        pwc_entry.get("title"),  # PWC 的标题可能与 ArXiv 不同
                        pwc_entry.get("pwc_url"),  # PWC 页面的 URL
                        area,  # 推导出的领域
                    ),
                )
                # 获取 RETURNING 子句返回的结果
                row = await cur.fetchone()
                # 确保返回了结果，且结果是整数 ID
                if row and isinstance(row[0], int):
                    return row[0]
                else:
                    # 如果插入后未能获取有效的 ID，记录错误
                    logger.error(
                        f"插入新论文后未能获取有效的 paper_id (int) (pwc_id={pwc_id}, arxiv_id={arxiv_id_base})。得到: {row}"
                    )
                    return None
            except Exception as e:
                # 如果插入过程中发生数据库错误或其他异常
                logger.error(
                    f"插入论文 (pwc_id={pwc_id}, arxiv={arxiv_id_base}) 时出错: {e}"
                )
                logger.error(traceback.format_exc())  # 记录完整堆栈跟踪
                return None


async def insert_pwc_relation(
    conn: psycopg.AsyncConnection,
    paper_id: int,
    relation_type: str,  # 关系类型，例如 "task", "method", "dataset"
    items: Optional[List[str]],  # 相关项目的名称列表
) -> None:
    """
    向 PWC 相关表中插入论文与任务/方法/数据集的关系记录。
    使用 psycopg 异步接口和 executemany 进行批量插入。

    Args:
        conn: 一个活跃的异步数据库连接。
        paper_id: 论文的主键 ID。
        relation_type: 关系的类型 ('task', 'method', 'dataset')。
        items: 相关项目名称的列表。
    """
    # 如果项目列表为空或为 None，则不执行任何操作
    if not items:
        return

    # 根据 relation_type 动态构建表名和列名
    # 例如，如果 relation_type 是 "task"，则表名为 "pwc_tasks"，列名为 "task_name"
    table_name = f"pwc_{relation_type}s"
    column_name = f"{relation_type}_name"

    # 准备要插入的数据元组列表，每个元组是 (paper_id, item_name)
    data_tuples = [(paper_id, item) for item in items]

    # 构建 SQL 插入语句，使用 f-string 插入表名和列名 (需确保 relation_type 来自受信任的源)
    # 使用 ON CONFLICT DO NOTHING 避免插入重复的关系
    sql = f"INSERT INTO {table_name} (paper_id, {column_name}) VALUES (%s, %s) ON CONFLICT DO NOTHING"

    # 使用连接创建异步游标
    async with conn.cursor() as cur:
        # 使用 executemany 批量执行插入操作，效率更高
        await cur.executemany(sql, data_tuples)

    logger.debug(
        f"为 paper_id {paper_id} 插入了 {len(data_tuples)} 条 {relation_type} 关系。"
    )


async def insert_pwc_repositories(
    conn: psycopg.AsyncConnection, paper_id: int, repos: Optional[List[Dict[str, Any]]]
) -> None:
    """
    向 `pwc_repositories` 表中插入或更新与论文相关的代码库信息。
    使用 psycopg 异步接口和 executemany。

    Args:
        conn: 一个活跃的异步数据库连接。
        paper_id: 论文的主键 ID。
        repos: 包含代码库信息的字典列表，每个字典应包含 'url', 'stars', 'is_official', 'framework' 等键。
    """
    # 如果代码库列表为空或为 None，则不执行任何操作
    if not repos:
        return

    # 准备要插入/更新的数据元组列表
    # 每个元组包含 (paper_id, url, stars, is_official, framework)
    data_tuples = [
        (
            paper_id,
            repo.get("url"),
            repo.get("stars"),  # 星标数
            repo.get("is_official"),  # 是否官方实现
            repo.get("framework"),  # 使用的框架 (如 PyTorch, TensorFlow)
        )
        for repo in repos
        if repo.get("url")  # 仅当 URL 存在时才处理该代码库
    ]

    # 如果没有有效的代码库数据，则返回
    if not data_tuples:
        return

    # 构建 SQL 插入/更新语句
    # ON CONFLICT (paper_id, url) DO UPDATE ... 表示如果 paper_id 和 url 的组合已存在，
    # 则更新 stars, is_official, framework 字段为新提供的值
    sql = """
        INSERT INTO pwc_repositories (paper_id, url, stars, is_official, framework)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (paper_id, url) DO UPDATE SET
            stars = EXCLUDED.stars,
            is_official = EXCLUDED.is_official,
            framework = EXCLUDED.framework
    """
    # 使用连接创建异步游标
    async with conn.cursor() as cur:
        # 使用 executemany 批量执行插入/更新操作
        await cur.executemany(sql, data_tuples)

    logger.debug(
        f"为 paper_id {paper_id} 插入/更新了 {len(data_tuples)} 条代码库记录。"
    )


async def insert_model_paper_link(
    conn: psycopg.AsyncConnection, hf_model_id: str, paper_id: Optional[int]
) -> None:
    """
    在 `model_paper_links` 表中插入一条连接 Hugging Face 模型和论文的记录。
    使用 psycopg 异步接口。

    Args:
        conn: 一个活跃的异步数据库连接。
        hf_model_id: Hugging Face 模型的 ID。
        paper_id: 论文的主键 ID。
    """
    # 如果 paper_id 无效 (例如论文插入失败)，则无法创建链接
    if paper_id is None:  # 显式检查 None
        logger.warning(f"无法链接模型 {hf_model_id}: paper_id 为 None 或无效。")
        return

    # 构建 SQL 插入语句
    # 假设 `model_paper_links` 表至少包含 hf_model_id 和 paper_id 列
    # ON CONFLICT (hf_model_id, paper_id) DO NOTHING 避免插入重复的链接
    sql = """
        INSERT INTO model_paper_links (hf_model_id, paper_id)
        VALUES (%s, %s)
        ON CONFLICT (hf_model_id, paper_id) DO NOTHING
    """
    # 使用连接创建异步游标
    async with conn.cursor() as cur:
        # 执行插入操作
        await cur.execute(sql, (hf_model_id, paper_id))

    logger.debug(
        f"已链接模型 {hf_model_id} 与 paper_id {paper_id} (如果链接尚不存在)。"
    )


# --- 主要处理逻辑 ---


async def process_batch(
    conn: psycopg.AsyncConnection, batch: List[Tuple[int, Dict[str, Any]]]
) -> int:
    """
    在单个数据库事务中处理一批 JSONL 记录。

    Args:
        conn: 从连接池获取的活跃异步数据库连接。
        batch: 一个列表，包含元组 (line_num, record_dict)，代表一批要处理的记录。

    Returns:
        成功处理（可能已提交到数据库）的记录数量。如果在处理过程中发生异常导致事务回滚，则返回 0。
    """
    processed_in_batch = 0  # 尝试处理的记录数
    successful_lines_in_batch = 0  # 成功处理且理论上已提交的记录数
    # 使用异步事务上下文管理器。如果内部代码块成功完成，事务将自动提交；
    # 如果发生任何异常，事务将自动回滚。
    async with conn.transaction():
        logger.debug(f"开始处理包含 {len(batch)} 条记录的批次事务...")
        # 遍历批次中的每条记录 (行号, 数据字典)
        for line_num, record in batch:
            try:
                # 假设记录处理成功，除非在处理过程中发生特定错误并更改此标志
                record_processed_successfully = True

                # --- 处理 Hugging Face 模型信息 ---
                hf_model_id = record.get("hf_model_id")
                if hf_model_id:
                    # 调用函数插入或更新模型信息
                    await insert_hf_model(conn, record)
                else:
                    # 如果关键的 hf_model_id 缺失，记录警告
                    logger.warning(f"记录在第 {line_num} 行缺少 hf_model_id。")
                    # 可以选择将此记录标记为处理失败
                    record_processed_successfully = False
                    # 也可以选择跳过此记录的后续处理 (例如论文关联)
                    # continue

                # --- 处理关联的论文信息 ---
                # linked_papers 字段应该是一个包含论文数据字典的列表
                linked_papers = record.get("linked_papers", [])
                # 做一步类型检查，确保它是列表
                if not isinstance(linked_papers, list):
                    logger.warning(
                        f"记录在第 {line_num} 行: 'linked_papers' 不是列表。将跳过论文处理。"
                    )
                    linked_papers = []  # 将其视为空列表，避免后续错误

                # 遍历模型关联的每篇论文数据
                for paper_data in linked_papers:
                    # 确保列表中的每个元素确实是字典
                    if not isinstance(paper_data, dict):
                        logger.warning(
                            f"跳过模型 {hf_model_id} 在第 {line_num} 行的无效论文条目: 不是字典类型。"
                        )
                        continue  # 处理下一篇论文

                    # --- 处理单篇论文 (获取或插入) ---
                    # 调用函数查找或插入论文，并获取其数据库 ID
                    paper_id = await get_or_insert_paper(conn, paper_data)

                    # --- 关联模型和论文 ---
                    # 只有当模型 ID 和论文 ID 都有效时才能创建链接
                    if (
                        hf_model_id and paper_id is not None
                    ):  # 显式检查 paper_id is not None
                        await insert_model_paper_link(conn, hf_model_id, paper_id)
                    # 可以添加日志记录链接未发生的情况（可选）
                    # elif paper_id is not None and not hf_model_id:
                    #     logger.debug(f"第 {line_num} 行的论文已处理，但缺少 HF 模型 ID。")
                    # elif hf_model_id and paper_id is None:
                    #     logger.debug(f"第 {line_num} 行的 HF 模型 {hf_model_id} 已处理，但论文插入/查找失败。")

                    # --- 处理 PWC 关系和代码库 (仅当论文 ID 有效时) ---
                    if paper_id is not None:
                        # 注意：pwc_entry 应从当前的 paper_data 中获取，而不是顶层 record
                        pwc_entry = paper_data.get("pwc_entry") or {}
                        # 插入任务、方法、数据集等关系
                        await insert_pwc_relation(
                            conn, paper_id, "task", pwc_entry.get("tasks")
                        )
                        await insert_pwc_relation(
                            conn, paper_id, "method", pwc_entry.get("methods")
                        )
                        await insert_pwc_relation(
                            conn, paper_id, "dataset", pwc_entry.get("datasets_used")
                        )
                        # 插入代码库信息
                        await insert_pwc_repositories(
                            conn, paper_id, pwc_entry.get("repositories")
                        )
                    else:
                        # 如果 get_or_insert_paper 返回 None，表示处理这篇论文失败
                        logger.warning(
                            f"未能获取或插入第 {line_num} 行 linked_papers 中的论文条目。论文数据: {paper_data}"
                        )
                        # 根据业务逻辑决定此失败是否应标记整个模型记录为失败
                        # record_processed_successfully = False

                # 只有当整个模型记录（包括其关联论文的处理）被认为是成功的，才增加成功计数
                # 可以根据需求调整成功的定义（例如，是否必须成功处理所有关联论文）
                if record_processed_successfully:
                    # processed_in_batch += 1 # 这个计数器似乎不再需要，用 successful_lines_in_batch 替代
                    successful_lines_in_batch += 1  # 增加成功处理的行数
                    logger.debug(f"成功处理了第 {line_num} 行的记录 (及其关联论文)。")
                else:
                    logger.warning(f"标记第 {line_num} 行的记录为处理时有错误/跳过。")
                    # 可以选择是否将有错误的行也计入尝试处理的总数
                    # processed_in_batch += 1

            except Exception as e:
                # 如果在处理单条记录时发生任何未预料的异常
                # 获取详细的 traceback 信息
                tb_str = traceback.format_exc()
                # 记录错误，包括行号、异常信息、记录内容和 traceback
                logger.error(
                    f"处理第 {line_num} 行记录时出错: {e}\n记录: {record}\nTraceback:\n{tb_str}"
                )
                # !!! 重新抛出异常 !!!
                # 这非常重要，因为 `async with conn.transaction():` 捕获到异常时会自动执行回滚。
                # 如果不重新抛出，事务管理器会认为代码块正常完成并尝试提交，这可能不是我们想要的。
                raise

    # 当 `async with conn.transaction():` 块结束时：
    # - 如果没有异常抛出，事务会自动提交。
    # - 如果有异常抛出（通过 raise），事务会自动回滚。
    logger.debug(
        f"批次事务处理完成 (自动提交或回滚)。此批次成功处理 {successful_lines_in_batch} 行。"
    )
    # 返回在此事务中成功处理的行数
    return successful_lines_in_batch


async def main(input_file_path: str, reset_db: bool, reset_checkpoint: bool) -> None:
    """
    主函数，负责协调从 JSONL 文件加载数据到 PostgreSQL 的整个过程。

    Args:
        input_file_path: 输入的 JSONL 文件路径。
        reset_db: 是否重置数据库（注意：此脚本不再负责重置数据库，此参数仅用于日志记录）。
        reset_checkpoint: 是否忽略并删除现有的检查点文件，从头开始加载。
    """
    logger.info(f"开始从文件加载数据: {input_file_path}")
    logger.info(f"重置检查点: {reset_checkpoint}")
    logger.info(f"重置数据库 (此脚本忽略，由外部处理): {reset_db}")

    pool: Optional[AsyncConnectionPool] = None  # 初始化连接池变量
    processed_count = 0  # 成功处理的总行数
    batch_count = 0  # 处理的批次数
    error_count = 0  # 解析或处理时跳过的行数
    current_batch: List[Tuple[int, Dict[str, Any]]] = []  # 当前正在构建的批次
    # 加载检查点，确定从哪一行开始处理
    start_line = _load_checkpoint(reset_checkpoint)

    # reset_db 标志现在仅用于日志记录，实际的数据库清理（如 TRUNCATE）
    # 应该在运行此脚本之前由外部机制（例如测试框架的 fixture）完成。

    try:
        # --- 初始化数据库连接池 ---
        logger.info(f"正在为数据库 {DATABASE_URL} 初始化连接池...")
        if not DATABASE_URL:
            logger.critical("DATABASE_URL 未配置。脚本退出。")
            return  # 提前退出

        # 创建异步连接池实例
        pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,  # 数据库连接字符串
            min_size=PG_POOL_MIN_SIZE,  # 最小连接数
            max_size=PG_POOL_MAX_SIZE,  # 最大连接数
            open=True,  # 在创建时就尝试建立初始连接
            # 可以根据需要调整连接或命令超时时间
            # timeout=60.0,
        )
        await pool.check()  # 检查连接池状态，确保至少一个连接可用
        logger.info("数据库连接池初始化并检查成功。")

        # --- 处理输入文件 ---
        # 以只读模式打开 JSONL 文件
        with open(input_file_path, "r", encoding="utf-8") as infile:  # 指定utf-8编码
            # 逐行读取文件，使用 enumerate 获取行号 (从 0 开始)
            last_processed_line = start_line  # 记录实际处理到的最后一行行号
            for i, line in enumerate(infile):
                line_num = i + 1  # 将行号转换为从 1 开始
                # 如果当前行号小于或等于检查点记录的行号，则跳过
                if line_num <= start_line:
                    continue

                # 尝试解析当前行
                try:
                    # 将 JSON 字符串解析为 Python 字典
                    data = json.loads(line)
                    # 基本的数据格式验证：必须是字典，且必须包含 'hf_model_id'
                    if not isinstance(data, dict) or not data.get("hf_model_id"):
                        logger.warning(
                            f"跳过第 {line_num} 行: 格式无效或缺少 'hf_model_id'。内容: {line.strip()}"
                        )
                        error_count += 1
                        continue  # 处理下一行

                    # 将有效的记录 (行号, 数据字典) 添加到当前批次
                    current_batch.append((line_num, data))

                    # 如果当前批次达到设定的大小
                    if len(current_batch) >= BATCH_SIZE:
                        batch_start_line = current_batch[0][0]
                        logger.info(
                            f"开始处理从第 {batch_start_line} 行开始的批次 (大小: {len(current_batch)})..."
                        )
                        # 从连接池获取一个连接来处理这个批次
                        async with pool.connection() as conn:
                            # 调用 process_batch 函数处理批次，该函数会在一个事务中完成
                            lines_processed_in_batch = await process_batch(
                                conn, current_batch
                            )
                        # 累加成功处理的行数
                        processed_count += lines_processed_in_batch
                        batch_count += 1  # 增加批次数
                        current_batch = []  # 清空当前批次，准备下一个
                        last_processed_line = line_num  # 更新最后处理的行号

                        # 检查是否达到了保存检查点的间隔
                        # (根据批次数判断，避免过于频繁地写文件)
                        if batch_count % (CHECKPOINT_INTERVAL // BATCH_SIZE or 1) == 0:
                            _save_checkpoint(last_processed_line)  # 保存检查点

                except json.JSONDecodeError:
                    # 如果行内容不是有效的 JSON
                    logger.error(
                        f"跳过第 {line_num} 行: 无效的 JSON。内容: {line.strip()}"
                    )
                    error_count += 1
                except Exception as e:
                    # 捕获处理单行时可能出现的其他意外错误
                    logger.error(
                        f"处理第 {line_num} 行时出错: {e}。内容: {line.strip()}",
                        exc_info=True,  # 记录堆栈跟踪
                    )
                    error_count += 1
                    # 在这里可以根据需要决定是继续处理下一行，还是中断整个脚本
                    # break

            # --- 处理最后一个未满的批次 ---
            if current_batch:
                batch_start_line = current_batch[0][0]
                logger.info(
                    f"开始处理最后一个批次 (从第 {batch_start_line} 行开始，大小: {len(current_batch)})..."
                )
                async with pool.connection() as conn:
                    lines_processed_in_batch = await process_batch(conn, current_batch)
                processed_count += lines_processed_in_batch
                # 更新最后处理的行号为文件中的最后一行
                last_processed_line = line_num

            # --- 最终保存检查点 ---
            # 确保即使没有达到检查点间隔，在脚本成功结束后也保存最终进度
            if last_processed_line > start_line:  # 只有当我们确实处理了新的行时才保存
                _save_checkpoint(last_processed_line)

    except FileNotFoundError:
        logger.critical(f"错误：输入文件未找到: {input_file_path}")
    except psycopg.Error as db_err:
        logger.critical(f"数据库连接或操作错误: {db_err}", exc_info=True)
    except Exception as e:
        # 捕获其他在主流程中可能发生的意外错误 (例如连接池初始化失败)
        logger.critical(f"发生未预料的严重错误: {e}", exc_info=True)
    finally:
        # --- 清理资源 ---
        if pool:
            # 异步关闭连接池，释放所有连接
            await pool.close()
            logger.info("数据库连接池已关闭。")
        # 记录最终的统计信息
        logger.info(
            f"数据加载完成。成功处理的总行数: {processed_count}。跳过/错误的总行数: {error_count}。"
        )


# --- 脚本入口点 ---
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从 JSONL 文件加载 AI Graph X 数据到 PostgreSQL 数据库。"
    )
    # 添加 --input-file 参数
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_JSONL_FILE,  # 使用之前定义的常量作为默认值
        help=f"指定输入的 JSONL 文件路径 (默认: {DEFAULT_INPUT_JSONL_FILE})",
    )
    # 添加 --reset 参数 (用于重置检查点)
    parser.add_argument(
        "--reset",
        action="store_true",  # 当出现此参数时，其值为 True，否则为 False
        help="忽略现有的检查点文件，从输入文件的第一行开始加载。",
    )
    # 添加 --reset-db 参数 (已弃用，仅作说明)
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="[已弃用/忽略] 数据库重置应在运行此脚本前由外部完成 (例如在测试环境中)。此标志将被忽略。",
    )

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 使用解析到的参数运行主异步函数
    # asyncio.run() 会启动并管理事件循环来执行 main 协程
    asyncio.run(main(args.input_file, args.reset_db, args.reset))
