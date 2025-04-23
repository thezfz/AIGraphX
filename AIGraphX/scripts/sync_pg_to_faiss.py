#!/usr/bin/env python
# -*- coding: utf-8 -*- # 指定编码为 UTF-8

# 文件作用说明：
# 该脚本负责将 PostgreSQL 数据库中 `papers` 表的数据同步到 Faiss 索引中。
# Faiss 是一个用于高效相似性搜索和稠密向量聚类的库。
# 这个脚本主要用于构建（或更新）论文的语义向量索引，以便后续进行语义搜索。
#
# 主要流程：
# 1. 加载配置：从环境变量或 .env 文件读取数据库连接信息、Faiss 文件路径、嵌入模型名称、日志级别等。
# 2. 初始化依赖：创建数据库连接池、初始化 PostgreSQL 仓库 (`PostgresRepository`) 和文本嵌入器 (`TextEmbedder`)。
# 3. 构建或重置索引：
#    - (可选) 如果指定了 `--reset` 参数，删除旧的 Faiss 索引文件和 ID 映射文件。
#    - 创建一个新的 Faiss 索引实例 (这里使用 `IndexFlatL2`，表示使用 L2 距离进行精确搜索)。
# 4. 批量处理：
#    - 从 PostgreSQL 数据库分批获取论文的 ID 和需要向量化的文本（通常是摘要 `summary`）。
#    - 使用 `TextEmbedder` 将每批文本转换为向量。
#    - 将生成的向量添加到 Faiss 索引中。
#    - 维护一个从 Faiss 内部索引 ID 到原始论文 `paper_id` 的映射。
# 5. 保存结果：
#    - 将构建好的 Faiss 索引保存到指定的文件 (`settings.faiss_index_path`)。
#    - 将 Faiss 索引 ID 到论文 ID 的映射保存为 JSON 文件 (`settings.faiss_mapping_path`)。
# 6. 清理资源：关闭数据库连接池。
#
# 交互对象：
# - 输入：PostgreSQL 数据库中的 `papers` 表。
# - 输出：
#   - Faiss 索引文件 (e.g., data/papers_index.faiss)。
#   - Faiss ID 到论文 ID 的映射文件 (e.g., data/papers_id_map.json)。
# - 配置：.env 文件和 `aigraphx/core/config.py` 中的 `settings` 对象。
# - 依赖：`PostgresRepository`, `TextEmbedder`。
# - 日志：输出到控制台。
#
# 注意：此脚本是一个独立的批处理脚本。

# 导入标准库
import asyncio  # 用于运行异步代码
import os       # 用于与操作系统交互 (文件路径、环境变量)
import json     # 用于读写 JSON 文件 (ID 映射)
import logging  # 用于日志记录
import sys      # 用于系统相关操作 (如修改 Python 路径)
import time     # 用于计算脚本执行时间
import traceback # 用于打印详细的错误堆栈信息
from typing import List, Tuple, AsyncGenerator, Optional, cast, Dict # 用于类型提示
import argparse # 用于解析命令行参数
import gc       # 用于垃圾回收

# 导入第三方库
import psycopg_pool  # 导入异步 PostgreSQL 连接池库
import faiss         # 导入 Faiss 库 (注意：可能需要额外安装)
import psycopg       # 添加 psycopg 导入
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.errors import OperationalError as PsycopgOperationalError
import numpy as np   # 导入 NumPy 库，用于处理向量 (Faiss 和 SentenceTransformer 都使用 NumPy 数组)
from dotenv import load_dotenv # 用于加载 .env 文件
from sentence_transformers import SentenceTransformer # 导入 SentenceTransformer 库，用于文本嵌入
from tqdm import tqdm # type: ignore[import-untyped] # 修正注释位置

# --- 项目内部导入 ---
# 动态调整 Python 搜索路径，以便能导入项目根目录下的模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 将项目根目录添加到 sys.path 的最前面

# 导入项目内部模块
# from aigraphx.core import config # 可以保留，如果脚本其他地方需要直接访问 config 模块
from aigraphx.repositories.postgres_repo import PostgresRepository # 导入 PG 仓库类
from aigraphx.vectorization.embedder import TextEmbedder # 导入文本嵌入器类

# 直接从 config.py 导入 Pydantic Settings 对象，推荐使用这种方式访问配置
from aigraphx.core.config import settings

# --- 配置加载 ---
# 定位并加载 .env 文件 (如果存在)，为 settings 对象提供环境变量值
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)
logger = logging.getLogger(__name__) # 在加载dotenv之后初始化
logger.info(f"从 {dotenv_path} 加载环境变量 (如果存在)。")

# --- 常量 ---
# 脚本中使用的常量，例如批处理大小，现在优先从 settings 对象读取
# BATCH_SIZE = settings.build_faiss_batch_size

# --- 日志设置 ---
# 配置日志记录器
logging.basicConfig(
    level=settings.log_level.upper(), # 从 settings 对象读取日志级别并转为大写
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s", # 日志格式，添加了函数名
    handlers=[
        logging.StreamHandler() # 输出到控制台
        # 可以取消注释以同时写入日志文件
        # logging.FileHandler('sync_pg_to_faiss.log')
    ],
)
# 获取当前脚本的日志记录器实例
# logger = logging.getLogger(__name__) # 已移到 dotenv 加载之后

# --- 数据库函数 (已移至仓库) ---
# 原本可能在此处定义的 fetch_papers_for_indexing 函数逻辑，
# 现在已封装在 PostgresRepository 类的 get_all_paper_ids_and_text 方法中，
# 实现了更好的代码组织和复用。

# --- 主要逻辑 ---
async def build_index(
    pg_repo: PostgresRepository, # PostgreSQL 仓库实例，用于获取数据
    embedder: TextEmbedder,     # 文本嵌入器实例，用于生成向量
    index_path: str,            # Faiss 索引文件的保存路径
    id_map_path: str,           # Faiss 内部 ID 到论文 ID 映射文件的保存路径
    batch_size: int,            # 处理数据的批次大小
    reset_index: bool = False,  # 是否在开始前删除已存在的索引和映射文件
) -> None:
    """
    从 PostgreSQL 数据库获取论文数据，生成文本嵌入，并构建/保存 Faiss 索引。

    Args:
        pg_repo: 用于访问 PostgreSQL 的仓库实例。
        embedder: 用于生成文本向量的嵌入器实例。
        index_path: 保存 Faiss 索引的文件路径 (例如 'data/papers_index.faiss')。
        id_map_path: 保存 Faiss 内部索引到论文 ID 映射的 JSON 文件路径 (例如 'data/papers_id_map.json')。
        batch_size: 每次从数据库获取和处理的论文数量。
        reset_index: 如果为 True，则在开始构建前删除已存在的 index_path 和 id_map_path 文件。
    """
    logger.info("开始为论文构建 Faiss 索引...")
    start_time = time.time() # 记录开始时间，用于计算总耗时

    # --- 重置逻辑 ---
    if reset_index:
        logger.warning("指定了 --reset 标志。将删除现有的 Faiss 索引和 ID 映射文件...")
        try:
            # 如果索引文件存在，则删除
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"已删除现有索引文件: {index_path}")
            # 如果 ID 映射文件存在，则删除
            if os.path.exists(id_map_path):
                os.remove(id_map_path)
                logger.info(f"已删除现有 ID 映射文件: {id_map_path}")
        except OSError as e:
            # 如果删除文件时出错，记录错误，但脚本可能仍会继续尝试创建新文件
            logger.error(f"删除现有 Faiss 文件时出错: {e}")

    # --- 确保存储目录存在 ---
    # 获取索引文件和映射文件所在的目录路径
    index_dir = os.path.dirname(index_path)
    map_dir = os.path.dirname(id_map_path)
    try:
        # 如果目录路径非空 (不是当前目录)，则创建目录，exist_ok=True 表示如果目录已存在则不报错
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        # 如果映射目录与索引目录不同，也创建它
        if map_dir and map_dir != index_dir:
            os.makedirs(map_dir, exist_ok=True)
    except OSError as e:
        # 如果创建目录时出错，记录错误并返回，无法继续
        logger.error(f"创建 Faiss 文件存储目录时出错: {e}")
        return

    # 初始化 Faiss 索引对象和 ID 映射字典
    faiss_index: Optional[faiss.Index] = None # Faiss 索引对象，初始为 None
    paper_id_map: Dict[int, int] = {} # 存储 Faiss 内部索引 -> 论文 paper_id 的映射
    papers_processed = 0 # 已处理的论文总数
    faiss_idx_counter = 0 # Faiss 内部索引的计数器 (从 0 开始递增)

    try:
        # --- 初始化嵌入器 ---
        # 获取嵌入器产生的向量维度
        embedding_dim = embedder.get_embedding_dimension()
        if embedding_dim is None or embedding_dim <= 0:
             logger.error("无法获取有效的嵌入维度。退出。")
             return
        logger.info(f"嵌入器已就绪。嵌入向量维度: {embedding_dim}")

        # --- 创建 Faiss 索引 ---
        # 使用 IndexFlatL2 创建一个简单的 Faiss 索引。
        # IndexFlatL2 进行精确的 L2 距离（欧氏距离）搜索，对于中小型数据集效果好且易于使用。
        # 对于超大规模数据集，可能需要考虑其他索引类型 (如 IndexIVFFlat, IndexHNSWFlat) 以平衡速度和精度。
        logger.info(f"正在创建 Faiss IndexFlatL2 索引，维度为 {embedding_dim}...")
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        # Faiss 也支持 IndexIDMap，可以直接将向量与原始 ID 关联，
        # 但这里使用外部字典映射，提供了更大的灵活性。
        # faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim)) # 另一种方式

        # --- 分批处理 ---
        logger.info(f"开始分批获取论文并生成嵌入向量 (批大小: {batch_size})...")
        batch_texts: List[str] = [] # 当前批次要嵌入的文本列表 (摘要)
        batch_ids: List[int] = [] # 当前批次的论文 ID 列表
        total_papers = 0 # 从数据库获取的总论文数

        # 使用 PostgreSQL 仓库提供的异步生成器获取所有论文的 ID 和摘要
        async for paper_id, summary in pg_repo.get_all_paper_ids_and_text():
            total_papers += 1
            # 确保摘要是字符串，如果为 None 则使用空字符串
            text_to_embed = summary if summary is not None else ""
            # 将文本和 ID 添加到当前批次
            batch_texts.append(text_to_embed)
            batch_ids.append(paper_id)

            # 当批次达到设定大小时，处理该批次
            if len(batch_texts) >= batch_size:
                batch_start_time = time.time() # 记录批次处理开始时间
                batch_num = papers_processed // batch_size + 1
                logger.info(
                    f"正在处理批次 {batch_num} (论文 {papers_processed + 1}-{papers_processed + len(batch_texts)})..."
                )
                # 调用嵌入器的批量嵌入方法
                embeddings_np: Optional[np.ndarray] = embedder.embed_batch(batch_texts)

                # 检查嵌入结果是否有效
                if embeddings_np is not None and embeddings_np.shape[0] > 0:
                    # 将生成的向量添加到 Faiss 索引中
                    # 对于 IndexFlatL2，内部索引是按添加顺序从 0 开始递增的
                    faiss_index.add(embeddings_np)
                    # 更新 ID 映射：将当前批次的论文 ID 与 Faiss 内部索引关联起来
                    for p_id in batch_ids:
                        paper_id_map[faiss_idx_counter] = p_id # key 是 Faiss 内部索引，value 是论文 ID
                        faiss_idx_counter += 1 # Faiss 内部索引递增
                    logger.info(
                        f"批次 {batch_num} 已添加到索引 ({embeddings_np.shape[0]} 个向量)。"
                        f"耗时: {time.time() - batch_start_time:.2f}秒。"
                        f"索引中总向量数: {faiss_index.ntotal}"
                    )
                else:
                    # 如果嵌入器未能为该批次生成向量
                    logger.warning(f"批次 {batch_num} 未能生成有效的嵌入向量。")

                # 更新已处理的论文总数
                papers_processed += len(batch_texts)
                # 清空当前批次，准备下一批
                batch_texts = []
                batch_ids = []

        # --- 处理最后一个未满的批次 ---
        if batch_texts: # 如果循环结束后仍有剩余数据
            batch_start_time = time.time()
            logger.info(
                f"正在处理最后一个批次 (论文 {papers_processed + 1}-{papers_processed + len(batch_texts)})..."
            )
            embeddings_np = embedder.embed_batch(batch_texts)
            if embeddings_np is not None and embeddings_np.shape[0] > 0:
                assert faiss_index is not None # 确保 faiss_index 已初始化
                faiss_index.add(embeddings_np)
                for p_id in batch_ids:
                    paper_id_map[faiss_idx_counter] = p_id
                    faiss_idx_counter += 1
                logger.info(
                    f"最后一个批次已添加到索引 ({embeddings_np.shape[0]} 个向量)。"
                    f"耗时: {time.time() - batch_start_time:.2f}秒。"
                    f"索引中总向量数: {faiss_index.ntotal}"
                )
            else:
                logger.warning("最后一个批次未能生成有效的嵌入向量。")
            # 更新总处理数
            papers_processed += len(batch_texts)

        # 检查是否从数据库获取到了论文
        if total_papers == 0:
            logger.warning("数据库中未找到带有摘要的论文。Faiss 索引将为空。")
        elif papers_processed != total_papers:
             logger.warning(f"从数据库获取了 {total_papers} 篇论文，但只处理了 {papers_processed} 篇。")


        # --- 保存索引和 ID 映射 ---
        # 仅当 Faiss 索引对象存在且包含向量时才进行保存
        if faiss_index is not None and faiss_index.ntotal > 0:
            logger.info("开始保存 Faiss 索引和 ID 映射...")
            try:
                # 保存 Faiss 索引到二进制文件
                logger.info(f"正在将 Faiss 索引写入到: {index_path}")
                faiss.write_index(faiss_index, index_path)
                logger.info("Faiss 索引写入成功。")

                # 将 ID 映射字典保存为 JSON 文件
                logger.info(f"正在将 ID 映射写入到: {id_map_path}")
                with open(id_map_path, "w", encoding='utf-8') as f:
                    # 使用 ensure_ascii=False 确保非 ASCII 字符（如果ID中包含）正确写入
                    json.dump(paper_id_map, f, ensure_ascii=False, indent=4) # indent=4 使 JSON 文件更易读
                logger.info("ID 映射写入成功。")
                logger.info("Faiss 索引和 ID 映射已成功保存。")
            except Exception as save_e:
                # 如果保存过程中发生错误
                logger.error(f"保存 Faiss 索引或 ID 映射时出错: {save_e}", exc_info=True)
        elif faiss_index is not None and faiss_index.ntotal == 0:
            # 如果索引为空，则不保存，并记录警告
            logger.warning("Faiss 索引为空，无需保存。")
        else:
            # 如果 faiss_index 对象本身就是 None (例如初始化失败)
            logger.error("Faiss 索引对象为 None，无法保存。")

    except Exception as e:
        # 捕获构建过程中的任何其他未预料错误
        logger.error(f"构建 Faiss 索引过程中发生意外错误: {e}")
        logger.error(traceback.format_exc()) # 打印详细的回溯信息

    finally:
        # 无论成功还是失败，都记录结束信息和总耗时
        end_time = time.time()
        logger.info(
            f"--- Faiss 索引构建流程结束，总耗时: {end_time - start_time:.2f} 秒 ---"
        )
        if faiss_index:
            logger.info(f"最终索引包含 {faiss_index.ntotal} 个向量。")
        else:
            logger.warning("Faiss 索引对象未成功创建或处理。")


# --- 主函数 ---
async def main() -> None:
    """
    脚本的主入口异步函数。
    负责解析命令行参数、初始化资源（数据库连接池、仓库、嵌入器）并调用 build_index 函数。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从 PostgreSQL 同步论文摘要到 Faiss 索引。"
    )
    # 添加 --reset 参数
    parser.add_argument(
        "--reset",
        action="store_true", # 如果命令行包含 --reset，则 args.reset 为 True
        help="在开始构建前删除现有的 Faiss 索引和 ID 映射文件。",
    )
    # 添加 --batch-size 参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.build_faiss_batch_size, # 从 settings 获取默认值
        help=f"每次处理的论文数量 (默认值来自配置: {settings.build_faiss_batch_size})。",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 初始化数据库连接池变量
    pool: Optional[psycopg_pool.AsyncConnectionPool] = None
    try:
        # --- 初始化数据库连接 ---
        # 从 settings 对象获取数据库 URL
        db_url = settings.database_url
        if not db_url:
            logger.error("settings 中未配置 DATABASE_URL。脚本退出。")
            return

        logger.info("正在创建数据库连接池...")
        # 使用 psycopg_pool 创建异步连接池
        pool = psycopg_pool.AsyncConnectionPool(
            conninfo=db_url, # 连接信息字符串
            min_size=settings.pg_pool_min_size, # 最小连接数 (从 settings 读取)
            max_size=settings.pg_pool_max_size, # 最大连接数 (从 settings 读取)
            # timeout=... # 可以设置连接超时
        )
        # 等待连接池准备就绪 (尝试建立最小数量的连接)
        await pool.wait()
        logger.info("数据库连接池创建成功。")
        # 实例化 PostgreSQL 仓库，传入连接池
        pg_repo = PostgresRepository(pool)

        # --- 初始化文本嵌入器 ---
        logger.info("正在初始化文本嵌入器...")
        # 使用 settings 中的配置实例化 TextEmbedder
        embedder = TextEmbedder(
            model_name=settings.sentence_transformer_model, # 使用的 Sentence Transformer 模型名称
            device=settings.embedder_device, # 使用的设备 ('cpu', 'cuda', etc.)
        )
        logger.info(f"文本嵌入器已初始化，模型: {settings.sentence_transformer_model}, 设备: {settings.embedder_device}")

        # --- 调用核心构建逻辑 ---
        # 使用解析的参数和初始化的对象调用 build_index 函数
        await build_index(
            pg_repo=pg_repo,
            embedder=embedder,
            index_path=settings.faiss_index_path, # Faiss 索引路径 (从 settings 读取)
            id_map_path=settings.faiss_mapping_path, # ID 映射路径 (从 settings 读取)
            batch_size=args.batch_size, # 使用命令行参数或 settings 默认值
            reset_index=args.reset, # 使用命令行参数
        )

    except psycopg_pool.PoolTimeout: # 捕获连接池超时错误
        logger.exception("数据库连接池操作超时。")
    except psycopg_pool.PoolClosed: # 捕获连接池已关闭错误
        logger.exception("尝试在已关闭的数据库连接池上操作。")
    except (PsycopgOperationalError, psycopg.Error) as db_err: # 捕获 psycopg 相关的数据库错误
         logger.exception(f"数据库操作失败: {db_err}")
    except Exception as e:
        # 捕获其他所有未预料的异常
        logger.exception(f"主流程执行过程中发生意外错误: {e}")
    finally:
        # --- 清理资源 ---
        # 无论是否发生错误，最后都尝试关闭数据库连接池
        if pool:
            logger.info("正在关闭 PostgreSQL 连接池...")
            await pool.close()
            logger.info("PostgreSQL 连接池已关闭。")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # 当脚本作为主程序运行时 (python scripts/sync_pg_to_faiss.py)
    # 使用 asyncio.run() 来运行异步的 main() 函数
    logger.info("脚本启动...")
    asyncio.run(main())
    logger.info("脚本执行完毕。")