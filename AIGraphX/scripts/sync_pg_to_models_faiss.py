#!/usr/bin/env python
# -*- coding: utf-8 -*- # 指定编码为 UTF-8

# 文件作用说明：
# 该脚本负责将 PostgreSQL 数据库中 `hf_models` 表的数据（主要是模型 ID 和用于嵌入的文本描述）
# 同步到 Faiss 索引中。这与 `sync_pg_to_faiss.py`（处理论文）类似，但专门为 Hugging Face 模型
# 创建一个独立的语义向量索引。这个索引将用于支持基于模型描述的语义搜索功能。
#
# 主要流程：
# 1. 加载配置：从 .env 文件和 settings 对象读取数据库连接、Faiss 文件路径、嵌入模型等配置。
# 2. 初始化依赖：创建数据库连接池、PostgreSQL 仓库 (`PostgresRepository`) 和文本嵌入器 (`TextEmbedder`)。
# 3. 构建或重置索引：
#    - (可选) 删除旧的模型 Faiss 索引和 ID 映射文件（如果指定 --reset）。
#    - 如果索引已存在且未指定重置，则跳过构建。
# 4. 批量处理：
#    - 从 PostgreSQL 数据库分批获取模型的 ID 和用于嵌入的文本。
#    - 使用 `TextEmbedder` 将文本批量转换为向量。
#    - 收集所有生成的向量和对应的模型 ID。
# 5. 构建与保存：
#    - 将收集到的所有向量合并成一个 NumPy 数组。
#    - 创建 Faiss 索引 (IndexFlatL2)。
#    - 将向量添加到 Faiss 索引。
#    - 创建 Faiss 内部索引到模型 ID (字符串) 的映射。
#    - 保存 Faiss 索引和 ID 映射到指定文件。
# 6. 清理资源：关闭数据库连接池。
#
# 交互对象：
# - 输入：PostgreSQL 数据库中的 `hf_models` 表。
# - 输出：
#   - 模型 Faiss 索引文件 (e.g., data/models_index.faiss)。
#   - 模型 Faiss ID 到模型 ID 的映射文件 (e.g., data/models_id_map.json)。
# - 配置：.env 文件和 `aigraphx/core/config.py` 中的 `settings` 对象。
# - 依赖：`PostgresRepository`, `TextEmbedder`。
# - 日志：输出到控制台。

# 导入标准库
import logging  # 日志记录
import argparse # 解析命令行参数
import asyncio  # 异步编程
import os       # 操作系统交互 (路径, 环境变量)
import json     # 处理 JSON (ID 映射)
import time     # 计时
import numpy as np # 处理数值数组 (向量)
import faiss    # Faiss 库，用于相似性搜索
import psycopg_pool # 异步 PostgreSQL 连接池
from typing import List, Optional, Dict # 类型提示
import psycopg # <--- 添加 psycopg 导入
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.errors import OperationalError as PsycopgOperationalError
from tqdm import tqdm  # type: ignore[import-untyped] # 导入 tqdm 库，用于显示进度条
from dotenv import load_dotenv

# --- 项目内部导入 ---
# 确保可以正确导入项目根目录下的模块
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从项目中导入配置和依赖类
from aigraphx.core.config import settings  # 导入 Pydantic Settings 对象
from aigraphx.repositories.postgres_repo import PostgresRepository # 导入 PG 仓库
from aigraphx.vectorization.embedder import TextEmbedder # 导入文本嵌入器

# --- 日志设置 ---
# 配置日志记录器，使用 settings 中的日志级别
logging.basicConfig(
    level=settings.log_level.upper(), # 从配置读取日志级别
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s" # 日志格式
)
logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

# --- 常量 ---
# 从 settings 对象获取配置常量
EMBEDDING_BATCH_SIZE = settings.build_faiss_batch_size  # 嵌入时每批处理的模型数量
PG_FETCH_BATCH_SIZE = 1000 # 从 PostgreSQL 一次性获取多少模型数据 (仓库方法内部使用，此脚本不直接用)
# 注意：PG_FETCH_BATCH_SIZE 可能应该移到 Repository 或 settings 中


async def build_index(
    pg_repo: PostgresRepository, # PG 仓库实例
    embedder: TextEmbedder,     # 文本嵌入器实例
    index_path: str,            # 模型 Faiss 索引保存路径
    id_map_path: str,           # 模型 ID 映射保存路径
    reset_index: bool = False,  # 是否重置索引
) -> None:
    """
    从 PostgreSQL 获取模型数据，生成嵌入向量，并构建/保存模型的 Faiss 索引。

    Args:
        pg_repo: 用于访问 PostgreSQL 的仓库实例。
        embedder: 用于生成文本向量的嵌入器实例。
        index_path: 保存模型 Faiss 索引的文件路径 (例如 'data/models_index.faiss')。
        id_map_path: 保存 Faiss 内部索引到模型 ID 映射的 JSON 文件路径 (例如 'data/models_id_map.json')。
        reset_index: 如果为 True，则在开始构建前删除已存在的 index_path 和 id_map_path 文件。
    """
    logger.info(f"开始为模型构建 Faiss 索引...")
    # 打印配置信息，方便调试
    logger.info(f"索引文件路径: {index_path}")
    logger.info(f"ID 映射文件路径: {id_map_path}")
    logger.info(f"是否重置现有索引: {reset_index}")

    # --- 重置逻辑 ---
    if reset_index:
        logger.info(f"正在删除现有索引文件（如果存在）于 {index_path} 和 {id_map_path}...")
        try:
            # 删除旧的索引文件
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"已删除: {index_path}")
            # 删除旧的 ID 映射文件
            if os.path.exists(id_map_path):
                os.remove(id_map_path)
                logger.info(f"已删除: {id_map_path}")
            logger.info("现有索引文件（如果存在）已删除。")
        except Exception as e:
            # 如果删除失败，记录警告，但可能仍然尝试继续
            logger.warning(f"删除现有索引文件时出错: {e}")

    # --- 检查索引是否已存在 (如果不重置) ---
    if os.path.exists(index_path) and not reset_index:
        logger.warning(
            f"索引文件 {index_path} 已存在且未指定 --reset。将跳过构建过程。"
            "如果需要重新构建，请使用 --reset 标志。"
        )
        # 在这里可以选择加载现有索引来验证维度等，但目前只是跳过
        return

    # --- 开始计时和初始化 ---
    start_time = time.time()
    all_model_ids: List[str] = []           # 存储所有成功处理的模型 ID
    all_embeddings_list: List[np.ndarray] = [] # 存储每个批次生成的 NumPy 向量数组
    total_models_processed = 0              # 从数据库获取并尝试处理的总模型数

    logger.info("开始从数据库获取模型数据并生成嵌入向量...")
    try:
        texts_batch: List[str] = [] # 当前批次的文本列表 (用于嵌入)
        ids_batch: List[str] = []   # 当前批次的模型 ID 列表

        # 使用 PG 仓库提供的异步生成器获取所有模型的 ID 和用于嵌入的文本
        # get_all_models_for_indexing 方法应返回 (model_id, text_to_embed)
        async for model_id, text in pg_repo.get_all_models_for_indexing():
            ids_batch.append(model_id)
            # 如果文本为 None，使用空字符串代替，避免嵌入器出错
            texts_batch.append(text if text is not None else "")
            total_models_processed += 1

            # 当批次大小达到阈值时，处理该批次
            if len(texts_batch) >= EMBEDDING_BATCH_SIZE:
                batch_num = (total_models_processed // EMBEDDING_BATCH_SIZE)
                logger.info(f"正在处理模型批次 {batch_num} (大小: {len(texts_batch)})...")
                # 调用嵌入器的批量嵌入方法
                embeddings = embedder.embed_batch(texts_batch)
                # 检查返回的嵌入向量是否有效
                if embeddings is not None and embeddings.shape[0] > 0:
                    # 将有效的嵌入向量 NumPy 数组添加到列表中
                    all_embeddings_list.append(embeddings)
                    # 将对应的模型 ID 列表扩展到总 ID 列表中
                    all_model_ids.extend(ids_batch)
                else:
                    # 如果嵌入失败或返回空，记录警告并跳过此批次
                    logger.warning(
                        f"嵌入器为以 ID {ids_batch[0]} 开头的批次返回 None 或空结果。跳过此批次。"
                    )

                # 清空当前批次，为下一批做准备
                texts_batch = []
                ids_batch = []
                # 打印进度信息
                logger.info(f"已处理模型总数: {total_models_processed}")

        # --- 处理最后一个未满的批次 ---
        if texts_batch: # 如果循环结束后仍有剩余数据
            logger.info(f"正在处理最后一个模型批次 (大小: {len(texts_batch)})...")
            embeddings = embedder.embed_batch(texts_batch)
            if embeddings is not None and embeddings.shape[0] > 0:
                all_embeddings_list.append(embeddings)
                all_model_ids.extend(ids_batch)
            else:
                logger.warning(
                    f"嵌入器为以 ID {ids_batch[0]} 开头的最后一个批次返回 None 或空结果。跳过此批次。"
                )

    except Exception as e:
        # 捕获在获取数据或生成嵌入过程中的任何异常
        logger.exception(f"获取模型或生成嵌入时出错: {e}")
        return # 中断构建过程

    # --- 检查是否有嵌入向量生成 ---
    if not all_embeddings_list:
        logger.error("未能生成任何嵌入向量。无法构建 Faiss 索引。")
        return

    logger.info(f"开始合并 {len(all_embeddings_list)} 个嵌入批次...")
    try:
        # 将列表中所有的 NumPy 数组沿第一个轴 (axis=0) 连接成一个大的 NumPy 数组
        # 使用 .astype('float32') 确保数据类型是 Faiss 通常期望的单精度浮点数
        embeddings_np = np.concatenate(all_embeddings_list, axis=0).astype("float32")
        logger.info(f"合并后的总嵌入向量形状: {embeddings_np.shape}")
        logger.info(f"收集到的总模型 ID 数量: {len(all_model_ids)}")

        # --- 验证向量数量和 ID 数量是否一致 ---
        if embeddings_np.shape[0] != len(all_model_ids):
            logger.error(
                f"嵌入向量数量 ({embeddings_np.shape[0]}) 与模型 ID 数量 ({len(all_model_ids)}) 不匹配！"
                "这可能表示在批处理或 ID 收集过程中存在错误。正在中止索引构建。"
            )
            return

    except ValueError as e:
        # 如果 NumPy 合并失败 (通常是因为各批次向量维度不一致)
        logger.error(f"合并嵌入向量时出错: {e}。请检查各批次的向量维度是否一致。")
        return
    except Exception as e:
        logger.error(f"处理嵌入向量时发生未知错误: {e}", exc_info=True)
        return


    # --- 构建 Faiss 索引 ---
    # 获取嵌入向量的维度
    embedding_dim = embedder.get_embedding_dimension()
    if embedding_dim is None or embedding_dim <= 0:
        logger.error("无法从嵌入器获取有效的向量维度。无法构建索引。")
        return

    logger.info(f"正在构建 Faiss 索引 (IndexFlatL2)，维度为 {embedding_dim}...")
    # 同样使用 IndexFlatL2 进行精确 L2 搜索
    index = faiss.IndexFlatL2(embedding_dim)

    logger.info(f"正在将 {embeddings_np.shape[0]} 个嵌入向量添加到索引中...")
    try:
        # 将合并后的 NumPy 向量数组添加到 Faiss 索引
        index.add(embeddings_np)
        logger.info(f"向量添加成功。索引中的总向量数: {index.ntotal}")
    except Exception as e:
        # 如果添加向量时出错
        logger.exception(f"向 Faiss 索引添加嵌入向量时出错: {e}")
        return

    # --- 创建 ID 映射 ---
    # 创建一个字典，将 Faiss 的内部顺序索引 (0, 1, 2, ...) 映射到原始的模型 ID (字符串)
    logger.info("正在创建 ID 映射...")
    id_map: Dict[int, str] = {i: model_id for i, model_id in enumerate(all_model_ids)}

    # --- 保存索引和 ID 映射文件 ---
    logger.info(f"正在保存 Faiss 索引到: {index_path}...")
    try:
        # 确保存储目录存在
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        # 将 Faiss 索引对象写入文件
        faiss.write_index(index, index_path)
        logger.info("模型 Faiss 索引保存成功。")
    except Exception as e:
        logger.exception(f"保存模型 Faiss 索引时出错: {e}")
        # 如果索引保存失败，后续保存映射文件可能意义不大，可以选择退出
        return

    logger.info(f"正在保存 ID 映射到: {id_map_path}...")
    try:
        # 确保存储目录存在
        os.makedirs(os.path.dirname(id_map_path), exist_ok=True)
        # 将 ID 映射字典写入 JSON 文件
        with open(id_map_path, "w", encoding='utf-8') as f:
            json.dump(id_map, f, ensure_ascii=False, indent=4)
        logger.info("模型 ID 映射保存成功。")
    except Exception as e:
        logger.exception(f"保存模型 ID 映射时出错: {e}")

    # --- 结束计时和日志 ---
    end_time = time.time()
    logger.info(
        f"模型 Faiss 索引构建完成，总耗时: {end_time - start_time:.2f} 秒。"
    )
    logger.info(f"最终索引包含 {index.ntotal} 个模型向量。")


async def main() -> None:
    """
    脚本的主异步执行函数。
    初始化数据库连接、仓库、嵌入器，并调用 build_index 函数。
    """
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从 PostgreSQL 构建 Hugging Face 模型的 Faiss 索引。"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="在开始构建前删除现有的模型 Faiss 索引和 ID 映射文件。",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 初始化连接池变量
    pool: Optional[psycopg_pool.AsyncConnectionPool] = None
    try:
        # --- 初始化数据库连接 ---
        # 验证数据库 URL 是否已配置
        db_url = settings.database_url
        if not db_url:
            logger.error("settings 中未配置 DATABASE_URL。脚本退出。")
            return

        # 创建并初始化 PG 连接池
        logger.info("正在初始化 PostgreSQL 连接池...")
        pool = psycopg_pool.AsyncConnectionPool(
            conninfo=db_url,
            min_size=settings.pg_pool_min_size,
            max_size=settings.pg_pool_max_size,
        )
        await pool.wait() # 等待连接池就绪
        pg_repo = PostgresRepository(pool=pool) # 实例化仓库，传入连接池
        logger.info("PostgreSQL 连接池初始化成功。")

        # --- 初始化嵌入器 ---
        logger.info("正在初始化文本嵌入器...")
        embedder = TextEmbedder(
            model_name=settings.sentence_transformer_model, # 从 settings 获取模型名
            device=settings.embedder_device, # 从 settings 获取设备
        )
        # 检查嵌入器是否就绪
        # if not await embedder.is_ready(): # <--- 移除此行
        #     logger.critical("文本嵌入器未能准备就绪，无法继续同步。")
        #     return
        logger.info("文本嵌入器初始化成功。")

        # --- 构建索引 ---
        # 调用 build_index 函数，传入所需的实例和配置路径
        await build_index(
            pg_repo=pg_repo,
            embedder=embedder,
            index_path=settings.models_faiss_index_path, # 使用 settings 中的模型索引路径
            id_map_path=settings.models_faiss_mapping_path, # 使用 settings 中的模型映射路径
            reset_index=args.reset, # 使用命令行参数
        )

    except psycopg_pool.PoolTimeout:
        logger.exception("数据库连接池操作超时。")
    except psycopg_pool.PoolClosed:
        logger.exception("尝试在已关闭的数据库连接池上操作。")
    except (PsycopgOperationalError, psycopg.Error) as db_err: # <--- 确保 psycopg.Error 可用
        logger.exception(f"数据库操作失败: {db_err}", exc_info=True)
    except Exception as e:
        logger.exception(f"主流程执行过程中发生意外错误: {e}")
    finally:
        # --- 清理资源 ---
        if pool:
            logger.info("正在关闭 PostgreSQL 连接池...")
            await pool.close()
            logger.info("PostgreSQL 连接池已关闭。")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # 当此脚本被直接执行时
    logger.info("开始执行脚本: sync_pg_to_models_faiss.py")
    # 使用 asyncio.run 运行异步 main 函数
    asyncio.run(main())
    logger.info("脚本执行完毕: sync_pg_to_models_faiss.py")