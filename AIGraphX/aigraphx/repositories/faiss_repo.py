# -*- coding: utf-8 -*-
"""
FaissRepository - Faiss 索引交互模块

该模块定义了 `FaissRepository` 类，封装了与 Faiss 索引文件和 ID 映射文件的交互逻辑。
Faiss 是一个用于高效相似性搜索和密集向量聚类的库。这个仓库负责加载预先构建好的
Faiss 索引和对应的 ID 映射，并提供执行相似性搜索的方法。

主要功能:
- 加载 Faiss 索引文件 (`.index`)。
- 加载 ID 映射文件 (`.json`)，该文件将 Faiss 内部的顺序索引映射回原始数据 ID（如论文 ID 或模型 ID）。
- 支持不同类型的原始 ID（整数或字符串）。
- 提供异步的相似性搜索方法 (`search_similar`)，输入查询向量，返回最相似的 K 个原始 ID 及其距离。
- 提供检查索引和映射是否成功加载的方法 (`is_ready`)。
- 提供获取索引中向量数量的方法 (`get_index_size`)。
- 使用 `asyncio.Lock` 确保对 Faiss 索引的并发访问安全（因为 Faiss 的 Python 绑定可能不是完全线程安全的）。

与其他文件的交互:
- **`aigraphx.services.search_service.SearchService`**: `SearchService` 依赖 `FaissRepository` 来执行语义搜索和混合搜索中的向量相似性查找部分。它会为不同的目标（论文、模型）创建不同的 `FaissRepository` 实例。
- **`aigraphx.scripts.build_vector_index`**: 构建 Faiss 索引的脚本会生成 `.index` 文件和 `.json` ID 映射文件，供 `FaissRepository` 加载和使用。
- **`numpy`**: 用于处理嵌入向量。
- **`faiss`**: Faiss 库本身，用于加载索引和执行搜索。
- **`json`**: 用于加载 JSON 格式的 ID 映射。
- **`asyncio`**: 用于异步锁和异步执行线程阻塞操作。

注意事项:
- Faiss 索引和 ID 映射文件必须预先存在且路径正确。
- `id_type` 参数必须与 ID 映射文件中的 ID 类型匹配。
- `search_similar` 方法返回的是距离（通常是 L2 距离），需要根据需要转换为相似度分数（如 SearchService 中的 `_convert_distance_to_score`）。
"""

import logging  # 日志记录
import os  # 操作系统交互，检查文件是否存在
import pickle  # 虽然未在此版本直接使用，但有时用于序列化 Python 对象
from typing import List, Tuple, Optional, Dict, Union  # 类型提示
import numpy as np  # NumPy 用于处理向量
import faiss  # type: ignore[import-untyped] # Faiss 库 (mypy 可能无法直接识别类型)
import json  # 用于加载 JSON 格式的 ID 映射
import asyncio  # 用于异步锁和异步执行线程阻塞操作
from typing import Literal  # 用于精确指定 id_type 的可能值
from unittest.mock import patch  # 如果需要模拟错误，保留 patch

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class FaissRepository:
    """
    用于与 Faiss 索引交互的仓库类。
    """

    def __init__(
        self,
        index_path: str = "data/papers_faiss.index",  # Faiss 索引文件路径
        id_map_path: str = "data/papers_faiss_ids.json",  # ID 映射文件路径
        id_type: Literal["int", "str"] = "int",  # 原始 ID 的预期类型 ('int' 或 'str')
    ):
        """
        初始化 FaissRepository，加载索引和 ID 映射。

        Args:
            index_path (str): 预构建的 Faiss 索引文件的路径。
            id_map_path (str): JSON 文件的路径，该文件将 Faiss 索引位置映射到原始 ID。
            id_type (Literal["int", "str"]): 原始 ID 的预期类型 ('int' 或 'str')。
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.index: Optional[faiss.Index] = None  # Faiss 索引对象，初始为 None
        # ID 映射字典：键是 Faiss 内部索引 (int)，值是原始 ID (Union[int, str])
        self.id_map: Dict[int, Union[int, str]] = {}
        self.id_type = id_type  # 存储预期的 ID 类型
        self._lock = asyncio.Lock()  # 异步锁，用于保护对索引的访问

        # 初始化时尝试加载索引和 ID 映射
        self._load_index()
        self._load_id_map()

    def _load_index(self) -> None:
        """
        从指定文件加载 Faiss 索引。
        """
        # 检查索引文件是否存在
        if not os.path.exists(self.index_path):
            logger.error(
                f"Faiss 索引文件未在 {self.index_path} 找到。搜索功能将无法工作。"
            )
            self.index = None
            return

        try:
            logger.info(f"正在从 {self.index_path} 加载 Faiss 索引...")
            # 使用 faiss.read_index 读取索引文件
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Faiss 索引加载成功。索引包含 {self.index.ntotal} 个向量。")
            # 如果加载的索引为空，发出警告
            if self.index.ntotal == 0:
                logger.warning("加载的 Faiss 索引为空。")
        except Exception as e:
            # 捕获加载过程中的任何异常
            logger.error(
                f"从 {self.index_path} 加载 Faiss 索引失败: {e}",
                exc_info=True,  # 记录堆栈跟踪
            )
            self.index = None  # 加载失败时将索引设为 None

    def _load_id_map(self) -> None:
        """
        从指定的 JSON 文件加载 ID 映射。
        """
        # 检查 ID 映射文件是否存在
        if not os.path.exists(self.id_map_path):
            logger.error(
                f"Faiss ID 映射文件未在 {self.id_map_path} 找到。搜索结果映射将失败。"
            )
            self.id_map = {}  # 文件不存在则映射为空
            return

        try:
            logger.info(f"正在从 {self.id_map_path} 加载 Faiss ID 映射...")
            with open(self.id_map_path, "r") as f:
                # JSON 加载后，键总是字符串，需要转换回整数
                loaded_map_str_keys = json.load(f)
                # 根据初始化时指定的 id_type 确定值转换函数
                value_converter = int if self.id_type == "int" else str
                # 构建最终的 id_map，转换键和值
                self.id_map = {
                    int(k): value_converter(v) for k, v in loaded_map_str_keys.items()
                }
            logger.info(f"Faiss ID 映射加载成功。映射包含 {len(self.id_map)} 个条目。")
        except (json.JSONDecodeError, ValueError, IOError) as e:
            # 捕获文件读取、JSON 解析或类型转换错误
            logger.error(
                f"从 {self.id_map_path} 加载或解析 Faiss ID 映射失败: {e}",
                exc_info=True,
            )
            self.id_map = {}  # 加载失败时映射为空

    def is_ready(self) -> bool:
        """
        检查索引和 ID 映射是否已成功加载并可用。

        Returns:
            bool: 如果索引已加载、包含向量且 ID 映射已加载且不为空，则返回 True。
        """
        # 检查索引对象是否存在
        index_exists = self.index is not None
        # 获取索引中的向量总数，如果索引为 None 则为 -1
        ntotal = self.index.ntotal if self.index else -1
        # 检查 ID 映射是否已加载且包含条目
        map_exists_and_not_empty = bool(self.id_map)
        # 最终准备状态：索引存在、向量数大于 0、映射存在且非空
        ready_status = index_exists and ntotal > 0 and map_exists_and_not_empty
        # 记录详细的检查信息，便于调试
        logger.info(
            f"[is_ready Check - Instance ID: {id(self)}] index_exists={index_exists}, index.ntotal={ntotal}, map_exists_and_not_empty={map_exists_and_not_empty} -> Returning: {ready_status}"
        )
        return ready_status

    async def search_similar(
        self, embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        在 Faiss 索引中搜索与查询嵌入向量相似的向量。

        Args:
            embedding (np.ndarray): 查询嵌入向量 (numpy 数组, float32)。
            k (int): 要检索的最近邻居的数量。

        Returns:
            List[Tuple[Union[int, str], float]]:
                一个元组列表，每个元组包含 (原始 ID, 距离)。
                原始 ID 的类型取决于初始化时指定的 `id_type`。
                如果索引未准备好或搜索失败，则返回空列表。
        """
        # 首先检查仓库是否准备就绪
        if not self.is_ready():
            logger.warning("Faiss 索引或 ID 映射未准备好。返回空的搜索结果。")
            return []

        # 再次确认索引对象存在 (理论上 is_ready 应该已检查)
        if self.index is None:
            logger.error("尝试搜索时索引为 None。")
            return []

        # --- 输入向量验证 ---
        if not isinstance(embedding, np.ndarray):
            logger.error("无效的查询嵌入类型。期望为 numpy 数组。")
            return []
        # 处理一维或二维（单行）输入向量
        if embedding.ndim == 1:
            query_vector = embedding.reshape(1, -1).astype(np.float32)
        elif embedding.ndim == 2 and embedding.shape[0] == 1:
            query_vector = embedding.astype(np.float32)
        else:
            logger.error(
                f"无效的查询嵌入形状: {embedding.shape}。期望为 (维度,) 或 (1, 维度)。"
            )
            return []

        # 检查查询向量维度是否与索引维度匹配
        if query_vector.shape[1] != self.index.d:
            logger.error(
                f"查询嵌入维度 ({query_vector.shape[1]}) 与索引维度 ({self.index.d}) 不匹配。"
            )
            return []

        # --- 确定实际搜索数量 K ---
        # 实际 K 不能超过索引中的向量总数
        actual_k = min(k, self.index.ntotal)
        if actual_k <= 0:
            logger.warning("搜索的 K 值为 0 或索引为空。返回空结果。")
            return []

        logger.debug(f"正在执行 Faiss 搜索，查找 {actual_k} 个邻居...")
        try:
            # 使用异步锁保护 Faiss 搜索操作
            async with self._lock:
                # Faiss 的 search 方法是阻塞的，使用 asyncio.to_thread 在单独线程中运行
                distances, indices = await asyncio.to_thread(
                    self.index.search, query_vector, actual_k
                )

            # --- 处理搜索结果 ---
            results: List[Tuple[Union[int, str], float]] = []
            # 确保返回的 indices 和 distances 数组不为空
            if indices.size > 0 and distances.size > 0:
                # Faiss 返回的是二维数组，即使只查询一个向量，也取第一行
                faiss_indices = indices[0]
                faiss_distances = distances[0]

                # 遍历返回的 Faiss 内部索引和距离
                for i, faiss_idx in enumerate(faiss_indices):
                    # Faiss 可能返回 -1 表示没有更多邻居
                    if faiss_idx == -1:
                        # logger.warning( # 通常不需要记录这个，是正常行为
                        #     f"Faiss 搜索在位置 {i} 返回无效索引 -1。跳过。"
                        # )
                        continue
                    # 从 ID 映射中查找原始 ID
                    original_id = self.id_map.get(int(faiss_idx))
                    if original_id is not None:
                        # 如果找到原始 ID，将其与距离一起添加到结果列表
                        distance = float(faiss_distances[i])
                        results.append((original_id, distance))
                    else:
                        # 如果在映射中找不到对应的原始 ID，记录警告
                        logger.warning(
                            f"无法在 id_map 中为 Faiss 索引 {faiss_idx} 找到原始 ID。跳过。"
                        )

            logger.debug(f"Faiss 搜索完成。找到 {len(results)} 个有效结果。")
            return results

        except Exception as e:
            # 捕获搜索过程中的任何异常
            logger.error(f"Faiss 搜索期间出错: {e}", exc_info=True)
            return []

    def get_index_size(self) -> int:
        """
        返回索引中当前包含的向量数量。
        """
        # 如果索引已加载，返回其 ntotal 属性，否则返回 0
        return self.index.ntotal if self.index else 0
