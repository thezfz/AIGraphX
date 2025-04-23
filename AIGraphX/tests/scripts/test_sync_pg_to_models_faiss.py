# -*- coding: utf-8 -*-
"""
文件目的：测试 `scripts/sync_pg_to_models_faiss.py` 脚本。

本测试文件 (`test_sync_pg_to_models_faiss.py`) 专注于验证 `sync_pg_to_models_faiss.py` 脚本的功能，
该脚本负责从 PostgreSQL 数据库读取 Hugging Face 模型数据，使用文本嵌入模型生成向量（基于模型元数据组合的文本），
并将这些向量构建成一个专门用于模型搜索的 Faiss 索引文件，同时保存一个 ID 映射文件。

测试策略主要是 **集成测试**：
- 使用真实的 PostgreSQL 测试数据库（通过 `conftest.py` 中的 fixture 提供连接和清理）。
- 使用真实的 `TextEmbedder` 来生成向量。
- 使用临时的 Faiss 索引和 ID 映射文件（通过 `test_settings_fixture` 提供路径）。
- 运行脚本中的核心索引构建函数 (`build_models_faiss_index`)。
- 通过查询测试数据库（确保数据已插入）和检查生成的 Faiss 文件来验证脚本的正确性。

主要交互：
- 导入 `pytest`, `numpy`, `unittest.mock`, `json`, `faiss`, `os`, `typing`, `pathlib`, `datetime`：用于测试框架、数据处理、模拟和文件/路径操作。
- 导入被测试的脚本函数：`build_index as build_models_faiss_index`。
- 导入真实的依赖组件：`PostgresRepository`, `TextEmbedder`。
- 导入测试 Fixtures：
    - 真实的 PG 仓库 `postgres_repository_fixture`。
    - 测试配置 `test_settings_fixture` (提供 Faiss 文件路径)。
- 定义测试数据：`TEST_MODEL_*_FAISS` 包含模拟的模型数据。
- 定义数据库辅助函数：`insert_faiss_model_pg_data` 用于在测试前准备数据库数据。
- 编写测试函数 (`test_build_models_faiss_integration`)：
    - 准备测试数据库数据。
    - 获取临时文件路径。
    - 初始化真实的文本嵌入器。
    - 调用核心索引构建函数。
    - 断言生成的 Faiss 索引文件和 ID 映射文件是否存在、内容是否正确、索引属性是否符合预期。

这个测试确保了模型 Faiss 索引构建脚本能够正确地处理模型数据，生成向量，并创建有效的搜索索引。
(注意：此文件目前只包含集成测试，没有像论文索引测试那样包含模拟依赖的单元测试。)
"""

import pytest # 导入 pytest 测试框架
import numpy as np # 导入 numpy 用于处理向量
from unittest.mock import (
    patch, # 用于模拟对象或函数
    AsyncMock, # 用于模拟异步对象/方法
    MagicMock, # 通用模拟对象
    mock_open, # 用于模拟文件打开操作
)
import json # 导入 json 库
import faiss  # type: ignore[import-untyped] # 导入 faiss 库 (忽略 mypy 类型检查错误)
import os # 导入 os 模块
from typing import List, Optional, Dict, Any, Tuple, cast # 导入类型提示
from pathlib import Path  # 导入 Path 对象，处理文件路径
from datetime import datetime # 导入 datetime 对象

# 导入要测试的函数，并重命名以避免命名冲突
from scripts.sync_pg_to_models_faiss import build_index as build_models_faiss_index

# 导入集成测试所需的真实组件
from aigraphx.repositories.postgres_repo import PostgresRepository # 真实的 PG 仓库类
from aigraphx.vectorization.embedder import TextEmbedder  # 真实的文本嵌入器类
from tests.conftest import (
    repository as postgres_repository_fixture, # 真实的 PG 仓库 fixture
    test_settings as test_settings_fixture,  # 测试配置 fixture，提供临时文件路径等
)
from aigraphx.core.config import Settings  # 导入 Settings 类型，用于类型提示

# --- 测试数据 ---
# 定义用于插入数据库的样本模型数据
TEST_MODEL_1_FAISS = {
    "hf_model_id": "test-faiss-model-1", # 模型 ID
    "hf_author": "faiss_author1", # 作者
    "hf_sha": "faiss_sha1", # SHA
    "hf_last_modified": datetime(2023, 7, 1, 10, 0, 0), # 最后修改时间 (datetime 对象)
    "hf_tags": json.dumps(["faiss", "model"]), # 标签 (JSON 字符串格式，模拟数据库存储)
    "hf_pipeline_tag": "feature-extraction", # Pipeline 标签
    "hf_downloads": 300, # 下载量
    "hf_likes": 30, # 点赞数
    "hf_library_name": "sentence-transformers", # 库名
    # 模拟脚本内部可能会构建的用于生成嵌入向量的文本
    "_index_text": "Model 1 for faiss testing. Author: faiss_author1. Tags: faiss, model.",
}
TEST_MODEL_2_FAISS = {
    "hf_model_id": "test-faiss-model-2",
    "hf_author": "faiss_author2",
    "hf_sha": "faiss_sha2",
    "hf_last_modified": datetime(2023, 7, 5, 12, 0, 0),
    "hf_tags": json.dumps(["image", "faiss"]), # 不同的标签
    "hf_pipeline_tag": "image-classification",
    "hf_downloads": 400,
    "hf_likes": 40,
    "hf_library_name": "timm",
    "_index_text": "Model 2 image classifier by faiss_author2.", # 不同的索引文本
}


# --- 辅助函数：在测试数据库中插入模型数据 ---
async def insert_faiss_model_pg_data(repo: PostgresRepository) -> List[str]:
    """
    辅助函数，用于在集成测试开始前向 PostgreSQL 测试数据库中插入样本模型数据。

    Args:
        repo: 已连接到测试数据库的 PostgresRepository 实例。

    Returns:
        List[str]: 插入的模型的 hf_model_id 列表。
    """
    model_ids: List[str] = [] # 用于存储插入的模型 ID
    # 获取数据库连接
    async with repo.pool.connection() as conn:
        # 创建游标
        async with conn.cursor() as cur:
            # 遍历样本模型数据
            for model_data in [TEST_MODEL_1_FAISS, TEST_MODEL_2_FAISS]:
                # 将模型 ID 添加到返回列表中
                model_ids.append(str(model_data["hf_model_id"]))
                # 执行插入语句
                # 使用 ON CONFLICT ... DO UPDATE 来处理可能的重复插入（如果测试重复运行）
                # 这里简单地用新数据更新 hf_author 字段
                await cur.execute(
                    """
                    INSERT INTO hf_models (
                        hf_model_id, hf_author, hf_sha, hf_last_modified, hf_tags,
                        hf_pipeline_tag, hf_downloads, hf_likes, hf_library_name
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (hf_model_id) DO UPDATE SET hf_author=EXCLUDED.hf_author
                    """,
                    (
                        model_data["hf_model_id"],
                        model_data["hf_author"],
                        model_data["hf_sha"],
                        model_data["hf_last_modified"],
                        model_data["hf_tags"], # 传递 JSON 字符串
                        model_data["hf_pipeline_tag"],
                        model_data["hf_downloads"],
                        model_data["hf_likes"],
                        model_data["hf_library_name"],
                    ),
                )
            # 提交事务，确保数据对后续操作可见
            await conn.commit()
    return model_ids # 返回插入的模型 ID 列表


# --- 测试用例 (集成测试) ---

@pytest.mark.asyncio # 标记为异步测试
# @pytest.mark.skip(reason="在重构期间跳过较慢的嵌入测试") # 可选跳过标记
async def test_build_models_faiss_integration(
    postgres_repository_fixture: PostgresRepository, # 请求真实 PG 仓库 fixture
    test_settings_fixture: Settings,  # 请求测试配置 fixture
    tmp_path: Path, # 请求临时路径 fixture (虽然未使用，但保持一致性)
) -> None:
    """
    集成测试：测试使用真实的数据库和嵌入器构建模型 Faiss 索引。
    """
    # --- 准备 ---
    pg_repo = postgres_repository_fixture # 获取仓库实例
    test_settings = test_settings_fixture # 获取测试配置实例
    print(f"\n[DEBUG] test_build_models_faiss_integration: Using DB pool: {pg_repo.pool}")
    print(f"[DEBUG] test_build_models_faiss_integration: Using index path: {test_settings.models_faiss_index_path}")
    print(f"[DEBUG] test_build_models_faiss_integration: Using map path: {test_settings.models_faiss_mapping_path}")

    # 1. 设置：向测试 PG 数据库插入样本模型数据
    print("[DEBUG] test_build_models_faiss_integration: Inserting test model data into PG...")
    inserted_model_ids = await insert_faiss_model_pg_data(pg_repo)
    assert len(inserted_model_ids) == 2, "未能成功插入测试模型数据"
    print(f"[DEBUG] test_build_models_faiss_integration: Inserted model IDs: {inserted_model_ids}")

    # 2. 设置：定义临时文件路径（直接使用 test_settings 中的路径）
    # 从测试配置中获取已设置为临时路径的文件名和完整路径
    index_file_name = Path(test_settings.models_faiss_index_path).name
    map_file_name = Path(test_settings.models_faiss_mapping_path).name
    temp_index_path = Path(test_settings.models_faiss_index_path)
    temp_map_path = Path(test_settings.models_faiss_mapping_path)

    # 确保这些路径的父目录存在
    temp_index_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] test_build_models_faiss_integration: Ensured directory exists: {temp_index_path.parent}")

    # 3. 设置：实例化真实的 TextEmbedder
    print("[DEBUG] test_build_models_faiss_integration: Initializing TextEmbedder...")
    try:
        embedder = TextEmbedder() # 初始化嵌入器
        dimension = embedder.get_embedding_dimension() # 获取嵌入维度
        assert isinstance(dimension, int) and dimension > 0, "嵌入器维度无效"
        print(f"[DEBUG] test_build_models_faiss_integration: Embedder initialized with dimension {dimension}.")
    except Exception as e:
        pytest.fail(
            f"初始化 TextEmbedder 失败: {e}。请确保模型已下载或网络可访问。"
        )

    # --- 脚本依赖假设 ---
    # 假设 build_models_faiss_index 使用传入的 pg_repo 和 embedder。
    # 并且假设它会正确调用仓库的 `get_all_models_for_indexing()` 方法来获取需要索引的数据。

    # --- 执行 ---
    # 4. 调用脚本中的索引构建函数
    print("[DEBUG] test_build_models_faiss_integration: Calling build_models_faiss_index...")
    await build_models_faiss_index(
        pg_repo=pg_repo, # 传入真实仓库
        embedder=embedder, # 传入真实嵌入器
        index_path=str(temp_index_path), # 使用测试配置中的路径
        id_map_path=str(temp_map_path), # 使用测试配置中的路径
        reset_index=True, # !!! 关键：设置为 True 确保每次测试都在干净状态下构建 !!!
    )
    print("[DEBUG] test_build_models_faiss_integration: build_models_faiss_index finished.")

    # --- 断言 ---
    # 5. 检查 Faiss 索引文件和 ID 映射文件是否已在临时路径中创建
    print("[DEBUG] test_build_models_faiss_integration: Asserting file existence...")
    assert temp_index_path.exists(), f"模型 Faiss 索引文件未找到: {temp_index_path}"
    assert temp_map_path.exists(), f"模型 ID 映射文件未找到: {temp_map_path}"

    # 6. 检查 ID 映射文件的内容
    print("[DEBUG] test_build_models_faiss_integration: Asserting ID map content...")
    with open(temp_map_path, "r") as f:
        loaded_id_map = json.load(f)
    # JSON 加载后 key 是字符串，需要转回整数
    loaded_id_map = {int(k): v for k, v in loaded_id_map.items()}
    # 映射大小应等于插入的模型数量
    assert len(loaded_id_map) == len(inserted_model_ids), "模型 ID 映射大小不匹配"
    # 映射的值（模型ID）应与插入的ID一致（比较集合）
    assert set(loaded_id_map.values()) == set(inserted_model_ids), "模型 ID 映射中的模型 ID 不匹配"
    # 映射的键（Faiss 内部索引）应是 0 到 n-1
    assert set(loaded_id_map.keys()) == set(range(len(inserted_model_ids))), "模型 ID 映射的 Faiss 索引键不正确"

    # 7. 检查生成的 Faiss 索引的基本属性
    print("[DEBUG] test_build_models_faiss_integration: Asserting Faiss index properties...")
    try:
        loaded_index = faiss.read_index(str(temp_index_path)) # 读取索引
        # 向量总数应等于插入的模型数量
        assert loaded_index.ntotal == len(inserted_model_ids), "模型 Faiss 索引中的向量总数不匹配"
        # 维度应与嵌入器维度一致
        assert loaded_index.d == dimension, "模型 Faiss 索引的维度与嵌入器维度不匹配"
    except Exception as e:
        pytest.fail(f"读取或验证生成的模型 Faiss 索引失败: {e}")

    print("[DEBUG] test_build_models_faiss_integration: All assertions passed.")


# 注意：此文件目前没有为模型索引构建脚本提供模拟的单元测试。
# 如果将来需要添加单元测试（例如，为了更快地测试错误处理或特定逻辑而不依赖真实数据库/嵌入器），
# 它们也应该进行更新以使用 test_settings fixture 来获取文件路径。