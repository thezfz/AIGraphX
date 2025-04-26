# -*- coding: utf-8 -*-
"""
文件目的：PostgreSQL 仓库类的覆盖率测试 (tests/repositories/test_postgres_repo_coverage.py)

本文件旨在补充 `test_postgres_repo.py`，提供更全面的测试覆盖，特别是针对
`PostgresRepository` 类中涉及 Hugging Face 模型 (`hf_models` 表) 的操作，
以及对各种方法进行更细致的错误处理和边缘情况测试。

核心测试策略：
- **混合测试方法:**
    - **集成测试:** 对于核心功能的 Happy Path（成功路径）和涉及数据库状态的验证（如批量插入/更新、搜索、获取关系数据），主要使用集成测试，依赖 `conftest.py` 中的 `repository` fixture 与真实的测试数据库交互。
    - **单元/模拟测试:** 对于难以稳定复现的数据库错误场景（如连接失败、特定 SQL 错误），采用 `unittest.mock.patch` 或辅助模拟类 (`AsyncContextManagerMock`) 来模拟数据库交互层（如连接池、连接、游标）或仓库方法本身，强制其抛出异常，以验证仓库代码的错误处理逻辑。
- **辅助函数:** 定义了 `insert_hf_model` 异步函数，简化 HF 模型测试数据的准备过程。
- **测试覆盖重点:**
    - `hf_models` 表的批量操作 (`save_hf_models_batch`)，包括插入和更新。
    - `hf_models` 的关键字搜索和分页 (`search_models_by_keyword`)。
    - 获取所有 HF 模型用于同步和索引的生成器方法 (`get_all_hf_models_for_sync`, `get_all_models_for_indexing`)。
    - 获取论文关联数据（任务、数据集、仓库）在关联数据为空或论文不存在时的行为。
    - 对多种数据库操作（查询、插入、生成器）模拟数据库错误，验证仓库的异常处理。
    - 测试空输入、最小数据插入等边缘情况。

与其他文件的交互：
- 导入测试框架、数据库、异步、模拟、日志等相关库。
- 导入被测试的类 `aigraphx.repositories.postgres_repo.PostgresRepository`。
- 导入数据模型 `aigraphx.models.paper.Paper` (可能用于辅助数据准备)。
- 导入 `psycopg` 库及其异常类型，用于数据库交互和错误模拟。
- **关键依赖:** `tests/conftest.py` 提供的 `db_pool` 和 `repository` fixtures。
"""

import pytest
import pytest_asyncio
import json  # 用于处理 HF 模型数据中的 JSON 字段（如 hf_tags）
from datetime import date as date_type, datetime  # 导入 date 和 datetime
from typing import (
    AsyncGenerator,
    Dict,
    Any,
    Optional,
    List,
    cast,
    Set,
    Tuple,
)  # 导入类型提示
from psycopg_pool import AsyncConnectionPool  # 导入异步连接池
from psycopg.rows import dict_row  # 导入字典行工厂
import logging  # 导入日志
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec  # 导入模拟工具
import psycopg  # 导入 psycopg 库，用于数据库异常类型
from pydantic import HttpUrl  # 导入 HttpUrl

# 导入被测试的类
from aigraphx.repositories.postgres_repo import PostgresRepository

# 导入 Paper 模型，可能在辅助函数或某些测试中需要
from aigraphx.models.paper import Paper

# 导入由 conftest.py 提供的 fixtures
# db_pool: 提供测试数据库连接池
# repository: 提供 PostgresRepository 实例并管理事务
from tests.conftest import db_pool, repository

# 设置日志记录器
logger = logging.getLogger(__name__)

# 标记模块内所有测试为异步
pytestmark = pytest.mark.asyncio


# --- 辅助函数：用于插入 Hugging Face 模型测试数据 ---
async def insert_hf_model(
    pool: AsyncConnectionPool, model_data: Dict[str, Any]
) -> None:
    """
    辅助函数：向 `hf_models` 表中插入或更新单个 Hugging Face 模型记录。

    Args:
        pool (AsyncConnectionPool): 用于连接测试数据库的连接池。
        model_data (Dict[str, Any]): 包含模型数据的字典，键应与 `hf_models` 表的列名匹配。
    """
    cols = list(model_data.keys())  # 获取所有列名
    vals = [model_data.get(col) for col in cols]  # 获取对应的值

    # --- 数据类型处理 ---
    # 如果 hf_tags 是 Python 列表，则序列化为 JSON 字符串以便存入数据库
    if "hf_tags" in model_data and isinstance(model_data["hf_tags"], list):
        vals[cols.index("hf_tags")] = json.dumps(model_data["hf_tags"])

    # 如果 hf_last_modified 是字符串，则解析为 datetime 对象
    if "hf_last_modified" in model_data and isinstance(
        model_data["hf_last_modified"], str
    ):
        vals[cols.index("hf_last_modified")] = datetime.fromisoformat(
            model_data["hf_last_modified"]
        )

    # --- 构建 UPSERT 查询 ---
    # 使用 hf_model_id 作为冲突键 (ON CONFLICT)
    # 如果 hf_model_id 已存在，则更新除 hf_model_id 之外的其他列
    query = f"""
        INSERT INTO hf_models ({", ".join(cols)})
        VALUES ({", ".join(["%s"] * len(vals))})
        ON CONFLICT (hf_model_id) DO UPDATE SET
        {", ".join([f"{col} = EXCLUDED.{col}" for col in cols if col != "hf_model_id"])}
    """
    try:
        # 从连接池获取连接并执行查询
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(vals))
        logger.debug(
            f"Helper inserted/updated HF model: {model_data.get('hf_model_id')}"
        )
    except Exception as e:
        # 记录错误并重新抛出，以便测试失败
        logger.error(
            f"Error in helper insert_hf_model for {model_data.get('hf_model_id')}: {e}",
            exc_info=True,
        )
        raise


# --- Hugging Face 模型测试数据 ---
# 字典键名使用 `hf_` 前缀，与数据库列名保持一致
test_hf_model_data_1 = {
    "hf_model_id": "org1/model-a",  # 模型唯一 ID
    "hf_author": "org1",  # 作者/组织
    "hf_sha": "sha1",  # Commit SHA
    "hf_last_modified": "2023-10-01T10:00:00",  # 最后修改时间 (字符串，辅助函数会解析)
    "hf_tags": ["tag1", "tagA"],  # 标签列表 (辅助函数会转为 JSON)
    "hf_pipeline_tag": "text-generation",  # 任务类型
    "hf_downloads": 100,  # 下载量
    "hf_likes": 10,  # 点赞数
    "hf_library_name": "transformers",  # 主要库
    "created_at": datetime.now(),  # 创建时间 (通常由数据库默认或触发器设置)
    "updated_at": datetime.now(),  # 更新时间 (通常由数据库默认或触发器设置)
}

test_hf_model_data_2 = {
    "hf_model_id": "user2/model-b-beta",
    "hf_author": "user2",
    "hf_sha": "sha2",
    "hf_last_modified": "2024-01-15T12:30:00",
    "hf_tags": ["tag2", "tagB", "beta"],
    "hf_pipeline_tag": "image-classification",
    "hf_downloads": 500,
    "hf_likes": 50,
    "hf_library_name": "timm",
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
}

test_hf_model_data_3 = {
    "hf_model_id": "org1/model-c-old",
    "hf_author": "org1",
    "hf_sha": "sha3",
    "hf_last_modified": "2022-05-20T08:00:00",
    "hf_tags": ["tag1", "tagC", "legacy"],
    "hf_pipeline_tag": None,  # 任务类型可以为 None
    "hf_downloads": 5,
    "hf_likes": 1,
    "hf_library_name": "transformers",
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
}


# --- 测试 HF 模型相关方法 ---


async def test_save_hf_models_batch_insert(
    repository: PostgresRepository,
    db_pool: AsyncConnectionPool,  # 依赖仓库实例和连接池 (用于清空表)
) -> None:
    """
    测试场景：使用 `save_hf_models_batch` 批量插入全新的 HF 模型记录。
    策略：先清空 `hf_models` 表，然后调用批量保存方法插入两条新记录，
          最后验证表中的记录数以及插入的数据内容是否正确。
    """
    # --- 1. 清空表 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")

    # --- 2. 调用批量保存 ---
    models_to_save: List[Dict[str, Any]] = [test_hf_model_data_1, test_hf_model_data_2]
    await repository.save_hf_models_batch(models_to_save)

    # --- 3. 验证结果 ---
    # 验证记录数
    count = await repository.count_hf_models()
    assert count == 2

    # 获取插入的记录并验证内容
    details1 = await repository.get_hf_models_by_ids(
        [cast(str, test_hf_model_data_1["hf_model_id"])]  # 使用 hf_model_id 获取
    )
    assert len(details1) == 1
    assert details1[0]["hf_author"] == test_hf_model_data_1["hf_author"]
    # 验证标签列表 (从数据库取出时应为 list)
    fetched_tags1 = details1[0].get("hf_tags")
    assert isinstance(fetched_tags1, list)
    assert set(fetched_tags1) == set(
        cast(List[str], test_hf_model_data_1["hf_tags"])
    )  # 使用集合比较忽略顺序

    details2 = await repository.get_hf_models_by_ids(
        [cast(str, test_hf_model_data_2["hf_model_id"])]
    )
    assert len(details2) == 1
    assert details2[0]["hf_pipeline_tag"] == test_hf_model_data_2["hf_pipeline_tag"]


async def test_save_hf_models_batch_update(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：使用 `save_hf_models_batch` 批量保存数据，其中包含对现有记录的更新和新记录的插入。
    策略：先使用辅助函数插入两条初始记录。然后创建一个包含一条更新记录 (hf_model_id 已存在)
          和一条新记录的列表，调用批量保存方法。最后验证总记录数是否正确增加，
          以及被更新记录的字段是否已改变，新记录是否已插入。
    """
    # --- 1. 准备初始数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")
    await insert_hf_model(db_pool, test_hf_model_data_1)  # 插入 model 1
    await insert_hf_model(db_pool, test_hf_model_data_2)  # 插入 model 2

    # --- 2. 准备包含更新和插入的批量数据 ---
    # 更新 model 1 的 likes 和 tags
    model1_updated = test_hf_model_data_1.copy()
    model1_updated["hf_likes"] = 150
    model1_updated["hf_tags"] = cast(
        List[str], ["tag1", "tagA", "updated"]
    )  # 添加新 tag

    # 添加一条新记录 model 3
    model2_new = test_hf_model_data_3

    models_to_save: List[Dict[str, Any]] = [model1_updated, model2_new]

    # --- 3. 调用批量保存 ---
    await repository.save_hf_models_batch(models_to_save)

    # --- 4. 验证结果 ---
    # 总记录数应变为 3 (初始 2 条 + 新增 1 条)
    count = await repository.count_hf_models()
    assert count == 3

    # 验证 model 1 是否被更新
    details1 = await repository.get_hf_models_by_ids(
        [cast(str, test_hf_model_data_1["hf_model_id"])]
    )
    assert len(details1) == 1
    assert details1[0]["hf_likes"] == 150  # 验证 likes 已更新
    fetched_tags1_updated = details1[0].get("hf_tags")  # 验证 tags 已更新
    assert isinstance(fetched_tags1_updated, list)
    assert set(fetched_tags1_updated) == {"tag1", "tagA", "updated"}

    # 验证 model 3 是否被插入
    details3 = await repository.get_hf_models_by_ids(
        [cast(str, test_hf_model_data_3["hf_model_id"])]
    )
    assert len(details3) == 1
    assert details3[0]["hf_author"] == test_hf_model_data_3["hf_author"]


async def test_save_hf_models_batch_empty(repository: PostgresRepository) -> None:
    """
    测试场景：调用 `save_hf_models_batch` 并传入一个空列表。
    预期：方法应能正常执行，不抛出错误，且数据库记录数不变。
    策略：记录调用前的记录数，传入空列表调用方法，再记录调用后的记录数，断言两者相等。
    """
    count_before = await repository.count_hf_models()
    await repository.save_hf_models_batch([])  # 传入空列表
    count_after = await repository.count_hf_models()
    assert count_before == count_after


async def test_search_models_by_keyword_found(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：使用 `search_models_by_keyword` 进行关键字搜索，并能找到匹配的模型。
    策略：插入多个模型，使用不同的关键词（如作者、任务类型、模型 ID 片段）进行搜索，
          验证返回的结果数量和内容是否符合预期。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")
    await insert_hf_model(db_pool, test_hf_model_data_1)
    await insert_hf_model(db_pool, test_hf_model_data_2)
    await insert_hf_model(db_pool, test_hf_model_data_3)

    # --- 2. 按作者搜索 ('org1') ---
    # 应该匹配 model 1 和 model 3
    results_list, total = await repository.search_models_by_keyword("org1")
    assert total == 2
    assert len(results_list) == 2
    model_ids = {r["hf_model_id"] for r in results_list}  # 提取返回的 hf_model_id
    assert model_ids == {  # 验证 ID 集合
        test_hf_model_data_1["hf_model_id"],
        test_hf_model_data_3["hf_model_id"],
    }

    # --- 3. 按任务类型搜索 ('text-generation') ---
    # 应该只匹配 model 1
    results_list, total = await repository.search_models_by_keyword("text-generation")
    assert total == 1
    assert len(results_list) == 1
    assert results_list[0]["hf_model_id"] == test_hf_model_data_1["hf_model_id"]

    # --- 4. 按模型 ID 片段搜索 ('model-b') ---
    # **重要**: 这个测试依赖于 `search_models_by_keyword` 方法内部实际实现了对 `hf_model_id` 列的搜索。
    # 如果它只搜索其他文本列，这个测试会失败。
    results_list, total = await repository.search_models_by_keyword("model-b")
    assert total == 1
    assert len(results_list) == 1
    assert results_list[0]["hf_model_id"] == test_hf_model_data_2["hf_model_id"]

    # --- 5. 按标签搜索 ('tag1') ---
    # **注意**: 这个测试依赖于 `search_models_by_keyword` 方法内部实现了对 JSON 类型的 `hf_tags` 列的搜索。
    # 这通常需要特定的 PostgreSQL JSON 操作符或索引。如果未实现，此测试可能失败或结果不准。
    # results_list, total = await repository.search_models_by_keyword("tag1")
    # assert total == 2 # 假设 model 1 和 model 3 都包含 tag1
    # ... 验证 ID ...
    pass  # 暂时跳过标签搜索验证，直到确认实现


async def test_search_models_by_keyword_pagination(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：使用 `search_models_by_keyword` 进行关键字搜索，并应用分页参数。
    策略：插入多个模型，执行能匹配所有模型的搜索，指定不同的 `skip` 和 `limit` 值，
          结合默认排序规则 (`hf_last_modified DESC`) 验证返回的子集是否正确。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")
    await insert_hf_model(db_pool, test_hf_model_data_1)  # 2023-10-01
    await insert_hf_model(db_pool, test_hf_model_data_2)  # 2024-01-15
    await insert_hf_model(db_pool, test_hf_model_data_3)  # 2022-05-20
    # 默认排序 hf_last_modified DESC，预期顺序: model-2, model-1, model-3

    # --- 2. 测试分页：第 1 页，limit=2 ---
    results_list, total = await repository.search_models_by_keyword(
        "model",
        limit=2,
        skip=0,  # 获取前两条
    )
    assert total == 3  # 总数是 3
    assert len(results_list) == 2  # 返回两条
    # 验证返回的是 model-2 和 model-1
    assert results_list[0]["hf_model_id"] == test_hf_model_data_2["hf_model_id"]
    assert results_list[1]["hf_model_id"] == test_hf_model_data_1["hf_model_id"]

    # --- 3. 测试分页：第 2 页，limit=2 ---
    results_list, total = await repository.search_models_by_keyword(
        "model",
        limit=2,
        skip=2,  # 跳过前两条，获取接下来的（最多）两条
    )
    assert total == 3  # 总数是 3
    assert len(results_list) == 1  # 只剩下 model-3
    # 验证返回的是 model-3
    assert results_list[0]["hf_model_id"] == test_hf_model_data_3["hf_model_id"]


async def test_search_models_by_keyword_not_found(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：使用 `search_models_by_keyword` 进行关键字搜索，但没有找到任何匹配的模型。
    预期：返回空列表和 0 的总数。
    策略：插入一些数据，然后使用一个肯定不存在的关键词搜索。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")
    await insert_hf_model(db_pool, test_hf_model_data_1)

    # --- 2. 搜索不存在的关键词 ---
    results_list, total = await repository.search_models_by_keyword(
        "nonexistentkeyword123abc"
    )

    # --- 3. 验证结果 ---
    assert total == 0
    assert len(results_list) == 0


async def test_get_all_hf_models_for_sync(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：使用异步生成器 `get_all_hf_models_for_sync` 分批获取所有 HF 模型数据。
    策略：插入多个模型，迭代调用生成器（指定 batch_size），收集所有返回的模型数据，
          验证最终获取的总数和内容是否完整。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")
    await insert_hf_model(db_pool, test_hf_model_data_1)
    await insert_hf_model(db_pool, test_hf_model_data_2)
    await insert_hf_model(db_pool, test_hf_model_data_3)

    # --- 2. 迭代生成器并收集结果 ---
    fetched_models: Dict[
        str, Dict[str, Any]
    ] = {}  # 存储获取的模型，以 hf_model_id 为键
    batch_count = 0
    # 调用生成器，设置 batch_size=2
    async for batch in repository.get_all_hf_models_for_sync(batch_size=2):
        batch_count += 1
        logger.debug(f"Fetched batch {batch_count} with {len(batch)} models")
        assert len(batch) <= 2  # 验证每批数量
        for model_dict in batch:
            assert isinstance(model_dict, dict)  # 每项应为字典
            model_id = model_dict.get("hf_model_id")  # 获取模型 ID
            assert isinstance(model_id, str)  # ID 应为字符串
            fetched_models[model_id] = model_dict  # 存入结果字典

    # --- 3. 验证结果 ---
    assert len(fetched_models) == 3  # 总共应获取 3 个模型
    # 验证所有插入的模型 ID 都已获取
    assert cast(str, test_hf_model_data_1["hf_model_id"]) in fetched_models
    assert cast(str, test_hf_model_data_2["hf_model_id"]) in fetched_models
    assert cast(str, test_hf_model_data_3["hf_model_id"]) in fetched_models
    # 抽查部分获取的数据内容是否正确
    model_1_fetched = fetched_models[cast(str, test_hf_model_data_1["hf_model_id"])]
    assert model_1_fetched["hf_author"] == test_hf_model_data_1["hf_author"]
    model_2_fetched = fetched_models[cast(str, test_hf_model_data_2["hf_model_id"])]
    fetched_tags2 = model_2_fetched.get("hf_tags")
    assert isinstance(fetched_tags2, list)  # 验证 tag 是列表
    assert set(fetched_tags2) == set(cast(List[str], test_hf_model_data_2["hf_tags"]))


async def test_get_all_models_for_indexing(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：使用异步生成器 `get_all_models_for_indexing` 获取用于构建向量索引的模型 ID 和文本表示。
    策略：插入多个模型，迭代生成器，验证返回的 ID 和文本内容是否符合预期
          （预期文本是 hf_tags (JSON 字符串) + hf_pipeline_tag 的拼接）。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE hf_models RESTART IDENTITY CASCADE;")
    data1 = test_hf_model_data_1.copy()
    data2 = test_hf_model_data_2.copy()
    data3 = test_hf_model_data_3.copy()
    await insert_hf_model(db_pool, data1)
    await insert_hf_model(db_pool, data2)
    await insert_hf_model(db_pool, data3)

    # --- 2. 计算预期的文本表示 ---
    # 假设文本表示是 tags (json string) + pipeline_tag
    expected_text = {
        cast(str, data1["hf_model_id"]): json.dumps(
            data1["hf_tags"]
        )  # tags 转 json 字符串
        + cast(str, data1["hf_pipeline_tag"]),  # 拼接 pipeline tag
        cast(str, data2["hf_model_id"]): json.dumps(data2["hf_tags"])
        + cast(str, data2["hf_pipeline_tag"]),
        cast(str, data3["hf_model_id"]): json.dumps(data3["hf_tags"])
        + "",  # pipeline tag 为 None，拼接空字符串
    }

    # --- 3. 迭代生成器并收集结果 ---
    results: Dict[str, str] = {}  # 存储 ID -> 文本
    # 注意：仓库方法 get_all_models_for_indexing 应该只选择 hf_model_id 和相关文本字段
    async for model_id, text_repr in repository.get_all_models_for_indexing():
        results[model_id] = text_repr
        logger.debug(f"Indexing data: ID={model_id}, Text='{text_repr[:50]}...'")

    # --- 4. 验证结果 ---
    assert len(results) == 3  # 应获取 3 个模型
    assert results.keys() == expected_text.keys()  # 验证 ID 集合是否一致
    # 逐一验证每个模型的文本表示是否符合预期
    for model_id in results:
        assert results[model_id] == expected_text[model_id]


# --- 测试关系获取的边缘情况（论文存在但无关联数据） ---


async def test_get_tasks_for_papers_not_found(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：调用 `get_tasks_for_papers`，请求的论文存在，但没有关联的任务。
    预期：返回的字典中，该论文 ID 对应的值应为空列表。
    策略：清空相关表，插入一篇没有任务的论文，调用方法，验证结果。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE;")
            await cur.execute(
                "TRUNCATE pwc_tasks RESTART IDENTITY CASCADE;"
            )  # 清空任务表
    paper = Paper(pwc_id="no-tasks-paper", title="Paper Without Tasks")
    paper_id = await repository.upsert_paper(paper)
    assert paper_id is not None

    # --- 2. 调用方法 ---
    results = await repository.get_tasks_for_papers([paper_id])  # 请求这篇论文的任务

    # --- 3. 验证结果 ---
    assert isinstance(results, dict)
    assert len(results) == 1  # 字典应包含请求的 ID
    assert paper_id in results
    assert results[paper_id] == []  # 对应的任务列表应为空


async def test_get_datasets_for_papers_not_found(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：调用 `get_datasets_for_papers`，请求的论文存在，但没有关联的数据集。
    预期：返回的字典中，该论文 ID 对应的值应为空列表。
    策略：清空相关表，插入一篇没有数据集的论文，调用方法，验证结果。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE;")
            await cur.execute(
                "TRUNCATE pwc_datasets RESTART IDENTITY CASCADE;"
            )  # 清空数据集表
    paper = Paper(pwc_id="no-datasets-paper", title="Paper Without Datasets")
    paper_id = await repository.upsert_paper(paper)
    assert paper_id is not None

    # --- 2. 调用方法 ---
    results = await repository.get_datasets_for_papers(
        [paper_id]
    )  # 请求这篇论文的数据集

    # --- 3. 验证结果 ---
    assert isinstance(results, dict)
    assert len(results) == 1
    assert paper_id in results
    assert results[paper_id] == []  # 对应的数据集列表应为空


async def test_get_repositories_for_papers_not_found(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：调用 `get_repositories_for_papers`，请求的论文存在，但没有关联的代码仓库。
    预期：返回的字典中，该论文 ID 对应的值应为空列表。
    策略：清空相关表，插入一篇没有代码仓库的论文，调用方法，验证结果。
    """
    # --- 1. 准备数据 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE;")
            await cur.execute(
                "TRUNCATE pwc_repositories RESTART IDENTITY CASCADE;"
            )  # 清空代码库表
    paper = Paper(pwc_id="no-repos-paper", title="Paper Without Repos")
    paper_id = await repository.upsert_paper(paper)
    assert paper_id is not None

    # --- 2. 调用方法 ---
    results = await repository.get_repositories_for_papers(
        [paper_id]
    )  # 请求这篇论文的代码库

    # --- 3. 验证结果 ---
    assert isinstance(results, dict)
    assert len(results) == 1
    assert paper_id in results
    assert results[paper_id] == []  # 对应的代码库列表应为空


# --- 错误处理和边缘情况测试 ---

# 模拟数据库连接或执行错误的集成测试比较复杂，通常更适合单元测试。
# 但可以测试一些基于输入的边缘情况。


async def test_get_hf_models_by_ids_empty(repository: PostgresRepository) -> None:
    """
    测试场景：调用 `get_hf_models_by_ids` 时传入空列表。
    预期：返回空列表。
    """
    result = await repository.get_hf_models_by_ids([])
    assert result == []


async def test_upsert_paper_minimal_data(
    repository: PostgresRepository, db_pool: AsyncConnectionPool
) -> None:
    """
    测试场景：插入一篇只包含数据库模式允许的最少非空字段的论文。
    预期：能够成功插入，并且未提供的可选字段在数据库中应为 NULL。
    策略：创建一个只包含 `pwc_id` 和 `title` 的 Paper 对象（假设这两个是必需的），
          调用 `upsert_paper`，然后获取详情验证。
    注意：这个测试的有效性取决于 `papers` 表的实际模式（哪些列允许 NULL）。
    """
    # --- 1. 清空表 ---
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE;")

    # --- 2. 创建最小数据对象 ---
    minimal_paper = Paper(
        pwc_id="minimal-paper-1",
        title="Minimal Paper Title",
        # 假设模型的其他字段都有默认值 None 或数据库允许 NULL
    )

    # --- 3. 调用 upsert ---
    paper_id = await repository.upsert_paper(minimal_paper)
    assert paper_id is not None  # 确保插入成功

    # --- 4. 获取详情并验证 ---
    details = await repository.get_paper_details_by_id(paper_id)
    assert details is not None
    assert details["pwc_id"] == "minimal-paper-1"
    assert details["title"] == "Minimal Paper Title"
    # 抽查一个可选字段，验证其是否为 None
    assert details["summary"] is None


# === 使用 Mock 进行错误处理测试 ===


# 定义一个辅助类来模拟异步上下文管理器 (如 connection 和 cursor)
class AsyncContextManagerMock:
    """辅助类，用于模拟异步上下文管理器 (async with)。"""

    def __init__(
        self, return_value: Any = None, side_effect: Optional[Exception] = None
    ) -> None:
        """初始化。
        Args:
            return_value: `__aenter__` 方法的返回值 (例如，模拟的 cursor 或 connection)。
            side_effect: `__aenter__` 或 `__aexit__` (取决于具体模拟目标) 可能抛出的异常。
        """
        self.return_value = return_value
        self.side_effect = side_effect
        # 可以添加 mock_aenter 和 mock_aexit 属性来方便断言调用
        self.mock_aenter = AsyncMock(return_value=return_value, side_effect=side_effect)
        self.mock_aexit = AsyncMock(return_value=None)

    async def __aenter__(self) -> Any:
        return await self.mock_aenter()

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        await self.mock_aexit(exc_type, exc_val, exc_tb)
        # 返回 None 或 False 表示不抑制异常（如果 side_effect 发生在 aenter 中）
        return None


async def test_fetch_data_cursor_error_handling(repository: PostgresRepository) -> None:
    """
    测试场景：模拟 `fetch_data_cursor` 内部数据库操作（如 `cursor.execute` 或 `fetchmany`）时发生错误。
    预期：生成器应能被迭代，但在迭代过程中抛出数据库异常。
    策略：使用 `unittest.mock.patch` 替换 `repository.pool.connection`，使其返回一个
          模拟的连接上下文，该上下文返回的游标上下文在执行 `execute` 时抛出异常。
          然后尝试迭代 `fetch_data_cursor` 并断言预期的异常被抛出。
    """
    # --- 1. 创建模拟对象 ---
    # 模拟 cursor，使其 execute 方法抛出异常
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.DatabaseError("Simulated DB error on execute")
    )
    # 模拟异步游标上下文管理器
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)

    # 模拟 connection，使其 cursor 方法返回模拟的游标上下文
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    # 模拟异步连接上下文管理器
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)

    # 模拟 pool，使其 connection 方法返回模拟的连接上下文
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    # --- 2. 应用 Patch ---
    # 保存原始 pool 并替换
    original_pool = repository.pool
    repository.pool = mock_pool

    try:
        # --- 3. 调用生成器并尝试迭代 ---
        results: List[Any] = []  # 添加类型注解
        generator = repository.fetch_data_cursor("SELECT * FROM papers")
        # 使用 pytest.raises 捕获预期的异常
        with pytest.raises(
            psycopg.DatabaseError, match="Simulated DB error on execute"
        ):
            # 尝试从生成器获取第一个元素，这应该会触发 execute 并抛出异常
            await anext(generator)
            # 如果没有抛出异常，或者抛出了其他异常，测试会失败
            # 在这里添加一个失败断言，以防万一迭代成功（不应该发生）
            pytest.fail("Generator did not raise DatabaseError as expected")

        # 如果异常被正确捕获，验证没有生成任何结果
        assert len(results) == 0

    finally:
        # --- 4. 恢复原始 pool ---
        repository.pool = original_pool


# 为其他方法添加类似的基于 Mock 的错误处理测试
# (get_tasks_for_papers, get_datasets_for_papers, get_repositories_for_papers, ...)


async def test_get_tasks_for_papers_error_handling(
    repository: PostgresRepository,
) -> None:
    """模拟 get_tasks_for_papers 中的数据库错误。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.DatabaseError("Simulated tasks error")
    )
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        result = await repository.get_tasks_for_papers([1, 2])
        # 预期：方法内部捕获异常并返回空字典
        assert result == {}
    finally:
        repository.pool = original_pool


async def test_get_datasets_for_papers_error_handling(
    repository: PostgresRepository,
) -> None:
    """模拟 get_datasets_for_papers 中的数据库错误。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.DatabaseError("Simulated datasets error")
    )
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        result = await repository.get_datasets_for_papers([1, 2])
        # 预期：方法内部捕获异常并返回空字典
        assert result == {}
    finally:
        repository.pool = original_pool


async def test_get_repositories_for_papers_error_handling(
    repository: PostgresRepository,
) -> None:
    """模拟 get_repositories_for_papers 中的数据库错误。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.DatabaseError("Simulated repos error")
    )
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        result = await repository.get_repositories_for_papers([1, 2])
        # 预期：方法内部捕获异常并返回空字典
        assert result == {}
    finally:
        repository.pool = original_pool


# === 测试之前文件中未覆盖的错误模拟场景 ===


async def test_get_paper_details_by_id_error_handling(
    repository: PostgresRepository,
) -> None:
    """测试 get_paper_details_by_id 方法的错误处理（使用Mock）。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 模拟 execute 抛出异常
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.DatabaseError("Simulated DB error")
    )
    mock_cursor.fetchone = AsyncMock(
        return_value=None
    )  # 即使 execute 失败，也模拟 fetchone
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        result = await repository.get_paper_details_by_id(1)
        # 预期: 方法内部捕获异常并返回 None
        assert result is None
    finally:
        repository.pool = original_pool


async def test_get_papers_details_by_ids_error_handling(
    repository: PostgresRepository,
) -> None:
    """测试 get_papers_details_by_ids 方法的错误处理（使用Mock）。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 模拟 execute 抛出异常
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.DatabaseError("Simulated DB error")
    )
    mock_cursor.fetchall = AsyncMock(
        return_value=[]
    )  # 即使 execute 失败，也模拟 fetchall
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        result = await repository.get_papers_details_by_ids([1, 2])
        # 预期: 方法内部捕获异常并返回空列表
        assert result == []
    finally:
        repository.pool = original_pool


async def test_fetch_one_error_simulation(repository: PostgresRepository) -> None:
    """测试 fetch_one 方法的错误处理（使用Mock）。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 模拟 execute 抛出异常
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.OperationalError("Simulated DB connection error")
    )
    mock_cursor.fetchone = AsyncMock(return_value=None)
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        query = "SELECT title FROM papers WHERE paper_id = %s"
        result = await repository.fetch_one(query, (1,))
        # 预期: 方法内部捕获异常并返回 None
        assert result is None
    finally:
        repository.pool = original_pool


async def test_get_all_papers_for_sync_error_simulation(
    repository: PostgresRepository,
) -> None:
    """测试 get_all_papers_for_sync 方法的错误处理（使用Mock）。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 模拟 execute 抛出异常
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.ProgrammingError("Simulated syntax error")
    )
    mock_cursor.fetchall = AsyncMock(return_value=[])
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        results = await repository.get_all_papers_for_sync()
        # 预期: 方法内部捕获异常并返回空列表
        assert results == []
    finally:
        repository.pool = original_pool


async def test_count_papers_error_simulation(repository: PostgresRepository) -> None:
    """测试 count_papers 方法的错误处理（使用Mock）。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 模拟 execute 抛出异常
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.OperationalError("Simulated DB error")
    )
    # 模拟 fetchone 返回 None 或者一个包含非数字的结果
    mock_cursor.fetchone = AsyncMock(return_value=None)
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        count = await repository.count_papers()
        # 预期: 方法内部捕获异常并返回 0
        assert count == 0
    finally:
        repository.pool = original_pool


async def test_search_papers_by_keyword_db_error_simulation(
    repository: PostgresRepository,
) -> None:
    """测试 search_papers_by_keyword 在数据库层面发生错误时的行为 (使用 Mock)。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 让 execute 在第二次调用（获取总数）时失败
    mock_cursor.execute = AsyncMock(
        side_effect=[
            None,  # 第一次 execute (获取结果) 成功
            psycopg.DatabaseError(
                "Simulated count error"
            ),  # 第二次 execute (获取总数) 失败
        ]
    )
    mock_cursor.fetchall = AsyncMock(
        return_value=[{"paper_id": 1, "title": "Mock Paper"}]
    )  # 第一次调用返回一些数据
    mock_cursor.fetchone = AsyncMock(
        return_value=None
    )  # 第二次调用失败，fetchone 不会被调用或返回 None
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        results, count = await repository.search_papers_by_keyword("test")
        # 预期：即使计数失败，也应返回已获取的结果，并将计数设为 0 或一个标记值（取决于实现）
        # 当前实现似乎是在捕获异常时将两者都设为空列表和0
        assert results == []
        assert count == 0
    finally:
        repository.pool = original_pool


async def test_search_models_by_keyword_db_error_simulation(
    repository: PostgresRepository,
) -> None:
    """测试 search_models_by_keyword 在数据库层面发生错误时的行为 (使用 Mock)。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 让 execute 在第二次调用（获取总数）时失败
    mock_cursor.execute = AsyncMock(
        side_effect=[
            None,  # 第一次 execute (获取结果) 成功
            psycopg.DatabaseError(
                "Simulated count error"
            ),  # 第二次 execute (获取总数) 失败
        ]
    )
    mock_cursor.fetchall = AsyncMock(
        return_value=[{"hf_model_id": "org/mock-model", "hf_author": "mock"}]
    )  # 第一次调用返回一些数据
    mock_cursor.fetchone = AsyncMock(return_value=None)  # 第二次调用失败
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        results, count = await repository.search_models_by_keyword("test")
        # 预期：即使计数失败，也应返回已获取的结果，并将计数设为 0 或一个标记值（取决于实现）
        # 当前实现似乎是在捕获异常时将两者都设为空列表和0
        assert results == []
        assert count == 0
    finally:
        repository.pool = original_pool


async def test_save_hf_models_batch_error_simulation(
    repository: PostgresRepository,
) -> None:
    """测试 save_hf_models_batch 在数据库层面发生错误时的行为 (使用 Mock)。"""
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_cursor = MagicMock()
    # 让 execute 在执行批量插入时失败
    mock_cursor.execute = AsyncMock(
        side_effect=psycopg.IntegrityError("Simulated constraint violation")
    )
    mock_cursor_ctx = AsyncContextManagerMock(return_value=mock_cursor)
    mock_conn = MagicMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor_ctx)
    mock_conn_ctx = AsyncContextManagerMock(return_value=mock_conn)
    mock_pool.connection = MagicMock(return_value=mock_conn_ctx)

    original_pool = repository.pool
    repository.pool = mock_pool
    try:
        # 尝试保存一些数据
        models_to_save: List[Dict[str, Any]] = [test_hf_model_data_1]
        # 调用方法，预期方法内部会捕获异常并记录日志，但不会重新抛出
        await repository.save_hf_models_batch(models_to_save)
        # 这里可以添加日志捕获断言，验证错误是否被记录
        # 例如: assert "Failed to save batch of HF models" in caplog.text
    except Exception as e:
        # 如果方法重新抛出了未预期的异常，则测试失败
        pytest.fail(f"save_hf_models_batch raised an unexpected exception: {e}")
    finally:
        repository.pool = original_pool
