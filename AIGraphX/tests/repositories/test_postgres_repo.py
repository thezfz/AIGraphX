# -*- coding: utf-8 -*-
"""
文件目的：测试 PostgreSQL 仓库类 (tests/repositories/test_postgres_repo.py)

本文件包含针对 `aigraphx.repositories.postgres_repo.PostgresRepository` 类的集成测试用例。
`PostgresRepository` 封装了所有与 PostgreSQL 数据库交互的操作，是数据持久化的核心组件。

核心测试策略：
- **集成测试:** 测试直接与一个隔离的测试 PostgreSQL 数据库实例进行交互。不模拟数据库连接或 SQL 执行，以确保测试反映真实的数据库行为。
- **Fixture 管理:**
    - 依赖 `conftest.py` 提供的 `db_pool` fixture 来获取到测试数据库的异步连接池。
    - 依赖 `conftest.py` 提供的 `repository` fixture，该 fixture 基于 `db_pool` 创建 `PostgresRepository` 实例，并在每个测试函数执行前后自动处理数据库事务（开始事务 -> 执行测试 -> 回滚事务），保证测试隔离性。
- **数据准备与清理:** 在每个测试函数内部按需插入测试数据。由于事务回滚，测试数据不会持久化到数据库中。对于需要预置数据的测试，可以使用 `@pytest_asyncio.fixture` (如 `setup_simple_data`)。
- **异步执行:** 所有数据库操作都是异步的，因此测试函数使用 `async def` 并标记为 `@pytest.mark.asyncio`。
- **测试覆盖:** 覆盖 `PostgresRepository` 中的关键方法，包括：
    - 数据的插入与更新 (`upsert_paper`)
    - 基于不同条件的数据检索 (`get_papers_details_by_ids`, `get_paper_details_by_pwc_id`, `get_paper_details_by_id`)
    - 关键字搜索与过滤 (`search_papers_by_keyword`)
    - 分页与排序逻辑
    - 使用异步生成器获取数据 (`get_all_paper_ids_and_text`, `fetch_data_cursor`)
    - 关系数据获取 (`get_tasks_for_papers`, etc.) - 虽然部分实现为空
    - 数据库统计 (`count_papers`, `count_hf_models`)
    - 连接池管理 (`close`)

与其他文件的交互：
- 导入 `pytest`, `pytest_asyncio` 等测试框架和异步支持库。
- 导入 `datetime`, `typing`, `pydantic` 等 Python 标准库和类型库。
- 导入 `psycopg_pool`, `psycopg` 用于数据库交互（主要在 Repository 内部）。
- 导入 `logging` 用于调试输出。
- 导入 `unittest.mock` 用于模拟 `close` 方法测试。
- 导入被测试的类 `aigraphx.repositories.postgres_repo.PostgresRepository`。
- 导入数据模型 `aigraphx.models.paper.Paper` 用于创建测试数据。
- **关键依赖:** `tests/conftest.py` 提供的 `db_pool` 和 `repository` fixtures。
"""

import pytest
import pytest_asyncio  # 导入 pytest-asyncio 用于异步 fixture 支持
import os  # 导入 os 模块 (在此文件中似乎未使用)
import json  # 导入 json 模块 (在此文件中似乎未使用)
import datetime  # 导入 datetime 模块，但未使用其成员，下面导入了 date
import asyncio  # 导入 asyncio 库，用于异步操作，如此处的 anext
from typing import AsyncGenerator, Dict, Any, Optional, List, cast  # 导入类型提示工具
from pydantic import (
    HttpUrl,
)  # 从 pydantic 导入 HttpUrl 类型，用于 Paper 模型中的 URL 字段
from psycopg_pool import AsyncConnectionPool  # 导入异步连接池类，由 fixture 提供
from psycopg.rows import dict_row  # 导入 psycopg 的行工厂，用于将查询结果转为字典
from dotenv import (
    load_dotenv,
)  # 导入用于加载 .env 文件的函数 (在此测试文件中未使用，由 conftest 处理)
import logging  # 导入日志库
from unittest.mock import AsyncMock, MagicMock, patch  # 导入模拟工具
from datetime import (
    date as date_type,
)  # 从 datetime 模块导入 date 类，并重命名为 date_type 以免与变量名冲突
import unittest  # 导入 unittest 库 (在此文件中似乎未使用)
import numpy as np  # 导入 numpy (在此文件中似乎未使用)
from psycopg.connection_async import AsyncConnection  # 导入异步连接类型 (未使用)

# 导入被测试的仓库类
from aigraphx.repositories.postgres_repo import PostgresRepository

# 导入数据模型类
from aigraphx.models.paper import Paper

# 导入由 conftest.py 文件提供的 fixtures。这些 fixtures 负责数据库连接和事务管理。
# `db_pool`: 提供连接到测试数据库的 AsyncConnectionPool。
# `repository`: 基于 db_pool 创建 PostgresRepository 实例，并自动管理事务。
from tests.conftest import db_pool, repository

# 为当前测试模块设置日志记录器
logger = logging.getLogger(__name__)

# --- 移除测试数据库配置 ---
# 测试数据库的 URL 和跳过逻辑现在完全由 conftest.py 处理

# pytest 标记，指示此模块中的所有 `async def` 测试函数都应使用 pytest-asyncio
# 提供的默认函数作用域事件循环来运行。
pytestmark = pytest.mark.asyncio

# --- 测试数据 ---
# 定义一些字典格式的测试论文数据。
# 注意：这些数据的结构应与 `Paper` Pydantic 模型和数据库 `papers` 表的模式匹配。
# - `authors` 和 `categories` 是字符串列表。
# - `published_date` 是 `datetime.date` 对象。
# - URL 字段是字符串，将在创建 `Paper` 对象时由 Pydantic 验证/转换为 `HttpUrl`。

test_paper_data_1 = {
    "pwc_id": "test-paper-1",  # 论文在 PapersWithCode 上的 ID
    "arxiv_id_base": "2401.00001",  # ArXiv ID (不含版本)
    "title": "Test Paper One",
    "summary": "Summary one.",
    "pdf_url": "http://example.com/pdf/1",  # PDF 链接 (字符串)
    "published_date": date_type(2024, 1, 1),  # 发表日期 (date 对象)
    "authors": ["Author A", "Author B"],  # 作者列表
    "area": "Computer Vision",  # 研究领域 (如果模型和表中有此字段)
    "primary_category": "cs.CV",  # ArXiv 主要分类
    "categories": ["cs.CV", "cs.AI"],  # ArXiv 所有分类列表
    "pwc_title": "Test Paper One PWC Title",  # 在 PWC 上的标题 (可能与 ArXiv 不同)
    "pwc_url": "http://paperswithcode.com/paper/test-paper-1",  # PWC 链接 (字符串)
    "doi": "10.xxxx/xxxx",  # DOI (可选)
}

test_paper_data_2 = {
    "pwc_id": "test-paper-2",
    "arxiv_id_base": "2402.00002",
    "title": "Test Paper Two",
    "summary": None,  # 摘要可以为 None
    "pdf_url": None,  # PDF 链接可以为 None (对应 Optional[HttpUrl])
    "published_date": date_type(2024, 2, 15),
    "authors": ["Author D"],
    "area": "NLP",
    "primary_category": "cs.CL",
    "categories": ["cs.CL"],
    "pwc_title": "Test Paper Two PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-2",
    "doi": None,  # DOI 可以为 None
}

test_paper_data_3 = {
    "pwc_id": "test-paper-3",
    "arxiv_id_base": "2312.00003",
    "title": "Test Paper Three (CV)",
    "summary": "Summary three.",
    "pdf_url": "http://example.com/pdf/3",
    "published_date": date_type(2023, 12, 25),
    "authors": ["Author A", "Author E"],
    "area": "Computer Vision",
    "primary_category": "cs.CV",
    "categories": ["cs.CV", "cs.LG"],
    "pwc_title": "Test Paper Three PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-3",
    "doi": None,
}

test_paper_data_4 = {
    "pwc_id": "test-paper-4",
    "arxiv_id_base": "2205.00004",
    "title": "Old NLP Paper",
    "summary": "An older NLP paper.",
    "pdf_url": None,
    "published_date": date_type(2022, 5, 10),
    "authors": ["Author F"],
    "area": "NLP",
    "primary_category": "cs.CL",
    "categories": ["cs.CL", "cs.IR"],
    "pwc_title": "Old NLP Paper PWC Title",
    "pwc_url": "http://paperswithcode.com/paper/test-paper-4",
    "doi": None,
}

# --- 测试用例 ---


async def test_get_papers_details_by_ids(repository: PostgresRepository) -> None:
    """
    测试场景：通过内部 `paper_id` 列表检索多篇论文的详细信息。
    策略：插入多篇测试论文，获取它们的 `paper_id`，然后调用 `get_papers_details_by_ids` 方法，
          验证返回结果的数量、类型和内容。同时测试传入不存在的 ID 和空列表的情况。
    """
    # --- 1. 准备数据 ---
    # 使用测试数据创建 Paper 对象。使用 cast 是为了告诉类型检查器，
    # 我们确信字典中的值符合 Paper 模型对应字段的类型（或 Optional 类型）。
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    paper2 = Paper(
        pwc_id=cast(str, test_paper_data_2["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_2["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_2["title"]),
        summary=cast(Optional[str], test_paper_data_2["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_2["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_2["published_date"]),
        authors=cast(List[str], test_paper_data_2["authors"]),
        area=cast(Optional[str], test_paper_data_2["area"]),
        primary_category=cast(Optional[str], test_paper_data_2["primary_category"]),
        categories=cast(List[str], test_paper_data_2["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_2["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_2["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_2["doi"]),
    )
    paper3 = Paper(
        pwc_id=cast(str, test_paper_data_3["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_3["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_3["title"]),
        summary=cast(Optional[str], test_paper_data_3["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_3["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_3["published_date"]),
        authors=cast(List[str], test_paper_data_3["authors"]),
        area=cast(Optional[str], test_paper_data_3["area"]),
        primary_category=cast(Optional[str], test_paper_data_3["primary_category"]),
        categories=cast(List[str], test_paper_data_3["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_3["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_3["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_3["doi"]),
    )

    # 使用 repository fixture 提供的实例插入数据，upsert_paper 返回内部 paper_id
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    assert id1 is not None  # 确保插入成功并返回了 ID
    assert id2 is not None
    assert id3 is not None

    # --- 2. 测试获取存在的 ID ---
    ids_to_fetch = [id2, id1]  # 要获取的 ID 列表，顺序可以打乱
    results = await repository.get_papers_details_by_ids(ids_to_fetch)

    # --- 3. 验证结果 ---
    assert len(results) == 2  # 应返回两个结果
    assert isinstance(results[0], dict)  # 结果列表中的每个元素应为字典
    assert isinstance(results[1], dict)

    # 将结果列表转换为以 paper_id 为键的字典，方便按 ID 查找和验证
    results_map = {r["paper_id"]: r for r in results}
    assert id1 in results_map  # 确保请求的 ID 都在结果中
    assert id2 in results_map

    # 验证返回的字典内容是否与插入的数据一致
    assert results_map[id1]["pwc_id"] == test_paper_data_1["pwc_id"]
    assert results_map[id1]["title"] == test_paper_data_1["title"]
    assert (
        results_map[id1]["authors"] == test_paper_data_1["authors"]
    )  # 列表类型也应匹配

    assert results_map[id2]["pwc_id"] == test_paper_data_2["pwc_id"]
    assert results_map[id2]["summary"] == test_paper_data_2["summary"]  # 验证 None 值

    # --- 4. 测试获取混合存在与不存在的 ID ---
    results_mixed = await repository.get_papers_details_by_ids(
        [id1, 99999]
    )  # 99999 是一个不存在的 ID
    assert len(results_mixed) == 1  # 只应返回存在的 ID 对应的数据
    assert results_mixed[0]["paper_id"] == id1

    # --- 5. 测试传入空列表 ---
    results_empty = await repository.get_papers_details_by_ids([])
    assert results_empty == []  # 传入空列表，应返回空列表


async def test_get_paper_details_by_pwc_id_found(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：通过 `pwc_id` (PapersWithCode ID) 检索单篇论文的详细信息，且论文存在。
    策略：插入一篇测试论文，然后使用其 `pwc_id` 调用 `get_paper_details_by_pwc_id`，
          验证返回的字典内容是否正确。
    """
    # --- 1. 准备数据 ---
    paper1 = Paper(
        pwc_id=cast(str, test_paper_data_1["pwc_id"]),
        arxiv_id_base=cast(Optional[str], test_paper_data_1["arxiv_id_base"]),
        title=cast(Optional[str], test_paper_data_1["title"]),
        summary=cast(Optional[str], test_paper_data_1["summary"]),
        pdf_url=cast(Optional[HttpUrl], test_paper_data_1["pdf_url"]),
        published_date=cast(Optional[date_type], test_paper_data_1["published_date"]),
        authors=cast(List[str], test_paper_data_1["authors"]),
        area=cast(Optional[str], test_paper_data_1["area"]),
        primary_category=cast(Optional[str], test_paper_data_1["primary_category"]),
        categories=cast(List[str], test_paper_data_1["categories"]),
        pwc_title=cast(Optional[str], test_paper_data_1["pwc_title"]),
        pwc_url=cast(Optional[HttpUrl], test_paper_data_1["pwc_url"]),
        doi=cast(Optional[str], test_paper_data_1["doi"]),
    )
    id1 = await repository.upsert_paper(paper1)
    assert id1 is not None

    # --- 2. 调用方法 ---
    # 使用插入数据的 pwc_id 进行查询
    details = await repository.get_paper_details_by_pwc_id(
        str(test_paper_data_1["pwc_id"])
    )

    # --- 3. 验证结果 ---
    assert details is not None  # 应找到结果
    assert isinstance(details, dict)  # 结果应为字典
    assert details["paper_id"] == id1  # 验证内部 ID
    assert details["pwc_id"] == test_paper_data_1["pwc_id"]  # 验证 pwc_id
    assert details["title"] == test_paper_data_1["title"]  # 验证其他字段
    assert details["summary"] == test_paper_data_1["summary"]
    assert details["authors"] == test_paper_data_1["authors"]
    assert details["published_date"] == test_paper_data_1["published_date"]


async def test_get_paper_details_by_pwc_id_not_found(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：通过 `pwc_id` 检索论文信息，但该 `pwc_id` 不存在于数据库中。
    策略：直接使用一个不存在的 `pwc_id` 调用 `get_paper_details_by_pwc_id`，验证返回结果为 None。
    """
    details = await repository.get_paper_details_by_pwc_id("non-existent-pwc-id")
    assert details is None


async def test_search_papers_by_keyword_no_filters(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：执行基本的论文关键字搜索，不带任何过滤器（如日期、领域）或特定的排序。
    策略：插入多篇包含不同关键词的论文，然后使用不同的查询词调用 `search_papers_by_keyword`，
          验证返回结果的数量和 ID 是否符合预期（基于标题和摘要的全文搜索）。
    """
    # --- 1. 准备数据 ---
    # 插入四篇测试论文
    paper1 = Paper(
        **cast(Dict[str, Any], test_paper_data_1)
    )  # 使用 ** 解包字典 (需要 cast)
    paper2 = Paper(**cast(Dict[str, Any], test_paper_data_2))
    paper3 = Paper(**cast(Dict[str, Any], test_paper_data_3))
    paper4 = Paper(**cast(Dict[str, Any], test_paper_data_4))
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert all([id1, id2, id3, id4])  # 确保所有 ID 都有效

    # --- 2. 测试搜索 "Paper" ---
    # "Paper" 出现在所有论文的标题或摘要中
    results_list, total = await repository.search_papers_by_keyword("Paper")
    assert total == 4  # 总数应为 4
    assert len(results_list) == 4  # 返回的列表长度也应为 4 (默认 limit 很大)
    # 验证返回的论文 ID 是否包含所有插入的论文 ID (使用集合比较，忽略顺序)
    returned_ids = {item["paper_id"] for item in results_list}
    expected_ids = {id1, id2, id3, id4}
    assert returned_ids == expected_ids

    # --- 3. 测试搜索 "One" ---
    # "One" 只出现在 paper1 的标题中
    results_one_list, total_one = await repository.search_papers_by_keyword("One")
    assert total_one == 1  # 总数应为 1
    assert len(results_one_list) == 1
    assert results_one_list[0]["paper_id"] == id1  # 返回的应是 paper1

    # --- 4. 测试搜索不存在的词 ---
    results_none, total_none = await repository.search_papers_by_keyword("NoSuchTerm")
    assert total_none == 0  # 总数应为 0
    assert results_none == []  # 返回空列表


async def test_search_papers_by_keyword_with_limit_skip(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：执行论文关键字搜索，并应用分页参数 `limit` 和 `skip`。
    策略：插入多篇论文，执行搜索，指定 `skip` 和 `limit`，并结合默认排序规则 (`published_date DESC`, `paper_id DESC`)
          来验证返回的子集是否正确。
    """
    # --- 1. 准备数据 ---
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))  # 2024-01-01
    paper2 = Paper(**cast(Dict[str, Any], test_paper_data_2))  # 2024-02-15
    paper3 = Paper(**cast(Dict[str, Any], test_paper_data_3))  # 2023-12-25
    paper4 = Paper(**cast(Dict[str, Any], test_paper_data_4))  # 2022-05-10
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert id1 is not None and id2 is not None and id3 is not None and id4 is not None

    query = "paper"  # 匹配所有 4 篇论文

    # --- 2. 计算预期顺序 ---
    # 默认排序是 published_date DESC, paper_id DESC。
    # 根据测试数据日期：paper2 (最新) -> paper1 -> paper3 -> paper4 (最旧)
    # 如果日期相同，则按 paper_id DESC 排序（假设 id2 > id1 > id3 > id4）
    # 所以预期的 ID 顺序是: id2, id1, id3, id4

    # --- 3. 测试分页：获取第 2 页 (skip=2, limit=2) ---
    results_list, total_count = await repository.search_papers_by_keyword(
        query=query,
        skip=2,  # 跳过前 2 条记录 (id2, id1)
        limit=2,  # 获取接下来的 2 条记录
        sort_by="published_date",  # 显式指定排序以确保一致性
        sort_order="desc",
    )

    # --- 4. 验证结果 ---
    # 从返回的字典列表中提取 paper_id
    returned_ids = [item["paper_id"] for item in results_list]

    assert total_count == 4  # 总匹配数仍为 4
    assert len(results_list) == 2  # 返回列表长度应为 limit (2)
    # 断言返回的 ID 列表是否为预期顺序中的第 3、4 个元素 (id3, id4)
    assert returned_ids == [id3, id4]


async def test_search_papers_by_keyword_with_filters(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：执行论文关键字搜索，并同时应用日期范围 (`published_after`, `published_before`)
          和研究领域 (`filter_area`) 过滤器。
    策略：插入具有不同日期和领域的论文，执行带过滤条件的搜索，验证返回结果是否只包含
          同时满足所有条件的论文。
    """
    # --- 1. 准备数据 ---
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))  # CV, 2024-01-01
    paper2 = Paper(**cast(Dict[str, Any], test_paper_data_2))  # NLP, 2024-02-15
    paper3 = Paper(**cast(Dict[str, Any], test_paper_data_3))  # CV, 2023-12-25
    paper4 = Paper(**cast(Dict[str, Any], test_paper_data_4))  # NLP, 2022-05-10
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert id1 is not None and id2 is not None and id3 is not None and id4 is not None

    # --- 2. 定义搜索条件 ---
    query = "paper"  # 匹配所有论文的关键词
    date_from = date_type(2024, 1, 1)  # 发表日期 >= 2024-01-01
    date_to = date_type(2024, 12, 31)  # 发表日期 <= 2024-12-31
    area = ["NLP"]  # 研究领域为 NLP (注意：参数期望列表类型)

    # --- 3. 执行带过滤的搜索 ---
    results_list, total_count = await repository.search_papers_by_keyword(
        query=query,
        published_after=date_from,
        published_before=date_to,
        filter_area=area,  # 传入 area 列表
    )

    # --- 4. 验证结果 ---
    # 从返回结果中提取 paper_id 集合
    returned_ids = {item["paper_id"] for item in results_list}

    # 只有 paper2 (NLP, 2024-02-15) 同时满足所有条件
    assert total_count == 1
    assert len(results_list) == 1
    assert returned_ids == {id2}


async def test_get_all_paper_ids_and_text(repository: PostgresRepository) -> None:
    """
    测试场景：使用异步生成器 `get_all_paper_ids_and_text` 获取所有论文的 ID 和用于向量化的文本内容。
    策略：插入包含不同标题和摘要（包括 None 或空字符串）的论文，然后迭代生成器，
          验证返回的 ID 和文本内容是否正确（优先使用摘要，若无则回退到标题）。
    """
    # --- 1. 准备数据 ---
    # 插入 4 篇标准测试论文
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))  # 有摘要
    paper2 = Paper(**cast(Dict[str, Any], test_paper_data_2))  # 无摘要，有标题
    paper3 = Paper(**cast(Dict[str, Any], test_paper_data_3))  # 有摘要
    paper4 = Paper(**cast(Dict[str, Any], test_paper_data_4))  # 有摘要
    # 再插入一篇只有标题没有摘要的论文
    paper_no_summary_title = Paper(
        pwc_id="test-paper-5",
        arxiv_id_base="2403.00005",
        title="Paper Without Summary",  # 有标题
        summary=None,  # 摘要为 None
        pdf_url=None,
        published_date=date_type(2024, 3, 1),
        authors=["Author G"],
        area="Other",
        primary_category="cs.XX",
        categories=["cs.XX"],
        pwc_title="Paper Without Summary PWC Title",
        pwc_url=cast(Optional[HttpUrl], "http://paperswithcode.com/paper/test-paper-5"),
        doi=None,
    )
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    id5 = await repository.upsert_paper(paper_no_summary_title)
    assert all([id1, id2, id3, id4, id5])  # 确保所有插入都成功

    # --- 2. (可选) 添加诊断日志，检查插入后的数据库状态 ---
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM papers;")
            count_result = await cur.fetchone()
            logger.info(
                f"[DIAGNOSTIC test_get_all_paper_ids_and_text] Count after upsert: {count_result[0] if count_result else 'None'}"
            )

    # --- 3. 迭代生成器并收集结果 ---
    results = {}  # 用于存储 ID -> 文本内容的字典
    logger.info("[test_get_all_paper_ids_and_text] Starting manual iteration...")
    generator = repository.get_all_paper_ids_and_text()  # 获取异步生成器
    items_yielded = 0
    try:
        # 使用 anext() 手动迭代异步生成器 (需要 import asyncio)
        while True:
            paper_id, text_content = await anext(generator)
            logger.info(
                f"[test_get_all_paper_ids_and_text] Yielded: ID={paper_id}, Text='{text_content[:50]}...'"  # 记录日志
            )
            results[paper_id] = text_content  # 存储结果
            items_yielded += 1
    except StopAsyncIteration:
        # 生成器迭代完成时会抛出 StopAsyncIteration 异常
        logger.info("[test_get_all_paper_ids_and_text] StopAsyncIteration caught.")
        pass
    except Exception as e:
        # 捕获可能的其他异常
        logger.error(
            f"[test_get_all_paper_ids_and_text] Error during iteration: {e}",
            exc_info=True,
        )

    logger.info(
        f"[test_get_all_paper_ids_and_text] Manual iteration finished. Items yielded: {items_yielded}"
    )

    # --- 4. 验证结果 ---
    assert len(results) == 5  # 应返回 5 篇论文

    # 验证文本内容是否符合预期（优先摘要，其次标题）
    assert id1 is not None  # 确保 id 不是 None
    assert results[id1] == test_paper_data_1["summary"]  # paper1 有摘要
    assert id2 is not None
    assert results[id2] == test_paper_data_2["title"]  # paper2 无摘要，回退到标题
    assert id3 is not None
    assert results[id3] == test_paper_data_3["summary"]  # paper3 有摘要
    assert id4 is not None
    assert results[id4] == test_paper_data_4["summary"]  # paper4 有摘要
    assert id5 is not None
    assert results[id5] == paper_no_summary_title.title  # paper5 无摘要，回退到标题


# --- 定义一个辅助 fixture，用于在测试前插入一些简单数据 ---
@pytest_asyncio.fixture
async def setup_simple_data(repository: PostgresRepository) -> None:
    """
    Pytest Fixture: 插入几篇基础论文数据。

    这个 fixture 依赖于 `repository` fixture，用于在需要基础数据的测试函数
    执行之前，向数据库中插入 `test_paper_data_1` 和 `test_paper_data_2`。
    由于 `repository` fixture 会在测试后回滚事务，这些数据不会持久存在。

    Args:
        repository (PostgresRepository): 由 conftest.py 提供的仓库实例。
    """
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))
    paper2 = Paper(**cast(Dict[str, Any], test_paper_data_2))
    await repository.upsert_paper(paper1)
    await repository.upsert_paper(paper2)
    logger.info("[setup_simple_data] Finished inserting simple data.")


async def test_fetch_data_cursor(
    repository: PostgresRepository,
    setup_simple_data: Any,  # 使用 setup_simple_data 插入数据
) -> None:
    """
    测试场景：使用异步生成器 `fetch_data_cursor` 分批获取查询结果。
    策略：使用 `setup_simple_data` fixture 准备数据，然后构造一个 SQL 查询。
          分别使用不同的 `batch_size` (1 和 5) 调用 `fetch_data_cursor`，
          迭代生成器并验证返回的行数、内容和唯一性。最后测试查询无结果的情况。
    """

    # --- 1. (可选) 诊断日志：检查数据是否已插入 ---
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM papers;")
            count_result = await cur.fetchone()
            logger.info(
                f"[DIAGNOSTIC test_fetch_data_cursor] Count before fetch: {count_result[0] if count_result else 'None'}"
            )

    # --- 2. 测试 batch_size = 1 ---
    query = "SELECT paper_id, pwc_id, title FROM papers ORDER BY paper_id"  # 定义查询
    batch_size = 1  # 每次获取 1 行
    count = 0
    ids_seen = set()  # 用于记录已获取的 ID，检查重复

    logger.info(
        "[test_fetch_data_cursor] Starting manual iteration 1 (batch_size=1)..."
    )
    generator_b1 = repository.fetch_data_cursor(query, (), batch_size)  # 获取生成器
    items_yielded_b1 = 0
    try:
        # 手动迭代生成器
        while True:
            row = await anext(generator_b1)
            logger.info(f"[test_fetch_data_cursor] Yielded (b1): {row}")
            assert isinstance(row, dict)  # 每行应为字典
            assert "paper_id" in row  # 检查关键字段是否存在
            assert "pwc_id" in row
            assert "title" in row
            assert row["paper_id"] not in ids_seen  # 确保 ID 未重复
            ids_seen.add(row["paper_id"])
            count += 1
            items_yielded_b1 += 1
            # 验证特定行的数据内容
            if row["pwc_id"] == test_paper_data_1["pwc_id"]:
                assert row["title"] == test_paper_data_1["title"]
            elif row["pwc_id"] == test_paper_data_2["pwc_id"]:
                assert row["title"] == test_paper_data_2["title"]
    except StopAsyncIteration:
        # 正常结束
        logger.info("[test_fetch_data_cursor] StopAsyncIteration caught (b1).")
        pass
    except Exception as e:
        logger.error(
            f"[test_fetch_data_cursor] Error during iteration (b1): {e}", exc_info=True
        )

    logger.info(
        f"[test_fetch_data_cursor] Manual iteration 1 finished. Items yielded: {items_yielded_b1}"
    )
    # `setup_simple_data` 插入了 2 条记录
    assert count == 2

    # --- 3. 测试 batch_size = 5 ---
    count_b2 = 0
    ids_seen_b2 = set()
    logger.info(
        "[test_fetch_data_cursor] Starting manual iteration 2 (batch_size=5)..."
    )
    generator_b5 = repository.fetch_data_cursor(query, (), batch_size=5)
    items_yielded_b5 = 0
    try:
        while True:
            row_b2 = await anext(generator_b5)
            logger.info(f"[test_fetch_data_cursor] Yielded (b5): {row_b2}")
            assert isinstance(row_b2, dict)
            assert row_b2["paper_id"] not in ids_seen_b2
            ids_seen_b2.add(row_b2["paper_id"])
            count_b2 += 1
            items_yielded_b5 += 1
    except StopAsyncIteration:
        logger.info("[test_fetch_data_cursor] StopAsyncIteration caught (b5).")
        pass
    except Exception as e:
        logger.error(
            f"[test_fetch_data_cursor] Error during iteration (b5): {e}", exc_info=True
        )

    logger.info(
        f"[test_fetch_data_cursor] Manual iteration 2 finished. Items yielded: {items_yielded_b5}"
    )
    assert count_b2 == 2  # 仍然只应获取 2 条记录

    # --- 4. 测试查询无结果的情况 ---
    query_empty = (
        "SELECT paper_id FROM papers WHERE pwc_id = %s"  # 查询一个不存在的 pwc_id
    )
    count_empty = 0
    logger.info(
        "[test_fetch_data_cursor] Starting manual iteration 3 (empty result)..."
    )
    generator_empty = repository.fetch_data_cursor(query_empty, ("no-such-id",), 10)
    items_yielded_empty = 0
    try:
        while True:
            _ = await anext(generator_empty)  # 尝试获取下一项
            logger.info(
                "[test_fetch_data_cursor] Yielded (empty): Unexpected item!"
            )  # 如果执行到这里，说明有问题
            count_empty += 1
            items_yielded_empty += 1
    except StopAsyncIteration:
        # 正常结束 (因为没有结果)
        logger.info("[test_fetch_data_cursor] StopAsyncIteration caught (empty).")
        pass
    except Exception as e:
        logger.error(
            f"[test_fetch_data_cursor] Error during iteration (empty): {e}",
            exc_info=True,
        )

    logger.info(
        f"[test_fetch_data_cursor] Manual iteration 3 finished. Items yielded: {items_yielded_empty}"
    )
    assert count_empty == 0  # 确认没有获取到任何项


async def test_search_papers_by_keyword_sort_by_date(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：执行论文关键字搜索，并明确指定按 `published_date` 排序（升序和降序）。
    策略：插入具有不同发表日期的论文，执行带 `sort_by` 和 `sort_order` 参数的搜索，
          验证返回结果的顺序是否正确。
    """
    # --- 1. 准备数据 ---
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))  # 2024-01-01
    paper2 = Paper(**cast(Dict[str, Any], test_paper_data_2))  # 2024-02-15
    paper3 = Paper(**cast(Dict[str, Any], test_paper_data_3))  # 2023-12-25
    paper4 = Paper(**cast(Dict[str, Any], test_paper_data_4))  # 2022-05-10
    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4)
    assert id1 is not None and id2 is not None and id3 is not None and id4 is not None

    query = "paper"  # 匹配所有论文

    # --- 2. 按日期降序测试 ---
    results_list_desc, total_count_desc = await repository.search_papers_by_keyword(
        query=query,
        sort_by="published_date",
        sort_order="desc",  # 指定降序
    )
    returned_ids_desc = [item["paper_id"] for item in results_list_desc]
    assert total_count_desc == 4
    # 预期顺序：最新 -> 最旧 (id2, id1, id3, id4)
    assert returned_ids_desc == [id2, id1, id3, id4]

    # --- 3. 按日期升序测试 ---
    results_list_asc, total_count_asc = await repository.search_papers_by_keyword(
        query=query,
        sort_by="published_date",
        sort_order="asc",  # 指定升序
    )
    returned_ids_asc = [item["paper_id"] for item in results_list_asc]
    assert total_count_asc == 4
    # 预期顺序：最旧 -> 最新 (id4, id3, id1, id2)
    assert returned_ids_asc == [id4, id3, id1, id2]


async def test_get_paper_details_by_id_integration(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：通过内部 `paper_id` 检索单篇论文的详细信息。
    策略：插入一篇论文，获取其返回的 `paper_id`，然后使用该 ID 调用 `get_paper_details_by_id`，
          验证返回的字典内容。同时测试使用不存在的 `paper_id` 进行查询。
    """
    # --- 1. 准备数据 ---
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))
    inserted_id = await repository.upsert_paper(paper1)
    assert inserted_id is not None

    # --- 2. 使用存在的 ID 查询 ---
    details = await repository.get_paper_details_by_id(inserted_id)
    assert details is not None
    assert isinstance(details, dict)
    assert details["paper_id"] == inserted_id
    assert details["pwc_id"] == test_paper_data_1["pwc_id"]
    assert details["title"] == test_paper_data_1["title"]
    assert details["authors"] == test_paper_data_1["authors"]

    # --- 3. 使用不存在的 ID 查询 ---
    details_none = await repository.get_paper_details_by_id(
        999999
    )  # 一个极不可能存在的 ID
    assert details_none is None


async def test_upsert_paper_update(repository: PostgresRepository) -> None:
    """
    测试场景：测试 `upsert_paper` 方法的更新逻辑。
    策略：先插入一篇论文，然后使用相同的 `pwc_id` 但修改了其他字段的数据再次调用 `upsert_paper`。
          验证第二次调用返回的 `paper_id` 与第一次相同（表示是更新操作），然后获取该论文的详情，
          验证其字段已被更新为新值。
    """
    # --- 1. 插入初始版本 ---
    paper_initial = Paper(**cast(Dict[str, Any], test_paper_data_1))
    id_initial = await repository.upsert_paper(paper_initial)
    assert id_initial is not None

    # --- 2. 创建更新后的数据 (pwc_id 保持不变) ---
    updated_data = test_paper_data_1.copy()  # 复制初始数据
    updated_data["title"] = "Updated Test Paper One Title"  # 修改标题
    updated_data["summary"] = "Updated summary."  # 修改摘要
    updated_data["authors"] = ["Author A", "Author B", "Author C"]  # 修改作者列表
    paper_updated = Paper(
        **cast(Dict[str, Any], updated_data)
    )  # 创建更新后的 Paper 对象

    # --- 3. 再次调用 upsert ---
    id_updated = await repository.upsert_paper(paper_updated)
    # 验证返回的 ID 是否与初始插入的 ID 相同
    assert id_updated == id_initial

    # --- 4. 获取详情并验证更新 ---
    details = await repository.get_paper_details_by_id(
        id_initial
    )  # 使用初始 ID 获取详情
    assert details is not None
    # 验证字段是否已更新
    assert details["title"] == "Updated Test Paper One Title"
    assert details["summary"] == "Updated summary."
    assert details["authors"] == ["Author A", "Author B", "Author C"]
    # 验证未修改的字段（如 pwc_id）是否保持不变
    assert details["pwc_id"] == test_paper_data_1["pwc_id"]


async def test_get_all_papers_for_sync_empty_db(repository: PostgresRepository) -> None:
    """
    测试场景：当数据库 `papers` 表为空时，调用 `get_all_papers_for_sync` 方法。
    预期：应返回一个空列表。
    策略：先清空 `papers` 表，然后调用方法并断言结果。
    """
    # --- 1. 清空数据库 ---
    # 直接执行 TRUNCATE 命令来清空表，并重置序列
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")

    # --- 2. 调用方法 ---
    results = await repository.get_all_papers_for_sync()

    # --- 3. 验证结果 ---
    assert isinstance(results, list)
    assert len(results) == 0


async def test_get_all_papers_for_sync_with_data(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：调用 `get_all_papers_for_sync` 方法，数据库中包含有摘要和无摘要的论文。
    预期：方法应只返回那些 `summary` 字段不为空或空字符串的论文记录。
    策略：插入几篇论文，包括摘要为空或 None 的情况，调用方法，验证返回结果的数量和 ID。
    """
    # --- 1. 清空数据库 (确保测试环境干净) ---
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
            # 如果有 pwc_tasks 等关联表，也需要清空或设置级联删除
            # await cur.execute("TRUNCATE pwc_tasks RESTART IDENTITY CASCADE")

    # --- 2. 准备数据 ---
    paper1 = Paper(
        pwc_id="test-sync-1",
        title="Paper With Summary",
        summary="This is a summary for syncing.",  # 有效摘要
        authors=["Author X"],
    )
    paper2 = Paper(
        pwc_id="test-sync-2",
        title="Paper Without Summary",
        summary="",  # 空字符串摘要，也应被排除
        authors=["Author Y"],
    )
    paper3 = Paper(
        pwc_id="test-sync-3",
        title="Paper With Summary 2",
        summary="Another summary for testing sync.",  # 有效摘要
        authors=["Author Z"],
    )
    paper4_null = Paper(  # 添加摘要为 None 的情况
        pwc_id="test-sync-4",
        title="Paper With Null Summary",
        summary=None,  # None 摘要，也应被排除
        authors=["Author W"],
    )

    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3)
    id4 = await repository.upsert_paper(paper4_null)

    assert id1 is not None
    assert id2 is not None
    assert id3 is not None
    assert id4 is not None

    # --- 3. 调用方法 ---
    results = await repository.get_all_papers_for_sync()

    # --- 4. 验证结果 ---
    assert isinstance(results, list)
    # 应该只返回 paper1 和 paper3 (摘要非空且非空字符串)
    assert len(results) == 2

    # 验证返回的论文 ID 是否正确
    paper_ids = {result["paper_id"] for result in results}
    assert id1 in paper_ids
    assert id3 in paper_ids
    assert id2 not in paper_ids  # 不应包含 id2
    assert id4 not in paper_ids  # 不应包含 id4


async def test_count_papers_empty_db(repository: PostgresRepository) -> None:
    """
    测试场景：当 `papers` 表为空时，调用 `count_papers` 方法。
    预期：应返回 0。
    策略：清空表，调用方法，断言结果。
    """
    # --- 1. 清空数据库 ---
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")

    # --- 2. 调用方法 ---
    count = await repository.count_papers()

    # --- 3. 验证结果 ---
    assert count == 0


async def test_count_papers_with_data(repository: PostgresRepository) -> None:
    """
    测试场景：当 `papers` 表中有数据时，调用 `count_papers` 方法。
    预期：应返回正确的论文数量。
    策略：清空表，插入 N 条记录，调用方法，断言结果为 N。
    """
    # --- 1. 清空数据库 ---
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")

    # --- 2. 准备数据 ---
    paper1 = Paper(
        pwc_id="test-count-1", title="Test Count Paper 1", authors=["Author X"]
    )
    paper2 = Paper(
        pwc_id="test-count-2", title="Test Count Paper 2", authors=["Author Y"]
    )
    await repository.upsert_paper(paper1)
    await repository.upsert_paper(paper2)

    # --- 3. 调用方法 ---
    count = await repository.count_papers()

    # --- 4. 验证结果 ---
    assert count == 2  # 插入了 2 条记录


async def test_search_papers_by_keyword_empty_result(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：使用一个数据库中肯定不存在的关键词执行关键字搜索。
    预期：应返回空的结果列表和 0 的总数。
    策略：插入一些数据（确保数据库不完全为空），然后使用非常独特的关键词搜索，断言结果。
    """
    # --- 1. 准备数据 (确保表不为空) ---
    paper = Paper(
        pwc_id="test-search-empty",
        title="Specific Title For Test",
        summary="Specific summary for empty search test",
        authors=["Author Empty"],
    )
    await repository.upsert_paper(paper)

    # --- 2. 使用不存在的关键词搜索 ---
    results, count = await repository.search_papers_by_keyword(
        "NonExistentKeywordXYZ123"  # 一个极不可能存在的关键词
    )

    # --- 3. 验证结果 ---
    assert isinstance(results, list)
    assert len(results) == 0  # 结果列表应为空
    assert count == 0  # 总数应为 0


async def test_search_papers_by_keyword_invalid_sort(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：执行关键字搜索，但提供一个无效的 `sort_by` 字段名。
    预期：方法应能优雅处理（可能忽略无效字段，使用默认排序），并返回结果。
    策略：插入数据，使用无效 `sort_by` 参数调用搜索，验证仍能返回结果。
    注意：数据库本身在遇到无效 ORDER BY 子句时可能会报错，但 Repository 实现
          可能（也应该）在生成 SQL 前验证 sort_by 参数的有效性。此测试更像是在
          验证 Repository 是否有此保护机制或默认行为。如果 Repository 直接拼接
          字符串导致 SQL 错误，此测试会失败。
    """
    # --- 1. 准备数据 ---
    paper = Paper(
        pwc_id="test-sort-invalid",
        title="Test Invalid Sort",
        summary="Test paper for invalid sort test",
        authors=["Author Sort"],
    )
    await repository.upsert_paper(paper)

    # --- 2. 使用无效排序字段搜索 ---
    # 使用 type: ignore 来抑制类型检查器关于 "invalid_column" 不是有效字面量的警告
    results, count = await repository.search_papers_by_keyword(
        "Test",  # 假设 "Test" 匹配插入的数据
        sort_by="invalid_column",  # type: ignore[arg-type] # 无效的列名
    )

    # --- 3. 验证结果 ---
    assert isinstance(results, list)
    assert count > 0  # 应该找到匹配项
    # 结果列表不应为空，即使排序字段无效（应回退到默认排序）
    assert len(results) > 0


async def test_get_hf_models_by_ids_empty_ids(repository: PostgresRepository) -> None:
    """
    测试场景：调用 `get_hf_models_by_ids` 方法，并传入一个空的 ID 列表。
    预期：应返回一个空列表。
    策略：直接使用空列表调用方法并断言。
    """
    results = await repository.get_hf_models_by_ids([])

    assert isinstance(results, list)
    assert len(results) == 0


async def test_get_paper_details_by_id_nonexistent(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：调用 `get_paper_details_by_id` 方法，使用一个数据库中不存在的 `paper_id`。
    预期：应返回 None。
    策略：使用一个非常大的、不太可能存在的整数 ID 调用方法并断言。
    """
    # 使用一个非常大的、几乎不可能存在的 ID
    result = await repository.get_paper_details_by_id(999999999)

    assert result is None


# === upsert_paper 测试 ===
# test_upsert_paper_update 已经测试了更新逻辑。下面添加一个纯粹的插入测试。
async def test_upsert_paper_insert(repository: PostgresRepository) -> None:
    """
    测试场景：测试 `upsert_paper` 方法的插入逻辑。
    策略：插入一篇全新的论文（确保 pwc_id 不存在），验证返回的 paper_id 是有效的整数，
          然后通过该 paper_id 获取详情，验证数据是否正确插入。
    """
    # --- 1. 准备新数据 ---
    paper_new = Paper(
        pwc_id="upsert-insert-new",  # 一个新的 pwc_id
        title="Upsert Insert Test",
        arxiv_id_versioned="2404.00001v1",
        arxiv_id_base="2404.00001",
        summary="Summary for insert test",
        pdf_url=HttpUrl("https://arxiv.org/pdf/2404.00001.pdf"),
        authors=["Insert Author"],
        published_date=date_type(2024, 4, 1),
        categories=["cs.AI"],
        area="AI",
    )

    # --- 2. 调用 upsert ---
    result_id = await repository.upsert_paper(paper_new)

    # --- 3. 验证返回的 ID ---
    assert isinstance(result_id, int)  # 返回的应是整数类型的 paper_id
    assert result_id > 0  # ID 应该是正数

    # --- 4. 获取详情并验证插入的数据 ---
    details = await repository.get_paper_details_by_id(result_id)
    assert details is not None
    assert details["pwc_id"] == "upsert-insert-new"
    assert details["title"] == "Upsert Insert Test"
    assert details["summary"] == "Summary for insert test"
    assert details["authors"] == ["Insert Author"]


# === count_hf_models 测试 ===
# 这个测试依赖于 hf_models 表的存在和结构。
# 假设该表存在，并且在测试开始时是空的（由 fixture 的事务回滚保证）。
async def test_count_hf_models(repository: PostgresRepository) -> None:
    """
    测试场景：调用 `count_hf_models` 方法计数 Hugging Face 模型。
    预期：在没有插入数据的情况下，返回 0。
    策略：直接调用方法并断言结果。
    注意：要测试计数非零的情况，需要先实现并调用插入 HF 模型的方法（如 `upsert_hf_model`）。
    """
    # 假设 hf_models 表存在且通过事务回滚保证为空
    count_initial = await repository.count_hf_models()
    assert count_initial == 0

    # TODO: 如果/当存在 upsert_hf_model 方法时，添加测试：
    # 1. 插入一些 HF 模型数据
    # 2. 再次调用 count_hf_models
    # 3. 断言计数是否等于插入的数量


# === fetch_one 测试 ===
async def test_fetch_one_success(repository: PostgresRepository) -> None:
    """
    测试场景：使用 `fetch_one` 方法成功获取单条记录。
    策略：先插入一条数据，然后构造 SQL 查询（带参数），调用 `fetch_one`，
          验证返回结果非 None 且内容正确。
    """
    # --- 1. 准备数据 ---
    paper1 = Paper(**cast(Dict[str, Any], test_paper_data_1))
    paper_id = await repository.upsert_paper(paper1)
    assert paper_id is not None

    # --- 2. 调用 fetch_one ---
    query = "SELECT title, pwc_id FROM papers WHERE paper_id = %s"
    result = await repository.fetch_one(query, (paper_id,))  # 使用元组传递参数

    # --- 3. 验证结果 ---
    assert result is not None  # 应找到记录
    assert isinstance(result, dict)  # 返回结果应为字典
    assert result["pwc_id"] == test_paper_data_1["pwc_id"]
    assert result["title"] == test_paper_data_1["title"]


async def test_fetch_one_not_found(repository: PostgresRepository) -> None:
    """
    测试场景：使用 `fetch_one` 方法查询不存在的记录。
    预期：应返回 None。
    策略：构造 SQL 查询查询一个不存在的 ID，调用 `fetch_one`，断言结果为 None。
    """
    query = "SELECT title FROM papers WHERE paper_id = %s"
    result = await repository.fetch_one(query, (99999,))  # 查询不存在的 ID
    assert result is None


# === get_tasks/datasets/repositories_for_papers 测试 ===
# 这些测试目前只验证了传入空 ID 列表的情况。需要补充测试用例来验证
# 在关联表 (pwc_tasks, pwc_datasets, pwc_repositories) 中有数据时，
# 这些方法是否能正确地将关系数据聚合到对应的 paper_id 上。


async def test_get_tasks_for_papers_empty_ids(repository: PostgresRepository) -> None:
    """
    测试场景：调用 `get_tasks_for_papers` 方法，并传入一个空的 paper_id 列表。
    预期：应返回一个空字典。
    策略：直接使用空列表调用方法并断言。
    """
    result = await repository.get_tasks_for_papers([])

    assert isinstance(result, dict)
    assert len(result) == 0


async def test_get_datasets_for_papers_empty_ids(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：调用 `get_datasets_for_papers` 方法，并传入一个空的 paper_id 列表。
    预期：应返回一个空字典。
    策略：直接使用空列表调用方法并断言。
    """
    result = await repository.get_datasets_for_papers([])

    assert isinstance(result, dict)
    assert len(result) == 0


async def test_get_repositories_for_papers_empty_ids(
    repository: PostgresRepository,
) -> None:
    """
    测试场景：调用 `get_repositories_for_papers` 方法，并传入一个空的 paper_id 列表。
    预期：应返回一个空字典。
    策略：直接使用空列表调用方法并断言。
    """
    result = await repository.get_repositories_for_papers([])

    assert isinstance(result, dict)
    assert len(result) == 0


async def test_get_tasks_for_papers_with_tasks(repository: PostgresRepository) -> None:
    """
    测试场景：获取具有关联任务的论文的任务列表。
    预期：返回一个字典，键是 paper_id，值是该论文关联的任务名称列表。
    策略：插入论文和关联的任务数据，调用 `get_tasks_for_papers`，验证返回字典的结构和内容。
    """
    # --- 1. 清空相关表 (确保隔离性) ---
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE papers RESTART IDENTITY CASCADE")
            await cur.execute(
                "TRUNCATE pwc_tasks RESTART IDENTITY CASCADE"
            )  # 清空任务表

    # --- 2. 准备论文数据 ---
    paper1 = Paper(
        pwc_id="test-tasks-1", title="Test Tasks Paper 1", authors=["Author Tasks"]
    )
    paper2 = Paper(
        pwc_id="test-tasks-2", title="Test Tasks Paper 2", authors=["Author Tasks"]
    )
    paper3_no_tasks = Paper(  # 没有任务的论文
        pwc_id="test-tasks-3", title="Test No Tasks Paper", authors=["Author Tasks"]
    )

    id1 = await repository.upsert_paper(paper1)
    id2 = await repository.upsert_paper(paper2)
    id3 = await repository.upsert_paper(paper3_no_tasks)
    assert id1 is not None
    assert id2 is not None
    assert id3 is not None

    # --- 3. 准备任务数据 (直接插入到 pwc_tasks 表) ---
    # 假设 pwc_tasks 表结构为 (task_id SERIAL PRIMARY KEY, paper_id INTEGER REFERENCES papers(paper_id), task_name TEXT)
    async with repository.pool.connection() as conn:
        async with conn.cursor() as cur:
            # 为 paper1 添加两个任务，为 paper2 添加一个任务
            await cur.execute(
                """
                INSERT INTO pwc_tasks (paper_id, task_name) VALUES
                (%s, %s), (%s, %s), (%s, %s)
                """,
                (id1, "Task A", id1, "Task B", id2, "Task C"),
            )

    # --- 4. 调用方法 ---
    # 请求 id1, id2, id3 的任务
    result = await repository.get_tasks_for_papers([id1, id2, id3])

    # --- 5. 验证结果 ---
    assert isinstance(result, dict)
    assert len(result) == 3  # 结果字典应包含所有请求的 paper_id
    # 验证 id1 的任务列表
    assert id1 in result
    assert isinstance(result[id1], list)
    assert set(result[id1]) == {"Task A", "Task B"}  # 使用集合忽略顺序比较
    # 验证 id2 的任务列表
    assert id2 in result
    assert isinstance(result[id2], list)
    assert set(result[id2]) == {"Task C"}
    # 验证 id3 的任务列表（应为空）
    assert id3 in result
    assert isinstance(result[id3], list)
    assert len(result[id3]) == 0


# TODO: 为 get_datasets_for_papers 和 get_repositories_for_papers 添加类似的测试用例
# 需要：
# 1. 假设或确认 pwc_datasets 和 pwc_repositories 表的结构。
# 2. 在测试中插入相应的关联数据。
# 3. 调用方法并验证返回的字典结构和内容。


# === close 测试 ===
async def test_close_connection_pool(repository: PostgresRepository) -> None:
    """
    测试场景：调用 `close` 方法来关闭仓库持有的数据库连接池。
    策略：使用 `unittest.mock.AsyncMock` 创建一个模拟的连接池对象，
          替换掉 `repository` 实例中的真实连接池，然后调用 `close` 方法，
          最后验证模拟连接池的 `close` 方法是否被调用。
    """
    # --- 1. 保存原始连接池以备恢复 ---
    original_pool = repository.pool
    assert original_pool is not None  # 确保原始池存在

    # --- 2. 创建模拟连接池并替换 ---
    mock_pool = AsyncMock(spec=AsyncConnectionPool)  # 创建模拟池，spec 确保接口匹配
    repository.pool = mock_pool  # 将仓库的 pool 属性替换为模拟对象

    try:
        # --- 3. 调用 close 方法 ---
        await repository.close()

        # --- 4. 验证模拟对象的 close 方法是否被调用 ---
        mock_pool.close.assert_awaited_once()  # 验证异步 close 方法被调用了一次
    finally:
        # --- 5. 恢复原始连接池 ---
        # 无论测试成功与否，都恢复原始的 pool 属性，避免影响其他测试
        repository.pool = original_pool
