# type: ignore # 文件级别的类型检查忽略指令（可能是历史遗留或特定需要）
# -*- coding: utf-8 -*-
"""
文件目的：测试 `aigraphx.services.search_service.SearchService` 类。

本测试文件 (`test_search_service.py`) 专注于验证 `SearchService` 的核心搜索逻辑，
包括语义搜索和关键词搜索，以及它们对论文 (papers) 和模型 (models) 两种目标的处理。

测试策略主要是 **单元测试**：
- 使用 `pytest` 和 `unittest.mock` 来模拟（Mock）`SearchService` 的依赖项：
    - `TextEmbedder` (用于将查询转换为向量)。
    - `FaissRepository` (用于执行向量相似度搜索，区分论文和模型索引)。
    - `PostgresRepository` (用于执行关键词搜索和获取结果详情)。
- 从 `conftest.py` 中导入预定义的模拟依赖项 fixtures。
- 通过配置模拟依赖项的返回值或 `side_effect` 来控制不同的测试场景（成功、失败、部分结果、无结果等）。
- 调用 `SearchService` 的核心方法：`perform_semantic_search` 和 `perform_keyword_search`。
- 断言服务方法的返回值（通常是分页结果模型，如 `PaginatedPaperSearchResult`, `PaginatedHFModelSearchResult`）是否符合预期（例如，总数、跳过的数量、限制数量、结果项内容和顺序、分数计算）。
- 断言模拟的依赖项方法是否以正确的参数被调用（或未被调用）。
- 测试各种参数组合的影响，如分页参数 (`page`, `page_size`)、过滤参数 (`date_from`, `date_to`, `area`)、排序参数 (`sort_by`, `sort_order`)。

主要交互：
- 导入 `pytest`, `asyncio`, `unittest.mock`, `numpy`：用于测试、异步、模拟和数值计算。
- 导入 `typing`, `json`, `datetime`：用于类型提示、JSON 处理（如果需要）和日期处理。
- 导入被测试的服务类和相关类型：`SearchService`, `SearchTarget`, `SortOrderLiteral` 等。
- 导入服务使用的 Pydantic 模型：各种 `SearchResultItem` 和分页结果模型。
- （已移除）之前在此文件定义的模拟数据和 Fixtures 已移至 `conftest.py`。
- 编写测试函数 (`test_*`)：
    - 语义搜索测试（论文和模型）：覆盖成功、嵌入失败、Faiss 失败、PG 失败、部分 PG 结果、无 Faiss 结果、分页逻辑、过滤条件等场景。
    - 关键词搜索测试（论文和模型）：覆盖成功、过滤与排序、分页逻辑等场景。
    - （可选）混合搜索场景测试。

这些测试确保 `SearchService` 能够正确地协调其依赖项，执行不同类型的搜索，处理各种参数和边界情况，并返回格式正确、内容符合预期的分页结果。
"""

import pytest  # 导入 pytest 测试框架
import asyncio  # 导入 asyncio 模块
from unittest.mock import AsyncMock, MagicMock, patch  # 导入模拟工具
import numpy as np  # 导入 numpy 用于向量操作（如果需要）
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Set,
    Literal,
    Union,
    cast,
    Any,
)  # 导入类型提示
import json  # 导入 json 模块
from datetime import date  # 导入 date 类型

# Fixtures 现在定义在 conftest.py 文件中
# 模拟数据现在定义在 conftest.py 文件中

# 导入被测试的服务类和相关类型
from aigraphx.services.search_service import (
    SearchService,  # 被测试的服务类
    SearchTarget,  # 搜索目标类型 (Literal["papers", "models"])
    PaperSortByLiteral,  # 论文排序字段类型
    ModelSortByLiteral,  # 模型排序字段类型
    SortOrderLiteral,  # 排序顺序类型 (Literal["asc", "desc"])
    ResultItem as ServiceResultItem,  # 服务内部使用的 ResultItem 类型别名 (如果存在)
)

# 移除 TextEmbedder, FaissRepository, PostgresRepository 的导入，因为它们仅在 conftest.py 的 fixtures 中需要
# from aigraphx.repositories.faiss_repo import FaissRepository
# from aigraphx.repositories.postgres_repo import PostgresRepository
# from aigraphx.vectorization.embedder import TextEmbedder

# 导入搜索服务使用的 Pydantic 模型
from aigraphx.models.search import (
    SearchResultItem,  # 论文搜索结果项模型
    HFSearchResultItem,  # Hugging Face 模型搜索结果项模型
    PaginatedPaperSearchResult,  # 论文搜索分页结果模型
    PaginatedSemanticSearchResult,  # (可能已弃用或内部使用) 语义搜索分页结果模型
    PaginatedHFModelSearchResult,  # 模型搜索分页结果模型
    AnySearchResultItem,  # (可能已弃用或内部使用) 任意搜索结果项的联合类型
)

# --- 已移除: Mock 数据定义 ---
# （现在应位于 conftest.py 或测试用例内部）

# --- 已移除: Fixture 定义 ---
# (mock_embedder, mock_faiss_paper_repo, mock_faiss_model_repo, mock_pg_repo, search_service)
# 这些 fixtures 现在由 conftest.py 提供，并会被自动注入到请求它们的测试函数中。


# --- 语义搜索测试 ---
@pytest.mark.asyncio  # 标记为异步测试
async def test_perform_semantic_search_success(
    search_service: SearchService,  # 请求 conftest.py 提供的 search_service fixture
    mock_embedder,  # 请求 conftest.py 提供的 mock_embedder fixture
    mock_faiss_paper_repo,  # 请求 conftest.py 提供的 mock_faiss_paper_repo fixture
    mock_pg_repo,  # 请求 conftest.py 提供的 mock_pg_repo fixture
):
    """
    测试场景：成功执行论文的语义搜索。
    预期行为：
    1. 调用 embedder 获取查询向量。
    2. 使用查询向量调用 Faiss 论文仓库进行相似度搜索。
    3. 使用 Faiss 返回的 ID 列表调用 PG 仓库获取论文详情。
    4. 根据 Faiss 返回的距离计算得分。
    5. 对结果按得分排序并进行分页。
    6. 返回正确格式的 PaginatedPaperSearchResult。
    """
    # --- 准备 ---
    query = "test query"  # 测试查询语句
    page = 2  # 请求第二页
    page_size = 1  # 每页大小为 1
    target = cast(SearchTarget, "papers")  # 明确指定搜索目标为论文
    # 模拟 Faiss 仓库返回的相似度结果 (paper_id, distance) 列表
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]  # 注意顺序，距离越小越相似
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # 模拟 embedder 返回的查询向量
    mock_embedding = mock_embedder.embed.return_value

    # --- 执行 ---
    # 调用被测试的语义搜索方法
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 1. 验证 embedder 被调用
    mock_embedder.embed.assert_called_once_with(query)
    # 2. 验证 Faiss 仓库被调用
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC  # 获取服务中定义的默认 k 值
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding,
        k=expected_faiss_k,  # 验证调用参数
    )
    # 3. 验证 PG 仓库被调用
    #    获取 Faiss 返回的 ID 列表
    expected_pg_ids = [item[0] for item in mock_faiss_return]
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with(expected_pg_ids)
    # 4. 验证返回结果类型和分页信息
    assert isinstance(results, PaginatedPaperSearchResult), (
        "返回类型应为 PaginatedPaperSearchResult"
    )
    assert results.total == 3, (
        "总数应为 3 (基于 PG 返回的有效结果)"
    )  # total 来自 PG 返回的有效结果数
    assert results.skip == (page - 1) * page_size, "跳过的数量不正确"
    assert results.limit == page_size, "限制数量不正确"
    assert len(results.items) == page_size, "返回的项目数量不正确"
    # 5. 验证返回结果内容和顺序
    #    结果按分数降序排列 (距离升序)，分页取第 2 页，大小为 1，应取到 ID=3 的项
    assert results.items[0].paper_id == 3, "分页后的结果项不正确"
    #    验证得分计算（假设是 1 / (1 + distance)）
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.3)), "得分计算不正确"


@pytest.mark.asyncio
async def test_perform_semantic_search_embedding_fails(
    search_service: SearchService, mock_embedder, mock_pg_repo: AsyncMock
):
    """
    测试场景：文本嵌入失败（embedder 返回 None）。
    预期行为：搜索应提前终止，返回空结果，不调用 Faiss 或 PG。
    """
    # --- 准备 ---
    # 配置模拟 embedder 返回 None
    mock_embedder.embed.return_value = None
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回空的分页结果
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == [], "结果列表应为空"
    assert results.total == 0, "总数应为 0"
    # 验证 embedder 被调用
    mock_embedder.embed.assert_called_once()
    # 验证 Faiss 和 PG *未* 被调用
    search_service.faiss_repo_papers.search_similar.assert_not_called()
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_semantic_search_faiss_fails(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """
    测试场景：调用 Faiss 仓库进行搜索时发生异常。
    预期行为：搜索应在 Faiss 步骤失败，返回空结果，不调用 PG。
    """
    # --- 准备 ---
    # 配置模拟 Faiss 仓库抛出异常
    mock_faiss_paper_repo.search_similar.side_effect = Exception("Faiss Error")
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回空结果
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == [], "结果列表应为空"
    assert results.total == 0, "总数应为 0"
    # 验证 Faiss 仓库被调用（因为它引发了异常）
    mock_faiss_paper_repo.search_similar.assert_called_once()
    # 验证 PG 仓库 *未* 被调用
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_semantic_search_pg_fails(
    search_service: SearchService, mock_pg_repo, mock_faiss_paper_repo
):
    """
    测试场景：Faiss 搜索成功，但在调用 PG 仓库获取详情时发生异常。
    预期行为：搜索应在 PG 步骤失败，返回空结果。
    """
    # --- 准备 ---
    # 配置模拟 Faiss 返回一个结果
    mock_faiss_paper_repo.search_similar.return_value = [(1, 0.1)]
    # 配置模拟 PG 仓库抛出异常
    mock_pg_repo.get_papers_details_by_ids.side_effect = Exception("PG Error")
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回空结果
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == [], "结果列表应为空"
    assert results.total == 0, "总数应为 0"
    # 验证 PG 仓库被调用（因为它引发了异常）
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1])


@pytest.mark.asyncio
async def test_perform_semantic_search_partial_pg_results(
    search_service: SearchService, mock_pg_repo, mock_faiss_paper_repo
):
    """
    测试场景：Faiss 返回了多个 ID，但 PG 仓库只找到了部分 ID 的详情。
    预期行为：应只返回能在 PG 中找到详情的结果项，并相应调整 total 计数。
    """
    # --- 准备 ---
    # 模拟 Faiss 返回 ID 1, 99, 2
    mock_faiss_paper_repo.search_similar.return_value = [(1, 0.1), (99, 0.2), (2, 0.3)]
    # mock_pg_repo 的默认 side_effect (在 conftest.py 定义) 会模拟只找到 ID 1 和 2 的情况
    page = 1
    page_size = 1  # 请求第一页，大小为 1
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证 Faiss 和 PG 都被调用
    mock_faiss_paper_repo.search_similar.assert_called_once()
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 99, 2])
    # 验证返回结果
    assert isinstance(results, PaginatedPaperSearchResult)
    # total 应为 2，因为 PG 只找到了 ID 1 和 2
    assert results.total == 2, "总数应为 PG 找到的结果数"
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size
    assert len(results.items) == page_size, "返回的项目数量不正确"
    # 结果按分数排序 (距离升序)，ID 1 距离最近，应该在第一页
    assert results.items[0].paper_id == 1, "分页结果应为 ID=1"
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.1)), "得分计算不正确"


@pytest.mark.asyncio
async def test_perform_semantic_search_no_faiss_results(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """
    测试场景：Faiss 搜索没有返回任何结果。
    预期行为：搜索应提前终止，返回空结果，不调用 PG。
    """
    # --- 准备 ---
    # 配置模拟 Faiss 返回空列表
    mock_faiss_paper_repo.search_similar.return_value = []
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回空结果
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == [], "结果列表应为空"
    assert results.total == 0, "总数应为 0"
    # 验证 Faiss 被调用
    mock_faiss_paper_repo.search_similar.assert_called_once()
    # 验证 PG *未* 被调用
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_semantic_search_pagination_skip_exceeds_total(
    search_service: SearchService,
    mock_embedder: MagicMock,  # 需要 embedder 来返回值
    mock_faiss_paper_repo,
    mock_pg_repo,
):
    """
    测试场景：请求的分页 skip 值（基于 page 和 page_size）超出了实际结果总数。
    预期行为：应返回空的结果列表，但 total, skip, limit 仍然反映请求。
    """
    # --- 准备 ---
    query = "query"
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]  # Faiss 返回 3 个结果
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value
    page = 2  # 请求第 2 页
    page_size = 3  # 每页大小为 3
    # (page - 1) * page_size = 1 * 3 = 3. skip 为 3。
    # 由于总结果只有 3 个（索引 0, 1, 2），跳过 3 个后，没有结果了。
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证 Faiss 和 PG 调用
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 3, 2])
    # 验证返回结果
    assert isinstance(results, PaginatedPaperSearchResult)
    # items 列表应为空
    assert results.items == [], "结果列表应为空，因为 skip 超出总数"
    # total 仍应反映 PG 找到的总数
    assert results.total == 3, "总数应为 PG 找到的结果数"
    # skip 和 limit 应反映请求的参数
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size


@pytest.mark.asyncio
async def test_perform_semantic_search_pagination_limit_zero(
    search_service: SearchService,
    mock_embedder: MagicMock,
    mock_faiss_paper_repo,
    mock_pg_repo,
):
    """
    测试场景：请求的分页 page_size (limit) 为 0。
    预期行为：应返回空的结果列表，但 total, skip, limit 仍然反映请求。
    """
    # --- 准备 ---
    query = "query"
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value
    page = 1
    page_size = 0  # 请求大小为 0
    target = cast(SearchTarget, "papers")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证 Faiss 和 PG 调用
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 3, 2])
    # 验证返回结果
    assert isinstance(results, PaginatedPaperSearchResult)
    # items 列表应为空
    assert results.items == [], "结果列表应为空，因为 limit 为 0"
    # total 仍应反映 PG 找到的总数
    assert results.total == 3
    # skip 和 limit 应反映请求的参数
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size  # limit 为 0


@pytest.mark.asyncio
async def test_perform_semantic_search_with_filters(
    search_service: SearchService,
    mock_embedder: MagicMock,
    mock_faiss_paper_repo,
    mock_pg_repo,
):
    """
    测试场景：执行语义搜索，并应用日期过滤条件。
    预期行为：
    1. Faiss 搜索照常执行（因为过滤发生在获取详情后）。
    2. 从 PG 获取所有 Faiss 返回 ID 的详情。
    3. 服务层根据过滤条件（日期）筛选 PG 返回的结果。
    4. 返回符合过滤条件的分页结果。
    """
    # --- 准备 ---
    query = "query"
    # Faiss 返回 ID 1, 3, 2
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    # 设置日期过滤条件
    date_from = date(2023, 1, 10)  # conftest 模拟数据中 ID 1 和 2 的日期在此之后
    date_to = None

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query=query,
        target=target,
        page=page,
        page_size=page_size,
        date_from=date_from,  # 传入过滤条件
        date_to=date_to,
    )

    # --- 断言 ---
    # 验证 Faiss 和 PG 调用
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    # PG 仍然获取所有 Faiss 返回的 ID
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 3, 2])
    # 验证返回结果
    assert isinstance(results, PaginatedPaperSearchResult)
    # total 应为 2，因为 ID 3 的日期 (2023-01-01) 不满足 date_from 条件
    assert results.total == 2, "过滤后的总数应为 2"
    assert len(results.items) == 2, "过滤后的项目数应为 2"
    # 验证返回的项目是符合条件的 ID 1 和 2，并按分数排序
    assert results.items[0].paper_id == 1
    assert results.items[1].paper_id == 2
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.1))


# --- 关键词搜索测试 (论文) ---
# 注意：这部分测试逻辑已根据服务层重构进行更新
# SearchService 现在直接调用 PG 仓库的 search_papers_by_keyword 方法，
# 该方法预期直接返回包含详情的字典列表和总数，不再需要额外的 get_details 调用。


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_success(
    search_service: SearchService, mock_pg_repo
):
    """
    测试场景：成功执行论文的关键词搜索。
    预期行为：
    1. 调用 PG 仓库的 search_papers_by_keyword 方法，传递查询、分页、排序等参数。
    2. PG 仓库返回结果字典列表和总数。
    3. 服务层将字典列表转换为 SearchResultItem 列表。
    4. 返回正确格式的 PaginatedPaperSearchResult。
    """
    # --- 准备 ---
    query = "keyword"
    page = 1
    page_size = 2

    # 通过 fixture 访问 conftest.py 中定义的模拟数据
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[
        101
    ]  # 假设 PG 仓库 fixture 中有此映射
    MOCK_PAPER_3 = mock_pg_repo.paper_details_map[3]

    # --- 模拟设置 ---
    # 配置模拟 PG 仓库的 search_papers_by_keyword 方法
    mock_pg_repo.search_papers_by_keyword.side_effect = (
        None  # 清除可能存在的默认 side_effect
    )
    # 让它返回包含完整详情的字典列表和总数
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_KEY_1, MOCK_PAPER_3],  # 返回的字典列表
        3,  # 假设总共有 3 个匹配项
    )

    # --- 执行 ---
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回类型和分页信息
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 3
    assert len(result.items) == 2
    assert result.skip == 0
    assert result.limit == page_size

    # 验证返回的 items 类型和内容
    assert all(isinstance(item, SearchResultItem) for item in result.items), (
        "所有项目应为 SearchResultItem 类型"
    )
    # 使用 pwc_id (或其他唯一标识) 验证返回了正确的论文
    assert result.items[0].pwc_id == MOCK_PAPER_KEY_1["pwc_id"]
    assert result.items[1].pwc_id == MOCK_PAPER_3["pwc_id"]

    # 验证 PG 仓库的 search_papers_by_keyword 方法是否以正确的参数被调用
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,  # 验证查询词
        skip=(page - 1) * page_size,  # 验证 skip 值
        limit=page_size,  # 验证 limit 值
        sort_by="published_date",  # 验证默认排序字段
        sort_order="desc",  # 验证默认排序顺序
        published_after=None,  # 验证默认日期过滤
        published_before=None,
        filter_area=None,  # 验证默认区域过滤
    )
    # 移除对 get_papers_details_by_ids 的断言，因为它不再被调用
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_filter_sort(
    search_service: SearchService, mock_pg_repo
):
    """
    测试场景：执行论文关键词搜索，并应用过滤和排序参数。
    预期行为：
    1. 调用 PG 仓库的 search_papers_by_keyword 方法，传递所有参数。
    2. PG 仓库（模拟）返回符合条件的结果字典列表和总数。
    3. 服务层返回正确的 PaginatedPaperSearchResult。
    """
    # --- 准备 ---
    query = "filter sort"
    page = 1
    page_size = 10
    date_from = date(2023, 1, 10)
    date_to = date(2023, 2, 1)
    area = ["NLP"]  # 区域过滤（列表）
    sort_by = cast(PaperSortByLiteral, "title")  # 按标题排序
    sort_order = cast(SortOrderLiteral, "asc")  # 升序

    # 访问 conftest.py 中的模拟数据
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]

    # --- 模拟设置 ---
    # 模拟 search_papers_by_keyword 处理了过滤和排序，只返回符合条件的 MOCK_PAPER_KEY_1
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_KEY_1],  # 返回符合条件的字典列表
        1,  # 总共只有 1 个符合条件
    )

    # --- 执行 ---
    result = await search_service.perform_keyword_search(
        query=query,
        target="papers",
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        area=area,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    # --- 断言 ---
    # 验证返回结果
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].pwc_id == MOCK_PAPER_KEY_1["pwc_id"]
    # 检查返回项的 area 字段是否符合过滤条件（虽然是由 mock 保证的）
    assert result.items[0].area == area[0]

    # 验证 PG 仓库方法调用时传递了所有过滤和排序参数
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=(page - 1) * page_size,
        limit=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        published_after=date_from,
        published_before=date_to,
        filter_area=area,  # 验证 area 参数被传递
    )
    # 移除对 get_papers_details_by_ids 的断言
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_pagination_skip_exceeds(
    search_service: SearchService, mock_pg_repo
):
    """
    测试场景：关键词搜索时，请求的分页 skip 值超出了总结果数。
    预期行为：服务层应返回空的结果列表，但 total, skip, limit 反映请求。
    """
    # --- 准备 ---
    query = "skip test"
    page = 5  # 请求第 5 页
    page_size = 10  # 每页 10 条，skip = (5-1)*10 = 40

    # --- 模拟设置 ---
    # 模拟 PG 仓库对于这个高的 skip 值返回空列表，但告知总数是 40
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [],  # 当前页无结果
        40,  # 总共有 40 条匹配
    )

    # --- 执行 ---
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回结果
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 40  # total 反映总数
    assert len(result.items) == 0  # items 为空
    assert result.skip == (page - 1) * page_size  # skip 反映请求
    assert result.limit == page_size  # limit 反映请求

    # 验证 PG 仓库方法被正确调用
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=(page - 1) * page_size,
        limit=page_size,
        sort_by="published_date",  # Default
        sort_order="desc",  # Default
        published_after=None,
        published_before=None,
        filter_area=None,
    )
    # 移除对 get_papers_details_by_ids 的断言
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_pagination_limit_zero(
    search_service: SearchService, mock_pg_repo
):
    """
    测试场景：关键词搜索时，请求的分页 page_size (limit) 为 0。
    预期行为：服务层应返回空的结果列表，但 total, skip, limit 反映请求。
    """
    # --- 准备 ---
    query = "limit zero"
    page = 1
    page_size = 0  # 请求 0 条

    # --- 模拟设置 ---
    # 模拟 PG 仓库对于 limit=0 返回空列表，但告知总数
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [],  # 当前页无结果 (因为 limit=0)
        5,  # 假设总共有 5 条匹配
    )

    # --- 执行 ---
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回结果
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 5  # total 反映总数
    assert len(result.items) == 0  # items 为空
    assert result.skip == (page - 1) * page_size
    assert result.limit == page_size  # limit 为 0

    # 验证 PG 仓库方法被正确调用，limit=0
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=(page - 1) * page_size,
        limit=page_size,  # limit 参数为 0
        sort_by="published_date",  # Default
        sort_order="desc",  # Default
        published_after=None,
        published_before=None,
        filter_area=None,
    )
    # 移除对 get_papers_details_by_ids 的断言
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


# 注意：此测试用例与 test_perform_keyword_search_papers_filter_sort 非常相似，
# 可能是冗余的，或者用于测试特定组合。保留注释。
@pytest.mark.asyncio
async def test_perform_keyword_search_papers_filter_sort_duplicate(
    search_service: SearchService, mock_pg_repo
):
    """
    测试场景：执行论文关键词搜索，包含过滤和排序参数。
    (与 test_perform_keyword_search_papers_filter_sort 类似)
    """
    # --- 准备 ---
    query = "keyword filter duplicate"
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    date_from = date(2023, 1, 1)
    date_to = date(2023, 12, 31)
    area = ["CV"]
    sort_by = cast(PaperSortByLiteral, "title")
    sort_order = cast(SortOrderLiteral, "desc")

    # 定义此测试所需的模拟数据 (如果 conftest 中的不够用)
    MOCK_PAPER_4 = {
        "paper_id": 4,
        "pwc_id": "paper4",
        "title": "Model 4 Paper",
        "summary": "This is a test paper 4",
        "pdf_url": "https://example.org/paper4.pdf",
        "published_date": date(2023, 2, 15),
        "authors": ["Author 1", "Author 2"],
        "area": "CV",
        # 添加其他 SearchResultItem 可能需要的字段，即使它们不是必需的
        "score": None,
    }

    MOCK_PAPER_5 = {
        "paper_id": 5,
        "pwc_id": "paper5",
        "title": "Model 5 Paper",
        "summary": "This is a test paper 5",
        "pdf_url": "https://example.org/paper5.pdf",
        "published_date": date(2023, 3, 20),
        "authors": ["Author 3", "Author 4"],
        "area": "CV",
        "score": None,
    }

    # --- 模拟设置 ---
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    # 模拟 PG 返回符合条件的字典列表和总数
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_4, MOCK_PAPER_5],  # 返回两个模拟结果
        2,  # 总数为 2
    )

    # --- 执行 ---
    results = await search_service.perform_keyword_search(
        query=query,
        target=target,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        area=area,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    # --- 断言 ---
    # 验证 PG 仓库调用
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=0,
        limit=5,
        published_after=date_from,
        published_before=date_to,
        filter_area=area,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    # 验证返回结果
    assert results.total == 2
    assert len(results.items) == 2
    # 不再断言 get_papers_details_by_ids 调用
    # mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([4, 5])


# --- 语义搜索测试 (模型) ---
@pytest.mark.asyncio
async def test_perform_semantic_search_models_success(
    search_service: SearchService, mock_embedder, mock_faiss_model_repo, mock_pg_repo
):
    """
    测试场景：成功执行模型的语义搜索。
    预期行为：与论文语义搜索类似，但使用模型的 Faiss 仓库和 PG 的模型详情获取方法。
    """
    # --- 准备 ---
    query = "test model query"
    page = 1
    page_size = 2
    target = cast(SearchTarget, "models")  # 目标为模型
    # 模拟模型 Faiss 仓库返回 (model_id, distance) 列表
    mock_faiss_return = [("org/model1", 0.1), ("user/model3", 0.2), ("org/model2", 0.3)]
    mock_faiss_model_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query=query,
        target=target,
        page=page,
        page_size=page_size,
        sort_order="desc",  # 按得分降序
    )

    # --- 断言 ---
    # 1. 验证 embedder 调用
    mock_embedder.embed.assert_called_once_with(query)
    # 2. 验证模型 Faiss 仓库调用
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_model_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    # 3. 验证 PG 获取模型详情的方法被调用
    expected_pg_ids = [item[0] for item in mock_faiss_return]  # 获取模型 ID 列表
    mock_pg_repo.get_hf_models_by_ids.assert_awaited_once_with(expected_pg_ids)
    # 4. 验证返回结果类型和分页
    assert isinstance(results, PaginatedHFModelSearchResult), (
        "返回类型应为 PaginatedHFModelSearchResult"
    )
    # conftest 中的模拟 PG 只会返回 ID org/model1, org/model2, user/model3 的详情
    assert results.total == 3, "总数应为 PG 找到的模型数"
    assert len(results.items) == page_size, "返回项目数不正确"
    # 5. 验证返回内容和顺序（按分数降序，距离升序）
    assert results.items[0].model_id == "org/model1"  # 距离 0.1，分数最高
    assert results.items[1].model_id == "user/model3"  # 距离 0.2
    # 验证得分计算
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.1))
    assert results.items[1].score == pytest.approx(1.0 / (1.0 + 0.2))


@pytest.mark.asyncio
async def test_perform_semantic_search_models_faiss_repo_not_ready(
    search_service: SearchService,
    mock_faiss_model_repo,  # 只需要模型的Faiss模拟仓库
):
    """
    测试场景：模型 Faiss 仓库未准备就绪 (is_ready() 返回 False)。
    预期行为：搜索应提前终止，返回空结果，不调用 embedder, PG 或 Faiss 搜索。
    """
    # --- 准备 ---
    # 配置模型 Faiss 仓库的 is_ready 返回 False
    mock_faiss_model_repo.is_ready.return_value = False
    query = "test query"
    target = cast(SearchTarget, "models")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=1, page_size=10
    )

    # --- 断言 ---
    # 验证返回空结果
    assert isinstance(results, PaginatedHFModelSearchResult)
    assert results.items == []
    assert results.total == 0
    # 验证 Faiss 仓库的 search_similar *未* 被调用
    mock_faiss_model_repo.search_similar.assert_not_called()
    # 理论上 embedder 和 PG 也不会被调用，但断言 search_similar 已足够


@pytest.mark.asyncio
async def test_perform_semantic_search_models_pg_fails(
    search_service: SearchService, mock_faiss_model_repo, mock_pg_repo, mock_embedder
):
    """
    测试场景：模型 Faiss 搜索成功，但在调用 PG 获取模型详情时发生异常。
    预期行为：搜索应在 PG 步骤失败，返回空结果。
    """
    # --- 准备 ---
    # 配置 Faiss 返回结果
    mock_faiss_model_repo.search_similar.return_value = [("org/model1", 0.1)]
    # 配置 PG 获取模型详情时抛出异常
    mock_pg_repo.get_hf_models_by_ids.side_effect = Exception("PG Error")
    target = cast(SearchTarget, "models")

    # --- 执行 ---
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=1, page_size=10
    )

    # --- 断言 ---
    # 验证返回空结果
    assert isinstance(results, PaginatedHFModelSearchResult)
    assert results.items == []
    assert results.total == 0
    # 验证 Faiss 被调用
    mock_faiss_model_repo.search_similar.assert_called_once()
    # 验证 PG 被调用（并引发异常）
    mock_pg_repo.get_hf_models_by_ids.assert_awaited_once_with(["org/model1"])
