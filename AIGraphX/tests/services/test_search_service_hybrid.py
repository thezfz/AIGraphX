# -*- coding: utf-8 -*-
"""
文件目的：测试搜索服务中的混合搜索功能 (tests/services/test_search_service_hybrid.py)

本文件专门针对 `aigraphx.services.search_service.SearchService` 类中的
`perform_hybrid_search` 方法进行单元测试和集成测试（主要通过 Mock）。
混合搜索旨在结合语义搜索（基于向量相似度）和关键字搜索（基于文本匹配）的优势，
提供更全面、更相关的搜索结果。

核心测试策略：
- **依赖模拟:** 利用 `tests/services/conftest.py` 提供的 fixtures，获取一个
  `SearchService` 实例，其所有依赖（Embedder, Faiss Repos, PG Repo, Neo4j Repo）
  都已被模拟对象替换。
- **场景覆盖:** 测试覆盖了混合搜索的多种场景：
    - 成功执行，两种搜索都有结果。
    - 语义搜索失败（如嵌入失败）但关键字搜索成功。
    - 关键字搜索失败但语义搜索成功。
    - 两种搜索都失败。
    - 两种搜索都成功但其中一种或两种返回空结果。
    - 应用分页逻辑到融合后的结果。
    - 应用过滤器（如日期、领域）到融合后的结果。
- **行为验证:** 通过 `unittest.mock` 的断言方法 (`assert_called_once_with`,
  `assert_not_awaited`, etc.) 验证 `SearchService` 是否按预期调用了其依赖项的方法。
- **结果验证:** 断言 `perform_hybrid_search` 返回的结果类型 (`PaginatedPaperSearchResult`)
  以及结果中的 `total`, `skip`, `limit` 和 `items` 是否符合预期。

与其他文件的交互：
- 导入 `pytest`, `asyncio`, `unittest.mock` 等测试和异步库。
- 导入 `aigraphx.services.search_service` 中的 `SearchService` 类和相关类型定义。
- 导入 `aigraphx.models.search` 中的 Pydantic 模型用于类型提示和结果校验。
- **关键依赖:** `tests/services/conftest.py` 中定义的 fixtures，特别是 `search_service`
  及其模拟依赖 (`mock_embedder`, `mock_faiss_paper_repo`, `mock_pg_repo`, etc.)。
"""
# type: ignore # 可以在文件开头添加，忽略整个文件的类型检查错误，如果需要的话
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch # 导入模拟工具
import numpy as np # 导入 numpy 用于处理向量
from typing import List, Tuple, Dict, Optional, Set, Literal, Union, cast, Any # 导入类型提示
import json # 导入 json
from datetime import date # 导入 date 类型

# 导入被测试的服务类和相关类型
from aigraphx.services.search_service import (
    SearchService,
    SearchTarget, # 搜索目标类型 (papers/models)
    PaperSortByLiteral, # 论文排序字段字面量类型
    ModelSortByLiteral, # 模型排序字段字面量类型
    SortOrderLiteral, # 排序顺序字面量类型 (asc/desc)
    ResultItem as ServiceResultItem, # 服务内部可能使用的结果项类型别名
)
# 导入 API/服务层使用的 Pydantic 模型
from aigraphx.models.search import (
    SearchResultItem, # 论文搜索结果项
    HFSearchResultItem, # HF 模型搜索结果项
    PaginatedPaperSearchResult, # 分页的论文结果
    PaginatedSemanticSearchResult, # 分页的语义结果
    PaginatedHFModelSearchResult, # 分页的 HF 模型结果
    AnySearchResultItem, # 论文或模型结果项的联合类型
    SearchFilterModel, # 搜索过滤器模型
)
# 移除对其他测试文件中 MOCK 数据的直接导入，现在依赖 conftest 中的 fixture 提供模拟数据
# from .test_search_service import (
#    MOCK_PAPER_1_DETAIL_DICT, MOCK_PAPER_2_DETAIL_DICT, MOCK_PAPER_3_DETAIL_DICT,
#    MOCK_PAPER_KEY_1_DETAIL_DICT, MOCK_PAPER_KEY_2_DETAIL_DICT
# )

# 标记模块内所有测试为异步
pytestmark = pytest.mark.asyncio

# --- 恢复的测试用例 ---
# 这些测试用例之前可能存在于其他文件中，现在集中在这里测试混合搜索逻辑。

async def test_perform_keyword_search_papers_empty_query(
    search_service: SearchService, # 注入配置了模拟依赖的 SearchService 实例
    mock_pg_repo: MagicMock # 注入模拟的 PostgresRepository 实例
) -> None:
    """
    测试场景：执行关键字搜索，但提供的查询字符串为空。
    预期：服务应直接返回空结果，而不调用底层仓库的搜索方法。
    """
    query = "" # 空查询字符串
    page = 1
    page_size = 10
    target = cast(SearchTarget, "papers") # 明确目标类型

    # 调用关键字搜索方法
    result = await search_service.perform_keyword_search(
        query=query, target=target, page=page, page_size=page_size
    )

    # --- 断言 ---
    # 验证返回结果是正确的分页类型
    assert isinstance(result, PaginatedPaperSearchResult)
    # 验证总数和项数都为 0
    assert result.total == 0
    assert len(result.items) == 0
    # 验证分页参数
    assert result.skip == (page - 1) * page_size
    assert result.limit == page_size
    # 验证底层 PG 仓库的搜索方法未被调用
    mock_pg_repo.search_papers_by_keyword.assert_not_awaited()


# 注意: 以下混合搜索测试假设 SearchService 中存在 perform_hybrid_search 方法
# 并且该方法内部会调用类似 _perform_semantic_search_internal,
# _perform_keyword_search_internal, _fetch_details_for_semantic_results,
# _fuse_results, _apply_filters_and_sort, _paginate_results 等辅助方法。
# 如果实际实现不同，测试断言需要相应调整。

async def test_perform_hybrid_search_success(
    search_service: SearchService, # 注入 SearchService 实例
    mock_embedder: MagicMock, # 注入模拟的 Embedder
    mock_faiss_paper_repo: MagicMock, # 注入模拟的 Faiss 仓库 (论文)
    mock_pg_repo: MagicMock # 注入模拟的 PG 仓库
) -> None:
    """
    测试场景：成功执行混合搜索，语义搜索和关键字搜索都返回结果。
    预期：服务能正确调用嵌入器、Faiss 仓库、PG 仓库，融合结果并返回分页数据。
    策略：配置 Faiss 和 PG 模拟仓库返回不同的结果集（可以有重叠），
          然后调用混合搜索，验证所有依赖项是否被正确调用，以及最终结果的结构和总数。
    """
    query = "hybrid paper"
    target = cast(SearchTarget, "papers")
    page = 1
    page_size = 3

    # --- 模拟语义搜索部分 ---
    # Faiss 仓库返回 (paper_id, score) 列表，模拟找到 1, 3, 2
    mock_faiss_return = [(1, 0.1), (3, 0.2), (2, 0.4)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # PG 仓库的 get_papers_details_by_ids 会被调用以获取这些 ID 的详情
    # 这个行为已在 conftest.py 的 mock_pg_repo fixture 中配置好

    # --- 模拟关键字搜索部分 ---
    # 从 mock_pg_repo fixture 中获取预定义的模拟详情数据
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]
    MOCK_PAPER_2 = mock_pg_repo.paper_details_map[2]
    # 配置 PG 仓库的关键字搜索返回包含 PK1 和 P2 的字典列表，总数为 2
    mock_pg_repo.search_papers_by_keyword.side_effect = None # 清除 conftest 中可能存在的默认 side_effect
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_KEY_1, MOCK_PAPER_2], # 关键字结果列表
        2, # 关键字结果总数
    )

    # --- 调用混合搜索 ---
    # 创建一个空的 SearchFilterModel 实例，因为这里不测试过滤
    filters = SearchFilterModel(
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query,
        target=target,
        page=page,
        page_size=page_size,
        filters=filters # 传递过滤器对象
    )

    # --- 断言依赖调用 ---
    # 1. 嵌入器应被调用一次以获取查询向量
    mock_embedder.embed.assert_called_once_with(query)
    # 2. Faiss 仓库的 search_similar 应被调用一次
    mock_faiss_paper_repo.search_similar.assert_called_once()
    # 3. PG 仓库的关键字搜索应被调用一次，用于获取初始关键字结果
    #    limit 参数现在使用内部常量 DEFAULT_TOP_N_KEYWORD (假设为 30)
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=0,
        limit=30, # 内部用于获取融合前数据的限制值
        sort_by="published_date", # 内部默认排序
        sort_order="desc",
        published_after=None, # 无过滤
        published_before=None,
        filter_area=None,
    )
    # 4. PG 仓库的 get_papers_details_by_ids 应被调用一次，以获取所有潜在结果的详情
    #    (包括语义结果 ID 和关键字结果 ID 的并集)
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once()
    #    获取调用时的参数 (即 ID 列表)
    call_args_list = mock_pg_repo.get_papers_details_by_ids.await_args[0][0]
    #    验证 ID 列表是否包含所有预期的 ID (来自语义: 1, 2, 3; 来自关键字: 101, 2)
    assert set(call_args_list) == {1, 2, 3, 101} # 并集是 {1, 2, 3, 101}

    # --- 断言返回结果 ---
    assert isinstance(results, PaginatedPaperSearchResult) # 验证返回类型
    # 总数应为融合后的唯一结果数 (语义={1,2,3}, 关键字={101,2}, 并集={1,2,3,101}, 总数=4)
    assert results.total == 4
    assert len(results.items) == page_size # 返回项数应等于请求的 page_size
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size

    # 验证返回项的内容（由于融合算法不确定，只检查部分 ID 是否存在）
    result_ids = {item.paper_id for item in results.items}
    assert 1 in result_ids # 语义结果 ID
    assert 101 in result_ids # 关键字结果 ID
    # 可能包含 2 或 3，取决于融合排序


async def test_perform_hybrid_search_embedding_fails(
    search_service: SearchService, 
    mock_embedder: MagicMock, 
    mock_pg_repo: MagicMock,
    mock_faiss_paper_repo: MagicMock # 添加缺少的 mock_faiss_paper_repo 参数
) -> None:
    """
    测试场景：混合搜索时，文本嵌入失败 (语义搜索部分失败)。
    预期：混合搜索应能回退到只返回关键字搜索的结果。
    策略：配置模拟嵌入器的 `embed` 方法返回 None，配置 PG 关键字搜索返回结果，
          调用混合搜索，验证嵌入器被调用、Faiss 未被调用、PG 关键字搜索被调用，
          并且最终结果只包含关键字搜索的部分。
    """
    query = "embedding fail hybrid"
    target = cast(SearchTarget, "papers")
    # --- 模拟嵌入失败 ---
    mock_embedder.embed.return_value = None

    # --- 模拟关键字搜索成功 ---
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([MOCK_PAPER_KEY_1], 1)

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10, filters=filters
    )

    # --- 断言依赖调用 ---
    mock_embedder.embed.assert_called_once() # 嵌入被尝试
    mock_faiss_paper_repo.search_similar.assert_not_called() # Faiss 搜索因嵌入失败未执行
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once() # 关键字搜索执行

    # --- 断言返回结果 ---
    # 结果应只包含关键字搜索的部分
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == MOCK_PAPER_KEY_1["paper_id"]
    assert results.items[0].score is None


async def test_perform_hybrid_search_one_fails(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock
) -> None:
    """
    测试场景：混合搜索时，关键字搜索失败，但语义搜索成功。
    预期：混合搜索应能回退到只返回语义搜索的结果。
    策略：配置 Faiss 搜索返回结果，配置 PG 关键字搜索抛出异常，
          调用混合搜索，验证 Faiss 和 PG 都被尝试调用，并且最终结果只包含语义部分。
    """
    query = "keyword fail hybrid"
    target = cast(SearchTarget, "papers")

    # --- 模拟语义搜索成功 ---
    mock_faiss_return = [(1, 0.1)] # 语义找到 paper 1
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # PG 的 get_papers_details_by_ids 会被调用获取 paper 1 的详情

    # --- 模拟关键字搜索失败 ---
    mock_pg_repo.search_papers_by_keyword.side_effect = Exception("Keyword DB Error")

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10, filters=filters
    )

    # --- 断言依赖调用 ---
    mock_faiss_paper_repo.search_similar.assert_called_once() # 语义搜索执行
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once() # 关键字搜索被尝试
    # 获取语义结果 ID 1 的详情应被调用
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1], scores={1: pytest.approx(0.1)}) # 传递了ID和分数

    # --- 断言返回结果 ---
    # 结果应只包含语义搜索的部分
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == 1
    assert results.items[0].score is not None # 语义结果应有分数


async def test_perform_hybrid_search_both_fail(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock, 
    mock_embedder: MagicMock
) -> None:
    """
    测试场景：混合搜索时，语义搜索和关键字搜索都失败。
    预期：混合搜索应返回空结果。
    策略：配置 Faiss 搜索和 PG 关键字搜索都抛出异常，调用混合搜索，
          验证嵌入器、Faiss 和 PG 都被尝试调用，并且最终结果为空。
    """
    query = "both fail hybrid"
    target = cast(SearchTarget, "papers")

    # --- 模拟语义搜索失败 ---
    mock_faiss_paper_repo.search_similar.side_effect = Exception("Faiss Error")

    # --- 模拟关键字搜索失败 ---
    mock_pg_repo.search_papers_by_keyword.side_effect = Exception("Keyword DB Error")

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10, filters=filters
    )

    # --- 断言依赖调用 ---
    mock_embedder.embed.assert_called_once() # 嵌入被尝试
    mock_faiss_paper_repo.search_similar.assert_called_once() # Faiss 被尝试
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once() # 关键字搜索被尝试
    # 因为两种搜索都失败，没有有效的 ID，所以不应调用获取详情的方法
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()

    # --- 断言返回结果 ---
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 0
    assert len(results.items) == 0


async def test_perform_hybrid_search_semantic_empty(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock
) -> None:
    """
    测试场景：混合搜索时，语义搜索成功但返回空结果，关键字搜索成功。
    预期：混合搜索结果应只包含关键字搜索的部分。
    策略：配置 Faiss 返回空列表，配置 PG 关键字搜索返回结果，调用混合搜索，
          验证 Faiss 和 PG 都被调用，最终结果只包含关键字部分。
    """
    query = "semantic empty hybrid"
    target = cast(SearchTarget, "papers")
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]

    # --- 模拟语义搜索为空 ---
    mock_faiss_paper_repo.search_similar.return_value = []

    # --- 模拟关键字搜索成功 ---
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([MOCK_PAPER_KEY_1], 1)

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10, filters=filters
    )

    # --- 断言依赖调用 ---
    mock_faiss_paper_repo.search_similar.assert_called_once() # 语义搜索被调用
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once() # 关键字搜索被调用
    # 应该调用了 get_papers_details_by_ids，但只包含关键字搜索的 ID
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([101], scores={}) # 确保调用，scores为空


    # --- 断言返回结果 ---
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == MOCK_PAPER_KEY_1["paper_id"]


async def test_perform_hybrid_search_keyword_empty(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock
) -> None:
    """
    测试场景：混合搜索时，关键字搜索成功但返回空结果，语义搜索成功。
    预期：混合搜索结果应只包含语义搜索的部分。
    策略：配置 Faiss 搜索返回结果，配置 PG 关键字搜索返回空列表，调用混合搜索，
          验证 Faiss 和 PG 都被调用，最终结果只包含语义部分。
    """
    query = "keyword empty hybrid"
    target = cast(SearchTarget, "papers")

    # --- 模拟语义搜索成功 ---
    mock_faiss_return = [(1, 0.1)] # 语义找到 paper 1
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # PG get_papers_details_by_ids 会被调用

    # --- 模拟关键字搜索为空 ---
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([], 0) # 返回空列表和 0

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10, filters=filters
    )

    # --- 断言依赖调用 ---
    mock_faiss_paper_repo.search_similar.assert_called_once() # 语义搜索被调用
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once() # 关键字搜索被调用
    # 只应获取语义结果 ID 1 的详情
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1], scores={1: pytest.approx(0.1)}) # 传递了ID和分数

    # --- 断言返回结果 ---
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == 1
    assert results.items[0].score is not None # 语义结果有分数


async def test_perform_hybrid_search_both_empty(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock
) -> None:
    """
    测试场景：混合搜索时，语义搜索和关键字搜索都成功执行但都返回空结果。
    预期：混合搜索应返回空结果。
    策略：配置 Faiss 和 PG 都返回空列表，调用混合搜索，验证依赖调用和最终结果。
    """
    query = "both empty hybrid"
    target = cast(SearchTarget, "papers")

    # --- 模拟两种搜索都为空 ---
    mock_faiss_paper_repo.search_similar.return_value = []
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([], 0)

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10, filters=filters
    )

    # --- 断言依赖调用 ---
    mock_faiss_paper_repo.search_similar.assert_called_once() # 语义搜索被调用
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once() # 关键字搜索被调用
    # 因为没有 ID，不应调用获取详情的方法
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()

    # --- 断言返回结果 ---
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 0
    assert len(results.items) == 0


async def test_perform_hybrid_search_pagination_after_fusion(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock
) -> None:
    """
    测试场景：验证混合搜索的分页是在结果融合之后应用的。
    预期：返回结果应是融合和重新排序后的完整结果集的正确分页。
    策略：配置语义和关键字搜索返回不同的、部分重叠的结果集。请求第 2 页，
          验证返回的总数是融合后的唯一结果总数，返回的项数符合分页大小，
          并且 skip 值正确。由于融合算法不确定，不严格断言返回的具体 ID 顺序，
          但验证它们属于预期的全集。
    """
    query = "hybrid pagination"
    target = cast(SearchTarget, "papers")
    page = 2  # 请求第 2 页
    page_size = 2  # 每页 2 项

    # --- 模拟搜索结果 ---
    # 语义: 返回 ID 1, 2, 3
    mock_faiss_paper_repo.search_similar.return_value = [(1, 0.1), (2, 0.2), (3, 0.3)]
    # 关键字: 返回 ID 3, 4, 5 (ID 3 重叠)
    keyword_papers = [
        mock_pg_repo.paper_details_map.get(3, {}), # 获取 paper 3 详情
        mock_pg_repo.paper_details_map.get(4, {}), # 获取 paper 4 详情
        mock_pg_repo.paper_details_map.get(5, {}), # 获取 paper 5 详情
    ]
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (keyword_papers, 3)

    # --- 调用混合搜索 ---
    filters = SearchFilterModel( # 空过滤器，显式设置 None
        published_after=None,
        published_before=None,
        filter_area=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
        sort_by=None,
        sort_order=None
    )
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=page, page_size=page_size, filters=filters
    )

    # --- 断言分页属性 ---
    assert isinstance(results, PaginatedPaperSearchResult)
    # 融合后的唯一 ID 集合是 {1, 2, 3, 4, 5}，总数为 5
    assert results.total == 5
    assert len(results.items) == page_size # 返回项数应为 page_size (2)
    assert results.skip == (page - 1) * page_size # skip 值应为 (2-1)*2 = 2
    assert results.limit == page_size

    # --- 断言返回项内容 (不依赖具体排序) ---
    result_ids = {item.paper_id for item in results.items}
    # 验证返回的 ID 属于融合后的全集 {1, 2, 3, 4, 5}
    assert all(paper_id in {1, 2, 3, 4, 5} for paper_id in result_ids)

    # --- 断言依赖调用 ---
    # 获取详情时应请求所有可能的 ID (语义 + 关键字)
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once()
    call_args_list = mock_pg_repo.get_papers_details_by_ids.await_args[0][0]
    assert set(call_args_list) == {1, 2, 3, 4, 5} # 请求了融合前的所有 ID


async def test_perform_hybrid_search_filters_after_fusion(
    search_service: SearchService, 
    mock_faiss_paper_repo: MagicMock, 
    mock_pg_repo: MagicMock
) -> None:
    """
    测试场景：验证混合搜索的过滤器是在结果融合之后应用的。
    预期：返回的结果应只包含那些既出现在融合结果中、又满足过滤条件的项。
    策略：配置语义和关键字搜索结果，创建一个过滤器（如日期和领域），
          确保部分融合结果满足过滤器，部分不满足。调用带过滤器的混合搜索，
          验证返回的总数和项只包含满足条件的。
    """
    query = "hybrid filters"
    target = cast(SearchTarget, "papers")
    # --- 定义过滤器 ---
    filters = SearchFilterModel(
        published_after=date(2023, 1, 10), # 日期在此之后
        filter_area=["CV"], # 领域是 CV (修正为列表)
        sort_by="score", # 按分数排序 (融合后的)
        sort_order="desc",
        # 显式设置其他过滤器为 None
        published_before=None,
        pipeline_tag=None,
        filter_authors=None,
        filter_library_name=None,
        filter_tags=None,
        filter_author=None,
    )

    # --- 模拟搜索结果 ---
    # 语义: 返回 ID 1, 2, 3
    mock_faiss_return = [(1, 0.1), (2, 0.3), (3, 0.4)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return

    # --- 准备模拟详情数据 (需要满足/不满足过滤条件) ---
    # 修改 paper 1: 满足日期和领域过滤器
    mock_pg_repo.paper_details_map[1]["published_date"] = date(2023, 2, 15)
    mock_pg_repo.paper_details_map[1]["area"] = "CV"
    # Paper 2 (来自 conftest): area=NLP, 不满足过滤器
    # Paper 3 (来自 conftest): area=CV, 但日期 2023-1-5, 不满足过滤器
    # 添加 paper 102: 满足日期和领域过滤器
    mock_pg_repo.paper_details_map[102] = {
        "paper_id": 102, "pwc_id": "pwc_102", "title": "Filter Test Paper 102",
        "summary": "Matches filters", "published_date": date(2023, 3, 1),
        "area": "CV", "authors": ["Author 102"], "pdf_url": "https://example.com/102.pdf",
    }

    # --- 模拟关键字搜索结果 ---
    # 假设关键字搜索返回 paper 1, 102, 2 (其中 1, 102 满足过滤器, 2 不满足)
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [
            mock_pg_repo.paper_details_map[1],
            mock_pg_repo.paper_details_map[102],
            mock_pg_repo.paper_details_map[2],
        ],
        3, # 总数 3
    )

    # --- 调用带过滤器的混合搜索 ---
    results = await search_service.perform_hybrid_search(
        query=query,
        target=target,
        filters=filters, # 传入过滤器对象
        page=1, # 添加默认分页参数
        page_size=10
    )

    # --- 断言结果 ---
    assert isinstance(results, PaginatedPaperSearchResult)
    # 融合前的 ID 集合: {1, 2, 3} U {1, 102, 2} = {1, 2, 3, 102}
    # 满足过滤器的 ID: {1, 102}
    assert results.total == 2 # 总数应为满足过滤条件的数量
    assert len(results.items) == 2 # 返回项数也是 2

    # 验证返回的 ID 是否正确
    result_ids = {item.paper_id for item in results.items}
    assert 1 in result_ids
    assert 102 in result_ids
    assert 2 not in result_ids
    assert 3 not in result_ids

    # 验证返回项都满足过滤条件
    for item in results.items:
        assert item.area == "CV"
        assert item.published_date and item.published_date >= date(2023, 1, 10)

# 文件末尾确保没有多余字符或语法错误