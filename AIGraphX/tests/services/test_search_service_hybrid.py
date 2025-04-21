# tests/services/test_search_service_hybrid.py
# Contains tests potentially removed during previous edits, focusing on hybrid search.
# type: ignore
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Literal, Union, cast, Any
import json
from datetime import date

# Assuming fixtures (search_service, mock_pg_repo, etc.) are available via conftest or main test file
from aigraphx.services.search_service import (
    SearchService,
    SearchTarget,
    PaperSortByLiteral,
    ModelSortByLiteral,
    SortOrderLiteral,
    ResultItem as ServiceResultItem,
)
from aigraphx.models.search import (
    SearchResultItem,
    HFSearchResultItem,
    PaginatedPaperSearchResult,
    PaginatedSemanticSearchResult,
    PaginatedHFModelSearchResult,
    AnySearchResultItem,
    SearchFilterModel,
)
# Removed direct import of MOCK data, it's now expected to be available via fixtures (e.g., mock_pg_repo.paper_details_map)
# from .test_search_service import (
#    MOCK_PAPER_1_DETAIL_DICT, MOCK_PAPER_2_DETAIL_DICT, MOCK_PAPER_3_DETAIL_DICT,
#    MOCK_PAPER_KEY_1_DETAIL_DICT, MOCK_PAPER_KEY_2_DETAIL_DICT
# )

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# --- Restored Tests ---


async def test_perform_keyword_search_papers_empty_query(
    search_service: SearchService, mock_pg_repo
):
    """Test keyword search with an empty query string."""
    query = ""
    page = 1
    page_size = 10

    # Expect the service to return an empty result without calling the repo
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 0
    assert len(result.items) == 0
    assert result.skip == 0
    assert result.limit == page_size
    mock_pg_repo.search_papers_by_keyword.assert_not_awaited()


# Note: The following hybrid search tests assume the existence of the
# `perform_hybrid_search` method in SearchService and related helper methods.
# If `perform_hybrid_search` was also deleted or significantly changed,
# these tests will need further review and potential modification.


async def test_perform_hybrid_search_success(
    search_service: SearchService, mock_embedder, mock_faiss_paper_repo, mock_pg_repo
):
    """Test successful hybrid search combining semantic and keyword results."""
    query = "hybrid paper"
    target = cast(SearchTarget, "papers")
    page = 1
    page_size = 3

    # Access mock data via the mock_pg_repo fixture now
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]
    MOCK_PAPER_2 = mock_pg_repo.paper_details_map[2]

    # --- Mock Semantic Part ---
    mock_faiss_return = [(1, 0.1), (3, 0.2), (2, 0.4)]  # Semantic results: P1, P3, P2
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # PG returns details for semantic results
    # (mock_get_details_by_ids handles this based on fixture setup)

    # --- Mock Keyword Part ---
    # search_papers_by_keyword returns dicts directly
    mock_pg_repo.search_papers_by_keyword.side_effect = None  # Clear default
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_KEY_1, MOCK_PAPER_2],  # Keyword results: PK1, P2
        2,
    )

    # --- Call Hybrid Search ---
    results = await search_service.perform_hybrid_search(
        query=query,
        target=target,
        page=page,
        page_size=page_size,
    )

    # --- Assertions ---
    mock_embedder.embed.assert_called_once_with(query)
    mock_faiss_paper_repo.search_similar.assert_called_once()
    # Keyword search should be called with default pagination/sorting for initial fetch
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=0,
        limit=30,  # 现在使用固定值30而不是DEFAULT_TOP_N_KEYWORD
        sort_by="published_date",
        sort_order="desc",  # Defaults likely used
        published_after=None,
        published_before=None,
        filter_area=None,
    )
    # Details for semantic IDs should be fetched - 不再断言详细的ID列表顺序，采用更灵活的检查
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once()
    # 验证传入的参数包含了所有预期的ID
    call_args = mock_pg_repo.get_papers_details_by_ids.await_args[0][0]
    assert set(call_args) == {1, 2, 3, 101}  # 确保所有ID都被包含，不关心顺序

    assert isinstance(results, PaginatedPaperSearchResult)
    # Total should reflect unique papers found by either method before pagination
    # Semantic: {1, 2, 3}, Keyword: {101, 2}. Union: {1, 2, 3, 101}. Total = 4
    assert results.total == 4
    assert len(results.items) == page_size  # 3 items requested

    # Check specific fused results (exact order/scores depend on fusion logic)
    # Example: Check if the top IDs are present
    result_ids = {item.paper_id for item in results.items}
    assert 1 in result_ids  # From semantic
    assert 101 in result_ids  # From keyword


async def test_perform_hybrid_search_embedding_fails(
    search_service: SearchService, mock_embedder, mock_pg_repo
):
    """Test hybrid search when semantic search fails due to embedding error."""
    query = "embedding fail hybrid"
    target = cast(SearchTarget, "papers")
    mock_embedder.embed.return_value = None  # Simulate embedding failure

    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]

    # Keyword part should still work
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([MOCK_PAPER_KEY_1], 1)

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10
    )

    mock_embedder.embed.assert_called_once_with(query)
    search_service.faiss_repo_papers.search_similar.assert_not_called()  # Semantic search aborted
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once()  # Keyword search ran

    # Results should only contain keyword results
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == MOCK_PAPER_KEY_1["paper_id"]
    assert (
        results.items[0].score is None
    )  # Keyword results might not have score in this case


async def test_perform_hybrid_search_one_fails(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """Test hybrid search when keyword search fails but semantic search works."""
    query = "keyword fail hybrid"
    target = cast(SearchTarget, "papers")

    # Semantic part works
    mock_faiss_return = [(1, 0.1)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # mock_pg_repo.get_papers_details_by_ids will be called by semantic part

    # Keyword part fails
    mock_pg_repo.search_papers_by_keyword.side_effect = Exception("Keyword DB Error")

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10
    )

    search_service.faiss_repo_papers.search_similar.assert_called_once()
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once()  # It was attempted
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with(
        [1]
    )  # Semantic details fetched

    # Results should only contain semantic results
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == 1
    assert results.items[0].score is not None  # Semantic results have scores


async def test_perform_hybrid_search_both_fail(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo, mock_embedder
):
    """Test hybrid search when both semantic and keyword searches fail."""
    query = "both fail hybrid"
    target = cast(SearchTarget, "papers")

    # Semantic part fails (e.g., Faiss error)
    mock_faiss_paper_repo.search_similar.side_effect = Exception("Faiss Error")

    # Keyword part fails
    mock_pg_repo.search_papers_by_keyword.side_effect = Exception("Keyword DB Error")

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10
    )

    mock_embedder.embed.assert_called_once()  # Embedding attempted
    search_service.faiss_repo_papers.search_similar.assert_called_once()  # Faiss attempted
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once()  # Keyword attempted
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()  # Neither provided IDs

    # Should return empty result
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 0
    assert len(results.items) == 0


async def test_perform_hybrid_search_semantic_empty(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """Test hybrid search when semantic search returns no results."""
    query = "semantic empty hybrid"
    target = cast(SearchTarget, "papers")
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]

    # Semantic part returns empty
    mock_faiss_paper_repo.search_similar.return_value = []

    # Keyword part returns results
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([MOCK_PAPER_KEY_1], 1)

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10
    )

    search_service.faiss_repo_papers.search_similar.assert_called_once()
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once()

    # Should contain only keyword results
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == MOCK_PAPER_KEY_1["paper_id"]


async def test_perform_hybrid_search_keyword_empty(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """Test hybrid search when keyword search returns no results."""
    query = "keyword empty hybrid"
    target = cast(SearchTarget, "papers")

    # Semantic part returns results
    mock_faiss_return = [(1, 0.1)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    # mock_pg_repo.get_papers_details_by_ids handled by fixture

    # Keyword part returns empty
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([], 0)

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10
    )

    search_service.faiss_repo_papers.search_similar.assert_called_once()
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once()
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1])

    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 1
    assert len(results.items) == 1
    assert results.items[0].paper_id == 1
    assert results.items[0].score is not None


async def test_perform_hybrid_search_both_empty(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """Test hybrid search when both searches return no results."""
    query = "both empty hybrid"
    target = cast(SearchTarget, "papers")

    # Both searches return empty
    mock_faiss_paper_repo.search_similar.return_value = []
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = ([], 0)

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=1, page_size=10
    )

    search_service.faiss_repo_papers.search_similar.assert_called_once()
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once()
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()

    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 0
    assert len(results.items) == 0


async def test_perform_hybrid_search_pagination_after_fusion(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """Test hybrid search pagination, ensuring items are fused first then paginated."""
    query = "hybrid pagination"
    target = cast(SearchTarget, "papers")
    page = 2  # Request page 2
    page_size = 2  # With 2 items per page

    # Setup paper ID results
    # Semantic search returns 3 papers with IDs 1, 2, 3 with different distances
    mock_faiss_paper_repo.search_similar.return_value = [(1, 0.1), (2, 0.2), (3, 0.3)]

    # Keyword search returns 3 papers with IDs 3, 4, 5
    # Note: Paper 3 appears in both semantic and keyword results (intentional)
    keyword_papers = [
        mock_pg_repo.paper_details_map.get(3, {}),
        mock_pg_repo.paper_details_map.get(4, {}),
        mock_pg_repo.paper_details_map.get(5, {}),
    ]
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (keyword_papers, 3)

    # Total union of IDs is {1, 2, 3, 4, 5} = 5 unique papers
    # If we request page 2 with size 2, we should get papers at positions 2-3 (0-indexed)
    # Specific papers depend on fusion algorithm's ranking

    # 使用新的方法签名
    results = await search_service.perform_hybrid_search(
        query=query, target=target, page=page, page_size=page_size
    )

    # Verify we got a valid result with expected pagination properties
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 5  # Total unique papers across both search types
    assert len(results.items) == 2  # We asked for 2 per page
    assert results.skip == (page - 1) * page_size  # Skip should be correct
    assert results.limit == page_size

    # We can't assert on the exact order since that depends on the RRF fusion
    # implementation, but we can verify the items come from our expected set
    result_ids = {item.paper_id for item in results.items}
    assert all(paper_id in {1, 2, 3, 4, 5} for paper_id in result_ids)

    # Details for all potential papers should be fetched
    # Don't assert on specific IDs that were returned as the full set is needed for
    # fusion before pagination
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once()


async def test_perform_hybrid_search_filters_after_fusion(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    """Test hybrid search with filters applied after fusion of results."""
    query = "hybrid filters"
    target = cast(SearchTarget, "papers")
    # 创建过滤器对象
    filters = SearchFilterModel(
        published_after=date(2023, 1, 10),
        filter_area="CV",
        sort_by="score",
        sort_order="desc",
    )

    # Semantic results: Papers 1, 2, 3
    mock_faiss_return = [(1, 0.1), (2, 0.3), (3, 0.4)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return

    # Access mock data via mock_pg_repo
    # Modify paper 1 to have a date that passes the filter
    mock_pg_repo.paper_details_map[1]["published_date"] = date(2023, 2, 15)
    mock_pg_repo.paper_details_map[1]["area"] = "CV"

    # Add a paper that will match the filters
    mock_pg_repo.paper_details_map[102] = {
        "paper_id": 102,
        "pwc_id": "pwc_102",
        "title": "Filter Test Paper 102",
        "summary": "A paper that matches the filter criteria",
        "published_date": date(2023, 3, 1),
        "area": "CV",
        "authors": ["Author 102"],
        "pdf_url": "https://example.com/102.pdf",
    }

    # Keyword search should still return both papers
    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [
            mock_pg_repo.paper_details_map[1],
            mock_pg_repo.paper_details_map[102],
            mock_pg_repo.paper_details_map[2],
        ],
        3,
    )

    # Run hybrid search with filters
    results = await search_service.perform_hybrid_search(
        query=query,
        target=target,
        filters=filters,  # 使用新的过滤器参数
    )

    # Assertions
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 2  # Only 2 papers match the filter criteria
    assert len(results.items) == 2

    result_ids = {item.paper_id for item in results.items}
    assert 1 in result_ids  # Paper 1 matches filter
    assert 102 in result_ids  # Paper 102 matches filter
    assert 2 not in result_ids  # Paper 2 filtered out
    assert 3 not in result_ids  # Paper 3 filtered out

    # All papers should have correct area and date
    for item in results.items:
        assert item.area == "CV"
        assert item.published_date and item.published_date >= date(2023, 1, 10)


# Ensure no trailing characters or syntax errors at the end of the file.
