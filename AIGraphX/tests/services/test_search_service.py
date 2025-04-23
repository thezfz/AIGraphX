# type: ignore
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Literal, Union, cast, Any
import json
from datetime import date

# Fixtures are now in conftest.py
# Mock data is now in conftest.py

from aigraphx.services.search_service import (
    SearchService,
    SearchTarget,
    PaperSortByLiteral,
    ModelSortByLiteral,
    SortOrderLiteral,
    ResultItem as ServiceResultItem,
)

# Removed TextEmbedder, FaissRepository, PostgresRepository imports as they are only needed for fixtures in conftest
from aigraphx.models.search import (
    SearchResultItem,
    HFSearchResultItem,
    PaginatedPaperSearchResult,
    PaginatedSemanticSearchResult,
    PaginatedHFModelSearchResult,
    AnySearchResultItem,
)

# --- REMOVED Mock Data Definitions ---

# --- REMOVED Fixture Definitions (mock_embedder, mock_faiss_paper_repo, mock_faiss_model_repo, mock_pg_repo, search_service) ---


# --- Semantic Search Tests ---
@pytest.mark.asyncio
async def test_perform_semantic_search_success(
    search_service: SearchService,
    mock_embedder,
    mock_faiss_paper_repo,
    mock_pg_repo,  # Fixtures provided by conftest
):
    query = "test query"
    page = 2
    page_size = 1
    target = cast(SearchTarget, "papers")
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value

    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size
    )

    mock_embedder.embed.assert_called_once_with(query)
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    expected_pg_ids = [item[0] for item in mock_faiss_return]
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with(expected_pg_ids)
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 3
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size
    assert len(results.items) == page_size
    assert results.items[0].paper_id == 3
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.3))


@pytest.mark.asyncio
async def test_perform_semantic_search_embedding_fails(
    search_service: SearchService, mock_embedder, mock_pg_repo: AsyncMock
):
    mock_embedder.embed.return_value = None
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == []
    assert results.total == 0
    mock_embedder.embed.assert_called_once()
    search_service.faiss_repo_papers.search_similar.assert_not_called()
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_semantic_search_faiss_fails(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    mock_faiss_paper_repo.search_similar.side_effect = Exception("Faiss Error")
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == []
    assert results.total == 0
    mock_faiss_paper_repo.search_similar.assert_called_once()
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_semantic_search_pg_fails(
    search_service: SearchService, mock_pg_repo, mock_faiss_paper_repo
):
    mock_faiss_paper_repo.search_similar.return_value = [(1, 0.1)]
    mock_pg_repo.get_papers_details_by_ids.side_effect = Exception("PG Error")
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == []
    assert results.total == 0
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1])


@pytest.mark.asyncio
async def test_perform_semantic_search_partial_pg_results(
    search_service: SearchService, mock_pg_repo, mock_faiss_paper_repo
):
    mock_faiss_paper_repo.search_similar.return_value = [(1, 0.1), (99, 0.2), (2, 0.3)]
    page = 1
    page_size = 1
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )
    mock_faiss_paper_repo.search_similar.assert_called_once()
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 99, 2])
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 2
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size
    assert len(results.items) == page_size
    assert results.items[0].paper_id == 1
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.1))


@pytest.mark.asyncio
async def test_perform_semantic_search_no_faiss_results(
    search_service: SearchService, mock_faiss_paper_repo, mock_pg_repo
):
    mock_faiss_paper_repo.search_similar.return_value = []
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=page, page_size=page_size
    )
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == []
    assert results.total == 0
    mock_faiss_paper_repo.search_similar.assert_called_once()
    mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_semantic_search_pagination_skip_exceeds_total(
    search_service: SearchService,
    mock_embedder: MagicMock,
    mock_faiss_paper_repo,
    mock_pg_repo,
):
    query = "query"
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value
    page = 2
    page_size = 3
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size
    )
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 3, 2])
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == []
    assert results.total == 3
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size


@pytest.mark.asyncio
async def test_perform_semantic_search_pagination_limit_zero(
    search_service: SearchService,
    mock_embedder: MagicMock,
    mock_faiss_paper_repo,
    mock_pg_repo,
):
    query = "query"
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value
    page = 1
    page_size = 0
    target = cast(SearchTarget, "papers")
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size
    )
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 3, 2])
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.items == []
    assert results.total == 3
    assert results.skip == (page - 1) * page_size
    assert results.limit == page_size


@pytest.mark.asyncio
async def test_perform_semantic_search_with_filters(
    search_service: SearchService,
    mock_embedder: MagicMock,
    mock_faiss_paper_repo,
    mock_pg_repo,
):
    query = "query"
    mock_faiss_return = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock_faiss_paper_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    date_from = date(2023, 1, 10)
    date_to = None
    results = await search_service.perform_semantic_search(
        query=query,
        target=target,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
    )
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_paper_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([1, 3, 2])
    assert isinstance(results, PaginatedPaperSearchResult)
    assert results.total == 2
    assert len(results.items) == 2
    assert results.items[0].paper_id == 1
    assert results.items[1].paper_id == 2
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.1))


# --- Keyword Search Tests (Papers) ---
@pytest.mark.asyncio
async def test_perform_keyword_search_papers_success(
    search_service: SearchService, mock_pg_repo
):
    query = "keyword"
    page = 1
    page_size = 2

    # Access mock data via fixture
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]
    MOCK_PAPER_3 = mock_pg_repo.paper_details_map[3]

    # --- MOCK SETUP ---
    # search_papers_by_keyword now returns dicts
    mock_pg_repo.search_papers_by_keyword.side_effect = (
        None  # Clear default side effect
    )
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_KEY_1, MOCK_PAPER_3],
        3,  # Assume total of 3 matching papers for this query
    )

    # --- CALL --- #
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    # --- ASSERTIONS --- #
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 3
    assert len(result.items) == 2
    assert result.skip == 0
    assert result.limit == page_size

    # Check item types and content (using pwc_id as unique identifier)
    assert all(isinstance(item, SearchResultItem) for item in result.items)
    assert result.items[0].pwc_id == MOCK_PAPER_KEY_1["pwc_id"]
    assert result.items[1].pwc_id == MOCK_PAPER_3["pwc_id"]

    # Check that search_papers_by_keyword was called correctly
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=(page - 1) * page_size,
        limit=page_size,
        sort_by="published_date",
        sort_order="desc",
        published_after=None,
        published_before=None,
        filter_area=None,
    )
    # REMOVED: Assertion for get_papers_details_by_ids - no longer called
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_filter_sort(
    search_service: SearchService, mock_pg_repo
):
    query = "filter sort"
    page = 1
    page_size = 10
    date_from = date(2023, 1, 10)
    date_to = date(2023, 2, 1)
    area = ["NLP"]
    sort_by = cast(PaperSortByLiteral, "title")
    sort_order = cast(SortOrderLiteral, "asc")

    # Access mock data via fixture
    MOCK_PAPER_KEY_1 = mock_pg_repo.paper_details_map[101]

    # --- MOCK SETUP ---
    # Simulate search_papers_by_keyword handling filters/sort internally
    # and returning the filtered/sorted dicts
    mock_pg_repo.search_papers_by_keyword.side_effect = None  # Clear default
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_KEY_1],  # Only paper key 1 matches filters/sort
        1,  # Total count matching filters
    )

    # --- CALL --- #
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

    # --- ASSERTIONS --- #
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].pwc_id == MOCK_PAPER_KEY_1["pwc_id"]
    assert (
        result.items[0].area == area[0]
    )  # Check if filter applied (mock simulates it)

    # Check search_papers_by_keyword was called with filters/sort
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=(page - 1) * page_size,
        limit=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        published_after=date_from,
        published_before=date_to,
        filter_area=area,
    )
    # REMOVED: Assertion for get_papers_details_by_ids
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_pagination_skip_exceeds(
    search_service: SearchService, mock_pg_repo
):
    query = "skip test"
    page = 5  # Request page 5
    page_size = 10

    # --- MOCK SETUP ---
    # Simulate DB returning 0 results for this high skip value, but a total count
    mock_pg_repo.search_papers_by_keyword.side_effect = None  # Clear default
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [],  # No results for this page
        40,  # Assume 40 total matching results exist
    )

    # --- CALL --- #
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    # --- ASSERTIONS --- #
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 40
    assert len(result.items) == 0  # No items returned for this page
    assert result.skip == (page - 1) * page_size  # Skip should reflect requested page
    assert result.limit == page_size

    # Check search_papers_by_keyword was called
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
    # REMOVED: Assertion for get_papers_details_by_ids
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_pagination_limit_zero(
    search_service: SearchService, mock_pg_repo
):
    query = "limit zero"
    page = 1
    page_size = 0  # Request 0 items

    # --- MOCK SETUP ---
    # Simulate DB returning 0 results due to limit, but a total count
    mock_pg_repo.search_papers_by_keyword.side_effect = None  # Clear default
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [],  # No results for limit=0
        5,  # Assume 5 total matching results exist
    )

    # --- CALL --- #
    result = await search_service.perform_keyword_search(
        query=query, target="papers", page=page, page_size=page_size
    )

    # --- ASSERTIONS --- #
    assert isinstance(result, PaginatedPaperSearchResult)
    assert result.total == 5
    assert len(result.items) == 0  # No items returned
    assert result.skip == (page - 1) * page_size
    assert result.limit == page_size  # Limit should be 0

    # Check search_papers_by_keyword was called with limit=0
    mock_pg_repo.search_papers_by_keyword.assert_awaited_once_with(
        query=query,
        skip=(page - 1) * page_size,
        limit=page_size,  # limit=0
        sort_by="published_date",  # Default
        sort_order="desc",  # Default
        published_after=None,
        published_before=None,
        filter_area=None,
    )
    # REMOVED: Assertion for get_papers_details_by_ids
    # mock_pg_repo.get_papers_details_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_keyword_search_papers_filter_sort_duplicate(
    search_service: SearchService, mock_pg_repo
):
    query = "keyword filter duplicate"
    page = 1
    page_size = 5
    target = cast(SearchTarget, "papers")
    date_from = date(2023, 1, 1)
    date_to = date(2023, 12, 31)
    area = ["CV"]
    sort_by = cast(PaperSortByLiteral, "title")
    sort_order = cast(SortOrderLiteral, "desc")

    # 定义测试需要的模拟数据
    MOCK_PAPER_4 = {
        "paper_id": 4,
        "pwc_id": "paper4",
        "title": "Model 4 Paper",
        "summary": "This is a test paper 4",
        "pdf_url": "https://example.org/paper4.pdf",
        "published_date": date(2023, 2, 15),
        "authors": ["Author 1", "Author 2"],
        "area": "CV",
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
    }

    mock_pg_repo.search_papers_by_keyword.side_effect = None
    mock_pg_repo.search_papers_by_keyword.return_value = (
        [MOCK_PAPER_4, MOCK_PAPER_5],
        2,
    )

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
    assert results.total == 2
    assert len(results.items) == 2
    # 不应再断言调用get_papers_details_by_ids，因为search_papers_by_keyword已直接返回完整字典
    # mock_pg_repo.get_papers_details_by_ids.assert_awaited_once_with([4, 5])


# --- Tests for Semantic Search (Models) ---
@pytest.mark.asyncio
async def test_perform_semantic_search_models_success(
    search_service: SearchService, mock_embedder, mock_faiss_model_repo, mock_pg_repo
):
    query = "test model query"
    page = 1
    page_size = 2
    target = cast(SearchTarget, "models")
    mock_faiss_return = [("org/model1", 0.1), ("user/model3", 0.2), ("org/model2", 0.3)]
    mock_faiss_model_repo.search_similar.return_value = mock_faiss_return
    mock_embedding = mock_embedder.embed.return_value

    results = await search_service.perform_semantic_search(
        query=query, target=target, page=page, page_size=page_size, sort_order="desc"
    )

    mock_embedder.embed.assert_called_once_with(query)
    expected_faiss_k = SearchService.DEFAULT_TOP_N_SEMANTIC
    mock_faiss_model_repo.search_similar.assert_called_once_with(
        mock_embedding, k=expected_faiss_k
    )
    expected_pg_ids = [item[0] for item in mock_faiss_return]
    mock_pg_repo.get_hf_models_by_ids.assert_awaited_once_with(expected_pg_ids)
    assert isinstance(results, PaginatedHFModelSearchResult)
    assert results.total == 3
    assert len(results.items) == page_size
    assert results.items[0].model_id == "org/model1"
    assert results.items[1].model_id == "user/model3"
    assert results.items[0].score == pytest.approx(1.0 / (1.0 + 0.1))
    assert results.items[1].score == pytest.approx(1.0 / (1.0 + 0.2))


@pytest.mark.asyncio
async def test_perform_semantic_search_models_faiss_repo_not_ready(
    search_service: SearchService, mock_faiss_model_repo
):
    mock_faiss_model_repo.is_ready.return_value = False
    query = "test query"
    target = cast(SearchTarget, "models")
    results = await search_service.perform_semantic_search(
        query=query, target=target, page=1, page_size=10
    )
    assert isinstance(results, PaginatedHFModelSearchResult)
    assert results.items == []
    assert results.total == 0
    mock_faiss_model_repo.search_similar.assert_not_called()


@pytest.mark.asyncio
async def test_perform_semantic_search_models_pg_fails(
    search_service: SearchService, mock_faiss_model_repo, mock_pg_repo, mock_embedder
):
    mock_faiss_model_repo.search_similar.return_value = [("org/model1", 0.1)]
    mock_pg_repo.get_hf_models_by_ids.side_effect = Exception("PG Error")
    target = cast(SearchTarget, "models")
    results = await search_service.perform_semantic_search(
        query="query", target=target, page=1, page_size=10
    )
    assert isinstance(results, PaginatedHFModelSearchResult)
    assert results.items == []
    assert results.total == 0
    mock_faiss_model_repo.search_similar.assert_called_once()
    mock_pg_repo.get_hf_models_by_ids.assert_awaited_once_with(["org/model1"])
