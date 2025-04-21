import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient  # 添加 AsyncClient 导入
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np  # Import numpy for mock embedding
from fastapi import FastAPI
from typing import List, Dict, Optional, Union, Any  # Added Any
from datetime import date, datetime, timezone  # Import date, datetime, and timezone

# Import original dependency functions for override keys
from aigraphx.api.v1 import dependencies as deps

# Import Service classes for spec in Mock
from aigraphx.services.search_service import SearchService, SearchTarget
from aigraphx.api.v1.dependencies import (
    get_search_service,
)  # Import the dependency getter
from aigraphx.models.search import (
    SearchResultItem,
    HFSearchResultItem,
    PaginatedPaperSearchResult,
    PaginatedSemanticSearchResult,
    PaginatedHFModelSearchResult,  # Import the new model
    SearchFilterModel,
)

# Remove direct import of app
# from aigraphx.main import app

# Remove direct TestClient instantiation
# client = TestClient(app)

# Mark all async tests in this module to use the default function-scoped event loop
pytestmark = pytest.mark.asyncio

# Helper function to generate search query params

# --- Mock Data (Papers Only) ---
MOCK_PAPER_RESULT = SearchResultItem(
    paper_id=1,
    pwc_id="pwc-1",
    title="Test Paper",
    summary="Test abstract",
    score=0.9,  # Score might be set by service mock
    pdf_url="http://example.com/abs",
    published_date=date(2023, 1, 1),
    authors=["Auth1"],
    area="CV",  # Use str date for JSON consistency
)

# --- Mock Data (Models) ---
# Parse datetime string before creating the model instance
now_dt_str = datetime.now(timezone.utc).isoformat()
parsed_now_dt = HFSearchResultItem.parse_last_modified(now_dt_str)
assert parsed_now_dt is not None

MOCK_MODEL_RESULT = HFSearchResultItem(
    model_id="org/test-model",
    author="Org",
    pipeline_tag="text-generation",
    library_name="transformers",
    tags=["test", "model"],
    likes=100,
    downloads=1000,
    last_modified=parsed_now_dt,  # Pass the datetime object
    score=0.8,
)

# --- Test Cases (Papers Search Only) ---


@pytest.mark.asyncio
async def test_search_semantic_papers_success(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test successful semantic search for papers, returning paginated results."""
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedSemanticSearchResult(
        items=[MOCK_PAPER_RESULT],  # 使用类型兼容的 SearchResultItem
        total=25,
        skip=0,
        limit=5,
    )
    mock_service.perform_semantic_search.return_value = mock_paginated_result

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        response = await client.get(
            "/api/v1/search/papers/?q=test&search_type=semantic&skip=0&limit=5"
        )  # 添加 await
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # Calculate expected page based on skip/limit
        expected_skip = 0
        expected_limit = 5
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_semantic_search.assert_awaited_once_with(
            query="test",
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # Service still needs target
            sort_by="score",  # API defaults to 'score' for semantic if None
            sort_order="desc",  # API default is desc
        )
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_keyword_papers_success(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test successful keyword search for papers, returning paginated results."""
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedPaperSearchResult(
        items=[MOCK_PAPER_RESULT], total=50, skip=5, limit=15
    )
    mock_service.perform_keyword_search.return_value = mock_paginated_result

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        response = await client.get(
            "/api/v1/search/papers/?q=graph&search_type=keyword&skip=5&limit=15"
        )  # 添加 await
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # Calculate expected page based on skip/limit
        expected_skip = 5
        expected_limit = 15
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_keyword_search.assert_awaited_once_with(
            query="graph",
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # Service still needs target
            date_from=None,
            date_to=None,
            area=None,
            sort_by="published_date",  # API defaults to 'published_date' for keyword if None
            sort_order="desc",  # API default is desc
        )
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_papers_with_all_filters_sort(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test searching papers with specific filters and sorting applied."""
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedPaperSearchResult(
        items=[MOCK_PAPER_RESULT], total=1, skip=0, limit=5
    )
    mock_service.perform_keyword_search.return_value = mock_paginated_result

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    test_query = "filtered search"
    test_after = "2023-01-01"
    test_before = "2023-12-31"
    test_area = "CV"
    test_sort_by = "published_date"
    test_sort_order = "asc"
    test_skip = 0
    test_limit = 5
    test_search_type = "keyword"

    try:
        response = await client.get(  # 添加 await
            f"/api/v1/search/papers/?q={test_query}"
            f"&search_type={test_search_type}"
            f"&skip={test_skip}"
            f"&limit={test_limit}"
            f"&date_from={test_after}"
            f"&date_to={test_before}"
            f"&area={test_area}"
            f"&sort_by={test_sort_by}"
            f"&sort_order={test_sort_order}"
        )
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # Calculate expected page based on skip/limit
        expected_skip = test_skip
        expected_limit = test_limit
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_keyword_search.assert_awaited_once_with(
            query=test_query,
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # Service still needs target
            date_from=date.fromisoformat(test_after),
            date_to=date.fromisoformat(test_before),
            area=test_area,
            sort_by=test_sort_by,
            sort_order=test_sort_order,
        )
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_papers_invalid_search_type(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test paper search with invalid search type."""
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service
    try:
        response = await client.get(
            "/api/v1/search/papers/?q=any&search_type=invalid"
        )  # 添加 await
        assert response.status_code == 422  # FastAPI validation error
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_hybrid_papers_success(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test successful hybrid search for papers."""
    mock_service = AsyncMock(spec=SearchService)
    mock_hybrid_result = PaginatedPaperSearchResult(
        items=[MOCK_PAPER_RESULT], total=15, skip=0, limit=10
    )
    mock_service.perform_hybrid_search.return_value = mock_hybrid_result

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        response = await client.get(
            "/api/v1/search/papers/?q=hybrid&search_type=hybrid&skip=0&limit=10"
        )  # 添加 await
        assert response.status_code == 200
        assert response.json() == mock_hybrid_result.model_dump(mode="json")

        # Calculate expected page based on skip/limit
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1

        # Create the expected filters object that the service should receive
        expected_filters = SearchFilterModel(
            published_after=None,
            published_before=None,
            filter_area=None,
            sort_by=None,  # API passes None if not specified
            sort_order="desc",  # API default
            pipeline_tag=None, # Add missing argument
        )

        mock_service.perform_hybrid_search.assert_awaited_once_with(
            query="hybrid",
            page=expected_page,
            page_size=expected_limit,
            target="papers",
            # Check for the filters object
            filters=expected_filters,
        )
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_type, target",
    [
        ("semantic", "papers"),
        ("keyword", "papers"),
        ("hybrid", "papers"),
        ("semantic", "models"),
        ("keyword", "models"),  # Added model keyword error case
    ],
)
async def test_search_service_error(
    client: AsyncClient, test_app: FastAPI, search_type: str, target: str
) -> None:  # 修改类型注解
    """Test API returns 500 when the corresponding service method fails."""
    mock_service = AsyncMock(spec=SearchService)

    # Mock the specific method expected to be called to raise an error
    if search_type == "semantic":
        mock_service.perform_semantic_search.side_effect = Exception("Service error")
    elif search_type == "keyword" and target == "papers":
        mock_service.perform_keyword_search.side_effect = Exception("Service error")
    elif search_type == "hybrid":
        mock_service.perform_hybrid_search.side_effect = Exception("Service error")
    elif search_type == "keyword" and target == "models":
        mock_service.perform_keyword_search.side_effect = Exception("Service error")
    else:  # Skip combinations not expected to call these methods
        pytest.skip(
            f"Skipping service error test for {search_type} with target {target}"
        )

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        base_url = f"/api/v1/search/{target}/"
        response = await client.get(
            f"{base_url}?q=error&search_type={search_type}"
        )  # 添加 await
        assert response.status_code == 500
    finally:
        test_app.dependency_overrides = original_overrides  # Clean up


# --- Test Cases (Models Search) ---


@pytest.mark.asyncio
async def test_search_semantic_models_success(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test successful semantic search for models."""
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedSemanticSearchResult(
        items=[MOCK_MODEL_RESULT],  # 使用类型兼容的 HFSearchResultItem
        total=10,
        skip=0,
        limit=5,
    )
    mock_service.perform_semantic_search.return_value = mock_paginated_result

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        response = await client.get(
            "/api/v1/search/models/?q=test&search_type=semantic&skip=0&limit=5"
        )  # 添加 await
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # Calculate expected page based on skip/limit
        expected_skip = 0
        expected_limit = 5
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_semantic_search.assert_awaited_once_with(
            query="test",
            page=expected_page,
            page_size=expected_limit,
            target="models",
            sort_by="score",  # Add default sort for models if service expects it
            sort_order="desc",
        )
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_keyword_models_success(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test successful keyword search for models."""
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedHFModelSearchResult(
        items=[MOCK_MODEL_RESULT],  # Type needs to match target
        total=30,
        skip=0,
        limit=10,
    )
    mock_service.perform_keyword_search.return_value = mock_paginated_result

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        response = await client.get(
            "/api/v1/search/models/?q=bert&search_type=keyword&skip=0&limit=10"
        )  # 添加 await
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # Calculate expected page based on skip/limit
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_keyword_search.assert_awaited_once_with(
            query="bert",
            page=expected_page,
            page_size=expected_limit,
            target="models",
            sort_by="likes",  # Add default sort for models if service expects it
            sort_order="desc",
        )
    finally:
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_models_invalid_search_type(
    client: AsyncClient, test_app: FastAPI
) -> None:  # 修改类型注解
    """Test model search with invalid search type."""
    mock_service = AsyncMock(spec=SearchService)
    # NOTE: Don't need to set up mock return values since FastAPI will reject the request

    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        response = await client.get(
            "/api/v1/search/models/?q=any&search_type=hybrid"
        )  # 添加 await
        assert response.status_code == 422  # FastAPI validation error
        # API should prevent hybrid because it's not valid for models in the ModelsSearchType enum
    finally:
        test_app.dependency_overrides = original_overrides


# --- Test Cases (Parameter Validation - Papers) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "param, value",
    [
        ("skip", -1),
        ("limit", -1),
        ("date_from", "invalid-date"),
        ("date_to", "2023/01/01"),  # Example invalid format
        ("sort_order", "descending"),  # Invalid enum value
        ("sort_by", "title"),  # Assuming 'title' is not valid for papers sort_by
        ("search_type", "fuzzy"),  # Invalid search type enum
    ],
)
async def test_search_papers_invalid_query_params(
    client: AsyncClient, test_app: FastAPI, param: str, value: Any
) -> None:  # Add mock_faiss_ready param type hint
    """Test paper search with invalid query parameters."""
    # --- Setup Mock Service Override ---
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service
    # --- End Setup ---

    try:
        url = f"/api/v1/search/papers/?q=test&{param}={value}"
        response = await client.get(url)
        assert response.status_code == 422  # Expect FastAPI validation error
    finally:
        # --- Teardown Mock Service Override ---
        test_app.dependency_overrides = original_overrides
        # --- End Teardown ---


# --- Test Cases (Parameter Validation - Models) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "param, value",
    [
        ("skip", -5),
        ("limit", 0),  # limit must be >= 1
        ("search_type", "hybrid"),  # Invalid enum value for models
    ],
)
async def test_search_models_invalid_query_params(
    client: AsyncClient, test_app: FastAPI, param: str, value: Any
) -> None:  # Add mock_faiss_ready param type hint
    """Test model search with invalid query parameters."""
    # --- Setup Mock Service Override ---
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service
    # --- End Setup ---

    try:
        url = f"/api/v1/search/models/?q=test&{param}={value}"
        response = await client.get(url)
        assert response.status_code == 422  # Expect FastAPI validation error
    finally:
        # --- Teardown Mock Service Override ---
        test_app.dependency_overrides = original_overrides
        # --- End Teardown ---


# ... existing error test ...
