# tests/services/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from typing import List, Tuple, Dict, Optional, cast, Any, Literal
from datetime import date

# Import necessary classes for fixtures
from aigraphx.services.search_service import SearchService
from aigraphx.vectorization.embedder import TextEmbedder
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# --- Mock Data Definitions ---
MOCK_PAPER_1_DETAIL_DICT = {
    "paper_id": 1,
    "pwc_id": "pwc-1",
    "title": "Paper 1",
    "summary": "Summary 1",
    "authors": ["Auth 1"],
    "published_date": date(2023, 1, 15),
    "pdf_url": "url1",
    "area": "CV",
}
MOCK_PAPER_2_DETAIL_DICT = {
    "paper_id": 2,
    "pwc_id": "pwc-2",
    "title": "Paper 2",
    "summary": "Summary 2",
    "authors": ["Auth 2"],
    "published_date": date(2023, 2, 10),
    "pdf_url": "url2",
    "area": "NLP",
}
MOCK_PAPER_3_DETAIL_DICT = {
    "paper_id": 3,
    "pwc_id": "pwc-3",
    "title": "Paper 3",
    "summary": "Summary 3",
    "authors": ["Auth 3"],
    "published_date": date(2023, 1, 5),
    "pdf_url": "url3",
    "area": "CV",
}
MOCK_PAPER_KEY_1_DETAIL_DICT = {
    "paper_id": 101,
    "pwc_id": "pwc-key-1",
    "title": "Keyword Paper 1",
    "summary": "Keyword Summary 1",
    "authors": ["Key Auth 1"],
    "published_date": date(2023, 2, 1),
    "pdf_url": "url_key1",
    "area": "NLP",
}
MOCK_PAPER_KEY_2_DETAIL_DICT = {
    "paper_id": 102,
    "pwc_id": "pwc-key-2",
    "title": "Keyword Paper 2",
    "summary": "Keyword Summary 2",
    "authors": ["Key Auth 2"],
    "published_date": date(2023, 2, 2),
    "pdf_url": "url_key2",
    "area": "CV",
}
MOCK_PAPER_4_DETAIL_DICT = {
    "paper_id": 4,
    "pwc_id": "pwc-test-4",
    "title": "Paper Z",
    "summary": "Summary Z",
    "authors": ["Auth Z"],
    "published_date": date(2023, 7, 7),
    "pdf_url": "http://example.com/4",
    "area": "CV",
}
MOCK_PAPER_5_DETAIL_DICT = {
    "paper_id": 5,
    "pwc_id": "pwc-test-5",
    "title": "Paper Y",
    "summary": "Summary Y",
    "authors": ["Auth Y"],
    "published_date": date(2023, 8, 8),
    "pdf_url": "http://example.com/5",
    "area": "CV",
}

# --- Fixtures ---


@pytest.fixture
def mock_embedder() -> MagicMock:
    mock = MagicMock(spec=TextEmbedder)
    mock.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
    mock.embed_batch = MagicMock(
        return_value=np.array([[0.1] * 384] * 10, dtype=np.float32)
    )
    return mock


@pytest.fixture
def mock_faiss_paper_repo() -> MagicMock:
    mock = MagicMock(spec=FaissRepository)
    mock.search_similar.return_value = [(1, 0.1), (3, 0.3), (2, 0.5)]
    mock.is_ready = MagicMock(return_value=True)
    mock.id_type = "int"
    return mock


@pytest.fixture
def mock_faiss_model_repo() -> MagicMock:
    mock = MagicMock(spec=FaissRepository)
    mock.search_similar.return_value = [("org/model-a", 0.2), ("another/model-b", 0.4)]
    mock.is_ready = MagicMock(return_value=True)
    mock.id_type = "str"
    return mock


@pytest.fixture
def mock_neo4j_repo() -> MagicMock:
    mock = AsyncMock(spec=Neo4jRepository)
    # Remove lines setting return values for non-existent methods
    # mock.get_related_papers_for_model.return_value = [] # Removed
    # mock.get_related_models_for_paper.return_value = [] # Removed
    # Add return values for methods that *might* be called implicitly or by other tests
    # For safety, let's mock methods we see are defined, if needed.
    # Example: if search service needed these, we'd uncomment and configure them.
    # mock.search_nodes.return_value = []
    # mock.get_neighbors.return_value = []
    # mock.get_related_nodes.return_value = []
    return mock


@pytest.fixture
def mock_pg_repo() -> MagicMock:
    mock = AsyncMock(spec=PostgresRepository)
    mock.paper_details_map = {
        1: MOCK_PAPER_1_DETAIL_DICT,
        2: MOCK_PAPER_2_DETAIL_DICT,
        3: MOCK_PAPER_3_DETAIL_DICT,
        4: MOCK_PAPER_4_DETAIL_DICT,
        5: MOCK_PAPER_5_DETAIL_DICT,
        101: MOCK_PAPER_KEY_1_DETAIL_DICT,
        102: MOCK_PAPER_KEY_2_DETAIL_DICT,
        999: {"paper_id": 999, "pwc_id": "pwc-err-1", "title": "Error Paper"},
    }

    async def mock_get_details_by_ids(
        paper_ids: List[int], scores: Optional[Dict[int, float]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for pid in paper_ids:
            if pid in mock.paper_details_map:
                detail = mock.paper_details_map[pid].copy()
                if scores and pid in scores:
                    detail["score"] = scores[pid]
                results.append(detail)
        return results

    mock.get_papers_details_by_ids.side_effect = mock_get_details_by_ids

    async def mock_search_papers_keyword_return_dicts(
        query: str,
        skip: int = 0,
        limit: int = 10,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        published_after: Optional[date] = None,
        published_before: Optional[date] = None,
        area: Optional[str] = None,
        filter_area: Optional[str] = None,
        sort_by: Optional[
            Literal["published_date", "title", "paper_id"]
        ] = "published_date",
        sort_order: Optional[Literal["asc", "desc"]] = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:
        date_from = date_from or published_after
        date_to = date_to or published_before
        area = area or filter_area
        
        # Simulate DB fetching details directly
        # Use details map for consistency
        all_paper_details = list(mock.paper_details_map.values())

        # --- Basic Filtering Simulation (Example) ---
        filtered_details = all_paper_details
        if date_from:
            filtered_details = [
                p
                for p in filtered_details
                if p.get("published_date") and p.get("published_date") >= date_from
            ]
        if date_to:
            filtered_details = [
                p
                for p in filtered_details
                if p.get("published_date") and p.get("published_date") <= date_to
            ]
        if area:
            filtered_details = [p for p in filtered_details if p.get("area") == area]
        # Add simple query matching for simulation
        if query:
            filtered_details = [
                p
                for p in filtered_details
                if query.lower() in p.get("title", "").lower()
                or query.lower() in p.get("summary", "").lower()
            ]

        # --- Sorting Simulation (Example) ---
        reverse = sort_order == "desc"
        sort_key = (
            sort_by
            if sort_by in ["published_date", "title", "paper_id"]
            else "published_date"
        )

        def get_key(paper_dict: Dict[str, Any]) -> Any:
            val = paper_dict.get(sort_key)
            # Handle None for sorting consistency
            if isinstance(val, date):
                return val
            if isinstance(val, str):
                return val.lower()
            if isinstance(val, int):
                return val
            # Fallback for None or unexpected types
            if sort_key == "published_date":
                return date.min
            if sort_key == "title":
                return ""
            if sort_key == "paper_id":
                return -1
            return None  # Should not happen with validated keys

        try:
            # Sort based on key, use paper_id as tie-breaker
            filtered_details.sort(
                key=lambda p: (get_key(p), p.get("paper_id", -1)), reverse=reverse
            )
        except TypeError as e:
            print(f"Sorting error in mock: {e}")  # Log error if sorting fails

        # --- Pagination ---
        total = len(filtered_details)
        paginated = filtered_details[skip : skip + limit]

        # Return list of dicts and total count
        return paginated, total

    # Set the default side_effect for search_papers_by_keyword
    mock.search_papers_by_keyword.side_effect = mock_search_papers_keyword_return_dicts

    async def mock_get_hf_details(model_ids: List[str]) -> List[Dict[str, Any]]:
        model_details_map = {
            "org/model-kw1": {
                "model_id": "org/model-kw1",
                "author": "Org",
                "pipeline_tag": "text-generation",
            },
            "org/model-kw3": {
                "model_id": "org/model-kw3",
                "author": "Org",
                "pipeline_tag": "text-generation",
            },
            "another/model-kw2": {
                "model_id": "another/model-kw2",
                "author": "Another",
                "pipeline_tag": "image-classification",
            },
            "org/model1": {
                "model_id": "org/model1",
                "author": "org",
                "likes": 100,
                "last_modified": "2023-01-01",
                "tags": ["tag1"],
                "pipeline_tag": "text",
                "downloads": 1000,
                "library_name": "transformers",
            },
            "user/model3": {
                "model_id": "user/model3",
                "author": "user",
                "likes": 50,
                "last_modified": "2023-01-03",
                "tags": ["tag1", "tag2"],
                "pipeline_tag": "image",
                "downloads": 500,
                "library_name": "diffusers",
            },
            "org/model2": {
                "model_id": "org/model2",
                "author": "org",
                "likes": 200,
                "last_modified": "2023-01-02",
                "tags": ["tag3"],
                "pipeline_tag": "text",
                "downloads": 2000,
                "library_name": "transformers",
            },
        }
        return cast(
            List[Dict[str, Any]],
            [model_details_map[mid] for mid in model_ids if mid in model_details_map],
        )

    mock.get_hf_models_by_ids.side_effect = mock_get_hf_details
    mock.search_models_by_keyword.return_value = ([], 0)
    return mock


@pytest.fixture
def search_service(
    mock_embedder: MagicMock,
    mock_faiss_paper_repo: MagicMock,
    mock_faiss_model_repo: MagicMock,
    mock_pg_repo: MagicMock,
    mock_neo4j_repo: MagicMock,
) -> SearchService:
    service = SearchService(
        embedder=mock_embedder,
        faiss_repo_papers=mock_faiss_paper_repo,
        faiss_repo_models=mock_faiss_model_repo,
        pg_repo=mock_pg_repo,
        neo4j_repo=mock_neo4j_repo,
    )
    return service
