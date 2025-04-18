import pytest
from pydantic import ValidationError
from typing import List, Optional, cast, Any, Dict, Union
from datetime import date, datetime, timezone

# Import models to test
from aigraphx.models.search import (
    SearchResultItem,
    HFSearchResultItem,
    SearchType,
    PaginatedPaperSearchResult,
)


# --- Tests for SearchResultItem Model ---
def test_searchresultitem_creation_minimal() -> None:
    """Test minimal SearchResultItem creation."""
    item = SearchResultItem(pwc_id="paper1", score=0.85)
    assert item.pwc_id == "paper1"
    assert item.score == 0.85
    assert item.title is None
    assert item.summary is None
    assert item.published_date is None
    assert item.authors is None


def test_searchresultitem_creation_full() -> None:
    """Test full SearchResultItem creation."""
    published_date = date(2024, 1, 15)
    item = SearchResultItem(
        pwc_id="paper2",
        title="Result Title",
        summary="Result abstract.",
        score=0.9,
        published_date=published_date,
        authors=["Author X"],
    )
    assert item.pwc_id == "paper2"
    assert item.title == "Result Title"
    assert item.summary == "Result abstract."
    assert item.score == 0.9
    assert item.published_date == published_date
    assert item.authors == ["Author X"]


def test_searchresultitem_missing_required() -> None:
    """Test SearchResultItem creation fails if required fields are missing."""
    # Only pwc_id is truly required now
    with pytest.raises(ValidationError) as excinfo:
        # Try creating without pwc_id (score is optional), remove invalid paper_id
        SearchResultItem(title="T", score=0.5)  # type: ignore[call-arg] # 故意不提供pwc_id参数以测试验证
    # Check the errors list for the missing field
    errors = excinfo.value.errors()
    assert any(err["type"] == "missing" and err["loc"] == ("pwc_id",) for err in errors)


# --- Tests for HFSearchResultItem Model ---
def test_hfsearchresultitem_creation_minimal() -> None:
    """Test minimal HFSearchResultItem creation."""
    # Explicitly pass None for optional fields to satisfy mypy if needed
    item = HFSearchResultItem(
        model_id="org/model1",
        score=0.7,
        author=None,
        pipeline_tag=None,
        library_name=None,
        tags=None,
        likes=None,
        downloads=None,
        last_modified=None,
    )
    assert item.model_id == "org/model1"
    assert item.score == 0.7
    assert item.author is None
    assert item.tags is None
    # Add assertions for other None fields if necessary
    assert item.pipeline_tag is None
    assert item.library_name is None
    assert item.likes is None
    assert item.downloads is None
    assert item.last_modified is None


def test_hfsearchresultitem_creation_full() -> None:
    """Test full HFSearchResultItem creation."""
    last_modified_str = "2024-03-10T10:00:00Z"
    # Explicitly parse the string to datetime before passing
    parsed_dt = HFSearchResultItem.parse_last_modified(last_modified_str)
    assert parsed_dt is not None  # Ensure parsing was successful

    # Replace **data with explicit arguments
    item = HFSearchResultItem(
        model_id="org/model2",
        author="Org",
        pipeline_tag="text-generation",
        library_name="transformers",
        tags=["llm", "chat"],
        likes=100,
        downloads=5000,
        last_modified=parsed_dt,  # Pass the datetime object
        score=0.95,
    )
    assert item.model_id == "org/model2"
    assert item.author == "Org"
    assert item.pipeline_tag == "text-generation"
    assert item.tags == ["llm", "chat"]
    assert item.likes == 100
    assert item.score == 0.95
    assert isinstance(item.last_modified, datetime)
    assert item.last_modified.year == 2024
    assert item.last_modified.tzinfo == timezone.utc


def test_hfsearchresultitem_missing_required() -> None:
    """Test HFSearchResultItem creation fails if required fields are missing."""
    # Test missing model_id (score is provided)
    with pytest.raises(ValidationError) as excinfo_model:
        HFSearchResultItem(  # type: ignore[call-arg] # 故意不提供model_id参数以测试验证
            # model_id参数被故意删除
            score=0.6,  # Provide score
            author=None,
            pipeline_tag=None,
            library_name=None,
            tags=None,
            likes=None,
            downloads=None,
            last_modified=None,  # Provide None for optionals
        )
    errors_model = excinfo_model.value.errors()
    assert any(
        err["type"] == "missing" and err["loc"] == ("model_id",) for err in errors_model
    )

    # Test missing score (model_id is provided)
    with pytest.raises(ValidationError) as excinfo_score:
        HFSearchResultItem(  # type: ignore[call-arg] # 故意不提供score参数以测试验证
            model_id="org/model3",  # Provide model_id
            # score参数被故意删除
            author=None,
            pipeline_tag=None,
            library_name=None,
            tags=None,
            likes=None,
            downloads=None,
            last_modified=None,  # Provide None for optionals
        )
    errors_score = excinfo_score.value.errors()
    assert any(
        err["type"] == "missing" and err["loc"] == ("score",) for err in errors_score
    )


# --- Tests for PaginatedPaperSearchResult Model ---
def test_paginated_paper_search_result_creation_success() -> None:
    """Test successful creation of PaginatedPaperSearchResult."""
    item1 = SearchResultItem(pwc_id="p1", score=0.9)
    item2 = SearchResultItem(pwc_id="p2", score=0.8)
    paginated_result = PaginatedPaperSearchResult(
        items=[item1, item2], total=10, skip=0, limit=2
    )
    assert len(paginated_result.items) == 2
    assert paginated_result.total == 10
    assert paginated_result.skip == 0
    assert paginated_result.limit == 2
    assert paginated_result.items[0] == item1


def test_paginated_paper_search_result_creation_empty() -> None:
    """Test PaginatedPaperSearchResult creation with empty items."""
    paginated_result = PaginatedPaperSearchResult(items=[], total=0, skip=10, limit=5)
    assert paginated_result.items == []
    assert paginated_result.total == 0
    assert paginated_result.skip == 10
    assert paginated_result.limit == 5


def test_paginated_paper_search_result_missing_required() -> None:
    """Test PaginatedPaperSearchResult creation fails if required fields are missing."""
    item1 = SearchResultItem(pwc_id="p1", score=0.9)
    with pytest.raises(ValidationError, match="items"):
        PaginatedPaperSearchResult(items=cast(Any, None), total=1, skip=0, limit=1)
    with pytest.raises(ValidationError, match="total"):
        PaginatedPaperSearchResult(
            items=[item1], total=cast(Any, None), skip=0, limit=1
        )
    with pytest.raises(ValidationError, match="skip"):
        PaginatedPaperSearchResult(
            items=[item1], total=1, skip=cast(Any, None), limit=1
        )
    with pytest.raises(ValidationError, match="limit"):
        PaginatedPaperSearchResult(
            items=[item1], total=1, skip=0, limit=cast(Any, None)
        )


# Note: The SearchResponse model currently defines results as List[SearchResultItem].
# The API endpoint uses List[Union[SearchResultItem, HFSearchResultItem]].
# If the API is intended to return mixed types, the SearchResponse model definition
# might need to be updated accordingly, e.g., using a Union or a base class.
# The current tests reflect the *model's* definition.
