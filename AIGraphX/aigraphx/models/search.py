from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Union, Any, TypeVar, Generic
from datetime import date, datetime, timezone

# Define supported search types
SearchType = Literal["semantic", "keyword", "hybrid"]


class SearchResultItem(BaseModel):
    """Represents a single search result item."""

    paper_id: Optional[int] = None
    pwc_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    # Score is now Optional to handle cases like keyword search results
    score: Optional[float] = Field(
        default=None,
        description="Relevance score (0-1 range for semantic/hybrid, higher is better; None for keyword-only).",
    )
    pdf_url: Optional[str] = None
    published_date: Optional[date] = None
    authors: Optional[List[str]] = None  # Assuming decoded list
    area: Optional[str] = None  # Add area field


class HFSearchResultItem(BaseModel):
    """Represents a Hugging Face model in search results."""

    model_id: str = Field(..., description="The unique Hugging Face model ID.")
    author: Optional[str] = Field(None, description="The author or organization.")
    pipeline_tag: Optional[str] = Field(
        None, description="The primary task pipeline tag."
    )
    library_name: Optional[str] = Field(
        None, description="The library associated with the model (e.g., transformers)."
    )
    tags: Optional[List[str]] = Field(
        None, description="List of tags associated with the model."
    )
    likes: Optional[int] = Field(None, description="Number of likes.")
    downloads: Optional[int] = Field(None, description="Number of downloads.")
    last_modified: Optional[datetime] = Field(
        None, description="Last modification timestamp."
    )
    score: float = Field(
        ..., description="Relevance score from the search (0.0 to 1.0)."
    )

    @field_validator("last_modified", mode="before")
    @classmethod
    def parse_last_modified(cls, v: Any) -> Optional[datetime]:
        """Parse and validate the last_modified field, returning datetime object."""
        if isinstance(v, str):
            try:
                # Handle 'Z' suffix for UTC
                dt_parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
                # Ensure timezone-aware (assuming UTC if no offset)
                if dt_parsed.tzinfo is None:
                    return dt_parsed.replace(tzinfo=timezone.utc)
                else:
                    return dt_parsed
            except ValueError as e:
                # Raise ValueError for Pydantic to catch as validation error
                raise ValueError(
                    f"Invalid datetime format for last_modified: '{v}'"
                ) from e
        elif isinstance(v, datetime):
            # Ensure timezone-aware if already datetime
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            else:
                return v
        elif v is None:
            return None  # Return None if input is None
        else:
            # Raise ValueError instead of TypeError for invalid types
            raise ValueError(
                "last_modified must be a valid ISO 8601 string or datetime object"
            )


class PaginatedPaperSearchResult(BaseModel):
    """Response model for paginated paper search results."""

    items: List[SearchResultItem] = Field(
        ..., description="List of paper search results on the current page."
    )
    total: int = Field(..., description="Total number of papers matching the query.")
    skip: int = Field(..., description="Number of items skipped (offset).")
    limit: int = Field(..., description="Maximum number of items per page.")


# Define a Union type for different search result items
AnySearchResultItem = Union[SearchResultItem, HFSearchResultItem]


class PaginatedSemanticSearchResult(BaseModel):
    """Response model for paginated semantic search results (papers or models)."""

    items: List[AnySearchResultItem] = Field(
        ...,
        description="List of semantic search results (papers or models) on the current page.",
    )
    total: int = Field(
        ..., description="Total number of candidate results found before pagination."
    )
    skip: int = Field(..., description="Number of items skipped (offset).")
    limit: int = Field(..., description="Maximum number of items per page.")


class PaginatedHFModelSearchResult(BaseModel):
    """Response model for paginated Hugging Face model search results."""

    items: List[HFSearchResultItem] = Field(
        ..., description="List of HF model search results on the current page."
    )
    total: int = Field(..., description="Total number of HF models matching the query.")
    skip: int = Field(..., description="Number of items skipped (offset).")
    limit: int = Field(..., description="Maximum number of items per page.")


# 添加泛型分页模型，用于混合搜索结果
T = TypeVar("T")


class PaginatedModel(BaseModel, Generic[T]):
    """通用分页模型，可以用于任何类型的搜索结果。"""

    items: List[T] = Field(..., description="当前页的搜索结果列表。")
    total: int = Field(..., description="与查询匹配的项目总数。")
    skip: int = Field(..., description="跳过的项目数量（偏移量）。")
    limit: int = Field(..., description="每页最大项目数。")


class SearchFilterModel(BaseModel):
    """搜索过滤器模型，用于定义混合搜索的过滤条件。"""

    published_after: Optional[date] = Field(
        None, description="筛选在此日期之后发布的论文（包含该日期）。"
    )
    published_before: Optional[date] = Field(
        None, description="筛选在此日期之前发布的论文（包含该日期）。"
    )
    filter_area: Optional[str] = Field(
        None, description="按研究领域筛选论文（例如 CV、NLP 等）。"
    )
    sort_by: Optional[str] = Field(
        None, description="结果排序依据（score、published_date、title 等）。"
    )
    sort_order: Optional[Literal["asc", "desc"]] = Field(
        "desc", description="排序顺序，asc（升序）或 desc（降序）。"
    )
