import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Union, Optional, Literal, cast
from datetime import date
import time
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# Service layer
from aigraphx.services.search_service import SearchService

# API Models
from aigraphx.models.search import (
    SearchResultItem,
    PaginatedPaperSearchResult,
    PaginatedSemanticSearchResult,
    PaginatedHFModelSearchResult,
    HFSearchResultItem,
    SearchFilterModel,
)

# Define the response type for paper search
PapersSearchApiResponse = Union[
    PaginatedPaperSearchResult, PaginatedSemanticSearchResult
]
# Define the response type for model search
ModelsSearchApiResponse = Union[
    PaginatedHFModelSearchResult, PaginatedSemanticSearchResult
]  # Semantic can return models

# Dependency injection
from aigraphx.api.v1 import dependencies as deps
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.vectorization.embedder import TextEmbedder

router = APIRouter()
logger = logging.getLogger(__name__)


# --- Renamed and Updated Papers Endpoint ---
@router.get(
    "/papers/",  # Renamed route
    response_model=PapersSearchApiResponse,
    summary="Search Papers",
    description="Performs semantic, keyword, or hybrid search specifically for papers, with optional filtering and sorting.",
)
async def search_papers(
    # Remove Request parameter
    # request: Request, # Removed Request
    # --- Core Search Parameters ---
    q: str = Query(..., description="The search query string."),
    search_type: Literal["semantic", "keyword", "hybrid"] = Query(
        "semantic", description="Type of search to perform."
    ),
    # --- Pagination Parameters ---
    skip: int = Query(0, description="Number of results to skip for pagination.", ge=0),
    limit: int = Query(
        10, description="Maximum number of results to return.", ge=1, le=100
    ),
    # --- Filtering Parameters (Paper Specific) ---
    date_from: Optional[date] = Query(
        None, description="Filter papers published on or after this date (YYYY-MM-DD)."
    ),
    date_to: Optional[date] = Query(
        None, description="Filter papers published on or before this date (YYYY-MM-DD)."
    ),
    area: Optional[str] = Query(
        None,
        description="Filter papers by research area (exact match, case-sensitive).",
    ),
    # --- Sorting Parameters (Paper Specific) ---
    sort_by: Optional[Literal["score", "published_date"]] = Query(
        None,
        description="Field to sort paper results by. Defaults: 'score' for semantic/hybrid, 'published_date' for keyword.",
    ),
    sort_order: Optional[Literal["asc", "desc"]] = Query(
        "desc", description="Sort order: 'asc' or 'desc'."
    ),
    # --- Hybrid Search Specific Parameters (Added) ---
    rrf_k: int = Query(
        60, description="Reciprocal Rank Fusion k parameter (for hybrid search).", ge=1
    ),
    top_n_semantic: int = Query(
        100,
        description="Initial number of semantic results to fetch for fusion (hybrid search).",
        ge=1,
    ),
    top_n_keyword: int = Query(
        100,
        description="Initial number of keyword results to fetch for fusion (hybrid search).",
        ge=1,
    ),
    # --- Dependency Injection ---
    # Restore direct dependency on SearchService
    search_service: SearchService = Depends(deps.get_search_service),
) -> PapersSearchApiResponse:
    """Endpoint to search specifically for papers."""
    target = cast(Literal["papers", "models", "all"], "papers")  # 显式转换为Literal类型
    log_params = {
        "query": q,
        "search_type": search_type,
        "target": target,
        "skip": skip,
        "limit": limit,
        "date_from": date_from,
        "date_to": date_to,
        "area": area,
        "sort_by": sort_by,
        "sort_order": sort_order,
        # Log RRF params if they are relevant (for hybrid)
        "rrf_k": rrf_k if search_type == "hybrid" else None,
        "top_n_semantic": top_n_semantic if search_type == "hybrid" else None,
        "top_n_keyword": top_n_keyword if search_type == "hybrid" else None,
    }
    logger.info(f"Received paper search request with parameters: {log_params}")

    try:
        if search_type == "semantic":
            # Determine default sort_by for semantic if None
            effective_sort_by = sort_by if sort_by is not None else "score"
            results = await search_service.perform_semantic_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
            )
        elif search_type == "keyword":
            # Determine default sort_by for keyword if None
            effective_sort_by = sort_by if sort_by is not None else "published_date"
            results = await search_service.perform_keyword_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                date_from=date_from,
                date_to=date_to,
                area=area,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
            )
        elif search_type == "hybrid":
            # Create filter object
            search_filters = SearchFilterModel(
                published_after=date_from,
                published_before=date_to,
                filter_area=area,
                sort_by=sort_by,
                sort_order=sort_order or "desc",  # Ensure sort_order is not None
            )

            # FIXED: Pass filters object, remove individual args and RRF params
            results = await search_service.perform_hybrid_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                filters=search_filters,
            )
        # else: # Should be unreachable due to Literal validation
        #     raise HTTPException(status_code=400, detail="Invalid search_type specified.")

        logger.info(
            f"Paper search successful, returning {len(results.items)} items out of {results.total} total."
        )
        # 确保返回类型兼容
        if isinstance(results, PaginatedSemanticSearchResult):
            # 已经是PaginatedSemanticSearchResult类型，可以直接返回
            return results
        elif isinstance(results, PaginatedHFModelSearchResult):
            # 不应该出现这种情况，但如果发生了，需要转换为合法的返回类型
            paper_items = [
                SearchResultItem(**item.model_dump()) for item in results.items
            ]
            return PaginatedPaperSearchResult(
                items=paper_items,
                total=results.total,
                skip=results.skip,
                limit=results.limit,
            )
        # 否则是PaginatedPaperSearchResult类型，直接返回
        return results

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"An error occurred during paper search: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during paper search."
        )


# --- New Models Endpoint ---
@router.get(
    "/models/",  # New route for models
    response_model=ModelsSearchApiResponse,
    summary="Search Models",
    description="Performs semantic or keyword search specifically for models.",
)
async def search_models(
    # Remove Request parameter
    # request: Request, # Removed Request
    # --- Core Search Parameters ---
    q: str = Query(..., description="The search query string."),
    # Note: search_type only allows semantic or keyword for models
    search_type: Literal["semantic", "keyword"] = Query(
        "semantic", description="Type of search to perform."
    ),
    # --- Pagination Parameters ---
    skip: int = Query(0, description="Number of results to skip for pagination.", ge=0),
    limit: int = Query(
        10, description="Maximum number of results to return.", ge=1, le=100
    ),
    # --- Filtering Parameters (Model Specific - Example: Add later if needed) ---
    # filter_pipeline_tag: Optional[str] = Query(None, description="Filter models by pipeline tag."),
    # --- Sorting Parameters (Model Specific - Example) ---
    sort_by: Optional[Literal["score", "likes", "downloads", "last_modified"]] = Query(
        None,
        description="Field to sort model results by. Defaults: 'score' for semantic, 'likes' for keyword.",
    ),
    sort_order: Optional[Literal["asc", "desc"]] = Query(
        "desc", description="Sort order: 'asc' or 'desc'."
    ),
    # --- Dependency Injection ---
    # Restore direct dependency on SearchService
    search_service: SearchService = Depends(deps.get_search_service),
) -> ModelsSearchApiResponse:
    """Endpoint to search specifically for models."""
    target = cast(Literal["papers", "models", "all"], "models")  # 显式转换为Literal类型
    log_params = {
        "query": q,
        "search_type": search_type,
        "target": target,
        "skip": skip,
        "limit": limit,
        # Add model-specific filters/sorts here if defined
        # "filter_pipeline_tag": filter_pipeline_tag,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    logger.info(f"Received model search request with parameters: {log_params}")

    try:
        if search_type == "semantic":
            # Determine default sort_by for semantic if None
            effective_sort_by = sort_by if sort_by is not None else "score"
            results = await search_service.perform_semantic_search(
                query=q,
                target=target,
                page=skip // limit + 1,  # Convert skip/limit to page
                page_size=limit,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",  # 确保sort_order非None
            )
        elif search_type == "keyword":
            # Determine default sort_by for keyword if None
            effective_sort_by = sort_by if sort_by is not None else "likes"
            results = await search_service.perform_keyword_search(
                query=q,
                target=target,
                page=skip // limit + 1,  # Convert skip/limit to page
                page_size=limit,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",  # 确保sort_order非None
            )
        # else: # Should be unreachable due to Literal validation
        #     raise HTTPException(status_code=400, detail="Invalid search_type specified for models.")

        logger.info(
            f"Model search successful, returning {len(results.items)} items out of {results.total} total."
        )
        # Convert to response model
        items = [HFSearchResultItem(**item.model_dump()) for item in results.items]
        response = PaginatedHFModelSearchResult(  # 不使用类型标注，避免类型错误
            items=items,
            total=results.total,
            skip=results.skip,
            limit=results.limit,
        )
        return response

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"An error occurred during model search: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during model search."
        )
