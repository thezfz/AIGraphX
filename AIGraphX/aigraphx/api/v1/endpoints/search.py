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
    area: Optional[List[str]] = Query(
        None,
        description="Filter papers by research area(s). 支持多选过滤。",
    ),
    # 新增作者过滤
    filter_authors: Optional[List[str]] = Query(
        None,
        description="Filter papers by author name(s). Supports multiple, uses partial matching (OR logic).",
    ),
    # --- Sorting Parameters (Paper Specific) ---
    sort_by: Optional[
        Literal["score", "published_date", "title"]
    ] = Query(  # 确保这里的 Literal 匹配 PG Repo
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
    # 为每个请求生成唯一ID以便跟踪
    request_id = id(search_service)
    target = cast(Literal["papers", "models", "all"], "papers")  # 显式转换为Literal类型

    # 收集所有参数并记录详细日志
    log_params = {
        "query": q,
        "search_type": search_type,
        "target": target,
        "skip": skip,
        "limit": limit,
        "date_from": date_from,
        "date_to": date_to,
        "area": area,
        "filter_authors": filter_authors,
        "sort_by": sort_by,
        "sort_order": sort_order,
        # Log RRF params if they are relevant (for hybrid)
        "rrf_k": rrf_k if search_type == "hybrid" else None,
        "top_n_semantic": top_n_semantic if search_type == "hybrid" else None,
        "top_n_keyword": top_n_keyword if search_type == "hybrid" else None,
    }
    logger.debug(
        f"[search_papers] [{request_id}] 开始处理论文搜索请求，参数: {log_params}"
    )

    try:
        start_time = time.time()

        if search_type == "semantic":
            # Determine default sort_by for semantic if None
            effective_sort_by = sort_by if sort_by is not None else "score"
            logger.debug(
                f"[search_papers] [{request_id}] 执行语义搜索，有效排序字段: {effective_sort_by}"
            )

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
            logger.debug(
                f"[search_papers] [{request_id}] 执行关键词搜索，有效排序字段: {effective_sort_by}"
            )

            results = await search_service.perform_keyword_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                date_from=date_from,
                date_to=date_to,
                area=area,
                filter_authors=filter_authors,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
            )
        elif search_type == "hybrid":
            # Create filter object
            search_filters = SearchFilterModel(
                published_after=date_from,
                published_before=date_to,
                filter_area=area,
                filter_authors=filter_authors,
                sort_by=sort_by,
                sort_order=sort_order or "desc",  # Ensure sort_order is not None
                pipeline_tag=None,  # Add missing argument
                filter_library_name=None,
                filter_tags=None,
                filter_author=None,
            )
            logger.debug(
                f"[search_papers] [{request_id}] 执行混合搜索，过滤器: {search_filters.model_dump()}"
            )

            # FIXED: Pass filters object, remove individual args and RRF params
            results = await search_service.perform_hybrid_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                filters=search_filters,
            )

        # 计算处理时间并记录结果信息
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"[search_papers] [{request_id}] 论文搜索成功完成，返回 {len(results.items)} 项，共 {results.total} 项，耗时: {process_time:.2f}ms"
        )

        # 如有需要，记录返回的数据摘要
        if results.items and logger.isEnabledFor(logging.DEBUG):
            try:
                # 根据结果类型安全地获取样本数据
                if isinstance(results, PaginatedPaperSearchResult):
                    sample_data = [
                        {"title": item.title, "paper_id": item.paper_id}
                        for item in results.items[:2]
                    ]
                else:
                    # 对于PaginatedSemanticSearchResult或其他类型，使用更安全的方式
                    sample_data = [
                        {"id": str(i), "type": type(item).__name__}
                        for i, item in enumerate(results.items[:2])
                    ]
                logger.debug(
                    f"[search_papers] [{request_id}] 返回数据样例: {sample_data}"
                )
            except Exception as e:
                logger.warning(
                    f"[search_papers] [{request_id}] 无法记录返回数据样例: {str(e)}"
                )

        # 确保返回类型兼容
        if isinstance(results, PaginatedSemanticSearchResult):
            # 已经是PaginatedSemanticSearchResult类型，可以直接返回
            return results
        elif isinstance(results, PaginatedHFModelSearchResult):
            # 不应该出现这种情况，但如果发生了，需要转换为合法的返回类型
            logger.warning(
                f"[search_papers] [{request_id}] 收到意外的PaginatedHFModelSearchResult类型，正在转换"
            )
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
        logger.warning(
            f"[search_papers] [{request_id}] HTTP异常: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        # 使用exception()方法自动包含堆栈跟踪
        logger.exception(f"[search_papers] [{request_id}] 论文搜索过程中发生错误")
        # 记录更多诊断信息
        logger.error(
            f"[search_papers] [{request_id}] 异常类型: {type(e).__name__}, 详情: {str(e)}"
        )
        logger.error(f"[search_papers] [{request_id}] 搜索参数: {log_params}")

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during paper search: {str(e)}",
        )


# --- New Models Endpoint ---
@router.get(
    "/models/",  # New route for models
    response_model=ModelsSearchApiResponse,
    summary="Search Models",
    description="Performs semantic, keyword, or hybrid search specifically for models.",
)
async def search_models(
    # Remove Request parameter
    # request: Request, # Removed Request
    # --- Core Search Parameters ---
    q: str = Query(..., description="The search query string."),
    # Note: search_type now allows semantic, keyword or hybrid for models
    search_type: Literal["semantic", "keyword", "hybrid"] = Query(
        "semantic", description="Type of search to perform."
    ),
    # --- Pagination Parameters ---
    skip: int = Query(0, description="Number of results to skip for pagination.", ge=0),
    limit: int = Query(
        10, description="Maximum number of results to return.", ge=1, le=100
    ),
    # --- Filtering Parameters (Model Specific - Example: Add later if needed) ---
    pipeline_tag: Optional[str] = Query(
        None, description="Filter models by Hugging Face pipeline tag."
    ),
    # 新增模型过滤器
    filter_library_name: Optional[str] = Query(
        None,
        description="Filter models by library name (e.g., 'transformers', case-insensitive exact match).",
    ),
    filter_tags: Optional[List[str]] = Query(
        None, description="Filter models by tags (must contain all specified tags)."
    ),
    filter_author: Optional[str] = Query(
        None,
        description="Filter models by author/organization name (case-insensitive partial match).",
    ),
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
    # 为每个请求生成唯一ID以便跟踪
    request_id = id(search_service)
    target = cast(Literal["papers", "models", "all"], "models")  # 显式转换为Literal类型

    # 收集所有参数并记录详细日志
    log_params = {
        "query": q,
        "search_type": search_type,
        "target": target,
        "skip": skip,
        "limit": limit,
        "pipeline_tag": pipeline_tag,
        "filter_library_name": filter_library_name,
        "filter_tags": filter_tags,
        "filter_author": filter_author,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    logger.debug(
        f"[search_models] [{request_id}] 开始处理模型搜索请求，参数: {log_params}"
    )

    try:
        start_time = time.time()

        if search_type == "semantic":
            # Determine default sort_by for semantic if None
            effective_sort_by = sort_by if sort_by is not None else "score"
            logger.debug(
                f"[search_models] [{request_id}] 执行语义搜索，有效排序字段: {effective_sort_by}"
            )

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
            effective_sort_by = (
                sort_by if sort_by is not None else "last_modified"
            )  # 模型特有默认值
            logger.debug(
                f"[search_models] [{request_id}] 执行关键词搜索，有效排序字段: {effective_sort_by}"
            )

            results = await search_service.perform_keyword_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
                pipeline_tag=pipeline_tag,
                filter_library_name=filter_library_name,
                filter_tags=filter_tags,
                filter_author=filter_author,
            )
        elif search_type == "hybrid":
            # 处理混合搜索
            effective_sort_by = sort_by if sort_by is not None else "score"
            logger.debug(
                f"[search_models] [{request_id}] 执行混合搜索，有效排序字段: {effective_sort_by}"
            )

            # 创建过滤器对象
            search_filters = SearchFilterModel(
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
                pipeline_tag=pipeline_tag,
                filter_library_name=filter_library_name,
                filter_tags=filter_tags,
                filter_author=filter_author,
                published_after=None,
                published_before=None,
                filter_area=None,
                filter_authors=None,
            )
            logger.debug(
                f"[search_models] [{request_id}] 执行混合搜索，过滤器: {search_filters.model_dump()}"
            )

            # 调用服务层方法
            results = await search_service.perform_hybrid_search(
                query=q,
                target=target,
                page=skip // limit + 1,
                page_size=limit,
                filters=search_filters,
            )

        # 计算处理时间并记录结果信息
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"[search_models] [{request_id}] 模型搜索成功完成，返回 {len(results.items)} 项，共 {results.total} 项，耗时: {process_time:.2f}ms"
        )

        # Check the type before returning to satisfy mypy and ensure correctness
        if not isinstance(
            results, (PaginatedHFModelSearchResult, PaginatedSemanticSearchResult)
        ):
            logger.error(
                f"[search_models] [{request_id}] Search service returned unexpected type: {type(results).__name__} for target 'models'"
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Unexpected search result type",
            )

        # 如有需要，记录返回的数据摘要
        if results.items and logger.isEnabledFor(logging.DEBUG):
            try:
                # 安全记录模型搜索结果
                sample_ids = [str(i) for i in range(min(2, len(results.items)))]
                result_type = type(results).__name__
                sample_data = {"result_type": result_type, "sample_ids": sample_ids}
                logger.debug(
                    f"[search_models] [{request_id}] 返回数据样例: {sample_data}"
                )
            except Exception as e:
                logger.warning(
                    f"[search_models] [{request_id}] 无法记录返回数据样例: {str(e)}"
                )

        return results

    except HTTPException as http_exc:
        logger.warning(
            f"[search_models] [{request_id}] HTTP异常: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        # 使用exception()方法自动包含堆栈跟踪
        logger.exception(f"[search_models] [{request_id}] 模型搜索过程中发生错误")
        # 记录更多诊断信息
        logger.error(
            f"[search_models] [{request_id}] 异常类型: {type(e).__name__}, 详情: {str(e)}"
        )
        logger.error(f"[search_models] [{request_id}] 搜索参数: {log_params}")

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during model search: {str(e)}",
        )


# --- 新增端点，获取所有可用的论文领域列表 ---
@router.get(
    "/paper-areas/",
    response_model=List[str],
    summary="获取论文领域列表",
    description="获取系统中所有可用的论文研究领域(area)列表，用于前端过滤器选项。",
)
async def get_paper_areas(
    search_service: SearchService = Depends(deps.get_search_service),
) -> List[str]:
    """获取所有可用的论文领域列表，供前端过滤器使用。"""
    try:
        areas = await search_service.get_available_paper_areas()
        logger.info(f"成功获取 {len(areas)} 个论文领域")
        return areas
    except Exception as e:
        logger.exception(f"获取论文领域列表时出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取论文领域列表失败: {str(e)}")
