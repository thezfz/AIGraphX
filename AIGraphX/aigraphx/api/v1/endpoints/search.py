# -----------------------------------------------------------------------------
# 文件名: aigraphx/api/v1/endpoints/search.py
#
# 描述:
# 这个文件定义了与"搜索"功能相关的 API 端点。它负责处理来自客户端的
# 搜索请求，允许用户根据不同的条件（关键词、语义、混合）和目标
# （论文、模型）来查找信息。此外，它还提供了一个端点来获取可用于
# 过滤的论文研究领域列表。
#
# 主要功能:
# - 定义 FastAPI 路由 (`APIRouter`) 来组织搜索相关的端点。
# - 提供一个专门用于搜索论文的端点 (`/papers/`)，支持语义、关键词和混合搜索，
#   并带有过滤（日期、领域、作者）和排序选项。
# - 提供一个专门用于搜索 Hugging Face 模型的端点 (`/models/`)，同样支持
#   语义、关键词和混合搜索，并带有过滤（pipeline_tag、库名、标签、作者）
#   和排序选项。
# - 提供一个获取所有可用论文研究领域列表的端点 (`/paper-areas/`)，
#   方便前端构建筛选器。
#
# 交互:
# - 路由 (`router`) 会被包含在 `aigraphx/api/v1/api.py` 中，
#   并整合到 `aigraphx/main.py` 的主 FastAPI 应用。
# - 搜索端点函数主要依赖于 `SearchService` (通过 `Depends(deps.get_search_service)`)
#   来执行实际的搜索逻辑。`SearchService` 会根据请求的类型（语义、关键词、混合）
#   和目标（论文、模型）调用相应的 Repository（`PostgresRepository`,
#   `FaissRepository`）和向量化工具（`TextEmbedder`）。
# - `SearchService` 实例由 `aigraphx/api/v1/dependencies.py` 的
#   `get_search_service` 提供。
# - 端点函数使用 Pydantic 模型 (来自 `aigraphx/models/search.py`)
#   来定义请求参数（通过 Query 和 Path）、请求体（如果需要）以及响应体
#   的数据结构。`SearchFilterModel` 用于统一管理搜索过滤器。
# - 使用 `typing.Union` 和 `Literal` 来精确定义请求参数和响应模型的类型。
# - 使用 `logging` 模块记录详细的请求信息、处理步骤和潜在错误。
# - 使用 `HTTPException` 处理预期错误（如无效参数）和意外错误。
# - 使用 `time` 模块计算请求处理时间。
# -----------------------------------------------------------------------------

# 导入标准库
import logging  # 日志记录
from typing import Union, Optional, Literal, cast, List, Any, get_args  # 类型提示
from datetime import date  # 日期类型，用于日期过滤
import time  # 时间相关操作，用于计算请求处理耗时

# 导入 Pydantic 相关，用于数据验证和模型定义
from pydantic import BaseModel, Field

# 添加缺失的导入
from fastapi import APIRouter, Depends, HTTPException, Query

# 导入服务层 (Service Layer)
# SearchService 封装了所有搜索相关的业务逻辑
from aigraphx.services.search_service import SearchService

# 导入 API 数据模型 (Pydantic Models)
# 这些模型定义了搜索结果项、分页结果等的结构
from aigraphx.models.search import (
    SearchResultItem,  # 单个搜索结果项（通用，或用于论文）
    PaginatedPaperSearchResult,  # 分页的论文搜索结果（主要用于关键词）
    PaginatedSemanticSearchResult,  # 分页的语义搜索结果（通用）
    PaginatedHFModelSearchResult,  # 分页的 Hugging Face 模型搜索结果
    HFSearchResultItem,  # 单个 Hugging Face 模型搜索结果项
    SearchFilterModel,  # 用于封装所有搜索过滤条件的模型
)

# --- 定义 API 响应联合类型 ---
# 一个端点可能根据 search_type 返回不同结构的结果，使用 Union 定义可能的响应类型。
# 这样 FastAPI 的文档和类型检查能更好地处理这种情况。

# 定义论文搜索 API 可能的响应类型
PapersSearchApiResponse = Union[
    PaginatedPaperSearchResult, PaginatedSemanticSearchResult
]
# 定义模型搜索 API 可能的响应类型
# 注意：语义搜索也可以返回模型结果，所以包含 PaginatedSemanticSearchResult
ModelsSearchApiResponse = Union[
    PaginatedHFModelSearchResult, PaginatedSemanticSearchResult
]

# -----------------------------------------------------------------------------
# --- 依赖注入 ---
# 导入集中的依赖提供模块 `dependencies` 并使用别名 `deps`
from aigraphx.api.v1 import dependencies as deps
# 虽然端点主要依赖 SearchService，但这里保留了导入 Repository 和 Embedder 的注释，
# 以便理解 SearchService 内部可能依赖哪些组件。实际注入由 `deps.get_search_service` 处理。
# from aigraphx.repositories.postgres_repo import PostgresRepository
# from aigraphx.repositories.faiss_repo import FaissRepository
# from aigraphx.vectorization.embedder import TextEmbedder
# -----------------------------------------------------------------------------


# --- 路由和日志设置 ---
router = APIRouter()  # 创建搜索相关的路由
logger = logging.getLogger(__name__)  # 获取日志记录器


# -----------------------------------------------------------------------------
# --- API 端点: 搜索论文 ---
# -----------------------------------------------------------------------------
@router.get(
    "/papers/",  # 端点路径，表示搜索论文资源
    response_model=PapersSearchApiResponse,  # 指定可能的响应模型类型
    summary="搜索论文",  # API 文档摘要
    description="执行针对论文的语义、关键词或混合搜索，支持可选的过滤和排序。",  # API 文档描述
)
async def search_papers(
    # --- 核心搜索参数 ---
    q: str = Query(..., description="搜索查询字符串。"),  # 必需的查询参数 q
    search_type: Literal["semantic", "keyword", "hybrid"] = Query(
        "semantic",
        description="要执行的搜索类型。",  # 搜索类型，默认为 semantic
    ),
    # --- 分页参数 ---
    skip: int = Query(
        0, description="用于分页要跳过的结果数量。", ge=0
    ),  # 跳过的条目数，必须 >= 0
    limit: int = Query(
        10,
        description="返回结果的最大数量。",
        ge=1,
        le=100,  # 每页数量，必须在 1 到 100 之间
    ),
    # --- 过滤参数 (论文特定) ---
    date_from: Optional[date] = Query(
        None,
        description="过滤在此日期之后（含）发布的论文 (YYYY-MM-DD)。",  # 可选的起始日期
    ),
    date_to: Optional[date] = Query(
        None,
        description="过滤在此日期之前（含）发布的论文 (YYYY-MM-DD)。",  # 可选的结束日期
    ),
    # `Query(None, ...)` 表示参数是可选的。
    # `List[str]` 表示可以接受多个 area 值，例如 /papers/?area=CV&area=NLP
    area: Optional[List[str]] = Query(
        None,
        description="按研究领域过滤论文。支持多选过滤 (使用 OR 逻辑)。",  # 可选的研究领域列表
    ),
    # 新增作者过滤参数
    filter_authors: Optional[List[str]] = Query(
        None,
        description="按作者姓名过滤论文。支持多个，使用部分匹配 (OR 逻辑)。",  # 可选的作者列表
    ),
    # --- 排序参数 (论文特定) ---
    # 排序字段，限制为指定的几个值
    # 注意：这里的 Literal 值需要与 PostgresRepository 中支持的排序字段匹配
    sort_by: Optional[
        Literal[
            "score", "published_date", "title"
        ]  # "score" 主要用于语义/混合, "published_date" 主要用于关键词
    ] = Query(
        None,
        description="论文结果排序字段。默认: 语义/混合为 'score', 关键词为 'published_date'。",
    ),
    sort_order: Optional[Literal["asc", "desc"]] = Query(
        "desc",
        description="排序顺序: 'asc' (升序) 或 'desc' (降序)。",  # 排序顺序，默认降序
    ),
    # --- 混合搜索特定参数 (如果 search_type='hybrid') ---
    # 这些参数用于 Reciprocal Rank Fusion (RRF) 算法，该算法结合语义和关键词搜索结果
    rrf_k: int = Query(
        60,
        description="倒数排名融合 k 参数 (用于混合搜索)。",
        ge=1,  # RRF 算法的 k 值
    ),
    top_n_semantic: int = Query(
        100,
        description="为融合获取的初始语义结果数量 (混合搜索)。",  # 初始获取的语义结果数
        ge=1,
    ),
    top_n_keyword: int = Query(
        100,
        description="为融合获取的初始关键词结果数量 (混合搜索)。",  # 初始获取的关键词结果数
        ge=1,
    ),
    # --- 依赖注入 ---
    # 注入 SearchService 实例
    search_service: SearchService = Depends(deps.get_search_service),
) -> PapersSearchApiResponse:  # 函数返回上面定义的联合类型
    """
    专门用于搜索论文的 API 端点。
    根据 `search_type` 参数，调用 `SearchService` 中不同的搜索方法。
    处理过滤、排序和分页逻辑。
    """
    # 为每个请求生成一个唯一的标识符，方便在日志中追踪同一个请求的处理过程
    request_id = id(search_service)
    # 明确指定搜索目标为 "papers"。
    # 使用 cast 是为了帮助类型检查器理解 target 的具体类型是 "papers"，
    # 尽管 SearchService 的方法可能接受更广泛的类型 ("papers", "models", "all")。
    target = cast(Literal["papers", "models", "all"], "papers")

    # --- 收集所有接收到的参数，用于详细日志记录 ---
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
        # 仅当搜索类型为 hybrid 时记录 RRF 相关参数才有意义
        "rrf_k": rrf_k if search_type == "hybrid" else None,
        "top_n_semantic": top_n_semantic if search_type == "hybrid" else None,
        "top_n_keyword": top_n_keyword if search_type == "hybrid" else None,
    }
    # 记录 DEBUG 级别的日志，包含请求 ID 和所有参数
    logger.debug(
        f"[search_papers] [{request_id}] 开始处理论文搜索请求，参数: {log_params}"
    )

    try:
        # 记录请求开始处理的时间点
        start_time = time.time()

        # --- 根据搜索类型调用不同的服务层方法 ---
        if search_type == "semantic":
            # 如果用户没有指定排序字段，为语义搜索设置默认排序字段为 'score'
            effective_sort_by = sort_by if sort_by is not None else "score"
            logger.debug(
                f"[search_papers] [{request_id}] 执行语义搜索，有效排序字段: {effective_sort_by}"
            )
            # 调用服务层的语义搜索方法
            results = await search_service.perform_semantic_search(
                query=q,
                target=target,  # 目标是论文
                page=skip // limit + 1,  # 将 skip/limit 转换为页码 (从 1 开始)
                page_size=limit,  # 每页大小
                sort_by=effective_sort_by,  # 使用确定的排序字段
                sort_order=sort_order
                or "desc",  # 如果 sort_order 为 None，则默认为 "desc"
                # 注意：语义搜索目前不直接处理日期/领域/作者过滤，这些过滤可能在服务层内部
                # 获取到 ID 后再应用，或者语义搜索本身就侧重于内容相似度。
            )
        elif search_type == "keyword":
            # 如果用户没有指定排序字段，为关键词搜索设置默认排序字段为 'published_date'
            effective_sort_by = sort_by if sort_by is not None else "published_date"
            logger.debug(
                f"[search_papers] [{request_id}] 执行关键词搜索，有效排序字段: {effective_sort_by}"
            )
            # 调用服务层的关键词搜索方法
            results = await search_service.perform_keyword_search(
                query=q,
                target=target,  # 目标是论文
                page=skip // limit + 1,
                page_size=limit,
                date_from=date_from,  # 传递日期过滤器
                date_to=date_to,
                area=area,  # 传递领域过滤器
                filter_authors=filter_authors,  # 传递作者过滤器
                sort_by=effective_sort_by,  # 传递排序字段
                sort_order=sort_order or "desc",  # 传递排序顺序
            )
        elif search_type == "hybrid":
            # --- 对于混合搜索，首先创建一个过滤器对象 ---
            # 使用 SearchFilterModel 统一封装所有过滤和排序参数，传递给服务层。
            # 这样可以避免函数参数列表过长，也更方便扩展。
            search_filters = SearchFilterModel(
                published_after=date_from,
                published_before=date_to,
                filter_area=area,
                filter_authors=filter_authors,
                sort_by=sort_by,  # 注意：混合搜索后的最终排序可能基于融合分数，这里传入的 sort_by 可能用于初始检索阶段
                sort_order=sort_order or "desc",
                # 为模型添加的过滤器，这里对于论文搜索可以设为 None
                pipeline_tag=None,
                filter_library_name=None,
                filter_tags=None,
                filter_author=None,
            )
            logger.debug(
                f"[search_papers] [{request_id}] 执行混合搜索，过滤器: {search_filters.model_dump()}"  # model_dump() 用于获取模型的字典表示
            )

            # 调用服务层的混合搜索方法
            # 注意：混合搜索内部会调用语义和关键词搜索，并使用 RRF 融合结果。
            # RRF 相关参数 (rrf_k, top_n_semantic, top_n_keyword) 在服务层内部使用，
            # 这里不再需要单独传递。
            results = await search_service.perform_hybrid_search(
                query=q,
                target=target,  # 目标是论文
                page=skip // limit + 1,
                page_size=limit,
                filters=search_filters,  # 传递统一的过滤器对象
                # 移除旧的单独参数传递
                # rrf_k=rrf_k,
                # top_n_semantic=top_n_semantic,
                # top_n_keyword=top_n_keyword,
            )
        else:
            # 如果 search_type 是无效的值 (理论上 FastAPI 的 Literal 会阻止这种情况)
            logger.error(
                f"[search_papers] [{request_id}] 无效的搜索类型: {search_type}"
            )
            raise HTTPException(
                status_code=400, detail=f"无效的搜索类型: {search_type}"
            )

        # --- 请求处理完成 ---
        # 计算处理时间（毫秒）
        process_time = (time.time() - start_time) * 1000
        # 记录 INFO 级别的日志，说明搜索成功完成、返回结果数量和耗时
        logger.info(
            f"[search_papers] [{request_id}] 论文搜索成功完成，返回 {len(results.items) if results else 0} 项，共 {results.total if results else 0} 项，耗时: {process_time:.2f}ms"
        )

        # --- （可选）记录返回数据的样例，仅在 DEBUG 级别启用时 ---
        if results and results.items and logger.isEnabledFor(logging.DEBUG):
            try:
                # 根据实际返回的结果类型，安全地提取一些样例信息用于日志
                # PaginatedPaperSearchResult 包含 title 和 paper_id
                if isinstance(results, PaginatedPaperSearchResult):
                    sample_data = [
                        {"title": item.title, "paper_id": item.paper_id}
                        for item in results.items[:2]  # 只取前 2 条作为样例
                    ]
                # PaginatedSemanticSearchResult 的 items 是 SearchResultItem，也有 title 和 id
                elif isinstance(results, PaginatedSemanticSearchResult):
                    sample_data = []
                    for item in results.items[:2]:
                        # 安全地获取属性，并处理类型
                        item_id = getattr(item, "id", "N/A")
                        item_title = getattr(
                            item, "title", "N/A"
                        )  # SearchResultItem 也有 title
                        item_score_val = getattr(item, "score", None)
                        item_score = (
                            str(item_score_val) if item_score_val is not None else None
                        )
                        sample_data.append(
                            {
                                "title": item_title,
                                "id": str(item_id),
                                "score": item_score,
                            }
                        )
                # 其他未知类型（理论上不应出现）
                else:
                    sample_data = [
                        {"id": str(i), "type": type(item).__name__}
                        for i, item in enumerate(results.items[:2])
                    ]
                logger.debug(
                    f"[search_papers] [{request_id}] 返回数据样例: {sample_data}"
                )
            except Exception as e:
                # 如果记录样例时出错，只记录警告，不影响正常响应
                logger.warning(
                    f"[search_papers] [{request_id}] 无法记录返回数据样例: {str(e)}"
                )

        # --- 返回结果 ---
        # FastAPI 会根据 response_model (PapersSearchApiResponse) 自动验证和序列化 results
        # 添加类型检查确保返回值符合预期
        if not isinstance(results, get_args(PapersSearchApiResponse)):
            logger.error(
                f"[search_papers] [{request_id}] 返回了非预期的类型: {type(results)}. 应为 PapersSearchApiResponse."
            )
            raise HTTPException(
                status_code=500, detail="搜索服务返回了意外的结果类型。"
            )
        # mypy 仍然可能推断类型为Any，显式转换
        return cast(PapersSearchApiResponse, results)

    except HTTPException as http_exc:
        # 如果是预期的 HTTPException (例如服务层或仓库层抛出的 404 或 400)，记录警告并重新抛出
        logger.warning(
            f"[search_papers] [{request_id}] HTTP异常: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        # 如果是其他未预料到的异常
        logger.exception(
            f"[search_papers] [{request_id}] 处理论文搜索请求时发生意外错误: {e}"
        )
        # 返回 500 内部服务器错误
        raise HTTPException(
            status_code=500, detail="处理论文搜索请求时发生内部服务器错误。"
        )


# -----------------------------------------------------------------------------
# --- API 端点: 搜索模型 ---
# -----------------------------------------------------------------------------
@router.get(
    "/models/",  # 新增的专门用于搜索模型的路径
    response_model=ModelsSearchApiResponse,  # 指定模型搜索可能的响应模型
    summary="搜索模型",
    description="执行针对模型的语义、关键词或混合搜索。",
)
async def search_models(
    # --- 核心搜索参数 ---
    q: str = Query(..., description="搜索查询字符串。"),
    # 模型搜索同样支持这三种类型
    search_type: Literal["semantic", "keyword", "hybrid"] = Query(
        "semantic", description="要执行的搜索类型。"
    ),
    # --- 分页参数 ---
    skip: int = Query(0, description="用于分页要跳过的结果数量。", ge=0),
    limit: int = Query(10, description="返回结果的最大数量。", ge=1, le=100),
    # --- 过滤参数 (模型特定) ---
    # 注意：这里的过滤器是示例，需要根据实际数据和 SearchService 的实现来确定
    pipeline_tag: Optional[str] = Query(
        None,
        description="按 Hugging Face pipeline 标签过滤模型。",  # 例如 'text-generation', 'image-classification'
    ),
    filter_library_name: Optional[str] = Query(
        None,
        description="按库名称过滤模型 (例如 'transformers', 不区分大小写的精确匹配)。",  # 例如 'transformers', 'diffusers'
    ),
    filter_tags: Optional[List[str]] = Query(
        None,
        description="按标签过滤模型 (必须包含所有指定的标签)。",  # 例如 ['pytorch', 'en']
    ),
    filter_author: Optional[str] = Query(
        None,
        description="按作者/组织名称过滤模型 (不区分大小写的部分匹配)。",  # 例如 'google', 'openai'
    ),
    # --- 排序参数 (模型特定) ---
    # 模型可以按下载量、点赞数等排序
    sort_by: Optional[Literal["score", "likes", "downloads", "last_modified"]] = Query(
        None,
        description="模型结果排序字段。默认: 语义为 'score', 关键词为 'likes'。",
    ),
    sort_order: Optional[Literal["asc", "desc"]] = Query(
        "desc", description="排序顺序: 'asc' 或 'desc'。"
    ),
    # RRF 参数，仅混合搜索需要，处理逻辑同论文搜索
    rrf_k: int = Query(60, ge=1),
    top_n_semantic: int = Query(100, ge=1),
    top_n_keyword: int = Query(100, ge=1),
    # --- 依赖注入 ---
    search_service: SearchService = Depends(deps.get_search_service),
) -> ModelsSearchApiResponse:
    """
    专门用于搜索 Hugging Face 模型的 API 端点。
    根据 search_type 调用 SearchService 的方法，处理模型特定的过滤和排序。
    """
    request_id = id(search_service)
    # 明确指定搜索目标为 "models"
    target = cast(Literal["papers", "models", "all"], "models")

    # --- 收集参数并记录日志 ---
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
        "rrf_k": rrf_k if search_type == "hybrid" else None,
        "top_n_semantic": top_n_semantic if search_type == "hybrid" else None,
        "top_n_keyword": top_n_keyword if search_type == "hybrid" else None,
    }
    logger.debug(
        f"[search_models] [{request_id}] 开始处理模型搜索请求，参数: {log_params}"
    )

    try:
        start_time = time.time()

        # --- 根据搜索类型调用服务 ---
        if search_type == "semantic":
            effective_sort_by = sort_by if sort_by is not None else "score"
            logger.debug(
                f"[search_models] [{request_id}] 执行语义搜索，有效排序字段: {effective_sort_by}"
            )
            results = await search_service.perform_semantic_search(
                query=q,
                target=target,  # 目标是模型
                page=skip // limit + 1,
                page_size=limit,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
                # 语义搜索可能不直接支持模型特定过滤，需要服务层内部处理
            )
        elif search_type == "keyword":
            effective_sort_by = (
                sort_by if sort_by is not None else "likes"
            )  # 模型默认按点赞数排序
            logger.debug(
                f"[search_models] [{request_id}] 执行关键词搜索，有效排序字段: {effective_sort_by}"
            )
            results = await search_service.perform_keyword_search(
                query=q,
                target=target,  # 目标是模型
                page=skip // limit + 1,
                page_size=limit,
                sort_by=effective_sort_by,
                sort_order=sort_order or "desc",
                # 传递模型特定的过滤器
                pipeline_tag=pipeline_tag,
                filter_library_name=filter_library_name,
                filter_tags=filter_tags,
                filter_author=filter_author,
                # 论文相关的过滤器设为 None
                date_from=None,
                date_to=None,
                area=None,
                filter_authors=None,
            )
        elif search_type == "hybrid":
            # 创建包含模型过滤器的 SearchFilterModel 实例
            search_filters = SearchFilterModel(
                pipeline_tag=pipeline_tag,
                filter_library_name=filter_library_name,
                filter_tags=filter_tags,
                filter_author=filter_author,
                sort_by=sort_by,
                sort_order=sort_order or "desc",
                # 论文相关的过滤器设为 None
                published_after=None,
                published_before=None,
                filter_area=None,
                filter_authors=None,
            )
            logger.debug(
                f"[search_models] [{request_id}] 执行混合搜索，过滤器: {search_filters.model_dump()}"
            )

            results = await search_service.perform_hybrid_search(
                query=q,
                target=target,  # 目标是模型
                page=skip // limit + 1,
                page_size=limit,
                filters=search_filters,  # 传递过滤器对象
                # RRF 参数由服务层内部处理
                # rrf_k=rrf_k,
                # top_n_semantic=top_n_semantic,
                # top_n_keyword=top_n_keyword,
            )
        else:
            logger.error(
                f"[search_models] [{request_id}] 无效的搜索类型: {search_type}"
            )
            raise HTTPException(
                status_code=400, detail=f"无效的搜索类型: {search_type}"
            )

        # --- 请求处理完成 ---
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"[search_models] [{request_id}] 模型搜索成功完成，返回 {len(results.items) if results else 0} 项，共 {results.total if results else 0} 项，耗时: {process_time:.2f}ms"
        )

        # --- （可选）记录返回数据样例 ---
        if results and results.items and logger.isEnabledFor(logging.DEBUG):
            try:
                # PaginatedHFModelSearchResult
                if isinstance(results, PaginatedHFModelSearchResult):
                    sample_data = [
                        {"model_id": item.model_id, "likes": item.likes}
                        for item in results.items[:2]
                    ]
                # PaginatedSemanticSearchResult (items are SearchResultItem)
                elif isinstance(results, PaginatedSemanticSearchResult):
                    sample_data = []
                    for item in results.items[:2]:
                        # 安全地获取属性，并处理类型
                        item_id = getattr(item, "id", "N/A")
                        item_type = getattr(item, "type", "N/A")
                        item_score_val = getattr(item, "score", None)
                        item_score = (
                            str(item_score_val) if item_score_val is not None else None
                        )
                        sample_data.append(
                            {"id": str(item_id), "type": item_type, "score": item_score}
                        )
                else:
                    sample_data = [
                        {"id": str(i), "type": type(item).__name__}
                        for i, item in enumerate(results.items[:2])
                    ]
                logger.debug(
                    f"[search_models] [{request_id}] 返回数据样例: {sample_data}"
                )
            except Exception as e:
                logger.warning(
                    f"[search_models] [{request_id}] 无法记录返回数据样例: {str(e)}"
                )

        # --- 返回结果 ---
        # 添加类型检查确保返回值符合预期
        if not isinstance(results, get_args(ModelsSearchApiResponse)):
            logger.error(
                f"[search_models] [{request_id}] 返回了非预期的类型: {type(results)}. 应为 ModelsSearchApiResponse."
            )
            raise HTTPException(
                status_code=500, detail="搜索服务返回了意外的结果类型。"
            )
        # mypy 仍然可能推断类型为Any，显式转换
        return cast(ModelsSearchApiResponse, results)

    except HTTPException as http_exc:
        logger.warning(
            f"[search_models] [{request_id}] HTTP异常: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.exception(
            f"[search_models] [{request_id}] 处理模型搜索请求时发生意外错误: {e}"
        )
        raise HTTPException(
            status_code=500, detail="处理模型搜索请求时发生内部服务器错误。"
        )


# -----------------------------------------------------------------------------
# --- API 端点: 获取论文领域列表 ---
# -----------------------------------------------------------------------------
@router.get(
    "/paper-areas/",  # 端点路径
    response_model=List[str],  # 响应模型是一个字符串列表
    summary="获取论文领域列表",
    description="获取系统中所有可用的论文研究领域(area)列表，用于前端过滤器选项。",
)
async def get_paper_areas(
    # 只需要 SearchService 来调用获取领域的方法
    search_service: SearchService = Depends(deps.get_search_service),
) -> List[str]:
    """
    获取所有不同的论文研究领域列表。
    """
    request_id = id(search_service)
    logger.info(f"[get_paper_areas] [{request_id}] 收到获取论文领域列表的请求")
    try:
        start_time = time.time()
        # 由 get_available_paper_areas 改为 get_distinct_paper_areas (确认此方法存在于SearchService)
        areas = await search_service.get_distinct_paper_areas()  # type: ignore
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"[get_paper_areas] [{request_id}] 获取到 {len(areas)} 个论文领域，耗时: {process_time:.2f}ms"
        )
        # 确保返回的是List[str]类型
        string_areas: List[str] = []
        for area in areas:
            if isinstance(area, str):
                string_areas.append(area)
            elif area is not None:
                string_areas.append(str(area))
        return string_areas
    except Exception as e:
        # 这里假设 `get_distinct_paper_areas` 不会抛出 HTTPException，
        # 如果有特定错误（如数据库连接失败），SearchService 应该处理或记录。
        # 如果发生任何其他异常，记录错误并返回 500。
        logger.exception(
            f"[get_paper_areas] [{request_id}] 获取论文领域列表时发生意外错误: {e}"
        )
        raise HTTPException(
            status_code=500, detail="获取论文领域列表时发生内部服务器错误。"
        )
