# aigraphx/services/search_service.py

import logging
import numpy as np
import asyncio
import math
import json
from datetime import date, datetime, timezone
from typing import (
    List,
    Dict,
    Optional,
    Literal,
    Union,
    Tuple,
    Set,
    TypeAlias,
    Type,
    cast,
    get_args,
    Callable,
    Coroutine,
    Any,
    Sequence,  # Import Sequence for type hinting variance
)
from fastapi import HTTPException, status  # Add imports
from pydantic import ValidationError

# Import necessary components
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.vectorization.embedder import TextEmbedder
from aigraphx.models.search import (
    SearchResultItem,
    HFSearchResultItem,
    PaginatedPaperSearchResult,
    PaginatedSemanticSearchResult,
    PaginatedHFModelSearchResult,
    AnySearchResultItem,
    PaginatedModel,
    SearchFilterModel,
)
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# --- Type Aliases for Clarity ---
SearchTarget: TypeAlias = Literal["papers", "models", "all"]
# Refined SortBy types to avoid ambiguity when None is allowed by function signature
PaperSortByLiteral: TypeAlias = Literal["score", "published_date", "title"]
ModelSortByLiteral: TypeAlias = Literal["score", "likes", "downloads", "last_modified"]
SortOrderLiteral: TypeAlias = Literal["asc", "desc"]

FaissID: TypeAlias = Union[int, str]
ResultItem: TypeAlias = Union[SearchResultItem, HFSearchResultItem]
PaginatedResult: TypeAlias = Union[
    PaginatedPaperSearchResult,
    PaginatedHFModelSearchResult,
    PaginatedSemanticSearchResult,  # For generic/error cases
]

logger = logging.getLogger(__name__)


class SearchService:
    """
    Provides services for searching papers and models using semantic, keyword,
    and hybrid approaches.
    """

    DEFAULT_RRF_K: int = 60
    DEFAULT_TOP_N_SEMANTIC: int = 100
    DEFAULT_TOP_N_KEYWORD: int = 100

    def __init__(
        self,
        embedder: Optional[TextEmbedder],
        faiss_repo_papers: FaissRepository,
        faiss_repo_models: FaissRepository,
        pg_repo: PostgresRepository,
        neo4j_repo: Optional[Neo4jRepository],
    ):
        """Initializes the service with necessary dependencies."""
        self.embedder = embedder
        self.faiss_repo_papers = faiss_repo_papers
        self.faiss_repo_models = faiss_repo_models
        self.pg_repo = pg_repo
        self.neo4j_repo = neo4j_repo
        logger.info("SearchService initialized.")
        if self.neo4j_repo is None:
            logger.warning(
                "SearchService initialized without a Neo4j repository. Graph features may be unavailable."
            )

    def _convert_distance_to_score(self, distance: float) -> float:
        """
        Converts Faiss distance (non-negative, typically L2) to a
        similarity score (0, 1]. Lower distance means higher score.
        Uses the formula: score = 1 / (1 + distance)
        """
        if distance < 0:
            logger.warning(
                f"Received negative Faiss distance: {distance}. Clamping to 0."
            )
            distance = 0.0
        # Add a small epsilon to prevent potential division by zero and ensure score > 0
        return 1.0 / (1.0 + distance + 1e-9)

    async def _get_paper_details_for_ids(
        self, paper_ids: List[int], scores: Optional[Dict[int, Optional[float]]] = None
    ) -> List[SearchResultItem]:
        """从数据库获取指定ID列表的论文详细信息。

        Args:
            paper_ids: 论文ID列表
            scores: 可选的ID到分数的映射

        Returns:
            包含论文详细信息的SearchResultItem列表
        """
        if not paper_ids:
            return []

        try:
            # 获取论文详细信息
            paper_details_list = await self.pg_repo.get_papers_details_by_ids(paper_ids)
            if not paper_details_list:
                return []

            # 创建ID到详细信息的映射，并按原始顺序填充结果
            paper_details_map = {
                details.get("paper_id"): details for details in paper_details_list
            }
            results: List[SearchResultItem] = []

            for paper_id in paper_ids:
                if details := paper_details_map.get(paper_id):
                    # 获取分数，如果存在
                    score = None
                    if scores is not None:
                        score = scores.get(paper_id)  # 可能为None

                    # 构建SearchResultItem
                    try:
                        item = SearchResultItem(
                            paper_id=paper_id,
                            pwc_id=details.get("pwc_id", ""),
                            title=details.get("title", ""),
                            summary=details.get("summary", ""),
                            score=score,  # 允许None
                            pdf_url=details.get("pdf_url", ""),
                            published_date=details.get("published_date"),
                            authors=details.get("authors", []),
                            area=details.get("area", ""),
                        )
                        results.append(item)
                    except ValidationError as ve:
                        # 记录验证错误详细信息
                        logger.error(
                            f"Validation error creating SearchResultItem for paper_id={paper_id}: {ve}"
                        )
                        # 继续处理其他项

            return results

        except Exception as e:
            logger.error(f"Error fetching paper details: {str(e)}", exc_info=True)
            return []

    async def _get_model_details_for_ids(
        self, model_ids: List[str], scores: Optional[Dict[str, float]] = None
    ) -> List[HFSearchResultItem]:
        """
        Helper to fetch HF model details from Postgres for a list of model IDs.
        """
        # 添加日志记录
        logger.debug(f"[_get_model_details_for_ids] Received {len(model_ids)} IDs: {model_ids[:10]}...")
        if not model_ids:
            logger.debug("[_get_model_details_for_ids] No model IDs provided, returning empty list.")
            return []

        logger.debug(f"[_get_model_details_for_ids] Fetching details for {len(model_ids)} model IDs from PG.")
        result_items_map: Dict[str, HFSearchResultItem] = {}
        model_details_list: List[Dict[str, Any]] = []  # Initialize

        # Check if the method exists before calling
        if not hasattr(self.pg_repo, "get_hf_models_by_ids"):
            logger.error(
                "[_get_model_details_for_ids] PostgresRepository does not have method 'get_hf_models_by_ids'."
            )
            return []

        try:
            model_details_list = await self.pg_repo.get_hf_models_by_ids(model_ids)
            logger.debug(f"[_get_model_details_for_ids] Fetched {len(model_details_list)} detail records from PG.")
        except Exception as e:
            logger.error(
                f"[_get_model_details_for_ids] Failed to fetch details from PG: {e}",
                exc_info=True,
            )
            return []  # Return empty on DB error
        
        if not model_details_list:
            logger.warning("[_get_model_details_for_ids] PG returned no details for the given model IDs.")
            return [] 

        # 修正：使用正确的数据库列名 'hf_model_id' 作为键
        details_map = {
            detail.get("hf_model_id"): detail
            for detail in model_details_list
            if detail.get("hf_model_id")
        }
        logger.debug(f"[_get_model_details_for_ids] Created details_map with {len(details_map)} entries.")

        # 添加日志记录处理结果数量
        processed_count = 0
        skipped_count = 0
        creation_errors = 0

        for model_id in model_ids: # model_id 是从 Faiss 来的 ID
            detail_result = details_map.get(model_id)
            if detail_result is None:
                # This log might be less frequent now, but good to keep
                logger.warning(f"[_get_model_details_for_ids] Details not found in details_map for model_id {model_id} (from Faiss). This might indicate inconsistency between Faiss and DB.")
                skipped_count += 1
                continue

            # Process fields safely
            processed_tags: Optional[List[str]] = None
            tags_list = detail_result.get("tags")
            if isinstance(tags_list, str):
                try:
                    parsed_tags = json.loads(tags_list)
                    if isinstance(parsed_tags, list) and all(
                        isinstance(t, str) for t in parsed_tags
                    ):
                        processed_tags = parsed_tags
                    else:
                        logger.warning(
                            f"[_get_model_details_for_ids] Decoded tags for model {model_id} is not a list of strings."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"[_get_model_details_for_ids] Failed to decode tags JSON for model_id {model_id}: {tags_list}"
                    )
            elif isinstance(tags_list, list) and all(
                isinstance(t, str) for t in tags_list
            ):
                processed_tags = tags_list

            score_float = scores.get(model_id, 0.0) if scores else 0.0

            processed_last_modified_str: Optional[str] = (
                None  # Initialize as Optional[str]
            )
            last_modified_val = detail_result.get("last_modified")
            if isinstance(last_modified_val, (datetime, date)):
                processed_last_modified_str = last_modified_val.isoformat()
            elif isinstance(last_modified_val, str):
                try:
                    dt_obj = datetime.fromisoformat(
                        last_modified_val.replace("Z", "+00:00")
                    )
                    processed_last_modified_str = dt_obj.isoformat()  # Store as string
                except ValueError:
                    try:
                        d = date.fromisoformat(last_modified_val)
                        processed_last_modified_str = datetime.combine(
                            d, datetime.min.time()
                        ).isoformat()
                    except ValueError:
                        logger.warning(
                            f"[_get_model_details_for_ids] Could not parse last_modified string '{last_modified_val}' for model {model_id}, using as is."
                        )
                        processed_last_modified_str = (
                            last_modified_val  # Pass through if unparseable
                        )
            elif last_modified_val is not None:
                logger.warning(
                    f"[_get_model_details_for_ids] Unexpected type for last_modified: {type(last_modified_val)}, converting to str."
                )
                processed_last_modified_str = str(last_modified_val)

            final_last_modified_dt: Optional[datetime] = None
            if processed_last_modified_str:
                try:
                    final_last_modified_dt = datetime.fromisoformat(
                        processed_last_modified_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    logger.warning(
                        f"[_get_model_details_for_ids] Could not parse final last_modified string '{processed_last_modified_str}' back to datetime for model {model_id}. Setting to None."
                    )

            try:
                # 修正：创建实例时，使用 detail_result.get('hf_model_id') 对应 Pydantic 的 model_id
                # 并确认其他字段名与数据库列名匹配 (hf_pipeline_tag, hf_likes, hf_downloads, hf_author, hf_library_name)
                result_item = HFSearchResultItem(
                    model_id=str(detail_result.get("hf_model_id", "")), # 修正: 对应 Pydantic 字段
                    pipeline_tag=str(detail_result.get("hf_pipeline_tag", "")), # 修正: 数据库列名
                    likes=int(detail_result.get("hf_likes", 0))
                    if detail_result.get("hf_likes") is not None
                    else None,
                    downloads=int(detail_result.get("hf_downloads", 0))
                    if detail_result.get("hf_downloads") is not None
                    else None,
                    last_modified=final_last_modified_dt,
                    score=score_float,
                    tags=processed_tags,
                    author=str(detail_result.get("hf_author", "")), # 修正: 数据库列名
                    library_name=str(detail_result.get("hf_library_name", "")), # 修正: 数据库列名
                )
                # 使用 hf_model_id (数据库中的真实ID) 作为 result_items_map 的键
                db_model_id = detail_result.get("hf_model_id")
                if db_model_id:
                     result_items_map[str(db_model_id)] = result_item
                     processed_count += 1
                else:
                    logger.warning(f"[_get_model_details_for_ids] Skipping item due to missing 'hf_model_id' in detail_result: {detail_result}")
                    creation_errors += 1
            except Exception as item_creation_error:
                logger.error(
                    f"[_get_model_details_for_ids] Error creating HFSearchResultItem for hf_model_id {detail_result.get('hf_model_id')}: {item_creation_error}",
                    exc_info=True,
                )
                creation_errors += 1
                continue  # Skip this model if item creation fails

        # Preserve original order using Faiss IDs, looking up in the potentially re-keyed map
        ordered_results = [
            result_items_map[mid] for mid in model_ids if mid in result_items_map
        ]
        logger.debug(
            f"[_get_model_details_for_ids] Completed processing. Processed: {processed_count}, Skipped (Not Found in Map): {skipped_count}, Creation Errors: {creation_errors}. Returning {len(ordered_results)} items."
        )
        return ordered_results

    def _filter_results_by_date(
        self,
        items: Sequence[ResultItem],  # Use Sequence for covariance
        published_after: Optional[date],
        published_before: Optional[date],
    ) -> List[ResultItem]:
        """Filter results by date range."""
        if not items:
            return []

        if not published_after and not published_before:
            return list(items)  # No filters to apply

        filtered_items = []
        for item in items:
            if not hasattr(item, "published_date"):
                continue  # Skip if item doesn't have a date (shouldn't happen for papers)

            item_date = getattr(item, "published_date")
            if not item_date:
                continue  # Skip if date is None

            # Apply date_from filter (inclusive)
            if published_after and item_date < published_after:
                continue

            # Apply date_to filter (inclusive)
            if published_before and item_date > published_before:
                continue

            filtered_items.append(item)

        return filtered_items

    def _apply_sorting_and_pagination(
        self,
        items: Sequence[ResultItem],  # Use Sequence
        sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]],
        sort_order: SortOrderLiteral,
        page: int,
        page_size: int,
    ) -> Tuple[List[ResultItem], int, int, int]:
        """Sorts and paginates a list of search results."""
        items_to_sort = list(items)  # Convert Sequence to list for sorting

        # --- Sorting ---
        if sort_by:
            reverse = sort_order == "desc"

            def get_sort_key(
                item: ResultItem,
            ) -> Any:  # Return Any for comparison flexibility
                # Define default values for None to ensure consistent sorting
                # Use values that place None appropriately (e.g., very early or very late)
                min_date = date.min
                min_datetime = datetime.min
                min_score = -math.inf
                min_int = -1
                min_str = ""

                try:
                    if isinstance(item, SearchResultItem):
                        if sort_by == "published_date":
                            return (
                                item.published_date if item.published_date else min_date
                            )
                        elif sort_by == "title":
                            return item.title.lower() if item.title else min_str
                        elif sort_by == "score":
                            # 返回元组 (score, paper_id) 以便在分数相同时稳定排序
                            primary_key = (
                                item.score if item.score is not None else min_score
                            )
                            secondary_key = (
                                item.paper_id
                            )  # 假设 paper_id 总是存在且可比较
                            return (primary_key, secondary_key)
                        else:
                            logger.warning(
                                f"Unsupported sort key '{sort_by}' for SearchResultItem."
                            )
                            return None  # Will be filtered out
                    elif isinstance(item, HFSearchResultItem):
                        if sort_by == "likes":
                            return item.likes if item.likes is not None else min_int
                        elif sort_by == "downloads":
                            return (
                                item.downloads
                                if item.downloads is not None
                                else min_int
                            )
                        elif sort_by == "last_modified":
                            return (
                                item.last_modified
                                if item.last_modified
                                else min_datetime
                            )
                        elif sort_by == "score":
                            # 返回元组 (score, model_id) 以便在分数相同时稳定排序
                            primary_key = (
                                item.score if item.score is not None else min_score
                            )
                            # Ignore Mypy error as it seems to be a false positive
                            secondary_key = (
                                item.model_id  # type: ignore[assignment]
                            )  # 假设 model_id 总是存在且可比较
                            return (primary_key, secondary_key)
                        else:
                            logger.warning(
                                f"Unsupported sort key '{sort_by}' for HFSearchResultItem."
                            )
                            return None  # Will be filtered out
                    else:
                        logger.warning(
                            f"Unsupported item type for sorting: {type(item)}"
                        )
                        return None
                except Exception as key_error:
                    logger.error(
                        f"Error getting sort key '{sort_by}' for item {getattr(item, 'paper_id', getattr(item, 'model_id', 'UNKNOWN'))}: {key_error}"
                    )
                    return None  # Treat error during key access as un-sortable

            try:
                # Filter out items where sort key is None before sorting
                valid_items = [
                    item for item in items_to_sort if get_sort_key(item) is not None
                ]
                if len(valid_items) < len(items_to_sort):
                    logger.warning(
                        f"Sorting skipped {len(items_to_sort) - len(valid_items)} items due to missing sort key '{sort_by}' or invalid type."
                    )

                # --- 添加调试日志：排序前 ---
                logger.debug(
                    f"_apply_sorting_and_pagination: Items BEFORE sort (sort_by='{sort_by}', reverse={reverse}): {[(getattr(i, 'paper_id', getattr(i, 'model_id', 'N/A')), getattr(i, 'score', None)) for i in valid_items]}"
                )
                # --- 结束调试日志 ---

                valid_items.sort(key=get_sort_key, reverse=reverse)

                # --- 添加调试日志：排序后 ---
                logger.debug(
                    f"_apply_sorting_and_pagination: Items AFTER sort: {[(getattr(i, 'paper_id', getattr(i, 'model_id', 'N/A')), getattr(i, 'score', None)) for i in valid_items]}"
                )
                # --- 结束调试日志 ---

                items_to_paginate = valid_items
            except TypeError as sort_error:
                logger.error(
                    f"TypeError during sorting by '{sort_by}': {sort_error}. Items might have incompatible types for comparison. Returning original order.",
                    exc_info=True,
                )
                items_to_paginate = items_to_sort  # Return original order on type error
            except Exception as e:
                logger.error(
                    f"Unexpected error during sorting by '{sort_by}': {e}",
                    exc_info=True,
                )
                items_to_paginate = (
                    items_to_sort  # Return original order on other errors
                )

        else:  # No sorting requested
            items_to_paginate = items_to_sort

        # --- Pagination ---
        total_items = len(items_to_paginate)
        skip = (page - 1) * page_size
        end_index = skip + page_size
        paginated_items = items_to_paginate[skip:end_index]

        logger.debug(
            f"Paginated {total_items} items to {len(paginated_items)} for page {page} (size {page_size})."
        )
        return (
            paginated_items,
            total_items,
            skip,
            page_size,
        )  # Return calculated skip/limit

    async def perform_semantic_search(
        self,
        query: str,
        target: SearchTarget,
        page: int = 1,
        page_size: int = 10,
        top_n: int = DEFAULT_TOP_N_SEMANTIC,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]] = "score",
        sort_order: SortOrderLiteral = "desc",
    ) -> PaginatedResult:
        """
        Performs semantic search using Faiss and fetches details.
        """
        # 添加日志
        logger.info(
            f"[perform_semantic_search] Target: {target}, Query: '{query}', Page: {page}, PageSize: {page_size}"
        )
        if self.embedder is None:
            logger.error(
                "[perform_semantic_search] Embedder not available."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Text embedding service is required for semantic search but is not available.",
            )
        
        # --- Check if embedder is available --- #
        if self.embedder is None:
            logger.error(
                "Semantic search requested, but embedder is not available in SearchService."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Text embedding service is required for semantic search but is not available.",
            )
        # --- End Check ---

        logger.info(
            f"Performing semantic search: query='{query}', target='{target}', "
            f"page={page}, page_size={page_size}, top_n={top_n}, sort='{sort_by}'"
        )

        # --- Basic Validation ---
        skip = (page - 1) * page_size
        if target == "all":
            logger.error(f"Semantic search target 'all' is not implemented yet.")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )
        if target not in get_args(SearchTarget):
            logger.error(f"Invalid search target '{target}'.")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- Pre-define target-specific configurations ---
        target_configs = {
            "papers": {
                "faiss_repo": self.faiss_repo_papers,
                "fetch_details_func": self._get_paper_details_for_ids,
                "id_type": int,
                "ResultModel": SearchResultItem,
                "PaginatedModel": PaginatedPaperSearchResult,
                "sort_options": get_args(PaperSortByLiteral),
                "EmptyModel": PaginatedPaperSearchResult,
            },
            "models": {
                "faiss_repo": self.faiss_repo_models,
                "fetch_details_func": self._get_model_details_for_ids,
                "id_type": str,
                "ResultModel": HFSearchResultItem,  # Use the imported class directly
                "PaginatedModel": PaginatedHFModelSearchResult,  # Use the imported class directly
                "sort_options": get_args(ModelSortByLiteral),
                "EmptyModel": PaginatedHFModelSearchResult,
            },
        }

        config = target_configs.get(target)
        if not config:
            logger.error(f"[perform_semantic_search] Invalid target: {target}")
            return PaginatedSemanticSearchResult(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        # Assign variables from config with explicit casting for type checker
        faiss_repo: FaissRepository = cast(FaissRepository, config["faiss_repo"])
        fetch_details_func: Callable[..., Coroutine[Any, Any, List[ResultItem]]] = cast(
            Callable[..., Coroutine[Any, Any, List[ResultItem]]],
            config["fetch_details_func"],
        )
        id_type: Type = cast(Type, config["id_type"])
        PaginatedModel: Type[PaginatedResult] = cast(Type[PaginatedResult], config["PaginatedModel"])
        target_sort_options: Tuple = cast(Tuple, config["sort_options"])
        EmptyModel: Type[PaginatedResult] = cast(Type[PaginatedResult], config["EmptyModel"])

        # --- Query Validation ---
        if not query:
            logger.warning("Semantic search attempted with empty query.")
            return EmptyModel(
                items=[], total=0, skip=skip, limit=page_size
            )  # Use specific EmptyModel

        # --- Embedding ---
        try:
            embedding = self.embedder.embed(query)
            if embedding is None:
                logger.error("[perform_semantic_search] Failed to generate embedding.")
                return EmptyModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)
            logger.debug("[perform_semantic_search] Embedding generated successfully.")
        except Exception as embed_error:
            logger.error(f"[perform_semantic_search] Error generating embedding: {embed_error}", exc_info=True)
            return PaginatedSemanticSearchResult(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        # --- Faiss Repo Check ---
        if not faiss_repo.is_ready():
            logger.error(f"[perform_semantic_search] Faiss repository for target '{target}' is not ready.")
            return PaginatedModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        # --- Validate Faiss ID Type ---
        if getattr(faiss_repo, "id_type", None) != id_type.__name__:
            logger.error(
                f"[perform_semantic_search] Mismatched ID types: Faiss repo for '{target}' expects '{getattr(faiss_repo, 'id_type', 'UNKNOWN')}' but service configured for '{id_type.__name__}'"
            )
            return PaginatedModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        # --- Perform Faiss Search ---
        search_results_raw: List[Tuple[FaissID, float]] = []
        try:
            search_results_raw = await faiss_repo.search_similar(embedding, k=top_n)
            # 添加日志记录原始结果
            logger.debug(
                f"[perform_semantic_search] Faiss search for '{target}' returned {len(search_results_raw)} raw results. Sample: {search_results_raw[:5]}"
            )
        except Exception as e:
            logger.error(
                f"[perform_semantic_search] Error during Faiss search for target '{target}': {e}", exc_info=True
            )
            return PaginatedModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        # --- Process Faiss Results ---
        if not search_results_raw:
            logger.info(f"[perform_semantic_search] Faiss search for '{target}' returned no results.")
            return PaginatedModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        result_ids: list = []
        result_scores: dict = {}
        for original_id_union, distance in search_results_raw:
            # Validate ID type against the expected type from config
            if isinstance(original_id_union, id_type):
                original_id = cast(Union[int, str], original_id_union)
                score = self._convert_distance_to_score(distance)
                result_ids.append(original_id)
                result_scores[original_id] = score
            else:
                logger.warning(
                    f"[perform_semantic_search] Skipping Faiss result. Expected ID type '{id_type.__name__}', got {type(original_id_union)} for ID {original_id_union}"
                )

        if not result_ids:
            logger.warning(
                "[perform_semantic_search] No valid IDs extracted from Faiss results after type check."
            )
            return PaginatedModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)
        
        # 添加日志记录提取的ID
        logger.debug(f"[perform_semantic_search] Extracted {len(result_ids)} valid IDs for target '{target}'. Sample: {result_ids[:10]}")

        # --- Fetch Details from Postgres ---
        all_items_list: List[ResultItem] = []
        try:
            logger.debug(f"[perform_semantic_search] Fetching details for {len(result_ids)} IDs using {fetch_details_func.__name__}...")
            all_items_list = await fetch_details_func(result_ids, result_scores)
            # 添加日志记录获取到的详情数量
            logger.debug(f"[perform_semantic_search] Fetched {len(all_items_list)} items with details for target '{target}'.")
        except Exception as fetch_error:
            logger.error(
                f"[perform_semantic_search] Error fetching details for target '{target}': {fetch_error}",
                exc_info=True,
            )
            return PaginatedModel(items=[], total=0, skip=(page - 1) * page_size, limit=page_size)

        # --- Filter by Date ---
        filtered_items: List[ResultItem] = self._filter_results_by_date(
            all_items_list, date_from, date_to
        )

        # --- Sort and Paginate ---
        # --- Validate Sort Key ---
        valid_sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]] = None
        if sort_by:
            if sort_by in target_sort_options:
                valid_sort_by = sort_by
            elif sort_by == "score":  # Score is always valid after semantic search
                valid_sort_by = "score"
            else:
                logger.warning(
                    f"[perform_semantic_search] Invalid sort_by key '{sort_by}' for target '{target}'. Defaulting to 'score'."
                )
                valid_sort_by = "score"

        paginated_items_list: List[ResultItem]
        total_items: int
        calculated_skip: int
        calculated_limit: int
        paginated_items_list, total_items, calculated_skip, calculated_limit = (
            self._apply_sorting_and_pagination(
                filtered_items,
                sort_by=valid_sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
            )
        )

        # --- Construct Final Response ---
        # Pass the original paginated list and cast for type checker.
        # We are confident based on the target that the items are of the correct type.
        # 添加日志记录最终返回数量
        logger.debug(f"[perform_semantic_search] Returning {len(paginated_items_list)} items for target '{target}' after filtering/sorting/pagination.")

        return PaginatedModel(
            items=cast(List[Any], paginated_items_list),
            total=total_items,
            skip=calculated_skip,
            limit=calculated_limit,
        )

    async def perform_keyword_search(
        self,
        query: str,
        target: SearchTarget,
        page: int = 1,
        page_size: int = 10,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        area: Optional[str] = None,  # Paper specific
        pipeline_tag: Optional[str] = None, # Model specific: 添加 pipeline_tag 参数
        # Allow target-specific sort options
        sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]] = None,
        sort_order: SortOrderLiteral = "desc",
    ) -> PaginatedResult:
        """
        Performs keyword search using Postgres capabilities.
        Postgres repo method handles filtering, sorting, and LIMIT/OFFSET.
        """
        # 添加日志
        logger.info(
            f"[perform_keyword_search] Target: {target}, Query: '{query}', Page: {page}, PageSize: {page_size}"
        )

        # --- Basic Validation ---
        skip = (page - 1) * page_size
        if target == "all":
            logger.error(f"Keyword search target 'all' is not implemented.")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )
        if target not in get_args(SearchTarget):
            logger.error(f"Invalid keyword search target '{target}'.")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )
        if not query:
            logger.warning("Keyword search attempted with empty query.")
            EmptyModel = (
                PaginatedPaperSearchResult
                if target == "papers"
                else PaginatedHFModelSearchResult
            )
            return EmptyModel(items=[], total=0, skip=skip, limit=page_size)

        # --- Assign target-specific functions, models, and sort options ---
        search_pg_func: Callable[..., Coroutine[Any, Any, Tuple[list, int]]]
        fetch_details_func: Callable[..., Coroutine[Any, Any, list]]
        PaginatedModel: Type[PaginatedResult]
        valid_pg_sort_by: Optional[Union[str]] = None  # Type PG repo expects
        target_sort_options: Tuple = ()

        if target == "papers":
            if not hasattr(self.pg_repo, "search_papers_by_keyword"):
                logger.error(
                    "[perform_keyword_search] PostgresRepository missing 'search_papers_by_keyword' method."
                )
                return PaginatedPaperSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )
            search_pg_func = self.pg_repo.search_papers_by_keyword
            fetch_details_func = self._get_paper_details_for_ids
            PaginatedModel = PaginatedPaperSearchResult
            target_sort_options = get_args(PaperSortByLiteral)
            # Default sort for paper keyword search
            if sort_by is None:
                sort_by = "published_date"
            # Validate sort_by for papers (PG repo might handle this, but good to check)
            if (
                sort_by not in target_sort_options or sort_by == "score"
            ):  # score not valid for PG keyword
                logger.warning(
                    f"[perform_keyword_search] Invalid or unsupported sort_by '{sort_by}' for paper keyword search. Using 'published_date'."
                )
                valid_pg_sort_by = "published_date"
            else:
                valid_pg_sort_by = cast(
                    Optional[Literal["published_date", "title", "paper_id"]],
                    sort_by,  # Added paper_id
                )  # Cast to valid PG keys

        elif target == "models":
            if not hasattr(self.pg_repo, "search_models_by_keyword"):
                logger.error(
                    "[perform_keyword_search] PostgresRepository missing 'search_models_by_keyword' method."
                )
                return PaginatedHFModelSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )
            search_pg_func = self.pg_repo.search_models_by_keyword
            fetch_details_func = self._get_model_details_for_ids
            PaginatedModel = PaginatedHFModelSearchResult
            target_sort_options = get_args(ModelSortByLiteral)
            # Default sort for model keyword search
            if sort_by is None:
                sort_by = "last_modified"  # Or perhaps 'likes' or 'downloads'?
            # Validate sort_by for models (PG repo might handle this)
            if (
                sort_by not in target_sort_options or sort_by == "score"
            ):  # score not valid for PG keyword
                logger.warning(
                    f"[perform_keyword_search] Invalid or unsupported sort_by '{sort_by}' for model keyword search. Using 'last_modified'."
                )
                valid_pg_sort_by = "last_modified"
            else:
                valid_pg_sort_by = cast(
                    Optional[Literal["likes", "downloads", "last_modified"]], sort_by
                )  # Cast to valid PG keys

        # --- Execute Keyword Search in PG (Fetches Details and Total Count) ---
        pg_results: List[
            Dict[str, Any]
        ] = []  # Results from PG repo (List[Dict] for both targets now)
        total_items: int = 0
        try:
            # 记录传递给PG repo的参数
            pg_params = {
                "query": query,
                "limit": page_size,
                "skip": skip,
                "sort_by": valid_pg_sort_by,
                "sort_order": sort_order,
                # Pass target-specific arguments, including pipeline_tag for models
                **({
                    "published_after": date_from,
                    "published_before": date_to,
                    "filter_area": area,
                 } if target == "papers" else
                 {
                     "pipeline_tag": pipeline_tag, # 添加 pipeline_tag 到模型参数
                 } if target == "models" else {})
            }
            logger.debug(f"[perform_keyword_search] Calling {search_pg_func.__name__} with params: {pg_params}")
            
            pg_results, total_items = await search_pg_func(**pg_params)
            
            # 记录从PG repo获取的结果
            logger.debug(f"[perform_keyword_search] PG search returned {len(pg_results)} raw items, total count: {total_items}")

        except Exception as search_error:
            logger.error(
                f"[perform_keyword_search] Error executing keyword search for target '{target}': {search_error}",
                exc_info=True,
            )
            # Return empty PaginatedModel based on target
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # --- Construct Final Response (Type Casting) ---
        if target == "papers":
            # Convert dicts to SearchResultItem
            try:
                # 处理可能仍是 JSON 字符串的字段，并处理单个转换错误
                paper_items: List[SearchResultItem] = []
                for item in pg_results:
                    # 处理可能是 JSON 字符串的作者字段
                    if "authors" in item and isinstance(item["authors"], str):
                        try:
                            item["authors"] = json.loads(item["authors"])
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"[perform_keyword_search] Could not decode authors JSON for paper_id {item.get('paper_id')}"
                            )
                            item["authors"] = []

                    # 创建 SearchResultItem 实例
                    try:
                        paper_items.append(SearchResultItem(**item))
                    except Exception as item_error:
                        logger.error(
                            f"[perform_keyword_search] Error creating SearchResultItem from dict: {item_error}\nData: {item}",
                            exc_info=True,
                        )
                        # 跳过不符合规范的结果项
                        continue

                # 返回正确类型的结果
                return PaginatedPaperSearchResult(
                    items=paper_items, total=total_items, skip=skip, limit=page_size
                )
            except Exception as e:
                logger.error(
                    f"[perform_keyword_search] Error converting paper keyword results to SearchResultItem: {e}",
                    exc_info=True,
                )
                return PaginatedPaperSearchResult(
                    items=[], total=total_items, skip=skip, limit=page_size
                )
        elif target == "models":
            # Convert dicts to HFSearchResultItem
            try:
                model_items: List[HFSearchResultItem] = []
                for item in pg_results:
                    try:
                        # Manually map DB fields to Pydantic model fields
                        # Provide default score for keyword search
                        
                        # Process tags safely (copied from _get_model_details_for_ids)
                        processed_tags: Optional[List[str]] = None
                        tags_list = item.get("hf_tags") # Use correct DB column name
                        if isinstance(tags_list, str):
                            try:
                                parsed_tags = json.loads(tags_list)
                                if isinstance(parsed_tags, list) and all(isinstance(t, str) for t in parsed_tags):
                                    processed_tags = parsed_tags
                            except json.JSONDecodeError:
                                logger.warning(f"[perform_keyword_search] Could not decode hf_tags JSON for model {item.get('hf_model_id')}")
                        elif isinstance(tags_list, list) and all(isinstance(t, str) for t in tags_list):
                             processed_tags = tags_list
                             
                        # Process last_modified safely (copied and adapted from _get_model_details_for_ids)
                        final_last_modified_dt: Optional[datetime] = None
                        last_modified_val = item.get("hf_last_modified") # Use correct DB column name
                        if isinstance(last_modified_val, (datetime, date)):
                             final_last_modified_dt = last_modified_val if isinstance(last_modified_val, datetime) else datetime.combine(last_modified_val, datetime.min.time())
                             # Ensure timezone aware
                             if final_last_modified_dt.tzinfo is None:
                                 final_last_modified_dt = final_last_modified_dt.replace(tzinfo=timezone.utc) 
                        elif isinstance(last_modified_val, str):
                            try:
                                dt_obj = datetime.fromisoformat(last_modified_val.replace("Z", "+00:00"))
                                if dt_obj.tzinfo is None:
                                     dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                                final_last_modified_dt = dt_obj
                            except ValueError:
                                logger.warning(f"[perform_keyword_search] Could not parse last_modified string '{last_modified_val}' for model {item.get('hf_model_id')}. Setting to None.")
                        
                        model_instance = HFSearchResultItem(
                            model_id=str(item.get("hf_model_id", "")), 
                            author=str(item.get("hf_author", "")),
                            pipeline_tag=str(item.get("hf_pipeline_tag", "")),
                            library_name=str(item.get("hf_library_name", "")),
                            tags=processed_tags,
                            likes=int(item["hf_likes"]) if item.get("hf_likes") is not None else None,
                            downloads=int(item["hf_downloads"]) if item.get("hf_downloads") is not None else None,
                            last_modified=final_last_modified_dt,
                            score=0.0, # Assign default score for keyword results
                            # sha=item.get("hf_sha") # sha is not in HFSearchResultItem model
                        )
                        model_items.append(model_instance)
                    except ValidationError as val_err:
                         logger.error(
                            f"[perform_keyword_search] Validation error creating HFSearchResultItem: {val_err}\nData: {item}",
                            exc_info=False # Don't need full stack usually
                        )
                         continue # Skip invalid item
                    except Exception as item_error:
                        logger.error(
                            f"[perform_keyword_search] Unexpected error creating HFSearchResultItem from dict: {item_error}\nData: {item}",
                            exc_info=True,
                        )
                        continue

                # Return typed results
                return PaginatedHFModelSearchResult(
                    items=model_items, total=total_items, skip=skip, limit=page_size
                )
            except Exception as e:
                logger.error(
                    f"[perform_keyword_search] Error converting model keyword results to HFSearchResultItem: {e}",
                    exc_info=True,
                )
                return PaginatedHFModelSearchResult(
                    items=[], total=total_items, skip=skip, limit=page_size
                )
        else:
            # Fallback for unexpected target (should not happen due to earlier checks)
            logger.error(f"[perform_keyword_search] Keyword search reached end with unexpected target: {target}")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

    async def perform_hybrid_search(
        self,
        query: str,
        target: SearchTarget = "papers",  # 当前仅支持论文
        page: int = 1,
        page_size: int = 10,
        filters: Optional[SearchFilterModel] = None,
    ) -> PaginatedPaperSearchResult:
        """
        执行混合搜索（同时结合语义和关键词搜索）

        Args:
            query: 搜索查询字符串
            target: 搜索目标类型（当前仅支持 papers）
            page: 页码，从1开始
            page_size: 每页结果数量
            filters: 可选的过滤条件

        Returns:
            PaginatedPaperSearchResult: 包含搜索结果的分页模型
        """
        logger.info(
            f"执行混合搜索: 查询='{query}', 目标='{target}', 页码={page}, 页大小={page_size}"
        )

        # 计算分页参数
        skip = (page - 1) * page_size

        # 处理空查询
        if not query.strip():
            return PaginatedPaperSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # 提取过滤条件
        published_after = None
        published_before = None
        filter_area = None
        sort_by = None
        sort_order = "desc"

        if filters:
            published_after = filters.published_after
            published_before = filters.published_before
            filter_area = filters.filter_area
            sort_by = filters.sort_by
            if filters.sort_order:
                sort_order = filters.sort_order

        # --- Step 1: 执行语义搜索 ---
        semantic_results_map: Dict[int, float] = {}
        try:
            # 检查嵌入器是否可用
            if self.embedder is None:
                logger.warning("嵌入器不可用，无法执行语义搜索部分")
            else:
                # 生成嵌入向量
                embedding = self.embedder.embed(query)
                if embedding is not None:
                    # 执行相似度搜索
                    semantic_results = await self.faiss_repo_papers.search_similar(
                        embedding,
                        k=30,  # 获取更多结果，以便后续排序和过滤
                    )
                    # 转换为ID -> 分数的映射
                    semantic_results_map = {
                        int(paper_id): self._convert_distance_to_score(distance)
                        for paper_id, distance in semantic_results
                        if isinstance(paper_id, (int, str))
                    }
        except Exception as e:
            logger.error(f"Hybrid: Error in semantic search: {e}", exc_info=True)

        # --- Step 2: 执行关键词搜索 ---
        keyword_results_map: Dict[int, Dict[str, Any]] = {}
        try:
            keyword_results, _ = await self.pg_repo.search_papers_by_keyword(
                query=query,
                limit=30,  # 同样获取更多结果
                skip=0,
                sort_by="published_date",
                sort_order="desc",
                published_after=published_after,
                published_before=published_before,
                filter_area=filter_area,
            )

            # 转换为ID -> 详情的映射
            keyword_results_map = {
                int(result.get("paper_id", 0)): result
                for result in keyword_results
                if result.get("paper_id") is not None
            }
        except Exception as e:
            logger.error(f"Hybrid: Error in keyword search: {e}", exc_info=True)

        # --- Step 3: 合并结果ID ---
        semantic_ids = set(semantic_results_map.keys())
        keyword_ids = set(keyword_results_map.keys())
        all_combined_ids: Set[int] = semantic_ids.union(keyword_ids)
        # 记录合并后的初始唯一ID总数 (潜在总数)
        initial_unique_ids_count = len(all_combined_ids)

        if not all_combined_ids:
            logger.info(
                "Hybrid search: No results from either semantic or keyword searches."
            )
            return PaginatedPaperSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- Step 4: 获取所有论文详情 ---
        all_paper_details: Dict[int, Dict[str, Any]] = {}
        try:
            # 获取语义搜索结果的详情（这些详情会从数据库获取）
            paper_details_list = await self.pg_repo.get_papers_details_by_ids(
                list(all_combined_ids)
            )
            # 创建ID到详情的映射
            all_paper_details = {
                int(details.get("paper_id", 0)): details
                for details in paper_details_list
                if details.get("paper_id") is not None
            }
        except Exception as fetch_error:
            logger.error(
                f"Hybrid: Error fetching paper details: {fetch_error}", exc_info=True
            )
            if not all_paper_details:
                return PaginatedPaperSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )

        # --- Step 5: Combine results using Reciprocal Rank Fusion (RRF) ---
        # (确保 RRF 和分数计算只针对有详情的 ID)
        valid_ids_with_details = set(all_paper_details.keys())

        # Semantic Ranks (only for valid IDs)
        semantic_ranks: Dict[int, int] = {}
        if semantic_results_map:
            valid_semantic_results = {
                pid: score
                for pid, score in semantic_results_map.items()
                if pid in valid_ids_with_details
            }
            semantic_ranks = {
                pid: rank + 1
                for rank, (pid, _) in enumerate(
                    sorted(
                        valid_semantic_results.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
            }

        # Keyword Ranks (only for valid IDs)
        keyword_ranks: Dict[int, int] = {}
        if keyword_results_map:
            valid_keyword_results = {
                pid: details
                for pid, details in keyword_results_map.items()
                if pid in valid_ids_with_details
            }
            keyword_ranks = {
                int(pid): rank + 1
                for rank, pid in enumerate(
                    valid_keyword_results.keys()
                )  # 使用有效 ID 列表
            }

        # 合并所有ID并计算RRF分数 (只针对有详情的 ID)
        combined_scores: Dict[int, Optional[float]] = {}
        # all_combined_ids_with_details = set(all_paper_details.keys()) # 重复，使用 valid_ids_with_details

        # 检查是否仅有关键词搜索结果 (基于有效 ID)
        is_keyword_only = bool(keyword_ranks) and not semantic_ranks

        # RRF参数k
        rrf_k = self.DEFAULT_RRF_K  # 默认RRF k参数

        for paper_id in valid_ids_with_details:  # 只迭代有详情的 ID
            score = 0.0
            sem_rank = semantic_ranks.get(paper_id)
            kw_rank = keyword_ranks.get(paper_id)

            # RRF公式
            if sem_rank is not None:
                score += 1.0 / (rrf_k + sem_rank)
            if kw_rank is not None:
                score += 1.0 / (rrf_k + kw_rank)

            # 存储合并分数
            if score > 0:
                if is_keyword_only:
                    combined_scores[paper_id] = None
                else:
                    combined_scores[paper_id] = score

        # --- Step 6: 创建SearchResultItem对象 --- (只为有详情的 ID 创建)
        all_items: List[SearchResultItem] = []
        for paper_id in valid_ids_with_details:  # 只迭代有详情的 ID
            if details := all_paper_details.get(paper_id):
                # 处理可能是JSON字符串的作者字段
                authors = details.get("authors", [])
                if isinstance(authors, str):
                    try:
                        authors = json.loads(authors)
                    except (json.JSONDecodeError, TypeError):
                        authors = []

                try:
                    item = SearchResultItem(
                        paper_id=paper_id,
                        pwc_id=details.get("pwc_id", ""),
                        title=details.get("title", ""),
                        summary=details.get("summary", ""),
                        score=combined_scores.get(paper_id),
                        pdf_url=details.get("pdf_url", ""),
                        published_date=details.get("published_date"),
                        authors=authors,
                        area=details.get("area", ""),
                    )
                    all_items.append(item)
                except ValidationError as ve:
                    logger.error(
                        f"[perform_hybrid_search] Validation error creating SearchResultItem for paper_id={paper_id}: {ve}"
                    )
                except Exception as e:
                    logger.error(
                        f"[perform_hybrid_search] Error creating SearchResultItem for paper_id={paper_id}: {e}"
                    )

        # --- Step 7: 应用过滤器 --- (过滤 all_items)
        filtered_items: List[SearchResultItem] = all_items

        # 日期过滤
        if published_after or published_before:
            filtered_items = [
                item
                for item in filtered_items
                if item.published_date is not None
                and (published_after is None or item.published_date >= published_after)
                and (
                    published_before is None or item.published_date <= published_before
                )
            ]

        # 领域过滤
        if filter_area:
            filtered_items = [
                item
                for item in filtered_items
                if item.area and item.area.lower() == filter_area.lower()
            ]

        # --- Step 8: 排序和分页 ---
        total_items_after_filtering = len(filtered_items)

        # Determine sort key, default to 'score' if not provided or invalid
        final_sort_by: Optional[PaperSortByLiteral] = None
        current_sort_by_from_filter = filters.sort_by if filters else None

        if current_sort_by_from_filter:
            if current_sort_by_from_filter in get_args(PaperSortByLiteral):
                final_sort_by = cast(PaperSortByLiteral, current_sort_by_from_filter)
            else:
                logger.warning(
                    f"[perform_hybrid_search] Invalid sort_by '{current_sort_by_from_filter}' in filter for hybrid paper search. Defaulting to 'score'."
                )
                final_sort_by = "score"
        else:  # No sort specified in filter, default to score
            final_sort_by = "score"

        # Determine sort order
        final_sort_order: SortOrderLiteral = "desc"  # Default desc
        if (
            filters
            and filters.sort_order
            and filters.sort_order in get_args(SortOrderLiteral)
        ):
            final_sort_order = filters.sort_order
        # Ensure final_sort_order is always a valid Literal for the function call
        final_sort_order = cast(SortOrderLiteral, final_sort_order)

        (
            paginated_items_list_uncasted,
            total_items,
            calculated_skip,
            calculated_limit,
        ) = self._apply_sorting_and_pagination(
            filtered_items,
            sort_by=final_sort_by,  # Now defaults to 'score'
            sort_order=final_sort_order,  # Use determined sort order
            page=page,
            page_size=page_size,
        )

        paginated_items_list = cast(
            List[SearchResultItem], paginated_items_list_uncasted
        )

        # --- Step 9: 返回分页结果 ---
        # 修改返回类型为 PaginatedPaperSearchResult
        return PaginatedPaperSearchResult(
            items=paginated_items_list,
            total=total_items_after_filtering,  # 使用 *过滤后* (分页前) 的总数
            skip=calculated_skip,
            limit=calculated_limit,
        )
