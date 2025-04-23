# aigraphx/services/search_service.py

# 文件作用说明：
# 该文件定义了 `SearchService` 类，它是 AIGraphX 后端的核心服务层之一。
# 主要职责是处理来自 API 端点（`aigraphx/api/v1/endpoints/search.py`）的搜索请求。
# 它封装了执行不同类型搜索（语义、关键词、混合）的逻辑，并协调与各个数据仓库（PostgreSQL, Faiss, Neo4j）
# 以及文本嵌入器（`aigraphx/vectorization/embedder.py`）的交互。
#
# 主要交互对象：
# - API 端点 (`aigraphx/api/v1/endpoints/search.py`): 接收来自此处的搜索请求。
# - PostgreSQL 仓库 (`aigraphx/repositories/postgres_repo.py`): 用于关键词搜索和获取论文/模型详细信息。
# - Faiss 仓库 (`aigraphx/repositories/faiss_repo.py`): 用于语义搜索（相似度查找）。
# - Neo4j 仓库 (`aigraphx/repositories/neo4j_repo.py`): (可选) 用于图相关的搜索或信息补充。
# - 文本嵌入器 (`aigraphx/vectorization/embedder.py`): 用于将文本查询转换为向量以进行语义搜索。
# - Pydantic 模型 (`aigraphx/models/search.py`): 用于定义搜索结果的数据结构和分页。
# - 依赖注入 (`aigraphx/api/v1/dependencies.py`): 该文件中的 `get_search_service` 函数负责创建 `SearchService` 实例并注入到 API 端点。
#
# 核心功能：
# 1.  **语义搜索 (`perform_semantic_search`)**:
#     - 接收查询文本和目标类型（论文/模型）。
#     - 使用嵌入器将查询文本转换为向量。
#     - 在对应的 Faiss 索引中查找相似向量。
#     - 从 PostgreSQL 获取匹配 ID 的详细信息。
#     - 排序和分页后返回结果。
# 2.  **关键词搜索 (`perform_keyword_search`)**:
#     - 接收查询文本、目标类型和各种过滤器。
#     - 调用 PostgreSQL 仓库执行基于文本匹配的搜索（如 `tsvector`）。
#     - PostgreSQL 仓库直接返回过滤、排序和分页后的结果。
#     - 将 PG 返回的字典列表转换为 Pydantic 模型列表。
# 3.  **混合搜索 (`perform_hybrid_search`)**:
#     - 同时执行语义搜索和关键词搜索。
#     - 使用 Reciprocal Rank Fusion (RRF) 算法合并两种搜索的结果，并重新计算排名分数。
#     - 获取合并后 ID 的详细信息。
#     - 应用过滤器、排序和分页后返回结果。
# 4.  **辅助方法**: 包括距离到分数的转换、根据 ID 获取详情、结果过滤、排序和分页等。

# 导入标准库
import logging  # 用于日志记录，帮助调试和跟踪程序运行状态
import numpy as np  # 虽然在当前代码中没有直接显式使用，但 Faiss 内部可能依赖它处理向量
import asyncio  # 用于支持异步操作，例如数据库查询和 Faiss 搜索
import math  # 用于数学运算，例如在排序时处理无穷大或在 RRF 中计算
import json  # 用于处理 JSON 数据，例如解析存储在数据库中的 JSON 字符串 (tags, authors)
from datetime import date, datetime, timezone  # 用于处理日期和时间，例如过滤发布日期
from typing import (  # 用于类型提示，增强代码可读性和健壮性
    List,  # 表示列表类型，例如 List[int] 表示整数列表
    Dict,  # 表示字典类型，例如 Dict[str, float] 表示字符串到浮点数的映射
    Optional,  # 表示一个值可以是指定类型，也可以是 None
    Literal,  # 表示一个变量只能是指定的几个字符串或值之一
    Union,  # 表示一个变量可以是多种指定类型之一
    Tuple,  # 表示元组类型
    Set,  # 表示集合类型
    TypeAlias,  # 用于创建类型别名，提高可读性
    Type,  # 表示类型本身，例如 Type[int]
    cast,  # 用于告诉类型检查器一个变量的确定类型，即使它不能被静态推断出来
    get_args,  # 用于获取 Literal 或 Union 等类型的参数
    Callable,  # 表示可调用对象，例如函数
    Coroutine,  # 表示协程对象 (async def 函数返回的类型)
    Any,  # 表示任意类型
    Sequence,  # 表示序列类型 (如 list, tuple)，支持协变性 (covariance)
)

# 导入第三方库
from fastapi import HTTPException, status  # FastAPI 框架的组件，HTTPException 用于主动返回 HTTP 错误响应，status 包含 HTTP 状态码常量
from pydantic import ValidationError  # Pydantic 库的组件，用于数据验证错误

# 导入项目内部组件
from aigraphx.repositories.postgres_repo import PostgresRepository  # 导入 PostgreSQL 数据仓库类
from aigraphx.repositories.faiss_repo import FaissRepository  # 导入 Faiss 数据仓库类
from aigraphx.vectorization.embedder import TextEmbedder  # 导入文本嵌入器类
from aigraphx.models.search import (  # 导入搜索相关的 Pydantic 模型
    SearchResultItem,  # 单个论文搜索结果项的模型
    HFSearchResultItem,  # 单个 Hugging Face 模型搜索结果项的模型
    PaginatedPaperSearchResult,  # 分页后的论文搜索结果模型
    PaginatedSemanticSearchResult,  # 通用的分页语义搜索结果模型 (或用于错误情况)
    PaginatedHFModelSearchResult,  # 分页后的 Hugging Face 模型搜索结果模型
    AnySearchResultItem,  # (未在当前代码中使用) 联合类型，表示任何类型的搜索结果项
    PaginatedModel,  # (未在当前代码中使用) 通用的分页模型
    SearchFilterModel,  # 搜索过滤条件的模型
)
from aigraphx.repositories.neo4j_repo import Neo4jRepository  # 导入 Neo4j 数据仓库类

# --- 类型别名 (Type Aliases) ---
# 使用 TypeAlias 可以让复杂的类型注解更清晰易懂

# 定义搜索目标的类型别名，只能是 "papers", "models", 或 "all"
SearchTarget: TypeAlias = Literal["papers", "models", "all"]

# 定义论文搜索结果排序依据的类型别名
PaperSortByLiteral: TypeAlias = Literal["score", "published_date", "title"]
# 定义模型搜索结果排序依据的类型别名
ModelSortByLiteral: TypeAlias = Literal["score", "likes", "downloads", "last_modified"]
# 定义排序顺序的类型别名，只能是 "asc" (升序) 或 "desc" (降序)
SortOrderLiteral: TypeAlias = Literal["asc", "desc"]

# 定义 Faiss 索引中 ID 的类型别名，可以是整数 (论文) 或字符串 (模型)
FaissID: TypeAlias = Union[int, str]
# 定义单个搜索结果项的类型别名，可以是论文或模型
ResultItem: TypeAlias = Union[SearchResultItem, HFSearchResultItem]
# 定义分页后的搜索结果的类型别名，可以是论文、模型或通用/错误类型
PaginatedResult: TypeAlias = Union[
    PaginatedPaperSearchResult,
    PaginatedHFModelSearchResult,
    PaginatedSemanticSearchResult,  # 用于通用情况或发生错误时的返回类型
]

# 获取当前模块的日志记录器实例
# 使用 `__name__` 可以让日志记录器知道这条日志是从哪个模块发出的
logger = logging.getLogger(__name__)


class SearchService:
    """
    搜索服务类。
    提供了使用语义、关键词和混合方法搜索论文和模型的功能。
    这是应用程序核心业务逻辑的一部分，负责协调数据访问和搜索算法。
    """

    # RRF (Reciprocal Rank Fusion) 算法中的 k 参数，用于调整融合结果中排名靠后项的权重。
    # 较大的 k 值意味着排名靠后的结果影响更大。
    DEFAULT_RRF_K: int = 60
    # 语义搜索默认从 Faiss 获取的 top_n 结果数量
    DEFAULT_TOP_N_SEMANTIC: int = 100
    # 关键词搜索默认从 Postgres 获取的 top_n 结果数量 (注意：实际分页在 PG 完成，这里可能更多是用于混合搜索的初始获取量)
    DEFAULT_TOP_N_KEYWORD: int = 100

    def __init__(
        self,
        embedder: Optional[TextEmbedder],  # 文本嵌入器实例，用于语义搜索
        faiss_repo_papers: FaissRepository,  # 论文 Faiss 仓库实例
        faiss_repo_models: FaissRepository,  # 模型 Faiss 仓库实例
        pg_repo: PostgresRepository,  # PostgreSQL 仓库实例
        neo4j_repo: Optional[Neo4jRepository],  # Neo4j 仓库实例 (可选)
    ):
        """
        初始化 SearchService。

        这个方法在服务实例化时被调用 (通常由依赖注入系统完成)。
        它接收所有必需的依赖项 (仓库、嵌入器) 并将它们存储为实例属性，
        以便在服务的其他方法中使用。

        Args:
            embedder: 用于将文本转换为向量的嵌入器。如果为 None，则无法进行语义搜索。
            faiss_repo_papers: 用于搜索论文向量索引的 Faiss 仓库。
            faiss_repo_models: 用于搜索模型向量索引的 Faiss 仓库。
            pg_repo: 用于访问 PostgreSQL 数据库（关键词搜索、获取详情）的仓库。
            neo4j_repo: 用于访问 Neo4j 图数据库的仓库。如果为 None，图相关功能将不可用。
        """
        # 将传入的依赖项赋值给实例变量
        self.embedder = embedder
        self.faiss_repo_papers = faiss_repo_papers
        self.faiss_repo_models = faiss_repo_models
        self.pg_repo = pg_repo
        self.neo4j_repo = neo4j_repo
        # 记录服务初始化信息
        logger.info("SearchService 初始化完成。")
        # 如果 Neo4j 仓库未提供，则记录警告信息
        if self.neo4j_repo is None:
            logger.warning(
                "SearchService 初始化时未提供 Neo4j 仓库。图数据库相关功能可能不可用。"
            )

    def _convert_distance_to_score(self, distance: float) -> float:
        """
        将 Faiss 返回的距离（通常是 L2 距离，非负）转换为相似度得分（范围在 (0, 1]）。
        距离越小，表示向量越相似，得分越高。

        使用公式: score = 1 / (1 + distance)

        Args:
            distance: Faiss 返回的距离值。

        Returns:
            相似度得分，值在 (0, 1] 之间。
        """
        # Faiss 的距离通常不应为负数，如果遇到负数，记录警告并将其视为 0
        if distance < 0:
            logger.warning(f"接收到负的 Faiss 距离: {distance}。将其修正为 0。")
            distance = 0.0
        # 加上一个很小的正数 (epsilon) 以防止分母为零 (当 distance 为 -1 时理论上可能，虽然不应发生)
        # 并确保分数总是大于 0
        return 1.0 / (1.0 + distance + 1e-9)

    async def _get_paper_details_for_ids(
        self, paper_ids: List[int], scores: Optional[Dict[int, Optional[float]]] = None
    ) -> List[SearchResultItem]:
        """
        一个内部辅助方法，用于从 PostgreSQL 数据库批量获取指定 ID 列表的论文详细信息。

        Args:
            paper_ids: 一个包含论文 ID (整数) 的列表。
            scores: 一个可选的字典，将论文 ID 映射到它们的分数 (例如来自语义搜索)。
                    分数可以是浮点数，也可以是 None (例如混合搜索中仅关键词匹配的情况)。

        Returns:
            一个包含 `SearchResultItem` 对象的列表，每个对象代表一篇论文的详细信息。
            列表的顺序与输入的 `paper_ids` 顺序一致。如果某个 ID 找不到详情，则不会包含在结果中。
        """
        # 如果输入的 ID 列表为空，直接返回空列表
        if not paper_ids:
            return []

        try:
            # 调用 PostgreSQL 仓库的方法来获取论文详情
            # `get_papers_details_by_ids` 应该返回一个字典列表，每个字典包含一篇论文的字段
            paper_details_list = await self.pg_repo.get_papers_details_by_ids(paper_ids)

            # 如果数据库没有返回任何详情，也返回空列表
            if not paper_details_list:
                return []

            # 为了方便查找和保持顺序，创建一个从论文 ID 到其详情字典的映射
            paper_details_map = {
                # 使用 .get() 安全地获取 paper_id，避免因缺少键而引发 KeyError
                details.get("paper_id"): details
                for details in paper_details_list
            }

            # 初始化结果列表
            results: List[SearchResultItem] = []

            # 遍历输入的 paper_ids 列表，以确保结果顺序与输入一致
            for paper_id in paper_ids:
                # 从映射中查找当前 paper_id 的详情
                if details := paper_details_map.get(paper_id):
                    # 如果详情存在，则尝试构建 SearchResultItem 对象

                    # 获取该论文的分数（如果提供了 scores 字典）
                    score = None  # 默认为 None
                    if scores is not None:
                        # 从 scores 字典中获取分数，如果 ID 不存在则返回 None
                        score = scores.get(paper_id)

                    # 尝试使用 Pydantic 模型创建 SearchResultItem 实例
                    # Pydantic 会自动进行数据类型验证和转换
                    try:
                        item = SearchResultItem(
                            paper_id=paper_id,
                            pwc_id=details.get("pwc_id", ""),  # 使用 .get 提供默认值
                            title=details.get("title", ""),
                            summary=details.get("summary", ""),
                            score=score,  # 允许 score 为 None
                            pdf_url=details.get("pdf_url", ""),
                            published_date=details.get("published_date"),  # 直接传递日期对象
                            authors=details.get(
                                "authors", []
                            ),  # 获取作者列表，默认为空列表
                            area=details.get("area", ""),
                        )
                        # 将成功创建的 item 添加到结果列表
                        results.append(item)
                    except ValidationError as ve:
                        # 如果 Pydantic 验证失败 (例如类型不匹配)，记录错误日志
                        # 但不中断整个过程，继续处理下一个论文 ID
                        logger.error(
                            f"为 paper_id={paper_id} 创建 SearchResultItem 时发生验证错误: {ve}"
                        )
                        # 可以选择在这里添加更多调试信息，例如打印 'details' 字典的内容

            # 返回包含所有成功创建的 SearchResultItem 的列表
            return results

        except Exception as e:
            # 如果在数据库查询或处理过程中发生任何其他异常，记录错误日志
            logger.error(
                f"获取论文详情时出错: {str(e)}", exc_info=True
            )  # exc_info=True 会记录完整的堆栈跟踪信息
            # 在出错时返回空列表，避免将错误传递给上层调用者
            return []

    async def _get_model_details_for_ids(
        self, model_ids: List[str], scores: Optional[Dict[str, float]] = None
    ) -> List[HFSearchResultItem]:
        """
        一个内部辅助方法，用于从 PostgreSQL 数据库批量获取指定 Hugging Face 模型 ID 列表的模型详细信息。

        Args:
            model_ids: 一个包含 Hugging Face 模型 ID (字符串) 的列表。这些 ID 通常来自 Faiss 搜索。
            scores: 一个可选的字典，将模型 ID 映射到它们的分数 (例如来自语义搜索)。

        Returns:
            一个包含 `HFSearchResultItem` 对象的列表，每个对象代表一个模型的详细信息。
            列表的顺序与输入的 `model_ids` 顺序一致。如果某个 ID 找不到详情，则不会包含在结果中。
        """
        # 记录接收到的 ID 数量和前几个 ID 示例
        logger.debug(
            f"[_get_model_details_for_ids] 收到 {len(model_ids)} 个模型 ID: {model_ids[:10]}..."
        )
        # 如果输入的 ID 列表为空，直接返回空列表
        if not model_ids:
            logger.debug("[_get_model_details_for_ids] 没有提供模型 ID，返回空列表。")
            return []

        # 记录将要从 PG 获取详情的 ID 数量
        logger.debug(
            f"[_get_model_details_for_ids] 正在从 PG 获取 {len(model_ids)} 个模型 ID 的详情。"
        )
        # 初始化用于存储结果的字典 (ID -> HFSearchResultItem 对象)
        result_items_map: Dict[str, HFSearchResultItem] = {}
        # 初始化用于存储从数据库获取的原始详情列表
        model_details_list: List[Dict[str, Any]] = []

        # 检查 PostgresRepository 是否有预期的 'get_hf_models_by_ids' 方法
        # 这是一个防御性编程措施，防止因方法名更改或缺失导致运行时错误
        if not hasattr(self.pg_repo, "get_hf_models_by_ids"):
            logger.error(
                "[_get_model_details_for_ids] PostgresRepository 缺少 'get_hf_models_by_ids' 方法。"
            )
            return []

        try:
            # 调用 PostgreSQL 仓库的方法来获取模型详情
            model_details_list = await self.pg_repo.get_hf_models_by_ids(model_ids)
            # 记录从 PG 获取到的记录数量
            logger.debug(
                f"[_get_model_details_for_ids] 从 PG 获取了 {len(model_details_list)} 条详情记录。"
            )
        except Exception as e:
            # 如果数据库查询失败，记录错误并返回空列表
            logger.error(
                f"[_get_model_details_for_ids] 从 PG 获取详情失败: {e}", exc_info=True
            )
            return []

        # 如果数据库没有返回任何详情，记录警告并返回空列表
        if not model_details_list:
            logger.warning(
                "[_get_model_details_for_ids] PG 没有为给定的模型 ID 返回任何详情。"
            )
            return []

        # 创建一个从模型 ID 到其详情字典的映射，方便后续查找
        # 使用数据库中的 'hf_model_id' 列作为字典的键
        details_map = {
            # 安全地获取 hf_model_id，如果不存在则该项不会被加入字典
            detail.get("hf_model_id"): detail
            for detail in model_details_list
            if detail.get("hf_model_id")
        }
        # 记录创建的映射的大小
        logger.debug(
            f"[_get_model_details_for_ids] 创建了包含 {len(details_map)} 个条目的 details_map。"
        )

        # 初始化计数器，用于统计处理结果
        processed_count = 0  # 成功处理的模型数量
        skipped_count = 0  # 因在 details_map 中未找到而跳过的模型数量
        creation_errors = 0  # 创建 HFSearchResultItem 对象时出错的数量

        # 遍历输入的 model_ids 列表 (这些 ID 来自 Faiss)
        for model_id in model_ids:
            # 在 details_map 中查找当前模型 ID 的详情
            detail_result = details_map.get(model_id)
            # 如果找不到详情
            if detail_result is None:
                # 记录警告，这可能表示 Faiss 索引和数据库之间存在不一致
                logger.warning(
                    f"[_get_model_details_for_ids] 在 details_map 中找不到 model_id {model_id} (来自 Faiss) 的详情。可能表示 Faiss 和数据库不一致。"
                )
                skipped_count += 1
                continue  # 跳过处理下一个 ID

            # --- 安全地处理各个字段 ---

            # 处理 tags 字段 (可能存储为 JSON 字符串或已经是列表)
            processed_tags: Optional[List[str]] = None  # 初始化为 None
            tags_list = detail_result.get("tags")  # 获取原始 tags 数据
            if isinstance(tags_list, str):  # 如果是字符串，尝试解析 JSON
                try:
                    parsed_tags = json.loads(tags_list)
                    # 确保解析结果是字符串列表
                    if isinstance(parsed_tags, list) and all(
                        isinstance(t, str) for t in parsed_tags
                    ):
                        processed_tags = parsed_tags
                    else:
                        logger.warning(
                            f"[_get_model_details_for_ids] 模型 {model_id} 解码后的 tags 不是字符串列表。"
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"[_get_model_details_for_ids] 解码模型 ID {model_id} 的 tags JSON 失败: {tags_list}"
                    )
            elif isinstance(tags_list, list) and all(
                isinstance(t, str) for t in tags_list
            ):
                # 如果已经是字符串列表，直接使用
                processed_tags = tags_list

            # 获取分数 (如果提供了 scores 字典)，默认为 0.0
            score_float = scores.get(model_id, 0.0) if scores else 0.0

            # 处理 last_modified 字段 (可能是 datetime, date, 或字符串)
            processed_last_modified_str: Optional[str] = None  # 初始化为可选字符串
            last_modified_val = detail_result.get("last_modified")  # 获取原始值
            if isinstance(last_modified_val, (datetime, date)):
                # 如果是 datetime 或 date 对象，转换为 ISO 格式字符串
                processed_last_modified_str = last_modified_val.isoformat()
            elif isinstance(last_modified_val, str):
                # 如果是字符串，尝试按 ISO 格式解析
                try:
                    # 尝试解析为 datetime，处理 'Z' 时区标识符
                    dt_obj = datetime.fromisoformat(
                        last_modified_val.replace("Z", "+00:00")
                    )
                    processed_last_modified_str = dt_obj.isoformat()
                except ValueError:
                    # 如果按 datetime 解析失败，尝试按 date 解析
                    try:
                        d = date.fromisoformat(last_modified_val)
                        # 将 date 转换为 datetime (时间设为午夜) 再转字符串
                        processed_last_modified_str = datetime.combine(
                            d, datetime.min.time()
                        ).isoformat()
                    except ValueError:
                        # 如果两种解析都失败，记录警告并直接使用原始字符串
                        logger.warning(
                            f"[_get_model_details_for_ids] 无法解析模型 {model_id} 的 last_modified 字符串 '{last_modified_val}'，将按原样使用。"
                        )
                        processed_last_modified_str = last_modified_val
            elif last_modified_val is not None:
                # 如果是其他非 None 类型，记录警告并转换为字符串
                logger.warning(
                    f"[_get_model_details_for_ids] last_modified 类型意外: {type(last_modified_val)}，将转换为字符串。"
                )
                processed_last_modified_str = str(last_modified_val)

            # 将处理后的字符串（如果存在）尝试转换回 datetime 对象，用于 Pydantic 模型
            final_last_modified_dt: Optional[datetime] = None
            if processed_last_modified_str:
                try:
                    # 再次尝试解析，确保处理 'Z'
                    final_last_modified_dt = datetime.fromisoformat(
                        processed_last_modified_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    logger.warning(
                        f"[_get_model_details_for_ids] 无法将最终的 last_modified 字符串 '{processed_last_modified_str}' 解析回 datetime 对象 (模型 {model_id})。将设置为 None。"
                    )

            # --- 尝试创建 HFSearchResultItem 实例 ---
            try:
                # 修复 mypy 错误：在调用 int() 前检查 None
                likes_val = detail_result.get("hf_likes")
                likes_int = int(likes_val) if likes_val is not None else None

                downloads_val = detail_result.get("hf_downloads")
                downloads_int = int(downloads_val) if downloads_val is not None else None

                # 使用从数据库获取的值创建 Pydantic 模型实例
                # 注意：这里的键名 (如 model_id, pipeline_tag) 必须与 HFSearchResultItem 模型定义的字段名匹配
                # 而 .get() 中的键名 (如 'hf_model_id', 'hf_pipeline_tag') 必须与数据库列名匹配
                result_item = HFSearchResultItem(
                    # 显式转换为字符串，以防数据库返回非字符串类型
                    model_id=str(detail_result.get("hf_model_id", "")),
                    pipeline_tag=str(detail_result.get("hf_pipeline_tag", "")),
                    # 使用安全转换后的 int 或 None
                    likes=likes_int,
                    downloads=downloads_int,
                    last_modified=final_last_modified_dt,  # 使用处理后的 datetime 对象
                    score=score_float,
                    tags=processed_tags,  # 使用处理后的 tags 列表
                    author=str(detail_result.get("hf_author", "")),
                    library_name=str(detail_result.get("hf_library_name", "")),
                )

                # 将成功创建的 result_item 存入 result_items_map
                # 使用数据库中的 hf_model_id 作为键，确保一致性
                db_model_id = detail_result.get("hf_model_id")
                if db_model_id:
                    result_items_map[str(db_model_id)] = result_item
                    processed_count += 1
                else:
                    # 如果数据库记录中缺少 hf_model_id，记录警告并跳过
                    logger.warning(
                        f"[_get_model_details_for_ids] 因缺少 'hf_model_id' 而跳过项: {detail_result}"
                    )
                    creation_errors += 1
            except Exception as item_creation_error:
                # 如果创建 Pydantic 模型实例时发生任何错误 (包括验证错误)
                logger.error(
                    f"[_get_model_details_for_ids] 为 hf_model_id {detail_result.get('hf_model_id')} 创建 HFSearchResultItem 时出错: {item_creation_error}",
                    exc_info=True,
                )
                creation_errors += 1
                continue  # 跳过此模型，继续处理下一个

        # --- 按原始顺序整理结果 ---
        # 遍历输入的 model_ids (来自 Faiss 的原始顺序)
        # 从 result_items_map 中查找对应的 HFSearchResultItem 对象
        ordered_results = [
            result_items_map[mid] for mid in model_ids if mid in result_items_map
        ]

        # 记录最终处理统计信息和返回结果数量
        logger.debug(
            f"[_get_model_details_for_ids] 处理完成。已处理: {processed_count}, 跳过(未找到): {skipped_count}, 创建错误: {creation_errors}。返回 {len(ordered_results)} 个项目。"
        )
        # 返回按原始顺序排列的结果列表
        return ordered_results

    def _filter_results_by_date(
        self,
        items: Sequence[ResultItem],  # 使用 Sequence 允许传入列表或元组等序列类型
        published_after: Optional[date],  # 过滤起始日期 (包含)
        published_before: Optional[date],  # 过滤结束日期 (包含)
    ) -> List[ResultItem]:
        """
        根据发布日期范围过滤搜索结果项列表。

        Args:
            items: 包含 ResultItem (SearchResultItem 或 HFSearchResultItem) 的序列。
            published_after: 起始日期 (包含)。如果为 None，则不应用起始日期过滤。
            published_before: 结束日期 (包含)。如果为 None，则不应用结束日期过滤。

        Returns:
            一个列表，只包含发布日期在指定范围内的结果项。
            如果结果项没有 'published_date' 属性或该属性为 None，则会被过滤掉。
        """
        # 如果输入列表为空，直接返回空列表
        if not items:
            return []

        # 如果没有设置任何日期过滤器，直接返回原始列表 (转换为 list)
        if not published_after and not published_before:
            return list(items)

        # 初始化用于存储过滤后结果的列表
        filtered_items = []
        # 遍历每个结果项
        for item in items:
            # 检查项是否有 'published_date' 属性 (主要针对论文 SearchResultItem)
            if not hasattr(item, "published_date"):
                continue  # 如果没有，跳过此项

            # 获取发布日期
            item_date = getattr(item, "published_date")
            # 如果发布日期为 None，也跳过此项
            if not item_date:
                continue

            # 应用起始日期过滤 (如果设置了 published_after)
            # 如果项的日期早于起始日期，则跳过
            if published_after and item_date < published_after:
                continue

            # 应用结束日期过滤 (如果设置了 published_before)
            # 如果项的日期晚于结束日期，则跳过
            if published_before and item_date > published_before:
                continue

            # 如果通过了所有日期过滤，则将该项添加到结果列表
            filtered_items.append(item)

        # 返回过滤后的列表
        return filtered_items

    def _apply_sorting_and_pagination(
        self,
        items: Sequence[ResultItem],  # 输入的结果项序列
        sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]],  # 排序字段
        sort_order: SortOrderLiteral,  # 排序顺序 ('asc' 或 'desc')
        page: int,  # 页码 (从 1 开始)
        page_size: int,  # 每页大小
    ) -> Tuple[List[ResultItem], int, int, int]:
        """
        对搜索结果列表进行排序和分页。

        Args:
            items: 要处理的结果项序列。
            sort_by: 用于排序的字段名。可以是论文或模型特有的字段，或者是 'score'。
                     如果为 None，则不进行排序。
            sort_order: 排序顺序 ('asc' 或 'desc')。
            page: 请求的页码。
            page_size: 每页的结果数量。

        Returns:
            一个元组，包含：
            - paginated_items (List[ResultItem]): 排序和分页后的结果列表。
            - total_items (int): 排序前 (但可能已过滤) 的总项目数。
            - skip (int): 计算出的偏移量。
            - page_size (int): 传入的每页大小 (limit)。
        """
        # 将输入的序列转换为列表，以便进行原地排序
        items_to_sort = list(items)

        # --- 排序 ---
        if sort_by:  # 仅当指定了 sort_by 字段时才进行排序
            # 判断是否为降序排序
            reverse = sort_order == "desc"

            # 定义一个内部函数 `get_sort_key`，用于获取每个项用于排序的键值
            def get_sort_key(
                item: ResultItem,
            ) -> Any:  # 返回 Any 类型，因为不同字段类型不同 (date, str, float, int)
                """获取用于排序的键值，处理 None 情况和不同类型。"""
                # 为 None 值定义默认值，以确保排序行为一致且可预测
                # 这些默认值通常应放在排序顺序的开头或末尾
                min_date = date.min  # 最小日期
                min_datetime = datetime.min.replace(tzinfo=timezone.utc) # 最小带时区日期时间
                min_score = -math.inf  # 负无穷大，用于分数排序时将 None 排在最前面 (升序)
                min_int = -1  # 负数，用于整数排序时将 None 排在最前面 (升序)
                min_str = ""  # 空字符串，用于字符串排序时将 None 排在最前面 (升序)

                try:
                    # 根据项的类型 (论文或模型) 和 sort_by 字段获取排序键
                    if isinstance(item, SearchResultItem):  # 如果是论文
                        if sort_by == "published_date":
                            # 如果 published_date 为 None，使用 min_date
                            return (
                                item.published_date if item.published_date else min_date
                            )
                        elif sort_by == "title":
                            # 如果 title 为 None，使用 min_str；比较时转换为小写以忽略大小写
                            return item.title.lower() if item.title else min_str
                        elif sort_by == "score":
                            # 如果 score 为 None，使用 min_score
                            # 返回元组 (score, paper_id)，当分数相同时，使用 paper_id 作为第二排序键，确保排序稳定
                            primary_key = (
                                item.score if item.score is not None else min_score
                            )
                            secondary_key = item.paper_id  # 假设 paper_id 总是存在且可比较
                            return (primary_key, secondary_key)
                        else:
                            # 如果 sort_by 对于论文类型无效，记录警告并返回 None
                            logger.warning(
                                f"不支持的 SearchResultItem 排序键 '{sort_by}'。"
                            )
                            return None # 返回 None 表示此项无法参与排序

                    elif isinstance(item, HFSearchResultItem):  # 如果是模型
                        if sort_by == "likes":
                            # 修复 mypy 错误：显式检查 None
                            likes_val = item.likes
                            return likes_val if likes_val is not None else min_int
                        elif sort_by == "downloads":
                            # 修复 mypy 错误：显式检查 None
                            downloads_val = item.downloads
                            return downloads_val if downloads_val is not None else min_int
                        elif sort_by == "last_modified":
                            # 如果 last_modified 为 None，使用 min_datetime
                            # 确保比较的是带时区的 datetime 对象
                            dt = item.last_modified
                            if dt and dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc) # 假设 UTC
                            return dt if dt else min_datetime
                        elif sort_by == "score":
                            # 如果 score 为 None，使用 min_score
                            # 返回元组 (score, model_id)，确保排序稳定
                            primary_key = (
                                item.score if item.score is not None else min_score
                            )
                            # mypy 在这里可能会误报类型错误，暂时忽略
                            secondary_key = (
                                item.model_id  # type: ignore[assignment]
                            )  # 假设 model_id 总是存在且可比较
                            return (primary_key, secondary_key)
                        else:
                            # 如果 sort_by 对于模型类型无效，记录警告并返回 None
                            logger.warning(
                                f"不支持的 HFSearchResultItem 排序键 '{sort_by}'。"
                            )
                            return None

                    else:
                        # 如果项的类型不是预期的论文或模型，记录警告并返回 None
                        logger.warning(f"不支持的排序项类型: {type(item)}")
                        return None
                except Exception as key_error:
                    # 如果在获取排序键的过程中发生异常，记录错误并返回 None
                    item_id = getattr(
                        item, "paper_id", getattr(item, "model_id", "未知ID")
                    )
                    logger.error(
                        f"获取项 {item_id} 的排序键 '{sort_by}' 时出错: {key_error}"
                    )
                    return None

            # 尝试进行排序
            try:
                # 首先，过滤掉那些无法获取有效排序键的项 (get_sort_key 返回 None 的项)
                valid_items = [
                    item for item in items_to_sort if get_sort_key(item) is not None
                ]
                # 如果有项被过滤掉，记录警告
                if len(valid_items) < len(items_to_sort):
                    logger.warning(
                        f"排序跳过了 {len(items_to_sort) - len(valid_items)} 个项，因为它们的排序键 '{sort_by}' 缺失或类型无效。"
                    )

                # 记录排序前的部分数据（ID 和分数），用于调试
                logger.debug(
                    f"_apply_sorting_and_pagination: 排序前 (sort_by='{sort_by}', reverse={reverse}): {[(getattr(i, 'paper_id', getattr(i, 'model_id', 'N/A')), getattr(i, 'score', None)) for i in valid_items[:20]]}" # 最多显示前20条
                )

                # 对有效项进行排序
                valid_items.sort(key=get_sort_key, reverse=reverse)

                # 记录排序后的部分数据（ID 和分数），用于调试
                logger.debug(
                    f"_apply_sorting_and_pagination: 排序后: {[(getattr(i, 'paper_id', getattr(i, 'model_id', 'N/A')), getattr(i, 'score', None)) for i in valid_items[:20]]}" # 最多显示前20条
                )

                # 将排序后的有效项列表用于后续分页
                items_to_paginate = valid_items
            except TypeError as sort_error:
                # 如果排序过程中发生 TypeError (通常是因为比较了不兼容的类型)
                logger.error(
                    f"按 '{sort_by}' 排序时发生 TypeError: {sort_error}。项目可能包含不兼容的类型进行比较。将返回原始顺序。",
                    exc_info=True,
                )
                # 在发生类型错误时，回退到使用原始顺序的列表进行分页
                items_to_paginate = items_to_sort
            except Exception as e:
                # 如果发生其他意外错误
                logger.error(
                    f"按 '{sort_by}' 排序时发生意外错误: {e}", exc_info=True
                )
                # 在发生其他错误时，也回退到使用原始顺序
                items_to_paginate = items_to_sort

        else:  # 如果没有指定 sort_by，则不排序，直接使用原始列表进行分页
            items_to_paginate = items_to_sort

        # --- 分页 ---
        # 计算总项目数 (排序后的或原始的)
        total_items = len(items_to_paginate)
        # 计算要跳过的项目数 (数据库查询中的 OFFSET)
        skip = (page - 1) * page_size
        # 计算分页结束的索引
        end_index = skip + page_size
        # 从列表中切片获取当前页的项目
        paginated_items = items_to_paginate[skip:end_index]

        # 记录分页信息
        logger.debug(
            f"将 {total_items} 个项目分页到 {len(paginated_items)} 个 (页码 {page}, 大小 {page_size})。"
        )
        # 返回分页后的列表、总项目数、跳过数量和页面大小
        return (paginated_items, total_items, skip, page_size)

    async def perform_semantic_search(
        self,
        query: str, # 搜索查询字符串
        target: SearchTarget, # 搜索目标 ('papers', 'models')
        page: int = 1, # 页码
        page_size: int = 10, # 每页大小
        top_n: int = DEFAULT_TOP_N_SEMANTIC, # 从 Faiss 获取的初始结果数量
        date_from: Optional[date] = None, # 起始日期过滤器
        date_to: Optional[date] = None, # 结束日期过滤器
        sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]] = "score", # 排序字段，默认为 'score'
        sort_order: SortOrderLiteral = "desc", # 排序顺序，默认为 'desc'
    ) -> PaginatedResult:
        """
        执行语义搜索。

        此方法处理语义搜索的完整流程：
        1. 验证输入和依赖项（嵌入器）。
        2. 使用嵌入器将查询文本转换为向量。
        3. 在指定的 Faiss 索引（论文或模型）中搜索相似向量。
        4. 处理 Faiss 返回的 ID 和距离，转换为分数。
        5. 根据获取的 ID 从 PostgreSQL 获取详细信息。
        6. 应用日期过滤器。
        7. 对结果进行排序和分页。
        8. 返回分页后的结果模型 (`PaginatedPaperSearchResult` 或 `PaginatedHFModelSearchResult`)。

        Args:
            query: 用户输入的搜索查询文本。
            target: 指定搜索目标是 'papers' 还是 'models'。
            page: 请求的页码 (从 1 开始)。
            page_size: 每页返回的结果数量。
            top_n: 从 Faiss 向量索引中检索的最相似结果的初始数量。后续会根据此列表获取详情并分页。
            date_from: (仅对论文) 过滤结果的起始发布日期 (包含)。
            date_to: (仅对论文) 过滤结果的结束发布日期 (包含)。
            sort_by: 用于对最终结果排序的字段。默认为 'score'。
            sort_order: 排序顺序 ('asc' 或 'desc')。默认为 'desc'。

        Returns:
            一个分页结果对象 (`PaginatedPaperSearchResult` 或 `PaginatedHFModelSearchResult`)，
            包含当前页的项目列表和总项目数等信息。

        Raises:
            HTTPException: 如果嵌入器服务不可用 (状态码 503)。
        """
        # 记录开始执行语义搜索的信息
        logger.info(
            f"[perform_semantic_search] 目标: {target}, 查询: '{query}', 页码: {page}, 页大小: {page_size}"
        )

        # --- 检查嵌入器是否可用 ---
        # 语义搜索强依赖于文本嵌入器将查询转换为向量
        if self.embedder is None:
            logger.error("[perform_semantic_search] 嵌入器 (Embedder) 不可用。")
            # 如果嵌入器未初始化，则无法执行语义搜索，抛出 HTTP 503 错误
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="文本嵌入服务对于语义搜索是必需的，但当前不可用。",
            )
        # --- 结束检查 ---

        # 记录更详细的搜索参数
        logger.info(
            f"执行语义搜索: query='{query}', target='{target}', "
            f"page={page}, page_size={page_size}, top_n={top_n}, sort='{sort_by}'"
        )

        # --- 基本验证和准备 ---
        # 计算分页的偏移量
        skip = (page - 1) * page_size
        # 当前版本不支持 'all' 目标的语义搜索
        if target == "all":
            logger.error("语义搜索目标 'all' 尚未实现。")
            # 返回空的通用分页结果
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )
        # 验证 target 是否是预期的值 ('papers' 或 'models')
        if target not in get_args(SearchTarget): # get_args(SearchTarget) 会返回 ('papers', 'models', 'all')
            logger.error(f"无效的搜索目标 '{target}'。")
            # 返回空的通用分页结果
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- 定义目标特定的配置 ---
        # 使用字典来存储不同目标（论文/模型）所需的仓库、函数、模型等信息
        # 这样可以减少代码重复
        target_configs = {
            "papers": {
                "faiss_repo": self.faiss_repo_papers, # 使用论文 Faiss 仓库
                "fetch_details_func": self._get_paper_details_for_ids, # 使用获取论文详情的函数
                "id_type": int, # 论文 ID 是整数
                "ResultModel": SearchResultItem, # 结果项的模型是 SearchResultItem
                "PaginatedModel": PaginatedPaperSearchResult, # 分页结果的模型
                "sort_options": get_args(PaperSortByLiteral), # 论文允许的排序字段
                "EmptyModel": PaginatedPaperSearchResult, # 空结果时使用的模型
            },
            "models": {
                "faiss_repo": self.faiss_repo_models, # 使用模型 Faiss 仓库
                "fetch_details_func": self._get_model_details_for_ids, # 使用获取模型详情的函数
                "id_type": str, # 模型 ID 是字符串
                "ResultModel": HFSearchResultItem, # 结果项的模型是 HFSearchResultItem
                "PaginatedModel": PaginatedHFModelSearchResult, # 分页结果的模型
                "sort_options": get_args(ModelSortByLiteral), # 模型允许的排序字段
                "EmptyModel": PaginatedHFModelSearchResult, # 空结果时使用的模型
            },
        }

        # 根据传入的 target 获取对应的配置
        config = target_configs.get(target)
        # 如果 target 无效 (理论上前面已检查过，但再次确认)
        if not config:
            logger.error(f"[perform_semantic_search] 无效的目标: {target}")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # 从配置字典中解包出需要的变量，并使用 cast 进行类型提示，帮助静态分析器理解
        faiss_repo: FaissRepository = cast(FaissRepository, config["faiss_repo"])
        # fetch_details_func 是一个异步函数，接收 ID 列表和可选的分数，返回 ResultItem 列表
        fetch_details_func: Callable[..., Coroutine[Any, Any, List[ResultItem]]] = cast(
            Callable[..., Coroutine[Any, Any, List[ResultItem]]],
            config["fetch_details_func"],
        )
        id_type: Type = cast(Type, config["id_type"]) # ID 的类型 (int 或 str)
        PaginatedModel: Type[PaginatedResult] = cast(
            Type[PaginatedResult], config["PaginatedModel"]
        ) # 分页模型的类型
        target_sort_options: Tuple = cast(Tuple, config["sort_options"]) # 该目标允许的排序字段元组
        EmptyModel: Type[PaginatedResult] = cast(
            Type[PaginatedResult], config["EmptyModel"]
        ) # 空结果时使用的分页模型类型

        # --- 查询验证 ---
        # 如果查询字符串为空或仅包含空白，则不执行搜索
        if not query or not query.strip():
            logger.warning("尝试使用空查询进行语义搜索。")
            # 返回特定目标的空分页模型
            return EmptyModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 嵌入 (Embedding) ---
        # 将查询文本转换为向量
        embedding: Optional[np.ndarray] = None # 初始化为 None
        try:
            embedding = self.embedder.embed(query) # 调用嵌入器的 embed 方法
            # 如果嵌入失败 (返回 None)
            if embedding is None:
                logger.error("[perform_semantic_search] 生成嵌入向量失败。")
                # 返回特定目标的空分页模型
                return EmptyModel(items=[], total=0, skip=skip, limit=page_size)
            # 记录嵌入成功信息
            logger.debug("[perform_semantic_search] 成功生成嵌入向量。")
        except Exception as embed_error:
            # 如果嵌入过程中发生异常
            logger.error(
                f"[perform_semantic_search] 生成嵌入向量时出错: {embed_error}",
                exc_info=True,
            )
            # 返回通用的空分页结果
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- Faiss 仓库检查 ---
        # 检查对应的 Faiss 仓库是否已准备好 (例如，索引是否已加载)
        if not faiss_repo.is_ready():
            logger.error(
                f"[perform_semantic_search] 目标 '{target}' 的 Faiss 仓库尚未就绪。"
            )
            # 返回特定目标的空分页模型
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 验证 Faiss ID 类型 ---
        # 检查 Faiss 仓库内部配置的 ID 类型是否与服务配置的 ID 类型匹配
        # 这是为了防止因配置错误导致后续处理失败
        faiss_id_type_name = getattr(faiss_repo, "id_type", None) # 获取 Faiss 仓库记录的 ID 类型名称
        expected_id_type_name = id_type.__name__ # 获取服务期望的 ID 类型名称 (如 'int', 'str')
        if faiss_id_type_name != expected_id_type_name:
            logger.error(
                f"[perform_semantic_search] ID 类型不匹配: 目标 '{target}' 的 Faiss 仓库期望 '{faiss_id_type_name}'，但服务配置为 '{expected_id_type_name}'"
            )
            # 返回特定目标的空分页模型
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 执行 Faiss 搜索 ---
        # 初始化用于存储 Faiss 原始结果的列表
        search_results_raw: List[Tuple[FaissID, float]] = []
        try:
            # 调用 Faiss 仓库的 search_similar 方法执行向量相似度搜索
            # k=top_n 指定返回最相似的 top_n 个结果
            search_results_raw = await faiss_repo.search_similar(embedding, k=top_n)
            # 记录原始结果数量和示例
            logger.debug(
                f"[perform_semantic_search] 目标 '{target}' 的 Faiss 搜索返回 {len(search_results_raw)} 条原始结果。示例: {search_results_raw[:5]}"
            )
        except Exception as e:
            # 如果 Faiss 搜索出错
            logger.error(
                f"[perform_semantic_search] 目标 '{target}' 的 Faiss 搜索过程中出错: {e}",
                exc_info=True,
            )
            # 返回特定目标的空分页模型
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 处理 Faiss 结果 ---
        # 如果 Faiss 没有返回任何结果
        if not search_results_raw:
            logger.info(f"[perform_semantic_search] 目标 '{target}' 的 Faiss 搜索未返回结果。")
            # 返回特定目标的空分页模型
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # 初始化用于存储有效 ID 和对应分数的列表和字典
        result_ids: list = [] # 存储有效 ID (类型可能是 int 或 str)
        result_scores: dict = {} # 存储 ID 到分数的映射
        # 遍历 Faiss 返回的原始结果 (ID, 距离) 对
        for original_id_union, distance in search_results_raw:
            # 再次验证从 Faiss 获取的 ID 类型是否与当前目标期望的类型一致
            if isinstance(original_id_union, id_type):
                # 如果类型匹配，进行类型转换 (cast) 并处理
                original_id = cast(Union[int, str], original_id_union)
                # 将 Faiss 距离转换为 (0, 1] 的相似度分数
                score = self._convert_distance_to_score(distance)
                # 将有效 ID 添加到列表
                result_ids.append(original_id)
                # 将 ID 和分数存入字典
                result_scores[original_id] = score
            else:
                # 如果类型不匹配，记录警告并跳过此结果
                logger.warning(
                    f"[perform_semantic_search] 跳过 Faiss 结果。期望 ID 类型 '{id_type.__name__}'，但收到 {type(original_id_union)} (ID: {original_id_union})"
                )

        # 如果经过类型检查后没有剩余的有效 ID
        if not result_ids:
            logger.warning(
                "[perform_semantic_search] 类型检查后，从 Faiss 结果中未提取到有效 ID。"
            )
            # 返回特定目标的空分页模型
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # 记录提取到的有效 ID 数量和示例
        logger.debug(
            f"[perform_semantic_search] 为目标 '{target}' 提取到 {len(result_ids)} 个有效 ID。示例: {result_ids[:10]}"
        )

        # --- 从 Postgres 获取详情 ---
        # 初始化用于存储带有详细信息的结果项列表
        all_items_list: List[ResultItem] = []
        try:
            # 记录将要获取详情的 ID 数量和使用的函数名
            logger.debug(
                f"[perform_semantic_search] 正在使用 {fetch_details_func.__name__} 获取 {len(result_ids)} 个 ID 的详情..."
            )
            # 调用之前根据 target 选择的获取详情的异步函数
            # 将 ID 列表和分数字典传递给它
            all_items_list = await fetch_details_func(result_ids, result_scores)
            # 记录获取到的带详情的结果项数量
            logger.debug(
                f"[perform_semantic_search] 为目标 '{target}' 获取到 {len(all_items_list)} 个带详情的项目。"
            )
        except Exception as fetch_error:
            # 如果获取详情时出错
            logger.error(
                f"[perform_semantic_search] 获取目标 '{target}' 详情时出错: {fetch_error}",
                exc_info=True,
            )
            # 返回特定目标的空分页模型
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 按日期过滤 (仅对论文有效，但方法内部会处理) ---
        # 调用日期过滤辅助方法
        filtered_items: List[ResultItem] = self._filter_results_by_date(
            all_items_list, date_from, date_to
        )
        # 如果过滤后列表不等于原列表，可以记录一下过滤掉了多少项 (可选)
        if len(filtered_items) != len(all_items_list):
             logger.debug(f"日期过滤将结果从 {len(all_items_list)} 项减少到 {len(filtered_items)} 项。")


        # --- 排序和分页 ---
        # 验证排序键是否对当前目标有效
        valid_sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]] = None
        if sort_by:
            # 检查传入的 sort_by 是否在当前 target 允许的排序选项中
            if sort_by in target_sort_options:
                valid_sort_by = sort_by
            # 语义搜索总是会产生 'score'，所以 'score' 总是有效的
            elif sort_by == "score":
                valid_sort_by = "score"
            else:
                # 如果 sort_by 无效，记录警告并默认使用 'score'
                logger.warning(
                    f"[perform_semantic_search] 无效的排序键 '{sort_by}' 用于目标 '{target}'。将默认使用 'score'。"
                )
                valid_sort_by = "score"

        # 调用排序和分页的辅助方法
        paginated_items_list: List[ResultItem] # 排序分页后的项目列表
        total_items: int # 过滤后、分页前的总项目数
        calculated_skip: int # 计算出的偏移量
        calculated_limit: int # 页面大小
        paginated_items_list, total_items, calculated_skip, calculated_limit = (
            self._apply_sorting_and_pagination(
                filtered_items, # 使用过滤后的列表
                sort_by=valid_sort_by, # 使用验证/默认后的排序键
                sort_order=sort_order, # 使用传入的排序顺序
                page=page, # 使用传入的页码
                page_size=page_size, # 使用传入的页面大小
            )
        )

        # --- 构建最终响应 ---
        # 记录最终返回的项目数量
        logger.debug(
            f"[perform_semantic_search] 为目标 '{target}' 返回 {len(paginated_items_list)} 个项目 (经过过滤/排序/分页)。"
        )

        # 使用特定目标的分页模型 (PaginatedModel) 创建响应对象
        # 使用 cast 告诉类型检查器 paginated_items_list 的具体类型 (虽然它是 ResultItem 列表，但 PaginatedModel 需要更具体的类型)
        # 我们在这里可以确信类型是正确的，因为它是基于 target 选择的 PaginatedModel
        return PaginatedModel(
            items=cast(List[Any], paginated_items_list), # 将结果列表强制转换为 Any 列表以匹配 Pydantic 模型签名
            total=total_items, # 过滤后、分页前的总数
            skip=calculated_skip, # 计算出的偏移量
            limit=calculated_limit, # 页面大小
        )

    async def perform_keyword_search(
        self,
        query: str, # 搜索查询文本
        target: SearchTarget, # 搜索目标 ('papers' 或 'models')
        page: int = 1, # 页码
        page_size: int = 10, # 每页大小
        date_from: Optional[date] = None, # (论文) 起始日期过滤器
        date_to: Optional[date] = None, # (论文) 结束日期过滤器
        area: Optional[List[str]] = None, # (论文) 领域过滤器 (支持多选)
        pipeline_tag: Optional[str] = None, # (模型) pipeline_tag 过滤器
        # 新增：其他可选过滤器
        filter_authors: Optional[List[str]] = None, # (论文) 作者过滤器 (支持多选)
        filter_library_name: Optional[str] = None, # (模型) 库名称过滤器
        filter_tags: Optional[List[str]] = None, # (模型) 标签过滤器 (支持多选)
        filter_author: Optional[str] = None, # (模型) 作者 (hf_author) 过滤器
        # 允许指定排序字段和顺序
        sort_by: Optional[Union[PaperSortByLiteral, ModelSortByLiteral]] = None,
        sort_order: SortOrderLiteral = "desc",
    ) -> PaginatedResult:
        """
        执行关键词搜索。

        此方法利用 PostgreSQL 的全文搜索或其他文本匹配能力进行搜索。
        与语义搜索不同，关键词搜索的过滤、排序和分页逻辑主要委托给
        PostgreSQL 仓库层的方法 (`search_papers_by_keyword` 或 `search_models_by_keyword`) 处理，
        这些方法通常直接在 SQL 查询中使用 `WHERE` 子句、`ORDER BY` 和 `LIMIT`/`OFFSET`。

        流程：
        1. 验证输入参数。
        2. 根据 `target` 确定要调用的 PostgreSQL 仓库方法和返回的分页模型类型。
        3. 验证 `sort_by` 参数对当前目标是否有效，设置默认排序（如果未提供）。
        4. 构建传递给 PostgreSQL 仓库方法的参数字典，包含查询、分页、排序和所有过滤器。
        5. 调用相应的 PostgreSQL 仓库方法，该方法返回结果列表（通常是字典列表）和总项目数。
        6. 将返回的字典列表转换为对应的 Pydantic 模型列表 (`SearchResultItem` 或 `HFSearchResultItem`)。
        7. 使用 Pydantic 模型列表和总数构建并返回最终的分页结果对象。

        Args:
            query: 用户输入的搜索查询文本。
            target: 指定搜索目标是 'papers' 还是 'models'。
            page: 请求的页码 (从 1 开始)。
            page_size: 每页返回的结果数量。
            date_from: (论文) 过滤结果的起始发布日期 (包含)。
            date_to: (论文) 过滤结果的结束发布日期 (包含)。
            area: (论文) 过滤结果的领域列表 (匹配其中任何一个即可)。
            pipeline_tag: (模型) 过滤结果的 Hugging Face pipeline tag。
            filter_authors: (论文) 过滤结果的作者列表 (匹配其中任何一个即可)。
            filter_library_name: (模型) 过滤结果的库名称 (例如 'transformers', 'diffusers')。
            filter_tags: (模型) 过滤结果的标签列表 (需要匹配所有指定的标签，或根据 PG 实现决定)。
            filter_author: (模型) 过滤结果的作者 (hf_author)。
            sort_by: 用于排序的字段名。如果为 None 或无效，会使用目标类型的默认排序。不支持 'score'。
            sort_order: 排序顺序 ('asc' 或 'desc')。默认为 'desc'。

        Returns:
            一个分页结果对象 (`PaginatedPaperSearchResult` 或 `PaginatedHFModelSearchResult`)。
        """
        # 记录开始执行关键词搜索的信息
        logger.info(
            f"[perform_keyword_search] 目标: {target}, 查询: '{query}', 页码: {page}, 页大小: {page_size}"
        )

        # --- 基本验证和准备 ---
        # 计算分页偏移量
        skip = (page - 1) * page_size
        # 'all' 目标不支持
        if target == "all":
            logger.error(f"Keyword search target 'all' is not implemented.")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )
        # 验证 target 值
        if target not in get_args(SearchTarget):
            logger.error(f"Invalid keyword search target '{target}'.")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )
        # 空查询处理
        if not query or not query.strip():
            logger.warning("Keyword search attempted with empty query.")
            EmptyModel = (
                PaginatedPaperSearchResult
                if target == "papers"
                else PaginatedHFModelSearchResult
            )
            return EmptyModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 根据 target 分配特定的函数、模型和排序选项 ---
        # 定义将要调用的 PG 搜索函数 (签名应类似: **kwargs -> Coroutine[Any, Any, Tuple[list, int]])
        search_pg_func: Callable[..., Coroutine[Any, Any, Tuple[list, int]]]
        # 定义最终返回的分页模型类型
        PaginatedModel: Type[PaginatedResult]
        # 定义传递给 PG 仓库的有效排序字段名 (PG 仓库可能期望特定的字符串)
        valid_pg_sort_by: Optional[str] = None
        # 定义当前 target 允许的排序选项 (用于验证 sort_by 参数)
        target_sort_options: Tuple = ()

        if target == "papers":
            # 检查 PG 仓库是否有论文关键词搜索方法
            if not hasattr(self.pg_repo, "search_papers_by_keyword"):
                logger.error(
                    "[perform_keyword_search] PostgresRepository missing 'search_papers_by_keyword' method."
                )
                return PaginatedPaperSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )
            # 分配论文搜索函数
            search_pg_func = self.pg_repo.search_papers_by_keyword
            # 分配论文分页模型
            PaginatedModel = PaginatedPaperSearchResult
            # 获取论文允许的排序字段
            target_sort_options = get_args(PaperSortByLiteral)
            # 为论文关键词搜索设置默认排序字段
            if sort_by is None:
                sort_by = "published_date"
            # 验证 sort_by 对论文是否有效 ('score' 在关键词搜索中无效)
            if sort_by not in target_sort_options or sort_by == "score":
                logger.warning(
                    f"[perform_keyword_search] 无效或不支持的排序键 '{sort_by}' 用于论文关键词搜索。将使用 'published_date'。"
                )
                valid_pg_sort_by = "published_date" # PG 仓库期望的列名
            else:
                # 类型转换，确保传递给 PG 的是它能理解的字符串
                # 假设 PG 仓库期望的排序键与 Literal 定义一致
                valid_pg_sort_by = cast(
                    Optional[Literal["published_date", "title"]], # PG 可能支持按 title 排序
                    sort_by,
                )

        elif target == "models":
            # 检查 PG 仓库是否有模型关键词搜索方法
            if not hasattr(self.pg_repo, "search_models_by_keyword"):
                logger.error(
                    "[perform_keyword_search] PostgresRepository missing 'search_models_by_keyword' method."
                )
                return PaginatedHFModelSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )
            # 分配模型搜索函数
            search_pg_func = self.pg_repo.search_models_by_keyword
            # 分配模型分页模型
            PaginatedModel = PaginatedHFModelSearchResult
            # 获取模型允许的排序字段
            target_sort_options = get_args(ModelSortByLiteral)
            # 为模型关键词搜索设置默认排序字段 (可以根据业务调整，例如 'likes')
            if sort_by is None:
                sort_by = "last_modified"
            # 验证 sort_by 对模型是否有效 ('score' 无效)
            if sort_by not in target_sort_options or sort_by == "score":
                logger.warning(
                    f"[perform_keyword_search] 无效或不支持的排序键 '{sort_by}' 用于模型关键词搜索。将使用 'last_modified'。"
                )
                valid_pg_sort_by = "last_modified" # PG 仓库期望的列名
            else:
                # 类型转换
                valid_pg_sort_by = cast(
                    Optional[Literal["likes", "downloads", "last_modified"]], sort_by
                )

        # --- 执行 PostgreSQL 关键词搜索 ---
        # PG 仓库方法应该返回包含结果字典的列表和匹配的总项目数
        pg_results: List[
            Dict[str, Any]
        ] = []  # Results from PG repo (List[Dict] for both targets now)
        total_items: int = 0
        try:
            # 构建传递给 PG 仓库方法的参数字典
            pg_params: Dict[str, Any] = {
                "query": query,
                "limit": page_size, # 分页大小
                "skip": skip, # 分页偏移量
                "sort_by": valid_pg_sort_by, # 验证/默认后的排序字段
                "sort_order": sort_order, # 排序顺序
            }

            # 根据目标类型，添加特定于该目标的过滤参数
            if target == "papers":
                pg_params["published_after"] = date_from
                pg_params["published_before"] = date_to
                pg_params["filter_area"] = area # 传递领域过滤器
                pg_params["filter_authors"] = filter_authors # 传递作者过滤器
            elif target == "models":
                pg_params["pipeline_tag"] = pipeline_tag
                pg_params["filter_library_name"] = filter_library_name # 传递库名过滤器
                pg_params["filter_tags"] = filter_tags  # 新增
                pg_params["filter_author"] = filter_author  # 新增

            logger.debug(
                f"[perform_keyword_search] Calling {search_pg_func.__name__} with params: {pg_params}"
            )

            pg_results, total_items = await search_pg_func(**pg_params)

            # 记录从PG repo获取的结果
            logger.debug(
                f"[perform_keyword_search] PG search returned {len(pg_results)} raw items, total count: {total_items}"
            )

        except Exception as search_error:
            logger.error(
                f"[perform_keyword_search] Error executing keyword search for target '{target}': {search_error}",
                exc_info=True,
            )
            # Return empty PaginatedModel based on target
            return PaginatedModel(items=[], total=0, skip=skip, limit=page_size)

        # --- 构建最终响应 (将 PG 返回的字典列表转换为 Pydantic 模型列表) ---
        if target == "papers":
            try:
                # 初始化论文 Pydantic 模型列表
                paper_items: List[SearchResultItem] = []
                # 遍历从 PG 返回的每个结果字典
                for item_dict in pg_results:
                    # 安全地处理 'authors' 字段，它可能在数据库中存储为 JSON 字符串
                    if "authors" in item_dict and isinstance(item_dict["authors"], str):
                        try:
                            # 尝试解析 JSON 字符串为 Python 列表
                            item_dict["authors"] = json.loads(item_dict["authors"])
                        except (json.JSONDecodeError, TypeError):
                            # 如果解析失败，记录警告并将作者设置为空列表
                            logger.warning(
                                f"[perform_keyword_search] Could not decode authors JSON for paper_id {item_dict.get('paper_id')}"
                            )
                            item_dict["authors"] = [] # 保证 authors 字段是列表类型

                    # 尝试使用字典创建 SearchResultItem 实例
                    # Pydantic 会进行验证，如果字典缺少必需字段或类型不匹配会抛出 ValidationError
                    try:
                        paper_items.append(SearchResultItem(**item_dict))
                    except ValidationError as item_error:
                        # 如果单个项目验证失败，记录错误并跳过该项目
                        logger.error(
                            f"[perform_keyword_search] 从字典创建 SearchResultItem 时出错: {item_error}\n数据: {item_dict}",
                            exc_info=False, # 通常不需要完整堆栈
                        )
                        continue # 继续处理下一个项目
                    except Exception as e:
                         # 捕获其他可能的错误
                         logger.error(
                            f"[perform_keyword_search] 创建 SearchResultItem 时发生意外错误: {e}\n数据: {item_dict}",
                            exc_info=True,
                         )
                         continue

                # 使用成功转换的 Pydantic 模型列表和总数创建分页结果对象
                return PaginatedPaperSearchResult(
                    items=paper_items, total=total_items, skip=skip, limit=page_size
                )
            except Exception as e:
                # 如果在整个转换过程中发生意外错误
                logger.error(
                    f"[perform_keyword_search] 将论文关键词搜索结果转换为 SearchResultItem 时出错: {e}",
                    exc_info=True,
                )
                # 返回一个空的但包含正确总数的分页对象 (如果 total_items 已获取)
                return PaginatedPaperSearchResult(
                    items=[], total=total_items, skip=skip, limit=page_size
                )
        elif target == "models":
            try:
                # 初始化模型 Pydantic 模型列表
                model_items: List[HFSearchResultItem] = []
                # 遍历从 PG 返回的每个结果字典
                for item_dict in pg_results:
                    try:
                        # --- 手动处理 PG 返回的字典，以匹配 HFSearchResultItem 模型字段 ---
                        # 关键词搜索通常没有 'score'，这里我们赋一个默认值 0.0
                        # (注意：PG 返回的列名可能带有 'hf_' 前缀)

                        # 安全处理 'tags' (可能为 JSON 字符串) - 复用 _get_model_details_for_ids 中的逻辑
                        processed_tags: Optional[List[str]] = None
                        tags_list = item_dict.get("hf_tags") # 从 PG 结果获取 'hf_tags' 列
                        if isinstance(tags_list, str):
                            try:
                                parsed_tags = json.loads(tags_list)
                                if isinstance(parsed_tags, list) and all(
                                    isinstance(t, str) for t in parsed_tags
                                ):
                                    processed_tags = parsed_tags
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"[perform_keyword_search] Could not decode hf_tags JSON for model {item_dict.get('hf_model_id')}"
                                )
                        elif isinstance(tags_list, list) and all(
                            isinstance(t, str) for t in tags_list
                        ):
                            processed_tags = tags_list

                        # 安全处理 'last_modified' - 复用并调整 _get_model_details_for_ids 中的逻辑
                        final_last_modified_dt: Optional[datetime] = None
                        last_modified_val = item_dict.get("hf_last_modified") # 从 PG 结果获取
                        if isinstance(last_modified_val, (datetime, date)):
                            # 确保是 datetime 对象
                            final_last_modified_dt = (
                                last_modified_val
                                if isinstance(last_modified_val, datetime)
                                else datetime.combine(
                                    last_modified_val, datetime.min.time()
                                )
                            )
                            # 确保时区感知 (假设 UTC)
                            if final_last_modified_dt.tzinfo is None:
                                final_last_modified_dt = final_last_modified_dt.replace(
                                    tzinfo=timezone.utc
                                )
                        elif isinstance(last_modified_val, str):
                            try:
                                dt_obj = datetime.fromisoformat(
                                    last_modified_val.replace("Z", "+00:00")
                                )
                                if dt_obj.tzinfo is None:
                                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                                final_last_modified_dt = dt_obj
                            except ValueError:
                                logger.warning(
                                    f"[perform_keyword_search] Could not parse last_modified string '{last_modified_val}' for model {item_dict.get('hf_model_id')}. Setting to None."
                                )

                        # 创建 HFSearchResultItem 实例，手动映射字段
                        model_instance = HFSearchResultItem(
                            model_id=str(item_dict.get("hf_model_id", "")),
                            author=str(item_dict.get("hf_author", "")),
                            pipeline_tag=str(item_dict.get("hf_pipeline_tag", "")),
                            library_name=str(item_dict.get("hf_library_name", "")),
                            tags=processed_tags,
                            likes=int(item_dict["hf_likes"])
                            if item_dict.get("hf_likes") is not None
                            else None,
                            downloads=int(item_dict["hf_downloads"])
                            if item_dict.get("hf_downloads") is not None
                            else None,
                            last_modified=final_last_modified_dt,
                            score=0.0,  # Assign default score for keyword results
                            # sha=item.get("hf_sha") # sha is not in HFSearchResultItem model
                        )
                        model_items.append(model_instance)
                    except ValidationError as val_err:
                        logger.error(
                            f"[perform_keyword_search] Validation error creating HFSearchResultItem: {val_err}\nData: {item_dict}",
                            exc_info=False,  # Don't need full stack usually
                        )
                        continue  # Skip invalid item
                    except Exception as item_error:
                        logger.error(
                            f"[perform_keyword_search] Unexpected error creating HFSearchResultItem from dict: {item_error}\nData: {item_dict}",
                            exc_info=True,
                        )
                        continue

                # 使用成功转换的 Pydantic 模型列表和总数创建分页结果对象
                return PaginatedHFModelSearchResult(
                    items=model_items, total=total_items, skip=skip, limit=page_size
                )
            except Exception as e:
                # 如果在整个转换过程中发生意外错误
                logger.error(
                    f"[perform_keyword_search] 将模型关键词搜索结果转换为 HFSearchResultItem 时出错: {e}",
                    exc_info=True,
                )
                # 返回空的但包含正确总数的分页对象
                return PaginatedHFModelSearchResult(
                    items=[], total=total_items, skip=skip, limit=page_size
                )
        else:
            # 这个分支理论上不会到达，因为 target 在前面已经验证过
            logger.error(
                f"[perform_keyword_search] 关键词搜索结束时遇到意外的目标: {target}"
            )
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

    async def perform_hybrid_search(
        self,
        query: str, # 搜索查询文本
        target: SearchTarget = "papers", # 搜索目标，默认为 'papers'
        page: int = 1, # 页码
        page_size: int = 10, # 每页大小
        filters: Optional[SearchFilterModel] = None, # 可选的过滤器对象
    ) -> PaginatedResult:
        """
        执行混合搜索，结合语义搜索和关键词搜索的结果。

        这种方法旨在利用两种搜索方式的优点：语义搜索可以找到概念上相关但可能不包含
        确切关键词的结果，而关键词搜索可以精确匹配特定术语。

        流程：
        1. 验证输入（查询、目标）。
        2. 从 `filters` 对象中提取各种过滤条件（日期、领域、pipeline tag、排序等）。
        3. 根据 `target` 调用内部的特定混合搜索实现 (`_perform_hybrid_search_papers` 或 `_perform_hybrid_search_models`)。
        4. 特定实现会分别执行语义搜索和关键词搜索，获取各自的结果列表（通常包含 ID 和分数/详情）。
        5. 使用 RRF (Reciprocal Rank Fusion) 算法合并两个结果列表，为每个结果计算一个融合分数。
        6. 获取合并后 ID 的完整详细信息。
        7. 对合并后的结果应用 `filters` 中指定的过滤条件。
        8. 根据 `filters` 中指定的 `sort_by` 和 `sort_order` 对结果进行排序（默认按融合分数降序）。
        9. 对排序后的结果进行分页。
        10. 返回最终的分页结果对象。

        Args:
            query: 用户输入的搜索查询文本。
            target: 搜索目标 ('papers' 或 'models')。默认为 'papers'。
            page: 请求的页码 (从 1 开始)。
            page_size: 每页返回的结果数量。
            filters: 一个可选的 `SearchFilterModel` 对象，包含所有过滤和排序参数。

        Returns:
            一个分页结果对象 (`PaginatedPaperSearchResult` 或 `PaginatedHFModelSearchResult`)。
        """
        # 记录开始执行混合搜索的信息
        logger.info(
            f"执行混合搜索: 查询='{query}', 目标='{target}', 页码={page}, 页大小={page_size}"
        )

        # --- 准备和验证 ---
        # 计算分页偏移量
        skip = (page - 1) * page_size

        # 处理空查询
        if not query or not query.strip():
            logger.warning("尝试使用空查询进行混合搜索。")
            # 根据 target 返回对应的空分页模型
            if target == "papers":
                return PaginatedPaperSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )
            else: # target == "models" 或其他 (虽然理论上只有 models)
                return PaginatedHFModelSearchResult(
                    items=[], total=0, skip=skip, limit=page_size
                )

        # --- 从 filters 对象提取过滤和排序参数 ---
        # 初始化过滤器变量
        published_after: Optional[date] = None
        published_before: Optional[date] = None
        filter_area: Optional[List[str]] = None
        pipeline_tag: Optional[str] = None # 模型特有的过滤器
        filter_authors: Optional[List[str]] = None # 论文特有
        filter_library_name: Optional[str] = None # 模型特有
        filter_tags: Optional[List[str]] = None # 模型特有
        filter_author: Optional[str] = None # 模型特有
        # 修复 mypy 错误：使用不同的变量名接收初始的 Optional[str]
        sort_by_from_filter: Optional[str] = None
        sort_order: SortOrderLiteral = "desc" # 默认降序

        # 如果提供了 filters 对象，则从中提取值覆盖默认值
        if filters:
            published_after = filters.published_after
            published_before = filters.published_before
            filter_area = filters.filter_area
            pipeline_tag = filters.pipeline_tag
            filter_authors = filters.filter_authors
            filter_library_name = filters.filter_library_name
            filter_tags = filters.filter_tags
            filter_author = filters.filter_author
            # 将 Optional[str] 赋值给 sort_by_from_filter
            sort_by_from_filter = filters.sort_by
            # 确保 sort_order 是有效值 ('asc' 或 'desc')
            if filters.sort_order and filters.sort_order in get_args(SortOrderLiteral):
                sort_order = filters.sort_order

        # --- 根据目标调用相应的内部混合搜索方法 ---
        if target == "papers":
            # 调用论文混合搜索实现
            # 在调用时将 Optional[str] 转换为 Optional[PaperSortByLiteral]
            # 内部方法会处理 None 和验证
            return await self._perform_hybrid_search_papers(
                query=query,
                page=page,
                page_size=page_size,
                published_after=published_after,
                published_before=published_before,
                filter_area=filter_area,
                filter_authors=filter_authors,
                # 使用 cast 明确 sort_by 类型
                sort_by=cast(Optional[PaperSortByLiteral], sort_by_from_filter),
                sort_order=sort_order,
                filters=filters,
            )
        elif target == "models":
            # 调用模型混合搜索实现
            # 在调用时将 Optional[str] 转换为 Optional[ModelSortByLiteral]
            return await self._perform_hybrid_search_models(
                query=query,
                page=page,
                page_size=page_size,
                pipeline_tag=pipeline_tag,
                filter_library_name=filter_library_name,
                filter_tags=filter_tags,
                filter_author=filter_author,
                # 使用 cast 明确 sort_by 类型
                sort_by=cast(Optional[ModelSortByLiteral], sort_by_from_filter),
                sort_order=sort_order,
                filters=filters,
            )
        else:
            # 处理无效的 target
            logger.error(f"[perform_hybrid_search] 不支持的目标: {target}")
            return PaginatedSemanticSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

    async def _perform_hybrid_search_papers(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10,
        published_after: Optional[date] = None,
        published_before: Optional[date] = None,
        filter_area: Optional[List[str]] = None,
        filter_authors: Optional[List[str]] = None, # 添加作者过滤器参数
        sort_by: Optional[PaperSortByLiteral] = None, # 现在从外部传入
        sort_order: SortOrderLiteral = "desc", # 现在从外部传入
        filters: Optional[SearchFilterModel] = None, # 接收完整的 filters 对象
    ) -> PaginatedPaperSearchResult:
        """
        内部方法，专门为论文执行混合搜索（语义 + 关键词）。

        Args:
            query, page, page_size: 基本搜索和分页参数。
            published_after, published_before, filter_area, filter_authors: 论文特定的过滤条件。
            sort_by, sort_order: 从外部（`perform_hybrid_search`）传入的排序参数。
            filters: 完整的 `SearchFilterModel` 对象，可能包含此方法未直接使用的其他过滤器，
                     但传递它可以方便未来扩展或在排序/过滤逻辑中引用。

        Returns:
            `PaginatedPaperSearchResult`: 包含混合搜索结果的分页对象。
        """

        # 计算分页参数
        skip = (page - 1) * page_size

        # --- 步骤 1: 执行语义搜索 (获取 ID 和分数) ---
        semantic_results_map: Dict[int, float] = {} # 存储 paper_id -> semantic_score
        try:
            # 检查嵌入器是否可用
            if self.embedder is None:
                logger.warning("[_perform_hybrid_search_papers] 嵌入器不可用，无法执行语义搜索部分。")
            else:
                # 生成查询嵌入向量
                embedding = self.embedder.embed(query)
                if embedding is not None:
                    # 调用 Faiss 仓库进行相似度搜索
                    # 获取比 page_size 更多的结果 (例如 30 或 DEFAULT_TOP_N_SEMANTIC)，以便后续 RRF 和过滤有足够数据
                    semantic_results_raw = await self.faiss_repo_papers.search_similar(
                        embedding,
                        k=self.DEFAULT_TOP_N_SEMANTIC, # 使用常量
                    )
                    # 将 (faiss_id, distance) 转换为 (paper_id, score) 的字典
                    semantic_results_map = {
                        # 确保 ID 是整数
                        int(paper_id): self._convert_distance_to_score(distance)
                        for paper_id, distance in semantic_results_raw
                        # 确保 paper_id 不是 None 且可以转换为 int
                        if paper_id is not None and isinstance(paper_id, (int, np.integer)) # Faiss 可能返回 numpy int
                    }
                    logger.debug(f"语义搜索找到 {len(semantic_results_map)} 个初始结果。")
                else:
                     logger.error("[_perform_hybrid_search_papers] 生成查询嵌入失败。")
        except Exception as e:
            # 记录语义搜索部分的错误，但不中断流程
            logger.error(
                f"[_perform_hybrid_search_papers] 语义搜索部分出错: {e}", exc_info=True
            )

        # --- 步骤 2: 执行关键词搜索 (获取 ID 和初步详情) ---
        keyword_results_list: List[Dict[str, Any]] = [] # 存储 PG 返回的原始字典列表
        keyword_ids_ordered: List[int] = [] # 按 PG 返回顺序存储 ID，用于后续计算关键词排名
        try:
            # 调用 PG 仓库的关键词搜索方法
            # 同样获取更多结果 (例如 30 或 DEFAULT_TOP_N_KEYWORD)
            # 注意：这里的 sort_by/sort_order 是 PG 内部的初步排序，RRF 会重新计算分数和排名
            # 传递日期和领域过滤器给 PG，让它先做一轮过滤
            keyword_results_list, _ = await self.pg_repo.search_papers_by_keyword(
                query=query,
                limit=self.DEFAULT_TOP_N_KEYWORD, # 使用常量
                skip=0, # 从头开始获取
                # 传递过滤器给 PG
                published_after=published_after,
                published_before=published_before,
                filter_area=filter_area,
                filter_authors=filter_authors, # 传递作者过滤器
                # 可以指定一个默认的 PG 排序，比如按日期，即使 RRF 会覆盖分数
                sort_by="published_date",
                sort_order="desc",
            )

            # 提取关键词搜索结果的 ID，并保持 PG 返回的顺序
            for result in keyword_results_list:
                 paper_id_val = result.get("paper_id")
                 if paper_id_val is not None:
                     try:
                         keyword_ids_ordered.append(int(paper_id_val))
                     except ValueError:
                         logger.warning(f"无法将关键词搜索结果的 paper_id '{paper_id_val}' 转换为整数。")

            logger.debug(f"关键词搜索找到 {len(keyword_ids_ordered)} 个初始结果。")

        except Exception as e:
            # 记录关键词搜索部分的错误
            logger.error(
                f"[_perform_hybrid_search_papers] 关键词搜索部分出错: {e}", exc_info=True
            )

        # --- 步骤 3: 合并结果 ID ---
        # 获取所有唯一 ID (来自语义搜索和关键词搜索)
        semantic_ids = set(semantic_results_map.keys())
        keyword_ids = set(keyword_ids_ordered) # 从有序列表创建集合以去重
        all_combined_ids: Set[int] = semantic_ids.union(keyword_ids)

        # 如果两种搜索都没有结果，直接返回空
        if not all_combined_ids:
            logger.info("[_perform_hybrid_search_papers] 语义和关键词搜索均无结果。")
            return PaginatedPaperSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- 步骤 4: 获取所有合并后 ID 的论文详情 ---
        # (即使关键词搜索已返回部分详情，也需要批量获取所有 ID 的最新详情，确保数据一致性)
        all_paper_details_map: Dict[int, Dict[str, Any]] = {} # paper_id -> details_dict
        try:
            paper_details_list = await self.pg_repo.get_papers_details_by_ids(
                list(all_combined_ids) # 将集合转为列表传入
            )
            # 修复 mypy 错误：使用显式循环创建映射，并在调用 int() 前检查 None
            for details in paper_details_list:
                paper_id_val = details.get("paper_id")
                if paper_id_val is not None:
                    try:
                        paper_id_int = int(paper_id_val) # 安全转换
                        all_paper_details_map[paper_id_int] = details
                    except (ValueError, TypeError):
                         logger.warning(f"无法将论文详情中的 paper_id '{paper_id_val}' 转换为整数。")

            logger.debug(f"获取了 {len(all_paper_details_map)} 篇论文的详情。")
        except Exception as fetch_error:
            logger.error(
                f"[_perform_hybrid_search_papers] 获取合并后的论文详情时出错: {fetch_error}",
                exc_info=True,
            )
            # 如果获取详情失败，无法继续，返回空结果
            return PaginatedPaperSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- 步骤 5: 使用 RRF (Reciprocal Rank Fusion) 合并结果并计算融合分数 ---
        # RRF 通过结合不同搜索方法的排名来产生最终排名

        # 确定实际有详情的 ID 集合 (可能比 all_combined_ids 少)
        valid_ids_with_details = set(all_paper_details_map.keys())
        if len(valid_ids_with_details) < len(all_combined_ids):
             logger.warning(f"合并后的 {len(all_combined_ids) - len(valid_ids_with_details)} 个 ID 没有找到对应的详情。")


        # 计算语义搜索结果的排名 (仅针对有详情的 ID)
        semantic_ranks: Dict[int, int] = {} # paper_id -> rank (从 1 开始)
        if semantic_results_map:
            # 1. 过滤掉没有详情的语义结果
            valid_semantic_results = {
                pid: score
                for pid, score in semantic_results_map.items()
                if pid in valid_ids_with_details
            }
            # 2. 按分数降序排序
            sorted_semantic_results = sorted(
                valid_semantic_results.items(), key=lambda item: item[1], reverse=True
            )
            # 3. 生成排名 (rank 从 1 开始)
            semantic_ranks = {
                pid: rank + 1 for rank, (pid, _) in enumerate(sorted_semantic_results)
            }

        # 计算关键词搜索结果的排名 (仅针对有详情的 ID)
        keyword_ranks: Dict[int, int] = {} # paper_id -> rank (从 1 开始)
        if keyword_ids_ordered:
            # 1. 过滤掉没有详情的关键词 ID，并保持原始顺序
            valid_keyword_ids_ordered = [
                 pid for pid in keyword_ids_ordered if pid in valid_ids_with_details
            ]
            # 2. 生成排名 (rank 从 1 开始)
            keyword_ranks = {
                pid: rank + 1 for rank, pid in enumerate(valid_keyword_ids_ordered)
            }

        # 计算 RRF 融合分数 (仅针对有详情的 ID)
        combined_scores: Dict[int, Optional[float]] = {} # paper_id -> rrf_score (或 None)
        # 检查是否只有关键词结果（并且这些结果都有详情）
        # 注意：这里不能简单地检查 semantic_ranks 是否为空，因为可能语义搜索有结果但那些结果恰好没详情
        # 一个更可靠的判断可能是检查 valid_semantic_results 是否为空
        has_valid_semantic_results = bool(semantic_ranks) # 如果 semantic_ranks 有内容，说明至少有一个语义结果有详情
        has_valid_keyword_results = bool(keyword_ranks) # 如果 keyword_ranks 有内容，说明至少有一个关键词结果有详情

        is_keyword_only_effective = has_valid_keyword_results and not has_valid_semantic_results

        rrf_k = self.DEFAULT_RRF_K # RRF 的 k 参数

        for paper_id in valid_ids_with_details: # 只迭代有详情的 ID
            rrf_score = 0.0
            sem_rank = semantic_ranks.get(paper_id)
            kw_rank = keyword_ranks.get(paper_id)

            # 应用 RRF 公式: score = 1 / (k + rank1) + 1 / (k + rank2) + ...
            if sem_rank is not None:
                rrf_score += 1.0 / (rrf_k + sem_rank)
            if kw_rank is not None:
                rrf_score += 1.0 / (rrf_k + kw_rank)

            # 存储计算出的 RRF 分数
            if rrf_score > 0:
                 # 如果实际上只有有效的关键词结果，混合分数没有意义，设为 None
                if is_keyword_only_effective:
                     combined_scores[paper_id] = None
                else:
                     combined_scores[paper_id] = rrf_score
            # 如果分数是 0 (意味着该 ID 在两个有效排名中都不存在)，则不记录

        logger.debug(f"为 {len(combined_scores)} 篇论文计算了 RRF 分数。")

        # --- 步骤 6: 创建 SearchResultItem 对象列表 ---
        # 使用获取到的详情和计算出的 RRF 分数构建最终的结果项列表
        all_items: List[SearchResultItem] = []
        for paper_id in valid_ids_with_details: # 再次确保只处理有详情的 ID
            # 获取详情字典
            details_optional = all_paper_details_map.get(paper_id)
            if not details_optional: # 理论上不会发生，因为是基于这个 map 的 key 迭代的
                 continue
            # 修复 Linter 错误：使用 cast 明确 details 不为 None
            details = cast(Dict[str, Any], details_optional)

            # 安全地处理 'authors' 字段 (可能为 JSON str)
            authors = details.get("authors", [])
            if isinstance(authors, str):
                try:
                    authors = json.loads(authors)
                except (json.JSONDecodeError, TypeError):
                    authors = [] # 出错时设为空列表

            # 尝试创建 SearchResultItem 实例
            try:
                item = SearchResultItem(
                    paper_id=paper_id,
                    pwc_id=details.get("pwc_id", ""),
                    title=details.get("title", ""),
                    summary=details.get("summary", ""),
                    # 使用 RRF 计算出的分数，如果不存在则为 None
                    score=combined_scores.get(paper_id),
                    pdf_url=details.get("pdf_url", ""),
                    published_date=details.get("published_date"),
                    authors=authors, # 使用处理后的列表
                    area=details.get("area", ""),
                )
                all_items.append(item)
            except ValidationError as ve:
                logger.error(
                    f"[_perform_hybrid_search_papers] 创建 SearchResultItem (paper_id={paper_id}) 时发生验证错误: {ve}"
                )
            except Exception as e:
                logger.error(
                    f"[_perform_hybrid_search_papers] 创建 SearchResultItem (paper_id={paper_id}) 时发生意外错误: {e}"
                )

        # --- 步骤 7: 应用过滤器 --- (过滤 all_items)
        filtered_items: List[SearchResultItem] = all_items

        # 日期过滤
        if published_after or published_before:
            original_count = len(filtered_items)
            filtered_items = [
                item
                for item in filtered_items
                if item.published_date is not None # 确保有日期
                and (published_after is None or item.published_date >= published_after)
                and (published_before is None or item.published_date <= published_before)
            ]
            if len(filtered_items) < original_count:
                 logger.debug(f"混合搜索：日期过滤将结果从 {original_count} 减少到 {len(filtered_items)}")

        # 领域多选过滤
        if filter_area and len(filter_area) > 0:
            original_count = len(filtered_items)
            filter_area_lower = {area.lower() for area in filter_area} # 转换为小写集合以提高效率
            filtered_items = [
                item
                for item in filtered_items
                # 确保 item 有 area 且不为空，然后进行不区分大小写的匹配
                if item.area and item.area.lower() in filter_area_lower
            ]
            if len(filtered_items) < original_count:
                 logger.debug(f"混合搜索：领域过滤将结果从 {original_count} 减少到 {len(filtered_items)}")

        # --- 步骤 8: 排序和分页 ---
        total_items_after_filtering = len(filtered_items)

        # Determine sort key, default to 'score' if not provided or invalid
        final_sort_by: Optional[PaperSortByLiteral] = None
        current_sort_by_from_filter = filters.sort_by if filters else None

        if current_sort_by_from_filter:
            if current_sort_by_from_filter in get_args(PaperSortByLiteral):
                final_sort_by = cast(PaperSortByLiteral, current_sort_by_from_filter)
            else:
                logger.warning(
                    f"[_perform_hybrid_search_papers] Invalid sort_by '{current_sort_by_from_filter}' in filter for hybrid paper search. Defaulting to 'score'."
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

        # --- 步骤 9: 返回分页结果 ---
        return PaginatedPaperSearchResult(
            items=paginated_items_list,
            total=total_items_after_filtering,  # 使用 *过滤后* (分页前) 的总数
            skip=calculated_skip,
            limit=calculated_limit,
        )

    async def _perform_hybrid_search_models(
        self,
        query: str,
        page: int = 1,
        page_size: int = 10,
        # 模型特定过滤器
        pipeline_tag: Optional[str] = None,
        filter_library_name: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
        filter_author: Optional[str] = None,
        # 排序参数
        sort_by: Optional[ModelSortByLiteral] = None, # 从外部传入
        sort_order: SortOrderLiteral = "desc", # 从外部传入
        filters: Optional[SearchFilterModel] = None, # 完整的 filters 对象
    ) -> PaginatedHFModelSearchResult:
        """
        内部方法，专门为模型执行混合搜索（语义 + 关键词）。

        流程与 `_perform_hybrid_search_papers` 类似，但使用模型相关的仓库、
        ID 类型、过滤器和结果模型。

        Args:
            query, page, page_size: 基本搜索和分页参数。
            pipeline_tag, filter_library_name, filter_tags, filter_author: 模型特定的过滤条件。
            sort_by, sort_order: 从外部传入的排序参数。
            filters: 完整的 `SearchFilterModel` 对象。

        Returns:
            `PaginatedHFModelSearchResult`: 包含混合搜索结果的分页对象。
        """

        # 计算分页参数
        skip = (page - 1) * page_size

        # --- 步骤 1: 执行语义搜索 (获取 ID 和分数) ---
        semantic_results_map: Dict[str, float] = {} # 存储 model_id -> semantic_score
        try:
            if self.embedder is None:
                logger.warning("[_perform_hybrid_search_models] 嵌入器不可用，无法执行语义搜索部分。")
            else:
                embedding = self.embedder.embed(query)
                if embedding is not None:
                    # 调用模型 Faiss 仓库
                    semantic_results_raw = await self.faiss_repo_models.search_similar(
                        embedding,
                        k=self.DEFAULT_TOP_N_SEMANTIC, # 获取足够多的初始结果
                    )
                    # 将 (faiss_id, distance) 转换为 (model_id, score) 的字典
                    semantic_results_map = {
                        # 确保 ID 是字符串
                        str(model_id): self._convert_distance_to_score(distance)
                        for model_id, distance in semantic_results_raw
                        if model_id is not None # 确保 model_id 不是 None
                    }
                    logger.debug(f"语义搜索找到 {len(semantic_results_map)} 个初始模型结果。")
                else:
                     logger.error("[_perform_hybrid_search_models] 生成查询嵌入失败。")
        except Exception as e:
            logger.error(
                f"[_perform_hybrid_search_models] 语义搜索部分出错: {e}", exc_info=True
            )

        # --- 步骤 2: 执行关键词搜索 (获取 ID 和初步详情) ---
        keyword_results_list: List[Dict[str, Any]] = [] # 存储 PG 返回的原始字典
        keyword_ids_ordered: List[str] = [] # 按 PG 顺序存储模型 ID
        try:
            # 调用 PG 仓库的模型关键词搜索方法
            # 传递模型相关的过滤器
            keyword_results_list, _ = await self.pg_repo.search_models_by_keyword(
                query=query,
                limit=self.DEFAULT_TOP_N_KEYWORD, # 获取足够多的初始结果
                skip=0,
                # 传递过滤器给 PG
                pipeline_tag=pipeline_tag,
                filter_library_name=filter_library_name,
                filter_tags=filter_tags,
                filter_author=filter_author,
                # 同样可以指定 PG 内部的默认排序
                sort_by=cast(Optional[Literal["likes", "downloads", "last_modified"]],"last_modified"),
                sort_order="desc",
            )

            # 提取关键词搜索结果的模型 ID，保持顺序
            for result in keyword_results_list:
                 model_id_val = result.get("hf_model_id") # PG 返回的是 hf_model_id
                 if model_id_val is not None:
                      keyword_ids_ordered.append(str(model_id_val)) # 确保是字符串

            logger.debug(f"关键词搜索找到 {len(keyword_ids_ordered)} 个初始模型结果。")

        except Exception as e:
            logger.error(
                f"[_perform_hybrid_search_models] 关键词搜索部分出错: {e}", exc_info=True
            )

        # --- 步骤 3: 合并结果 ID ---
        semantic_ids = set(semantic_results_map.keys())
        keyword_ids = set(keyword_ids_ordered)
        all_combined_ids: Set[str] = semantic_ids.union(keyword_ids)

        if not all_combined_ids:
            logger.info("[_perform_hybrid_search_models] 语义和关键词搜索均无结果。")
            return PaginatedHFModelSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- 步骤 4: 获取所有合并后 ID 的模型详情 ---
        all_model_details_map: Dict[str, Dict[str, Any]] = {} # model_id -> details_dict
        try:
            # 调用 PG 仓库批量获取模型详情
            model_details_list = await self.pg_repo.get_hf_models_by_ids(
                list(all_combined_ids)
            )
            # 创建 ID 到详情的映射
            all_model_details_map = {
                str(details.get("hf_model_id")): details # 确保 key 是 str
                for details in model_details_list
                if details.get("hf_model_id") is not None
            }
            logger.debug(f"获取了 {len(all_model_details_map)} 个模型的详情。")
        except Exception as fetch_error:
            logger.error(
                f"[_perform_hybrid_search_models] 获取合并后的模型详情时出错: {fetch_error}",
                exc_info=True,
            )
            return PaginatedHFModelSearchResult(
                items=[], total=0, skip=skip, limit=page_size
            )

        # --- 步骤 5: 使用 RRF 合并结果并计算融合分数 ---
        valid_ids_with_details = set(all_model_details_map.keys())
        if len(valid_ids_with_details) < len(all_combined_ids):
             logger.warning(f"合并后的 {len(all_combined_ids) - len(valid_ids_with_details)} 个模型 ID 没有找到对应的详情。")

        # 计算语义排名 (只对有详情的 ID)
        semantic_ranks: Dict[str, int] = {} # model_id -> rank
        if semantic_results_map:
            valid_semantic_results = {
                mid: score
                for mid, score in semantic_results_map.items()
                if mid in valid_ids_with_details
            }
            sorted_semantic_results = sorted(
                valid_semantic_results.items(), key=lambda item: item[1], reverse=True
            )
            semantic_ranks = {
                mid: rank + 1 for rank, (mid, _) in enumerate(sorted_semantic_results)
            }

        # 计算关键词排名 (只对有详情的 ID)
        keyword_ranks: Dict[str, int] = {} # model_id -> rank
        if keyword_ids_ordered:
            valid_keyword_ids_ordered = [
                mid for mid in keyword_ids_ordered if mid in valid_ids_with_details
            ]
            keyword_ranks = {
                mid: rank + 1 for rank, mid in enumerate(valid_keyword_ids_ordered)
            }

        # 计算融合分数
        combined_scores: Dict[str, Optional[float]] = {} # model_id -> rrf_score (或 None)
        has_valid_semantic_results = bool(semantic_ranks)
        has_valid_keyword_results = bool(keyword_ranks)
        is_keyword_only_effective = has_valid_keyword_results and not has_valid_semantic_results

        rrf_k = self.DEFAULT_RRF_K

        for model_id in valid_ids_with_details:
            rrf_score = 0.0
            sem_rank = semantic_ranks.get(model_id)
            kw_rank = keyword_ranks.get(model_id)

            if sem_rank is not None:
                rrf_score += 1.0 / (rrf_k + sem_rank)
            if kw_rank is not None:
                rrf_score += 1.0 / (rrf_k + kw_rank)

            if rrf_score > 0:
                if is_keyword_only_effective:
                     combined_scores[model_id] = None # 纯关键词结果，分数设为 None
                else:
                     combined_scores[model_id] = rrf_score

        logger.debug(f"为 {len(combined_scores)} 个模型计算了 RRF 分数。")


        # --- 步骤 6: 创建 HFSearchResultItem 对象列表 ---
        all_items: List[HFSearchResultItem] = []
        for model_id in valid_ids_with_details:
            details = all_model_details_map.get(model_id)
            if not details:
                 continue

            # 安全地处理字段并创建实例
            try:
                # 处理 tags (可能为 JSON str)
                tags = details.get("hf_tags", []) # 从详情字典获取
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []

                # 处理 last_modified (需要转为 datetime 对象)
                final_last_modified_dt: Optional[datetime] = None
                last_modified_val = details.get("hf_last_modified")
                if isinstance(last_modified_val, (datetime, date)):
                    # 确保是 datetime 对象
                    final_last_modified_dt = (
                        last_modified_val
                        if isinstance(last_modified_val, datetime)
                        else datetime.combine(last_modified_val, datetime.min.time())
                    )
                    # 确保时区感知 (假设 UTC)
                    if final_last_modified_dt.tzinfo is None:
                        final_last_modified_dt = final_last_modified_dt.replace(
                            tzinfo=timezone.utc
                        )
                elif isinstance(last_modified_val, str):
                    try:
                        dt_obj = datetime.fromisoformat(last_modified_val.replace("Z", "+00:00"))
                        if dt_obj.tzinfo is None:
                            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                        final_last_modified_dt = dt_obj
                    except ValueError:
                        pass # 解析失败则为 None

                # 修复 mypy 错误：在调用 int() 前检查 None
                likes_val = details.get("hf_likes")
                likes_int = int(likes_val) if likes_val is not None else None

                downloads_val = details.get("hf_downloads")
                downloads_int = int(downloads_val) if downloads_val is not None else None

                # 创建 HFSearchResultItem 实例
                item = HFSearchResultItem(
                    model_id=model_id, # 使用迭代的 model_id
                    author=str(details.get("hf_author", "")),
                    pipeline_tag=str(details.get("hf_pipeline_tag", "")),
                    library_name=str(details.get("hf_library_name", "")),
                    tags=tags, # 使用处理后的列表
                    likes=likes_int, # 使用安全转换后的 int 或 None
                    downloads=downloads_int, # 使用安全转换后的 int 或 None
                    last_modified=final_last_modified_dt, # 使用处理后的 datetime
                    score=combined_scores.get(model_id), # 使用 RRF 分数
                )
                all_items.append(item)
            except ValidationError as ve:
                logger.error(
                    f"[_perform_hybrid_search_models] 创建 HFSearchResultItem (model_id={model_id}) 时发生验证错误: {ve}"
                )
            except Exception as e:
                logger.error(
                    f"[_perform_hybrid_search_models] 创建 HFSearchResultItem (model_id={model_id}) 时发生意外错误: {e}"
                )

        # --- 步骤 7: 应用过滤器 (在 RRF 之后，排序之前) ---
        filtered_items: List[HFSearchResultItem] = all_items

        # pipeline_tag 过滤
        if pipeline_tag:
            original_count = len(filtered_items)
            pipeline_tag_lower = pipeline_tag.lower()
            filtered_items = [
                item
                for item in filtered_items
                # 确保 item 有 pipeline_tag 且不为空，然后进行不区分大小写的比较
                if item.pipeline_tag and item.pipeline_tag.lower() == pipeline_tag_lower
            ]
            if len(filtered_items) < original_count:
                 logger.debug(f"混合搜索：pipeline_tag 过滤将结果从 {original_count} 减少到 {len(filtered_items)}")


        # 应用 library_name 过滤器
        if filter_library_name:
             original_count = len(filtered_items)
             filter_library_name_lower = filter_library_name.lower()
             filtered_items = [
                 item
                 for item in filtered_items
                 if item.library_name and item.library_name.lower() == filter_library_name_lower
             ]
             if len(filtered_items) < original_count:
                 logger.debug(f"混合搜索：library_name 过滤将结果从 {original_count} 减少到 {len(filtered_items)}")

        # 应用 tags 过滤器 (需要匹配所有指定的标签)
        if filter_tags:
             original_count = len(filtered_items)
             filter_tags_lower = {tag.lower() for tag in filter_tags}
             filtered_items = [
                 item
                 for item in filtered_items
                 # 确保 item 有 tags 列表
                 if item.tags
                 # 检查 item 的 tags (转为小写集合) 是否包含所有过滤标签
                 and filter_tags_lower.issubset({tag.lower() for tag in item.tags})
             ]
             if len(filtered_items) < original_count:
                 logger.debug(f"混合搜索：tags 过滤将结果从 {original_count} 减少到 {len(filtered_items)}")

        # 应用 author 过滤器
        if filter_author:
             original_count = len(filtered_items)
             filter_author_lower = filter_author.lower()
             filtered_items = [
                 item
                 for item in filtered_items
                 if item.author and item.author.lower() == filter_author_lower
             ]
             if len(filtered_items) < original_count:
                 logger.debug(f"混合搜索：author 过滤将结果从 {original_count} 减少到 {len(filtered_items)}")


        # --- 步骤 8: 排序和分页 ---
        total_items_after_filtering = len(filtered_items)

        # 确定最终排序字段，默认为 'score'
        final_sort_by: Optional[ModelSortByLiteral] = None
        current_sort_by_from_filter = filters.sort_by if filters else None

        if current_sort_by_from_filter:
            if current_sort_by_from_filter in get_args(ModelSortByLiteral):
                final_sort_by = cast(ModelSortByLiteral, current_sort_by_from_filter)
                if final_sort_by != 'score' and is_keyword_only_effective:
                     logger.warning(f"结果仅来自关键词搜索，但排序依据为 '{final_sort_by}' 而不是 'score'。排序可能不符合预期。")
            else:
                logger.warning(
                    f"[_perform_hybrid_search_models] Filter 中提供了无效的排序键 '{current_sort_by_from_filter}'。将默认使用 'score'。"
                )
                final_sort_by = "score"
        else:
            final_sort_by = "score" # 混合搜索默认按 RRF 分数排序

        # 确定最终排序顺序
        final_sort_order: SortOrderLiteral = "desc"
        if (
            filters
            and filters.sort_order
            and filters.sort_order in get_args(SortOrderLiteral)
        ):
            final_sort_order = filters.sort_order

        # 调用排序和分页辅助方法
        (
            paginated_items_list_uncasted,
            _,
            calculated_skip,
            calculated_limit,
        ) = self._apply_sorting_and_pagination(
            filtered_items, # 对过滤后的结果排序分页
            sort_by=final_sort_by,
            sort_order=final_sort_order,
            page=page,
            page_size=page_size,
        )

        # 转换类型
        paginated_items_list = cast(
            List[HFSearchResultItem], paginated_items_list_uncasted
        )

        # --- 步骤 9: 返回分页结果 ---
        return PaginatedHFModelSearchResult(
            items=paginated_items_list,
            total=total_items_after_filtering, # 使用过滤后的总数
            skip=calculated_skip,
            limit=calculated_limit,
        )

    async def get_available_paper_areas(self) -> List[str]:
        """
        获取系统中所有可用的、唯一的论文领域（Area）列表。
        这个方法通常被 API 端点调用，用于给前端提供领域过滤的选项。

        Returns:
            List[str]: 一个包含所有唯一论文领域名称的列表，按字母顺序排序。
                       如果获取失败或没有领域，则返回空列表。
        """
        try:
            # 检查 PostgreSQL 仓库是否有获取唯一领域的方法
            if hasattr(self.pg_repo, "get_unique_paper_areas"):
                # 调用仓库方法
                areas = await self.pg_repo.get_unique_paper_areas()
                # 过滤掉 None 或空字符串
                valid_areas = [area for area in areas if area]
                # 记录获取到的领域数量
                logger.info(f"获取到 {len(valid_areas)} 个有效的唯一论文领域。")
                # 返回排序后的列表
                return sorted(valid_areas)
            else:
                # 如果方法不存在，记录错误
                logger.error("PostgreSQL 仓库中缺少 get_unique_paper_areas 方法。")
                return []
        except Exception as e:
            # 如果在获取过程中发生任何错误，记录日志
            logger.error(f"获取论文领域列表时出错: {e}", exc_info=True)
            # 返回空列表
            return []
