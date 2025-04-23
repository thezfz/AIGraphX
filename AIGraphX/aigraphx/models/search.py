"""
搜索相关 Pydantic 模型 (Search Related Pydantic Models)

功能 (Function):
这个模块定义了与搜索功能相关的 Pydantic 模型，主要用于：
1.  **API 响应序列化**: 定义搜索结果的结构，包括单个结果项 (`SearchResultItem`, `HFSearchResultItem`) 和分页后的整体响应 (`PaginatedPaperSearchResult`, `PaginatedSemanticSearchResult`, `PaginatedHFModelSearchResult`, 通用 `PaginatedModel`)。FastAPI 使用这些模型来序列化搜索接口的返回数据。
2.  **API 请求体验证**: (虽然此文件主要用于响应) 将来可能定义用于接收搜索请求参数的模型，例如包含查询词、过滤条件等的请求体模型。
3.  **搜索过滤与排序**: 定义 `SearchFilterModel`，用于结构化地表示搜索的过滤条件（如日期范围、领域、作者、标签等）和排序偏好（排序字段、排序顺序）。
4.  **类型定义与约束**: 定义支持的搜索类型 (`SearchType`)，并使用 Pydantic 的验证功能（如 `@field_validator`）确保数据（特别是日期时间）的格式正确。

交互 (Interaction):
- 依赖 (Depends on):
    - `pydantic`: 用于定义模型、字段、验证器等。
    - `typing`: 用于提供类型提示 (List, Optional, Literal, Union, Any, TypeVar, Generic)。
    - `datetime`: 用于处理日期和时间类型，特别是时区处理。
- 被导入 (Imported by):
    - `aigraphx.api.v1.endpoints.search`: 搜索相关的 API 端点会使用这些模型作为 `response_model` 来序列化响应，并可能将查询参数解析到 `SearchFilterModel` 或类似模型中。
    - `aigraphx.services.search_service`: 搜索服务层可能会构建并返回这些模型的实例，也可能接收 `SearchFilterModel` 作为参数来执行过滤和排序逻辑。

设计原则 (Design Principles):
- **清晰的搜索结果结构 (Clear Search Result Structure):** 为不同类型的搜索目标（论文、HF模型）定义了专门的结果项模型，并提供了包含分页信息的包装模型。
- **灵活的过滤与排序 (Flexible Filtering & Sorting):** `SearchFilterModel` 提供了一种结构化的方式来传递复杂的过滤和排序要求。
- **类型安全 (Type Safety):** 利用 Pydantic 和 typing 确保数据类型正确，特别是日期和枚举类型。
- **可重用性 (Reusability):** 定义了通用的分页模型 `PaginatedModel` 和结果项联合类型 `AnySearchResultItem`，提高了代码的复用性。
"""

# 导入 Pydantic 核心组件
from pydantic import BaseModel, Field, field_validator, ConfigDict

# 导入 typing 模块提供类型提示
# Literal: 用于定义枚举类型，变量值必须是指定的字面量之一
# Union: 用于表示一个变量可以是多种类型之一
# TypeVar, Generic: 用于定义泛型类 (PaginatedModel)
from typing import List, Optional, Literal, Union, Any, TypeVar, Generic

# 导入 datetime 模块处理日期和时间，timezone 用于处理时区
from datetime import date, datetime, timezone

# 定义支持的搜索类型常量
# 使用 Literal 来限制 search_type 参数只能是这三个字符串之一
SearchType = Literal["semantic", "keyword", "hybrid"]


class SearchResultItem(BaseModel):
    """
    表示搜索结果中的单个论文条目。
    """

    # 论文在 PostgreSQL 中的内部 ID (可选，可能不总是需要返回)
    paper_id: Optional[int] = None
    # Papers with Code ID，通常是主要标识符，必需
    pwc_id: str
    # 论文标题 (可选)
    title: Optional[str] = None
    # 论文摘要或用于搜索结果展示的简短总结 (可选)
    summary: Optional[str] = None
    # 搜索结果的相关性得分 (可选)
    # 对于语义搜索或混合搜索，这通常是一个 0 到 1 之间的浮点数，值越高表示越相关。
    # 对于纯关键词搜索，可能没有明确的相关性得分，因此设为可选，默认为 None。
    score: Optional[float] = Field(
        default=None,
        description="相关性得分 (语义/混合搜索范围 0-1，越高越好；纯关键词搜索可能为 None)。",
        ge=0.0,  # (可选约束) 确保分数大于等于 0
        le=1.0,  # (可选约束) 确保分数小于等于 1
    )
    # 论文 PDF 链接 (可选)
    pdf_url: Optional[str] = None
    # 发表日期 (可选)
    published_date: Optional[date] = None
    # 作者列表 (可选)，假设是解码后的字符串列表
    authors: Optional[List[str]] = None  # Assuming decoded list
    # 论文所属领域 (可选)
    area: Optional[str] = None  # Add area field

    # 模型配置
    model_config = ConfigDict(from_attributes=True)


class HFSearchResultItem(BaseModel):
    """
    表示搜索结果中的单个 Hugging Face 模型条目。
    """

    # Hugging Face 模型 ID，必需
    model_id: str = Field(..., description="唯一的 Hugging Face 模型 ID。")
    # 作者或组织 (可选)
    author: Optional[str] = Field(None, description="作者或组织。")
    # 主要任务标签 (可选)
    pipeline_tag: Optional[str] = Field(None, description="主要的任务流水线标签。")
    # 关联库名称 (可选)
    library_name: Optional[str] = Field(
        None, description="模型关联的库 (例如, transformers)。"
    )
    # 标签列表 (可选)
    tags: Optional[List[str]] = Field(None, description="与模型关联的标签列表。")
    # 点赞数 (可选)
    likes: Optional[int] = Field(None, description="点赞次数。")
    # 下载数 (可选)
    downloads: Optional[int] = Field(None, description="下载次数。")
    # 最后修改时间戳 (可选)
    last_modified: Optional[datetime] = Field(None, description="最后修改时间戳。")
    # 相关性得分 (可选)，例如来自向量搜索
    score: Optional[float] = Field(
        None,
        description="搜索的相关性得分 (0.0 到 1.0)。",
        ge=0.0,  # (可选约束)
        le=1.0,  # (可选约束)
    )

    # 自定义验证器，用于解析和验证 last_modified 字段
    # 确保返回的是带有时区的 datetime 对象 (UTC)
    @field_validator("last_modified", mode="before")
    @classmethod
    def parse_last_modified(cls, v: Any) -> Optional[datetime]:
        """
        解析并验证 'last_modified' 字段，确保返回带时区的 datetime 对象 (UTC)。

        处理字符串、无时区的 datetime 和带时区的 datetime。
        """
        # 如果输入是字符串
        if isinstance(v, str):
            try:
                # 替换 'Z' 并解析为 datetime 对象
                dt_parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
                # 确保结果是时区感知的 (如果解析结果没有时区信息，则假定为 UTC)
                # fromisoformat 在解析 '+00:00' 时会自动设置 UTC 时区
                # return dt_parsed # 直接返回即可，因为它已经是时区感知的
                # 为了更健壮，显式检查并设置
                if dt_parsed.tzinfo is None:
                    return dt_parsed.replace(tzinfo=timezone.utc)
                else:
                    # 如果已经有时区，确保它是 UTC 或转换为 UTC (根据需要)
                    return dt_parsed.astimezone(timezone.utc)
            except ValueError as e:
                # 抛出 ValueError 以便 Pydantic 捕获
                raise ValueError(f"无效的 'last_modified' 日期时间格式: '{v}'") from e
        # 如果输入已经是 datetime 对象
        elif isinstance(v, datetime):
            # 确保它是时区感知的，如果不是则设为 UTC
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            else:
                # 如果已经有时区，确保它是 UTC 或转换为 UTC
                return v.astimezone(timezone.utc)
        # 如果输入是 None
        elif v is None:
            return None
        # 其他类型无效
        else:
            raise ValueError(
                "'last_modified' 必须是有效的 ISO 8601 字符串或 datetime 对象"
            )

    # 模型配置
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


# --- 分页响应模型 (Paginated Response Models) ---


class PaginatedPaperSearchResult(BaseModel):
    """
    用于论文搜索结果的分页响应模型。
    """

    # 当前页的论文搜索结果列表，必需
    items: List[SearchResultItem] = Field(..., description="当前页的论文搜索结果列表。")
    # 匹配查询的总论文数量，必需
    total: int = Field(..., description="匹配查询的总论文数。")
    # 跳过的项目数 (分页偏移量)，必需
    skip: int = Field(..., description="跳过的项目数 (偏移量)。")
    # 每页最大项目数，必需
    limit: int = Field(..., description="每页最大项目数。")

    # 模型配置
    model_config = ConfigDict(from_attributes=True)


# 定义一个联合类型，表示搜索结果项可以是论文或 HF 模型
AnySearchResultItem = Union[SearchResultItem, HFSearchResultItem]


class PaginatedSemanticSearchResult(BaseModel):
    """
    用于语义搜索结果的分页响应模型 (可以包含论文或模型)。
    """

    # 当前页的结果列表，可以是论文或模型的混合，必需
    items: List[AnySearchResultItem] = Field(
        ...,
        description="当前页的语义搜索结果列表 (论文或模型)。",
    )
    # 在分页前找到的总候选结果数，必需
    # 注意：这可能与最终过滤后的总数不同，取决于语义搜索的实现
    total: int = Field(..., description="分页前找到的总候选结果数。")
    # 跳过的项目数，必需
    skip: int = Field(..., description="跳过的项目数 (偏移量)。")
    # 每页最大项目数，必需
    limit: int = Field(..., description="每页最大项目数。")

    # 模型配置
    model_config = ConfigDict(from_attributes=True)


class PaginatedHFModelSearchResult(BaseModel):
    """
    用于 Hugging Face 模型搜索结果的分页响应模型。
    """

    # 当前页的 HF 模型搜索结果列表，必需
    items: List[HFSearchResultItem] = Field(
        ..., description="当前页的 HF 模型搜索结果列表。"
    )
    # 匹配查询的总 HF 模型数量，必需
    total: int = Field(..., description="匹配查询的总 HF 模型数。")
    # 跳过的项目数，必需
    skip: int = Field(..., description="跳过的项目数 (偏移量)。")
    # 每页最大项目数，必需
    limit: int = Field(..., description="每页最大项目数。")

    # 模型配置
    model_config = ConfigDict(from_attributes=True)


# --- 通用分页模型 (Generic Paginated Model) ---

# 定义一个类型变量 T，用于泛型编程
T = TypeVar("T")


# 定义一个泛型分页模型，可以容纳任何类型的列表项
class PaginatedModel(BaseModel, Generic[T]):
    """
    通用的分页响应模型。

    可以使用 `PaginatedModel[SearchResultItem]` 或 `PaginatedModel[HFSearchResultItem]`
    来表示特定类型的分页结果，增加了代码的复用性。
    尤其适用于混合搜索或未来可能出现的其他搜索类型。
    """

    # 当前页的结果列表，类型由泛型参数 T 决定，必需
    items: List[T] = Field(..., description="当前页的搜索结果列表。")
    # 匹配查询的总项目数，必需
    total: int = Field(..., description="匹配查询的总项目数。")
    # 跳过的项目数，必需
    skip: int = Field(..., description="跳过的项目数 (偏移量)。")
    # 每页最大项目数，必需
    limit: int = Field(..., description="每页最大项目数。")

    # 模型配置
    model_config = ConfigDict(from_attributes=True)


# --- 搜索过滤器模型 (Search Filter Model) ---


class SearchFilterModel(BaseModel):
    """
    定义搜索时可用的过滤和排序选项。

    这个模型可以被 API 端点用来解析请求参数（查询参数或请求体），
    然后传递给服务层来执行具体的过滤和排序逻辑。
    所有字段都是可选的，允许用户只指定他们关心的过滤/排序条件。
    """

    # --- 过滤条件 (Filtering Criteria) ---
    # 按论文发表日期过滤 (之后)
    published_after: Optional[date] = Field(
        None, description="筛选在此日期之后发布的论文 (包含该日期)。"
    )
    # 按论文发表日期过滤 (之前)
    published_before: Optional[date] = Field(
        None, description="筛选在此日期之前发布的论文 (包含该日期)。"
    )
    # 按论文研究领域过滤 (支持多选，例如 ["CV", "NLP"])
    filter_area: Optional[List[str]] = Field(
        None, description="按研究领域筛选论文 (例如 CV、NLP 等)，支持多选。"
    )
    # 按 HF 模型任务类型过滤 (例如 "text-generation")
    pipeline_tag: Optional[str] = Field(
        None,
        description="按任务类型筛选模型 (例如 text-generation、image-classification 等)。",
    )
    # 按论文档案作者过滤 (支持模糊匹配，多选表示 OR 关系)
    filter_authors: Optional[List[str]] = Field(
        None, description="按作者名称筛选论文 (模糊匹配，支持多选，任一匹配即可)。"
    )
    # 按 HF 模型库名称过滤 (精确匹配，忽略大小写)
    filter_library_name: Optional[str] = Field(
        None,
        description="按库名称筛选模型 (例如 transformers、diffusers，精确匹配，忽略大小写)。",
    )
    # 按 HF 模型标签过滤 (要求模型包含所有指定的标签)
    filter_tags: Optional[List[str]] = Field(
        None, description="按标签筛选模型 (要求所有提供的标签都存在)。"
    )
    # 按 HF 模型作者/组织过滤 (模糊匹配，忽略大小写)
    filter_author: Optional[str] = Field(
        None, description="按作者/组织名称筛选模型 (模糊匹配，忽略大小写)。"
    )

    # --- 排序选项 (Sorting Options) ---
    # 指定排序依据的字段名
    # 可选值取决于具体的搜索实现和返回的数据类型
    sort_by: Optional[str] = Field(
        None,
        description="结果排序依据 (score、published_date、title、likes、downloads、last_modified 等)。",
    )
    # 指定排序顺序 (升序或降序)，默认为降序 'desc'
    sort_order: Optional[Literal["asc", "desc"]] = Field(
        "desc", description="排序顺序，asc (升序) 或 desc (降序)。"
    )

    # 模型配置
    model_config = ConfigDict(from_attributes=True)
