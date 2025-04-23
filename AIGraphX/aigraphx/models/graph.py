"""
图数据相关 Pydantic 模型 (Graph Data Related Pydantic Models)

功能 (Function):
这个模块定义了用于表示和序列化图数据结构以及图中节点详细信息的 Pydantic 模型。
这些模型主要用于：
1.  **API 响应序列化**: 定义 API 端点返回图数据（例如，某个节点的邻居）或节点详细信息（论文详情、模型详情）时的响应体结构。FastAPI 会使用这些模型来验证和序列化返回的数据为 JSON。
2.  **数据校验**: (虽然在此文件中主要用于响应) Pydantic 模型也提供了数据校验功能，确保数据符合预期的格式和类型。
3.  **代码清晰度和类型提示**: 明确定义了数据结构，提高了代码的可读性和可维护性，并为类型检查工具（如 Mypy）提供了支持。

交互 (Interaction):
- 依赖 (Depends on):
    - `pydantic`: 核心库，用于定义模型、字段、验证器等。
    - `typing`: 用于提供类型提示 (List, Optional, Dict, Any, Union)。
    - `datetime`: 用于处理日期和时间类型。
- 被导入 (Imported by):
    - `aigraphx.api.v1.endpoints.*`: API 端点函数会使用这些模型作为 `response_model` 来定义和序列化响应数据。例如，返回图邻居的端点可能使用 `GraphData`，返回论文详情的端点可能使用 `PaperDetailResponse`。
    - `aigraphx.services.*`: 服务层的方法可能会构建并返回这些模型的实例，或者接收它们作为参数（尽管在此模块中主要用于响应）。
    - `aigraphx.repositories.*`: 仓库层在从数据库获取数据后，可能会将原始数据映射到这些模型实例上（尤其是在服务层或端点层进行转换时）。

设计原则 (Design Principles):
- **API 边界清晰 (Clear API Boundary):** 这些模型主要用于定义 API 的输出契约。
- **结构化数据 (Structured Data):** 使用 Pydantic 强制执行清晰的数据结构。
- **自包含性 (Self-Contained):** 模型定义包含了字段、类型、描述和（必要的）验证逻辑。
- **可扩展性 (Extensibility):** 容易添加新的字段或模型来支持未来的需求。
"""

# 导入 Pydantic 核心组件
from pydantic import (
    BaseModel,  # 所有 Pydantic 模型的基类
    Field,  # 用于为字段添加额外信息 (默认值, 描述, 别名等)
    ConfigDict,  # 用于配置模型行为 (例如 orm_mode/from_attributes)
    field_validator,  # 装饰器，用于定义自定义字段验证逻辑
)

# 导入 typing 模块提供类型提示
from typing import List, Optional, Dict, Any, Union

# 导入 datetime 模块处理日期和时间
from datetime import datetime, date


# --- 图数据结构基础模型 (Base Models for Graph Data Structure) ---


class Node(BaseModel):
    """
    表示图中的一个节点 (Node)。

    用于 GraphData 模型中，代表图谱中的一个实体，如论文、模型、概念等。
    """

    # 节点的唯一标识符，必需字段 (...)
    # 例如，可能是论文的 pwc_id，模型的 huggingface_id，或其他内部 ID
    id: str = Field(..., description="节点的唯一标识符 (例如, pwc_id 或 model_id)")

    # 节点的可选显示标签，用于可视化或展示
    # 例如，论文标题或模型名称
    label: Optional[str] = Field(
        None, description="节点的显示标签 (例如, 论文标题或模型名称)"
    )

    # 节点的类型，必需字段，用于区分不同种类的节点
    type: str = Field(
        ..., description="节点的类型 (例如, 'Paper', 'HFModel', 'Concept')"
    )

    # 一个字典，用于存储从 Neo4j 获取的节点的其他属性
    # 提供了灵活性，可以包含任意键值对
    properties: Dict[str, Any] = Field({}, description="节点的附加属性。")

    # Pydantic 模型配置 (可选，如果需要特定行为)
    # model_config = ConfigDict(...)


class Relationship(BaseModel):
    """
    表示图中连接两个节点的一条关系 (Edge / Relationship)。

    用于 GraphData 模型中，代表节点之间的连接，如引用、相关、使用等。
    """

    # 关系起始节点的 ID，必需字段
    source: str = Field(..., description="源节点的 ID。")

    # 关系目标节点的 ID，必需字段
    target: str = Field(..., description="目标节点的 ID。")

    # 关系的类型，必需字段，描述了连接的性质
    type: str = Field(
        ..., description="关系的类型 (例如, 'CITES', 'RELATED_TO', 'USES_MODEL')。"
    )

    # 一个字典，用于存储关系可能具有的属性 (如果 Neo4j 中的关系有属性的话)
    properties: Dict[str, Any] = Field({}, description="关系的属性。")

    # Pydantic 模型配置 (可选)
    # model_config = ConfigDict(...)


class GraphData(BaseModel):
    """
    用于图可视化或分析的图数据结构。

    通常由返回图邻居信息的 API 端点使用，包含节点列表和关系列表。
    """

    # 图中包含的节点列表，必需字段
    nodes: List[Node] = Field(..., description="图邻域中的节点列表。")

    # 连接这些节点的关系列表，必需字段
    relationships: List[Relationship] = Field(..., description="连接节点的关系列表。")

    # Pydantic 模型配置 (可选)
    # model_config = ConfigDict(...)


# --- 图相关实体的详细信息响应模型 (Detailed Information Response Models) ---


class PaperDetailResponse(BaseModel):
    """
    表示 API 返回的单个论文的详细信息。

    用于 `/papers/{pwc_id}` 或类似端点的响应模型。
    字段大多是可选的，因为数据库中的信息可能不完整。
    """

    # Papers with Code ID，通常作为主键
    pwc_id: str
    # 论文标题
    title: Optional[str] = None
    # 论文摘要
    abstract: Optional[str] = None
    # ArXiv ID (如果可用)
    arxiv_id: Optional[str] = None
    # 论文摘要页面的 URL (通常是 ArXiv 或 PWC 页面)
    url_abs: Optional[str] = None
    # 论文 PDF 文件的 URL
    url_pdf: Optional[str] = None
    # 发表日期 (使用 date 类型)
    published_date: Optional[date] = None
    # 作者列表
    authors: Optional[List[str]] = None
    # 相关任务列表
    tasks: Optional[List[str]] = None
    # 使用的方法列表
    methods: Optional[List[str]] = None
    # 使用的数据集列表
    datasets: Optional[List[str]] = None
    # 使用的框架列表 (例如 PyTorch, TensorFlow)
    frameworks: Optional[List[str]] = None
    # GitHub 仓库星标数 (如果关联了代码库并获取了星标)
    number_of_stars: Optional[int] = None
    # 论文所属领域
    area: Optional[str] = None
    # (可选) 将来可能在此模型中包含论文的邻居图信息
    # neighborhood: Optional[GraphData] = None

    # 模型配置
    model_config = ConfigDict(
        # from_attributes=True (以前的 orm_mode=True):
        # 允许 Pydantic 模型从对象的属性（而不仅仅是字典）创建实例。
        # 这在使用 ORM (如 SQLAlchemy) 或其他返回对象实例的库时很有用。
        # 即使不直接用 ORM，设置为 True 通常也是安全的，并增加了灵活性。
        from_attributes=True
    )


class HFModelDetail(BaseModel):
    """
    表示 API 返回的单个 Hugging Face 模型的详细信息。

    用于 `/models/{model_id}` 或类似端点的响应模型。
    """

    # Hugging Face 模型的唯一 ID (例如 "bert-base-uncased")，必需字段
    model_id: str = Field(..., description="唯一的 Hugging Face 模型 ID。")
    # 模型作者或所属组织
    author: Optional[str] = Field(None, description="作者或组织。")
    # 与模型版本关联的 Git 提交 SHA
    sha: Optional[str] = Field(None, description="与模型版本关联的 Git 提交 SHA。")
    # 最后修改时间戳 (使用 datetime 类型)
    last_modified: Optional[datetime] = Field(None, description="最后修改的时间戳。")
    # 模型关联的标签列表
    tags: Optional[List[str]] = Field(None, description="与模型关联的标签列表。")
    # 主要的任务流水线标签 (例如 "text-classification", "question-answering")
    pipeline_tag: Optional[str] = Field(None, description="主要的任务流水线标签。")
    # 下载次数
    downloads: Optional[int] = Field(None, description="下载次数。")
    # 点赞次数
    likes: Optional[int] = Field(None, description="点赞次数。")
    # 模型关联的库名称 (例如 "transformers", "diffusers")
    library_name: Optional[str] = Field(
        None, description="模型关联的库 (例如, transformers)。"
    )
    # 数据库中记录的创建时间
    created_at: Optional[datetime] = Field(
        None, description="数据库中记录创建的时间戳。"
    )
    # 数据库中记录的最后更新时间
    updated_at: Optional[datetime] = Field(
        None, description="数据库中记录最后更新的时间戳。"
    )

    # 自定义字段验证器，用于解析 'last_modified' 字段
    # mode='before': 在 Pydantic 标准验证和类型转换 *之前* 执行此验证器
    @field_validator("last_modified", mode="before")
    @classmethod  # 验证器必须是类方法
    def parse_last_modified(cls, v: Any) -> Optional[datetime]:
        """
        尝试将输入的 'last_modified' 值解析为 datetime 对象。

        处理两种情况：
        1. 输入是符合 ISO 8601 格式的字符串 (特别是 Hugging Face API 返回的带 'Z' 的格式)。
        2. 输入已经是 datetime 对象。
        3. 输入是 None。

        Args:
            v (Any): 输入的值。

        Returns:
            Optional[datetime]: 解析后的 datetime 对象，如果输入为 None 或解析失败则返回 None 或抛出 ValueError。

        Raises:
            ValueError: 如果输入字符串格式无效或输入类型不被支持。Pydantic 会捕获此错误并报告为验证失败。
        """
        # 如果输入是字符串
        if isinstance(v, str):
            try:
                # Hugging Face API 返回的时间字符串可能以 'Z' 结尾，代表 UTC
                # Python 的 fromisoformat 需要 '+00:00' 来表示 UTC 时区
                # 因此，将 'Z' 替换为 '+00:00'
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                # 如果字符串格式无效，抛出 ValueError
                # Pydantic 会捕获这个 ValueError 并生成验证错误信息
                raise ValueError(f"无效的 'last_modified' 日期时间格式: '{v}'") from e
        # 如果输入已经是 datetime 对象，直接返回
        elif isinstance(v, datetime):
            return v
        # 如果输入是 None，也直接返回
        elif v is None:
            return None
        # 对于其他不支持的类型，抛出 ValueError
        raise ValueError("'last_modified' 必须是有效的 ISO 8601 字符串或 datetime 对象")

    # 模型配置
    model_config = ConfigDict(
        from_attributes=True,  # 允许从对象属性创建模型实例
        # arbitrary_types_allowed=True:
        # 允许模型字段使用 Pydantic 本身无法直接验证的任意类型（例如数据库连接对象）。
        # 在这个模型中，虽然没有直接使用任意类型，但保留它通常是无害的，
        # 除非需要非常严格的类型控制。对于 datetime 类型，Pydantic 是可以处理的。
        arbitrary_types_allowed=True,
    )


# --- 未来可能添加的其他模型 ---
# 例如，如果需要返回任务或数据集的详细信息，可以在这里定义相应的模型
# class TaskDetailResponse(BaseModel): ...
# class DatasetDetailResponse(BaseModel): ...
