"""
论文核心 Pydantic 模型 (Paper Core Pydantic Model)

功能 (Function):
这个模块定义了一个核心的 `Paper` Pydantic 模型。与 `graph.py` 中的 `PaperDetailResponse` 不同，
这个模型更侧重于表示存储在数据库 (`papers` 表) 中的核心字段结构。它可能主要用于：
1.  **数据访问层 (Repository Layer)**: 在 `PostgresRepository` 中，将从数据库查询到的原始行数据映射为此模型实例，方便在代码中处理结构化的论文数据。
2.  **数据插入/更新**: 当需要向数据库插入或更新论文数据时，可以使用此模型来校验和传递数据。
3.  **内部数据传递**: 在服务层或脚本中传递核心的论文信息。

与 `PaperDetailResponse` 的区别:
- `Paper` 模型可能包含更多与数据库表结构直接对应的字段（例如 `arxiv_id_base`, `arxiv_id_versioned`, `primary_category`）。
- `PaperDetailResponse` 更侧重于 API 响应，可能只包含 API 需要暴露给客户端的字段，并且字段名可能经过调整以更符合 API 规范。

交互 (Interaction):
- 依赖 (Depends on):
    - `pydantic`: 用于定义模型、字段、配置等。
    - `typing`: 用于提供类型提示 (List, Optional)。
    - `datetime`: 用于处理日期类型 (`date` as `date_type`)。
- 被导入 (Imported by):
    - `aigraphx.repositories.postgres_repo`: 很可能使用此模型来处理从 `papers` 表查询或写入的数据。
    - `aigraphx.services.*`: 服务层可能使用此模型在内部传递核心论文数据。
    - `scripts/*`: 数据处理脚本（如 `load_postgres.py`）在处理和存储论文数据时可能会使用此模型。

设计原则 (Design Principles):
- **数据库映射 (Database Mapping):** 字段设计倾向于匹配数据库表的列。
- **核心属性 (Core Attributes):** 包含论文最核心、最常用的属性。
- **类型安全 (Type Safety):** 使用 Pydantic 和 typing 确保数据类型正确。
"""

# 导入 datetime 模块中的 date 类型，并重命名为 date_type 以避免与变量名冲突
from datetime import date as date_type

# 导入 typing 模块提供类型提示
from typing import List, Optional

# 导入 Pydantic 核心组件
# HttpUrl 是 Pydantic 提供的特殊类型，用于验证字符串是否为有效的 HTTP 或 HTTPS URL
from pydantic import BaseModel, Field, ConfigDict, HttpUrl


class Paper(BaseModel):
    """
    论文核心模型类。

    这个模型定义了与数据库 `papers` 表中核心列对应的字段。
    它主要在数据持久化和内部数据处理流程中使用。
    """

    # Papers with Code ID，作为主要标识符，必需字段，且最小长度为 1
    pwc_id: str = Field(..., min_length=1)
    # 论文标题 (可选)
    title: Optional[str] = None
    # ArXiv ID 的基础部分 (不包含版本号，例如 '2310.06825') (可选)
    arxiv_id_base: Optional[str] = None
    # 带版本的 ArXiv ID (例如 '2310.06825v1') (可选)
    arxiv_id_versioned: Optional[str] = None
    # 论文摘要 (可选)
    summary: Optional[str] = None
    # PDF 下载链接 (可选)，使用 HttpUrl 类型进行验证
    pdf_url: Optional[HttpUrl] = None
    # 发表日期 (可选)，使用 date 类型
    published_date: Optional[date_type] = None
    # 作者列表，默认为空列表，确保即使没有作者信息，该字段也始终是列表类型
    authors: List[str] = Field(default_factory=list)
    # 论文所属领域 (例如 'Computer Science') (可选)
    area: Optional[str] = None
    # ArXiv 上的主要分类 (例如 'cs.CL') (可选)
    primary_category: Optional[str] = None
    # ArXiv 上的所有分类列表，默认为空列表
    categories: List[str] = Field(default_factory=list)
    # Papers with Code 网站上显示的标题 (可能与原始标题略有不同) (可选)
    pwc_title: Optional[str] = None
    # Papers with Code 网站上该论文的链接 (可选)，使用 HttpUrl 类型验证
    pwc_url: Optional[HttpUrl] = None
    # 论文的 DOI (Digital Object Identifier) (可选)
    doi: Optional[str] = None

    # 模型配置
    model_config = ConfigDict(
        # 允许从对象属性创建模型实例，方便从数据库查询结果或其他对象转换
        from_attributes=True
    )
