"""
论文模型定义。
"""

from datetime import date as date_type
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, HttpUrl


class Paper(BaseModel):
    """论文模型类，对应数据库 'papers' 表的核心字段 (用于 Repository 层)。"""

    pwc_id: str = Field(..., min_length=1)
    title: Optional[str] = None
    arxiv_id_base: Optional[str] = None
    arxiv_id_versioned: Optional[str] = None
    summary: Optional[str] = None
    pdf_url: Optional[HttpUrl] = None
    published_date: Optional[date_type] = None
    authors: List[str] = Field(default_factory=list)
    area: Optional[str] = None
    primary_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    pwc_title: Optional[str] = None
    pwc_url: Optional[HttpUrl] = None
    doi: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
