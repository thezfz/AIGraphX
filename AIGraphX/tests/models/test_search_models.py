# -*- coding: utf-8 -*-
"""
文件目的：测试搜索结果相关的 Pydantic 模型。

本测试文件 (`test_search_models.py`) 专注于验证定义在 `aigraphx.models.search` 模块中的
用于表示搜索结果（包括单个结果项和分页结果）的 Pydantic 模型。

这些模型确保从搜索服务返回的数据结构符合预期的格式和类型，并且包含了必要的信息。

主要交互：
- 导入 `pytest`：用于测试框架和运行测试。
- 导入 `pydantic.ValidationError`：用于捕获和验证模型创建时的验证错误。
- 导入 `typing`：用于类型提示。
- 导入 `datetime`：用于处理日期时间字段。
- 导入被测试的模型：从 `aigraphx.models.search` 导入 `SearchResultItem`, `HFSearchResultItem`, `SearchType`, `PaginatedPaperSearchResult` 等。
- 编写测试函数 (`test_*`)：
    - `SearchResultItem` 测试 (`test_searchresultitem_*`)：
        - 测试最小化创建（只提供必需字段）。
        - 测试完整创建（提供所有字段）。
        - 测试缺少必需字段（如 `pwc_id`）时是否抛出 `ValidationError`。
    - `HFSearchResultItem` 测试 (`test_hfsearchresultitem_*`)：
        - 测试最小化创建。
        - 测试完整创建，包括对 `last_modified` 字段的日期时间字符串解析验证。
        - 测试缺少必需字段（如 `model_id`, `score`）时是否抛出 `ValidationError`。
    - `PaginatedPaperSearchResult` 测试 (`test_paginated_paper_search_result_*`)：
        - 测试成功创建包含结果项的分页对象。
        - 测试创建空结果的分页对象。
        - 测试缺少必需字段（如 `items`, `total`, `skip`, `limit`）时是否抛出 `ValidationError`。

这些测试对于确保 API 返回给前端或其他客户端的搜索结果数据结构是正确、一致且有效的至关重要。
"""

import pytest # 导入 pytest 测试框架
from pydantic import ValidationError # 从 Pydantic 导入验证错误异常类
from typing import List, Optional, cast, Any, Dict, Union # 导入类型提示工具
from datetime import date, datetime, timezone # 导入日期和时间相关类型

# 导入需要测试的模型
from aigraphx.models.search import (
    SearchResultItem, # 单个论文/通用搜索结果项模型
    HFSearchResultItem, # 单个 Hugging Face 模型搜索结果项模型
    SearchType, # 搜索类型枚举 (虽然这里没直接测枚举，但模型可能用到)
    PaginatedPaperSearchResult, # 论文搜索结果的分页模型
)


# --- 测试 SearchResultItem 模型 ---

def test_searchresultitem_creation_minimal() -> None:
    """测试 SearchResultItem 模型的最小化创建。"""
    # 只提供必需的 pwc_id 和可选但常用的 score
    item = SearchResultItem(pwc_id="paper1", score=0.85)
    # 断言必需字段被正确设置
    assert item.pwc_id == "paper1"
    # 断言提供的可选字段被正确设置
    assert item.score == 0.85
    # 断言其他未提供的可选字段默认为 None
    assert item.title is None
    assert item.summary is None
    assert item.published_date is None
    assert item.authors is None


def test_searchresultitem_creation_full() -> None:
    """测试 SearchResultItem 模型的完整创建。"""
    # 准备一个日期对象
    published_date = date(2024, 1, 15)
    # 提供所有字段的值进行创建
    item = SearchResultItem(
        pwc_id="paper2",
        title="Result Title",
        summary="Result abstract.",
        score=0.9,
        published_date=published_date,
        authors=["Author X"],
    )
    # 断言所有字段都被正确设置
    assert item.pwc_id == "paper2"
    assert item.title == "Result Title"
    assert item.summary == "Result abstract."
    assert item.score == 0.9
    assert item.published_date == published_date
    assert item.authors == ["Author X"]


def test_searchresultitem_missing_required() -> None:
    """测试当缺少必需字段时，SearchResultItem 创建是否失败。"""
    # 目前模型定义中，只有 pwc_id 是严格必需的 (score 有默认值 None)
    with pytest.raises(ValidationError) as excinfo: # 捕获预期的验证错误
        # 尝试创建实例，但不提供 pwc_id 参数
        # 使用 # type: ignore 告诉 mypy 忽略此处的参数缺失错误，因为这是测试目的
        SearchResultItem(title="T", score=0.5)  # type: ignore[call-arg]
    # 检查 ValidationError 的错误详情
    errors = excinfo.value.errors()
    # 断言错误列表中包含一个针对 'pwc_id' 字段的 'missing' 类型错误
    assert any(err["type"] == "missing" and err["loc"] == ("pwc_id",) for err in errors)


# --- 测试 HFSearchResultItem 模型 ---

def test_hfsearchresultitem_creation_minimal() -> None:
    """测试 HFSearchResultItem 模型的最小化创建。"""
    # 提供必需的 model_id 和 score
    # 为了代码清晰和满足 mypy (如果严格检查)，显式地为所有可选字段传递 None
    item = HFSearchResultItem(
        model_id="org/model1",
        score=0.7,
        author=None,
        pipeline_tag=None,
        library_name=None,
        tags=None,
        likes=None,
        downloads=None,
        last_modified=None,
    )
    # 断言必需字段
    assert item.model_id == "org/model1"
    assert item.score == 0.7
    # 断言可选字段为 None
    assert item.author is None
    assert item.tags is None
    assert item.pipeline_tag is None
    assert item.library_name is None
    assert item.likes is None
    assert item.downloads is None
    assert item.last_modified is None


def test_hfsearchresultitem_creation_full() -> None:
    """测试 HFSearchResultItem 模型的完整创建，特别是 last_modified 的解析。"""
    # 准备一个符合模型内部验证器格式的日期时间字符串
    last_modified_str = "2024-03-10T10:00:00Z"
    # 在传递给模型之前，手动调用模型定义的解析方法（如果想单独测试解析逻辑）
    # 或者直接传递字符串，让模型在初始化时解析
    parsed_dt = HFSearchResultItem.parse_last_modified(last_modified_str)
    assert parsed_dt is not None  # 确保解析成功

    # 使用所有字段创建实例
    item = HFSearchResultItem(
        model_id="org/model2",
        author="Org",
        pipeline_tag="text-generation",
        library_name="transformers",
        tags=["llm", "chat"],
        likes=100,
        downloads=5000,
        last_modified=parsed_dt,  # 传递解析后的 datetime 对象
        score=0.95,
    )
    # 断言所有字段都被正确设置
    assert item.model_id == "org/model2"
    assert item.author == "Org"
    assert item.pipeline_tag == "text-generation"
    assert item.tags == ["llm", "chat"]
    assert item.likes == 100
    assert item.score == 0.95
    # 验证 last_modified 的类型和值
    assert isinstance(item.last_modified, datetime)
    assert item.last_modified.year == 2024
    # 验证时区信息是否正确（应为 UTC）
    assert item.last_modified.tzinfo == timezone.utc


def test_hfsearchresultitem_missing_required() -> None:
    """测试当缺少必需字段时，HFSearchResultItem 创建是否失败。"""
    # --- 测试缺少 model_id ---
    with pytest.raises(ValidationError) as excinfo_model:
        HFSearchResultItem(  # type: ignore[call-arg] # 故意不提供 model_id
            # model_id 参数被故意删除
            score=0.6, # 提供 score
            author=None,
            pipeline_tag=None,
            library_name=None,
            tags=None,
            likes=None,
            downloads=None,
            last_modified=None, # 提供 None
        )
    errors_model = excinfo_model.value.errors()
    # 断言错误是关于 model_id 缺失
    assert any(
        err["type"] == "missing" and err["loc"] == ("model_id",) for err in errors_model
    )

    # --- 测试缺少 score ---
    with pytest.raises(ValidationError) as excinfo_score:
        HFSearchResultItem(  # type: ignore[call-arg] # 故意不提供 score
            model_id="org/model3", # 提供 model_id
            # score 参数被故意删除
            author=None,
            pipeline_tag=None,
            library_name=None,
            tags=None,
            likes=None,
            downloads=None,
            last_modified=None, # 提供 None
        )
    errors_score = excinfo_score.value.errors()
    # 断言错误是关于 score 缺失
    assert any(
        err["type"] == "missing" and err["loc"] == ("score",) for err in errors_score
    )


# --- 测试 PaginatedPaperSearchResult 模型 ---

def test_paginated_paper_search_result_creation_success() -> None:
    """测试 PaginatedPaperSearchResult 模型的成功创建。"""
    # 创建两个搜索结果项
    item1 = SearchResultItem(pwc_id="p1", score=0.9)
    item2 = SearchResultItem(pwc_id="p2", score=0.8)
    # 使用结果项列表和其他分页信息创建分页结果对象
    paginated_result = PaginatedPaperSearchResult(
        items=[item1, item2], total=10, skip=0, limit=2
    )
    # 断言 items 列表长度正确
    assert len(paginated_result.items) == 2
    # 断言 total, skip, limit 值正确
    assert paginated_result.total == 10
    assert paginated_result.skip == 0
    assert paginated_result.limit == 2
    # 断言 items 列表内容正确
    assert paginated_result.items[0] == item1


def test_paginated_paper_search_result_creation_empty() -> None:
    """测试当 items 列表为空时，PaginatedPaperSearchResult 模型的创建。"""
    # 创建一个 items 为空列表的分页结果对象
    paginated_result = PaginatedPaperSearchResult(items=[], total=0, skip=10, limit=5)
    # 断言 items 列表为空
    assert paginated_result.items == []
    # 断言 total, skip, limit 值正确
    assert paginated_result.total == 0
    assert paginated_result.skip == 10
    assert paginated_result.limit == 5


def test_paginated_paper_search_result_missing_required() -> None:
    """测试当缺少必需字段时，PaginatedPaperSearchResult 创建是否失败。"""
    # 准备一个结果项用于测试
    item1 = SearchResultItem(pwc_id="p1", score=0.9)

    # --- 测试缺少 items ---
    with pytest.raises(ValidationError, match="items"): # 预期错误消息包含 "items"
        # 使用 cast(Any, None) 来绕过类型检查，模拟传入 None
        PaginatedPaperSearchResult(items=cast(Any, None), total=1, skip=0, limit=1)

    # --- 测试缺少 total ---
    with pytest.raises(ValidationError, match="total"):
        PaginatedPaperSearchResult(
            items=[item1], total=cast(Any, None), skip=0, limit=1
        )

    # --- 测试缺少 skip ---
    with pytest.raises(ValidationError, match="skip"):
        PaginatedPaperSearchResult(
            items=[item1], total=1, skip=cast(Any, None), limit=1
        )

    # --- 测试缺少 limit ---
    with pytest.raises(ValidationError, match="limit"):
        PaginatedPaperSearchResult(
            items=[item1], total=1, skip=0, limit=cast(Any, None)
        )


# 附注：当前的 SearchResponse 模型（如果存在且被其他地方使用）定义 results 为 List[SearchResultItem]。
# 而 API 端点可能返回 List[Union[SearchResultItem, HFSearchResultItem]]。
# 如果 API 确实旨在返回混合类型的结果，SearchResponse 模型的定义可能需要更新
# （例如，使用 Union 或一个通用的基类）。
# 当前的测试反映的是 *已导入模型* 的定义。