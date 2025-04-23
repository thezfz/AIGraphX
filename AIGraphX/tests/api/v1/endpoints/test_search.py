# -*- coding: utf-8 -*-
"""
文件目的：测试搜索 API 端点 (tests/api/v1/endpoints/test_search.py)

本文件包含针对 `/api/v1/search/papers/` 和 `/api/v1/search/models/` 端点的进一步集成测试。
与 `test_search_api.py` 类似，主要目标是验证这些搜索 API 端点：
1.  能否正确处理针对 papers 和 models 的搜索请求（特别是默认的语义搜索）。
2.  能否正确调用依赖的 `SearchService` 中的 `perform_semantic_search` 方法。
3.  能否正确处理 `SearchService` 返回的分页结果对象 (`PaginatedHFModelSearchResult`, `PaginatedPaperSearchResult`)。
4.  能否在服务层发生错误时返回 500 内部服务器错误。
5.  能否根据请求参数（如 `q`, `skip`, `limit`）正确计算并传递 `page` 和 `page_size` 给服务层。

核心测试策略：
- 使用 `httpx.AsyncClient` 模拟异步 HTTP 客户端。
- 使用 `pytest` 测试框架及其 fixtures (`client`, `test_app`)。
- **在每个测试函数内部**创建 `unittest.mock.AsyncMock` 对象来模拟 `SearchService`。
- 通过 FastAPI 的 `test_app.dependency_overrides` 机制，使用模拟服务替换真实的 `SearchService` 依赖。
- 对成功场景 (200 OK) 和服务层失败场景 (500 Internal Server Error) 进行测试。
- 验证响应数据的结构和内容是否符合预期。
- 使用 `mock_*.assert_awaited_once_with(...)` 验证模拟服务的方法调用参数是否正确。

与其他文件的交互：
- 导入测试基础库 `pytest`, `httpx`, `unittest.mock`, `fastapi`, `datetime`, `logging`。
- 导入 Pydantic 模型 `aigraphx.models.search.*` 用于定义模拟数据和预期响应结构，特别是 `HFSearchResultItem` 和 `SearchResultItem` 以及分页模型。
- 导入原始依赖提供函数所在的模块 `aigraphx.api.v1.dependencies` (别名 `deps`)，以便引用 `get_search_service` 作为依赖覆盖的键。
- 依赖 `tests/conftest.py` 提供 `client` (AsyncClient 实例) 和 `test_app` (FastAPI 实例) fixtures。

注意：此文件中的测试似乎更侧重于验证 API 端点和服务层之间关于分页参数 (`skip`, `limit` vs `page`, `page_size`) 的转换逻辑，以及默认搜索类型（语义搜索）的处理。
"""

import pytest
from fastapi import status, FastAPI  # 导入 status 用于 HTTP 状态码常量, FastAPI 类用于类型提示和依赖覆盖
from httpx import AsyncClient  # 导入异步 HTTP 客户端
from unittest.mock import AsyncMock, patch  # 导入异步模拟 AsyncMock 和 patch (如果需要替换模块/类)
from datetime import datetime, timezone  # 导入日期时间处理
import logging  # 导入日志库 (虽然在此文件中未显式使用 logger)

# pytest 标记，指示此模块中的所有 `async def` 测试函数都应使用 pytest-asyncio
# 提供的默认函数作用域事件循环来运行。
pytestmark = pytest.mark.asyncio

# 导入必要的 Pydantic 模型
from aigraphx.models.search import (
    PaginatedHFModelSearchResult,  # 分页的 HF 模型搜索结果模型
    PaginatedPaperSearchResult,  # 分页的论文搜索结果模型
    SearchResultItem,  # 单个论文结果项模型
    HFSearchResultItem,  # 单个 HF 模型结果项模型
)

# 导入原始依赖提供函数所在的模块，用作依赖覆盖的键
from aigraphx.api.v1 import dependencies as deps


# --- 测试用例 ---

@pytest.mark.asyncio
async def test_search_hf_models_success(
    client: AsyncClient,  # 注入 httpx 异步客户端 fixture
    test_app: FastAPI,  # 注入 FastAPI 测试应用 fixture
) -> None:
    """
    测试场景：成功搜索 Hugging Face 模型 (GET /api/v1/search/models/)。
    预期：API 返回 200 OK 和正确的分页模型数据。
    策略：在测试函数内部创建并配置 SearchService 的模拟对象，覆盖依赖，发送请求，然后验证响应和模拟调用。
    """
    # --- 1. 在测试函数内部创建模拟服务 ---
    mock_search_service = AsyncMock()  # 创建 SearchService 的异步模拟

    # --- 2. 配置模拟数据和模拟方法的返回值 ---
    # 为了确保 last_modified 字段是正确的 datetime 类型，需要显式解析字符串
    dt1_str = "2023-01-01T12:00:00Z"
    dt2_str = "2023-01-02T12:00:00Z"
    # 使用 HFSearchResultItem 模型自带的解析方法
    parsed_dt1 = HFSearchResultItem.parse_last_modified(dt1_str)
    parsed_dt2 = HFSearchResultItem.parse_last_modified(dt2_str)
    assert parsed_dt1 is not None  # 确保解析成功
    assert parsed_dt2 is not None

    # 创建模拟的 HF 模型结果列表
    mock_results = [
        HFSearchResultItem(
            model_id="org/model1",
            score=0.9,
            last_modified=parsed_dt1,  # 使用解析后的 datetime 对象
            author="Organization1",
            pipeline_tag="text-generation",
            library_name="transformers",
            tags=["nlp"],
            likes=100,
            downloads=1000,
        ),
        HFSearchResultItem(
            model_id="org/model2",
            score=0.8,
            last_modified=parsed_dt2,
            author="Organization2",
            pipeline_tag="text-classification",
            library_name="transformers",
            tags=["nlp"],
            likes=200,
            downloads=2000,
        ),
    ]
    # 创建模拟的分页结果对象
    mock_paginated_result = PaginatedHFModelSearchResult(
        items=mock_results, total=len(mock_results), skip=0, limit=10
    )
    # 配置模拟服务的 perform_semantic_search 方法返回此分页结果
    # 注意：/models/ 端点默认进行语义搜索，因此模拟 perform_semantic_search
    mock_search_service.perform_semantic_search.return_value = mock_paginated_result

    # --- 3. 应用依赖覆盖 ---
    # 保存原始覆盖状态
    original_overrides = test_app.dependency_overrides.copy()
    # 将 get_search_service 依赖替换为返回模拟对象的 lambda 函数
    test_app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service

    try:
        # --- 4. 执行 API 调用 ---
        # 向 /api/v1/search/models/ 发送 GET 请求，带上查询参数 q, skip, limit
        # 未指定 search_type，因此 API 端点应默认为 semantic
        response = await client.get(
            "/api/v1/search/models/", params={"q": "test", "skip": 0, "limit": 10}
        )

        # --- 5. 断言响应 ---
        assert response.status_code == status.HTTP_200_OK  # 验证状态码
        response_data = response.json()  # 解析响应 JSON
        # 验证响应数据的结构和内容是否符合 PaginatedHFModelSearchResult
        assert isinstance(response_data, dict)
        assert response_data["total"] == len(mock_results)
        assert response_data["skip"] == 0
        assert response_data["limit"] == 10
        assert len(response_data["items"]) == len(mock_results)
        assert response_data["items"][0]["model_id"] == mock_results[0].model_id

        # --- 6. 验证模拟调用 ---
        # 根据 API 请求中的 skip 和 limit 计算服务层期望接收的 page 和 page_size
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1  # 页码通常从 1 开始
        # 验证模拟服务的 perform_semantic_search 方法是否被以预期的参数调用
        mock_search_service.perform_semantic_search.assert_awaited_once_with(
            query="test",
            page=expected_page,
            page_size=expected_limit,
            target="models",  # 目标是模型
            sort_by="score",  # 模型语义搜索默认按分数排序
            sort_order="desc",  # 默认降序
        )
    finally:
        # --- 7. 恢复依赖覆盖 ---
        # 清理依赖覆盖，确保不影响其他测试
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_semantic_success(
    client: AsyncClient,
    test_app: FastAPI,
) -> None:
    """
    测试场景：成功通过 /papers/ 端点进行语义搜索 (GET /api/v1/search/papers/?search_type=semantic)。
    预期：API 返回 200 OK 和正确的分页论文数据。
    策略：模拟 SearchService 的 `perform_semantic_search` 方法返回论文分页结果。
    注意：这个测试用例的名称与上一个有重叠，且似乎是针对 /papers/ 端点的语义搜索。
    """
    # --- 1. 创建模拟服务 ---
    mock_search_service = AsyncMock()

    # --- 2. 配置模拟数据和模拟方法 ---
    # 创建模拟的论文结果列表
    mock_papers = [
        SearchResultItem(pwc_id="pwc-1", paper_id=1, title="Test Paper 1", score=0.95),
        SearchResultItem(pwc_id="pwc-2", paper_id=2, title="Test Paper 2", score=0.90),
    ]
    # 创建模拟的分页结果对象 (PaginatedPaperSearchResult)
    # 注意：即使服务方法是 perform_semantic_search，API 端点的响应模型决定了这里可能
    # 需要模拟 PaginatedPaperSearchResult 或 PaginatedSemanticSearchResult (需要核对API实现)
    # 假设这里模拟返回 PaginatedPaperSearchResult
    mock_paginated_paper_result = PaginatedPaperSearchResult(
        items=mock_papers,
        total=len(mock_papers),
        skip=0,  # 默认 skip
        limit=10,  # 默认 limit
    )
    # 配置模拟服务的 perform_semantic_search 方法返回这个结果
    mock_search_service.perform_semantic_search.return_value = (
        mock_paginated_paper_result
    )

    # --- 3. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service

    try:
        # --- 4. 执行 API 调用 ---
        # 向 /api/v1/search/papers/ 发送 GET 请求，明确指定 search_type=semantic
        response = await client.get(
            "/api/v1/search/papers/", params={"q": "test", "search_type": "semantic"}
        )

        # --- 5. 断言响应 ---
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()

        # 验证响应数据是否与模拟的分页结果匹配
        # 使用 model_dump(mode="json") 来获取可比较的 JSON 兼容字典
        expected_response = mock_paginated_paper_result.model_dump(mode="json")
        assert response_data == expected_response

        # --- 6. 验证模拟调用 ---
        # API 端点默认 skip=0, limit=10
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1
        # 验证 perform_semantic_search 的调用参数
        mock_search_service.perform_semantic_search.assert_awaited_once_with(
            query="test",
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # 目标是论文
            # 注意：断言时不应包含值为 None 的默认过滤器参数，因为 FastAPI 可能不会传递它们
            # published_after=None,
            # published_before=None,
            # filter_area=None,
            sort_by="score",  # 论文语义搜索默认按分数排序
            sort_order="desc",  # 默认降序
        )
    finally:
        # --- 7. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# --- 针对 /models/ 端点的更多测试 ---


@pytest.mark.asyncio
async def test_search_models_semantic_success(
    client: AsyncClient,
    test_app: FastAPI,
) -> None:
    """
    测试场景：通过 /models/ 端点成功进行语义搜索 (不显式指定 search_type)。
    预期：API 返回 200 OK 和模型的分页结果。
    策略：模拟 SearchService 的 `perform_semantic_search` 方法。
    注意：这个测试用例名称与之前的模型搜索成功案例重名，但可能侧重于默认 search_type 的情况。
    """
    # --- 1. 创建模拟服务 ---
    mock_search_service = AsyncMock()

    # --- 2. 配置模拟数据和模拟方法 ---
    # 解析 datetime 字符串
    dt_str = "2023-02-01T12:00:00Z"
    parsed_dt = HFSearchResultItem.parse_last_modified(dt_str)
    assert parsed_dt is not None

    # 创建模拟模型结果
    mock_results = [
        HFSearchResultItem(
            model_id="org/model-sem1",
            score=0.7,
            last_modified=parsed_dt,
            author="Organization",
            pipeline_tag="text-generation",
            library_name="transformers",
            tags=["nlp"],
            likes=150,
            downloads=1500,
        ),
    ]
    # 创建模拟分页结果
    mock_paginated_result = PaginatedHFModelSearchResult(
        items=mock_results, total=len(mock_results), skip=0, limit=5
    )
    mock_search_service.perform_semantic_search.return_value = mock_paginated_result

    # --- 3. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service
    try:
        # --- 4. 执行 API 调用 ---
        # 请求 /models/ 端点，只提供 q, skip, limit 参数，依赖 API 默认使用语义搜索
        response = await client.get(
            "/api/v1/search/models/",
            params={"q": "semantic model", "skip": 0, "limit": 5},
        )

        # --- 5. 断言响应 ---
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["total"] == 1
        assert len(response_data["items"]) == 1
        assert response_data["items"][0]["model_id"] == "org/model-sem1"

        # --- 6. 验证模拟调用 ---
        expected_skip = 0
        expected_limit = 5
        expected_page = expected_skip // expected_limit + 1
        # 验证 perform_semantic_search 被以正确的参数调用
        mock_search_service.perform_semantic_search.assert_awaited_once_with(
            query="semantic model",
            page=expected_page,
            page_size=expected_limit,
            target="models",
            sort_by="score",
            sort_order="desc",
        )
    finally:
        # --- 7. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_models_semantic_service_fails(
    client: AsyncClient,
    test_app: FastAPI,
) -> None:
    """
    测试场景：当 SearchService 在处理模型语义搜索时失败。
    预期：API 端点返回 500 Internal Server Error。
    策略：模拟 SearchService 的 `perform_semantic_search` 方法抛出异常。
    """
    # --- 1. 创建模拟服务 ---
    mock_search_service = AsyncMock()
    # 配置模拟方法抛出异常
    mock_search_service.perform_semantic_search.side_effect = Exception("Service Error")

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service
    try:
        # --- 3. 执行 API 调用 ---
        # 发送一个模型搜索请求 (默认语义)
        response = await client.get(
            "/api/v1/search/models/", params={"q": "semantic model fail"}
        )
        # --- 4. 断言响应 ---
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # 验证响应体包含 "detail" 字段和指示错误的文本
        assert "detail" in response.json()
        # 检查错误消息是否指明了与模型搜索相关
        assert "model search" in response.json()["detail"] # 或者更通用的 "Internal server error"
    finally:
        # --- 5. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# --- 在模块级别准备一些可复用的模拟数据 ---
# 提前解析日期时间字符串，避免在测试函数内部重复解析
dt_now_str1 = datetime.now(timezone.utc).isoformat()
dt_now_str2 = datetime.now(timezone.utc).isoformat()
parsed_dt1 = HFSearchResultItem.parse_last_modified(dt_now_str1)
parsed_dt2 = HFSearchResultItem.parse_last_modified(dt_now_str2)
assert parsed_dt1 is not None
assert parsed_dt2 is not None

# 创建一个可复用的模型项目列表
MOCK_MODEL_ITEMS = [
    HFSearchResultItem(
        model_id="org/model-a",
        score=0.9,
        author="OrgA",
        pipeline_tag="text-summarization",
        library_name="transformers",
        tags=["summarization"],
        likes=50,
        downloads=500,
        last_modified=parsed_dt1,  # 使用预先解析的 datetime 对象
    ),
    HFSearchResultItem(
        model_id="user/model-b",
        score=0.8,
        author="UserB",
        pipeline_tag="image-classification",
        library_name="timm",
        tags=["cv", "resnet"],
        likes=25,
        downloads=200,
        last_modified=parsed_dt2,  # 使用预先解析的 datetime 对象
    ),
]


@pytest.mark.asyncio
async def test_search_models_keyword_success(
    client: AsyncClient,
    test_app: FastAPI,
) -> None:
    """
    测试场景：成功通过 /models/ 端点进行关键字搜索 (GET /api/v1/search/models/?search_type=keyword)。
    预期：API 返回 200 OK 和正确的分页模型数据。
    策略：模拟 SearchService 的 `perform_keyword_search` 方法返回模型分页结果。
    注意：此测试用例的实现不完整，缺少部分模拟设置和测试逻辑。
    """
    # --- 1. 创建模拟服务 ---
    mock_search_service = AsyncMock()

    # --- 2. 配置模拟数据和模拟方法 ---
    # 解析日期时间字符串
    dt_str = "2023-03-01T12:00:00Z"
    parsed_dt = HFSearchResultItem.parse_last_modified(dt_str)
    assert parsed_dt is not None

    # 创建模拟的模型结果列表 (仅包含一个示例)
    mock_results = [
        HFSearchResultItem(
            model_id="org/model-kw1",
            # 关键字搜索可能不直接返回语义分数，但模型定义需要 score 字段。
            # 服务层需要处理如何填充或省略 score。这里假设服务层会提供一个默认值或 None。
            # 如果服务层返回的字典中没有 score， Pydantic 模型会使用默认值或报错（如果无默认值且非 Optional）。
            # 为了模拟方便，这里提供一个值。
            score=0.6,
            last_modified=parsed_dt,
            author="OrgKeyword",
            pipeline_tag="fill-mask",
            library_name="pytorch",
            tags=["bert"],
            likes=50,
            downloads=500,
        )
    ]
    # 创建模拟分页结果
    mock_paginated_result = PaginatedHFModelSearchResult(
        items=mock_results, total=len(mock_results), skip=0, limit=10
    )
    # 配置模拟服务的 perform_keyword_search 方法
    mock_search_service.perform_keyword_search.return_value = mock_paginated_result

    # --- 3. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[deps.get_search_service] = lambda: mock_search_service

    try:
        # --- 4. 执行 API 调用 ---
        # 发送模型关键字搜索请求
        response = await client.get(
            "/api/v1/search/models/",
            params={"q": "keyword model", "search_type": "keyword", "skip": 0, "limit": 10}
        )

        # --- 5. 断言响应 ---
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        # 验证响应结构和内容
        expected_response = mock_paginated_result.model_dump(mode="json")
        assert response_data == expected_response
        assert response_data["items"][0]["model_id"] == "org/model-kw1"

        # --- 6. 验证模拟调用 ---
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1
        # 验证 perform_keyword_search 调用参数
        mock_search_service.perform_keyword_search.assert_awaited_once_with(
            query="keyword model",
            page=expected_page,
            page_size=expected_limit,
            target="models",
            sort_by="likes", # 假设模型关键字搜索默认按 likes 排序
            sort_order="desc", # 默认降序
            # 省略论文特有的过滤器参数断言
        )

    finally:
        # --- 7. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides

# --- 可以继续添加更多测试用例 ---
# 例如：
# - 测试带有模型特定过滤器（如 pipeline_tag, library_name, tags）的关键字搜索
# - 测试模型搜索结果为空的情况
# - 测试带有不同排序参数（sort_by, sort_order）的模型搜索
# - 如果 SearchService 增加了针对模型的 hybrid search，也需要添加相应测试