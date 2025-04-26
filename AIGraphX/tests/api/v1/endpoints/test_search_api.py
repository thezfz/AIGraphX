# -*- coding: utf-8 -*-
"""
文件目的：测试搜索 API 端点 (tests/api/v1/endpoints/test_search_api.py)

本文件包含针对 `/api/v1/search/papers/` 和 `/api/v1/search/models/` 端点的集成测试用例。
主要目标是验证这些搜索 API 端点：
1.  能否根据不同的搜索类型 (`semantic`, `keyword`, `hybrid` for papers) 和目标 (`papers`, `models`) 正确路由请求。
2.  能否正确解析和验证查询参数（`q`, `search_type`, `skip`, `limit`, `date_from`, `date_to`, `area`, `sort_by`, `sort_order` 等）。
3.  能否根据请求参数正确调用依赖的 `SearchService` 中的相应方法 (`perform_semantic_search`, `perform_keyword_search`, `perform_hybrid_search`)。
4.  能否处理 `SearchService` 返回的分页结果对象 (`Paginated...SearchResult`) 并将其正确序列化为 JSON 响应。
5.  能否在服务层发生错误时返回 500 错误。
6.  能否在接收到无效参数（如负数 `skip`, 无效日期格式, 无效枚举值）时返回 422 错误。

核心测试策略：
- 使用 `httpx.AsyncClient` 模拟异步 HTTP 客户端。
- 使用 `pytest` 测试框架及其 fixtures (`client`, `test_app`)。
- 使用 `unittest.mock.AsyncMock` 创建 `SearchService` 的模拟对象。
- 通过 FastAPI 的 `test_app.dependency_overrides` 机制，使用模拟服务替换真实的 `SearchService` 依赖。
- 覆盖不同搜索类型、目标以及包含各种过滤/排序条件的成功场景。
- 使用 `@pytest.mark.parametrize` 测试各种无效参数输入的场景。
- 验证模拟服务的方法调用是否符合预期（调用次数、参数）。

与其他文件的交互：
- 导入测试基础库 `pytest`, `unittest.mock`, `httpx`, `fastapi`。
- 导入 `numpy` 用于生成模拟嵌入向量（虽然在此版本中可能未直接使用）。
- 导入日期时间相关库 `datetime`, `date`, `timezone` 用于处理和生成模拟数据。
- 导入原始依赖提供函数 `aigraphx.api.v1.dependencies.get_search_service` 作为依赖覆盖的键。
- 导入服务类 `aigraphx.services.search_service.SearchService` 和枚举 `SearchTarget` 用于模拟和类型提示。
- 导入 Pydantic 模型 `aigraphx.models.search.*` 定义模拟数据和预期响应结构。
- 依赖 `tests/conftest.py` 提供 `client` 和 `test_app` fixtures。
"""

import pytest
from fastapi.testclient import (
    TestClient,
)  # 导入 FastAPI 的同步测试客户端（在此文件中未直接使用，推荐使用 AsyncClient）
from httpx import AsyncClient  # 导入异步 HTTP 客户端，用于发送测试请求
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)  # 导入模拟工具：AsyncMock (异步模拟), MagicMock (通用模拟), patch (修补/替换对象)
import numpy as np  # 导入 numpy，通常用于数值计算，这里可能用于创建模拟的向量嵌入（如果需要）
from fastapi import FastAPI  # 导入 FastAPI 应用类，主要用于类型提示和依赖覆盖
from typing import List, Dict, Optional, Union, Any  # 导入类型提示工具
from datetime import date, datetime, timezone  # 导入日期、时间、时区处理类

# 导入原始的依赖提供函数，用作依赖覆盖字典的键
from aigraphx.api.v1 import dependencies as deps

# 导入被测试端点依赖的服务类，用于模拟对象的 `spec` 参数，确保接口一致性
from aigraphx.services.search_service import SearchService, SearchTarget

# 明确导入依赖提供函数，用于在覆盖时指定键
from aigraphx.api.v1.dependencies import get_search_service

# 导入搜索相关的 Pydantic 模型，用于定义模拟数据和预期响应体结构
from aigraphx.models.search import (
    SearchResultItem,  # 单个论文搜索结果项的模型
    HFSearchResultItem,  # 单个 Hugging Face 模型搜索结果项的模型
    PaginatedPaperSearchResult,  # 分页的论文搜索结果模型 (通常用于关键字搜索)
    PaginatedSemanticSearchResult,  # 分页的语义搜索结果模型 (可能包含论文或模型)
    PaginatedHFModelSearchResult,  # 分页的 HF 模型搜索结果模型 (通常用于关键字搜索)
    SearchFilterModel,  # 搜索过滤器模型，用于封装各种过滤条件传递给服务层
)

# 移除直接导入主 app 的语句，测试应使用 conftest.py 中配置好的 test_app fixture
# from aigraphx.main import app

# 移除直接实例化同步 TestClient 的语句
# client = TestClient(app)

# pytest 标记，指示此模块中的所有 `async def` 测试函数都应使用 pytest-asyncio
# 提供的默认函数作用域事件循环来运行。
pytestmark = pytest.mark.asyncio


# --- 模拟数据 (论文) ---
# 创建一个 SearchResultItem 实例作为模拟论文搜索结果
MOCK_PAPER_RESULT = SearchResultItem(
    paper_id=1,
    pwc_id="pwc-1",
    title="Test Paper",
    summary="Test abstract",
    score=0.9,  # 分数通常由服务层设置，这里提供一个示例值
    pdf_url="http://example.com/abs",  # 注意：模型中似乎是 abs_url，这里可能需要统一
    published_date=date(2023, 1, 1),  # 使用 date 对象
    authors=["Auth1"],
    area="CV",
)

# --- 模拟数据 (模型) ---
# 获取当前 UTC 时间并格式化为 ISO 字符串
now_dt_str = datetime.now(timezone.utc).isoformat()
# 使用 HFSearchResultItem 模型定义的解析方法将字符串转换为 datetime 对象
# 这是必要的，因为模型字段期望 datetime 对象，而模拟数据可能从字符串开始
parsed_now_dt = HFSearchResultItem.parse_last_modified(now_dt_str)
# 断言确保解析成功
assert parsed_now_dt is not None

# 创建一个 HFSearchResultItem 实例作为模拟模型搜索结果
MOCK_MODEL_RESULT = HFSearchResultItem(
    model_id="org/test-model",
    author="Org",
    pipeline_tag="text-generation",
    library_name="transformers",
    tags=["test", "model"],
    likes=100,
    downloads=1000,
    last_modified=parsed_now_dt,  # 传入解析后的 datetime 对象
    score=0.8,  # 示例分数
)

# --- 测试用例 (仅论文搜索) ---


@pytest.mark.asyncio
async def test_search_semantic_papers_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功执行论文的语义搜索 (GET /api/v1/search/papers/?search_type=semantic)。
    预期：返回分页的语义搜索结果 (`PaginatedSemanticSearchResult`)。
    策略：覆盖 SearchService，使其 `perform_semantic_search` 方法返回模拟的分页结果。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)
    # 创建模拟的分页结果对象，包含一个模拟论文项
    mock_paginated_result = PaginatedSemanticSearchResult(
        items=[MOCK_PAPER_RESULT],  # 确保 items 列表中的类型是兼容的 (SearchResultItem)
        total=25,  # 模拟的总记录数
        skip=0,  # 模拟的跳过记录数
        limit=5,  # 模拟的每页限制数
    )
    # 配置模拟服务的 perform_semantic_search 方法返回这个模拟结果
    mock_service.perform_semantic_search.return_value = mock_paginated_result

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 3. 执行 API 调用 ---
        # 发送带有查询参数 q, search_type, skip, limit 的 GET 请求
        response = await client.get(
            "/api/v1/search/papers/?q=test&search_type=semantic&skip=0&limit=5"
        )

        # --- 4. 断言 ---
        assert response.status_code == 200
        # 验证响应 JSON 是否与模拟结果对象的 JSON 序列化形式匹配
        # model_dump(mode="json") 用于确保日期等类型被正确序列化
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # --- 5. 验证模拟调用 ---
        # 根据 API 请求中的 skip 和 limit 计算服务层期望接收的 page 和 page_size
        expected_skip = 0
        expected_limit = 5
        expected_page = expected_skip // expected_limit + 1  # 页码从 1 开始
        # 验证模拟服务的 perform_semantic_search 方法是否被以预期的参数调用
        mock_service.perform_semantic_search.assert_awaited_once_with(
            query="test",
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # 服务层方法需要知道目标是论文还是模型
            sort_by="score",  # API 端点对于语义搜索，若未指定 sort_by，默认为 'score'
            sort_order="desc",  # API 端点默认排序顺序为 'desc'
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_keyword_papers_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功执行论文的关键字搜索 (GET /api/v1/search/papers/?search_type=keyword)。
    预期：返回分页的论文搜索结果 (`PaginatedPaperSearchResult`)。
    策略：覆盖 SearchService，使其 `perform_keyword_search` 方法返回模拟的分页结果。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedPaperSearchResult(
        items=[MOCK_PAPER_RESULT], total=50, skip=5, limit=15
    )
    mock_service.perform_keyword_search.return_value = mock_paginated_result

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get(
            "/api/v1/search/papers/?q=graph&search_type=keyword&skip=5&limit=15"
        )

        # --- 4. 断言 ---
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # --- 5. 验证模拟调用 ---
        expected_skip = 5
        expected_limit = 15
        expected_page = expected_skip // expected_limit + 1
        # 验证 perform_keyword_search 的调用参数
        mock_service.perform_keyword_search.assert_awaited_once_with(
            query="graph",
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # 目标是论文
            date_from=None,  # 未提供日期过滤
            date_to=None,
            area=None,  # 未提供领域过滤
            sort_by="published_date",  # API 端点对于关键字搜索，若未指定 sort_by，默认为 'published_date'
            sort_order="desc",  # 默认排序顺序
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_papers_with_all_filters_sort(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：执行论文的关键字搜索，并应用所有可用的过滤器和自定义排序。
    预期：API 端点能正确解析所有过滤参数，并将其传递给服务层的 `perform_keyword_search` 方法。
    策略：覆盖 SearchService，使其 `perform_keyword_search` 返回模拟结果，并验证服务调用时的参数。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedPaperSearchResult(
        items=[MOCK_PAPER_RESULT], total=1, skip=0, limit=5
    )
    mock_service.perform_keyword_search.return_value = mock_paginated_result

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    # --- 定义测试参数 ---
    test_query = "filtered search"
    test_after = "2023-01-01"  # 日期过滤起始 (字符串)
    test_before = "2023-12-31"  # 日期过滤结束 (字符串)
    test_area = "CV"  # 领域过滤
    test_sort_by = "published_date"  # 自定义排序字段
    test_sort_order = "asc"  # 自定义排序顺序
    test_skip = 0
    test_limit = 5
    test_search_type = "keyword"

    try:
        # --- 3. 执行 API 调用 ---
        # 构建包含所有查询参数的 URL
        response = await client.get(
            f"/api/v1/search/papers/?q={test_query}"
            f"&search_type={test_search_type}"
            f"&skip={test_skip}"
            f"&limit={test_limit}"
            f"&date_from={test_after}"
            f"&date_to={test_before}"
            f"&area={test_area}"
            f"&sort_by={test_sort_by}"
            f"&sort_order={test_sort_order}"
        )

        # --- 4. 断言 ---
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # --- 5. 验证模拟调用 ---
        expected_skip = test_skip
        expected_limit = test_limit
        expected_page = expected_skip // expected_limit + 1
        # 验证 perform_keyword_search 的调用参数，注意日期参数应被转换为 date 对象
        mock_service.perform_keyword_search.assert_awaited_once_with(
            query=test_query,
            page=expected_page,
            page_size=expected_limit,
            target="papers",
            date_from=date.fromisoformat(
                test_after
            ),  # API 端点应将日期字符串解析为 date 对象
            date_to=date.fromisoformat(test_before),
            area=test_area,
            sort_by=test_sort_by,
            sort_order=test_sort_order,
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_papers_invalid_search_type(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：使用无效的 search_type 值请求论文搜索。
    预期：FastAPI 在参数验证阶段就应失败，返回 422 Unprocessable Entity。
    策略：发送带有无效 search_type 的请求，并断言状态码。不需要覆盖服务，因为请求在到达服务前就会失败。
    """
    # --- 1. (可选) 覆盖服务，虽然预计不会被调用 ---
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service
    try:
        # --- 2. 执行 API 调用 ---
        response = await client.get(
            "/api/v1/search/papers/?q=any&search_type=invalid"  # 使用无效的 search_type
        )
        # --- 3. 断言 ---
        assert response.status_code == 422  # FastAPI 基于 Pydantic Enum 的验证错误
    finally:
        # --- 4. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_hybrid_papers_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功执行论文的混合搜索 (GET /api/v1/search/papers/?search_type=hybrid)。
    预期：返回分页的论文搜索结果 (`PaginatedPaperSearchResult`)。
    策略：覆盖 SearchService，使其 `perform_hybrid_search` 方法返回模拟的分页结果。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)
    # 混合搜索通常返回 PaginatedPaperSearchResult，因为它结合了关键字和语义结果
    mock_hybrid_result = PaginatedPaperSearchResult(
        items=[MOCK_PAPER_RESULT], total=15, skip=0, limit=10
    )
    mock_service.perform_hybrid_search.return_value = mock_hybrid_result

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get(
            "/api/v1/search/papers/?q=hybrid&search_type=hybrid&skip=0&limit=10"
        )

        # --- 4. 断言 ---
        assert response.status_code == 200
        assert response.json() == mock_hybrid_result.model_dump(mode="json")

        # --- 5. 验证模拟调用 ---
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1

        # 创建服务层 perform_hybrid_search 期望接收的 filters 对象
        # 因为 API 端点没有传递具体的过滤参数，所以这里大部分是 None
        expected_filters = SearchFilterModel(
            published_after=None,
            published_before=None,
            filter_area=None,
            sort_by=None,  # API 端点未指定时传递 None
            sort_order="desc",  # API 默认值
            # 以下是模型特有的过滤器，对于论文搜索应为 None
            pipeline_tag=None,
            filter_authors=None,
            filter_library_name=None,
            filter_tags=None,
            filter_author=None,
        )

        # 验证 perform_hybrid_search 的调用参数
        mock_service.perform_hybrid_search.assert_awaited_once_with(
            query="hybrid",
            page=expected_page,
            page_size=expected_limit,
            target="papers",  # 目标是论文
            filters=expected_filters,  # 验证传递的 filters 对象
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_type, target",  # 参数化测试，覆盖不同搜索类型和目标的错误场景
    [
        ("semantic", "papers"),
        ("keyword", "papers"),
        ("hybrid", "papers"),
        ("semantic", "models"),
        ("keyword", "models"),  # 添加模型关键字搜索的错误场景
    ],
)
async def test_search_service_error(
    client: AsyncClient, test_app: FastAPI, search_type: str, target: str
) -> None:
    """
    测试场景：当对应的服务层搜索方法抛出异常时，API 端点应返回 500 错误。
    策略：参数化测试不同的 search_type 和 target 组合。覆盖 SearchService，使相应的搜索方法抛出异常。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)

    # --- 根据 search_type 和 target 配置相应的模拟方法抛出异常 ---
    if search_type == "semantic":
        mock_service.perform_semantic_search.side_effect = Exception("Service error")
    elif search_type == "keyword" and target == "papers":
        mock_service.perform_keyword_search.side_effect = Exception("Service error")
    elif search_type == "hybrid":
        mock_service.perform_hybrid_search.side_effect = Exception("Service error")
    elif search_type == "keyword" and target == "models":
        mock_service.perform_keyword_search.side_effect = Exception("Service error")
    else:
        # 跳过不适用的组合（例如，模型的 hybrid 搜索是不支持的，理论上不会调用该方法）
        pytest.skip(
            f"Skipping service error test for {search_type} with target {target}"
        )

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 3. 执行 API 调用 ---
        base_url = f"/api/v1/search/{target}/"  # 构建基础 URL
        response = await client.get(
            f"{base_url}?q=error&search_type={search_type}"  # 发送请求
        )
        # --- 4. 断言 ---
        assert response.status_code == 500  # 预期内部服务器错误
    finally:
        # --- 5. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# --- 测试用例 (模型搜索) ---


@pytest.mark.asyncio
async def test_search_semantic_models_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功执行模型的语义搜索 (GET /api/v1/search/models/?search_type=semantic)。
    预期：返回分页的语义搜索结果 (`PaginatedSemanticSearchResult`)，其中 items 是 HFSearchResultItem。
    策略：覆盖 SearchService，使其 `perform_semantic_search` 返回模拟的模型分页结果。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)
    mock_paginated_result = PaginatedSemanticSearchResult(
        items=[MOCK_MODEL_RESULT],  # items 包含模拟的模型结果
        total=10,
        skip=0,
        limit=5,
    )
    mock_service.perform_semantic_search.return_value = mock_paginated_result

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get(
            "/api/v1/search/models/?q=test&search_type=semantic&skip=0&limit=5"
        )

        # --- 4. 断言 ---
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # --- 5. 验证模拟调用 ---
        expected_skip = 0
        expected_limit = 5
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_semantic_search.assert_awaited_once_with(
            query="test",
            page=expected_page,
            page_size=expected_limit,
            target="models",  # 目标是模型
            sort_by="score",  # 假设模型语义搜索默认按分数排序
            sort_order="desc",
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_keyword_models_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功执行模型的关键字搜索 (GET /api/v1/search/models/?search_type=keyword)。
    预期：返回分页的模型搜索结果 (`PaginatedHFModelSearchResult`)。
    策略：覆盖 SearchService，使其 `perform_keyword_search` 返回模拟的模型分页结果。
    """
    # --- 1. 创建模拟服务 ---
    mock_service = AsyncMock(spec=SearchService)
    # 注意：关键字搜索模型应返回 PaginatedHFModelSearchResult
    mock_paginated_result = PaginatedHFModelSearchResult(
        items=[MOCK_MODEL_RESULT], total=30, skip=0, limit=10
    )
    mock_service.perform_keyword_search.return_value = mock_paginated_result

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get(
            "/api/v1/search/models/?q=bert&search_type=keyword&skip=0&limit=10"
        )

        # --- 4. 断言 ---
        assert response.status_code == 200
        assert response.json() == mock_paginated_result.model_dump(mode="json")

        # --- 5. 验证模拟调用 ---
        expected_skip = 0
        expected_limit = 10
        expected_page = expected_skip // expected_limit + 1
        mock_service.perform_keyword_search.assert_awaited_once_with(
            query="bert",
            page=expected_page,
            page_size=expected_limit,
            target="models",  # 目标是模型
            sort_by="likes",  # 假设模型关键字搜索默认按点赞数排序
            sort_order="desc",
            # 这里省略了日期、区域等论文特有的过滤器，因为它们不适用于模型搜索
            # 如果服务层需要接收完整的 SearchFilterModel，则应在这里断言相应的 None 值
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_search_models_invalid_search_type(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：使用无效的 search_type (如 hybrid) 请求模型搜索。
    预期：FastAPI 基于 ModelsSearchType Enum 进行验证，返回 422 错误。
    策略：发送带有无效 search_type 的请求。
    """
    # --- 1. (可选) 覆盖服务 ---
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 2. 执行 API 调用 ---
        # hybrid 对于模型搜索是无效的
        response = await client.get("/api/v1/search/models/?q=any&search_type=hybrid")
        # --- 3. 断言 ---
        assert response.status_code == 422
    finally:
        # --- 4. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# --- 测试用例 (参数验证 - 论文) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(  # 使用参数化测试各种无效的查询参数组合
    "param, value",
    [
        ("skip", -1),  # skip 不能为负
        ("limit", -1),  # limit 不能为负 (FastAPI 默认 > 0，但具体看实现)
        ("date_from", "invalid-date"),  # 无效日期格式
        ("date_to", "2023/01/01"),  # 无效日期格式 (示例)
        ("sort_order", "descending"),  # 无效的排序顺序枚举值
        (
            "sort_by",
            "title",
        ),  # 假设 'title' 不是有效的论文排序字段 (根据 PaperSortBy Enum)
        ("search_type", "fuzzy"),  # 无效的搜索类型枚举值 (根据 PapersSearchType Enum)
    ],
)
async def test_search_papers_invalid_query_params(
    client: AsyncClient, test_app: FastAPI, param: str, value: Any
) -> None:
    """
    测试场景：论文搜索请求包含无效的查询参数。
    预期：FastAPI 在验证阶段捕获错误，返回 422。
    策略：参数化测试不同的无效参数，发送请求并断言 422 状态码。
    """
    # --- 1. (可选) 覆盖服务 ---
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 2. 执行 API 调用 ---
        # 构建包含无效参数的 URL
        url = f"/api/v1/search/papers/?q=test&{param}={value}"
        response = await client.get(url)
        # --- 3. 断言 ---
        assert response.status_code == 422
    finally:
        # --- 4. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# --- 测试用例 (参数验证 - 模型) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(  # 参数化测试模型搜索的无效参数
    "param, value",
    [
        ("skip", -5),  # skip 不能为负
        ("limit", 0),  # limit 必须 >= 1
        (
            "search_type",
            "hybrid",
        ),  # 'hybrid' 对模型搜索无效 (根据 ModelsSearchType Enum)
        # 可以添加其他模型特有的无效参数测试，例如无效的 sort_by for models
    ],
)
async def test_search_models_invalid_query_params(
    client: AsyncClient, test_app: FastAPI, param: str, value: Any
) -> None:
    """
    测试场景：模型搜索请求包含无效的查询参数。
    预期：FastAPI 在验证阶段捕获错误，返回 422。
    策略：参数化测试不同的无效参数，发送请求并断言 422 状态码。
    """
    # --- 1. (可选) 覆盖服务 ---
    mock_service = AsyncMock(spec=SearchService)
    original_overrides = test_app.dependency_overrides.copy()
    test_app.dependency_overrides[get_search_service] = lambda: mock_service

    try:
        # --- 2. 执行 API 调用 ---
        url = f"/api/v1/search/models/?q=test&{param}={value}"
        response = await client.get(url)
        # --- 3. 断言 ---
        assert response.status_code == 422
    finally:
        # --- 4. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# 之前的 test_search_service_error 已经覆盖了服务层错误返回 500 的情况
# 无需重复添加仅针对错误的测试用例
