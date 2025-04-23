# -*- coding: utf-8 -*-
"""
文件目的：测试图谱 API 端点 (tests/api/v1/endpoints/test_graph_api.py)

这个文件包含了针对 `/api/v1/graph/` 路径下所有 API 端点的集成测试用例。
主要目标是验证这些 API 端点：
1.  能否正确处理不同类型的 HTTP 请求（GET）。
2.  能否正确解析路径参数和查询参数。
3.  能否正确调用依赖的 `GraphService` 中的相应方法。
4.  能否根据 `GraphService` 的返回结果（成功数据、空数据或异常）生成正确的 HTTP 响应（状态码、响应体）。

核心测试策略：
- 使用 `httpx.AsyncClient` 模拟异步 HTTP 客户端向测试应用发送请求。
- 使用 `pytest` 作为测试框架，利用其 fixture 机制管理测试环境（如 `client`, `test_app`）。
- 使用 `unittest.mock.AsyncMock` 创建 `GraphService` 的模拟对象。
- 通过 FastAPI 的 `test_app.dependency_overrides` 机制，在测试运行时将真实的 `GraphService` 依赖替换为模拟对象。这使得测试能够专注于 API 端点本身的逻辑，而不需要依赖真实的数据库或服务层实现。
- 对每个端点，覆盖成功、资源未找到 (404) 和内部服务器错误 (500) 等关键场景。
- 使用 `mock_*.assert_awaited_once_with(...)` 验证模拟服务的方法是否被以预期的参数调用。

与其他文件的交互：
- 导入 `pytest` 和 `unittest.mock` 用于测试和模拟。
- 导入 `httpx.AsyncClient` 发送异步 HTTP 请求。
- 导入 `fastapi.FastAPI` 用于类型提示和依赖覆盖。
- 导入 `aigraphx.api.v1.dependencies` (别名 `deps`) 以获取原始的依赖提供函数，作为覆盖字典的键。
- 导入 `aigraphx.services.graph_service.GraphService` 以便为模拟对象提供 `spec`，确保模拟对象具有与真实服务相同的接口。
- 导入 `aigraphx.models.graph` 中的 Pydantic 模型 (`GraphData`, `Node`, `Relationship`, `HFModelDetail`, `PaperDetailResponse`) 用于定义模拟数据和验证响应结构。
- 依赖 `tests/conftest.py` 文件提供必要的 pytest fixtures，特别是 `client` (基于 `test_app` 创建的 `AsyncClient`) 和 `test_app` (配置了 lifespan 和依赖覆盖的 FastAPI 应用实例)。对于 `/related` 端点的测试，还隐式使用了可能在 `conftest.py` 中定义的 `mock_graph_service_fixture`。

注意：对于依赖注入覆盖，此文件遵循最佳实践，即：
1. 在测试函数签名中注入 `test_app: FastAPI` fixture。
2. 将 `test_app.dependency_overrides` 应用于这个注入的 `test_app` 实例，而不是直接从主应用模块导入 `app`。
3. 在每个测试的 `finally` 块中恢复原始的 `dependency_overrides`，确保测试之间的隔离性。
"""

import pytest
from unittest.mock import AsyncMock, patch  # 导入 AsyncMock 用于异步模拟，patch 用于可能的猴子补丁（此文件未使用）
from httpx import AsyncClient  # 导入用于发送异步 HTTP 请求的客户端
from unittest.mock import (
    AsyncMock,
    MagicMock,
)  # 再次导入 AsyncMock (冗余)，MagicMock 用于创建通用模拟对象（此文件未使用）
from datetime import datetime, date  # 导入日期时间类，用于创建模拟数据

# 导入原始的依赖提供函数，它们将作为依赖覆盖字典中的键
from aigraphx.api.v1 import dependencies as deps

# 导入被测试端点所依赖的服务类，主要用于为模拟对象设置 `spec`
from aigraphx.services.graph_service import GraphService

# 导入 API 端点可能返回或服务层使用的 Pydantic 模型
from aigraphx.models.graph import (
    GraphData,  # 图数据模型（节点和关系）
    Node,  # 图节点模型
    Relationship,  # 图关系模型
    HFModelDetail,  # Hugging Face 模型详情模型
    PaperDetailResponse,  # 论文详情响应模型 (取代了旧的 SearchResultItem)
)

# from aigraphx.models.search import SearchResultItem # 旧模型注释，已被 PaperDetailResponse 替代
# from aigraphx.services.search_service import SearchService # 搜索服务未在此测试文件中直接模拟
from fastapi import FastAPI  # 导入 FastAPI 应用类，主要用于类型提示和访问依赖覆盖

# pytest 标记，指示此模块中的所有 `async def` 测试函数都应使用 pytest-asyncio
# 提供的默认函数作用域事件循环来运行。
pytestmark = pytest.mark.asyncio

# --- 模拟数据 ---
# 创建一些 Pydantic 模型实例作为模拟服务调用的返回值，或用于断言响应内容。

# 模拟 /papers/{pwc_id}/graph 端点的成功响应数据
MOCK_GRAPH_DATA = GraphData(
    nodes=[  # 图中的节点列表
        Node(id="pwc-1", label="Center Paper", type="Paper", properties={}),
        Node(id="pwc-2", label="Cited Paper", type="Paper", properties={}),
    ],
    relationships=[  # 图中的关系列表
        Relationship(source="pwc-1", target="pwc-2", type="CITES", properties={})
    ],
)

# 模拟 /models/{model_id} 端点的成功响应数据 (Hugging Face 模型详情)
MOCK_MODEL_DETAIL = HFModelDetail(
    model_id="hf-model-1",
    author="TestAuthor",
    sha="abcdef123",
    last_modified=datetime.now(),
    tags=["test"],
    pipeline_tag="text-classification",
    downloads=100,
    likes=10,
    library_name="transformers",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

# 模拟 /papers/{pwc_id} 端点的成功响应数据 (论文详情)
# 注意：这里使用了 PaperDetailResponse 模型，与实际端点返回的模型保持一致。
MOCK_PAPER_DETAIL = PaperDetailResponse(
    pwc_id="pwc-found",
    title="Mock Paper Detail",
    abstract="Mock abstract.",
    url_abs="http://example.com/abs",
    url_pdf="http://example.com/pdf",
    published_date=date(2023, 2, 15),
    authors=["Author A", "Author B"],
    tasks=["Task A"],
    datasets=["Dataset X"],
    methods=[],
    frameworks=[],
    number_of_stars=10,
    area="NLP",
)

# 模拟图邻居端点（未在此文件中直接测试，但可能由 GraphService 使用）的返回数据结构
# (注意：这不是一个直接的 Pydantic 模型，而是一个字典结构，可能用于前端可视化)
MOCK_PAPER_NEIGHBORHOOD_DATA = {
    "nodes": [
        {"id": "pwc-found", "label": "Paper", "title": "Found Paper Title"},
        {"id": "pwc-related-1", "label": "Paper", "title": "Related Paper 1"},
    ],
    "edges": [{"from": "pwc-found", "to": "pwc-related-1", "label": "CITES"}],
}

# --- 测试用例 ---


@pytest.mark.asyncio
# 测试函数签名包含 client 和 test_app fixtures
# client: httpx.AsyncClient 实例，由 conftest.py 提供，用于发送异步请求
# test_app: FastAPI 应用实例，由 conftest.py 提供，用于管理依赖覆盖
async def test_get_paper_graph_success(client: AsyncClient, test_app: FastAPI) -> None:
    """
    测试场景：成功获取论文图谱数据 (GET /api/v1/graph/papers/{pwc_id}/graph)。
    策略：覆盖 GraphService 依赖，使其返回预定义的模拟图谱数据。
    """
    # --- 1. 创建模拟服务 ---
    # 创建 GraphService 的异步模拟对象 (AsyncMock)
    # spec=GraphService 确保模拟对象具有与真实 GraphService 类相同的属性和方法签名
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟对象的 get_paper_graph 方法，使其在被调用时返回预定义的 MOCK_GRAPH_DATA
    # 这里使用 AsyncMock 是因为 get_paper_graph 是一个 async 方法
    mock_graph_service.get_paper_graph = AsyncMock(return_value=MOCK_GRAPH_DATA)

    # --- 2. 应用依赖覆盖 ---
    # 保存原始的依赖覆盖字典，以便在测试结束时恢复
    original_overrides = test_app.dependency_overrides.copy()
    # 定义覆盖规则：当 FastAPI 需要 `deps.get_graph_service` 这个依赖时，
    # 不再执行原始的 `get_graph_service` 函数，而是执行 lambda 函数，返回我们创建的模拟对象。
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    # 将覆盖规则更新到测试应用实例 (`test_app`) 的依赖覆盖字典中
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        # 使用模拟客户端向目标端点发送 GET 请求
        response = await client.get("/api/v1/graph/papers/pwc-1/graph")

        # --- 4. 断言 ---
        # 验证 HTTP 状态码是否为 200 OK
        assert response.status_code == 200
        # 解析响应的 JSON 数据
        data = response.json()
        # 验证响应数据的内容是否符合预期（例如，检查第一个节点的 ID）
        assert data["nodes"][0]["id"] == "pwc-1"
        # 可以进行更严格的检查，例如比较整个响应数据与模拟数据的 JSON 序列化结果
        # assert data == MOCK_GRAPH_DATA.model_dump(mode='json')

        # --- 5. 验证模拟调用 ---
        # 验证模拟服务的 get_paper_graph 方法是否被精确地调用了一次，
        # 并且参数是 "pwc-1"
        mock_graph_service.get_paper_graph.assert_awaited_once_with("pwc-1")
    finally:
        # --- 6. 恢复依赖覆盖 ---
        # 无论测试成功还是失败，在 finally 块中将依赖覆盖恢复到原始状态
        # 这对于确保测试之间的隔离性至关重要
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_paper_graph_not_found(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：请求的论文图谱数据未找到 (GET /api/v1/graph/papers/{pwc_id}/graph 返回 404)。
    策略：覆盖 GraphService 依赖，使其 `get_paper_graph` 方法返回 None。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟对象的 get_paper_graph 方法返回 None，模拟找不到数据的情况
    mock_graph_service.get_paper_graph.return_value = None

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        # 使用一个虚构的 ID ("not-found-id") 发送请求
        response = await client.get("/api/v1/graph/papers/not-found-id/graph")

        # --- 4. 断言 ---
        # 验证 HTTP 状态码是否为 404 Not Found
        assert response.status_code == 404
        # 验证响应 JSON 中的 "detail" 字段是否包含预期的错误信息
        assert "Graph data not found" in response.json()["detail"]

        # --- 5. 验证模拟调用 ---
        # 验证模拟服务的方法是否被以预期的参数调用
        mock_graph_service.get_paper_graph.assert_awaited_once_with("not-found-id")
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_paper_graph_error(client: AsyncClient, test_app: FastAPI) -> None:
    """
    测试场景：在获取论文图谱数据过程中发生内部错误 (GET /api/v1/graph/papers/{pwc_id}/graph 返回 500)。
    策略：覆盖 GraphService 依赖，使其 `get_paper_graph` 方法抛出异常。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟对象的 get_paper_graph 方法，使其在被调用时抛出指定的异常
    mock_graph_service.get_paper_graph = AsyncMock(
        side_effect=Exception("Service Graph Error")
    )

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        # 使用一个虚构的 ID ("error-id") 发送请求
        response = await client.get("/api/v1/graph/papers/error-id/graph")

        # --- 4. 断言 ---
        # 验证 HTTP 状态码是否为 500 Internal Server Error
        assert response.status_code == 500
        # 验证响应 JSON 中的 "detail" 字段是否包含预期的错误信息
        # 注意：通常全局异常处理器会返回一个通用的错误消息
        assert "Internal server error" in response.json()["detail"]

        # --- 5. 验证模拟调用 ---
        # 验证模拟服务的方法是否被以预期的参数调用
        mock_graph_service.get_paper_graph.assert_awaited_once_with("error-id")
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_model_details_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功获取模型详情 (GET /api/v1/graph/models/{model_id})。
    策略：覆盖 GraphService 依赖，使其返回预定义的模拟模型详情数据。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)

    # --- 配置模拟返回值 ---
    # 创建一个 HFModelDetail 实例作为模拟方法的返回值
    mock_return_data = HFModelDetail(
        model_id="hf-model-1",
        author="Mock Author",
        sha="mocksha123",
        last_modified=datetime.now(),
        tags=["tag1", "tag2"],
        pipeline_tag="text-classification",
        downloads=100,
        likes=10,
        library_name="transformers",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    # 配置模拟对象的 get_model_details 方法返回这个实例
    mock_graph_service.get_model_details.return_value = mock_return_data

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get("/api/v1/graph/models/hf-model-1")

        # --- 4. 断言 ---
        # 验证状态码
        assert response.status_code == 200
        # 解析响应数据
        data = response.json()
        # 验证关键字段的值
        assert data["model_id"] == "hf-model-1"
        assert data["author"] == "Mock Author"
        # 验证日期时间字段是否被序列化为字符串
        assert isinstance(data["last_modified"], str)

        # --- 5. 验证模拟调用 ---
        mock_graph_service.get_model_details.assert_awaited_once_with("hf-model-1")
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_model_details_not_found(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：请求的模型详情未找到 (GET /api/v1/graph/models/{model_id} 返回 404)。
    策略：覆盖 GraphService 依赖，使其 `get_model_details` 方法返回 None。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟方法返回 None
    mock_graph_service.get_model_details.return_value = None

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get("/api/v1/graph/models/not-found-model")

        # --- 4. 断言 ---
        assert response.status_code == 404
        # 验证错误详情
        assert "Model details not found" in response.json()["detail"]

        # --- 5. 验证模拟调用 ---
        mock_graph_service.get_model_details.assert_awaited_once_with("not-found-model")
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_model_details_error(client: AsyncClient, test_app: FastAPI) -> None:
    """
    测试场景：在获取模型详情过程中发生内部错误 (GET /api/v1/graph/models/{model_id} 返回 500)。
    策略：覆盖 GraphService 依赖，使其 `get_model_details` 方法抛出异常。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟方法抛出异常
    mock_graph_service.get_model_details.side_effect = Exception("Service Model Error")

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get("/api/v1/graph/models/error-model")

        # --- 4. 断言 ---
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

        # --- 5. 验证模拟调用 ---
        mock_graph_service.get_model_details.assert_awaited_once_with("error-model")
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_paper_details_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：成功获取论文详情 (GET /api/v1/graph/papers/{pwc_id})。
    策略：覆盖 GraphService 依赖，使其返回预定义的模拟论文详情数据 (PaperDetailResponse)。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟对象的 get_paper_details 方法返回预定义的 MOCK_PAPER_DETAIL
    mock_graph_service.get_paper_details.return_value = MOCK_PAPER_DETAIL

    # --- 2. 应用依赖覆盖 ---
    # 注意：这里仍然覆盖的是 get_graph_service，因为 GraphService 提供了 get_paper_details 方法
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. 执行 API 调用 ---
        # 使用 MOCK_PAPER_DETAIL 中的 pwc_id 发送请求
        response = await client.get("/api/v1/graph/papers/pwc-found")

        # --- 4. 断言 ---
        assert response.status_code == 200
        # 验证响应数据中的 pwc_id 是否与模拟数据一致
        assert response.json()["pwc_id"] == MOCK_PAPER_DETAIL.pwc_id

        # --- 5. 验证模拟调用 ---
        # 验证模拟服务的 get_paper_details 方法是否被以正确的参数调用
        mock_graph_service.get_paper_details.assert_awaited_once_with(
            pwc_id="pwc-found"
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_paper_details_not_found(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：请求的论文详情未找到 (GET /api/v1/graph/papers/{pwc_id} 返回 404)。
    策略：覆盖 GraphService 依赖，使其 `get_paper_details` 方法返回 None。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟方法返回 None
    mock_graph_service.get_paper_details.return_value = None

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)
    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get("/api/v1/graph/papers/pwc-not-found")

        # --- 4. 断言 ---
        assert response.status_code == 404
        # 验证错误详情
        assert "Paper with PWC ID" in response.json()["detail"]

        # --- 5. 验证模拟调用 ---
        mock_graph_service.get_paper_details.assert_awaited_once_with(
            pwc_id="pwc-not-found"
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
async def test_get_paper_details_service_error(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """
    测试场景：在获取论文详情过程中发生内部错误 (GET /api/v1/graph/papers/{pwc_id} 返回 500)。
    策略：覆盖 GraphService 依赖，使其 `get_paper_details` 方法抛出异常。
    """
    # --- 1. 创建模拟服务 ---
    mock_graph_service = AsyncMock(spec=GraphService)
    # 配置模拟方法抛出异常
    mock_graph_service.get_paper_details.side_effect = Exception(
        "Graph Service Internal Error"
    )

    # --- 2. 应用依赖覆盖 ---
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)
    try:
        # --- 3. 执行 API 调用 ---
        response = await client.get("/api/v1/graph/papers/pwc-error")

        # --- 4. 断言 ---
        assert response.status_code == 500
        # 验证错误详情，期望包含通用错误信息和请求的ID
        assert "An internal server error occurred" in response.json()["detail"]
        assert "pwc-error" in response.json()["detail"]

        # --- 5. 验证模拟调用 ---
        mock_graph_service.get_paper_details.assert_awaited_once_with(
            pwc_id="pwc-error"
        )
    finally:
        # --- 6. 恢复依赖覆盖 ---
        test_app.dependency_overrides = original_overrides


# --- 测试 GET /api/v1/graph/related/{start_node_label}/{start_node_prop}/{start_node_val} 端点 ---
# 注意：以下的测试使用了名为 `mock_graph_service_fixture` 的 Fixture，
# 这与前面的测试不同（前面是在测试函数内部创建 Mock 并覆盖 test_app）。
# 这个 Fixture 可能在 conftest.py 中定义，并自动应用了依赖覆盖。


@pytest.mark.asyncio
async def test_get_related_entities_success(
    client: AsyncClient, mock_graph_service_fixture: AsyncMock
) -> None:
    """
    测试场景：成功获取相关实体列表。
    策略：使用 `mock_graph_service_fixture` (假设已覆盖依赖)，配置其 `get_related_entities` 方法返回模拟数据。
    """
    # 定义请求参数
    start_label = "Paper"
    start_prop = "pwc_id"
    start_val = "test-paper"
    rel_type = "HAS_TASK"
    target_label = "Task"
    direction = "OUT"
    limit = 5

    # 配置模拟服务的返回值
    mock_return_data = [
        {"name": "Task A", "id": "task-a"},
        {"name": "Task B", "id": "task-b"},
    ]
    # 假设 mock_graph_service_fixture 已经是配置好的 GraphService 模拟对象
    mock_graph_service_fixture.get_related_entities.return_value = mock_return_data

    # 构建请求 URL 和查询参数
    url = f"/api/v1/graph/related/{start_label}/{start_prop}/{start_val}"
    params = {
        "relationship_type": rel_type,
        "target_node_label": target_label,
        "direction": direction,
        "limit": str(limit),  # 查询参数通常是字符串
    }
    # 发送 GET 请求
    response = await client.get(url, params=params)

    # 断言响应状态码和内容
    assert response.status_code == 200
    assert response.json() == mock_return_data

    # 验证模拟服务的方法是否被以正确的参数调用
    mock_graph_service_fixture.get_related_entities.assert_awaited_once_with(
        start_node_label=start_label,
        start_node_prop=start_prop,
        start_node_val=start_val,
        relationship_type=rel_type,
        target_node_label=target_label,
        direction=direction,
        limit=limit,
    )


@pytest.mark.asyncio
async def test_get_related_entities_not_found(
    client: AsyncClient, mock_graph_service_fixture: AsyncMock
) -> None:
    """
    测试场景：获取相关实体时，服务返回空列表（表示未找到或没有相关实体）。
    策略：配置模拟服务的 `get_related_entities` 方法返回空列表。
    """
    # 配置模拟服务返回空列表
    mock_graph_service_fixture.get_related_entities.return_value = []

    # 构建请求 URL 和参数
    url = "/api/v1/graph/related/Paper/pwc_id/not-found-paper"
    params = {"relationship_type": "HAS_TASK", "target_node_label": "Task"}
    # 发送请求
    response = await client.get(url, params=params)

    # 断言状态码为 200，响应体为空列表
    assert response.status_code == 200
    assert response.json() == []
    # 验证模拟服务的方法被调用
    mock_graph_service_fixture.get_related_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_related_entities_missing_params(client: AsyncClient) -> None:
    """
    测试场景：请求缺少必需的查询参数 (`relationship_type`, `target_node_label`)。
    预期：FastAPI 在处理依赖注入之前进行参数验证，返回 422 Unprocessable Entity。
    注意：此测试不涉及模拟服务，因为它在验证阶段就失败了。
    """
    # 构建缺少查询参数的 URL
    url = "/api/v1/graph/related/Paper/pwc_id/some-paper"
    # 发送请求
    response = await client.get(url)
    # 断言状态码为 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_related_entities_invalid_direction(client: AsyncClient) -> None:
    """
    测试场景：请求中的 `direction` 查询参数值无效 (不是 "IN", "OUT", 或 "BOTH")。
    预期：FastAPI 在处理依赖注入之前进行参数验证（基于 Enum），返回 422 Unprocessable Entity。
    注意：此测试不涉及模拟服务。
    """
    # 构建包含无效 direction 参数的 URL 和参数
    url = "/api/v1/graph/related/Paper/pwc_id/some-paper"
    params = {
        "relationship_type": "HAS_TASK",
        "target_node_label": "Task",
        "direction": "INVALID_DIRECTION",  # 无效值
    }
    # 发送请求
    response = await client.get(url, params=params)
    # 断言状态码为 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_related_entities_service_error(
    client: AsyncClient, mock_graph_service_fixture: AsyncMock
) -> None:
    """
    测试场景：获取相关实体时，依赖的 GraphService 内部发生意外错误。
    策略：配置模拟服务的 `get_related_entities` 方法抛出异常。
    预期：端点的异常处理器捕获异常，返回 500 Internal Server Error。
    """
    # 配置模拟服务抛出异常
    mock_graph_service_fixture.get_related_entities.side_effect = Exception(
        "Service layer crashed"
    )

    # 构建请求 URL 和参数
    url = "/api/v1/graph/related/Paper/pwc_id/error-paper"
    params = {"relationship_type": "HAS_TASK", "target_node_label": "Task"}
    # 发送请求
    response = await client.get(url, params=params)

    # 断言状态码为 500
    assert response.status_code == 500
    # 断言错误详情包含通用错误信息
    assert "Internal server error" in response.json()["detail"]
    # 验证模拟服务的方法被调用
    mock_graph_service_fixture.get_related_entities.assert_awaited_once()
