# -*- coding: utf-8 -*-
"""
文件目的：测试 FastAPI 依赖注入函数的健壮性和正确性。

本测试文件 (`test_dependencies.py`) 专注于验证位于 `aigraphx/api/v1/dependencies.py` 文件中定义的 FastAPI 依赖注入函数。
这些依赖函数负责在处理 API 请求时，创建和提供必要的服务实例、数据库连接、仓库对象等资源。

主要交互：
- 导入被测试的依赖函数：从 `aigraphx.api.v1.dependencies` 导入。
- 导入相关类型：从 `psycopg_pool`, `neo4j`, 以及项目内部模块（如 `vectorization`, `repositories`, `services`）导入类型提示所需的类。
- 使用 Pytest Fixtures：定义模拟对象（如 `mock_request`, `mock_state`）来模拟 FastAPI 请求上下文和应用状态 (`app.state`)，这是依赖函数运行的基础。
- 使用 `unittest.mock`：模拟（Mock）或修补（Patch）依赖函数的内部调用或外部依赖，以隔离测试单元并控制测试环境。例如，模拟数据库连接池、Neo4j驱动、Embedder服务等。
- 验证行为：
    - 成功场景：检查依赖函数在所有依赖项都可用时是否能正确返回预期的对象实例。
    - 失败/边界场景：检查当依赖项（如数据库连接池、Neo4j驱动、Faiss索引）不可用或未准备就绪时，依赖函数是否能按预期抛出 `HTTPException` 或记录警告/错误。
    - 异常处理：检查依赖函数在内部逻辑（如访问 `app.state`）遇到异常时的行为。

这个文件对于确保 API 层的稳定性和可靠性至关重要，因为它保证了 API 端点能够获得它们正常工作所需的所有依赖项。
"""

import pytest  # 导入 pytest 测试框架，用于编写和运行测试。
import pytest_asyncio  # 导入 pytest 的异步扩展，支持异步代码的测试 fixture。
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
)  # 从 FastAPI 框架导入核心类：FastAPI 应用、Request 请求对象、HTTPException 用于抛出 HTTP 错误。
from starlette.datastructures import (
    State,
)  # 从 Starlette（FastAPI 的基础）导入 State 类，用于管理应用级别的状态。
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)  # 从 Python 内置的 mock 库导入：AsyncMock (用于模拟异步对象), MagicMock (更灵活的模拟对象), patch (用于替换模块/类/方法)。
from typing import Any, Optional, Tuple  # 导入 Python 的类型提示工具。

# 导入被测试的依赖函数
# 这些函数定义在 aigraphx/api/v1/dependencies.py 中，负责提供 API 端点所需的各种依赖对象实例。
from aigraphx.api.v1.dependencies import (
    get_app_state,  # 获取 FastAPI 应用状态对象 (app.state)
    get_postgres_pool,  # 获取 PostgreSQL 连接池
    get_neo4j_driver,  # 获取 Neo4j 异步驱动实例
    get_embedder,  # 获取文本嵌入模型服务
    get_postgres_repository,  # 获取 PostgreSQL 数据仓库对象
    get_neo4j_repository,  # 获取 Neo4j 数据仓库对象
    get_faiss_repository_papers,  # 获取 Faiss 论文索引仓库对象
    get_faiss_repository_models,  # 获取 Faiss 模型索引仓库对象
    get_search_service,  # 获取搜索服务对象
    get_graph_service,  # 获取图服务对象
)

# 导入类型
# 这些是依赖函数返回或内部使用的对象的类型注解，有助于静态类型检查和代码可读性。
from psycopg_pool import AsyncConnectionPool  # PostgreSQL 异步连接池类型。
from neo4j import AsyncDriver  # Neo4j 异步驱动类型。
from aigraphx.vectorization.embedder import TextEmbedder  # 文本嵌入器类型。
from aigraphx.repositories.postgres_repo import (
    PostgresRepository,
)  # PostgreSQL 仓库类型。
from aigraphx.repositories.neo4j_repo import Neo4jRepository  # Neo4j 仓库类型。
from aigraphx.repositories.faiss_repo import FaissRepository  # Faiss 仓库类型。
from aigraphx.services.search_service import SearchService  # 搜索服务类型。
from aigraphx.services.graph_service import GraphService  # 图服务类型。


# --- Fixtures ---
# Fixtures 是 Pytest 的核心功能，用于设置测试环境和准备测试数据。
# 它们可以被测试函数作为参数请求，Pytest 会自动管理它们的创建和销毁。


@pytest.fixture
def mock_request() -> MagicMock:
    """
    创建一个模拟的 FastAPI Request 对象。

    这个 fixture 对于测试那些需要 `Request` 对象作为输入的依赖函数（如 `get_app_state`）非常重要。
    它模拟了一个带有 `app` 属性的请求对象，并且 `app` 对象又带有一个 `state` 属性 (类型为 `State`)。
    这模拟了 FastAPI 在处理请求时传递给依赖函数的真实请求对象结构。

    返回:
        MagicMock: 一个配置好的模拟 Request 对象。
    """
    # 创建一个 Request 对象的模拟实例
    mock = MagicMock(spec=Request)
    # 给模拟对象添加一个 app 属性，它也是一个模拟对象
    mock.app = MagicMock()
    # 给 app 模拟对象添加一个 state 属性，它是 Starlette 的 State 实例
    mock.app.state = State()
    return mock


@pytest.fixture
def mock_request_no_state() -> MagicMock:
    """
    创建一个没有 state 属性的模拟 Request 对象。

    这个 fixture 用于测试当 `request.app.state` 因为某种原因（例如，lifespan 初始化失败）不存在时的边界情况。
    它模拟了一个 `request.app` 对象，但故意删除了 `state` 属性。

    返回:
        MagicMock: 一个缺少 `state` 属性的模拟 Request 对象。
    """
    # 创建一个 Request 对象的模拟实例
    mock = MagicMock(spec=Request)
    # 添加 app 模拟对象
    mock.app = MagicMock()
    # 显式删除 app 模拟对象的 state 属性，模拟 state 不存在的情况
    if hasattr(mock.app, "state"):  # 防御性编程，确保属性存在再删除
        delattr(mock.app, "state")
    return mock


@pytest.fixture
def mock_request_wrong_state_type() -> MagicMock:
    """
    创建一个 state 属性类型不是 State 的模拟 Request 对象。

    这个 fixture 用于测试一个不太可能但可能的边缘情况：`request.app.state` 存在，但不是预期的 `State` 类型。
    这有助于验证依赖函数在这种意外情况下的健壮性。

    返回:
        MagicMock: 一个 `state` 属性类型错误的模拟 Request 对象。
    """
    # 创建一个 Request 对象的模拟实例
    mock = MagicMock(spec=Request)
    # 添加 app 模拟对象
    mock.app = MagicMock()
    # 将 app.state 设置为一个普通字典，而不是 State 对象
    mock.app.state = {}
    return mock


@pytest.fixture
def mock_state() -> State:
    """
    创建一个模拟的 State 对象，用于依赖测试。

    这个 fixture 提供了一个预先填充了各种模拟依赖项（如数据库连接池、驱动、仓库、嵌入器）的 `State` 对象。
    它使得测试那些直接依赖 `State` 对象（作为参数传入，而不是通过 `Request`）的依赖函数更加方便。
    这些模拟依赖项本身也是 `MagicMock` 对象，可以进一步配置它们的行为（例如，设置返回值）。

    返回:
        State: 一个包含模拟依赖项的 State 对象。
    """
    # 创建一个真实的 State 对象
    state = State()
    # 在 state 对象上设置各种模拟属性，模拟 lifespan 启动后应用状态中应包含的内容
    # 每个属性都设置为 MagicMock，并指定了 spec (模拟的原始类型)，这有助于保持接口一致性
    state.pg_pool = MagicMock(spec=AsyncConnectionPool)  # 模拟 PG 连接池
    state.neo4j_driver = MagicMock(spec=AsyncDriver)  # 模拟 Neo4j 驱动
    state.embedder = MagicMock(spec=TextEmbedder)  # 模拟文本嵌入器
    # 重要：确保模拟的 embedder 有一个 'model' 属性，因为 get_embedder 会检查它
    # 这里设置为 True，表示模型已加载
    state.embedder.model = True
    # 模拟 Faiss 论文仓库
    state.faiss_repo_papers = MagicMock(spec=FaissRepository)
    # 设置模拟仓库的 is_ready 方法返回 True，表示仓库已准备就绪
    state.faiss_repo_papers.is_ready.return_value = True
    # 模拟 Faiss 模型仓库
    state.faiss_repo_models = MagicMock(spec=FaissRepository)
    state.faiss_repo_models.is_ready.return_value = True
    return state


# --- 测试 get_app_state ---
# 这组测试验证 get_app_state 函数的行为。
# get_app_state 的主要职责是从传入的 Request 对象中安全地获取 app.state。


def test_get_app_state_success(mock_request: MagicMock) -> None:
    """
    测试场景：`request.app.state` 存在且类型正确。
    预期行为：`get_app_state` 应该成功返回 `request.app.state` 对象。
    """
    # 准备：从 fixture 获取模拟的 request 对象，并获取其预设的 state
    expected_state = mock_request.app.state

    # 执行：调用被测试的函数 `get_app_state`，传入模拟的 request 对象
    result = get_app_state(mock_request)

    # 验证：断言返回的结果 `result` 是否与预期的 `expected_state` 是同一个对象
    assert result is expected_state


def test_get_app_state_no_state_attribute(mock_request_no_state: MagicMock) -> None:
    """
    测试场景：`request.app` 对象上没有 `state` 属性。
    预期行为：`get_app_state` 应该捕获 `AttributeError` 并抛出 HTTP 500 错误。
    """
    # 执行与验证：使用 pytest.raises 上下文管理器来断言特定的异常是否被抛出
    with pytest.raises(HTTPException) as excinfo:
        # 在这个上下文中调用 get_app_state，传入缺少 state 的 request
        get_app_state(mock_request_no_state)

    # 验证异常的细节：
    # 1. 检查状态码是否为 500 (服务器内部错误)
    assert excinfo.value.status_code == 500
    # 2. 检查错误详情信息是否包含预期的文本
    assert "Application state not initialized" in excinfo.value.detail


def test_get_app_state_wrong_state_type(
    mock_request_wrong_state_type: MagicMock,
) -> None:
    """
    测试场景：`request.app.state` 存在，但不是预期的 `State` 类型。
    预期行为：函数应该记录一个警告，但仍然返回这个非预期的 state 对象（保持一定的容错性）。
    """
    # 执行：
    # 使用 patch 上下文管理器临时替换掉 `dependencies` 模块中的 `logger` 对象为 `mock_logger`
    # 这样我们可以检查 logger 是否被按预期调用
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        # 调用 get_app_state，传入 state 类型错误的 request
        result = get_app_state(mock_request_wrong_state_type)

    # 验证：
    # 1. 断言返回的结果仍然是 request 对象上的那个 state (即使类型不对)
    assert result is mock_request_wrong_state_type.app.state
    # 2. 断言 mock_logger 的 warning 方法被调用了一次
    mock_logger.warning.assert_called_once()
    # 3. 断言警告日志的内容包含了预期的提示信息
    assert "is not of type State" in mock_logger.warning.call_args[0][0]


# --- 测试 get_postgres_pool ---
# 这组测试验证 get_postgres_pool 函数的行为。
# get_postgres_pool 的职责是从 app.state 中安全地获取 PostgreSQL 连接池。


def test_get_postgres_pool_success() -> None:
    """
    测试场景：`state.pg_pool` 存在。
    预期行为：函数应该成功返回这个连接池对象。
    """
    # 准备：
    # 1. 创建一个模拟的 State 对象
    mock_state = MagicMock(spec=State)
    # 2. 创建一个模拟的连接池对象
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    # 3. 将模拟连接池设置到模拟 state 上
    mock_state.pg_pool = mock_pool

    # 执行：调用 get_postgres_pool，直接传入准备好的 mock_state (因为函数签名允许直接传入 state)
    result = get_postgres_pool(state=mock_state)

    # 验证：断言返回的结果是预期的模拟连接池对象
    assert result is mock_pool


def test_get_postgres_pool_not_available() -> None:
    """
    测试场景：`state` 对象上没有 `pg_pool` 属性。
    预期行为：函数应该捕获 `AttributeError` 并抛出 HTTP 503 错误 (服务不可用)。
    """
    # 准备：创建一个模拟的 State 对象，但不设置 pg_pool 属性
    mock_state = MagicMock(spec=State)
    # 故意不设置 mock_state.pg_pool

    # 执行与验证：使用 pytest.raises 捕获预期抛出的 HTTPException
    with pytest.raises(HTTPException) as excinfo:
        get_postgres_pool(state=mock_state)

    # 验证异常细节：
    # 1. 状态码应为 503
    assert excinfo.value.status_code == 503
    # 2. 错误详情信息应符合预期
    assert "Database connection pool is not available" in excinfo.value.detail


# --- 测试 get_neo4j_driver ---
# 这组测试验证 get_neo4j_driver 函数的行为。
# get_neo4j_driver 从 app.state 获取 Neo4j 驱动。与 PG 不同，如果驱动不存在，它会记录警告并返回 None，而不是抛出异常。


def test_get_neo4j_driver_success() -> None:
    """
    测试场景：`state.neo4j_driver` 存在。
    预期行为：函数应该成功返回这个驱动对象。
    """
    # 准备：
    mock_state = MagicMock(spec=State)
    mock_driver = MagicMock(spec=AsyncDriver)
    mock_state.neo4j_driver = mock_driver

    # 执行：调用 get_neo4j_driver
    result = get_neo4j_driver(state=mock_state)

    # 验证：返回结果应为模拟的驱动对象
    assert result is mock_driver


def test_get_neo4j_driver_not_available() -> None:
    """
    测试场景：`state` 对象上没有 `neo4j_driver` 属性。
    预期行为：函数应该记录一个警告，并返回 `None`。
    """
    # 准备：创建一个没有 neo4j_driver 属性的模拟 state
    mock_state = MagicMock(spec=State)
    # 故意不设置 mock_state.neo4j_driver

    # 执行：使用 patch 捕获 logger 调用
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_neo4j_driver(state=mock_state)

    # 验证：
    # 1. 返回结果应为 None
    assert result is None
    # 2. logger.warning 应被调用一次
    mock_logger.warning.assert_called_once()
    # 3. 警告日志内容应符合预期
    assert "Neo4j driver not initialized" in mock_logger.warning.call_args[0][0]


# --- 测试 get_embedder ---
# 这组测试验证 get_embedder 函数的行为。
# get_embedder 从 app.state 获取文本嵌入器实例，并检查其模型是否已加载。


def test_get_embedder_success() -> None:
    """
    测试场景：`state.embedder` 存在，并且其 `model` 属性为真值（表示模型已加载）。
    预期行为：函数应该成功返回嵌入器对象。
    """
    # 准备：
    mock_state = MagicMock(spec=State)
    mock_embedder = MagicMock(spec=TextEmbedder)
    mock_embedder.model = True  # 模拟模型已加载
    mock_state.embedder = mock_embedder

    # 执行：调用 get_embedder
    result = get_embedder(state=mock_state)

    # 验证：返回结果应为模拟的嵌入器对象
    assert result is mock_embedder


def test_get_embedder_not_available() -> None:
    """
    测试场景：`state` 对象上没有 `embedder` 属性。
    预期行为：函数应该捕获 `AttributeError` 并抛出 HTTP 503 错误。
    """
    # 准备：创建一个没有 embedder 属性的模拟 state
    mock_state = MagicMock(spec=State)
    # 故意不设置 mock_state.embedder

    # 执行与验证：捕获预期的 HTTPException
    with pytest.raises(HTTPException) as excinfo:
        get_embedder(state=mock_state)

    # 验证异常细节：
    assert excinfo.value.status_code == 503
    assert "Text embedding service is not available" in excinfo.value.detail


def test_get_embedder_no_model() -> None:
    """
    测试场景：`state.embedder` 存在，但其 `model` 属性为假值 (None, False 等，表示模型未加载)。
    预期行为：函数应该抛出 HTTP 503 错误。
    """
    # 准备：
    mock_state = MagicMock(spec=State)
    mock_embedder = MagicMock(spec=TextEmbedder)
    mock_embedder.model = None  # 模拟模型未加载
    mock_state.embedder = mock_embedder

    # 执行与验证：捕获预期的 HTTPException
    with pytest.raises(HTTPException) as excinfo:
        get_embedder(state=mock_state)

    # 验证异常细节：
    assert excinfo.value.status_code == 503
    assert "Text embedding service is not available" in excinfo.value.detail


# --- 测试 get_postgres_repository ---
# 这个测试验证 get_postgres_repository 函数。
# 这个函数比较简单，它接收一个连接池对象，然后用它来实例化 PostgresRepository。


def test_get_postgres_repository() -> None:
    """
    测试场景：给定一个模拟的 PostgreSQL 连接池。
    预期行为：函数应该返回一个 `PostgresRepository` 的实例，并且该实例的 `pool` 属性是传入的模拟连接池。
    """
    # 准备：创建一个模拟的连接池对象
    mock_pool = MagicMock(spec=AsyncConnectionPool)

    # 执行：调用 get_postgres_repository，传入模拟连接池
    result = get_postgres_repository(pool=mock_pool)

    # 验证：
    # 1. 确认返回的对象是 PostgresRepository 的实例
    assert isinstance(result, PostgresRepository)
    # 2. 确认实例的 pool 属性是传入的 mock_pool
    assert result.pool is mock_pool


# --- 测试 get_neo4j_repository ---
# 这组测试验证 get_neo4j_repository 函数。
# 这个函数接收一个 Neo4j 驱动对象。如果驱动存在，则用它实例化 Neo4jRepository；如果驱动为 None，则返回 None。


def test_get_neo4j_repository_with_driver() -> None:
    """
    测试场景：给定一个模拟的 Neo4j 驱动对象。
    预期行为：函数应该返回一个 `Neo4jRepository` 的实例，其 `driver` 属性是传入的模拟驱动。
    """
    # 准备：创建一个模拟的 Neo4j 驱动
    mock_driver = MagicMock(spec=AsyncDriver)

    # 执行：调用 get_neo4j_repository，传入模拟驱动
    result = get_neo4j_repository(driver=mock_driver)

    # 验证：
    # 1. 返回对象是 Neo4jRepository 实例
    assert isinstance(result, Neo4jRepository)
    # 2. 实例的 driver 属性是传入的 mock_driver
    assert result.driver is mock_driver


def test_get_neo4j_repository_without_driver() -> None:
    """
    测试场景：传入的 `driver` 参数为 `None`。
    预期行为：函数应该直接返回 `None`。
    """
    # 执行：调用 get_neo4j_repository，传入 None
    result = get_neo4j_repository(driver=None)

    # 验证：返回结果应为 None
    assert result is None


# --- 测试 get_faiss_repository_papers ---
# 这组测试验证 get_faiss_repository_papers 函数。
# 这个函数从 app.state 获取 Faiss 论文仓库实例，并检查其是否准备就绪 (is_ready)。


def test_get_faiss_repository_papers_success(mock_state: State) -> None:
    """
    测试场景：`state.faiss_repo_papers` 存在，并且其 `is_ready()` 方法返回 `True`。
    预期行为：函数应该成功返回这个 Faiss 仓库对象。
    """
    # 准备：使用 mock_state fixture，它已经包含了准备好的 faiss_repo_papers

    # 执行：调用 get_faiss_repository_papers，传入 mock_state
    result = get_faiss_repository_papers(state=mock_state)

    # 验证：
    # 1. 返回结果应为 state 中的 faiss_repo_papers 对象
    assert result is mock_state.faiss_repo_papers
    # 2. 确认模拟仓库的 is_ready 方法被调用了一次
    mock_state.faiss_repo_papers.is_ready.assert_called_once()


def test_get_faiss_repository_papers_not_available(mock_state: State) -> None:
    """
    测试场景：`state` 对象上没有 `faiss_repo_papers` 属性。
    预期行为：函数应该捕获 `AttributeError` 并抛出 HTTP 503 错误。
    """
    # 准备：将 mock_state 中的 faiss_repo_papers 设置为 None，模拟它不存在
    mock_state.faiss_repo_papers = None

    # 执行与验证：捕获预期的 HTTPException
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_papers(state=mock_state)

    # 验证异常细节：
    assert excinfo.value.status_code == 503
    assert "Papers vector search service is not available" in excinfo.value.detail


def test_get_faiss_repository_papers_not_ready(mock_state: State) -> None:
    """
    测试场景：`state.faiss_repo_papers` 存在，但其 `is_ready()` 方法返回 `False`。
    预期行为：函数应该抛出 HTTP 503 错误。
    """
    # 准备：修改 mock_state 中 faiss_repo_papers 的 is_ready 方法的返回值
    mock_state.faiss_repo_papers.is_ready.return_value = False

    # 执行与验证：捕获预期的 HTTPException
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_papers(state=mock_state)

    # 验证异常细节：
    assert excinfo.value.status_code == 503
    assert "Papers vector search service is not available" in excinfo.value.detail
    # 确认 is_ready 方法仍然被调用了
    mock_state.faiss_repo_papers.is_ready.assert_called_once()


# --- 测试 get_faiss_repository_models ---
# 这组测试与 get_faiss_repository_papers 非常相似，但针对的是模型 Faiss 仓库 (`faiss_repo_models`)。


def test_get_faiss_repository_models_success(mock_state: State) -> None:
    """
    测试场景：`state.faiss_repo_models` 存在且 `is_ready()` 返回 `True`。
    预期行为：函数应该成功返回 Faiss 模型仓库对象。
    """
    # 准备：使用 mock_state fixture

    # 执行：调用 get_faiss_repository_models
    result = get_faiss_repository_models(state=mock_state)

    # 验证：
    assert result is mock_state.faiss_repo_models
    mock_state.faiss_repo_models.is_ready.assert_called_once()


def test_get_faiss_repository_models_not_available(mock_state: State) -> None:
    """
    测试场景：`state` 对象上没有 `faiss_repo_models` 属性。
    预期行为：函数应该抛出 HTTP 503 错误。
    """
    # 准备：模拟仓库不存在
    mock_state.faiss_repo_models = None

    # 执行与验证：
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_models(state=mock_state)

    # 验证异常细节：
    assert excinfo.value.status_code == 503
    assert "Models vector search service is not available" in excinfo.value.detail


def test_get_faiss_repository_models_not_ready(mock_state: State) -> None:
    """
    测试场景：`state.faiss_repo_models` 存在但 `is_ready()` 返回 `False`。
    预期行为：函数应该抛出 HTTP 503 错误。
    """
    # 准备：模拟仓库未就绪
    mock_state.faiss_repo_models.is_ready.return_value = False

    # 执行与验证：
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_models(state=mock_state)

    # 验证异常细节：
    assert excinfo.value.status_code == 503
    assert "Models vector search service is not available" in excinfo.value.detail
    mock_state.faiss_repo_models.is_ready.assert_called_once()


# --- 测试 get_search_service ---
# 这组测试验证 get_search_service 函数。
# 这个函数是一个更复杂的依赖提供者，它聚合了多个其他依赖（仓库、嵌入器）来实例化 SearchService。


@pytest_asyncio.fixture
async def mock_request_with_state() -> MagicMock:
    """
    创建一个带有完整模拟状态的模拟 Request 对象（异步 fixture）。

    这个 fixture 用于测试 `get_search_service`，因为它需要从 `request.app.state` 中获取 `embedder`。
    它确保 `request.app.state.embedder` 存在并且 `model` 已加载。

    返回:
        MagicMock: 配置好的模拟 Request 对象。
    """
    mock = MagicMock(spec=Request)
    mock.app = MagicMock()
    state = State()
    state.embedder = MagicMock(spec=TextEmbedder)
    state.embedder.model = True  # 确保 embedder 和 model 都存在
    mock.app.state = state
    return mock


def test_get_search_service_success(
    mock_request_with_state: MagicMock, mock_state: State
) -> None:
    """
    测试场景：所有依赖项（Faiss 仓库、PG 仓库、Neo4j 仓库、Embedder）都可用。
    预期行为：函数应该成功创建一个 `SearchService` 实例，并将所有依赖项正确地传递给它。
    """
    # 准备：创建所有 SearchService 构造函数所需的模拟依赖项
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)

    # 执行：
    # 使用 patch 来模拟 get_app_state 函数的返回值，确保它返回我们准备好的 mock_state
    # 这是因为 get_search_service 内部会调用 get_app_state 来获取 embedder
    with patch("aigraphx.api.v1.dependencies.get_app_state", return_value=mock_state):
        # 调用 get_search_service，传入模拟的 request 和其他模拟的仓库依赖
        result = get_search_service(
            request=mock_request_with_state,  # 提供 request 以获取 embedder
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo,
        )

    # 验证：
    # 1. 确认返回的是 SearchService 实例
    assert isinstance(result, SearchService)
    # 2. 确认实例的各个属性被正确设置为了我们传入的模拟对象
    assert result.embedder is mock_state.embedder
    assert result.faiss_repo_papers is mock_faiss_papers
    assert result.faiss_repo_models is mock_faiss_models
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is mock_neo4j_repo


def test_get_search_service_no_embedder(mock_request_with_state: MagicMock) -> None:
    """
    测试场景：`request.app.state.embedder` 不存在。
    预期行为：函数应该能处理这种情况，创建 `SearchService` 实例，但其 `embedder` 属性为 `None`。
             （它不会抛出异常，因为 SearchService 的某些功能可能不需要 embedder）。
    """
    # 准备：修改模拟 request 的 state，使其 embedder 为 None
    mock_request_with_state.app.state.embedder = None
    # 准备其他模拟仓库
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)

    # 执行：使用 patch 捕获 logger 调用
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        # 调用 get_search_service
        result = get_search_service(
            request=mock_request_with_state,
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo,
        )

    # 验证：
    # 1. 仍然返回 SearchService 实例
    assert isinstance(result, SearchService)
    # 2. 实例的 embedder 属性应为 None
    assert result.embedder is None
    # 3. 确认其他仓库被正确设置
    assert result.faiss_repo_papers is mock_faiss_papers
    assert result.faiss_repo_models is mock_faiss_models
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is mock_neo4j_repo
    # 注意：在这种情况下，原代码逻辑似乎不会记录日志，所以我们不检查 mock_logger


def test_get_search_service_model_not_loaded(
    mock_request_with_state: MagicMock,
) -> None:
    """
    测试场景：`request.app.state.embedder` 存在，但其 `model` 属性为 `None` (模型未加载)。
    预期行为：函数应该记录一个警告，创建 `SearchService` 实例，但其 `embedder` 属性为 `None`。
    """
    # 准备：修改模拟 request 的 state，使其 embedder 的 model 属性为 None
    mock_request_with_state.app.state.embedder.model = None
    # 准备其他模拟仓库
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)

    # 执行：使用 patch 捕获 logger 调用
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        # 调用 get_search_service
        result = get_search_service(
            request=mock_request_with_state,
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo,
        )

    # 验证：
    # 1. 返回 SearchService 实例
    assert isinstance(result, SearchService)
    # 2. 实例的 embedder 属性应为 None
    assert result.embedder is None
    # 3. logger.warning 应被调用一次
    mock_logger.warning.assert_called_once()
    # 4. 警告日志内容应符合预期
    assert (
        "Embedder found in state but model not loaded"
        in mock_logger.warning.call_args[0][0]
    )


def test_get_search_service_get_app_state_exception(
    mock_request_with_state: MagicMock,
) -> None:
    """
    测试场景：内部调用 `get_app_state` 时抛出了异常。
    预期行为：函数应该捕获这个异常，记录一个错误，创建 `SearchService` 实例，但其 `embedder` 属性为 `None`。
    """
    # 准备：
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    # 定义一个当 get_app_state 被调用时要抛出的异常
    http_exception = HTTPException(
        status_code=500, detail="Test exception from get_app_state"
    )

    # 执行：
    # 使用 patch 来模拟 get_app_state，使其在被调用时抛出我们定义的异常 (side_effect)
    with patch(
        "aigraphx.api.v1.dependencies.get_app_state", side_effect=http_exception
    ) as mock_get_state:
        # 同时 patch logger 以便检查错误日志
        with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
            # 调用 get_search_service
            result = get_search_service(
                request=mock_request_with_state,
                faiss_repo_papers=mock_faiss_papers,
                faiss_repo_models=mock_faiss_models,
                pg_repo=mock_pg_repo,
                neo4j_repo=mock_neo4j_repo,
            )

    # 验证：
    # 1. 确认 get_app_state 被调用了
    mock_get_state.assert_called_once_with(mock_request_with_state)
    # 2. 确认返回了 SearchService 实例
    assert isinstance(result, SearchService)
    # 3. 确认实例的 embedder 属性为 None
    assert result.embedder is None
    # 4. 确认 logger.error 被调用了一次
    mock_logger.error.assert_called_once()
    # 5. 确认错误日志的内容
    assert "Could not get app state" in mock_logger.error.call_args[0][0]
    # 6. 确认原始异常信息被包含在日志参数中
    assert http_exception in mock_logger.error.call_args[0]


def test_get_search_service_general_exception(
    # 注意：这里不使用 mock_request_with_state，因为我们需要一个特殊的会抛异常的 embedder
) -> None:
    """
    测试场景：在检查 `state.embedder.model` 时发生了一个未预料到的通用异常 (例如 RuntimeError)。
    预期行为：函数应该捕获这个异常，记录一个错误，创建 `SearchService` 实例，但其 `embedder` 属性为 `None`。
    """
    # 准备：
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)

    # 创建一个特殊的 mock request，其 app.state.embedder 在访问 model 属性时会主动引发 RuntimeError
    mock_request = MagicMock(spec=Request)
    mock_app = MagicMock()
    mock_state_with_exception = State()

    # 定义一个特殊的类，其 model 属性访问会抛出异常
    class ExceptionEmbedder:
        @property
        def model(self) -> Any:
            raise RuntimeError("Simulated error accessing model")

    # 设置模拟对象关系
    mock_state_with_exception.embedder = ExceptionEmbedder()
    mock_app.state = mock_state_with_exception
    mock_request.app = mock_app

    # 执行：使用 patch 捕获 logger 调用
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        # 调用 get_search_service，传入这个特殊的 request
        result = get_search_service(
            request=mock_request,  # 使用我们特殊构造的 request
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo,
        )

    # 验证：
    # 1. 返回 SearchService 实例
    assert isinstance(result, SearchService)
    # 2. 实例的 embedder 属性应为 None
    assert result.embedder is None
    # 3. logger.error 应被调用一次
    mock_logger.error.assert_called_once()
    # 4. 错误日志内容应符合预期
    assert "Error checking for embedder" in mock_logger.error.call_args[0][0]
    # 5. 确认原始异常信息（RuntimeError）被包含在日志参数中
    assert isinstance(mock_logger.error.call_args[0][1], RuntimeError)


def test_get_search_service_no_neo4j_repo(
    mock_request_with_state: MagicMock, mock_state: State
) -> None:
    """
    测试场景：Neo4j 仓库 (`neo4j_repo`) 参数为 `None`。
    预期行为：函数应该记录一个警告，创建 `SearchService` 实例，其 `neo4j_repo` 属性为 `None`。
    """
    # 准备：
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    # Neo4j repo 显式设为 None

    # 执行：
    with patch("aigraphx.api.v1.dependencies.get_app_state", return_value=mock_state):
        with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
            # 调用 get_search_service，传入 neo4j_repo=None
            result = get_search_service(
                request=mock_request_with_state,
                faiss_repo_papers=mock_faiss_papers,
                faiss_repo_models=mock_faiss_models,
                pg_repo=mock_pg_repo,
                neo4j_repo=None,  # 传入 None
            )

    # 验证：
    # 1. 返回 SearchService 实例
    assert isinstance(result, SearchService)
    # 2. 实例的 neo4j_repo 属性应为 None
    assert result.neo4j_repo is None
    # 3. logger.warning 应被调用一次
    mock_logger.warning.assert_called_once()
    # 4. 警告日志内容应符合预期
    assert "Neo4j repository is None" in mock_logger.warning.call_args[0][0]


# --- 测试 get_graph_service ---
# 这组测试验证 get_graph_service 函数。
# 这个函数聚合 PG 和 Neo4j 仓库来实例化 GraphService。


def test_get_graph_service_with_neo4j() -> None:
    """
    测试场景：PG 仓库和 Neo4j 仓库都可用。
    预期行为：函数应该成功创建 `GraphService` 实例，并正确设置仓库属性，不记录警告。
    """
    # 准备：创建模拟的 PG 和 Neo4j 仓库
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)

    # 执行：使用 patch 捕获 logger 调用（主要为了验证它没被调用）
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        # 调用 get_graph_service
        result = get_graph_service(pg_repo=mock_pg_repo, neo4j_repo=mock_neo4j_repo)

    # 验证：
    # 1. 返回 GraphService 实例
    assert isinstance(result, GraphService)
    # 2. 确认仓库属性设置正确
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is mock_neo4j_repo
    # 3. 确认 logger.warning 没有被调用
    mock_logger.warning.assert_not_called()


def test_get_graph_service_without_neo4j() -> None:
    """
    测试场景：Neo4j 仓库 (`neo4j_repo`) 参数为 `None`。
    预期行为：函数应该记录一个警告，创建 `GraphService` 实例，其 `neo4j_repo` 属性为 `None`。
    """
    # 准备：只创建模拟的 PG 仓库
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    # Neo4j repo 设为 None

    # 执行：使用 patch 捕获 logger 调用
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        # 调用 get_graph_service，传入 neo4j_repo=None
        result = get_graph_service(pg_repo=mock_pg_repo, neo4j_repo=None)

    # 验证：
    # 1. 返回 GraphService 实例
    assert isinstance(result, GraphService)
    # 2. 确认 pg_repo 设置正确
    assert result.pg_repo is mock_pg_repo
    # 3. 确认 neo4j_repo 为 None
    assert result.neo4j_repo is None
    # 4. 确认 logger.warning 被调用一次
    mock_logger.warning.assert_called_once()
    # 5. 确认警告日志内容
    assert "Neo4j repository is None" in mock_logger.warning.call_args[0][0]
