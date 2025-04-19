import pytest
import pytest_asyncio
from fastapi import FastAPI, Request, HTTPException
from starlette.datastructures import State
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Optional, Tuple

# 导入被测试的依赖函数
from aigraphx.api.v1.dependencies import (
    get_app_state,
    get_postgres_pool,
    get_neo4j_driver,
    get_embedder,
    get_postgres_repository,
    get_neo4j_repository,
    get_faiss_repository_papers,
    get_faiss_repository_models,
    get_search_service,
    get_graph_service,
)

# 导入类型
from psycopg_pool import AsyncConnectionPool
from neo4j import AsyncDriver
from aigraphx.vectorization.embedder import TextEmbedder
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.services.search_service import SearchService
from aigraphx.services.graph_service import GraphService


# --- Fixtures ---

@pytest.fixture
def mock_request() -> MagicMock:
    """创建一个模拟的 Request 对象，带有 state 属性。"""
    mock = MagicMock(spec=Request)
    mock.app = MagicMock()
    mock.app.state = State()
    return mock


@pytest.fixture
def mock_request_no_state() -> MagicMock:
    """创建一个没有 state 属性的模拟 Request 对象。"""
    mock = MagicMock(spec=Request)
    mock.app = MagicMock()
    # 删除state属性
    delattr(mock.app, 'state')
    return mock


@pytest.fixture
def mock_request_wrong_state_type() -> MagicMock:
    """创建一个 state 属性类型不是 State 的模拟 Request 对象。"""
    mock = MagicMock(spec=Request)
    mock.app = MagicMock()
    mock.app.state = {}  # 使用字典而不是 State 对象
    return mock


@pytest.fixture
def mock_state() -> State:
    """创建一个模拟的 State 对象，用于依赖测试。"""
    state = State()
    # 设置一些默认值
    state.pg_pool = MagicMock(spec=AsyncConnectionPool)
    state.neo4j_driver = MagicMock(spec=AsyncDriver)
    state.embedder = MagicMock(spec=TextEmbedder)
    state.embedder.model = True  # 确保 embedder.model 存在
    state.faiss_repo_papers = MagicMock(spec=FaissRepository)
    state.faiss_repo_papers.is_ready.return_value = True
    state.faiss_repo_models = MagicMock(spec=FaissRepository)
    state.faiss_repo_models.is_ready.return_value = True
    return state


# --- 测试 get_app_state ---

def test_get_app_state_success(mock_request: MagicMock) -> None:
    """测试 get_app_state 在 state 存在时的成功情况。"""
    # 准备
    expected_state = mock_request.app.state
    
    # 执行
    result = get_app_state(mock_request)
    
    # 验证
    assert result is expected_state


def test_get_app_state_no_state_attribute(mock_request_no_state: MagicMock) -> None:
    """测试 get_app_state 在 state 不存在时应抛出 HTTPException。"""
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_app_state(mock_request_no_state)
    
    assert excinfo.value.status_code == 500
    assert "Application state not initialized" in excinfo.value.detail


def test_get_app_state_wrong_state_type(mock_request_wrong_state_type: MagicMock) -> None:
    """测试 get_app_state 在 state 类型不是 State 时的处理。"""
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_app_state(mock_request_wrong_state_type)
    
    # 验证
    assert result is mock_request_wrong_state_type.app.state
    mock_logger.warning.assert_called_once()
    assert "is not of type State" in mock_logger.warning.call_args[0][0]


# --- 测试 get_postgres_pool ---

def test_get_postgres_pool_success() -> None:
    """测试 get_postgres_pool 在 pool 存在时的成功情况。"""
    # 准备
    mock_state = MagicMock(spec=State)
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_state.pg_pool = mock_pool
    
    # 执行
    result = get_postgres_pool(state=mock_state)
    
    # 验证
    assert result is mock_pool


def test_get_postgres_pool_not_available() -> None:
    """测试 get_postgres_pool 在 pool 不存在时应抛出 HTTPException。"""
    # 准备
    mock_state = MagicMock(spec=State)
    # 没有设置 pg_pool 属性
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_postgres_pool(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Database connection pool is not available" in excinfo.value.detail


# --- 测试 get_neo4j_driver ---

def test_get_neo4j_driver_success() -> None:
    """测试 get_neo4j_driver 在 driver 存在时的成功情况。"""
    # 准备
    mock_state = MagicMock(spec=State)
    mock_driver = MagicMock(spec=AsyncDriver)
    mock_state.neo4j_driver = mock_driver
    
    # 执行
    result = get_neo4j_driver(state=mock_state)
    
    # 验证
    assert result is mock_driver


def test_get_neo4j_driver_not_available() -> None:
    """测试 get_neo4j_driver 在 driver 不存在时的处理（返回 None 而不是抛出异常）。"""
    # 准备
    mock_state = MagicMock(spec=State)
    # 没有设置 neo4j_driver 属性
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_neo4j_driver(state=mock_state)
    
    # 验证
    assert result is None
    mock_logger.warning.assert_called_once()
    assert "Neo4j driver not initialized" in mock_logger.warning.call_args[0][0]


# --- 测试 get_embedder ---

def test_get_embedder_success() -> None:
    """测试 get_embedder 在 embedder 存在且加载了模型时的成功情况。"""
    # 准备
    mock_state = MagicMock(spec=State)
    mock_embedder = MagicMock(spec=TextEmbedder)
    mock_embedder.model = True  # 确保 model 属性存在
    mock_state.embedder = mock_embedder
    
    # 执行
    result = get_embedder(state=mock_state)
    
    # 验证
    assert result is mock_embedder


def test_get_embedder_not_available() -> None:
    """测试 get_embedder 在 embedder 不存在时应抛出 HTTPException。"""
    # 准备
    mock_state = MagicMock(spec=State)
    # 没有设置 embedder 属性
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_embedder(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Text embedding service is not available" in excinfo.value.detail


def test_get_embedder_no_model() -> None:
    """测试 get_embedder 在 embedder 存在但没有加载模型时应抛出 HTTPException。"""
    # 准备
    mock_state = MagicMock(spec=State)
    mock_embedder = MagicMock(spec=TextEmbedder)
    mock_embedder.model = None  # 模型未加载
    mock_state.embedder = mock_embedder
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_embedder(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Text embedding service is not available" in excinfo.value.detail


# --- 测试 get_postgres_repository ---

def test_get_postgres_repository() -> None:
    """测试 get_postgres_repository 函数。"""
    # 准备
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    
    # 执行
    result = get_postgres_repository(pool=mock_pool)
    
    # 验证
    assert isinstance(result, PostgresRepository)
    assert result.pool is mock_pool


# --- 测试 get_neo4j_repository ---

def test_get_neo4j_repository_with_driver() -> None:
    """测试 get_neo4j_repository 在 driver 存在时的情况。"""
    # 准备
    mock_driver = MagicMock(spec=AsyncDriver)
    
    # 执行
    result = get_neo4j_repository(driver=mock_driver)
    
    # 验证
    assert isinstance(result, Neo4jRepository)
    assert result.driver is mock_driver


def test_get_neo4j_repository_without_driver() -> None:
    """测试 get_neo4j_repository 在 driver 不存在时的情况（返回 None）。"""
    # 执行
    result = get_neo4j_repository(driver=None)
    
    # 验证
    assert result is None


# --- 测试 get_faiss_repository_papers ---

def test_get_faiss_repository_papers_success(mock_state: State) -> None:
    """测试 get_faiss_repository_papers 在存储库存在且准备就绪时的成功情况。"""
    # 执行
    result = get_faiss_repository_papers(state=mock_state)
    
    # 验证
    assert result is mock_state.faiss_repo_papers
    mock_state.faiss_repo_papers.is_ready.assert_called_once()


def test_get_faiss_repository_papers_not_available(mock_state: State) -> None:
    """测试 get_faiss_repository_papers 在存储库不存在时应抛出 HTTPException。"""
    # 准备
    mock_state.faiss_repo_papers = None
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_papers(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Papers vector search service is not available" in excinfo.value.detail


def test_get_faiss_repository_papers_not_ready(mock_state: State) -> None:
    """测试 get_faiss_repository_papers 在存储库存在但未准备就绪时应抛出 HTTPException。"""
    # 准备
    mock_state.faiss_repo_papers.is_ready.return_value = False
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_papers(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Papers vector search service is not available" in excinfo.value.detail


# --- 测试 get_faiss_repository_models ---

def test_get_faiss_repository_models_success(mock_state: State) -> None:
    """测试 get_faiss_repository_models 在存储库存在且准备就绪时的成功情况。"""
    # 执行
    result = get_faiss_repository_models(state=mock_state)
    
    # 验证
    assert result is mock_state.faiss_repo_models
    mock_state.faiss_repo_models.is_ready.assert_called_once()


def test_get_faiss_repository_models_not_available(mock_state: State) -> None:
    """测试 get_faiss_repository_models 在存储库不存在时应抛出 HTTPException。"""
    # 准备
    mock_state.faiss_repo_models = None
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_models(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Models vector search service is not available" in excinfo.value.detail


def test_get_faiss_repository_models_not_ready(mock_state: State) -> None:
    """测试 get_faiss_repository_models 在存储库存在但未准备就绪时应抛出 HTTPException。"""
    # 准备
    mock_state.faiss_repo_models.is_ready.return_value = False
    
    # 执行与验证
    with pytest.raises(HTTPException) as excinfo:
        get_faiss_repository_models(state=mock_state)
    
    assert excinfo.value.status_code == 503
    assert "Models vector search service is not available" in excinfo.value.detail


# --- 测试 get_search_service ---

@pytest_asyncio.fixture
async def mock_request_with_state() -> MagicMock:
    """创建一个带有完整状态的模拟 Request 对象。"""
    mock = MagicMock(spec=Request)
    mock.app = MagicMock()
    state = State()
    state.embedder = MagicMock(spec=TextEmbedder)
    state.embedder.model = True
    mock.app.state = state
    return mock


def test_get_search_service_success(
    mock_request_with_state: MagicMock,
    mock_state: State
) -> None:
    """测试 get_search_service 在所有依赖都可用时的成功情况。"""
    # 准备
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.get_app_state", return_value=mock_state):
        result = get_search_service(
            request=mock_request_with_state,
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo
        )
    
    # 验证
    assert isinstance(result, SearchService)
    assert result.embedder is mock_state.embedder
    assert result.faiss_repo_papers is mock_faiss_papers
    assert result.faiss_repo_models is mock_faiss_models
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is mock_neo4j_repo


def test_get_search_service_no_embedder(
    mock_request_with_state: MagicMock
) -> None:
    """测试 get_search_service 在 embedder 不可用时的处理。"""
    # 准备
    mock_request_with_state.app.state.embedder = None
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_search_service(
            request=mock_request_with_state,
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo
        )
    
    # 验证
    assert isinstance(result, SearchService)
    assert result.embedder is None
    assert result.faiss_repo_papers is mock_faiss_papers
    assert result.faiss_repo_models is mock_faiss_models
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is mock_neo4j_repo


def test_get_search_service_model_not_loaded(
    mock_request_with_state: MagicMock
) -> None:
    """测试 get_search_service 在 embedder 存在但模型未加载时的处理。"""
    # 准备
    mock_request_with_state.app.state.embedder.model = None
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_search_service(
            request=mock_request_with_state,
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo
        )
    
    # 验证
    assert isinstance(result, SearchService)
    assert result.embedder is None  # embedder 应该被设置为 None
    mock_logger.warning.assert_called_once()
    assert "Embedder found in state but model not loaded" in mock_logger.warning.call_args[0][0]


def test_get_search_service_get_app_state_exception(
    mock_request_with_state: MagicMock
) -> None:
    """测试 get_search_service 在调用 get_app_state 抛出异常时的处理。"""
    # 准备
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    http_exception = HTTPException(status_code=500, detail="Test exception")
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.get_app_state", 
               side_effect=http_exception) as mock_get_state:
        with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
            result = get_search_service(
                request=mock_request_with_state,
                faiss_repo_papers=mock_faiss_papers,
                faiss_repo_models=mock_faiss_models,
                pg_repo=mock_pg_repo,
                neo4j_repo=mock_neo4j_repo
            )
    
    # 验证
    assert isinstance(result, SearchService)
    assert result.embedder is None
    mock_logger.error.assert_called_once()
    assert "Could not get app state" in mock_logger.error.call_args[0][0]


def test_get_search_service_general_exception(
    mock_request_with_state: MagicMock
) -> None:
    """测试 get_search_service 在检查 embedder 时发生一般异常的处理。"""
    # 准备
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    
    # 创建一个 mock 请求，其 app.state.embedder 配置为在访问 model 属性时引发异常
    mock_request = MagicMock(spec=Request)
    mock_app = MagicMock()
    mock_state = State()
    
    # 创建一个特殊的 embedder 类，其 model 属性在访问时抛出异常
    class ExceptionEmbedder:
        @property
        def model(self) -> Any:
            raise RuntimeError("Test error")
    
    # 设置 mock 对象关系
    mock_state.embedder = ExceptionEmbedder()
    mock_app.state = mock_state
    mock_request.app = mock_app
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_search_service(
            request=mock_request,
            faiss_repo_papers=mock_faiss_papers,
            faiss_repo_models=mock_faiss_models,
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo
        )
    
    # 验证
    assert isinstance(result, SearchService)
    mock_logger.error.assert_called_once()
    assert "Error checking for embedder" in mock_logger.error.call_args[0][0]


def test_get_search_service_no_neo4j_repo(
    mock_request_with_state: MagicMock,
    mock_state: State
) -> None:
    """测试 get_search_service 在 Neo4j 仓库不可用时的处理。"""
    # 准备
    mock_faiss_papers = MagicMock(spec=FaissRepository)
    mock_faiss_models = MagicMock(spec=FaissRepository)
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.get_app_state", return_value=mock_state):
        with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
            result = get_search_service(
                request=mock_request_with_state,
                faiss_repo_papers=mock_faiss_papers,
                faiss_repo_models=mock_faiss_models,
                pg_repo=mock_pg_repo,
                neo4j_repo=None
            )
    
    # 验证
    assert isinstance(result, SearchService)
    assert result.neo4j_repo is None
    mock_logger.warning.assert_called_once()
    assert "Neo4j repository is None" in mock_logger.warning.call_args[0][0]


# --- 测试 get_graph_service ---

def test_get_graph_service_with_neo4j() -> None:
    """测试 get_graph_service 在 Neo4j 仓库可用时的情况。"""
    # 准备
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    mock_neo4j_repo = MagicMock(spec=Neo4jRepository)
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_graph_service(
            pg_repo=mock_pg_repo,
            neo4j_repo=mock_neo4j_repo
        )
    
    # 验证
    assert isinstance(result, GraphService)
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is mock_neo4j_repo
    # 不应有警告日志
    mock_logger.warning.assert_not_called()


def test_get_graph_service_without_neo4j() -> None:
    """测试 get_graph_service 在 Neo4j 仓库不可用时的处理。"""
    # 准备
    mock_pg_repo = MagicMock(spec=PostgresRepository)
    
    # 执行
    with patch("aigraphx.api.v1.dependencies.logger") as mock_logger:
        result = get_graph_service(
            pg_repo=mock_pg_repo,
            neo4j_repo=None
        )
    
    # 验证
    assert isinstance(result, GraphService)
    assert result.pg_repo is mock_pg_repo
    assert result.neo4j_repo is None
    # 应有警告日志
    mock_logger.warning.assert_called_once()
    assert "Neo4j repository is None" in mock_logger.warning.call_args[0][0] 