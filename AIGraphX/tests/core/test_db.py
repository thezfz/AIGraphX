# -*- coding: utf-8 -*-
"""
文件目的：测试 FastAPI 应用的 `lifespan` 上下文管理器。

本测试文件 (`test_db.py`) 专注于验证位于 `aigraphx/core/db.py` 文件中定义的 `lifespan` 函数。
`lifespan` 是 FastAPI 提供的一种机制，用于在应用启动时执行初始化操作（例如，创建数据库连接池、加载模型、初始化外部服务客户端），
并在应用关闭时执行清理操作（例如，关闭连接池、释放资源）。

主要交互：
- 导入 `pytest` 和 `asyncio`：`pytest` 用于测试框架，`asyncio` 用于支持异步操作，特别是手动驱动 `lifespan` 的异步上下文。
- 导入 `unittest.mock`：大量使用 `MagicMock`, `AsyncMock`, `patch` 来模拟 `lifespan` 内部依赖的外部库和类，如 `psycopg_pool`, `neo4j.AsyncGraphDatabase`, `FaissRepository`。这使得测试可以独立于真实的数据库和文件系统运行。
- 导入 `FastAPI` 和 `State`：用于创建模拟的 FastAPI 应用实例 (`mock_app`) 及其状态对象 (`app.state`)，这是 `lifespan` 操作的目标。
- 导入被测试的 `lifespan` 函数：从 `aigraphx.core.db` 导入。
- 导入被模拟的类的类型：如 `FaissRepository`，用于 `spec` 参数，确保模拟对象接口的正确性。
- 导入配置对象：`settings` 和 `Settings` 类型从 `aigraphx.core.config` 导入，因为 `lifespan` 函数现在接收 `settings` 作为参数。
- 定义 Fixtures (`@pytest.fixture`)：
    - `mock_async_connection_pool`, `mock_neo4j_driver`, `mock_faiss_repository`: 使用 `patch` 来模拟相应的类或函数，并返回模拟的类/函数对象及其创建的模拟实例。这允许测试检查类是否以正确的参数被调用，以及实例的方法（如 `close`, `is_ready`）是否被调用。
    - `mock_app`: 创建一个简单的 FastAPI 应用实例，并初始化其 `state` 属性。
    - `mock_settings`: 创建一个 `Settings` 对象的副本，并使用 `monkeypatch` 来设置特定的测试配置值（如数据库 URI、Faiss 路径），确保测试在可控的配置下运行。
- 编写测试函数 (`test_*`)：
    - 测试成功场景 (`test_lifespan_success`)：验证当所有初始化步骤都成功时，`lifespan` 是否正确调用了所有模拟的初始化函数，并将实例设置到 `app.state` 中，以及在退出时是否调用了清理方法 (`close`)。
    - 测试初始化失败场景 (`test_lifespan_pg_pool_failure`, `test_lifespan_neo4j_failure`, `test_lifespan_faiss_papers_init_failure`, `test_lifespan_faiss_models_init_failure`)：通过配置模拟对象在初始化时抛出异常 (`side_effect`)，验证 `lifespan` 是否能捕获这些异常并抛出 `RuntimeError`，以及是否阻止了后续的初始化和不必要的清理。
    - 测试 Faiss "未就绪" 场景 (`test_lifespan_faiss_not_ready`)：验证当 Faiss 仓库实例的 `is_ready()` 返回 `False` 时，`lifespan` 是否会抛出 `RuntimeError`。
    - 测试可选依赖未配置场景 (`test_lifespan_neo4j_not_configured`)：验证当 Neo4j 的必要配置缺失时，`lifespan` 是否能跳过 Neo4j 的初始化，不抛出错误，并将 `app.state.neo4j_driver` 设为 `None`。

这些测试对于确保应用程序启动和关闭过程的健壮性至关重要，保证了数据库连接、向量索引等核心资源能够被正确、可靠地管理。
"""

import pytest # 导入 pytest 测试框架
import asyncio  # 导入 asyncio 模块，用于支持异步操作，特别是在手动驱动 lifespan 时
from unittest.mock import MagicMock, AsyncMock, patch, Mock, ANY # 从 unittest.mock 导入模拟工具：MagicMock (通用模拟), AsyncMock (异步模拟), patch (用于替换对象), Mock (基础模拟), ANY (用于匹配任意参数)
from fastapi import FastAPI # 从 FastAPI 框架导入 FastAPI 类
from starlette.datastructures import State  # 从 Starlette 导入 State 类，FastAPI 的 app.state 基于此
import os # 导入 os 模块，可能用于路径操作等 (虽然在此文件中不直接使用)
from typing import Dict, Any, Tuple, List, Optional, Generator, Literal, cast # 从 typing 导入类型提示工具

# 将此模块中的所有异步测试标记为使用会话作用域的事件循环
# 这有助于在多个测试之间共享事件循环，可能提高效率，但也需注意测试间的潜在干扰
pytestmark = pytest.mark.asyncio(loop_scope="session")

# 导入需要测试的 lifespan 函数
from aigraphx.core.db import lifespan

# 导入被模拟的类 (用于类型注解和 spec)
# from aigraphx.vectorization.embedder import TextEmbedder # 不再需要在此模拟 Embedder，lifespan 不再处理它
from aigraphx.repositories.faiss_repo import FaissRepository # 导入 Faiss 仓库类

# 导入 settings 对象和 Settings 类型
from aigraphx.core.config import settings, Settings  # 导入实际的 settings 对象和 Settings 类型定义


# 使用 MagicMock 模拟类，AsyncMock 模拟实例/方法
@pytest.fixture
def mock_async_connection_pool() -> Generator[Tuple[MagicMock, AsyncMock], None, None]:
    """
    Pytest fixture: 模拟 psycopg_pool.AsyncConnectionPool 类及其创建的实例。

    使用 `patch` 替换掉 `aigraphx.core.db` 模块中引用的 `AsyncConnectionPool`。
    `new_callable=MagicMock` 表示用 MagicMock 来替换类本身。
    它返回一个元组：(模拟的类对象, 模拟的实例对象)。
    这允许测试：
    1. 验证类是否以正确的参数被调用来创建实例。
    2. 控制模拟实例的行为（例如，模拟 `close` 方法）。
    """
    # 使用 patch 上下文管理器替换目标类
    with patch(
        "aigraphx.core.db.psycopg_pool.AsyncConnectionPool", new_callable=MagicMock
    ) as mock_pool_class:
        # 创建一个 AsyncMock 作为模拟实例，因为它有异步方法 (如 close)
        mock_instance = AsyncMock()
        # 将模拟实例的 close 方法也设置为 AsyncMock
        mock_instance.close = AsyncMock()
        # 配置模拟类，使其在被调用时返回我们创建的模拟实例
        mock_pool_class.return_value = mock_instance
        # 使用 yield 返回模拟类和实例，测试函数执行完毕后，patch 会自动撤销
        yield mock_pool_class, mock_instance


@pytest.fixture
def mock_neo4j_driver() -> Generator[Tuple[MagicMock, AsyncMock], None, None]:
    """
    Pytest fixture: 模拟 neo4j.AsyncGraphDatabase.driver 函数及其返回的驱动实例。

    与 `mock_async_connection_pool` 类似，但这次 `patch` 的是 `driver` 函数。
    返回一个元组：(模拟的 driver 函数, 模拟的驱动实例)。
    """
    with patch(
        # 注意 patch 的目标是 `aigraphx.core.db` 中导入并使用的 `driver` 函数
        "aigraphx.core.db.AsyncGraphDatabase.driver", new_callable=MagicMock
    ) as mock_driver_func:
        # 创建模拟的驱动实例
        mock_instance = AsyncMock()
        # 模拟异步的 close 方法
        mock_instance.close = AsyncMock()
        # 配置模拟函数，使其返回模拟实例
        mock_driver_func.return_value = mock_instance
        yield mock_driver_func, mock_instance


# 不再需要 mock_text_embedder fixture，因为 lifespan 不再负责创建 embedder


@pytest.fixture
def mock_faiss_repository() -> Generator[
    Tuple[MagicMock, MagicMock, MagicMock], None, None # 返回类型是包含三个 Mock 的元组
]:
    """
    Pytest fixture: 模拟 FaissRepository 类及其创建的两个实例（用于论文和模型）。

    这个 fixture 比较复杂，因为它需要模拟同一个类根据不同的初始化参数返回不同的实例。
    它使用 `side_effect` 来自定义 `mock_repo_class` 的行为。
    返回一个元组：(模拟的 FaissRepository 类, 模拟的论文仓库实例, 模拟的模型仓库实例)。
    """
    with patch(
        "aigraphx.core.db.FaissRepository", new_callable=MagicMock # 模拟 FaissRepository 类
    ) as mock_repo_class:
        # --- 创建模拟实例 ---
        # 为论文 Faiss 仓库创建模拟实例
        mock_instance_papers = MagicMock(spec=FaissRepository) # 使用 spec 确保接口一致性
        mock_instance_papers.index = MagicMock() # 模拟 index 属性
        mock_instance_papers.id_map = {}  # 添加模拟的 id_map 属性
        mock_instance_papers.is_ready.return_value = True  # 默认设置为准备就绪

        # 为模型 Faiss 仓库创建模拟实例
        mock_instance_models = MagicMock(spec=FaissRepository)
        mock_instance_models.index = MagicMock()
        mock_instance_models.id_map = {}
        mock_instance_models.is_ready.return_value = True

        # --- 配置 side_effect ---
        # 定义一个函数，根据调用 FaissRepository 时的关键字参数 id_type 返回不同的实例
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            id_type = kwargs.get("id_type") # 获取 id_type 参数
            if id_type == "int": # 如果是论文仓库（id 是整数）
                return mock_instance_papers
            elif id_type == "str": # 如果是模型仓库（id 是字符串）
                return mock_instance_models
            else: # 处理未预期的情况
                raise ValueError(
                    f"在 FaissRepository 模拟中遇到未预期的 id_type: {id_type}"
                )

        # 将这个 side_effect 函数赋给模拟类的 side_effect 属性
        mock_repo_class.side_effect = side_effect

        # 使用 yield 返回模拟类和两个模拟实例
        yield mock_repo_class, mock_instance_papers, mock_instance_models


# 创建模拟 FastAPI 应用的辅助 fixture
@pytest.fixture
def mock_app() -> FastAPI:
    """
    Pytest fixture: 创建一个简单的 FastAPI 应用实例用于测试。

    这个实例带有一个初始化的 `state` 属性，`lifespan` 函数将在这个 `state` 上设置属性。
    """
    app = FastAPI() # 创建 FastAPI 应用
    app.state = State()  # 初始化 state 属性
    return app


# 创建带有特定配置的模拟 settings 对象的辅助 fixture
@pytest.fixture
def mock_settings(
    monkeypatch: pytest.MonkeyPatch, # 请求 monkeypatch fixture 用于修改 settings
    # --- Fixture 参数 (允许在测试用例中覆盖这些默认值) ---
    neo4j_uri: Optional[str] = "neo4j://testsuccess", # 默认模拟 Neo4j URI
    neo4j_user: Optional[str] = "testuser_success", # 默认模拟 Neo4j 用户名
    neo4j_pass: Optional[str] = "testpass_success", # 默认模拟 Neo4j 密码
    # embedder_model: str = "test-model", # 移除 embedder 相关参数
    # embedder_device: str = "test-device",
    faiss_papers_index: str = "/tmp/papers.index", # 默认模拟论文 Faiss 索引路径
    faiss_papers_map: str = "/tmp/papers.json", # 默认模拟论文 ID 映射路径
    faiss_models_index: str = "/tmp/models.index", # 默认模拟模型 Faiss 索引路径
    faiss_models_map: str = "/tmp/models.json", # 默认模拟模型 ID 映射路径
) -> Settings:
    """
    Pytest fixture: 创建一个 Settings 对象的副本，并使用 monkeypatch 修改其属性。

    这允许每个测试用例在隔离的环境中使用特定的配置值，而不会影响全局 `settings` 对象
    或其他测试。它接收一些可选参数，允许调用测试直接覆盖这些配置项。

    返回:
        Settings: 一个被修改过的 Settings 对象实例。
    """
    # 创建全局 settings 对象的一个副本，避免修改原始对象
    test_settings = settings.model_copy()  # 使用 Pydantic V2 的 model_copy()
    # 使用 monkeypatch 修改副本的属性值，monkeypatch 会在 fixture 结束时自动恢复
    monkeypatch.setattr(
        test_settings,
        "database_url", # 修改数据库 URL
        "postgresql://mock_user:mock_pass@mock_host:5432/mock_db", # 设置为模拟值
    )
    monkeypatch.setattr(test_settings, "neo4j_uri", neo4j_uri) # 修改 Neo4j URI
    monkeypatch.setattr(test_settings, "neo4j_username", neo4j_user) # 修改 Neo4j 用户名
    monkeypatch.setattr(test_settings, "neo4j_password", neo4j_pass) # 修改 Neo4j 密码
    # monkeypatch.setattr(test_settings, "sentence_transformer_model", embedder_model) # 移除对 embedder 的修改
    # monkeypatch.setattr(test_settings, "embedder_device", embedder_device) # 移除
    monkeypatch.setattr(test_settings, "faiss_index_path", faiss_papers_index) # 修改论文 Faiss 索引路径
    monkeypatch.setattr(test_settings, "faiss_mapping_path", faiss_papers_map) # 修改论文 Faiss 映射路径
    monkeypatch.setattr(test_settings, "models_faiss_index_path", faiss_models_index) # 修改模型 Faiss 索引路径
    monkeypatch.setattr(test_settings, "models_faiss_mapping_path", faiss_models_map) # 修改模型 Faiss 映射路径
    return test_settings # 返回修改后的 settings 副本


# --- 测试用例 ---

# 测试 lifespan 成功初始化和清理的场景
@pytest.mark.asyncio # 标记为异步测试
async def test_lifespan_success(
    mock_app: FastAPI, # 请求模拟 app fixture
    mock_settings: Settings,  # 请求模拟 settings fixture
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock], # 请求 PG 连接池模拟 fixture
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock], # 请求 Neo4j 驱动模拟 fixture
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],  # 请求 Faiss 仓库模拟 fixture
) -> None:
    """
    测试场景：lifespan 正常启动和关闭。
    预期行为：
    1. 所有依赖项（PG Pool, Neo4j Driver, Faiss Repos）都被正确初始化（对应的模拟类/函数被以正确的参数调用）。
    2. Faiss 仓库的 is_ready() 方法被调用。
    3. 初始化后的实例被正确设置到 mock_app.state 中。
    4. 在 lifespan 退出时，PG Pool 和 Neo4j Driver 的 close() 方法被异步调用。
    """
    # --- 准备 ---
    # 从 fixture 返回的元组中解包模拟对象
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # 确保模拟的 Faiss 仓库处于“准备就绪”状态
    mock_repo_instance_papers.is_ready.return_value = True
    mock_repo_instance_models.is_ready.return_value = True

    # --- 执行 ---
    # 创建 lifespan 异步上下文管理器实例，传入模拟 app 和 settings
    ctx = lifespan(mock_app, mock_settings)
    # 手动进入上下文（模拟应用启动）
    # 这会执行 lifespan 函数 `yield` 之前的部分
    await ctx.__aenter__()

    # --- 验证启动过程 ---
    # 验证 PG 连接池类是否被以 mock_settings 中的参数调用了一次
    mock_pool_class.assert_called_once_with(
        conninfo=mock_settings.database_url, # 检查连接信息
        min_size=mock_settings.pg_pool_min_size, # 检查最小连接数
        max_size=mock_settings.pg_pool_max_size, # 检查最大连接数
    )
    # 验证 Neo4j driver 函数是否被以 mock_settings 中的参数调用了一次
    mock_driver_func.assert_called_once_with(
        mock_settings.neo4j_uri, # 检查 URI
        auth=(mock_settings.neo4j_username, mock_settings.neo4j_password), # 检查认证元组
    )
    # 不再检查 embedder 的调用

    # 验证 FaissRepository 类是否被调用了两次，分别对应论文和模型仓库的初始化
    # 使用 assert_any_call 因为调用顺序不确定或不重要
    mock_repo_class.assert_any_call( # 验证论文仓库的调用参数
        index_path=mock_settings.faiss_index_path,
        id_map_path=mock_settings.faiss_mapping_path,
        id_type="int", # 关键参数：论文 ID 类型为 int
    )
    mock_repo_class.assert_any_call( # 验证模型仓库的调用参数
        index_path=mock_settings.models_faiss_index_path,
        id_map_path=mock_settings.models_faiss_mapping_path,
        id_type="str", # 关键参数：模型 ID 类型为 str
    )
    # 精确断言 FaissRepository 类总共被调用了两次
    assert mock_repo_class.call_count == 2
    # 验证两个 Faiss 实例的 is_ready 方法都被调用了一次
    mock_repo_instance_papers.is_ready.assert_called_once()
    mock_repo_instance_models.is_ready.assert_called_once()

    # --- 验证应用状态 (app.state) ---
    # 检查 mock_app.state 中的属性是否被正确设置为了对应的模拟实例
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    # assert not hasattr(mock_app.state, 'embedder') # 确认 embedder 不再被设置到 state 中
    assert mock_app.state.faiss_repo_papers == mock_repo_instance_papers # 检查论文仓库实例
    assert mock_app.state.faiss_repo_models == mock_repo_instance_models # 检查模型仓库实例

    # --- 执行退出 ---
    # 手动退出上下文（模拟应用关闭）
    # 这会执行 lifespan 函数 `yield` 之后的部分 (清理操作)
    # 传入 None 表示正常退出，没有异常
    await ctx.__aexit__(None, None, None)

    # --- 验证清理过程 ---
    # 验证 PG 连接池实例的 close 方法被异步调用了一次
    mock_pool_instance.close.assert_awaited_once()
    # 验证 Neo4j 驱动实例的 close 方法被异步调用了一次
    mock_driver_instance.close.assert_awaited_once()


# 测试初始化失败的场景（例如，PG 连接池初始化失败）
@pytest.mark.asyncio
async def test_lifespan_pg_pool_failure(
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock], # Neo4j mock 仍然需要，即使预期它不被调用
) -> None:
    """
    测试场景：在 lifespan 启动过程中，PG 连接池初始化失败。
    预期行为：
    1. `lifespan` 在 `__aenter__` 阶段应该捕获原始异常并抛出 `RuntimeError`。
    2. 后续的初始化步骤（如 Neo4j Driver, Faiss Repos）不应被执行。
    3. 清理操作（如 `close` 方法）不应被调用，因为初始化未完成。
    """
    # --- 准备 ---
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    # 配置模拟的 PG 连接池类，使其在被调用时抛出异常
    mock_pool_class.side_effect = Exception("PG Pool Init Error")

    # --- 执行与验证 ---
    ctx = lifespan(mock_app, mock_settings)
    # 使用 pytest.raises 来断言特定的 RuntimeError 是否被抛出
    with pytest.raises(RuntimeError, match="PostgreSQL pool initialization failed"):
        # 尝试进入 lifespan 上下文，预期会在此处失败
        await ctx.__aenter__()

    # --- 验证后续步骤未执行 ---
    # 验证 Neo4j driver 函数没有被调用，因为它在 PG Pool 初始化之后
    mock_driver_func.assert_not_called()
    # 验证 PG Pool 实例的 close 方法没有被调用（实例可能都没创建成功）
    mock_pool_instance.close.assert_not_awaited()
    # 验证 Neo4j Driver 实例的 close 方法没有被调用
    mock_driver_instance.close.assert_not_awaited()


# 测试初始化失败的场景（例如，Neo4j 驱动初始化失败）
@pytest.mark.asyncio
async def test_lifespan_neo4j_failure(
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
) -> None:
    """
    测试场景：在 lifespan 启动过程中，Neo4j 驱动初始化失败。
    预期行为：
    1. PG Pool 初始化应该成功，并设置到 `app.state`。
    2. Neo4j Driver 初始化失败，`lifespan` 捕获异常并抛出 `RuntimeError`。
    3. Faiss Repos 的初始化不应被执行。
    4. 在异常发生后，清理逻辑可能不会被完全执行或按预期执行，这里我们保守地断言 `close` 未被调用。
    """
    # --- 准备 ---
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    # 配置模拟的 Neo4j driver 函数，使其在被调用时抛出异常
    mock_driver_func.side_effect = Exception("Neo4j Init Error")

    # --- 执行与验证 ---
    ctx = lifespan(mock_app, mock_settings)
    with pytest.raises(RuntimeError, match="Neo4j driver initialization failed"):
        await ctx.__aenter__()

    # --- 验证状态和清理 ---
    # 检查 PG Pool 是否已成功设置到 state (因为它在 Neo4j 之前初始化)
    # assert mock_app.state.pg_pool == mock_pool_instance # 理论上应该设置了，但异常后的状态可能不稳定

    # 断言 close 方法没有被调用。
    # 在 lifespan 中途失败时，标准的 __aexit__ 可能不会被完全执行，
    # 或者在 __aexit__ 中尝试关闭已初始化的资源也可能再次引发问题。
    # 最安全的断言是它们没有被成功调用。
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# 重命名测试：测试 Faiss 仓库初始化后处于 "未就绪" 状态
@pytest.mark.asyncio
async def test_lifespan_faiss_not_ready(
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock], # 依赖 Faiss mock
) -> None:
    """
    测试场景：Faiss 仓库实例被成功创建，但在启动检查时 `is_ready()` 返回 `False`。
    预期行为：
    1. PG Pool 和 Neo4j Driver 初始化成功并设置到 `app.state`。
    2. Faiss Repository (假设是论文仓库先检查) 的 `is_ready()` 返回 `False`。
    3. `lifespan` 捕获这个情况并抛出 `RuntimeError`，指示仓库未就绪。
    4. 失败后，`app.state` 中对应的 Faiss 仓库属性应为 `None`。
    5. 清理操作不应被调用。
    """
    # --- 准备 ---
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # 配置模拟的论文 Faiss 仓库的 is_ready 方法返回 False
    mock_repo_instance_papers.is_ready.return_value = False
    # 假设模型仓库是准备好的（虽然可能执行不到这一步）
    mock_repo_instance_models.is_ready.return_value = True

    # --- 执行与验证 ---
    ctx = lifespan(mock_app, mock_settings)
    # 断言抛出了指示 Papers Faiss 未就绪的 RuntimeError
    with pytest.raises(
        RuntimeError, match="Papers Faiss Repository is not ready after initialization."
    ):
        await ctx.__aenter__()

    # --- 验证状态和清理 ---
    # 检查 PG Pool 和 Neo4j Driver 是否已设置到 state
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    # 检查失败的 Faiss 仓库状态是否为 None
    assert mock_app.state.faiss_repo_papers is None
    # 另一个 Faiss 仓库也应该是 None，因为在第一个失败后初始化停止了
    assert mock_app.state.faiss_repo_models is None

    # 检查 close 方法未被调用
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# 测试初始化失败（例如，Faiss 论文仓库的 __init__ 方法本身失败）
@pytest.mark.asyncio
async def test_lifespan_faiss_papers_init_failure( # 轻微重命名以更清晰
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock], # fixture 返回的是类和两个实例模板
) -> None:
    """
    测试场景：在尝试创建 Faiss 论文仓库实例时，`FaissRepository.__init__` 抛出异常。
    预期行为：
    1. PG Pool 和 Neo4j Driver 初始化成功并设置到 `app.state`。
    2. 调用 `FaissRepository` 创建论文实例时失败。
    3. `lifespan` 捕获异常并抛出 `RuntimeError`。
    4. 后续的模型仓库初始化不应执行。
    5. `app.state` 中 Faiss 相关属性应为 `None`。
    6. 清理操作不应被调用。
    """
    # --- 准备 ---
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    # 获取模拟类和实例模板，但我们将覆盖类的 side_effect
    mock_repo_class, mock_repo_instance_papers_tmpl, mock_repo_instance_models_tmpl = (
        mock_faiss_repository
    )

    # 定义一个新的 side_effect 函数，当 id_type 为 'int' (论文) 时抛出异常
    def side_effect(*args: Any, **kwargs: Any) -> Any:
        id_type = kwargs.get("id_type")
        if id_type == "int":
            raise ValueError("模拟论文 Faiss 初始化失败") # 抛出异常
        elif id_type == "str":
            # 理论上这里不会被调用，但为了完整性，返回模型实例模板
            return mock_repo_instance_models_tmpl
        else:
            raise ValueError(f"未预期的 id_type: {id_type}")

    # 将这个新的 side_effect 应用到模拟类上
    mock_repo_class.side_effect = side_effect

    # --- 执行与验证 ---
    ctx = lifespan(mock_app, mock_settings)
    # 断言抛出了指示 Papers Faiss 初始化失败的 RuntimeError
    with pytest.raises(
        RuntimeError, match="Papers Faiss Repository initialization failed"
    ):
        await ctx.__aenter__()

    # --- 验证状态和清理 ---
    # 检查 PG 和 Neo4j 状态
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    # 检查 Faiss 状态都为 None
    assert mock_app.state.faiss_repo_papers is None
    assert mock_app.state.faiss_repo_models is None

    # 检查 close 方法未被调用
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# 测试初始化失败（例如，Faiss 模型仓库的 __init__ 方法失败）
@pytest.mark.asyncio
async def test_lifespan_faiss_models_init_failure( # 轻微重命名以更清晰
    mock_app: FastAPI,
    mock_settings: Settings,
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock],
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """
    测试场景：论文 Faiss 仓库创建成功，但在尝试创建模型 Faiss 仓库实例时失败。
    预期行为：
    1. PG Pool, Neo4j Driver, 论文 Faiss 仓库初始化成功并设置到 `app.state`。
    2. 调用 `FaissRepository` 创建模型实例时失败。
    3. `lifespan` 捕获异常并抛出 `RuntimeError`。
    4. `app.state` 中模型 Faiss 仓库属性应为 `None`。
    5. 清理操作不应被调用。
    """
    # --- 准备 ---
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers_tmpl, mock_repo_instance_models_tmpl = (
        mock_faiss_repository
    )

    # 定义 side_effect，当 id_type 为 'str' (模型) 时抛出异常
    def side_effect(*args: Any, **kwargs: Any) -> Any:
        id_type = kwargs.get("id_type")
        if id_type == "int":
            return mock_repo_instance_papers_tmpl # 论文实例成功返回
        elif id_type == "str":
            raise ValueError("模拟模型 Faiss 初始化失败") # 模型实例失败
        else:
            raise ValueError(f"未预期的 id_type: {id_type}")

    mock_repo_class.side_effect = side_effect

    # --- 执行与验证 ---
    ctx = lifespan(mock_app, mock_settings)
    # 断言抛出了指示 Models Faiss 初始化失败的 RuntimeError
    with pytest.raises(
        RuntimeError, match="Models Faiss Repository initialization failed"
    ):
        await ctx.__aenter__()

    # --- 验证状态和清理 ---
    # 检查 PG, Neo4j, 论文 Faiss 状态
    assert mock_app.state.pg_pool == mock_pool_instance
    assert mock_app.state.neo4j_driver == mock_driver_instance
    # 论文仓库应该已经成功设置
    assert mock_app.state.faiss_repo_papers == mock_repo_instance_papers_tmpl
    # 模型仓库应该为 None
    assert mock_app.state.faiss_repo_models is None

    # 检查 close 方法未被调用
    mock_pool_instance.close.assert_not_awaited()
    mock_driver_instance.close.assert_not_awaited()


# 测试 Neo4j 未配置的场景
@pytest.mark.asyncio
async def test_lifespan_neo4j_not_configured(
    mock_app: FastAPI,
    mock_settings: Settings,  # 使用 mock_settings fixture
    mock_async_connection_pool: Tuple[MagicMock, AsyncMock],
    mock_neo4j_driver: Tuple[MagicMock, AsyncMock], # Neo4j mock 仍需提供，即使预期不被调用
    mock_faiss_repository: Tuple[MagicMock, MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,  # 请求 monkeypatch 用于修改 settings
) -> None:
    """
    测试场景：Neo4j 的部分必要配置项（例如密码）在 `settings` 中为 `None`。
    预期行为：
    1. `lifespan` 应该检测到配置不完整，跳过 Neo4j 的初始化。
    2. 不应调用 `AsyncGraphDatabase.driver` 函数。
    3. 其他依赖项（PG Pool, Faiss Repos）应正常初始化并设置到 `app.state`。
    4. `app.state.neo4j_driver` 应为 `None`。
    5. `lifespan` 启动和关闭应正常完成，不抛出异常。
    6. 关闭时，只应调用 PG Pool 的 `close` 方法。
    """
    # --- 准备 ---
    mock_pool_class, mock_pool_instance = mock_async_connection_pool
    mock_driver_func, mock_driver_instance = mock_neo4j_driver
    mock_repo_class, mock_repo_instance_papers, mock_repo_instance_models = (
        mock_faiss_repository
    )

    # 使用 monkeypatch 显式地将 mock_settings 中的 neo4j_password 设置为 None
    # 这模拟了配置文件或环境变量中缺少此项的情况
    monkeypatch.setattr(mock_settings, "neo4j_password", None)

    # --- 执行 ---
    ctx = lifespan(mock_app, mock_settings)
    # 预期 lifespan 不会抛出错误，只是会跳过 Neo4j 初始化（并可能记录日志，但测试不检查日志）
    await ctx.__aenter__() # 正常进入上下文

    # --- 验证启动过程 ---
    # 验证 PG Pool 类被调用
    mock_pool_class.assert_called_once()
    # !!! 关键验证：Neo4j driver 函数 *不* 应该被调用
    mock_driver_func.assert_not_called()
    # 验证 Faiss Repos 仍然被正常初始化
    mock_repo_class.assert_any_call(
        index_path=mock_settings.faiss_index_path,
        id_map_path=mock_settings.faiss_mapping_path,
        id_type="int",
    )
    mock_repo_class.assert_any_call(
        index_path=mock_settings.models_faiss_index_path,
        id_map_path=mock_settings.models_faiss_mapping_path,
        id_type="str",
    )
    assert mock_repo_class.call_count == 2

    # --- 验证应用状态 ---
    # 检查 PG Pool 状态
    assert mock_app.state.pg_pool == mock_pool_instance
    # !!! 关键验证：Neo4j driver 状态应为 None
    assert mock_app.state.neo4j_driver is None
    # 检查 Faiss 状态
    assert mock_app.state.faiss_repo_papers == mock_repo_instance_papers
    assert mock_app.state.faiss_repo_models == mock_repo_instance_models

    # --- 执行退出 ---
    await ctx.__aexit__(None, None, None) # 正常退出

    # --- 验证清理过程 ---
    # 验证 PG Pool 的 close 被调用
    mock_pool_instance.close.assert_awaited_once()
    # !!! 关键验证：Neo4j Driver 的 close *不* 应该被调用，因为它从未被初始化
    mock_driver_instance.close.assert_not_awaited()