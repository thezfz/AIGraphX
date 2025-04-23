# -*- coding: utf-8 -*-
"""
文件目的：测试 `scripts/sync_pg_to_faiss.py` 脚本。

本测试文件 (`test_sync_pg_to_faiss.py`) 专注于验证 `sync_pg_to_faiss.py` 脚本的功能，
该脚本负责从 PostgreSQL 数据库读取论文数据，使用文本嵌入模型生成向量，
并将这些向量构建成一个 Faiss 索引文件，同时保存一个 ID 映射文件。

测试策略结合了 **集成测试** 和 **单元测试** 的元素：
- 集成测试 (`test_build_faiss_integration`)：
    - 使用真实的 PostgreSQL 测试数据库（通过 `conftest.py` 的 fixture）。
    - 使用真实的 `TextEmbedder` 来生成向量（需要下载模型并可能访问网络）。
    - 使用临时的 Faiss 索引和 ID 映射文件（通过 `tmp_path` 和 `test_settings_fixture`）。
    - 验证生成的 Faiss 索引和 ID 映射文件的正确性。
- 单元测试 (`test_build_faiss_success_*`, `test_build_faiss_embedding_error`, `test_build_faiss_save_error`)：
    - Mock（模拟）数据库仓库 (`PostgresRepository`) 和文本嵌入器 (`TextEmbedder`)。
    - Mock 文件系统操作（`os.path.exists`, `os.makedirs`, `os.remove`, `builtins.open`）和 `faiss` 库的调用。
    - 专注于测试 `build_index` 函数的核心逻辑，如循环、批处理、错误处理、文件保存调用等，而不依赖实际的数据库、嵌入模型或文件 I/O。

主要交互：
- 导入 `pytest`, `pytest_asyncio`, `pytest-mock`：用于测试框架、异步支持和模拟。
- 导入 `unittest.mock`：用于 `patch`, `AsyncMock`, `mock_open` 等模拟工具。
- 导入 `asyncpg`, `json`, `os`, `builtins`, `numpy`, `faiss`, `pathlib`, `datetime`：脚本和测试中使用的库。
- 导入被测试的脚本函数：`build_index as build_faiss_index`, `main as run_sync_faiss_main`。
- 导入脚本依赖的类：`PostgresRepository`, `TextEmbedder`。
- 导入测试 Fixtures：
    - 真实的 PG 仓库 `postgres_repository_fixture` (来自 `conftest.py`)。
    - 测试配置 `test_settings_fixture` (来自 `conftest.py`)，提供临时文件路径。
    - Mock Fixtures (`mock_faiss_index`, `mock_pg_repo`, `mock_embedder`)：为单元测试提供模拟对象。
- 定义测试数据：`TEST_PAPER_*_FAISS`。
- 定义数据库辅助函数：`insert_faiss_pg_data` 用于在集成测试前准备数据。
- 编写测试函数：覆盖成功构建（带/不带重置）、嵌入错误、保存错误等场景。

这些测试确保 Faiss 索引构建脚本能够正确地从数据库读取数据，调用嵌入模型，处理数据批次，并生成有效的索引和映射文件，同时也能妥善处理可能出现的错误。
"""

import pytest # 导入 pytest 测试框架
import pytest_asyncio # 导入 pytest 的异步扩展
from unittest.mock import patch, AsyncMock, mock_open, call, ANY, MagicMock # 导入模拟工具
import asyncpg  # type: ignore[import-untyped] # 导入 asyncpg，即使被 mock
import json # 导入 json 库
import os # 导入 os 模块
import builtins # 导入 builtins，用于 mock open
import numpy as np # 导入 numpy，用于处理向量
import faiss  # type: ignore[import-untyped] # 导入 faiss 库，用于索引操作 (忽略 mypy 类型检查错误)
from pathlib import Path  # 导入 Path 对象，处理文件路径
from datetime import date # 导入 date 类型
from typing import AsyncGenerator, Optional, List, Tuple, Any, cast, Dict, Callable # 导入类型提示

# 从要测试的脚本中导入特定函数/类
from scripts.sync_pg_to_faiss import (
    build_index as build_faiss_index, # 导入核心的索引构建函数，并重命名
    main as run_sync_faiss_main, # 导入脚本的主入口函数，并重命名
)

# 导入脚本中使用的、我们可能需要 mock 或实例化的类
from aigraphx.repositories.postgres_repo import PostgresRepository # 导入 PG 仓库类
from aigraphx.vectorization.embedder import TextEmbedder # 导入文本嵌入器类
from tests.conftest import (
    repository as postgres_repository_fixture, # 导入真实的 PG 仓库 fixture (来自 conftest.py)
    test_settings as test_settings_fixture,  # 导入测试配置 fixture (来自 conftest.py)
)
from aigraphx.core.config import Settings  # 导入 Settings 类，用于类型提示

# --- 测试数据 ---
# 定义用于插入数据库的样本论文数据
TEST_PAPER_1_FAISS = {
    "pwc_id": "test-faiss-pwc-1", # 论文的 PapersWithCode ID
    "title": "Faiss Paper 1", # 标题
    "summary": "This is the first paper summary for Faiss testing.", # 摘要，将用于生成嵌入向量
    "published_date": date(2023, 6, 1), # 发表日期
    "area": "Vision", # 领域
}
TEST_PAPER_2_FAISS = {
    "pwc_id": "test-faiss-pwc-2",
    "title": "Faiss Paper 2",
    "summary": "Second paper abstract, slightly different content.", # 不同的摘要
    "published_date": date(2023, 6, 10),
    "area": "NLP",
}


# --- 辅助函数：在测试数据库中插入数据 ---
async def insert_faiss_pg_data(repo: PostgresRepository) -> List[int]:
    """
    辅助函数，用于在集成测试开始前向 PostgreSQL 测试数据库中插入样本论文数据。

    Args:
        repo: 已连接到测试数据库的 PostgresRepository 实例。

    Returns:
        List[int]: 插入的论文的 paper_id 列表。
    """
    paper_ids = [] # 用于存储返回的 paper_id
    # 从仓库连接池获取一个连接
    async with repo.pool.connection() as conn:
        # 在连接上创建一个游标
        async with conn.cursor() as cur:
            # 遍历样本数据
            for paper_data in [TEST_PAPER_1_FAISS, TEST_PAPER_2_FAISS]:
                # 执行插入语句，并使用 RETURNING 获取新插入行的 paper_id
                await cur.execute(
                    """
                    INSERT INTO papers (pwc_id, title, summary, published_date, area)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING paper_id
                    """,
                    (
                        paper_data["pwc_id"],
                        paper_data["title"],
                        paper_data["summary"],
                        paper_data["published_date"],
                        paper_data["area"],
                    ),
                )
                # 获取查询结果
                result = await cur.fetchone()
                if result: # 确保 fetchone 返回了结果
                    paper_id = result[0]  # 获取 paper_id (元组的第一个元素)
                    paper_ids.append(paper_id) # 添加到列表中
        # !!! 重要：显式提交事务 !!!
        # 确保在连接关闭前，插入的数据对后续的操作（脚本的读取）可见。
        # 在 conftest.py 的 repo fixture 中，每个测试函数结束后会执行清理 (TRUNCATE)，
        # 但在函数执行期间，需要手动提交才能让脚本读到这里插入的数据。
        await conn.commit()
    return paper_ids # 返回插入的 ID 列表


# --- 测试用例 (集成测试) ---

@pytest.mark.asyncio # 标记为异步测试
# @pytest.mark.skip(reason="在重构期间跳过较慢的嵌入测试") # 可选：如果嵌入器初始化/运行缓慢，可以取消注释以跳过此测试
async def test_build_faiss_integration(
    postgres_repository_fixture: PostgresRepository, # 请求真实的 PG 仓库 fixture
    test_settings_fixture: Settings,  # 请求测试配置 fixture
    tmp_path: Path, # 请求临时路径 fixture (虽然此测试现在主要用 settings 里的路径)
) -> None:
    """
    集成测试：测试使用真实的数据库和嵌入器构建 Faiss 索引。
    """
    # --- 准备 ---
    pg_repo = postgres_repository_fixture # 获取仓库实例
    test_settings = test_settings_fixture # 获取测试配置实例 (包含临时文件路径)
    print(f"\n[DEBUG] test_build_faiss_integration: Using DB pool: {pg_repo.pool}")
    print(f"[DEBUG] test_build_faiss_integration: Using index path: {test_settings.faiss_index_path}")
    print(f"[DEBUG] test_build_faiss_integration: Using map path: {test_settings.faiss_mapping_path}")


    # 1. 设置：向测试 PG 数据库插入样本数据
    print("[DEBUG] test_build_faiss_integration: Inserting test data into PG...")
    inserted_paper_ids = await insert_faiss_pg_data(pg_repo)
    assert len(inserted_paper_ids) == 2, "未能成功插入测试数据"
    print(f"[DEBUG] test_build_faiss_integration: Inserted paper IDs: {inserted_paper_ids}")


    # 2. 设置：定义临时文件路径（直接使用 test_settings 中的路径）
    # test_settings_fixture 已经配置为使用会话级别的临时文件路径
    temp_index_path = Path(test_settings.faiss_index_path)
    temp_map_path = Path(test_settings.faiss_mapping_path)

    # 确保这些路径的父目录存在 (tmp_path_factory fixture 应该已经创建了基础临时目录)
    temp_index_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. 设置：实例化真实的 TextEmbedder
    print("[DEBUG] test_build_faiss_integration: Initializing TextEmbedder...")
    try:
        embedder = TextEmbedder() # 使用默认配置初始化
        # (可选) 检查嵌入维度是否符合预期
        dimension = embedder.get_embedding_dimension()
        assert isinstance(dimension, int) and dimension > 0, "嵌入器维度无效"
        print(f"[DEBUG] test_build_faiss_integration: Embedder initialized with dimension {dimension}.")
    except Exception as e:
        # 如果嵌入器初始化失败（例如，模型未下载或网络问题），则测试失败并给出提示
        pytest.fail(
            f"初始化 TextEmbedder 失败: {e}。请确保模型已下载或网络可访问。"
        )

    # --- 脚本依赖假设 ---
    # 假设 build_faiss_index 函数接受传入的 pg_repo 和 embedder 实例。

    # --- 执行 ---
    # 4. 调用脚本中的 build_index 函数
    print("[DEBUG] test_build_faiss_integration: Calling build_faiss_index...")
    await build_faiss_index(
        pg_repo=pg_repo, # 传入真实的仓库实例
        embedder=embedder, # 传入真实的嵌入器实例
        index_path=str(temp_index_path),  # 使用来自 test_settings 的路径
        id_map_path=str(temp_map_path),  # 使用来自 test_settings 的路径
        batch_size=32,  # 设置批处理大小
        reset_index=True,  # !!! 关键：设置为 True 确保每次测试都在干净状态下构建索引 !!!
    )
    print("[DEBUG] test_build_faiss_integration: build_faiss_index finished.")


    # --- 断言 ---
    # 5. 检查 Faiss 索引文件和 ID 映射文件是否已在临时路径中创建
    print("[DEBUG] test_build_faiss_integration: Asserting file existence...")
    assert temp_index_path.exists(), f"Faiss 索引文件未找到: {temp_index_path}"
    assert temp_map_path.exists(), f"ID 映射文件未找到: {temp_map_path}"

    # 6. 检查 ID 映射文件的内容
    print("[DEBUG] test_build_faiss_integration: Asserting ID map content...")
    with open(temp_map_path, "r") as f:
        loaded_id_map = json.load(f)
    # JSON 加载后 key 是字符串，需要转回整数进行比较
    loaded_id_map = {int(k): v for k, v in loaded_id_map.items()}
    # 映射的大小应等于插入的论文数量
    assert len(loaded_id_map) == len(inserted_paper_ids), "ID 映射大小不匹配"
    # 映射的值（论文ID）应与插入的ID一致（顺序可能不同，所以比较集合）
    assert set(loaded_id_map.values()) == set(inserted_paper_ids), "ID 映射中的论文 ID 不匹配"
    # 映射的键（Faiss 内部索引）应是从 0 到 n-1 的连续整数
    assert set(loaded_id_map.keys()) == set(range(len(inserted_paper_ids))), "ID 映射的 Faiss 索引键不正确"

    # 7. (可选但推荐) 检查生成的 Faiss 索引的基本属性
    print("[DEBUG] test_build_faiss_integration: Asserting Faiss index properties...")
    try:
        # 读取生成的索引文件
        loaded_index = faiss.read_index(str(temp_index_path))
        # 索引中的向量总数应等于插入的论文数量
        assert loaded_index.ntotal == len(inserted_paper_ids), "Faiss 索引中的向量总数不匹配"
        # 索引的维度应与嵌入器的维度一致
        assert loaded_index.d == dimension, "Faiss 索引的维度与嵌入器维度不匹配"
    except Exception as e:
        pytest.fail(f"读取或验证生成的 Faiss 索引失败: {e}")

    print("[DEBUG] test_build_faiss_integration: All assertions passed.")
    # TODO: 如果需要测试 reset_index=True 的具体行为（例如验证 os.remove 被调用），
    # 可能需要专门为此目的 patch os.remove。


# --- Mock 类定义 ---
# 为单元测试定义模拟类

class MockTextEmbedder:
    """模拟 TextEmbedder 类，用于单元测试。"""
    def __init__(
        self, model_name: Optional[str] = None, device: Optional[str] = None
    ) -> None:
        """模拟初始化过程。"""
        self.model_name = model_name
        self.device = device
        self.model = MagicMock()  # 模拟一个已加载的模型对象
        self._dimension = 384  # 设定一个示例嵌入维度
        print(f"模拟 TextEmbedder 使用模型 '{model_name}' 在设备 '{device}' 上初始化")

    def get_embedding_dimension(self) -> int:
        """返回模拟的嵌入维度。"""
        return self._dimension

    def embed_batch(self, texts: List[Optional[str]]) -> np.ndarray:
        """
        模拟批量嵌入文本。
        返回一个形状正确的随机 numpy 数组。
        """
        print(f"模拟 TextEmbedder 正在嵌入大小为 {len(texts)} 的批次")
        if not texts: # 处理空列表的情况
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        # 返回随机向量作为模拟嵌入结果
        return np.random.rand(len(texts), self._dimension).astype(np.float32)


class MockPostgresRepository:
    """模拟 PostgresRepository 类，用于单元测试。"""
    def __init__(self, pool: AsyncMock) -> None:
        """模拟初始化。"""
        self.pool = pool # 存储传入的模拟连接池 (虽然在此 mock 中未使用)
        # 预设一些用于 get_all_paper_ids_and_text 的模拟数据
        self.paper_data = [(1, "Summary 1"), (2, "Summary 2"), (3, "Summary 3")]
        print("模拟 PostgresRepository 初始化")


    async def get_all_paper_ids_and_text(self) -> AsyncGenerator[Tuple[int, str], None]:
        """
        模拟从数据库异步获取所有论文 ID 和文本（摘要）。
        使用 `yield` 实现异步生成器。
        """
        print("模拟 PostgresRepository 正在获取论文数据...")
        for item in self.paper_data:
            yield item # 逐个产生模拟数据
        print("模拟 PostgresRepository 获取数据完成。")
        # 异步生成器结束时隐式返回
        return

    async def close(self) -> None:
        """模拟关闭方法。"""
        print("模拟 PostgresRepository 关闭")
        pass # 什么也不做


# --- 测试 Fixtures (单元测试用) ---

@pytest.fixture
def mock_faiss_index(mocker: Any) -> MagicMock:
    """Pytest fixture: 模拟 Faiss 索引对象 (faiss.Index)。"""
    # 创建一个 Faiss Index 的模拟对象
    mock_index = MagicMock(spec=faiss.Index)
    mock_index.ntotal = 0 # 初始化向量总数为 0

    # 定义一个 mock 的 add 方法的行为
    # 当 add 被调用时，根据传入向量的数量增加 ntotal
    def mock_add(embeddings: Any) -> None:
        if isinstance(embeddings, np.ndarray):
            # 增加 ntotal 计数
            mock_index.ntotal += embeddings.shape[0]
            print(f"模拟 Faiss Index: 添加了 {embeddings.shape[0]} 个向量，当前总数: {mock_index.ntotal}")
        else:
            print("模拟 Faiss Index: add 收到非 numpy 数组输入")
            pass # 或者抛出错误

    # 将自定义的 mock_add 函数设置为模拟对象的 add 方法的 side_effect
    mock_index.add.side_effect = mock_add

    # Patch 掉 faiss.IndexFlatL2 类，使其在被调用时返回我们的 mock_index
    mocker.patch("faiss.IndexFlatL2", return_value=mock_index)
    # Patch 掉 faiss.write_index 函数，避免实际写入文件
    mocker.patch("faiss.write_index")
    return mock_index # 返回配置好的模拟 Faiss 索引对象


@pytest.fixture
def mock_pg_repo(mocker: Any) -> MockPostgresRepository:
    """Pytest fixture: 提供 MockPostgresRepository 的实例。"""
    # 只需要一个模拟的连接池对象来初始化 MockPostgresRepository
    return MockPostgresRepository(pool=AsyncMock())


@pytest.fixture
def mock_embedder(mocker: Any) -> MockTextEmbedder:
    """Pytest fixture: 提供 MockTextEmbedder 的实例。"""
    return MockTextEmbedder()


# --- 测试用例 (单元测试) ---

@pytest.mark.asyncio
@patch("os.path.exists", return_value=False) # 模拟文件不存在
@patch("os.makedirs") # 模拟创建目录函数
@patch("os.remove") # 模拟删除文件函数
@patch("builtins.open", new_callable=mock_open) # 模拟内置的 open 函数
@patch("json.dump") # 模拟 json.dump 函数
async def test_build_faiss_success_no_reset(
    mock_json_dump: MagicMock, # Mock 对象会被作为参数注入
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock, # 请求模拟 Faiss 索引 fixture
    mock_pg_repo: MockPostgresRepository, # 请求模拟 PG 仓库 fixture
    mock_embedder: MockTextEmbedder, # 请求模拟嵌入器 fixture
    test_settings_fixture: Settings,  # 请求测试配置 fixture (获取路径)
) -> None:
    """单元测试：测试在不重置索引的情况下成功构建 Faiss 索引。"""
    test_settings = test_settings_fixture # 获取测试配置
    print("\n[DEBUG] test_build_faiss_success_no_reset: Running...")


    # --- 执行 ---
    # 调用 build_index 函数，传入所有模拟依赖和来自测试配置的文件路径
    await build_faiss_index(
        # 使用 cast 将模拟对象转换为 Pydantic 期望的类型，以满足类型检查器
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path, # 使用配置中的路径
        id_map_path=test_settings.faiss_mapping_path, # 使用配置中的路径
        batch_size=32,  # 传递批处理大小参数
        reset_index=False, # 不重置索引
    )
    print("[DEBUG] test_build_faiss_success_no_reset: build_faiss_index finished.")

    # --- 断言 ---
    print("[DEBUG] test_build_faiss_success_no_reset: Starting assertions...")
    # 1. 验证是否从数据库获取了论文数据
    #    直接验证异步生成器是否被迭代比较困难，但可以通过检查下游操作（如 embedder 调用次数或 faiss 添加次数）间接验证。
    #    或者，如果 MockPostgresRepository 内部有计数器，可以检查。

    # 2. 验证 Faiss 索引是否被创建和填充
    #    验证 faiss.IndexFlatL2 是否以正确的维度被调用一次
    faiss.IndexFlatL2.assert_called_once_with(mock_embedder.get_embedding_dimension())
    #    验证模拟索引的 add 方法是否被调用（对于小数据量，可能只调用一次）
    mock_faiss_index.add.assert_called()
    #    验证最终索引中的向量总数是否等于模拟仓库提供的数据项数量
    assert mock_faiss_index.ntotal == len(mock_pg_repo.paper_data), "Faiss 索引中的向量总数不正确"
    print("[DEBUG] Asserted Faiss index creation and population.")

    # 3. 验证文件未被删除 (因为 reset_index=False)
    mock_os_remove.assert_not_called()
    print("[DEBUG] Asserted files were not removed.")

    # 4. 验证索引和 ID 映射文件是否被保存
    #    获取预期的目录路径
    expected_dir = os.path.dirname(test_settings.faiss_index_path)
    #    验证 os.makedirs 是否被调用以确保目录存在
    mock_os_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    #    验证 faiss.write_index 是否被调用一次
    faiss.write_index.assert_called_once()
    #    验证 ID 映射文件是否被打开以供写入
    mock_file_open.assert_called_once_with(test_settings.faiss_mapping_path, "w")
    #    构建预期的 ID 映射字典
    expected_id_map = {
        i: paper_id for i, (paper_id, _) in enumerate(mock_pg_repo.paper_data)
    }
    #    验证 json.dump 是否以正确的参数（预期映射和文件句柄）被调用
    mock_json_dump.assert_called_once_with(expected_id_map, mock_file_open())
    print("[DEBUG] Asserted index and ID map saving.")


@pytest.mark.asyncio
@patch("os.path.exists", return_value=True)  # 模拟文件 *确实* 存在
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
async def test_build_faiss_success_with_reset(
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder,
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """单元测试：测试在 *重置* 索引的情况下成功构建 Faiss 索引。"""
    test_settings = test_settings_fixture
    print("\n[DEBUG] test_build_faiss_success_with_reset: Running...")

    # --- 执行 ---
    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,
        reset_index=True, # !!! 设置 reset_index 为 True !!!
    )
    print("[DEBUG] test_build_faiss_success_with_reset: build_faiss_index finished.")


    # --- 断言 ---
    print("[DEBUG] test_build_faiss_success_with_reset: Starting assertions...")
    # 1. 验证 os.path.exists 是否被调用来检查两个文件的存在性
    mock_os_exists.assert_any_call(test_settings.faiss_index_path) # 检查索引文件
    mock_os_exists.assert_any_call(test_settings.faiss_mapping_path) # 检查映射文件
    print("[DEBUG] Asserted os.path.exists calls.")

    # 2. 验证 os.remove 是否被调用来删除两个文件
    #    构建预期的调用列表
    expected_remove_calls = [
        call(test_settings.faiss_index_path), # 对索引文件的调用
        call(test_settings.faiss_mapping_path), # 对映射文件的调用
    ]
    #    使用 assert_has_calls 验证这些调用都发生了（顺序不重要）
    mock_os_remove.assert_has_calls(expected_remove_calls, any_order=True)
    #    验证 os.remove 总共被调用了两次
    assert mock_os_remove.call_count == 2
    print("[DEBUG] Asserted os.remove calls.")

    # 3. 验证后续的保存操作仍然执行（基本检查）
    faiss.write_index.assert_called_once() # 验证索引写入被调用
    mock_json_dump.assert_called_once() # 验证 ID 映射写入被调用
    print("[DEBUG] Asserted saving operations occurred.")


@pytest.mark.asyncio
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
@patch("scripts.sync_pg_to_faiss.logger")  # Mock 脚本中使用的 logger
async def test_build_faiss_embedding_error(
    mock_logger: MagicMock, # 注入模拟的 logger
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder, # 注意这里用的是模拟嵌入器实例
    mocker: Any, # 请求 mocker fixture 用于 patch 实例方法
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """单元测试：测试在嵌入过程中发生错误时的处理情况。"""
    test_settings = test_settings_fixture
    print("\n[DEBUG] test_build_faiss_embedding_error: Running...")

    # --- 准备：模拟错误 ---
    # 定义一个要在嵌入时抛出的异常
    embed_error = Exception("模拟嵌入错误")
    # 使用 mocker.patch.object 来 patch *模拟嵌入器实例* 的 embed_batch 方法，
    # 使其在被调用时抛出定义的异常。
    mocker.patch.object(mock_embedder, "embed_batch", side_effect=embed_error)

    # --- 执行 ---
    # 调用 build_index，预期它会捕获嵌入错误并记录日志
    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder), # 传入的是配置了错误的模拟嵌入器
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,
        reset_index=False,
    )
    print("[DEBUG] test_build_faiss_embedding_error: build_faiss_index finished.")

    # --- 断言 ---
    print("[DEBUG] test_build_faiss_embedding_error: Starting assertions...")
    # 1. 验证模拟嵌入器的 embed_batch 方法确实被调用了
    #    需要通过 mock_embedder (我们 patch 了它的方法) 来访问其方法的 mock 属性
    mock_embedder.embed_batch.assert_called() # type: ignore[attr-defined] # 告诉 mypy 这个属性存在
    print("[DEBUG] Asserted embedder was called.")

    # 2. 验证错误日志是否被记录
    #    断言 mock_logger 的 error 方法至少被调用了一次
    assert mock_logger.error.call_count >= 1, "错误日志未被记录"
    #    获取第一次调用 error 方法的参数
    first_call_args, first_call_kwargs = mock_logger.error.call_args_list[0]
    #    断言调用时只有一个位置参数（错误消息）
    assert len(first_call_args) == 1
    #    断言错误消息的内容符合预期，包含了原始异常信息
    assert (
        first_call_args[0]
        == f"在构建 Faiss 索引期间发生意外错误: {embed_error}"
    ), "记录的错误日志消息不匹配"
    print("[DEBUG] Asserted error logging.")

    # 3. 验证文件保存操作在错误发生后没有被执行
    faiss.write_index.assert_not_called() # 索引不应被写入
    mock_file_open.assert_not_called() # ID 映射文件不应被打开
    mock_json_dump.assert_not_called() # json.dump 不应被调用
    print("[DEBUG] Asserted saving operations were not called.")


@pytest.mark.asyncio
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
@patch("os.remove")
@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
@patch("faiss.write_index", side_effect=IOError("磁盘已满"))  # 模拟 faiss.write_index 抛出 IO 错误
@patch("scripts.sync_pg_to_faiss.logger")  # Mock 脚本中的 logger
async def test_build_faiss_save_error(
    mock_logger: MagicMock, # 注入模拟 logger
    mock_write_index: MagicMock, # 注入模拟的 faiss.write_index (因为我们 patch 了它)
    mock_json_dump: MagicMock,
    mock_file_open: MagicMock,
    mock_os_remove: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_os_exists: MagicMock,
    mock_faiss_index: MagicMock,
    mock_pg_repo: MockPostgresRepository,
    mock_embedder: MockTextEmbedder,
    test_settings_fixture: Settings,  # Inject test_settings
) -> None:
    """单元测试：测试在保存 Faiss 索引文件时发生 IO 错误的处理情况。"""
    test_settings = test_settings_fixture
    print("\n[DEBUG] test_build_faiss_save_error: Running...")

    # --- 执行 ---
    # 调用 build_index，预期它会尝试保存，但在调用 faiss.write_index 时失败
    await build_faiss_index(
        pg_repo=cast(PostgresRepository, mock_pg_repo),
        embedder=cast(TextEmbedder, mock_embedder),
        index_path=test_settings.faiss_index_path,
        id_map_path=test_settings.faiss_mapping_path,
        batch_size=32,
        reset_index=False,
    )
    print("[DEBUG] test_build_faiss_save_error: build_faiss_index finished.")


    # --- 断言 ---
    print("[DEBUG] test_build_faiss_save_error: Starting assertions...")
    # 1. 验证 faiss.write_index (即 mock_write_index) 是否以正确的参数被调用
    mock_write_index.assert_called_once_with(
        mock_faiss_index, test_settings.faiss_index_path
    )
    print("[DEBUG] Asserted write_index was called.")

    # 2. 验证错误日志是否被记录
    #    断言 mock_logger 的 error 方法被调用一次
    mock_logger.error.assert_called_once_with(
        # 检查日志消息是否包含我们模拟的 IO 错误信息
        f"保存 Faiss 索引或 ID 映射时出错: {mock_write_index.side_effect}"
    )
    print("[DEBUG] Asserted error logging.")

    # 3. 验证 ID 映射文件没有被保存 (因为在保存索引时就出错了)
    mock_json_dump.assert_not_called()
    print("[DEBUG] Asserted json.dump was not called.")


# --- 如果需要，可以考虑添加针对脚本 main() 函数本身的测试 ---
# (例如，测试参数解析、资源初始化/清理调用等)