# -*- coding: utf-8 -*-
"""
文件目的：测试 Faiss 仓库类 (tests/repositories/test_faiss_repo.py)

本文件包含针对 `aigraphx.repositories.faiss_repo.FaissRepository` 类的单元测试和集成测试。
`FaissRepository` 类封装了与 Faiss 向量索引文件 (.index) 和 ID 映射文件 (.json) 的交互逻辑，
主要用于加载索引/映射，并提供基于向量相似度的搜索功能。

核心测试策略：
- 使用 `pytest` 测试框架。
- 利用 `pytest` 内建的 `tmp_path` fixture 在每个测试函数运行时创建独立的临时目录。
- 在 `repository` fixture 中，于 `tmp_path` 提供的临时目录内动态创建小型的 Faiss 索引文件和 ID 映射文件，包含预定义的测试向量和 ID。
- `repository` fixture 初始化一个 `FaissRepository` 实例，指向这些临时文件，并将其提供给测试函数。
- 测试覆盖了仓库的加载、属性检查、获取索引大小、成功的相似性搜索以及各种边界情况（如 k 值选择、查询向量维度错误）和错误处理（仓库未就绪）。
- 由于 `search_similar` 方法是异步的，测试函数使用 `async def` 并标记为 `@pytest.mark.asyncio`。
- 使用 `unittest.mock.patch` 来模拟 `is_ready` 方法，以测试在仓库未准备好时调用搜索方法的行为。

与其他文件的交互：
- 导入 `pytest` 和相关类型提示。
- 导入 `os` (未使用), `json` (读写 ID 映射), `numpy` (创建和处理向量), `faiss` (构建测试索引), `unittest.mock` (模拟), `pathlib` (处理 `tmp_path`)。
- 导入被测试的类 `aigraphx.repositories.faiss_repo.FaissRepository`。
- 通过 `tmp_path` 读写文件系统中的临时文件。
"""

import pytest
import os  # 导入 os 模块，虽然在此文件中 pathlib 更常用
import json  # 导入 json 模块，用于读写 ID 映射文件
import numpy as np  # 导入 numpy 用于创建和操作数值数组（向量）
import faiss  # 导入 faiss 库，用于创建和操作向量索引。注意：需要单独安装。类型提示标记为忽略未类型化的导入。
from unittest.mock import (
    patch,
    MagicMock,
)  # 导入 patch 用于模拟对象方法，MagicMock 用于创建模拟对象
from pathlib import (
    Path,
)  # 导入 Path 对象，用于处理文件路径，特别是 pytest 的 tmp_path fixture
from typing import Dict, List, Tuple, Any, Optional, Generator  # 导入类型提示

# 导入被测试的类
from aigraphx.repositories.faiss_repo import FaissRepository

# --- 测试数据常量 (动态生成时使用) ---
# 这些常量定义了将在 fixture 中动态创建的测试索引的维度和内容。
TEST_DIMENSION = 8  # 测试向量的维度
TEST_VECTORS = np.array(  # 定义测试向量 NumPy 数组
    [
        [0.1] * TEST_DIMENSION,  # 第一个向量，靠近 0.1
        [0.2] * TEST_DIMENSION,  # 第二个向量，靠近 0.2
        [0.9] * TEST_DIMENSION,  # 第三个向量，靠近 0.9
    ]
).astype(np.float32)  # Faiss 通常需要 float32 类型
TEST_IDS = [101, 202, 999]  # 与上述向量对应的原始 ID (例如 paper_id 或 model_db_id)
# 创建从 Faiss 内部索引 (0, 1, 2) 到原始 ID 的映射字典
TEST_ID_MAP = {i: id_ for i, id_ in enumerate(TEST_IDS)}
# 预期索引中包含的向量数量
TEST_EXPECTED_NUM_VECTORS = len(TEST_IDS)

# --- Fixtures ---


# 使用 pytest.fixture 定义一个 fixture，名为 'repository'
# scope="function" 表示这个 fixture 会为每个测试函数单独执行一次，确保隔离性
# 它接收 pytest 内建的 tmp_path fixture 作为参数，tmp_path 提供一个唯一的临时目录 Path 对象
@pytest.fixture(scope="function")
def repository(tmp_path: Path) -> Generator[FaissRepository, None, None]:
    """
    Pytest Fixture: 创建临时的 Faiss 索引和 ID 映射文件。

    这个 fixture 利用 pytest 的 `tmp_path` 功能：
    1. 在 `tmp_path` 指定的临时目录中，定义测试索引文件 (`test.index`) 和 ID 映射文件 (`test_ids.json`) 的路径。
    2. 使用预定义的 `TEST_VECTORS` 和 `TEST_DIMENSION` 构建一个简单的 Faiss 索引 (`IndexFlatL2`)。
    3. 将构建好的索引写入临时的 `test.index` 文件。
    4. 将预定义的 `TEST_ID_MAP` 写入临时的 `test_ids.json` 文件。
    5. 使用这两个临时文件的路径初始化一个 `FaissRepository` 实例。
    6. (可选) 检查仓库是否成功加载了临时数据。
    7. 使用 `yield` 将初始化好的 `FaissRepository` 实例提供给测试函数。
    8. 测试函数执行完毕后，`tmp_path` 会自动清理所有创建的临时文件和目录。

    Args:
        tmp_path (Path): Pytest 提供的临时目录路径对象。

    Yields:
        FaissRepository: 一个已使用临时文件初始化的 FaissRepository 实例。
    """
    # 1. 定义临时文件的完整路径
    index_path = tmp_path / "test.index"
    id_map_path = tmp_path / "test_ids.json"

    # 2. 构建并保存测试 Faiss 索引
    index = faiss.IndexFlatL2(TEST_DIMENSION)  # 创建一个简单的 L2 距离索引
    index.add(TEST_VECTORS)  # 向索引中添加测试向量
    faiss.write_index(index, str(index_path))  # 将索引写入临时文件

    # 3. 保存测试 ID 映射
    with open(id_map_path, "w") as f:
        json.dump(TEST_ID_MAP, f)  # 将 ID 映射字典写入临时 JSON 文件

    # 4. 使用临时文件路径初始化 FaissRepository
    repo = FaissRepository(index_path=str(index_path), id_map_path=str(id_map_path))

    # 5. (可选) 检查仓库是否初始化成功
    if not repo.is_ready():
        pytest.fail("FaissRepository failed to initialize with temporary test data.")

    # 6. 使用 yield 将初始化好的仓库实例传递给测试函数
    yield repo

    # 7. 清理：tmp_path fixture 会自动处理临时文件和目录的删除


# --- 测试用例 ---


def test_load_and_properties(repository: FaissRepository) -> None:
    """
    测试场景：验证仓库是否成功加载了临时索引和映射文件，并检查其基本属性。
    策略：使用 `repository` fixture 获取初始化的仓库实例，断言其内部状态。
    """
    assert repository.index is not None  # 确保索引对象已加载
    assert repository.index.d == TEST_DIMENSION  # 验证索引维度是否正确
    assert (
        repository.index.ntotal == TEST_EXPECTED_NUM_VECTORS
    )  # 验证索引中的向量数量是否正确
    # 验证加载后的 ID 映射字典内容是否与原始测试数据一致
    # 注意：从 JSON 加载后，键可能需要是整数类型，FaissRepository 内部应处理好类型转换
    assert repository.id_map == TEST_ID_MAP
    assert repository.is_ready() is True  # 验证仓库是否处于就绪状态


def test_get_index_size(repository: FaissRepository) -> None:
    """
    测试场景：验证 `get_index_size` 方法是否返回正确的向量数量。
    策略：调用 `get_index_size` 并断言其返回值。
    """
    assert repository.get_index_size() == TEST_EXPECTED_NUM_VECTORS


@pytest.mark.asyncio  # 标记为异步测试
async def test_search_similar_found(repository: FaissRepository) -> None:
    """
    测试场景：执行相似性搜索，并验证是否能找到预期中最近的向量及其对应的原始 ID。
    策略：构造一个靠近第一个测试向量的查询向量，调用 `search_similar`，断言返回结果的 ID 和顺序。
    """
    # 创建一个查询向量，其值接近第一个测试向量 ([0.1]*8)，该向量对应的原始 ID 是 101
    query_vec = np.array([[0.11] * TEST_DIMENSION]).astype(np.float32)
    k = 2  # 请求返回最近的 2 个邻居

    # 调用异步的 search_similar 方法
    results = await repository.search_similar(query_vec, k=k)

    # 断言返回结果的数量是否为 k
    assert len(results) == k
    # 断言第一个结果（最近的）的 ID 是否为 101
    assert results[0][0] == 101
    # 断言第一个结果的距离值是 float 类型
    assert isinstance(results[0][1], float)
    # 断言第二个结果（次近的）的 ID 是否为 202 (对应向量 [0.2]*8)
    assert results[1][0] == 202
    # 断言第一个结果的距离小于第二个结果的距离 (结果应按距离升序排列)
    assert results[0][1] < results[1][1]


@pytest.mark.asyncio
async def test_search_similar_k_too_large(repository: FaissRepository) -> None:
    """
    测试场景：当请求的 K 值大于索引中的向量总数时，进行相似性搜索。
    预期：应返回索引中所有存在的向量，按距离排序。
    策略：设置 K 值大于 `TEST_EXPECTED_NUM_VECTORS`，调用 `search_similar`，断言返回结果的数量。
    """
    # 创建一个靠近第三个测试向量 ([0.9]*8, ID 999) 的查询向量
    query_vec = np.array([[0.91] * TEST_DIMENSION]).astype(np.float32)
    # 设置 K 值大于索引中的向量总数
    request_k = TEST_EXPECTED_NUM_VECTORS + 5
    # 调用搜索方法
    results = await repository.search_similar(query_vec, k=request_k)
    # 断言返回的结果数量等于索引中的实际向量数量
    assert len(results) == TEST_EXPECTED_NUM_VECTORS
    # 断言最近的结果是否为 ID 999
    assert results[0][0] == 999


@pytest.mark.asyncio
async def test_search_similar_k_zero(repository: FaissRepository) -> None:
    """
    测试场景：当请求的 K 值为 0 时，进行相似性搜索。
    预期：应返回空列表。
    策略：设置 K=0，调用 `search_similar`，断言结果为空列表。
    """
    # 创建一个随机查询向量
    query_vec = np.random.rand(1, TEST_DIMENSION).astype(np.float32)
    # 调用搜索方法，设置 k=0
    results = await repository.search_similar(query_vec, k=0)
    # 断言返回结果为空列表
    assert results == []


@pytest.mark.asyncio
async def test_search_wrong_dimension(repository: FaissRepository, caplog: Any) -> None:
    """
    测试场景：使用维度与索引维度不匹配的查询向量进行搜索。
    预期：应返回空列表，并记录错误日志。
    策略：创建维度错误的查询向量，调用 `search_similar`，断言结果为空，并检查日志输出。
          `caplog` 是 pytest 的内建 fixture，用于捕获日志信息。
    """
    # 创建一个维度比索引维度大的查询向量
    wrong_dim_vec = np.random.rand(1, TEST_DIMENSION + 5).astype(np.float32)
    # 调用搜索方法
    results = await repository.search_similar(wrong_dim_vec, k=1)
    # 断言结果为空列表
    assert results == []
    # 断言捕获到的日志文本中包含维度不匹配的错误信息
    assert "does not match index dimension" in caplog.text


@pytest.mark.asyncio
# 使用 patch.object 来模拟 FaissRepository 实例的 is_ready 方法
# 在这个测试函数执行期间，所有对 repository.is_ready() 的调用都会返回 False
@patch.object(FaissRepository, "is_ready", return_value=False)
async def test_search_when_not_ready(
    mock_is_ready: MagicMock,  # patch 会将模拟对象作为参数注入测试函数
    repository: FaissRepository,
    caplog: Any,
) -> None:
    """
    测试场景：当仓库未处于就绪状态时（例如，索引或 ID 映射加载失败），调用搜索方法。
    预期：应返回空列表，并记录警告或错误日志。
    策略：使用 `@patch.object` 模拟 `is_ready` 方法使其返回 `False`，然后调用 `search_similar`，
          断言结果为空，并检查模拟方法是否被调用以及日志输出。
    """
    # 创建一个随机查询向量
    query_vec = np.random.rand(1, TEST_DIMENSION).astype(np.float32)
    # 调用搜索方法 (此时 repository.is_ready() 会返回 False)
    results = await repository.search_similar(query_vec, k=1)
    # 断言结果为空列表
    assert results == []
    # 验证被模拟的 is_ready 方法确实被调用了一次
    mock_is_ready.assert_called_once()
    # 断言捕获到的日志文本中包含仓库未就绪的警告或错误信息
    assert "index or ID map not ready" in caplog.text
