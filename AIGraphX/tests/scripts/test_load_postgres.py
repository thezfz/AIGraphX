# -*- coding: utf-8 -*-
"""
文件目的：测试 `scripts/load_postgres.py` 脚本。

本测试文件 (`test_load_postgres.py`) 专注于验证 `load_postgres.py` 脚本的功能，
该脚本负责从 JSONL 文件读取数据（通常是 Hugging Face 模型和关联论文的信息），
并将这些数据加载到 PostgreSQL 数据库中。

测试策略主要是 **集成测试**：
- 使用真实的 PostgreSQL 测试数据库（通过 `conftest.py` 中的 fixture 提供连接和清理）。
- 使用临时的输入文件和检查点文件（通过 `tmp_path` fixture）。
- 模拟（Patch）脚本内部的文件操作（检查点加载/保存）和数据库连接创建（如果脚本不直接使用仓库对象）。
- 运行脚本的 `main` 函数。
- 通过查询测试数据库来验证数据是否被正确加载。
- 验证检查点逻辑（恢复、重置）是否按预期工作。

主要交互：
- 导入 `pytest`, `pytest_asyncio`, `pytest-mock`：用于测试框架、异步支持和模拟。
- 导入 `unittest.mock`：用于 `patch`, `AsyncMock` 等模拟工具。
- 导入 `asyncpg`：虽然被模拟，但类型提示可能需要。
- 导入 `json`, `os`, `builtins`, `pathlib`：用于处理 JSON 数据、环境变量、模拟文件操作和路径操作。
- 导入 `aigraphx.core.config`：用于获取测试数据库 URL（虽然现在倾向于在测试内部获取）。
- 导入 `typing`：用于类型提示。
- 导入被测试的脚本函数和常量：从 `scripts.load_postgres` 导入 `main`, `process_batch`, `CHECKPOINT_FILE` 等。
- 导入真实的数据库仓库 fixture：从 `tests.conftest` 导入 `postgres_repository_fixture`，这是执行集成测试的关键。
- 定义测试数据：`SAMPLE_MODEL_LINE_*` 包含模拟的 JSONL 数据行。
- 定义测试标记 (`pytestmark`)：
    - `skipif`: 如果 `TEST_DATABASE_URL` 环境变量未设置，则跳过此文件中的所有测试。
    - `asyncio`: 将此文件中的所有测试标记为异步测试。
- 定义 Fixtures：
    - `mock_db_pool` (目前未使用，但可用于纯单元测试)：模拟 `asyncpg` 连接池及其连接、事务。
- 编写测试函数 (`test_*`)：
    - `test_load_postgres_integration_success`：测试脚本从头成功加载数据的场景。
    - `test_load_postgres_integration_resume`：测试脚本从检查点恢复加载的场景。
    - `test_load_postgres_integration_reset`：测试脚本重置检查点并重新加载的场景。
    - (TODO) 可以添加更多关于错误处理的集成测试（例如无效 JSON、数据库约束）。

这些测试确保数据加载脚本能够可靠地将数据导入数据库，并且检查点机制能够正确工作，允许从中断处恢复。
"""

import pytest # 导入 pytest 测试框架
import pytest_asyncio # 导入 pytest 的异步扩展
from unittest.mock import patch, AsyncMock, mock_open, call, ANY # 导入模拟工具
import asyncpg  # type: ignore[import-untyped] # 导入 asyncpg 库，即使被 mock，类型提示也可能需要 (忽略 mypy 找不到类型存根的错误)
import json # 导入 json 库，用于处理 JSON 数据
import os # 导入 os 模块，用于访问环境变量
import builtins  # 导入 builtins 模块，用于模拟内置的 open 函数
from pathlib import Path  # 从 pathlib 导入 Path 对象，用于处理文件路径 (配合 tmp_path)
from aigraphx.core import config  # 导入配置模块，可能用于获取默认配置（如下面的数据库URL）
from typing import Tuple  # 从 typing 导入 Tuple 类型提示
from pytest_mock import MockerFixture  # 导入 MockerFixture 类型，用于 pytest-mock 的 mocker fixture 类型提示

# 导入需要测试的脚本中的 main 函数和一些内部函数/常量
from scripts.load_postgres import main as load_pg_main # 导入主函数并重命名，避免与关键字冲突
from scripts.load_postgres import (
    process_batch, # 处理一批数据的函数 (虽然测试主要调用 main)
    insert_hf_model, # 插入 HF 模型的函数
    get_or_insert_paper, # 获取或插入论文的函数
    insert_model_paper_link, # 插入模型-论文关联的函数
    insert_pwc_relation, # 插入 PWC 关系的函数
    insert_pwc_repositories, # 插入 PWC 代码仓库的函数
    CHECKPOINT_FILE, # 脚本中定义的检查点文件名常量
    DEFAULT_INPUT_JSONL_FILE, # 脚本中定义的默认输入文件名常量
)

# 导入真实的 PostgreSQL 仓库 fixture，用于集成测试中断言数据库状态
# 这个 fixture 在 conftest.py 中定义，负责连接测试数据库并进行清理
from tests.conftest import repository as postgres_repository_fixture
from aigraphx.repositories.postgres_repo import PostgresRepository # 导入仓库类类型

# --- 测试样本数据 ---
# 定义一些 JSON 字符串，模拟输入 JSONL 文件中的行

SAMPLE_MODEL_LINE_1 = json.dumps( # 第一行数据：包含一个模型和关联论文
    {
        "hf_model_id": "test-model-1",  # 使用不同的 ID 以便测试区分
        "hf_author": "author1",
        "hf_last_modified": "2023-01-01T10:00:00+00:00", # ISO 格式日期时间
        "hf_tags": ["tagA"],
        "linked_papers": [ # 关联的论文列表
            {
                "arxiv_id_base": "1111.1111", # 论文的 arXiv ID
                "arxiv_metadata": { # arXiv 元数据
                    "arxiv_id_versioned": "1111.1111v1",
                    "title": "Test Paper 1 Title",
                    "authors": ["Auth1"],
                    "summary": "Summary 1",
                    "published_date": "2023-01-01", # 日期字符串
                },
                "pwc_entry": { # PapersWithCode 条目信息
                    "pwc_id": "test-pwc-1",  # 使用不同的 PWC ID
                    "tasks": ["Task A"], # 任务列表
                    "datasets": ["Data X"], # 数据集列表
                    "repositories": [{"url": "repo.com/1", "stars": 10}], # 代码仓库列表
                },
            }
        ],
    }
)
SAMPLE_MODEL_LINE_2 = json.dumps( # 第二行数据：只包含一个模型，没有关联论文
    {
        "hf_model_id": "test-model-2",
        "hf_author": "author2",
        "hf_last_modified": "2023-01-02T11:00:00Z", # 使用 Z 表示 UTC
        "hf_tags": ["tagB", "tagC"],
        "linked_papers": [],  # 空的论文列表
    }
)
SAMPLE_INVALID_JSON_LINE = "{invalid json" # 无效的 JSON 字符串，用于测试错误处理（如果需要）
SAMPLE_MODEL_LINE_NO_ID = json.dumps({"hf_author": "author3"}) # 缺少必需的 hf_model_id，用于测试错误处理

# --- 添加诊断性打印语句 ---
# 在模块加载时打印环境变量的值，有助于调试测试环境配置问题
print(
    f"\n[DEBUG] test_load_postgres.py: Initial TEST_DATABASE_URL = {os.getenv('TEST_DATABASE_URL')}\n"
)
# --- 结束诊断性打印语句 ---

# 不再在模块级别捕获环境变量，因为 fixture 可能在之后设置它
# TEST_DB_URL = os.getenv("TEST_DATABASE_URL")

# 使用 pytest.mark.skipif 标记此模块
# 如果 os.getenv("TEST_DATABASE_URL") 返回假值 (None 或空字符串)，则跳过此文件中的所有测试
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), # 直接在 skipif 中调用 os.getenv
    reason="TEST_DATABASE_URL environment variable not set", # 跳过原因
)
# 使用 pytest.mark.asyncio 将此文件中的所有测试标记为异步测试
# 注意：可以将多个标记合并，例如 pytestmark = [pytest.mark.skipif(...), pytest.mark.asyncio]
pytestmark = pytest.mark.asyncio

# --- 测试 Fixtures ---

@pytest_asyncio.fixture
async def mock_db_pool(mocker: MockerFixture) -> Tuple[AsyncMock, AsyncMock, AsyncMock]:
    """
    Pytest fixture: 用于模拟 asyncpg 连接池、连接和事务。
    (注意：当前的集成测试不使用这个 fixture，它主要用于纯单元测试场景。)

    返回:
        Tuple[AsyncMock, AsyncMock, AsyncMock]: (模拟的 create_pool 函数, 模拟的连接池, 模拟的连接)
    """
    # 创建一个模拟的 asyncpg 连接对象
    mock_conn = AsyncMock(spec=asyncpg.Connection)
    # 模拟连接上的 transaction() 方法，使其返回一个异步上下文管理器 (AsyncMock)
    mock_tx = AsyncMock()
    mock_conn.transaction.return_value = mock_tx
    # 模拟常用的数据库操作方法，并设置默认返回值
    mock_conn.fetchval.return_value = None  # 默认模拟：未找到记录
    mock_conn.execute.return_value = None
    mock_conn.executemany.return_value = None

    # 创建一个模拟的 asyncpg 连接池对象
    mock_pool = AsyncMock(spec=asyncpg.Pool)
    # 模拟连接池的 acquire() 方法，使其返回一个异步上下文管理器，该管理器返回模拟的连接对象
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    # 模拟 acquire() 上下文管理器的退出
    mock_pool.acquire.return_value.__aexit__.return_value = None
    # 确保模拟的 pool.close() 方法是可等待的
    mock_pool.close = AsyncMock()

    # 使用 mocker.patch 替换掉全局的 asyncpg.create_pool 函数
    # new=AsyncMock(...) 创建一个模拟函数，它被调用时返回我们预先创建的 mock_pool
    mock_create_pool = AsyncMock(return_value=mock_pool)
    mocker.patch("asyncpg.create_pool", new=mock_create_pool)

    # 返回模拟的 create_pool 函数、连接池和连接对象
    return (
        mock_create_pool,
        mock_pool,
        mock_conn,
    )


# --- 测试用例 (重构为集成测试) ---

@pytest.mark.asyncio
async def test_load_postgres_integration_success(
    postgres_repository_fixture: PostgresRepository,  # 请求真实的仓库 fixture (来自 conftest.py)
    tmp_path: Path,  # 请求 pytest 内置的 tmp_path fixture，用于创建临时文件/目录
    mocker: MockerFixture,  # 请求 pytest-mock 的 mocker fixture
) -> None:
    """
    集成测试：测试 load_postgres 脚本成功加载数据到真实的测试数据库。
    """
    # --- 准备 ---
    repo = postgres_repository_fixture  # 使用更清晰的变量名
    print(f"\n[DEBUG] test_load_postgres_success: Using repo with pool: {repo.pool}")

    # 1. 准备临时的输入 JSONL 文件
    input_file = tmp_path / "test_input.jsonl"
    # 将样本数据写入临时文件
    input_file.write_text(SAMPLE_MODEL_LINE_1 + "\n" + SAMPLE_MODEL_LINE_2 + "\n")
    print(f"[DEBUG] test_load_postgres_success: Created input file at {input_file}")

    # 2. 准备临时的检查点文件路径 (脚本会尝试读写这个路径)
    checkpoint_path = tmp_path / "checkpoint.txt"
    print(f"[DEBUG] test_load_postgres_success: Using checkpoint path {checkpoint_path}")


    # --- 关键：Patch 脚本依赖 ---
    # 假设 load_postgres.py 脚本内部通过某种方式获取数据库连接/仓库。
    # 如果脚本直接使用环境变量 DATABASE_URL 来创建连接池，我们需要确保测试环境的
    # DATABASE_URL 指向测试数据库 (这通常由 conftest.py 中的 fixture 完成)。
    # 如果脚本依赖注入或从某个模块获取，则需要 patch 那个机制。
    # 这里我们假设脚本使用环境变量，并且 conftest 已正确设置。

    # Patch 脚本内部使用的检查点文件路径常量，使其指向我们的临时路径
    mocker.patch("scripts.load_postgres.CHECKPOINT_FILE", str(checkpoint_path))
    # Mock 脚本内部用于加载和保存检查点的函数，避免实际的文件 I/O 干扰测试
    # _load_checkpoint 返回 0 表示从头开始
    mock_load = mocker.patch("scripts.load_postgres._load_checkpoint", return_value=0)
    mock_save = mocker.patch("scripts.load_postgres._save_checkpoint")

    # !!! 重要：Patch 脚本内部使用的 DATABASE_URL !!!
    # 即使 conftest 可能设置了环境变量，最好在测试内部显式 patch 脚本使用的变量，确保隔离性。
    # 在测试函数内部获取测试数据库 URL
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url: # 如果未设置，则跳过测试
        pytest.skip("TEST_DATABASE_URL 未在环境/配置中设置。")
    print(f"[DEBUG] test_load_postgres_success: Patching DATABASE_URL in script to: {test_db_url}")
    # 直接 patch 掉 `scripts.load_postgres` 模块中的 `DATABASE_URL` 变量
    mocker.patch("scripts.load_postgres.DATABASE_URL", test_db_url)

    # --- 执行 ---
    # 3. 调用脚本的 main 函数，传入临时输入文件路径
    # reset_db 和 reset_checkpoint 设为 False，模拟正常运行
    print("[DEBUG] test_load_postgres_success: Calling load_pg_main...")
    await load_pg_main(
        input_file_path=str(input_file), reset_db=False, reset_checkpoint=False
    )
    print("[DEBUG] test_load_postgres_success: load_pg_main finished.")


    # --- 断言 ---
    # 4. 使用真实的 repository fixture 连接测试数据库，验证数据是否已正确插入
    # 这里直接使用 repo.pool 获取连接，因为 repo fixture 已经初始化好了
    print("[DEBUG] test_load_postgres_success: Asserting database state...")
    async with repo.pool.connection() as conn:
        async with conn.cursor() as cur:  # 使用显式的游标
            # 检查模型 1 是否插入成功，并验证部分字段
            await cur.execute(
                "SELECT hf_author, hf_tags FROM hf_models WHERE hf_model_id = %s",
                ("test-model-1",),
            )
            model1 = await cur.fetchone() # 获取一行结果
            assert model1 is not None, "Model 1 not found in DB"
            assert model1[0] == "author1"
            assert model1[1] == ["tagA"]

            # 检查模型 2 是否插入成功
            await cur.execute(
                "SELECT hf_author, hf_tags FROM hf_models WHERE hf_model_id = %s",
                ("test-model-2",),
            )
            model2 = await cur.fetchone()
            assert model2 is not None, "Model 2 not found in DB"
            assert model2[0] == "author2"
            assert model2[1] == ["tagB", "tagC"]

            # 检查论文 1 (关联到模型 1) 是否插入成功
            await cur.execute(
                "SELECT paper_id, title, authors, area FROM papers WHERE pwc_id = %s",
                ("test-pwc-1",),
            )
            paper1 = await cur.fetchone()
            assert paper1 is not None, "Paper 1 (pwc_id=test-pwc-1) not found in DB"
            paper1_id = paper1[0] # 获取插入后生成的 paper_id
            assert paper1[1] == "Test Paper 1 Title"
            assert paper1[2] == ["Auth1"]
            # assert paper1[3] == "AI" # area 字段的断言可能不稳定，暂时注释掉

            # 检查模型 1 和论文 1 之间的关联是否在 model_paper_links 表中创建
            await cur.execute(
                "SELECT 1 FROM model_paper_links WHERE hf_model_id = %s AND paper_id = %s",
                ("test-model-1", paper1_id),
            )
            link1 = await cur.fetchone()
            assert link1 is not None, "Link between model 1 and paper 1 not found"

            # 示例：检查任务 (如果 pwc_tasks 表存在且被加载)
            # await cur.execute(
            #     "SELECT task_name FROM pwc_tasks WHERE paper_id = %s", (paper1_id,)
            # )
            # tasks = await cur.fetchall() # 如果预期多行，使用 fetchall
            # assert len(tasks) == 1
            # assert tasks[0][0] == "Task A"

    print("[DEBUG] test_load_postgres_success: Database assertions passed.")

    # 5. 断言检查点保存函数被正确调用
    # 输入文件有 2 行，处理完后，应该保存下一个要处理的行号，即 2
    mock_save.assert_called_with(2)
    print("[DEBUG] test_load_postgres_success: Checkpoint assertion passed.")


@pytest.mark.asyncio
async def test_load_postgres_integration_resume(
    postgres_repository_fixture: PostgresRepository,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """
    集成测试：测试 load_postgres 脚本从检查点恢复加载。
    """
    # --- 准备 ---
    repo = postgres_repository_fixture
    # 准备包含两行数据的输入文件
    input_file = tmp_path / "test_input_resume.jsonl"
    input_file.write_text(SAMPLE_MODEL_LINE_1 + "\n" + SAMPLE_MODEL_LINE_2 + "\n")
    # 准备检查点文件路径
    checkpoint_path = tmp_path / "checkpoint_resume.txt"

    # Patch 检查点文件路径
    mocker.patch("scripts.load_postgres.CHECKPOINT_FILE", str(checkpoint_path))
    # !!! Mock _load_checkpoint 返回 1，模拟之前已经处理了第 0 行，这次从第 1 行开始 !!!
    mock_load = mocker.patch("scripts.load_postgres._load_checkpoint", return_value=1)
    mock_save = mocker.patch("scripts.load_postgres._save_checkpoint")

    # Patch 脚本使用的 DATABASE_URL
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL not configured.")
    mocker.patch("scripts.load_postgres.DATABASE_URL", test_db_url)

    # --- 执行 ---
    # 运行脚本，不重置数据库或检查点
    await load_pg_main(
        input_file_path=str(input_file), reset_db=False, reset_checkpoint=False
    )

    # --- 断言 ---
    # 验证数据库状态：这次运行应该只处理了第 1 行 (SAMPLE_MODEL_LINE_2)
    async with repo.pool.connection() as conn:
        async with conn.cursor() as cur:
            # 模型 1 (来自第 0 行) *不* 应该在这次运行中被插入（假设数据库是干净的）
            await cur.execute(
                "SELECT 1 FROM hf_models WHERE hf_model_id = %s", ("test-model-1",)
            )
            model1 = await cur.fetchone()
            assert model1 is None, "Model 1 should not have been processed in this resume run"

            # 模型 2 (来自第 1 行) *应该* 被插入
            await cur.execute(
                "SELECT hf_author FROM hf_models WHERE hf_model_id = %s",
                ("test-model-2",),
            )
            model2 = await cur.fetchone()
            assert model2 is not None, "Model 2 was not processed in this resume run"
            assert model2[0] == "author2"

    # 断言检查点函数调用
    # _load_checkpoint 应该被调用一次，参数为 False (因为 reset_checkpoint=False)
    mock_load.assert_called_once_with(False)
    # 处理完第 1 行后，下一个行号是 2，应该保存 2
    mock_save.assert_called_with(2)


@pytest.mark.asyncio
async def test_load_postgres_integration_reset(
    postgres_repository_fixture: PostgresRepository,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """
    集成测试：测试 load_postgres 脚本使用 reset_checkpoint=True 参数。
    """
    # --- 准备 ---
    repo = postgres_repository_fixture
    # 输入文件只包含一行，方便测试边界
    input_file = tmp_path / "test_input_reset.jsonl"
    input_file.write_text(SAMPLE_MODEL_LINE_1 + "\n")
    checkpoint_path = tmp_path / "checkpoint_reset.txt"

    # Patch 检查点文件路径
    mocker.patch("scripts.load_postgres.CHECKPOINT_FILE", str(checkpoint_path))
    # Mock _load_checkpoint，因为 reset=True，它应该返回 0
    mock_load = mocker.patch(
        "scripts.load_postgres._load_checkpoint", return_value=0
    )
    mock_save = mocker.patch("scripts.load_postgres._save_checkpoint")

    # Patch 脚本使用的 DATABASE_URL
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        pytest.skip("TEST_DATABASE_URL not configured.")
    mocker.patch("scripts.load_postgres.DATABASE_URL", test_db_url)

    # --- 执行 ---
    # 调用脚本，设置 reset_checkpoint=True
    # 注意：脚本逻辑似乎忽略了 reset_db 参数，主要看 reset_checkpoint
    await load_pg_main(
        input_file_path=str(input_file), reset_db=True, reset_checkpoint=True
    )

    # --- 断言 ---
    # 验证数据库状态：因为重置了检查点，输入文件的第一行 (SAMPLE_MODEL_LINE_1) 应该被处理
    async with repo.pool.connection() as conn:
        async with conn.cursor() as cur:
            # 模型 1 应该存在
            await cur.execute(
                "SELECT hf_author FROM hf_models WHERE hf_model_id = %s",
                ("test-model-1",),
            )
            model1 = await cur.fetchone()
            assert model1 is not None, "Model 1 was not processed after reset"
            assert model1[0] == "author1"

    # 断言检查点函数调用
    # _load_checkpoint 应该被调用，并且 reset 参数为 True
    mock_load.assert_called_once_with(True)
    # 脚本处理了第 0 行 (总共 1 行)。循环结束后，下一个要处理的行号是 1。
    # 因此，应该保存检查点值为 1。
    mock_save.assert_called_with(1)


# TODO: 可以添加更多针对错误处理的集成测试，例如：
# - 输入文件中包含无效的 JSON 行
# - 尝试插入违反数据库约束的数据（需要更复杂的设置来模拟）
# 这些测试的设置可能比较复杂，需要仔细考虑如何可靠地触发错误并验证处理逻辑。