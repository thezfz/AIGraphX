# tests/scripts/test_sync_pg_to_neo4j.py
# -*- coding: utf-8 -*-
"""
文件目的：测试 `scripts/sync_pg_to_neo4j.py` 脚本。

本测试文件 (`test_sync_pg_to_neo4j.py`) 专注于验证 `sync_pg_to_neo4j.py` 脚本的功能，
该脚本负责从 PostgreSQL 数据库读取数据（Hugging Face 模型、论文、关联关系等），
并将这些数据同步（创建或更新）到 Neo4j 图数据库中，形成知识图谱。

测试策略主要是 **集成测试**：
- 使用真实的 PostgreSQL 测试数据库（通过 `postgres_repository_fixture`）。
- 使用真实的 Neo4j 测试数据库（通过 `neo4j_repo_fixture`）。这两个 fixture 均来自 `conftest.py`，负责数据库的连接、清理和模式（约束/索引）设置。
- 在测试 PG 数据库中插入样本数据。
- 运行脚本的核心同步逻辑 (`run_sync`)。
- 通过查询测试 Neo4j 数据库来验证节点和关系是否已按预期创建。

主要交互：
- 导入 `pytest`, `pytest_asyncio`, `unittest.mock`, `pytest_mock`, `datetime`, `typing`, `sys`, `os`, `json`, `pprint`, `asyncio`, `logging`：用于测试框架、异步支持、模拟、日期时间处理、类型提示、路径处理、JSON 处理、调试打印、日志记录等。
- 设置测试标记 (`pytestmark`)：标记所有测试为异步，并在会话范围内共享事件循环。
- 动态添加项目根目录到 `sys.path`：确保可以正确导入项目内部的模块（`aigraphx` 和 `scripts`）。
- 导入被测试的脚本函数：`run_sync`。
- 导入真实的仓库类：`PostgresRepository`, `Neo4jRepository`（用于类型提示和依赖）。
- 导入真实的仓库 Fixtures：`postgres_repository_fixture`, `neo4j_repo_fixture`。
- 定义测试常量和数据：`PG_FETCH_BATCH_SIZE_TEST`, `NEO4J_WRITE_BATCH_SIZE_TEST`, `TEST_HF_MODEL_*`, `TEST_PAPER_*`, `TEST_LINK_*`。
- 定义数据库辅助函数：`insert_pg_data` 用于在测试前准备 PG 数据。
- 编写测试函数 (`test_run_sync_integration`)：
    - 准备 PG 测试数据。
    - 调用 `run_sync` 函数，传入真实的 PG 和 Neo4j 仓库实例。
    - 连接到 Neo4j 测试数据库，执行 Cypher 查询以验证：
        - HFModel 节点是否创建，属性是否正确。
        - Paper 节点是否创建，属性是否正确。
        - MENTIONS 关系是否在正确的模型和论文节点之间创建。
        - (增强部分) Task, Dataset, Repository 节点及其与 Paper 节点的关系是否正确创建（需要额外插入 PWC 关系数据并可能再次运行同步）。

这些测试确保数据同步脚本能够可靠地将结构化数据转换为图数据，并在 Neo4j 中建立正确的节点和关系。
"""
import pytest # 导入 pytest 测试框架
import pytest_asyncio # 导入 pytest 的异步扩展
from unittest.mock import patch, AsyncMock, call, MagicMock # 导入模拟工具
from pytest_mock import MockerFixture # 导入 MockerFixture 类型，用于 mocker fixture
from datetime import datetime, date # 导入日期时间类型
from typing import Optional, Tuple # 导入类型提示
import sys # 导入 sys 模块，用于修改 Python 路径
import os # 导入 os 模块，用于路径操作和环境变量
import json # 导入 json 模块，用于处理 JSON 数据（如 tags, authors）
import pprint # 导入 pprint 模块，用于美观地打印复杂数据结构（调试用）
import asyncio # 导入 asyncio 模块
import logging # 导入日志模块

# 添加：在模块加载时打印原始环境变量值的调试日志
logger_conftest_top = logging.getLogger("conftest_top") # 获取一个日志记录器实例
raw_test_neo4j_db_env = os.getenv("TEST_NEO4J_DATABASE") # 获取环境变量原始值
# 使用 CRITICAL 级别确保此日志在测试输出中可见
logger_conftest_top.critical(
    f"[CONTEST TOP] 原始 os.getenv('TEST_NEO4J_DATABASE'): '{raw_test_neo4j_db_env}'"
)

# 将此模块中的所有异步测试标记为使用会话作用域的事件循环
pytestmark = pytest.mark.asyncio(loop_scope="session")

# 将项目根目录添加到 Python 路径，以便能够导入 aigraphx 和 scripts 模块
# __file__ 指向当前测试文件路径
# os.path.dirname(__file__) 获取当前文件所在目录
# os.path.join(..., "..", "..") 向上两级目录，到达项目根目录 (Backend/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path: # 如果根目录不在 sys.path 中
    sys.path.insert(0, project_root) # 将其插入到列表开头，优先搜索

# 在路径修改后，导入需要测试的函数和需要模拟/使用的类
from scripts.sync_pg_to_neo4j import (
    run_sync, # 导入核心的同步运行函数
)

# 导入仓库类（用于类型提示和作为 Patch 目标）
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# 从 conftest.py 导入真实的仓库 fixtures
from tests.conftest import repository as postgres_repository_fixture # PG 仓库 fixture
from tests.conftest import neo4j_repo_fixture # Neo4j 仓库 fixture

# --- 测试数据常量 ---
# 定义脚本中可能使用的批处理大小常量（用于测试中的理解或潜在的 Patch）
PG_FETCH_BATCH_SIZE_TEST = 10  # 假设 PG 数据获取批次大小
NEO4J_WRITE_BATCH_SIZE_TEST = 5 # 假设 Neo4j 数据写入批次大小

# 为集成测试定义不同的 ID 和数据，避免与其他测试冲突
TEST_HF_MODEL_1 = { # 第一个测试模型数据
    "hf_model_id": "test-sync-hf-1",
    "hf_author": "sync_auth1",
    "hf_sha": "sync_sha1",
    "hf_last_modified": datetime(2023, 5, 1, 10, 0, 0), # datetime 对象
    "hf_tags": json.dumps(["syncA", "syncB"]), # 标签存储为 JSON 字符串
    "hf_pipeline_tag": "text-classification",
    "hf_downloads": 150,
    "hf_likes": 15,
    "hf_library_name": "transformers",
}
TEST_HF_MODEL_2 = { # 第二个测试模型数据
    "hf_model_id": "test-sync-hf-2",
    "hf_author": "sync_auth2",
    "hf_sha": "sync_sha2",
    "hf_last_modified": datetime(2023, 5, 2, 11, 0, 0),
    "hf_tags": None, # 标签可以为 None
    "hf_pipeline_tag": "image-generation",
    "hf_downloads": 250,
    "hf_likes": 25,
    "hf_library_name": "diffusers",
}
TEST_PAPER_1 = { # 第一个测试论文数据
    # paper_id 由数据库自动生成
    "pwc_id": "test-sync-pwc-1", # PapersWithCode ID
    "arxiv_id_base": "sync.1111",
    "arxiv_id_versioned": "sync.1111v1",
    "title": "Sync Paper 1",
    "authors": json.dumps(["Sync Auth 1", "Sync Auth 2"]), # 作者列表存储为 JSON 字符串
    "summary": "Sync summary 1",
    "published_date": date(2023, 5, 1), # date 对象
    "area": "ML",
    "pwc_url": "sync_url1",
    "pdf_url": "sync_pdf1",
    "doi": "sync_doi1",
    "primary_category": "cs.LG",
    "categories": json.dumps(["cs.LG", "cs.AI"]), # 分类列表存储为 JSON 字符串
}
TEST_PAPER_2 = { # 第二个测试论文数据
    # paper_id 由数据库自动生成
    "pwc_id": "test-sync-pwc-2",
    "arxiv_id_base": "sync.2222",
    "arxiv_id_versioned": "sync.2222v1",
    "title": "Sync Paper 2",
    "authors": None, # 作者可以为 None
    "summary": "Sync summary 2",
    "published_date": date(2023, 5, 10),
    "area": "Robotics",
    "pwc_url": "sync_url2",
    "pdf_url": "sync_pdf2",
    "doi": "sync_doi2",
    "primary_category": "cs.RO",
    "categories": json.dumps(["cs.RO"]),
}
# 定义模型和论文之间的关联关系（用于插入 model_paper_links 表）
TEST_LINK_1 = {"hf_model_id": "test-sync-hf-1", "pwc_id": "test-sync-pwc-1"}


# --- 辅助函数：在测试 PG 数据库中插入数据 --- #
async def insert_pg_data(repo: PostgresRepository) -> Tuple[int, int]:
    """
    将样本模型和论文数据插入到测试 PostgreSQL 数据库中。

    Args:
        repo: 连接到测试数据库的 PostgresRepository 实例。

    Returns:
        Tuple[int, int]: 插入的两篇论文的 paper_id。
    """
    paper1_id = -1 # 初始化 paper_id
    paper2_id = -1
    # 获取数据库连接
    async with repo.pool.connection() as conn:
        # 创建游标
        async with conn.cursor() as cur:
            # --- 插入论文 ---
            await cur.execute(
                # 插入语句，包含主要字段，并使用 RETURNING 获取生成的 paper_id
                "INSERT INTO papers (pwc_id, title, summary, published_date, area, authors) VALUES (%s, %s, %s, %s, %s, %s) RETURNING paper_id",
                (
                    TEST_PAPER_1["pwc_id"],
                    TEST_PAPER_1["title"],
                    TEST_PAPER_1["summary"],
                    TEST_PAPER_1["published_date"],
                    TEST_PAPER_1["area"],
                    # authors 是列表，需要转换为 JSON 字符串插入
                    json.dumps(TEST_PAPER_1["authors"]),
                ),
            )
            paper1_row = await cur.fetchone() # 获取返回的结果行
            if not paper1_row: # 检查是否成功插入
                raise ValueError("插入 TEST_PAPER_1 失败")
            paper1_id = paper1_row[0] # 提取 paper_id

            await cur.execute(
                "INSERT INTO papers (pwc_id, title, summary, published_date, area, authors) VALUES (%s, %s, %s, %s, %s, %s) RETURNING paper_id",
                (
                    TEST_PAPER_2["pwc_id"],
                    TEST_PAPER_2["title"],
                    TEST_PAPER_2["summary"],
                    TEST_PAPER_2["published_date"],
                    TEST_PAPER_2["area"],
                    # 如果 authors 为 None，json.dumps 会返回 'null' 字符串
                    json.dumps(TEST_PAPER_2["authors"]),
                ),
            )
            paper2_row = await cur.fetchone()
            if not paper2_row:
                raise ValueError("插入 TEST_PAPER_2 失败")
            paper2_id = paper2_row[0]

            # --- 插入模型 --- (包括 hf_library_name)
            await cur.execute(
                "INSERT INTO hf_models (hf_model_id, hf_author, hf_tags, hf_pipeline_tag, hf_library_name) VALUES (%s, %s, %s, %s, %s)",
                (
                    TEST_HF_MODEL_1["hf_model_id"],
                    TEST_HF_MODEL_1["hf_author"],
                    TEST_HF_MODEL_1["hf_tags"], # 传递 JSON 字符串
                    TEST_HF_MODEL_1["hf_pipeline_tag"],
                    TEST_HF_MODEL_1["hf_library_name"],
                ),
            )
            await cur.execute(
                "INSERT INTO hf_models (hf_model_id, hf_author, hf_tags, hf_pipeline_tag, hf_library_name) VALUES (%s, %s, %s, %s, %s)",
                (
                    TEST_HF_MODEL_2["hf_model_id"],
                    TEST_HF_MODEL_2["hf_author"],
                    TEST_HF_MODEL_2["hf_tags"], # 传递 None 或 JSON 字符串
                    TEST_HF_MODEL_2["hf_pipeline_tag"],
                    TEST_HF_MODEL_2["hf_library_name"],
                ),
            )

            # --- 插入模型-论文关联 ---
            # 修正：model_paper_links 表现在只包含 hf_model_id 和 paper_id
            await cur.execute(
                "INSERT INTO model_paper_links (hf_model_id, paper_id) VALUES (%s, %s)",
                (TEST_HF_MODEL_1["hf_model_id"], paper1_id), # 模型1 -> 论文1
            )
            await cur.execute(
                "INSERT INTO model_paper_links (hf_model_id, paper_id) VALUES (%s, %s)",
                (TEST_HF_MODEL_2["hf_model_id"], paper2_id), # 模型2 -> 论文2
            )
            # 额外插入一个关联，用于后续关系检查：模型1 -> 论文2
            await cur.execute(
                "INSERT INTO model_paper_links (hf_model_id, paper_id) VALUES (%s, %s)",
                (TEST_HF_MODEL_1["hf_model_id"], paper2_id),
            )

        # 提交事务，确保所有插入都生效
        await conn.commit()
        print(f"[DEBUG] insert_pg_data: Inserted papers with IDs {paper1_id}, {paper2_id}")
    return paper1_id, paper2_id # 返回生成的论文 ID


# --- 测试用例 --- #

@pytest.mark.asyncio # 标记为异步测试
async def test_run_sync_integration(
    postgres_repository_fixture: PostgresRepository,  # 请求真实的 PG 仓库 fixture
    neo4j_repo_fixture: Neo4jRepository,  # 请求真实的 Neo4j 仓库 fixture
) -> None:
    """集成测试：测试 sync_pg_to_neo4j 脚本的核心同步逻辑。"""
    # --- 准备 ---
    pg_repo = postgres_repository_fixture # 获取 PG 仓库实例
    neo4j_repo = neo4j_repo_fixture # 获取 Neo4j 仓库实例
    print(f"\n[DEBUG] test_run_sync_integration: Using PG pool: {pg_repo.pool}")
    print(f"[DEBUG] test_run_sync_integration: Using Neo4j driver: {neo4j_repo.driver}")


    # 1. 设置：向测试 PostgreSQL 数据库插入样本数据
    print("[DEBUG] test_run_sync_integration: Inserting test data into PG...")
    paper1_id, paper2_id = await insert_pg_data(pg_repo)
    print(f"[DEBUG] test_run_sync_integration: PG data inserted, paper IDs: {paper1_id}, {paper2_id}")


    # --- 脚本依赖假设 ---
    # 假设 run_sync() 函数接受 pg_repo 和 neo4j_repo 实例作为参数。
    # 如果脚本内部自己创建仓库实例，则需要确保它使用了测试环境的配置（由 conftest.py 设置）。

    # --- 执行 ---
    # 2. 调用脚本的核心同步逻辑，传入真实的仓库实例
    print("[DEBUG] test_run_sync_integration: Calling run_sync...")
    await run_sync(pg_repo=pg_repo, neo4j_repo=neo4j_repo)
    print("[DEBUG] test_run_sync_integration: run_sync finished.")


    # --- 断言 ---
    # 3. 连接到测试 Neo4j 数据库，验证数据是否已正确同步
    print("[DEBUG] test_run_sync_integration: Asserting Neo4j state...")
    # 使用 Neo4j 仓库实例获取一个会话
    # neo4j_repo_fixture 保证了测试数据库是干净的，并且约束已创建
    async with neo4j_repo.driver.session(database=neo4j_repo.db_name) as session:
        # --- 验证 HFModel 节点 ---
        # 检查模型 1 是否创建，并验证部分属性
        result_hf1 = await session.run(
            # Cypher 查询：匹配 HFModel 节点，通过 model_id 查找，返回 author 和 library_name
            "MATCH (m:HFModel {model_id: $id}) RETURN m.author as author, m.library_name as lib",
            id=TEST_HF_MODEL_1["hf_model_id"], # 传入参数
        )
        record_hf1 = await result_hf1.single() # 获取单个结果记录
        assert record_hf1 is not None, f"Neo4j: Model {TEST_HF_MODEL_1['hf_model_id']} not found"
        assert record_hf1["author"] == TEST_HF_MODEL_1["hf_author"]
        assert record_hf1["lib"] == TEST_HF_MODEL_1["hf_library_name"]

        # 检查模型 2
        result_hf2 = await session.run(
            "MATCH (m:HFModel {model_id: $id}) RETURN m.author as author",
            id=TEST_HF_MODEL_2["hf_model_id"],
        )
        record_hf2 = await result_hf2.single()
        assert record_hf2 is not None, f"Neo4j: Model {TEST_HF_MODEL_2['hf_model_id']} not found"
        assert record_hf2["author"] == TEST_HF_MODEL_2["hf_author"]

        # --- 验证 Paper 节点 ---
        # 检查论文 1
        result_p1 = await session.run(
            # Cypher 查询：匹配 Paper 节点，通过 pwc_id 查找，返回 title 和 area
            "MATCH (p:Paper {pwc_id: $id}) RETURN p.title as title, p.area as area",
            id=TEST_PAPER_1["pwc_id"],
        )
        record_p1 = await result_p1.single()
        assert record_p1 is not None, f"Neo4j: Paper {TEST_PAPER_1['pwc_id']} not found"
        assert record_p1["title"] == TEST_PAPER_1["title"]
        assert record_p1["area"] == TEST_PAPER_1["area"]

        # 检查论文 2
        result_p2 = await session.run(
            "MATCH (p:Paper {pwc_id: $id}) RETURN p.title as title",
            id=TEST_PAPER_2["pwc_id"],
        )
        record_p2 = await result_p2.single()
        assert record_p2 is not None, f"Neo4j: Paper {TEST_PAPER_2['pwc_id']} not found"
        assert record_p2["title"] == TEST_PAPER_2["title"]

        # --- 验证关系 ---
        # 检查模型 1 和论文 1 之间的 MENTIONS 关系是否存在
        # 脚本中关系类型从 USES_PAPER 改为 MENTIONS
        result_link = await session.run(
            # Cypher 查询：匹配从 HFModel 到 Paper 的 MENTIONS 关系，返回关系数量
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_LINK_1["hf_model_id"], # 模型 ID
            pwc_id=TEST_LINK_1["pwc_id"], # 论文 PWC ID
        )
        record_link = await result_link.single()
        assert record_link is not None, "Neo4j: Relationship check M1->P1 failed"
        assert record_link["c"] == 1, f"Expected 1 M1->P1 MENTIONS relationship, found {record_link['c']}"

        # === 开始：增强的关系断言 ===
        # 在 PG 数据库中为论文 1 插入额外的关系数据（任务、数据集、仓库）
        # 然后再次运行同步脚本，并验证这些新关系是否已在 Neo4j 中创建。

        # 验证模型 1 -> 论文 2 的关系是否存在 (这个关系是在 insert_pg_data 中添加的)
        result_link_m1p2 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_1["hf_model_id"], # 模型 1
            pwc_id=TEST_PAPER_2["pwc_id"], # 论文 2
        )
        record_link_m1p2 = await result_link_m1p2.single()
        assert record_link_m1p2 is not None, (
            f"未找到关系 M1->P2：{TEST_HF_MODEL_1['hf_model_id']} -> {TEST_PAPER_2['pwc_id']}"
        )
        assert record_link_m1p2["c"] == 1, (
            f"预期找到 1 个 M1->P2 关系，实际找到 {record_link_m1p2['c']} 个"
        )

        # 验证模型 2 -> 论文 2 的关系是否存在
        result_link_m2p2 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_2["hf_model_id"], # 模型 2
            pwc_id=TEST_PAPER_2["pwc_id"], # 论文 2
        )
        record_link_m2p2 = await result_link_m2p2.single()
        assert record_link_m2p2 is not None, (
            f"未找到关系 M2->P2：{TEST_HF_MODEL_2['hf_model_id']} -> {TEST_PAPER_2['pwc_id']}"
        )
        assert record_link_m2p2["c"] == 1, (
            f"预期找到 1 个 M2->P2 关系，实际找到 {record_link_m2p2['c']} 个"
        )

        # --- 插入额外的 PWC 关系数据到 PG ---
        print("[DEBUG] test_run_sync_integration: Inserting PWC relations into PG for Paper 1...")
        async with pg_repo.pool.connection() as pg_conn:
            async with pg_conn.cursor() as pg_cur:
                # 为 paper1_id 插入两个任务
                await pg_cur.execute(
                    "INSERT INTO pwc_tasks (paper_id, task_name) VALUES (%s, %s), (%s, %s)",
                    (paper1_id, "Text Summarization", paper1_id, "Question Answering"),
                )
                # 为 paper1_id 插入一个数据集
                await pg_cur.execute(
                    "INSERT INTO pwc_datasets (paper_id, dataset_name) VALUES (%s, %s)",
                    (paper1_id, "SQuAD"),
                )
                # 为 paper1_id 插入一个代码仓库
                await pg_cur.execute(
                    "INSERT INTO pwc_repositories (paper_id, url) VALUES (%s, %s)",
                    (paper1_id, "https://github.com/sync/test-repo"),
                )
            await pg_conn.commit() # 提交事务
        print("[DEBUG] test_run_sync_integration: PWC relations inserted.")

        # --- 再次运行同步脚本以处理新的关系数据 ---
        # 注意：理想情况下，同步脚本应该是幂等的（多次运行结果相同）。
        # 但为了确保测试覆盖到关系处理，这里再次运行。
        # 如果同步很慢，可以考虑在第一次 run_sync 之前插入所有数据。
        print("\n--- 再次运行同步脚本以处理关系 ---")
        await run_sync(pg_repo=pg_repo, neo4j_repo=neo4j_repo)
        print("--- 第二次同步运行完成 ---")

        # --- 现在验证新创建的关系和节点是否存在于 Neo4j 中 ---
        print("[DEBUG] test_run_sync_integration: Asserting PWC relations in Neo4j...")
        # 检查论文 1 -> 任务 (Task) 的关系
        result_p1_tasks = await session.run(
            """
            MATCH (p:Paper {pwc_id: $pwc_id})-[:HAS_TASK]->(t:Task)
            RETURN count(t) as task_count, collect(t.name) as task_names
            """, # 返回任务数量和任务名称列表
            pwc_id=TEST_PAPER_1["pwc_id"], # 查询论文 1
        )
        record_p1_tasks = await result_p1_tasks.single()
        assert record_p1_tasks is not None, "Neo4j: 论文 1 的任务关系检查失败"
        assert record_p1_tasks["task_count"] == 2, (
            f"预期论文 1 有 2 个任务，实际找到 {record_p1_tasks['task_count']} 个"
        )
        # 比较任务名称列表（忽略顺序）
        assert sorted(record_p1_tasks["task_names"]) == sorted(
            ["Text Summarization", "Question Answering"]
        )

        # 检查论文 1 -> 数据集 (Dataset) 的关系
        result_p1_datasets = await session.run(
            """
            MATCH (p:Paper {pwc_id: $pwc_id})-[:USES_DATASET]->(d:Dataset)
            RETURN count(d) as dataset_count, d.name as dataset_name
            """, # 返回数据集数量和名称
            pwc_id=TEST_PAPER_1["pwc_id"],
        )
        record_p1_datasets = await result_p1_datasets.single()
        assert record_p1_datasets is not None, "Neo4j: 论文 1 的数据集关系检查失败"
        assert record_p1_datasets["dataset_count"] == 1, (
            f"预期论文 1 有 1 个数据集，实际找到 {record_p1_datasets['dataset_count']} 个"
        )
        assert record_p1_datasets["dataset_name"] == "SQuAD"

        # 检查论文 1 -> 代码仓库 (Repository) 的关系
        result_p1_repos = await session.run(
            """
            MATCH (p:Paper {pwc_id: $pwc_id})-[:HAS_REPOSITORY]->(r:Repository)
            RETURN count(r) as repo_count, r.url as repo_url
            """, # 返回仓库数量和 URL
            pwc_id=TEST_PAPER_1["pwc_id"],
        )
        record_p1_repos = await result_p1_repos.single()
        assert record_p1_repos is not None, "Neo4j: 论文 1 的代码仓库关系检查失败"
        assert record_p1_repos["repo_count"] == 1, (
            f"预期论文 1 有 1 个代码仓库，实际找到 {record_p1_repos['repo_count']} 个"
        )
        assert record_p1_repos["repo_url"] == "https://github.com/sync/test-repo"

        # === 结束：增强的关系断言 ===

        # （可选）检查之前验证过的其他 MENTIONS 关系是否仍然存在
        # 检查模型 1 -> 论文 2
        result_link_2 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_1["hf_model_id"],
            pwc_id=TEST_PAPER_2["pwc_id"],
        )
        record_link_2 = await result_link_2.single()
        assert record_link_2 is not None, "Neo4j: Relationship M1->P2 check failed after second sync"
        assert record_link_2["c"] == 1

        # 检查模型 2 -> 论文 2
        result_link_3 = await session.run(
            "MATCH (m:HFModel {model_id: $model_id})-[r:MENTIONS]->(p:Paper {pwc_id: $pwc_id}) RETURN count(r) as c",
            model_id=TEST_HF_MODEL_2["hf_model_id"],
            pwc_id=TEST_PAPER_2["pwc_id"],
        )
        record_link_3 = await result_link_3.single()
        assert record_link_3 is not None, "Neo4j: Relationship M2->P2 check failed after second sync"
        assert record_link_3["c"] == 1

    print("[DEBUG] test_run_sync_integration: Neo4j assertions passed.")
    # 注意：neo4j_repo_fixture 会在测试结束后自动清理 Neo4j 测试数据库并确保约束存在。


# TODO: 如果可能，在集成测试上下文中添加错误处理测试
# （例如，模拟 PG 在获取数据时连接错误 - 可能需要 patch PG 仓库实例的方法）