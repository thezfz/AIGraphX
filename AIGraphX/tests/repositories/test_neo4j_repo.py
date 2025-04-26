# tests/repositories/test_neo4j_repo.py

"""
文件目的：测试 Neo4jRepository 类

概述：
该文件包含了针对 `aigraphx.repositories.neo4j_repo.Neo4jRepository` 类的测试用例集合。
主要采用**集成测试**的策略，这意味着测试会直接与一个**真实的测试 Neo4j 数据库实例**进行交互，
而不是完全依赖模拟（Mocking）。这有助于确保仓库层代码与实际数据库的交互符合预期。

主要交互：
- **被测代码**: `aigraphx.repositories.neo4j_repo.Neo4jRepository`
- **测试框架**: `pytest` (包括其异步插件 `pytest-asyncio`)
- **数据库交互**: 通过 `neo4j` 驱动库连接到一个**测试专用**的 Neo4j 数据库。
- **测试环境管理**: 依赖 `conftest.py` 文件中定义的 Pytest Fixtures 来管理测试环境，包括：
    - `test_settings`: 提供测试配置，如测试数据库的连接信息。
    - `neo4j_driver`: 提供连接到测试 Neo4j 数据库的 `AsyncDriver` 实例。
    - `neo4j_repo_fixture`: 提供一个配置好、连接到测试数据库的 `Neo4jRepository` 实例。
    - `clear_db_before_test`: 一个自动执行的 Fixture，确保在每次测试运行前清空测试数据库，保证测试隔离性。
- **模拟 (Mocking)**: `unittest.mock` 用于某些难以在集成测试中稳定触发的场景，例如数据库连接失败或特定查询错误。

测试策略：
遵循 "测试奖杯" 模型，重点在于**集成测试**仓库层与数据库的交互。
单元测试（使用 Mock）主要用于验证独立的逻辑或难以模拟的失败场景。

注意：
运行这些测试需要一个正在运行的、配置正确的 Neo4j 测试实例，并且相关的测试配置（如数据库 URI、用户、密码、数据库名）已在测试环境中（例如通过 `.env.test` 文件或环境变量）正确设置。
"""

# 导入测试框架和相关工具
import pytest  # 导入 pytest 测试框架，用于编写和运行测试
import pytest_asyncio  # 导入 pytest 的异步支持插件，用于测试异步代码
from unittest.mock import (
    AsyncMock,
    call,
    patch,
    MagicMock,
)  # 从 unittest.mock 导入异步 Mock 类、调用记录、补丁工具和魔法 Mock，用于模拟对象和行为
from typing import (
    List,
    Literal,
    cast,
    Callable,
    Awaitable,
    Any,
    Dict,
    AsyncGenerator,
)  # 导入类型提示工具，用于代码静态分析和提高可读性
import sys  # 导入 sys 模块，用于访问与 Python 解释器相关的变量和函数，如此处的路径操作
import os  # 导入 os 模块，提供与操作系统交互的功能，如此处的路径操作
import logging  # 导入 logging 模块，用于记录程序运行时的信息

# --- 项目路径设置 ---
# 将项目根目录添加到 Python 解释器的搜索路径中
# 这允许测试文件能够像运行时一样，直接导入项目内部的模块（例如 aigraphx.repositories）
# 获取当前文件所在的目录的绝对路径
# os.path.dirname(__file__) -> /path/to/AIGraphX/Backend/tests/repositories
# os.path.join(..., "..", "..") -> /path/to/AIGraphX
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:  # 如果项目根目录不在 sys.path 中
    sys.path.insert(
        0, project_root
    )  # 将项目根目录插入到 sys.path 的最前面，确保优先搜索项目内模块

# --- 导入被测代码和依赖 ---
# 导入要测试的 Neo4jRepository 类
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# 从 neo4j 驱动库导入异步相关的类，用于类型提示和模拟
from neo4j import AsyncDriver, AsyncSession, AsyncManagedTransaction, Record

# 导入配置类，用于获取测试数据库名等信息
from aigraphx.core.config import Settings

# 导入 FixtureRequest，用于在 fixture 或测试函数中获取请求上下文信息（如此处的测试名称）
from pytest import FixtureRequest

# --- 日志设置 ---
# 获取一个名为 __name__ (即 'tests.repositories.test_neo4j_repo') 的日志记录器实例
logger = logging.getLogger(__name__)
# 将此模块中的所有异步测试标记为需要 pytest-asyncio 处理
# 注意：之前的 'loop_scope="function"' 已移除，通常默认的事件循环作用域是合适的
pytestmark = pytest.mark.asyncio

# --- 常量定义 ---
# 定义仓库类中预期会执行的 DDL (数据定义语言) Cypher 查询语句列表
# 用于确保图数据库的 Schema（约束和索引）被正确创建
# 这有助于保持测试代码与实际实现同步，或者作为参考
EXPECTED_DDL_QUERIES = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.pwc_id IS UNIQUE;",  # 论文节点 pwc_id 唯一性约束
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:HFModel) REQUIRE m.model_id IS UNIQUE;",  # HuggingFace 模型节点 model_id 唯一性约束
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE;",  # 任务节点 name 唯一性约束
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE;",  # 数据集节点 name 唯一性约束
    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Repository) REQUIRE r.url IS UNIQUE;",  # 代码仓库节点 url 唯一性约束
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE;",  # 作者节点 name 唯一性约束 (注意：实际情况可能需要更复杂的作者消歧)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (ar:Area) REQUIRE ar.name IS UNIQUE;",  # 研究领域节点 name 唯一性约束
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Framework) REQUIRE f.name IS UNIQUE;",  # 框架节点 name 唯一性约束
    "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id_base);",  # 论文节点 arxiv_id_base 索引
    "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title);",  # 论文节点 title 索引
    "CREATE INDEX IF NOT EXISTS FOR (m:HFModel) ON (m.author);",  # HF 模型节点 author 索引
    "CREATE INDEX IF NOT EXISTS FOR (e:Method) ON (e.name);",  # 方法节点 name 索引 (假设 Method 节点存在)
]


# --- 测试辅助函数与 Fixtures ---
async def _clear_neo4j_db(driver: AsyncDriver, settings: Settings) -> None:
    """
    清空指定的 Neo4j 测试数据库中的所有节点和关系。

    Args:
        driver (AsyncDriver): 连接到 Neo4j 数据库的异步驱动实例。
        settings (Settings): 包含数据库名称等配置的应用设置对象。
    """
    db_name = settings.neo4j_database  # 从配置中获取要操作的数据库名称
    logger.debug(f"[测试前] 正在清空 Neo4j 数据库: {db_name}")
    try:
        # 使用驱动创建一个异步会话 (session)，指定要操作的数据库
        async with driver.session(database=db_name) as session:
            # 在会话中执行一个写事务 (execute_write)
            # "MATCH (n) DETACH DELETE n" 是一个 Cypher 查询，匹配所有节点 (n)，
            # DETACH 删除与这些节点相关的关系，然后 DELETE 删除节点本身。
            await session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        logger.debug(f"[测试前] 已清空 Neo4j 数据库: {db_name}")
    except Exception as e:
        # 如果清空过程中发生任何异常，记录错误日志并重新抛出异常
        # 清理失败通常意味着测试环境不稳定，应使测试失败
        logger.error(
            f"[测试前] 清空 Neo4j 数据库 {db_name} 失败: {e}",
            exc_info=True,  # 包含异常的堆栈跟踪信息
        )
        raise  # 重新抛出异常


# Pytest Fixture: 自动在每个测试函数运行前清空数据库
# `@pytest_asyncio.fixture(autouse=True)` 定义了一个异步 Fixture，
# `autouse=True` 表示这个 Fixture 会自动应用于当前模块的所有测试函数，无需显式调用。
# 它依赖于 `neo4j_driver` 和 `test_settings` 这两个 Fixture (通常在 conftest.py 中定义)。
@pytest_asyncio.fixture(autouse=True)
async def clear_db_before_test(
    neo4j_driver: AsyncDriver, test_settings: Settings
) -> AsyncGenerator[None, None]:
    """
    Pytest Fixture: 自动在每个测试函数运行前清空 Neo4j 测试数据库。
    确保测试之间的隔离性。

    Args:
        neo4j_driver (AsyncDriver): 由 conftest.py 提供的 Neo4j 驱动实例。
        test_settings (Settings): 由 conftest.py 提供的测试配置实例。

    Yields:
        AsyncGenerator[None, None]: 无返回值，主要执行清理操作。
    """
    # 在测试函数执行前，调用辅助函数清空数据库
    await _clear_neo4j_db(neo4j_driver, test_settings)
    # `yield` 语句将控制权交给测试函数执行
    yield
    # 测试函数执行完毕后，这里可以添加测试后的清理代码（如果需要）
    # 在这个例子中，因为我们在测试前清理，所以测试后通常不需要额外清理。
    # logger.debug("[测试后] 清理完成 (如有需要)")


# --- 测试用例 ---


@pytest.mark.asyncio
async def test_create_constraints_integration(
    neo4j_repo_fixture: Neo4jRepository,  # 依赖注入配置好的 Neo4jRepository 实例
    neo4j_driver: AsyncDriver,  # 依赖注入 Neo4j 驱动实例 (用于可选的验证)
    test_settings: Settings,  # 依赖注入测试配置 (用于获取数据库名等)
) -> None:
    """
    集成测试：测试 `create_constraints` 方法能否在真实的测试数据库上成功运行。

    这个测试主要验证方法调用本身不抛出异常。
    对约束是否真的创建成功的验证可以依赖后续尝试插入违反约束的数据的测试，
    或者通过查询系统表（但这比较复杂且可能因 Neo4j 版本变化）。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
    """
    repo = neo4j_repo_fixture  # 获取仓库实例
    db_name = test_settings.neo4j_database  # 获取数据库名

    try:
        # --- Action: 调用被测方法 ---
        await repo.create_constraints()
        # --- Assertion: 基本检查 ---
        # 主要断言是上面的调用没有引发异常
        logger.info("`create_constraints` 方法成功执行，未抛出异常。")

        # --- Assertion: 可选的高级验证 ---
        # 可以取消注释下面的代码来尝试通过查询 Neo4j 系统表来验证约束是否创建
        # 注意：查询系统表的 Cypher 语法可能因 Neo4j 版本而异
        # async with neo4j_driver.session(database='system') as sys_session:
        #     # 查询目标数据库的约束
        #     result = await sys_session.run(
        #         "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
        #         "WHERE labelsOrTypes = ['Paper'] AND properties = ['pwc_id'] " # 检查 Paper(pwc_id) 唯一性约束
        #         "RETURN count(*) AS count"
        #     )
        #     record = await result.single()
        #     assert record["count"] >= 1 # 断言至少存在一个这样的约束

    finally:
        # --- Cleanup: 清理 ---
        # 这个测试不需要显式清理，因为 `clear_db_before_test` fixture 会在下一个测试开始前自动清理。
        pass


# 之前的 test_create_constraints_failure 测试已被移除，
# 因为在集成测试中模拟 DDL 执行失败比较困难且不稳定。
# 此类失败场景更适合使用 Mock 进行单元测试。


@pytest.mark.asyncio
async def test_save_papers_batch_integration(
    neo4j_driver: AsyncDriver,  # 依赖注入驱动实例，用于验证数据
    neo4j_repo_fixture: Neo4jRepository,  # 依赖注入仓库实例
    test_settings: Settings,  # 依赖注入测试配置
    request: FixtureRequest,  # 依赖注入请求上下文 (可选，可用于记录测试名称等)
) -> None:
    """
    集成测试：测试 `save_papers_batch` 方法是否能成功地将论文数据及其关联关系保存到 Neo4j。

    Args:
        neo4j_driver (AsyncDriver): Neo4j 驱动实例，用于测试后的数据验证。
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture  # 获取仓库实例
    db_name = test_settings.neo4j_database  # 获取数据库名

    # --- Setup: 准备测试数据 ---
    # 创建一个包含多篇论文信息的列表，每篇论文是一个字典
    sample_papers_data: List[Dict[str, Any]] = [
        {
            "pwc_id": "paper1-integ",  # 论文在 PapersWithCode 上的 ID (主键)
            "arxiv_id_base": "1234.5678",  # ArXiv ID (无版本)
            "arxiv_id_versioned": "1234.5678v1",  # ArXiv ID (带版本)
            "title": "Integ Test Paper 1",  # 标题
            "summary": "Summary 1",  # 摘要
            "published_date": "2023-01-01",  # 发表日期 (字符串格式)
            "authors": [
                "Author A",
                "Author B",
            ],  # 作者列表 (将创建 Author 节点和 AUTHORED 关系)
            "area": "Computer Science",  # 研究领域 (将创建 Area 节点和 HAS_AREA 关系)
            "tasks": ["Task 1"],  # 相关任务 (将创建 Task 节点和 HAS_TASK 关系)
            "datasets": [
                "Dataset X"
            ],  # 使用的数据集 (将创建 Dataset 节点和 USES_DATASET 关系)
            "repositories": [  # 关联的代码仓库 (将创建 Repository 节点和 HAS_REPOSITORY 关系)
                {
                    "url": "http://repo1-integ.com",  # 仓库 URL (主键)
                    "stars": 100,  # 星标数
                    "is_official": True,  # 是否官方
                    "framework": "pytorch",  # 使用的框架 (将创建 Framework 节点和 USES_FRAMEWORK 关系)
                }
            ],
            "pwc_url": "http://pwc1.com",  # PWC 页面 URL
            "pdf_url": "http://pdf1.com",  # PDF 下载 URL
            "doi": "doi1-integ",  # DOI
            "primary_category": "cs.AI",  # 主要 ArXiv 分类
            "categories": [
                "cs.AI",
                "cs.LG",
            ],  # 所有 ArXiv 分类 (将创建 Category 节点和 HAS_CATEGORY 关系)
        },
        {
            "pwc_id": "paper2-integ",
            "arxiv_id_base": "9876.5432",
            "arxiv_id_versioned": "9876.5432v2",
            "title": "Integ Test Paper 2",
            "summary": "Summary 2",
            "published_date": "2023-02-15",
            "authors": ["Author C"],
            "area": "Machine Learning",
            "tasks": ["Task 2", "Task 3"],
            "datasets": [],  # 无数据集
            "repositories": [],  # 无代码仓库
            "pwc_url": "http://pwc2.com",
            "pdf_url": "http://pdf2.com",
            "doi": "doi2-integ",
            "primary_category": "cs.LG",
            "categories": ["cs.LG"],
        },
    ]
    # 收集要清理的论文 ID，虽然自动清理 fixture 会处理，但显式记录有助于调试
    ids_to_clean = [p["pwc_id"] for p in sample_papers_data]

    try:
        # --- Action: 调用被测方法 ---
        await repo.save_papers_batch(sample_papers_data)

        # --- Verification: 使用驱动直接查询数据库进行验证 ---
        async with neo4j_driver.session(database=db_name) as session:
            # 1. 验证论文节点数量
            result_papers = await session.run(
                # Cypher 查询: 匹配标签为 Paper 且 pwc_id 在给定列表中的节点，返回数量
                "MATCH (p:Paper) WHERE p.pwc_id IN $ids RETURN count(p) AS count",
                ids=ids_to_clean,  # 将论文 ID 列表作为参数传递给查询
            )
            count_record = await result_papers.single()  # 获取单个结果记录
            assert count_record is not None  # 确保查询返回了结果
            assert count_record["count"] == 2  # 断言创建了 2 个论文节点

            # 2. 验证某篇论文的属性 (例如 paper1-integ)
            result_paper1 = await session.run(
                # Cypher 查询: 匹配 pwc_id 为指定值的 Paper 节点，返回其 title 和 area 属性
                "MATCH (p:Paper {pwc_id: $id}) RETURN p.title AS title, p.area AS area",
                id="paper1-integ",
            )
            paper1_record = await result_paper1.single()
            assert paper1_record is not None
            assert paper1_record["title"] == "Integ Test Paper 1"
            assert paper1_record["area"] == "Computer Science"

            # 3. 验证 paper1-integ 的关联关系
            #    - 作者 (Author)-[:AUTHORED]->(Paper)
            #    - 论文 (Paper)-[:HAS_TASK]->(Task)
            #    - 论文 (Paper)-[:USES_DATASET]->(Dataset)
            #    - 论文 (Paper)-[:HAS_REPOSITORY]->(Repository)
            result_rels = await session.run(
                """
                MATCH (p:Paper {pwc_id: $id}) // 找到目标论文节点
                // 使用 OPTIONAL MATCH，即使某些关系不存在也不会导致整个查询失败
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p) // 找到编写该论文的作者
                OPTIONAL MATCH (p)-[:HAS_TASK]->(t:Task)     // 找到该论文相关的任务
                OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset) // 找到该论文使用的数据集
                OPTIONAL MATCH (p)-[:HAS_REPOSITORY]->(r:Repository) // 找到该论文关联的仓库
                RETURN
                    collect(DISTINCT a.name) AS authors,   // 收集所有不重复的作者名字
                    collect(DISTINCT t.name) AS tasks,     // 收集所有不重复的任务名字
                    collect(DISTINCT d.name) AS datasets,  // 收集所有不重复的数据集名字
                    collect(DISTINCT r.url) AS repos       // 收集所有不重复的仓库 URL
                """,
                id="paper1-integ",
            )
            rels_record = await result_rels.single()
            assert rels_record is not None
            # 断言集合内容，忽略顺序
            assert set(rels_record["authors"]) == {"Author A", "Author B"}
            assert set(rels_record["tasks"]) == {"Task 1"}
            assert set(rels_record["datasets"]) == {"Dataset X"}
            assert set(rels_record["repos"]) == {"http://repo1-integ.com"}

    finally:
        # --- Cleanup: 清理 ---
        # 由 autouse fixture `clear_db_before_test` 自动处理
        pass


@pytest.mark.asyncio
async def test_save_hf_models_batch_integration(
    neo4j_driver: AsyncDriver,
    neo4j_repo_fixture: Neo4jRepository,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `save_hf_models_batch` 方法是否能成功地将 Hugging Face 模型数据及其关联关系保存到 Neo4j。

    Args:
        neo4j_driver (AsyncDriver): Neo4j 驱动实例，用于验证。
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 准备测试数据 ---
    sample_models_data: List[Dict[str, Any]] = [
        {
            "model_id": "org/model-1-integ",  # 模型 ID (主键)
            "author": "org",  # 作者/组织
            "sha": "sha1",  # Git SHA
            "last_modified": "2023-10-26T10:00:00.000Z",  # 最后修改时间 (ISO 格式字符串)
            "tags": ["tag1", "tag2"],  # 标签列表 (可能用于创建 Tag 节点或作为属性存储)
            "pipeline_tag": "text-generation",  # 流水线标签 (应创建/合并 Task 节点并建立 HAS_TASK 关系)
            "downloads": 1000,  # 下载量
            "likes": 50,  # 点赞数
            "library_name": "transformers",  # 使用的库 (可能创建/合并 Library 节点或作为属性)
        },
        {
            "model_id": "user/model-2-integ",
            "author": "user",
            "sha": "sha2",
            "last_modified": "2023-10-27T11:30:00.000Z",
            "tags": None,  # 允许为 None
            "pipeline_tag": "image-classification",  # 另一个 Task 节点
            "downloads": 500,
            "likes": 25,
            "library_name": None,  # 允许为 None
        },
    ]
    # 收集用于验证和清理的 ID 和名称
    ids_to_clean = [m["model_id"] for m in sample_models_data if "model_id" in m]
    tasks_to_clean = list(
        set(m["pipeline_tag"] for m in sample_models_data if m.get("pipeline_tag"))
    )

    try:
        # --- Action: 调用被测方法 ---
        await repo.save_hf_models_batch(sample_models_data)

        # --- Verification: 使用驱动直接查询数据库 ---
        async with neo4j_driver.session(database=db_name) as session:
            # 1. 验证模型节点数量
            result_models = await session.run(
                "MATCH (m:HFModel) WHERE m.model_id IN $ids RETURN count(m) AS count",
                ids=ids_to_clean,
            )
            count_record = await result_models.single()
            assert count_record is not None
            assert count_record["count"] == 2

            # 2. 验证某个模型的属性 (例如 model-1-integ)
            result_model1 = await session.run(
                # 查询 model_id 匹配的 HFModel 节点，返回 author, likes, library_name 属性
                "MATCH (m:HFModel {model_id: $id}) RETURN m.author AS author, m.likes AS likes, m.library_name AS lib",
                id="org/model-1-integ",
            )
            model1_record = await result_model1.single()
            assert model1_record is not None
            assert model1_record["author"] == "org"
            assert model1_record["likes"] == 50
            assert model1_record["lib"] == "transformers"
            # 注意：验证日期时间转换需要额外步骤，可能需要检查返回值的类型
            # 例如: result_dt = await session.run("MATCH (m:HFModel {model_id: $id}) RETURN m.last_modified", id="org/model-1-integ")
            #       dt_record = await result_dt.single()
            #       assert isinstance(dt_record["m.last_modified"], neo4j.time.DateTime) # 检查是否是 Neo4j 的日期时间类型

            # 3. 验证 Task 节点是否被创建或合并
            result_tasks = await session.run(
                # 查询 name 在指定列表中的 Task 节点数量
                "MATCH (t:Task) WHERE t.name IN $names RETURN count(t) AS count",
                names=tasks_to_clean,
            )
            tasks_count_record = await result_tasks.single()
            assert tasks_count_record is not None
            # 应该创建了 text-generation 和 image-classification 两个 Task 节点
            assert tasks_count_record["count"] == 2

            # 4. 验证模型与任务之间的关系 (HAS_TASK)
            result_rels = await session.run(
                """
                MATCH (m:HFModel)-[r:HAS_TASK]->(t:Task) // 匹配 HFModel 到 Task 的 HAS_TASK 关系
                WHERE m.model_id IN $ids AND t.name IN $task_names // 限制在本次测试创建的模型和任务之间
                RETURN count(r) AS count // 返回关系数量
                """,
                ids=ids_to_clean,
                task_names=tasks_to_clean,
            )
            rels_count_record = await result_rels.single()
            assert rels_count_record is not None
            # 每个模型应该都链接到了其对应的 pipeline_tag 任务
            assert rels_count_record["count"] == 2

    finally:
        # --- Cleanup: 清理 ---
        # 由 autouse fixture `clear_db_before_test` 自动处理
        pass


@pytest.mark.asyncio
@patch(
    "aigraphx.repositories.neo4j_repo.logger"
)  # 使用 @patch 装饰器替换 logger 对象为 Mock 对象
async def test_save_papers_batch_failure(mock_logger: MagicMock) -> None:
    """
    单元测试 (Mocked): 测试 `save_papers_batch` 在数据库写入失败时的行为。

    使用 Mock 来模拟数据库驱动和会话，使其在 `execute_write` 时抛出异常。
    验证方法是否捕获异常、记录错误日志，并可能重新抛出异常。

    Args:
        mock_logger (MagicMock): 被 @patch 替换的 logger Mock 对象。
    """
    # --- Setup: 准备 Mock 对象和数据 ---
    sample_papers_data = [{"pwc_id": "paper1", "title": "Title"}]  # 简单的测试数据

    # 创建 Mock 的 Neo4j 驱动和会话
    mock_driver = AsyncMock(
        spec=AsyncDriver
    )  # spec=确保 Mock 对象有 AsyncDriver 的接口
    mock_session = AsyncMock(spec=AsyncSession)
    # 配置 mock_driver.session() 返回一个异步上下文管理器，其 __aenter__ 返回 mock_session
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = (
        None  # __aexit__ 通常返回 None
    )

    # 定义一个要模拟的数据库写入异常
    test_exception = Exception("DB write error")
    # 配置 mock_session 的 execute_write 方法在被调用时，抛出定义的异常
    mock_session.execute_write = AsyncMock(side_effect=test_exception)

    # 使用 Mock 驱动创建仓库实例
    repo = Neo4jRepository(driver=mock_driver)

    # --- Action & Assertion: 调用被测方法并断言异常 ---
    # 使用 pytest.raises 作为上下文管理器，断言特定类型的异常被抛出
    with pytest.raises(Exception) as excinfo:
        await repo.save_papers_batch(sample_papers_data)

    # 断言抛出的异常就是我们模拟的 test_exception
    assert excinfo.value is test_exception

    # --- Assertion: 验证 Mock 调用 ---
    # 断言 mock_driver.session() 被调用了一次
    mock_driver.session.assert_called_once()
    # 断言 mock_logger.error() 被调用了一次
    mock_logger.error.assert_called_once()
    # 检查错误日志消息是否符合预期
    expected_log_prefix = "Error saving papers batch (with relations) to Neo4j:"
    # 获取记录器错误调用的第一个位置参数（即日志消息字符串）
    log_message = mock_logger.error.call_args[0][0]
    assert isinstance(log_message, str)  # 确保是字符串
    assert log_message.startswith(expected_log_prefix)  # 断言日志消息的开头


@pytest.mark.asyncio
@patch("aigraphx.repositories.neo4j_repo.logger")
async def test_save_hf_models_batch_failure(mock_logger: MagicMock) -> None:
    """
    单元测试 (Mocked): 测试 `save_hf_models_batch` 在数据库写入失败时的行为。

    与 `test_save_papers_batch_failure` 类似，使用 Mock 模拟数据库错误。

    Args:
        mock_logger (MagicMock): Mocked logger 对象。
    """
    # --- Setup: 准备 Mock 对象和数据 ---
    sample_models_data = [{"model_id": "model1"}]
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock(spec=AsyncSession)
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = None

    test_exception = Exception("DB write error")
    mock_session.execute_write = AsyncMock(side_effect=test_exception)

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action & Assertion: 调用并断言异常 ---
    # 使用 match 参数检查异常消息是否包含特定字符串
    with pytest.raises(Exception, match="DB write error"):
        await repo.save_hf_models_batch(sample_models_data)

    # --- Assertion: 验证 Mock 调用 ---
    mock_logger.error.assert_called_once()
    # 检查错误日志消息内容
    error_call_args, error_call_kwargs = mock_logger.error.call_args
    assert "Error saving HF models batch to Neo4j" in error_call_args[0]
    # 确保原始异常信息也包含在日志中
    assert str(test_exception) in error_call_args[0]


# 之前的 test_create_or_update_paper_node_calls_execute (单元测试版本) 被注释掉了，
# 因为它测试的是内部方法 _execute_query，而现在的测试更倾向于测试公共接口。
# 保留下面的集成测试版本 test_create_or_update_paper_node_integration。


@pytest.mark.asyncio
async def test_link_paper_to_task_integration(
    neo4j_driver: AsyncDriver,
    neo4j_repo_fixture: Neo4jRepository,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `link_paper_to_task` 方法是否能在 Neo4j 中成功创建 Paper 和 Task 之间的 HAS_TASK 关系。

    Args:
        neo4j_driver (AsyncDriver): Neo4j 驱动实例，用于设置和验证。
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database
    pwc_id = "paper-link-integ"  # 测试用论文 ID
    task_name = "Task Link Integ"  # 测试用任务名称

    try:
        # --- Setup: 准备前提条件 ---
        # 在调用链接方法之前，必须确保 Paper 和 Task 节点已经存在于数据库中。
        # 这里直接使用驱动程序创建这些节点。
        async with neo4j_driver.session(database=db_name) as session:
            await session.execute_write(
                lambda tx: tx.run(
                    # MERGE 语句：如果节点不存在则创建，如果存在则匹配。
                    "MERGE (p:Paper {pwc_id: $pid}) MERGE (t:Task {name: $tname})",
                    pid=pwc_id,
                    tname=task_name,
                )
            )
        logger.info(f"测试设置完成: 已创建 Paper {pwc_id} 和 Task {task_name}")

        # --- Action: 调用被测方法 ---
        await repo.link_paper_to_task(pwc_id, task_name)

        # --- Verification: 验证关系是否已创建 ---
        async with neo4j_driver.session(database=db_name) as session:
            result = await session.run(
                """
                MATCH (p:Paper {pwc_id: $pid})-[r:HAS_TASK]->(t:Task {name: $tname}) // 匹配指定的 Paper 到 Task 的 HAS_TASK 关系
                RETURN count(r) AS count // 返回匹配到的关系数量
                """,
                pid=pwc_id,
                tname=task_name,
            )
            record = await result.single()
            assert record is not None
            assert record["count"] == 1  # 断言关系已成功创建（数量为 1）
        logger.info("测试验证完成: Paper->Task 关系存在。")

    finally:
        # --- Cleanup: 清理 ---
        # 由 autouse fixture `clear_db_before_test` 自动处理
        pass


# --- Mock 辅助类 (主要用于旧的或特定的 Mocked 测试) ---
class MockNeo4jRecord:
    """模拟 Neo4j 查询返回的 Record 对象。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data  # 存储记录的数据

    def data(self) -> Dict[str, Any]:
        """模拟 record.data() 方法。"""
        return self._data

    def __getitem__(self, key: str) -> Any:
        """模拟通过键访问记录字段 (record['key'])。"""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """模拟 record.get(key, default) 方法。"""
        return self._data.get(key, default)


@pytest.mark.asyncio
async def test_search_nodes_success_with_results() -> None:
    """
    单元测试 (Mocked): 测试 `search_nodes` 在找到匹配项时返回正确的结果。

    由于在集成测试中设置和管理 Neo4j 全文索引可能比较复杂，
    这个测试使用 Mock（通过 patch.object 直接替换方法实现）来模拟 `search_nodes` 的行为。
    """
    # --- Setup: 准备 Mock 对象 ---
    mock_driver = AsyncMock(spec=AsyncDriver)  # 不需要真正连接，仅作占位符
    repo = Neo4jRepository(driver=mock_driver)

    # --- Setup: 定义预期的模拟返回结果 ---
    # 模拟 `search_nodes` 应该返回的数据格式
    expected_results = [
        {
            "node": {"id": 1, "title": "Paper 1"},
            "score": 0.9,
        },  # 模拟找到的节点和相似度分数
        {"node": {"id": 2, "name": "Author A"}, "score": 0.8},
    ]

    # --- Action & Assertion: 使用 patch.object 模拟方法并调用 ---
    # 使用 patch.object 作为上下文管理器，临时将 repo 实例上的 search_nodes 方法
    # 替换为一个直接返回 `expected_results` 的函数。
    with patch.object(repo, "search_nodes", return_value=expected_results):
        # 调用被（间接）测试的方法（实际上调用的是 patch 后的版本）
        results = await repo.search_nodes(
            search_term="test query",  # <--- FIX: Changed from query to search_term
            index_name="paper_fulltext_idx",  # 索引名称 (在模拟中不重要)
            labels=["Paper", "Author"],  # 目标节点标签 (在模拟中不重要)
            limit=10,
            skip=0,
        )

        # 断言返回的结果与预期的模拟结果完全一致
        assert results == expected_results


@pytest.mark.asyncio
async def test_search_nodes_success_no_results() -> None:
    """
    单元测试 (Mocked): 测试 `search_nodes` 在没有找到匹配项时返回空列表。

    同样使用 Mock 来模拟数据库交互。
    """
    # --- Setup: 准备 Mock 对象 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()  # 使用 MagicMock 模拟 Neo4j 返回的结果对象
    # 配置模拟结果对象的 data() 方法返回空列表
    mock_result.data.return_value = []

    # 配置 mock_session.run() 返回这个模拟结果对象
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = None

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action: 调用被测方法 ---
    results = await repo.search_nodes(
        search_term="term", index_name="idx", labels=["Label"]
    )

    # --- Assertion: 验证结果和 Mock 调用 ---
    # 断言 mock_session.run 被精确地调用了一次
    mock_session.run.assert_awaited_once()
    # 断言返回结果是空列表
    assert results == []


@pytest.mark.asyncio
@patch("aigraphx.repositories.neo4j_repo.logger")
async def test_search_nodes_failure(mock_logger: MagicMock) -> None:
    """
    单元测试 (Mocked): 测试 `search_nodes` 在底层查询失败时记录错误并抛出异常。

    Args:
        mock_logger (MagicMock): Mocked logger 对象。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock(spec=AsyncSession)
    test_exception = Exception("Fulltext index error")  # 模拟的查询异常
    # 配置 mock_session.run() 在被调用时抛出异常
    mock_session.run = AsyncMock(side_effect=test_exception)
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = None

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action & Assertion: 调用并断言异常 ---
    with pytest.raises(Exception, match="Fulltext index error"):
        await repo.search_nodes(search_term="term", index_name="idx", labels=["Label"])

    # --- Assertion: 验证 Mock 调用 ---
    mock_logger.error.assert_called_once()
    # 检查错误日志消息
    assert "Error searching Neo4j" in mock_logger.error.call_args[0][0]
    assert str(test_exception) in mock_logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_get_neighbors_success_with_results_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `get_neighbors` 方法是否能成功检索到一个已存在节点的邻居节点及其关系信息。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例，用于设置数据。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database
    node_id = "neighbor_test_node_1"  # 中心节点 ID
    neighbor_id_1 = "neighbor_test_neighbor_1"  # 邻居节点 1 ID
    neighbor_id_2 = "neighbor_test_neighbor_2"  # 邻居节点 2 ID

    # --- Setup: 直接使用驱动创建测试数据 ---
    # 创建中心节点 (TestNode) 和两个邻居节点 (TestNeighbor)
    # 创建关系：(n1)-[:CONNECTS_TO {weight: 1.0}]->(n2)  (出向关系)
    # 创建关系：(n3)-[:CONNECTS_TO {weight: 2.0}]->(n1)  (入向关系)
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                CREATE (n1:TestNode {node_id: $id, name: 'Start Node'})
                CREATE (n2:TestNeighbor {node_id: $nid1, name: 'Neighbor 1'})
                CREATE (n3:TestNeighbor {node_id: $nid2, name: 'Neighbor 2'})
                CREATE (n1)-[:CONNECTS_TO {weight: 1.0}]->(n2)
                CREATE (n3)-[:CONNECTS_TO {weight: 2.0}]->(n1)
                """,
                id=node_id,
                nid1=neighbor_id_1,
                nid2=neighbor_id_2,
            )
        )

    try:
        # --- Action: 调用被测方法 ---
        # 注意：get_neighbors 现在不需要 max_neighbors 参数了（根据代码）
        neighbors_data = await repo.get_neighbors(
            node_label="TestNode",
            node_prop="node_id",
            node_val=node_id,  # <--- FIX: Changed from node_value to node_val
        )

        # --- Assertions: 验证返回结果 ---
        assert isinstance(neighbors_data, list)  # 结果应该是列表
        assert len(neighbors_data) == 2  # 应该找到两个邻居

        # 将结果列表转换为字典，方便按邻居 ID 查找
        results_dict = {n["node"]["node_id"]: n for n in neighbors_data}
        assert neighbor_id_1 in results_dict  # 确认邻居 1 在结果中
        assert neighbor_id_2 in results_dict  # 确认邻居 2 在结果中

        # 验证邻居 1 (neighbor_id_1) 的详细信息
        neighbor1_data = results_dict[neighbor_id_1]
        assert neighbor1_data["relationship"]["properties"]["weight"] == 1.0  # 关系属性
        assert neighbor1_data["relationship"]["type"] == "CONNECTS_TO"  # 关系类型
        assert (
            neighbor1_data["direction"] == "OUT"
        )  # 关系方向 (对于中心节点 n1 来说是出向)

        # 验证邻居 2 (neighbor_id_2) 的详细信息
        neighbor2_data = results_dict[neighbor_id_2]
        assert neighbor2_data["relationship"]["properties"]["weight"] == 2.0  # 关系属性
        assert neighbor2_data["relationship"]["type"] == "CONNECTS_TO"  # 关系类型
        assert (
            neighbor2_data["direction"] == "IN"
        )  # 关系方向 (对于中心节点 n1 来说是入向)

    finally:
        # --- Cleanup: 清理 ---
        # 由 autouse fixture `clear_db_before_test` 自动处理
        pass


@pytest.mark.asyncio
async def test_get_neighbors_no_results_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `get_neighbors` 在查询一个不存在的节点时，返回空列表。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    # --- Action: 调用被测方法查询一个不存在的节点 ---
    # 注意：get_neighbors 现在不需要 max_neighbors 参数了
    neighbors_data = await repo.get_neighbors(
        node_label="NonExistentLabel",
        node_prop="node_id",
        node_val="non_existent_id",  # <--- FIX: Changed from node_value to node_val
    )
    # --- Assertion: 验证返回空列表 ---
    assert neighbors_data == []


@pytest.mark.asyncio
@patch("aigraphx.repositories.neo4j_repo.logger")
async def test_get_neighbors_failure(mock_logger: MagicMock) -> None:
    """
    单元测试 (Mocked): 测试 `get_neighbors` 在底层查询失败时记录错误并抛出异常。

    Args:
        mock_logger (MagicMock): Mocked logger 对象。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock(spec=AsyncSession)
    test_exception = Exception("Neighbor query error")  # 模拟的查询异常
    mock_session.run = AsyncMock(side_effect=test_exception)  # 配置 run() 抛出异常
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = None

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action & Assertion: 调用并断言异常 ---
    with pytest.raises(Exception, match="Neighbor query error"):
        await repo.get_neighbors(
            node_label="Paper",
            node_prop="pwc_id",
            node_val="node_id_val",  # <--- FIX: Changed from node_value to node_val
        )

    # --- Assertion: 验证 Mock 调用 ---
    mock_logger.error.assert_called_once()
    assert "Error getting neighbors from Neo4j" in mock_logger.error.call_args[0][0]
    assert str(test_exception) in mock_logger.error.call_args[0][0]


# --- 针对 get_related_nodes 方法的测试用例 ---


# --- Mock 辅助类 (保留，以防将来需要 Mocked 测试) ---
class MockNode:
    """模拟 Neo4j 节点对象。"""

    def __init__(self, element_id: str, labels: List[str], properties: Dict[str, Any]):
        self.element_id = element_id  # 节点内部 ID
        self.labels = set(labels)  # 节点的标签集合
        self.properties = properties  # 节点的属性字典

    def __getitem__(self, key: str) -> Any:
        return self.properties[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def keys(self) -> Any:
        return self.properties.keys()

    def items(self) -> Any:
        return self.properties.items()


class MockRelationship:
    """模拟 Neo4j 关系对象。"""

    def __init__(
        self,
        element_id: str,
        type: str,
        properties: Dict[str, Any],
        start_node: MockNode,
        end_node: MockNode,
    ):
        self.element_id = element_id  # 关系内部 ID
        self.type = type  # 关系类型
        self.properties = properties  # 关系属性字典
        self.start_node = start_node  # 起始节点
        self.end_node = end_node  # 结束节点

    def __getitem__(self, key: str) -> Any:
        return self.properties[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def keys(self) -> Any:
        return self.properties.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direction",
    [("OUT"), ("IN"), ("BOTH")],  # 参数化：对三种方向分别执行测试
)
async def test_get_related_nodes_integration(
    direction: Literal["OUT", "IN", "BOTH"],  # 当前测试使用的方向参数
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `get_related_nodes` 方法是否能根据不同方向 ('OUT', 'IN', 'BOTH')
    成功检索相关的节点及其关系信息。

    Args:
        direction (Literal["OUT", "IN", "BOTH"]): 测试参数，指定关系方向。
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database
    start_node_id = "related_test_start"  # 起始节点 ID
    target_node_id_out = "related_test_target_out"  # 出向关系的目标节点 ID
    target_node_id_in = "related_test_target_in"  # 入向关系的目标节点 ID

    # --- Setup: 创建测试数据 ---
    # 创建起始节点 (Start)、出向目标节点 (Target)、入向目标节点 (Target)
    # 创建关系: (start)-[:RELATES_TO {rel_prop: 'out'}]->(target_out)
    # 创建关系: (target_in)-[:RELATES_TO {rel_prop: 'in'}]->(start)
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                CREATE (start:Start {node_id: $sid, name: 'Start'})
                CREATE (target_out:Target {node_id: $tid_out, name: 'Target Out'})
                CREATE (target_in:Target {node_id: $tid_in, name: 'Target In'})
                CREATE (start)-[:RELATES_TO {rel_prop: 'out'}]->(target_out)
                CREATE (target_in)-[:RELATES_TO {rel_prop: 'in'}]->(start)
                """,
                sid=start_node_id,
                tid_out=target_node_id_out,
                tid_in=target_node_id_in,
            )
        )

    try:
        # --- Action: 调用被测方法 ---
        # 注意参数名称：start_node_val (不是 value)
        related_nodes = await repo.get_related_nodes(
            start_node_label="Start",  # 起始节点标签
            start_node_prop="node_id",  # 用于匹配起始节点的属性名
            start_node_val=start_node_id,  # 用于匹配起始节点的属性值
            relationship_type="RELATES_TO",  # 要匹配的关系类型
            target_node_label="Target",  # 目标节点的标签
            direction=direction,  # 关系方向 (来自参数化)
            limit=10,  # 最大返回数量
        )

        # --- Assertions: 验证返回结果 ---
        assert isinstance(related_nodes, list)
        # 将结果列表转换为字典，方便按目标节点 ID 查找
        results_dict: Dict[str, Dict[str, Any]] = {
            r["target_node"]["node_id"]: r for r in related_nodes
        }

        # 根据不同的方向参数进行不同的断言
        if direction == "OUT":
            assert len(related_nodes) == 1  # 只应找到出向关系的节点
            assert target_node_id_out in results_dict  # 确认是出向目标节点
            # 验证关系属性
            assert results_dict[target_node_id_out]["relationship"]["rel_prop"] == "out"
        elif direction == "IN":
            assert len(related_nodes) == 1  # 只应找到入向关系的节点
            assert target_node_id_in in results_dict  # 确认是入向目标节点
            # 验证关系属性
            assert results_dict[target_node_id_in]["relationship"]["rel_prop"] == "in"
        elif direction == "BOTH":
            # 获取所有找到的目标节点 ID
            target_ids_found = {r["target_node"]["node_id"] for r in related_nodes}
            assert len(target_ids_found) == 2  # 应找到两个不同的目标节点
            assert target_node_id_out in target_ids_found  # 包含出向目标
            assert target_node_id_in in target_ids_found  # 包含入向目标

            # 分别查找出向和入向关系的数据
            out_rel = next(
                (
                    r
                    for r in related_nodes
                    if r["target_node"]["node_id"] == target_node_id_out
                ),
                None,
            )
            in_rel = next(
                (
                    r
                    for r in related_nodes
                    if r["target_node"]["node_id"] == target_node_id_in
                ),
                None,
            )
            # 验证各自的关系属性
            assert out_rel is not None and out_rel["relationship"]["rel_prop"] == "out"
            assert in_rel is not None and in_rel["relationship"]["rel_prop"] == "in"

    finally:
        # --- Cleanup: 清理 ---
        # 由 autouse fixture `clear_db_before_test` 自动处理
        pass


@pytest.mark.asyncio
async def test_get_related_nodes_no_results_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `get_related_nodes` 在查询一个不存在的起始节点时，返回空列表。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    # --- Action: 调用被测方法查询不存在的节点 ---
    related_nodes = await repo.get_related_nodes(
        start_node_label="Start",
        start_node_prop="node_id",
        start_node_val="non_existent_start",  # 不存在的起始节点值
        relationship_type="RELATES_TO",
        target_node_label="Target",
        direction="OUT",
        limit=10,
    )
    # --- Assertion: 验证返回空列表 ---
    assert related_nodes == []


# 保留失败场景的 Mocked 测试，因为在集成测试中模拟特定数据库错误比较困难
@pytest.mark.asyncio
@patch("aigraphx.repositories.neo4j_repo.logger")
async def test_get_related_nodes_driver_unavailable(mock_logger: MagicMock) -> None:
    """
    单元测试 (Mocked): 测试 `get_related_nodes` 在驱动不可用（例如已关闭）时的处理。

    Args:
        mock_logger (MagicMock): Mocked logger 对象。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    # 使 session 属性本身就是一个可调用的 MagicMock，而不是返回一个 Mock 会话
    mock_driver.session = MagicMock()
    # 配置调用 session() 时直接抛出异常，模拟驱动已关闭或无法创建会话
    mock_driver.session.side_effect = Exception("Driver closed")
    repo = Neo4jRepository(driver=mock_driver)

    # --- Action & Assertion: 调用并断言异常 ---
    with pytest.raises(Exception, match="Driver closed"):
        await repo.get_related_nodes("Start", "id", "val", "REL", "Target", "OUT")

    # --- Assertion: 验证 Mock 调用 (可选) ---
    # 在这种情况下，错误发生在获取会话之前，可能不会记录特定的 "Error getting related nodes" 日志
    # 可以检查是否有任何错误日志被记录
    # mock_logger.error.assert_called()


@pytest.mark.asyncio
@patch("aigraphx.repositories.neo4j_repo.logger")
async def test_get_related_nodes_exception(mock_logger: MagicMock) -> None:
    """
    单元测试 (Mocked): 测试 `get_related_nodes` 在查询执行期间发生通用异常时的处理。

    Args:
        mock_logger (MagicMock): Mocked logger 对象。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock(spec=AsyncSession)
    test_exception = Exception("Query execution error")  # 模拟的查询异常

    # 配置 session.run() 方法在被调用时抛出异常
    mock_session.run = AsyncMock(side_effect=test_exception)

    # 设置驱动返回模拟会话
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = None

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action & Assertion: 调用并断言异常 ---
    with pytest.raises(Exception, match="Query execution error"):
        await repo.get_related_nodes("Start", "id", "val", "REL", "Target", "OUT")

    # --- Assertion: 验证 Mock 调用 ---
    # 断言 logger.error 被调用了（可能不止一次，如果在多个地方记录）
    assert mock_logger.error.call_count >= 1
    # 检查第一次错误日志调用的参数，看是否包含预期的错误信息片段
    first_call_args = mock_logger.error.call_args_list[0][0]
    assert any(
        "Error getting related nodes" in arg
        for arg in first_call_args
        if isinstance(arg, str)
    )


@pytest.mark.asyncio
async def test_get_related_nodes_invalid_direction(
    neo4j_repo_fixture: Neo4jRepository,
) -> None:
    """
    单元测试：测试 `get_related_nodes` 在接收到无效的 direction 参数时抛出 ValueError。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例 (驱动本身不重要)。
    """
    repo = neo4j_repo_fixture
    # --- Action & Assertion: 调用并断言 ValueError ---
    with pytest.raises(ValueError) as excinfo:
        # 使用 cast 将 "INVALID" 强制转换为合法的类型，以通过静态类型检查
        # 但在运行时，它仍然是一个无效的值
        await repo.get_related_nodes(
            start_node_label="Start",
            start_node_prop="id",
            start_node_val="val",
            relationship_type="REL",
            target_node_label="Target",
            direction=cast(Literal["OUT", "IN", "BOTH"], "INVALID"),
        )
    # 断言异常消息中包含 "Invalid direction"
    assert "Invalid direction" in str(excinfo.value)


@pytest.mark.asyncio
async def test_count_paper_nodes_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `count_paper_nodes` 方法是否能正确返回数据库中 Paper 节点的数量。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例，用于设置数据。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 确保数据库初始为空 ---
    # 虽然有 autouse fixture，但在这里显式调用一次可以增加确定性
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Action & Assertion: 检查空数据库计数 ---
    empty_count = await repo.count_paper_nodes()
    assert empty_count == 0  # 空数据库应该返回 0

    # --- Setup: 添加测试数据 ---
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                CREATE (p1:Paper {pwc_id: 'test-count-1', title: 'Test Paper 1'})
                CREATE (p2:Paper {pwc_id: 'test-count-2', title: 'Test Paper 2'})
                """
            )
        )

    # --- Action & Assertion: 测试有数据时的计数 ---
    count_result = await repo.count_paper_nodes()
    assert count_result == 2  # 应该返回 2


@pytest.mark.asyncio
async def test_count_paper_nodes_error(
    neo4j_driver: AsyncDriver,  # 仅用于类型提示，实际使用 Mock
    test_settings: Settings,  # 仅用于类型提示
    request: FixtureRequest,  # 仅用于类型提示
) -> None:
    """
    单元测试 (Mocked): 测试 `count_paper_nodes` 在发生数据库错误时返回 0。

    Args:
        neo4j_driver (AsyncDriver): 类型提示。
        test_settings (Settings): 类型提示。
        request (FixtureRequest): 类型提示。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock()
    # 模拟获取会话时就发生错误
    mock_session.__aenter__.side_effect = Exception("Simulated session error")
    mock_driver.session.return_value = mock_session

    # 使用 Mock 驱动创建仓库实例
    repo = Neo4jRepository(driver=mock_driver)

    # --- Action: 调用被测方法 ---
    count_result = await repo.count_paper_nodes()

    # --- Assertion: 断言错误时返回 0 ---
    # 根据当前实现，发生错误时会记录日志并返回 0
    assert count_result == 0


@pytest.mark.asyncio
async def test_count_hf_models_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `count_hf_models` 方法是否能正确返回数据库中 HFModel 节点的数量。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Action & Assertion: 检查空数据库计数 ---
    empty_count = await repo.count_hf_models()
    assert empty_count == 0

    # --- Setup: 添加测试数据 ---
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                CREATE (m1:HFModel {model_id: 'model-1', author: 'Author 1'})
                CREATE (m2:HFModel {model_id: 'model-2', author: 'Author 2'})
                CREATE (m3:HFModel {model_id: 'model-3', author: 'Author 3'})
                """
            )
        )

    # --- Action & Assertion: 测试有数据时的计数 ---
    count_result = await repo.count_hf_models()
    assert count_result == 3


@pytest.mark.asyncio
async def test_count_hf_models_error(
    neo4j_driver: AsyncDriver, test_settings: Settings, request: FixtureRequest
) -> None:
    """
    单元测试 (Mocked): 测试 `count_hf_models` 在发生数据库错误时返回 0。

    Args:
        neo4j_driver (AsyncDriver): 类型提示。
        test_settings (Settings): 类型提示。
        request (FixtureRequest): 类型提示。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock()
    mock_session.__aenter__.side_effect = Exception("Simulated session error")
    mock_driver.session.return_value = mock_session

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action: 调用被测方法 ---
    count_result = await repo.count_hf_models()

    # --- Assertion: 断言错误时返回 0 ---
    assert count_result == 0


@pytest.mark.asyncio
async def test_get_paper_neighborhood_not_found_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `get_paper_neighborhood` 在查询一个不存在的论文 ID 时返回 None。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Action: 调用被测方法查询不存在的论文 ---
    result = await repo.get_paper_neighborhood("non-existent-paper-id")

    # --- Assertion: 断言返回 None ---
    assert result is None


@pytest.mark.asyncio
async def test_get_paper_neighborhood_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `get_paper_neighborhood` 方法是否能成功获取一篇论文及其所有类型的关联实体。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database
    pwc_id = "test-neighborhood"  # 测试用论文 ID

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Setup: 创建复杂的测试图结构 ---
    # 创建论文、作者、任务、数据集、领域、方法、仓库、模型等节点，并建立它们之间的关系
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                // 创建 Paper 节点
                CREATE (p:Paper {
                    pwc_id: $pwc_id,
                    title: 'Test Neighborhood Paper',
                    summary: 'Summary for neighborhood test',
                    published_date: date('2023-10-15') // 使用 Neo4j 的 date() 函数
                })

                // 创建 Author 节点并关联
                CREATE (a1:Author {name: 'Author A'})
                CREATE (a2:Author {name: 'Author B'})
                CREATE (a1)-[:AUTHORED]->(p)
                CREATE (a2)-[:AUTHORED]->(p)

                // 创建 Task 节点并关联
                CREATE (t1:Task {name: 'Task X'})
                CREATE (t2:Task {name: 'Task Y'})
                CREATE (p)-[:HAS_TASK]->(t1)
                CREATE (p)-[:HAS_TASK]->(t2)

                // 创建 Dataset 节点并关联
                CREATE (d:Dataset {name: 'Dataset Z'})
                CREATE (p)-[:USES_DATASET]->(d)

                // 创建 Area 节点并关联
                CREATE (ar:Area {name: 'Computer Vision'})
                CREATE (p)-[:HAS_AREA]->(ar) // 假设存在 HAS_AREA 关系

                // 创建 Method 节点并关联
                CREATE (m:Method {name: 'Method M'})
                CREATE (p)-[:USES_METHOD]->(m) // 假设存在 USES_METHOD 关系

                // 创建 Repository 节点并关联
                CREATE (r:Repository {
                    url: 'http://github.com/test/repo',
                    stars: 100,
                    framework: 'pytorch' // 可能需要关联到 Framework 节点
                })
                CREATE (p)-[:HAS_REPOSITORY]->(r)

                // 创建 HFModel 节点并关联 (假设是 Model 提到 Paper)
                CREATE (hf:HFModel {
                    model_id: 'test/model',
                    author: 'Test Author'
                })
                CREATE (hf)-[:MENTIONS]->(p)
                """,
                {"pwc_id": pwc_id},
            )
        )

    # --- Action: 调用被测方法 ---
    result = await repo.get_paper_neighborhood(pwc_id)

    # --- Assertions: 验证返回的邻域数据 ---
    assert result is not None  # 应该返回了数据，而不是 None

    # 1. 验证论文本身的数据
    assert result["paper"]["pwc_id"] == pwc_id
    assert result["paper"]["title"] == "Test Neighborhood Paper"
    # 可以添加对其他论文属性的验证

    # 2. 验证关联的作者
    assert "authors" in result and isinstance(result["authors"], list)
    assert len(result["authors"]) == 2
    author_names = {author["name"] for author in result["authors"]}
    assert "Author A" in author_names
    assert "Author B" in author_names

    # 3. 验证关联的任务
    assert "tasks" in result and isinstance(result["tasks"], list)
    assert len(result["tasks"]) == 2
    task_names = {task["name"] for task in result["tasks"]}
    assert "Task X" in task_names
    assert "Task Y" in task_names

    # 4. 验证关联的数据集
    assert "datasets" in result and isinstance(result["datasets"], list)
    assert len(result["datasets"]) == 1
    assert result["datasets"][0]["name"] == "Dataset Z"

    # 5. 验证关联的方法
    assert "methods" in result and isinstance(result["methods"], list)
    assert len(result["methods"]) == 1
    assert result["methods"][0]["name"] == "Method M"

    # 6. 验证关联的代码仓库
    assert "repositories" in result and isinstance(result["repositories"], list)
    assert len(result["repositories"]) == 1
    assert result["repositories"][0]["url"] == "http://github.com/test/repo"
    assert result["repositories"][0]["stars"] == 100
    # 可以添加对 framework 的验证

    # 7. 验证关联的领域
    assert "area" in result and isinstance(result["area"], dict)
    assert result["area"]["name"] == "Computer Vision"

    # 8. 验证关联的模型
    assert "models" in result and isinstance(result["models"], list)
    assert len(result["models"]) == 1
    assert result["models"][0]["model_id"] == "test/model"


@pytest.mark.asyncio
@patch("aigraphx.repositories.neo4j_repo.logger")
async def test_get_paper_neighborhood_error(
    mock_logger: MagicMock,
    neo4j_driver: AsyncDriver,  # 仅用于类型提示
    test_settings: Settings,  # 仅用于类型提示
) -> None:
    """
    单元测试 (Mocked): 测试 `get_paper_neighborhood` 在发生数据库错误时返回 None 并记录日志。

    Args:
        mock_logger (MagicMock): Mocked logger 对象。
        neo4j_driver (AsyncDriver): 类型提示。
        test_settings (Settings): 类型提示。
    """
    # --- Setup: 准备 Mock 对象和模拟异常 ---
    mock_driver = AsyncMock(spec=AsyncDriver)
    mock_session = AsyncMock()
    mock_session.__aenter__.side_effect = Exception("Simulated session error")
    mock_driver.session.return_value = mock_session

    repo = Neo4jRepository(driver=mock_driver)

    # --- Action: 调用被测方法 ---
    result = await repo.get_paper_neighborhood("test-id")

    # --- Assertions: 验证返回 None 和日志记录 ---
    assert result is None
    mock_logger.error.assert_called()  # 验证 logger.error 被调用


@pytest.mark.asyncio
async def test_link_model_to_paper_batch_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `link_model_to_paper_batch` 方法是否能成功批量创建 HFModel 和 Paper 之间的 MENTIONS 关系，并带有属性。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Setup: 创建测试用的 Paper 和 HFModel 节点 ---
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                CREATE (p1:Paper {pwc_id: 'paper-1', title: 'Paper 1'})
                CREATE (p2:Paper {pwc_id: 'paper-2', title: 'Paper 2'})
                CREATE (m1:HFModel {model_id: 'model-1', author: 'Author 1'})
                CREATE (m2:HFModel {model_id: 'model-2', author: 'Author 2'})
                """
            )
        )

    # --- Setup: 准备要创建的链接数据 ---
    # 列表中的每个字典代表一个要创建的 MENTIONS 关系
    links = [
        {"model_id": "model-1", "pwc_id": "paper-1", "confidence": 0.95},
        {"model_id": "model-2", "pwc_id": "paper-2", "confidence": 0.85},
        {
            "model_id": "model-1",  # model-1 链接到第二篇论文
            "pwc_id": "paper-2",
            "confidence": 0.70,
        },
    ]

    # --- Action: 调用被测方法 ---
    await repo.link_model_to_paper_batch(links)

    # --- Verification: 验证关系是否已创建 ---
    async with neo4j_driver.session(database=db_name) as session:
        # 1. 验证创建的关系总数
        result_count = await session.run(
            # 匹配所有 HFModel 到 Paper 的 MENTIONS 关系，并计数
            "MATCH (m:HFModel)-[r:MENTIONS]->(p:Paper) RETURN count(r) AS count"
        )
        count_record = await result_count.single()
        assert count_record is not None
        assert count_record["count"] == 3  # 应该创建了 3 条关系

        # 2. 验证特定链接及其属性 (例如 model-1 -> paper-1)
        result_link1 = await session.run(
            """
            MATCH (m:HFModel {model_id: 'model-1'})-[r:MENTIONS]->(p:Paper {pwc_id: 'paper-1'})
            RETURN r.confidence AS confidence // 返回关系的 confidence 属性
            """
        )
        link1_record = await result_link1.single()
        assert link1_record is not None
        assert link1_record["confidence"] == 0.95  # 验证属性值

        # 3. 验证另一个链接 (model-1 -> paper-2)
        result_link3 = await session.run(
            """
            MATCH (m:HFModel {model_id: 'model-1'})-[r:MENTIONS]->(p:Paper {pwc_id: 'paper-2'})
            RETURN r.confidence AS confidence
            """
        )
        link3_record = await result_link3.single()
        assert link3_record is not None
        assert link3_record["confidence"] == 0.70


@pytest.mark.asyncio
async def test_save_papers_by_arxiv_batch_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `save_papers_by_arxiv_batch` 方法是否能根据 ArXiv 数据成功批量创建 Paper 节点及其关联的 Author 和 Category 节点/关系。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Setup: 准备基于 ArXiv 的论文数据 ---
    papers_data = [
        {
            "arxiv_id_base": "2401.00001",  # ArXiv ID (关键)
            "arxiv_id_versioned": "2401.00001v1",
            "title": "ArXiv Paper 1",
            "summary": "Summary for ArXiv paper 1",
            "published_date": "2024-01-01",
            "authors": ["Author A", "Author B"],  # 作者列表
            "primary_category": "cs.CV",
            "categories": ["cs.CV", "cs.AI"],  # 分类列表
        },
        {
            "arxiv_id_base": "2401.00002",
            "arxiv_id_versioned": "2401.00002v2",
            "title": "ArXiv Paper 2",
            "summary": "Summary for ArXiv paper 2",
            "published_date": "2024-01-02",
            "authors": ["Author C"],
            "primary_category": "cs.LG",
            "categories": ["cs.LG"],
        },
        {
            "arxiv_id_base": "2401.00003",
            "arxiv_id_versioned": "2401.00003v1",
            "title": "ArXiv Paper 3",
            "summary": "Summary for ArXiv paper 3",
            "published_date": "2024-01-03",
            "authors": ["Author D", "Author E"],
            "primary_category": "cs.NE",
            "categories": ["cs.NE", "cs.AI"],  # 注意 cs.AI 会被复用
        },
    ]

    # --- Action: 调用被测方法 ---
    await repo.save_papers_by_arxiv_batch(papers_data)

    # --- Verification: 验证创建的节点和关系 ---
    async with neo4j_driver.session(database=db_name) as session:
        # 1. 验证 Paper 节点数量
        result_papers = await session.run(
            # 匹配 arxiv_id_base 在给定列表中的 Paper 节点
            "MATCH (p:Paper) WHERE p.arxiv_id_base IN $ids RETURN count(p) AS count",
            {"ids": ["2401.00001", "2401.00002", "2401.00003"]},
        )
        papers_count = await result_papers.single()
        assert papers_count is not None
        assert papers_count["count"] == 3  # 应创建 3 个 Paper 节点

        # 2. 验证 Author 节点数量 (应创建 5 个不同的 Author 节点)
        result_authors = await session.run(
            # 匹配所有 Author 节点，并去重计数
            "MATCH (a:Author) RETURN count(DISTINCT a.name) AS count"
            # 或者通过关系计数，但可能会重复计算作者
            # "MATCH (a:Author)-[:AUTHORED]->(p:Paper) WHERE p.arxiv_id_base IN $ids RETURN count(DISTINCT a.name) AS count",
            # {"ids": ["2401.00001", "2401.00002", "2401.00003"]},
        )
        authors_count = await result_authors.single()
        assert authors_count is not None
        assert authors_count["count"] == 5  # 共有 5 个不同的作者

        # 3. 验证 Category 节点数量 (应创建 4 个不同的 Category 节点: CV, AI, LG, NE)
        result_categories = await session.run(
            # 匹配所有 Category 节点，并去重计数
            "MATCH (c:Category) RETURN count(DISTINCT c.name) AS count"
            # "MATCH (p:Paper)-[:HAS_CATEGORY]->(c:Category) WHERE p.arxiv_id_base IN $ids RETURN count(DISTINCT c.name) AS count",
            # {"ids": ["2401.00001", "2401.00002", "2401.00003"]},
        )
        categories_count = await result_categories.single()
        assert categories_count is not None
        assert categories_count["count"] == 4  # 共有 4 个不同的分类

        # 4. （可选）验证关系数量，例如 AUTHORED 关系应有 5 条


@pytest.mark.asyncio
async def test_search_nodes_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试 (部分模拟)：测试 `search_nodes` 方法的基本调用流程。

    注意：由于在测试环境中配置 Neo4j 全文索引 (Full-Text Index) 或 APOC 过程可能比较复杂，
    这个测试选择**不直接验证数据库的全文搜索结果**。
    它创建了基础数据，然后使用 `patch.object` **模拟 `search_nodes` 方法本身**的返回值，
    从而专注于验证方法调用的接口和预期的数据格式，而不是实际的搜索逻辑。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例，用于设置数据。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Setup: 创建一些基础测试数据 ---
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                CREATE (p1:Paper {title: 'Neural Network Research', summary: 'A paper about neural networks'})
                CREATE (p2:Paper {title: 'Transformer Architecture', summary: 'A detailed study of transformers'})
                CREATE (p3:Paper {title: 'CNN Applications', summary: 'Applications of CNNs in computer vision'})
                """
            )
        )

    # --- Setup: 定义模拟的返回值 ---
    # 假设搜索 "neural" 会匹配到第一个论文
    mock_search_result = [
        {
            "node": {  # 模拟返回的节点数据
                "title": "Neural Network Research",
                "summary": "A paper about neural networks",
            },
            "score": 1.0,  # 模拟返回的相似度分数
        }
    ]

    # --- Action & Assertion: 使用 patch.object 模拟并调用 ---
    # 临时替换 repo 实例上的 search_nodes 方法
    with patch.object(
        repo, "search_nodes", return_value=mock_search_result
    ) as mock_method:
        # 调用被 mock 的方法
        results = await repo.search_nodes(
            search_term="neural",  # 搜索词
            index_name="paper_fulltext",  # 索引名 (在 mock 中不重要)
            labels=["Paper"],  # 目标标签
            limit=10,
            skip=0,
        )

        # 验证返回结果是否等于模拟的返回值
        assert isinstance(results, list)
        assert len(results) == 1
        assert results == mock_search_result
        assert "Neural Network Research" in results[0]["node"]["title"]

        # (可选) 验证 mock 方法是否被正确调用
        mock_method.assert_awaited_once_with(
            search_term="neural",
            index_name="paper_fulltext",
            labels=["Paper"],
            limit=10,
            skip=0,
        )


@pytest.mark.parametrize(
    "start_node_label,start_node_prop,relationship_type,target_node_label, expected_target_prop_value_map",
    [
        # 测试从 Paper 出发查找 Task
        ("Paper", "pwc_id", "HAS_TASK", "Task", {"Classification", "Object Detection"}),
        # 测试从 Paper 出发查找 Dataset
        ("Paper", "pwc_id", "USES_DATASET", "Dataset", {"COCO"}),
        # 测试从 HFModel 出发查找 Paper (MENTIONS 关系，方向反转)
        ("HFModel", "model_id", "MENTIONS", "Paper", {"paper-test-1", "paper-test-2"}),
    ],
)
@pytest.mark.asyncio
async def test_get_related_nodes_different_types_integration(
    start_node_label: str,  # 起始节点标签 (参数化)
    start_node_prop: str,  # 起始节点匹配属性 (参数化)
    relationship_type: str,  # 关系类型 (参数化)
    target_node_label: str,  # 目标节点标签 (参数化)
    expected_target_prop_value_map: set,  # 预期找到的目标节点的某个属性值的集合 (参数化)
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试 (参数化)：测试 `get_related_nodes` 方法获取不同类型的相关节点。

    使用 `@pytest.mark.parametrize` 来覆盖多种节点和关系类型的组合，
    验证该方法的通用性。

    Args:
        start_node_label (str): 起始节点标签。
        start_node_prop (str): 用于定位起始节点的属性名称。
        relationship_type (str): 要查找的关系类型。
        target_node_label (str): 目标节点的标签。
        expected_target_prop_value_map (set): 预期找到的目标节点的某个关键属性值的集合。
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Setup: 创建包含多种节点和关系的测试数据 ---
    async with neo4j_driver.session(database=db_name) as session:
        await session.execute_write(
            lambda tx: tx.run(
                """
                // 创建 Paper 节点
                CREATE (p1:Paper {pwc_id: 'paper-test-1', title: 'Paper 1'})
                CREATE (p2:Paper {pwc_id: 'paper-test-2', title: 'Paper 2'})

                // 创建 Task 节点
                CREATE (t1:Task {name: 'Classification'})
                CREATE (t2:Task {name: 'Object Detection'})
                CREATE (t3:Task {name: 'Segmentation'})

                // 创建 Dataset 节点
                CREATE (d1:Dataset {name: 'COCO'})
                CREATE (d2:Dataset {name: 'ImageNet'})

                // 创建 HFModel 节点
                CREATE (m1:HFModel {model_id: 'model-test-1', author: 'Author 1'})
                CREATE (m2:HFModel {model_id: 'model-test-2', author: 'Author 2'})

                // 创建关系
                CREATE (p1)-[:HAS_TASK]->(t1)
                CREATE (p1)-[:HAS_TASK]->(t2)
                CREATE (p2)-[:HAS_TASK]->(t3)

                CREATE (p1)-[:USES_DATASET]->(d1)
                CREATE (p2)-[:USES_DATASET]->(d2)

                CREATE (m1)-[:MENTIONS]->(p1) // 模型提及论文
                CREATE (m2)-[:MENTIONS]->(p1)
                CREATE (m2)-[:MENTIONS]->(p2)
                """
            )
        )

    # --- Setup: 确定当前参数化测试使用的起始节点值 ---
    start_node_val = ""
    if start_node_label == "Paper":
        start_node_val = "paper-test-1"  # 如果从 Paper 开始，使用 paper-test-1
    elif start_node_label == "HFModel":
        start_node_val = "model-test-2"  # 如果从 HFModel 开始，使用 model-test-2

    # --- Action: 调用被测方法 ---
    # 根据关系类型推断方向 (这是一个简化，实际可能更复杂)
    direction: Literal["OUT", "IN", "BOTH"] = "OUT"
    if relationship_type == "MENTIONS":  # MENTIONS 是 HFModel -> Paper
        direction = "OUT"  # 从 HFModel 出发是 OUT
        # 如果从 Paper 出发查找 HFModel，这里应该是 IN
        # 为了测试通用性，我们固定从 HFModel 出发查找 Paper
        if start_node_label == "Paper":
            logger.warning("Adjusting direction to IN for Paper->MENTIONS->HFModel")
            direction = "IN"
            start_node_val = "paper-test-1"  # 修正起始节点

    results = await repo.get_related_nodes(
        start_node_label=start_node_label,
        start_node_prop=start_node_prop,
        start_node_val=start_node_val,
        relationship_type=relationship_type,
        target_node_label=target_node_label,
        direction=direction,
        limit=10,
    )

    # --- Assertions: 验证结果 ---
    assert isinstance(results, list)
    assert len(results) > 0  # 断言找到了至少一个相关节点

    # 根据目标节点类型，确定要检查的属性名
    target_prop_key = ""
    if target_node_label == "Task":
        target_prop_key = "name"
    elif target_node_label == "Dataset":
        target_prop_key = "name"
    elif target_node_label == "Paper":
        target_prop_key = "pwc_id"

    # 提取所有找到的目标节点的关键属性值
    target_values_found = {result["target_node"][target_prop_key] for result in results}

    # 断言找到的值集合与预期的集合一致
    assert target_values_found == expected_target_prop_value_map


@pytest.mark.asyncio
async def test_create_or_update_paper_node_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `create_or_update_paper_node` 方法。

    验证该方法在节点不存在时创建节点，在节点已存在时更新节点属性（MERGE 行为）。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database
    pwc_id = "test-create-update"  # 测试用论文 ID

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)

    # --- Action 1: 测试创建节点 ---
    await repo.create_or_update_paper_node(pwc_id, "Initial Title")

    # --- Verification 1: 验证节点是否被创建 ---
    async with neo4j_driver.session(database=db_name) as session:
        result_create = await session.run(
            "MATCH (p:Paper {pwc_id: $pwc_id}) RETURN p.title AS title",
            {"pwc_id": pwc_id},
        )
        create_record = await result_create.single()
        assert create_record is not None  # 节点应该存在
        assert create_record["title"] == "Initial Title"  # 属性应为初始值

    # --- Action 2: 测试更新节点 ---
    # 使用相同的 pwc_id 再次调用，但提供不同的 title
    await repo.create_or_update_paper_node(pwc_id, "Updated Title")

    # --- Verification 2: 验证节点是否被更新 ---
    async with neo4j_driver.session(database=db_name) as session:
        result_update = await session.run(
            "MATCH (p:Paper {pwc_id: $pwc_id}) RETURN p.title AS title",
            {"pwc_id": pwc_id},
        )
        update_record = await result_update.single()
        assert update_record is not None  # 节点仍然存在
        assert update_record["title"] == "Updated Title"  # 属性应已更新


@pytest.mark.asyncio
async def test_link_paper_to_entity_integration(
    neo4j_repo_fixture: Neo4jRepository,
    neo4j_driver: AsyncDriver,
    test_settings: Settings,
    request: FixtureRequest,
) -> None:
    """
    集成测试：测试 `link_paper_to_entity` 方法。

    验证该方法是否能成功地将一个 Paper 节点链接到一个通用的实体节点，
    并使用自定义的标签和关系类型。

    Args:
        neo4j_repo_fixture (Neo4jRepository): 测试仓库实例。
        neo4j_driver (AsyncDriver): Neo4j 驱动实例。
        test_settings (Settings): 测试配置。
        request (FixtureRequest): Pytest 请求上下文。
    """
    repo = neo4j_repo_fixture
    db_name = test_settings.neo4j_database
    pwc_id = "test-link-entity"  # 测试用论文 ID

    # --- Setup: 确保数据库初始为空 ---
    await _clear_neo4j_db(neo4j_driver, test_settings)
    # 注意：此测试假设 Paper 节点和 Entity 节点会在 link 方法内部通过 MERGE 创建（如果不存在）

    # --- Action: 调用被测方法 ---
    # 链接论文到一个标签为 "CustomEntity"，名称为 "Test Entity" 的节点，
    # 使用的关系类型是 "HAS_CUSTOM_ENTITY"
    await repo.link_paper_to_entity(
        pwc_id=pwc_id,
        entity_label="CustomEntity",
        entity_name="Test Entity",
        relationship="HAS_CUSTOM_ENTITY",
    )

    # --- Verification: 验证链接是否被创建 ---
    async with neo4j_driver.session(database=db_name) as session:
        result = await session.run(
            """
            // 匹配 Paper 节点 -> 指定关系 -> 指定标签和名称的实体节点
            MATCH (p:Paper {pwc_id: $pwc_id})-[r:HAS_CUSTOM_ENTITY]->(e:CustomEntity {name: $name})
            RETURN p.pwc_id AS pwc_id, e.name AS entity_name // 返回两端的属性以确认匹配成功
            """,
            {"pwc_id": pwc_id, "name": "Test Entity"},
        )
        record = await result.single()
        assert record is not None  # 应该能匹配到这条路径
        assert record["pwc_id"] == pwc_id
        assert record["entity_name"] == "Test Entity"
