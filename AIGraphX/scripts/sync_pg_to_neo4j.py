#!/usr/bin/env python
"""
AIGraphX项目 - PostgreSQL到Neo4j同步脚本

本文件是AIGraphX项目中负责将PostgreSQL数据库中的数据同步到Neo4j图数据库的脚本。
在整个项目数据处理流程中，该脚本位于数据采集和数据库加载之后，负责构建知识图谱。

主要功能:
1. 从PostgreSQL数据库中获取模型、论文和它们之间的关系数据
2. 将模型数据同步到Neo4j数据库，创建模型节点
3. 将论文数据同步到Neo4j数据库，创建论文节点
4. 丰富论文节点，添加任务、数据集和代码库关系
5. 创建模型与论文之间的链接关系

该脚本在项目API服务使用前必须执行，因为图服务依赖Neo4j中构建好的知识图谱进行图查询。

执行流程:
1. 同步HF模型数据到Neo4j (sync_hf_models)
2. 同步论文数据到Neo4j (sync_papers_and_relations)
   - 分离PWC和ArXiv论文处理
   - 为PWC论文添加任务、数据集、代码库关系
3. 同步模型-论文链接关系 (sync_model_paper_links)

脚本支持--reset参数，可以在同步前清空Neo4j数据库。
"""

import asyncio
import logging
import os
import traceback  # 导入traceback模块
from typing import Optional, Dict, Any, List, Tuple  # 导入List和Tuple类型
import json
import sys  # 导入sys用于路径操作
import argparse  # 导入argparse解析命令行参数

# 第三方库导入
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver
from psycopg_pool import AsyncConnectionPool  # 导入PostgreSQL连接池

# 需要asyncpg处理连接池创建和异常处理
import asyncpg  # type: ignore[import-untyped]

# 调整导入路径，假设脚本从项目根目录运行或处理回退情况
try:
    # 首先尝试直接导入(如果作为模块运行或设置了PYTHONPATH)
    from aigraphx.repositories.postgres_repo import PostgresRepository
    from aigraphx.repositories.neo4j_repo import Neo4jRepository
except ImportError:
    # 如果作为独立脚本运行的回退方案
    print("INFO: 初始导入失败，尝试修改路径...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        print(f"INFO: 正在添加项目根目录到sys.path: {project_root}")
        sys.path.insert(0, project_root)

    # 修改路径后重试导入
    # --- 开始修正的代码块 ---
    try:
        # 第二个try下的正确缩进导入
        from aigraphx.repositories.postgres_repo import PostgresRepository
        from aigraphx.repositories.neo4j_repo import Neo4jRepository

        # 第二个try下的正确缩进打印
        print("INFO: 路径修改后成功导入模块。")
    except ImportError as e:  # 与第二个try正确对齐的except
        # 第二个except下的正确缩进打印语句
        print(f"严重错误: 即使在路径修改后也无法导入所需模块: {e}")
        print("严重错误: 确保脚本从项目根目录运行或正确设置了PYTHONPATH。")
        # 第二个except下的正确缩进sys.exit
        sys.exit(1)  # 如果无法导入核心组件则退出
    # --- 结束修正的代码块 ---


# --- 配置加载 ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# PostgreSQL连接配置
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("错误: 未设置DATABASE_URL环境变量。", file=sys.stderr)
    sys.exit(1)

# Neo4j连接配置
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    print(
        "错误: Neo4j连接详情(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)未完全设置。",
        file=sys.stderr,
    )
    sys.exit(1)

# --- 日志设置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- 常量 ---
PG_FETCH_BATCH_SIZE = 500  # 每次从PG获取多少条记录
NEO4J_WRITE_BATCH_SIZE = 200  # 每批向Neo4j写入多少条记录

# --- 同步逻辑 ---


async def sync_hf_models(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """将HF模型从PostgreSQL同步到Neo4j。

    Args:
        pg_repo: PostgreSQL仓库实例
        neo4j_repo: Neo4j仓库实例
        batch_size: 批处理大小，默认100
    """
    logger.info("开始HFModel同步...")
    models_synced = 0
    query = "SELECT * FROM hf_models ORDER BY hf_model_id"  # 排序以获得确定性批次
    models_to_process: List[Dict[str, Any]] = []

    try:
        # 修正调用: 为batch_size使用关键字参数
        async for model_record in pg_repo.fetch_data_cursor(
            query, batch_size=batch_size
        ):
            model_data = dict(model_record)
            # 确保标签作为列表加载，如果存储为JSON字符串
            if isinstance(model_data.get("hf_tags"), str):
                try:
                    model_data["hf_tags"] = json.loads(model_data["hf_tags"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"无法解码模型{model_data.get('hf_model_id')}的hf_tags JSON"
                    )
                    model_data["hf_tags"] = None

            model_data = {
                "model_id": model_data.get("hf_model_id"),
                "author": model_data.get("hf_author"),
                "sha": model_data.get("hf_sha"),
                "last_modified": model_data.get("hf_last_modified"),
                "tags": model_data.get("hf_tags") or [],
                "pipeline_tag": model_data.get("hf_pipeline_tag"),
                "downloads": model_data.get("hf_downloads"),
                "likes": model_data.get("hf_likes"),
                "library_name": model_data.get("hf_library_name"),
            }
            models_to_process.append(model_data)

            if len(models_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.save_hf_models_batch(models_to_process)
                    models_synced += len(models_to_process)
                    logger.info(f"已同步{models_synced}个HF模型...")
                except Exception as e:
                    logger.error(f"将HF模型批次保存到Neo4j时出错: {e}")
                    # 如果已在全局导入了traceback，则无需在此导入
                    logger.error(traceback.format_exc())
                finally:
                    models_to_process = []
    except Exception as e:
        logger.error(f"从Postgres获取HF模型时出错: {e}")
        # 无需导入traceback
        logger.error(traceback.format_exc())

    # 同步剩余模型
    if models_to_process:
        try:
            await neo4j_repo.save_hf_models_batch(models_to_process)
            models_synced += len(models_to_process)
        except Exception as e:
            logger.error(f"保存最后一批HF模型到Neo4j时出错: {e}")
            # 无需导入traceback
            logger.error(traceback.format_exc())

    logger.info(f"HFModel同步完成。共同步模型: {models_synced}个")


async def sync_papers_and_relations(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> int:
    """从PG获取论文及其关系并将它们同步到Neo4j。

    Args:
        pg_repo: PostgreSQL仓库实例
        neo4j_repo: Neo4j仓库实例
        batch_size: 批处理大小，默认100

    Returns:
        int: 同步的论文总数
    """
    logger.info("开始论文和关系同步...")
    papers_synced_arxiv = 0
    papers_synced_pwc = 0

    logger.info("获取并同步论文节点...")
    paper_query = """
        SELECT
            p.paper_id, p.pwc_id, p.arxiv_id_base, p.arxiv_id_versioned, p.title,
            p.authors, p.summary, p.published_date, p.area, p.pwc_url,
            p.pdf_url, p.doi, p.primary_category, p.categories
        FROM papers p
    """
    papers_to_process: List[Dict[str, Any]] = []
    arxiv_only_papers: List[Dict[str, Any]] = []  # 仅通过arxiv_id标识的论文
    try:
        # 修正调用: 为batch_size使用关键字参数
        async for paper_record in pg_repo.fetch_data_cursor(
            paper_query, batch_size=batch_size
        ):
            paper_data = dict(paper_record)

            # 将JSON字符串字段转换回列表(如果需要)(取决于PG repo处理)
            if isinstance(paper_data.get("authors"), str):
                try:
                    paper_data["authors"] = json.loads(paper_data["authors"])
                except json.JSONDecodeError:
                    paper_data["authors"] = []
            if isinstance(paper_data.get("categories"), str):
                try:
                    paper_data["categories"] = json.loads(paper_data["categories"])
                except json.JSONDecodeError:
                    paper_data["categories"] = []
            if paper_data.get("published_date"):
                paper_data["published_date"] = paper_data["published_date"].isoformat()
            # 初始化关系键 - 后续为pwc_id论文添加丰富信息
            paper_data["tasks"] = []
            paper_data["datasets"] = []
            paper_data["repositories"] = []

            # 按标识符可用性对论文进行分类
            if paper_data.get("pwc_id"):
                # 如果论文有PWC ID，用附加关系丰富它
                # 并使用基于pwc_id的Neo4j方法保存
                papers_to_process.append(paper_data)
            elif paper_data.get("arxiv_id_base"):
                # 如果论文只有ArXiv ID，我们将使用不同的方法保存
                arxiv_only_papers.append(paper_data)
            else:
                logger.warning(
                    f"论文id={paper_data.get('paper_id')}既没有pwc_id也没有arxiv_id。跳过。"
                )
                continue

            # 批处理 - PWC ID论文
            if len(papers_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    enriched_papers = await enrich_papers_with_relations(
                        pg_repo, papers_to_process
                    )
                    await neo4j_repo.save_papers_batch(enriched_papers)
                    papers_synced_pwc += len(enriched_papers)
                    logger.info(
                        f"已同步{papers_synced_pwc}篇PWC论文和{papers_synced_arxiv}篇仅ArXiv论文..."
                    )
                except Exception as e:
                    logger.error(f"将论文批次保存到Neo4j时出错: {e}")
                    # 无需导入traceback
                    logger.error(traceback.format_exc())
                finally:
                    papers_to_process = []

            # 批处理 - 仅ArXiv论文
            if len(arxiv_only_papers) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.save_papers_by_arxiv_batch(arxiv_only_papers)
                    papers_synced_arxiv += len(arxiv_only_papers)
                    logger.info(
                        f"已同步{papers_synced_pwc}篇PWC论文和{papers_synced_arxiv}篇仅ArXiv论文..."
                    )
                except Exception as e:
                    logger.error(f"将arxiv论文批次保存到Neo4j时出错: {e}")
                    # 无需导入traceback
                    logger.error(traceback.format_exc())
                finally:
                    arxiv_only_papers = []

    except Exception as e:
        logger.error(f"从PostgreSQL获取论文时出错: {e}")
        logger.error(traceback.format_exc())

    # 处理剩余的PWC论文
    if papers_to_process:
        try:
            enriched_papers = await enrich_papers_with_relations(
                pg_repo, papers_to_process
            )
            await neo4j_repo.save_papers_batch(enriched_papers)
            papers_synced_pwc += len(enriched_papers)
        except Exception as e:
            logger.error(f"将剩余PWC论文批次保存到Neo4j时出错: {e}")
            logger.error(traceback.format_exc())

    # 处理剩余的ArXiv论文
    if arxiv_only_papers:
        try:
            await neo4j_repo.save_papers_by_arxiv_batch(arxiv_only_papers)
            papers_synced_arxiv += len(arxiv_only_papers)
        except Exception as e:
            logger.error(f"将剩余ArXiv论文批次保存到Neo4j时出错: {e}")
            logger.error(traceback.format_exc())

    total_papers = papers_synced_pwc + papers_synced_arxiv
    logger.info(
        f"论文同步完成。共同步{total_papers}篇论文 "
        f"(PWC: {papers_synced_pwc}, 仅ArXiv: {papers_synced_arxiv})"
    )
    return total_papers


async def enrich_papers_with_relations(
    pg_repo: PostgresRepository, paper_batch: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """用关系信息丰富论文数据对象列表。

    Args:
        pg_repo: PostgreSQL仓库实例
        paper_batch: 要丰富的论文数据列表

    Returns:
        List[Dict[str, Any]]: 丰富后的论文数据列表
    """
    if not paper_batch:
        return []

    # 使用paper_id映射，因为之前的代码使用paper_id作为键
    paper_map = {
        int(p["paper_id"]): p for p in paper_batch if p.get("paper_id") is not None
    }
    paper_ids = list(paper_map.keys())

    if not paper_ids:
        logger.warning("Enrichment called with batch containing no valid paper_ids.")
        return paper_batch

    # --- 确保关系键存在 ---
    for paper_data in paper_map.values():
        paper_data.setdefault("tasks", [])
        paper_data.setdefault("datasets", [])
        paper_data.setdefault("methods", []) # PWC论文可能有关联的方法，需要初始化
        paper_data.setdefault("repositories", [])

    # --- 获取关系数据 ---
    # 使用 Repository 的方法代替 fetch_all
    try:
        # 任务关系
        tasks_map = await pg_repo.get_tasks_for_papers(paper_ids)
        for paper_id, tasks_list in tasks_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["tasks"] = tasks_list

        # 数据集关系
        datasets_map = await pg_repo.get_datasets_for_papers(paper_ids)
        for paper_id, datasets_list in datasets_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["datasets"] = datasets_list

        # 方法关系 (假设存在 get_methods_for_papers)
        # 如果没有这个方法，mypy可能还会报错，或者这里会抛出AttributeError
        try:
             methods_map = await pg_repo.get_methods_for_papers(paper_ids)
             for paper_id, methods_list in methods_map.items():
                 if paper_id in paper_map:
                     paper_map[paper_id]["methods"] = methods_list
        except AttributeError:
             logger.warning("PostgresRepository does not have 'get_methods_for_papers'. Methods not enriched.")
        except Exception as e_meth:
            logger.error(f"Error fetching methods relations: {e_meth}", exc_info=True)


        # 代码库关系 (假设存在 get_repositories_for_papers)
        repos_map = await pg_repo.get_repositories_for_papers(paper_ids)
        for paper_id, repo_urls in repos_map.items():
             if paper_id in paper_map:
                 # 保持与之前代码相似的结构，包含 url 键
                 paper_map[paper_id]["repositories"] = [
                     {"url": url} for url in repo_urls if url
                 ]


    except Exception as e:
        logger.error(f"获取论文关系数据时出错: {e}")
        logger.error(traceback.format_exc())
        # 出错时返回原始批次（带有默认空列表）
        return list(paper_map.values())

    # 返回丰富后的论文列表
    enriched_papers = list(paper_map.values())
    if enriched_papers:
        first_id = enriched_papers[0].get('paper_id', 'N/A')
        logger.debug(f"Enriched paper batch. Example paper ID {first_id} tasks: {enriched_papers[0].get('tasks')}")
    return enriched_papers


async def sync_model_paper_links(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """将模型-论文链接从PostgreSQL同步到Neo4j。

    Args:
        pg_repo: PostgreSQL仓库实例
        neo4j_repo: Neo4j仓库实例
        batch_size: 批处理大小，默认100
    """
    logger.info("开始模型-论文链接同步...")
    links_synced = 0
    query = """
        SELECT 
            mp.hf_model_id, 
            mp.paper_id,
            p.arxiv_id_base,
            p.pwc_id
        FROM model_papers mp
        JOIN papers p ON mp.paper_id = p.paper_id
        ORDER BY mp.hf_model_id
    """
    links_to_process = []

    try:
        async for link_record in pg_repo.fetch_data_cursor(
            query, batch_size=batch_size
        ):
            link_data = dict(link_record)

            # 跳过缺少必要标识符的链接
            if not link_data.get("hf_model_id"):
                logger.warning(f"链接缺少hf_model_id: {link_data}. 跳过。")
                continue

            # 我们需要论文的某种ID (pwc_id或arxiv_id)
            if not link_data.get("arxiv_id_base") and not link_data.get("pwc_id"):
                logger.warning(
                    f"链接到{link_data.get('hf_model_id')}的论文ID {link_data.get('paper_id')} "
                    f"既没有arxiv_id也没有pwc_id。跳过。"
                )
                continue

            links_to_process.append(link_data)

            if len(links_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    # 修正方法名
                    await neo4j_repo.link_model_to_paper_batch(links_to_process)
                    links_synced += len(links_to_process)
                    logger.info(f"已同步{links_synced}个模型-论文链接...")
                except Exception as e:
                    logger.error(f"将模型-论文链接批次保存到Neo4j时出错: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    links_to_process = []

    except Exception as e:
        logger.error(f"从Postgres获取模型-论文链接时出错: {e}")
        logger.error(traceback.format_exc())

    # 同步剩余链接
    if links_to_process:
        try:
            # 修正方法名
            await neo4j_repo.link_model_to_paper_batch(links_to_process)
            links_synced += len(links_to_process)
        except Exception as e:
            logger.error(f"将剩余模型-论文链接批次保存到Neo4j时出错: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"模型-论文链接同步完成。共同步链接: {links_synced}个")


async def run_sync(
    pg_repo: Optional[PostgresRepository] = None,
    neo4j_repo: Optional[Neo4jRepository] = None,
) -> int:
    """运行完整的数据同步过程。

    Args:
        pg_repo: 可选的PostgreSQL仓库实例
        neo4j_repo: 可选的Neo4j仓库实例

    Returns:
        int: 同步的论文总数
    """
    # 创建连接池(如果未提供)
    should_close_pg = False
    should_close_neo4j = False
    pg_pool = None
    neo4j_driver = None
    neo4j_db = "neo4j"  # 默认Neo4j数据库名

    try:
        # 设置PostgreSQL连接(如果需要)
        if pg_repo is None:
            logger.info("创建PostgreSQL连接池...")
            # 添加断言确保DATABASE_URL不是None
            assert DATABASE_URL is not None, "DATABASE_URL must be set"
            pg_pool = AsyncConnectionPool(
                conninfo=DATABASE_URL,
                min_size=1,
                max_size=5,
            )
            pg_repo = PostgresRepository(pg_pool)
            should_close_pg = True
            logger.info("已创建PostgreSQL连接和仓库。")

        # 设置Neo4j连接(如果需要)
        if neo4j_repo is None:
            logger.info("创建Neo4j连接...")
            # 添加断言确保URI, USER, PASSWORD不是None
            assert NEO4J_URI is not None, "NEO4J_URI must be set"
            assert NEO4J_USER is not None, "NEO4J_USER must be set"
            assert NEO4J_PASSWORD is not None, "NEO4J_PASSWORD must be set"
            neo4j_driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # 移除 database 参数
            neo4j_repo = Neo4jRepository(neo4j_driver)
            should_close_neo4j = True
            logger.info("已创建Neo4j连接和仓库。")

        # 验证连接 (移除)
        # if not await pg_repo.check_connection():
        #     logger.error("PostgreSQL连接测试失败")
        #     return 0
        #
        # if not await neo4j_repo.check_connection():
        #     logger.error("Neo4j连接测试失败")
        #     return 0

        # 执行同步
        # 1. 同步模型
        await sync_hf_models(pg_repo, neo4j_repo)

        # 2. 同步论文和关系
        papers_synced = await sync_papers_and_relations(pg_repo, neo4j_repo)

        # 3. 同步模型-论文链接
        await sync_model_paper_links(pg_repo, neo4j_repo)

        logger.info("所有同步过程完成。")
        return papers_synced

    except Exception as e:
        logger.error(f"同步执行期间出错: {e}")
        logger.error(traceback.format_exc())
        return 0
    finally:
        # 清理资源
        if should_close_pg and pg_pool:
            logger.info("关闭PostgreSQL连接池...")
            await pg_pool.close()

        if should_close_neo4j and neo4j_driver:
            logger.info("关闭Neo4j驱动...")
            await neo4j_driver.close()


async def main(reset_neo4j: bool) -> None:
    """主函数 - 带有可选的Neo4j重置。

    Args:
        reset_neo4j: 如果为True，则在同步前重置Neo4j数据库
    """
    # 创建必要的连接
    pg_pool = None
    neo4j_driver = None

    try:
        logger.info("初始化连接...")

        # 创建PostgreSQL连接池
        # 添加断言确保DATABASE_URL不是None
        assert DATABASE_URL is not None, "DATABASE_URL environment variable must be set"
        pg_pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            min_size=2,
            max_size=10,
        )
        logger.info("PostgreSQL连接池创建成功。")

        # 创建Neo4j驱动
        # 添加断言确保URI, USER, PASSWORD不是None
        assert NEO4J_URI is not None, "NEO4J_URI environment variable must be set"
        assert NEO4J_USER is not None, "NEO4J_USER environment variable must be set"
        assert NEO4J_PASSWORD is not None, "NEO4J_PASSWORD environment variable must be set"
        neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info("Neo4j驱动创建成功。")

        # 创建仓库
        pg_repo = PostgresRepository(pg_pool)
        # 移除 database 参数
        neo4j_repo = Neo4jRepository(neo4j_driver)

        # 如果请求，重置Neo4j数据库
        if reset_neo4j:
            logger.warning("重置标志指定。正在清空Neo4j数据库...")
            # 修正方法名
            await neo4j_repo.reset_database()
            logger.info("Neo4j数据库已清空。")

        # 运行同步
        papers_synced = await run_sync(pg_repo, neo4j_repo)
        logger.info(f"同步完成。共同步{papers_synced}篇论文。")

    except Exception as e:
        logger.error(f"主函数执行期间出错: {e}")
        logger.error(traceback.format_exc())

    finally:
        # 清理资源
        if pg_pool:
            logger.info("关闭PostgreSQL连接池...")
            await pg_pool.close()

        if neo4j_driver:
            logger.info("关闭Neo4j驱动...")
            await neo4j_driver.close()


if __name__ == "__main__":
    # 设置命令行解析
    parser = argparse.ArgumentParser(
        description="将AIGraphX数据从PostgreSQL同步到Neo4j"
    )
    parser.add_argument("--reset", action="store_true", help="在同步前清空Neo4j数据库")
    args = parser.parse_args()

    # 在Windows上需要特定的事件循环策略
    # 修改判断条件
    if sys.platform == "win32":  # Windows
        # 保留原来的策略，如果 mypy 报错可能是环境问题或 mypy 配置问题
        # 如果持续报错，可以考虑移除这部分或尝试 ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行主函数
    asyncio.run(main(args.reset))
    logger.info("程序完成。")
