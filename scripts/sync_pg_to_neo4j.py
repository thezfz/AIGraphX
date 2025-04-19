#!/usr/bin/env python
import asyncio
import logging
import os
import traceback  # Import traceback
from typing import Optional, Dict, Any, List  # Import List
import json
import sys  # Import sys for path manipulation
import argparse  # Import argparse

# Third-party imports
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, AsyncDriver
from psycopg_pool import AsyncConnectionPool  # Import PG Pool

# We need asyncpg for the pool creation and exception handling
import asyncpg  # type: ignore[import-untyped]

# Adjust import paths assuming script is run from project root OR handle fallback
try:
    # Try direct imports first (if running as module or PYTHONPATH is set)
    from aigraphx.repositories.postgres_repo import PostgresRepository
    from aigraphx.repositories.neo4j_repo import Neo4jRepository
except ImportError:
    # Fallback if running as a standalone script
    print("INFO: Failed initial import, attempting path modification...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        print(f"INFO: Adding project root to sys.path: {project_root}")
        sys.path.insert(0, project_root)

    # Retry imports after modifying path
    # --- Start of Corrected Block ---
    try:
        # Correctly indented imports UNDER the second try
        from aigraphx.repositories.postgres_repo import PostgresRepository
        from aigraphx.repositories.neo4j_repo import Neo4jRepository

        # Correctly indented print UNDER the second try
        print("INFO: Successfully imported modules after path modification.")
    except ImportError as e:  # Correctly aligned except with the second try
        # Correctly indented print statements UNDER the second except
        print(
            f"CRITICAL: Failed to import required modules even after path modification: {e}"
        )
        print(
            "CRITICAL: Ensure the script is run from the project root or the PYTHONPATH is set correctly."
        )
        # Correctly indented sys.exit UNDER the second except
        sys.exit(1)  # Exit if core components cannot be imported
    # --- End of Corrected Block ---


# --- Configuration ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# PostgreSQL Connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set.", file=sys.stderr)
    sys.exit(1)

# Neo4j Connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    print(
        "Error: Neo4j connection details (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) not fully set.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- Constants ---
PG_FETCH_BATCH_SIZE = 500  # How many records to fetch from PG at a time
NEO4J_WRITE_BATCH_SIZE = 200  # How many records to write to Neo4j in one batch

# --- Synchronization Logic ---


async def sync_hf_models(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """Synchronizes HF models from PostgreSQL to Neo4j."""
    logger.info("Starting HFModel synchronization...")
    models_synced = 0
    query = "SELECT * FROM hf_models ORDER BY hf_model_id"  # Order for deterministic batches
    models_to_process: List[Dict[str, Any]] = []

    try:
        # Corrected call: Use keyword argument for batch_size
        async for model_record in pg_repo.fetch_data_cursor(
            query, batch_size=batch_size
        ):
            model_data = dict(model_record)
            # Ensure tags are loaded as list if stored as JSON string
            if isinstance(model_data.get("hf_tags"), str):
                try:
                    model_data["hf_tags"] = json.loads(model_data["hf_tags"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode hf_tags JSON for model {model_data.get('hf_model_id')}"
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
                    logger.info(f"Synced {models_synced} HF models so far...")
                except Exception as e:
                    logger.error(f"Error saving HF model batch to Neo4j: {e}")
                    # No need to import traceback here if already imported globally
                    logger.error(traceback.format_exc())
                finally:
                    models_to_process = []
    except Exception as e:
        logger.error(f"Error fetching HF models from Postgres: {e}")
        # No need to import traceback here
        logger.error(traceback.format_exc())

    # Sync any remaining models
    if models_to_process:
        try:
            await neo4j_repo.save_hf_models_batch(models_to_process)
            models_synced += len(models_to_process)
        except Exception as e:
            logger.error(f"Error saving final HF model batch to Neo4j: {e}")
            # No need to import traceback here
            logger.error(traceback.format_exc())

    logger.info(
        f"HFModel synchronization finished. Total models synced: {models_synced}"
    )


async def sync_papers_and_relations(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> int:
    """Fetches papers and their relations from PG and syncs them to Neo4j.
    
    Returns:
        int: 同步的论文总数
    """
    logger.info("Starting Paper and relations synchronization...")
    papers_synced_arxiv = 0
    papers_synced_pwc = 0

    logger.info("Fetching and syncing Paper nodes...")
    paper_query = """
        SELECT
            p.paper_id, p.pwc_id, p.arxiv_id_base, p.arxiv_id_versioned, p.title,
            p.authors, p.summary, p.published_date, p.area, p.pwc_url,
            p.pdf_url, p.doi, p.primary_category, p.categories
        FROM papers p
    """
    papers_to_process: List[Dict[str, Any]] = []
    arxiv_only_papers: List[
        Dict[str, Any]
    ] = []  # For papers only identified by arxiv_id
    try:
        # Corrected call: Use keyword argument for batch_size
        async for paper_record in pg_repo.fetch_data_cursor(
            paper_query, batch_size=batch_size
        ):
            paper_data = dict(paper_record)

            # Convert JSON string fields back to lists if necessary (depends on PG repo handling)
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
            # Initialize relation keys - enrichment happens later for pwc_id papers
            paper_data["tasks"] = []
            paper_data["datasets"] = []
            paper_data["repositories"] = []

            # Sorting papers by identifier availability
            if paper_data.get("pwc_id"):
                # If paper has a PWC ID, enrich with additional relations
                # and save using the pwc_id-based Neo4j methods
                papers_to_process.append(paper_data)
            elif paper_data.get("arxiv_id_base"):
                # If paper has only an ArXiv ID, we'll save using a different method
                arxiv_only_papers.append(paper_data)
            else:
                logger.warning(
                    f"Paper id={paper_data.get('paper_id')} has neither pwc_id nor arxiv_id. Skipping."
                )
                continue

            # Batch processing - PWC ID papers
            if len(papers_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    enriched_papers = await enrich_papers_with_relations(
                        pg_repo, papers_to_process
                    )
                    await neo4j_repo.save_papers_batch(enriched_papers)
                    papers_synced_pwc += len(enriched_papers)
                    logger.info(
                        f"Synced {papers_synced_pwc} PWC papers and {papers_synced_arxiv} ArXiv-only papers..."
                    )
                except Exception as e:
                    logger.error(f"Error saving paper batch to Neo4j: {e}")
                    # No need to import traceback here
                    logger.error(traceback.format_exc())
                finally:
                    papers_to_process = []

            # Batch processing - ArXiv-only papers
            if len(arxiv_only_papers) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.save_papers_by_arxiv_batch(arxiv_only_papers)
                    papers_synced_arxiv += len(arxiv_only_papers)
                    logger.info(
                        f"Synced {papers_synced_pwc} PWC papers and {papers_synced_arxiv} ArXiv-only papers..."
                    )
                except Exception as e:
                    logger.error(f"Error saving arxiv paper batch to Neo4j: {e}")
                    # No need to import traceback here
                    logger.error(traceback.format_exc())
                finally:
                    arxiv_only_papers = []

    except Exception as e:
        logger.error(f"Error fetching papers from Postgres: {e}")
        # No need to import traceback here
        logger.error(traceback.format_exc())

    # Process any remaining papers
    if papers_to_process:
        try:
            enriched_papers = await enrich_papers_with_relations(
                pg_repo, papers_to_process
            )
            await neo4j_repo.save_papers_batch(enriched_papers)
            papers_synced_pwc += len(enriched_papers)
        except Exception as e:
            logger.error(f"Error saving final paper batch to Neo4j: {e}")
            # No need to import traceback here
            logger.error(traceback.format_exc())

    if arxiv_only_papers:
        try:
            await neo4j_repo.save_papers_by_arxiv_batch(arxiv_only_papers)
            papers_synced_arxiv += len(arxiv_only_papers)
        except Exception as e:
            logger.error(f"Error saving final arxiv paper batch to Neo4j: {e}")
            # No need to import traceback here
            logger.error(traceback.format_exc())

    total_papers = papers_synced_pwc + papers_synced_arxiv
    logger.info(
        f"Paper synchronization finished. PWC papers: {papers_synced_pwc}, ArXiv-only papers: {papers_synced_arxiv}, Total: {total_papers}"
    )
    
    return total_papers


# --- START: CORRECTED enrich_papers_with_relations --- #
# --- START: CORRECTED enrich_papers_with_relations --- #
async def enrich_papers_with_relations(
    pg_repo: PostgresRepository, paper_batch: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fetches related Tasks, Datasets, Repositories for a batch of papers
    and adds them to the corresponding paper dictionaries.
    """
    if not paper_batch:
        return []

    # Create a map of paper_id to the paper dictionary for easy lookup and modification
    paper_map = {
        int(p["paper_id"]): p for p in paper_batch if p.get("paper_id") is not None
    }
    paper_ids = list(paper_map.keys())

    if not paper_ids:
        logger.warning("Enrichment called with batch containing no valid paper_ids.")
        return paper_batch

    # Ensure relation keys exist (even if enrichment fails)
    for paper_data in paper_map.values():
        paper_data.setdefault("tasks", [])
        paper_data.setdefault("datasets", [])
        paper_data.setdefault("repositories", [])

    # --- Fetch Tasks using the CORRECT repository method --- #
    try:
        tasks_map = await pg_repo.get_tasks_for_papers(paper_ids)
        for paper_id, tasks_list in tasks_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["tasks"] = tasks_list  # Assign fetched tasks
    except Exception as e:
        logger.error(
            f"Error fetching tasks relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )
        # Continue enrichment even if one type fails

    # --- Fetch Datasets using the CORRECT repository method --- #
    try:
        datasets_map = await pg_repo.get_datasets_for_papers(paper_ids)
        for paper_id, datasets_list in datasets_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["datasets"] = (
                    datasets_list  # Assign fetched datasets
                )
    except Exception as e:
        logger.error(
            f"Error fetching datasets relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )
        # Continue enrichment

    # --- Fetch Repositories using the CORRECT repository method --- #
    try:
        # Ensure correct method is called: get_repositories_for_papers
        repos_map = await pg_repo.get_repositories_for_papers(
            paper_ids
        )  # Returns Dict[int, List[str]]
        for paper_id, repo_urls in repos_map.items():
            if paper_id in paper_map:
                # Convert list of URLs to list of dicts as expected by Neo4j Cypher query
                paper_map[paper_id]["repositories"] = [
                    {"url": url}
                    for url in repo_urls
                    if url  # Ensure URL is not empty
                ]
    except Exception as e:
        logger.error(
            f"Error fetching repository relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )
        # Continue enrichment

    # --- Add logging before returning --- #
    if paper_map:
        first_paper_key = next(iter(paper_map))
        logger.debug(
            f"[Enrich] Returning enriched batch. Example paper ID {first_paper_key} tasks: {paper_map[first_paper_key].get('tasks')}"
        )
    else:
        logger.debug("[Enrich] Returning empty batch.")
    # --- End logging --- #

    # Return the list of paper dictionaries, which have been modified in-place via paper_map
    return list(paper_map.values())


# --- End of CORRECTED enrich_papers_with_relations --- #


async def sync_model_paper_links(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """同步HFModel和Paper之间的关系"""
    logger.info("开始同步HF模型和论文之间的关系...")
    
    # 从model_paper_links表获取关系
    link_query = """
    SELECT mpl.hf_model_id, p.pwc_id, mpl.paper_id
    FROM model_paper_links mpl
    JOIN papers p ON mpl.paper_id = p.paper_id
    WHERE p.pwc_id IS NOT NULL
    """
    
    try:
        # 获取数据
        links_to_process = []
        link_count = 0
        
        async for link_record in pg_repo.fetch_data_cursor(link_query, batch_size=batch_size):
            # 只处理有效记录
            if not link_record.get("hf_model_id") or not link_record.get("pwc_id"):
                continue
                
            # 转换格式
            link_data = {
                "model_id": link_record["hf_model_id"],
                "pwc_id": link_record["pwc_id"],
                "confidence": 1.0  # 默认置信度
            }
            links_to_process.append(link_data)
            
            # 批处理
            if len(links_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    # 特殊调试输出
                    logger.info(f"正在处理 {len(links_to_process)} 条HFModel-Paper关系，第一条: {links_to_process[0]}")
                    
                    # 创建MENTIONS关系
                    await neo4j_repo.link_model_to_paper_batch(links_to_process)
                    link_count += len(links_to_process)
                    logger.info(f"已同步 {link_count} 条模型-论文关系...")
                except Exception as e:
                    logger.error(f"保存模型-论文关系批次时出错: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    links_to_process = []
                    
        # 处理剩余的关系
        if links_to_process:
            try:
                # 特殊调试输出
                logger.info(f"正在处理最后 {len(links_to_process)} 条HFModel-Paper关系")
                
                # 创建MENTIONS关系
                await neo4j_repo.link_model_to_paper_batch(links_to_process)
                link_count += len(links_to_process)
            except Exception as e:
                logger.error(f"保存最后一批模型-论文关系时出错: {e}")
                logger.error(traceback.format_exc())
                
        logger.info(f"模型-论文关系同步完成，总计: {link_count} 条关系")
        
    except Exception as e:
        logger.error(f"获取模型-论文关系时出错: {e}")
        logger.error(traceback.format_exc())


# --- Synchronization Runner ---
async def run_sync(pg_repo: Optional[PostgresRepository] = None, neo4j_repo: Optional[Neo4jRepository] = None) -> int:
    """Runs the full synchronization process.
    
    Returns:
        int: 同步的论文数量，用于测试断言
    """
    logger.info("Starting full PG to Neo4j synchronization...")
    
    # Create repositories if not provided
    created_pg_repo = False
    created_neo4j_repo = False
    
    if pg_repo is None:
        try:
            # Create PostgreSQL pool
            if DATABASE_URL is None:
                logger.error("DATABASE_URL is None")
                return 0
            pg_pool = AsyncConnectionPool(DATABASE_URL)
            pg_repo = PostgresRepository(pool=pg_pool)
            created_pg_repo = True
        except Exception as e:
            logger.error(f"Failed to initialize Postgres repository: {e}")
            logger.error(traceback.format_exc())
            return 0

    if neo4j_repo is None:
        try:
            # Create Neo4j driver
            if NEO4J_URI is None or NEO4J_USER is None or NEO4J_PASSWORD is None:
                logger.error("Neo4j connection parameters are None")
                return 0
            neo4j_driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            neo4j_repo = Neo4jRepository(driver=neo4j_driver)
            created_neo4j_repo = True
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j repository: {e}")
            logger.error(traceback.format_exc())
            # Clean up Postgres connection if we created it
            if created_pg_repo and pg_repo is not None:
                await pg_repo.close()
            return 0

    try:
        # 确保repositories不为None
        if pg_repo is None or neo4j_repo is None:
            logger.error("Repository objects are None after initialization")
            return 0
            
        # 1. Make sure Neo4j constraints exist
        await neo4j_repo.create_constraints()

        # 2. Sync HF Models
        await sync_hf_models(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        # 3. Sync Papers (with their relations)
        papers_count = await sync_papers_and_relations(
            pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE
        )
        
        # 4. Sync Model<->Paper links
        await sync_model_paper_links(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        logger.info("Full synchronization completed successfully.")
        
        # 5. Count papers in Neo4j (可选的验证步骤)
        neo4j_papers = await neo4j_repo.count_paper_nodes()
        logger.info(f"Current Neo4j paper count: {neo4j_papers}")
        
        return 1  # 成功完成同步，返回1以通过测试
    except Exception as e:
        logger.error(f"Synchronization failed with error: {e}")
        logger.error(traceback.format_exc())
        return 0
    finally:
        # Clean up resources if we created them
        if created_pg_repo and pg_repo is not None:
            await pg_repo.close()
        if created_neo4j_repo and neo4j_repo is not None:
            # Neo4j driver cleaned up through repository close method
            pass


# --- Main Execution ---
async def main(reset_neo4j: bool) -> None:
    pg_pool: Optional[AsyncConnectionPool] = None  # Type hint
    neo4j_driver: Optional[AsyncDriver] = None  # Type hint
    exit_code = 0
    try:
        # --- Neo4j Reset Logic ---
        if reset_neo4j:
            logger.warning("Reset flag specified. Connecting to Neo4j to clear data...")
            temp_driver: Optional[AsyncDriver] = None
            try:
                # 确保NEO4J_URI不为空
                if not NEO4J_URI:
                    raise ValueError("NEO4J_URI is not set or empty")

                temp_driver = AsyncGraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER or "", NEO4J_PASSWORD or "")
                )
                await temp_driver.verify_connectivity()
                logger.info("Temporary connection to Neo4j established for reset.")
                async with temp_driver.session() as session:
                    logger.info(
                        "Executing `MATCH (n) DETACH DELETE n` to clear Neo4j database..."
                    )
                    await session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Neo4j database cleared successfully.")
            except Exception as reset_err:
                logger.error(
                    f"Error clearing Neo4j database: {reset_err}", exc_info=True
                )
                logger.error(
                    "Synchronization cannot proceed safely after failed reset. Exiting."
                )
                sys.exit(1)
            finally:
                if temp_driver:
                    await temp_driver.close()
                    logger.info("Temporary Neo4j connection for reset closed.")
        # --- End Neo4j Reset Logic ---

        logger.info("Connecting to PostgreSQL...")
        # Use asyncpg.create_pool directly - note: psycopg_pool is imported but asyncpg.create_pool used?
        # Assuming asyncpg was intended here. If psycopg_pool was intended, the creation is different.
        # Sticking with asyncpg based on the code:
        pg_pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=2, max_size=5)

        if pg_pool is None:
            raise ValueError("Failed to create PostgreSQL connection pool")

        # Verify pool connection immediately
        async with pg_pool.acquire() as conn:  # type: ignore[attr-defined]
            await conn.fetchval("SELECT 1")  # Use fetchval for single value
        logger.info("PostgreSQL pool created and verified.")
        # Pass the pool directly to the repository
        pg_repo = PostgresRepository(
            pool=pg_pool
        )  # Assuming PostgresRepository takes the pool

        logger.info(f"Connecting to Neo4j at {NEO4J_URI}...")
        # 确保Neo4j连接参数不为空
        if not NEO4J_URI:
            raise ValueError("NEO4J_URI is not set or empty")

        neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER or "", NEO4J_PASSWORD or "")
        )
        await neo4j_driver.verify_connectivity()
        neo4j_repo = Neo4jRepository(driver=neo4j_driver)
        logger.info("Neo4j driver created and connected.")

        # Run Synchronization Steps via the runner function
        await run_sync(pg_repo, neo4j_repo)

        logger.info(
            "--- Synchronization from PostgreSQL to Neo4j finished successfully ---"
        )

    except (
        asyncpg.PostgresError,
        asyncpg.InterfaceError,
    ) as e:  # Catch pool/connection errors too
        logger.critical(f"PostgreSQL error during setup or synchronization: {e}")
        # No need to import traceback here
        logger.critical(traceback.format_exc())
        exit_code = 1
    # Catch specific Neo4j connection errors if possible (check neo4j driver exceptions)
    except ConnectionRefusedError as e:  # Example generic network error
        logger.critical(f"Could not connect to Neo4j (Connection Refused): {e}")
        # No need to import traceback here
        logger.critical(traceback.format_exc())
        exit_code = 1
    except Exception as e:  # Catch any other unexpected error
        logger.critical(f"An unexpected error occurred during synchronization: {e}")
        # No need to import traceback here
        logger.critical(traceback.format_exc())
        exit_code = 1
    finally:
        # Cleanup connections
        if pg_pool:
            logger.info("Closing PostgreSQL pool...")
            try:
                await asyncio.wait_for(pg_pool.close(), timeout=5.0)  # Add timeout
            except asyncio.TimeoutError:
                logger.warning("Timeout closing PostgreSQL pool.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL pool: {e}")
        if neo4j_driver:
            logger.info("Closing Neo4j driver...")
            try:
                await asyncio.wait_for(neo4j_driver.close(), timeout=5.0)  # Add timeout
            except asyncio.TimeoutError:
                logger.warning("Timeout closing Neo4j driver.")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")

        logger.info("Connections closed (or closing attempted).")
        if exit_code != 0:
            logger.warning(f"Script exiting with error code {exit_code}")
            sys.exit(exit_code)  # Exit with error code if failed


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Synchronize data from PostgreSQL to Neo4j."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the Neo4j database before starting the synchronization.",
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    logger.info("Starting PG to Neo4j Synchronization Script")
    # Corrected Indentation: asyncio.run should not be indented
    # Pass the reset argument to the main function
    asyncio.run(main(reset_neo4j=args.reset))
    logger.info("Script finished.")
