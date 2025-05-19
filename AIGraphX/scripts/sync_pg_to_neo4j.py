#!/usr/bin/env python
import asyncio
import logging
import os
import traceback  # Import traceback
from typing import Optional, Dict, Any, List, Union  # Import List and Union
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
    query = "SELECT hf_model_id, hf_author, hf_sha, hf_last_modified, hf_downloads, hf_likes, hf_tags, hf_pipeline_tag, hf_library_name, hf_readme_content, hf_dataset_links FROM hf_models ORDER BY hf_model_id"
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

            # Prepare data for Neo4j
            # The neo4j_repo.save_hf_models_batch expects the ID key in the dictionary to be 'model_id' (lowercase)
            # which it then uses for the 'modelId' (camelCase) property in Cypher.
            model_data_for_neo4j = {
                "model_id": model_data.get("hf_model_id"), # Key for the repo method
                "author": model_data.get("hf_author"),
                "sha": model_data.get("hf_sha"),
                "last_modified": model_data.get("hf_last_modified"),
                "tags": model_data.get("hf_tags") or [],
                "pipeline_tag": model_data.get("hf_pipeline_tag"),
                "downloads": model_data.get("hf_downloads"),
                "likes": model_data.get("hf_likes"),
                "library_name": model_data.get("hf_library_name"),
                "hf_readme_content": model_data.get("hf_readme_content"),
                "hf_dataset_links": json.loads(model_data["hf_dataset_links"])
                                    if isinstance(model_data.get("hf_dataset_links"), str)
                                    else (model_data.get("hf_dataset_links") or []),
            }
            
            model_data_for_neo4j_cleaned = {k: v for k, v in model_data_for_neo4j.items() if v is not None}

            models_to_process.append(model_data_for_neo4j_cleaned)

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
            p.pdf_url, p.doi, p.primary_category, p.categories,
            p.conference
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
            
            # Ensure conference is included (it's fetched by the query now)
            # paper_data["conference"] will be set from dict(paper_record)

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
async def enrich_papers_with_relations(
    pg_repo: PostgresRepository, paper_batch: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fetches related Tasks, Datasets, Methods, and Repositories for a batch of papers
    and adds them to the corresponding paper dictionaries.
    """
    if not paper_batch:
        return []

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
        paper_data.setdefault("methods", [])  # Add default for methods
        paper_data.setdefault("repositories", [])

    # --- Fetch Tasks using the CORRECT repository method --- #
    try:
        tasks_map = await pg_repo.get_tasks_for_papers(paper_ids)
        for paper_id, tasks_list in tasks_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["tasks"] = tasks_list
    except Exception as e:
        logger.error(
            f"Error fetching tasks relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    # --- Fetch Datasets using the CORRECT repository method --- #
    try:
        datasets_map = await pg_repo.get_datasets_for_papers(paper_ids)
        for paper_id, datasets_list in datasets_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["datasets"] = datasets_list
    except Exception as e:
        logger.error(
            f"Error fetching datasets relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    # --- Fetch Methods using the CORRECT repository method --- #
    # Assuming a method pg_repo.get_methods_for_papers exists
    try:
        methods_map = await pg_repo.get_methods_for_papers(paper_ids)  # Call the method
        for paper_id, methods_list in methods_map.items():
            if paper_id in paper_map:
                paper_map[paper_id]["methods"] = methods_list  # Assign fetched methods
    except AttributeError:
        logger.error(
            f"PostgresRepository does not have a 'get_methods_for_papers' method. Methods cannot be enriched."
        )
    except Exception as e:
        logger.error(
            f"Error fetching methods relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    # --- Fetch Repositories using the CORRECT repository method --- #
    try:
        repos_map = await pg_repo.get_repositories_for_papers(paper_ids)
        for paper_id, repo_list_of_dicts in repos_map.items(): # Expecting a list of dicts
            if paper_id in paper_map:
                # Directly assign the list of repository dictionaries
                # Each dict should contain: url, stars, is_official, framework, license, language
                paper_map[paper_id]["repositories"] = repo_list_of_dicts
    except Exception as e:
        logger.error(
            f"Error fetching repository relations for paper IDs {paper_ids[:10]}...: {e}",
            exc_info=True,
        )

    if paper_map:
        first_paper_key = next(iter(paper_map))
        logger.debug(
            f"[Enrich] Returning enriched batch. Example paper ID {first_paper_key} tasks: {paper_map[first_paper_key].get('tasks')}, methods: {paper_map[first_paper_key].get('methods')}"  # Log methods too
        )
    else:
        logger.debug("[Enrich] Returning empty batch.")

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

        async for link_record in pg_repo.fetch_data_cursor(
            link_query, batch_size=batch_size
        ):
            # 只处理有效记录
            if not link_record.get("hf_model_id") or not link_record.get("pwc_id"):
                continue

            # 转换格式
            link_data = {
                "model_id": link_record["hf_model_id"],
                "pwc_id": link_record["pwc_id"],
                "confidence": 1.0,  # 默认置信度
            }
            links_to_process.append(link_data)

            # 批处理
            if len(links_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    # 特殊调试输出
                    logger.info(
                        f"正在处理 {len(links_to_process)} 条HFModel-Paper关系，第一条: {links_to_process[0]}"
                    )

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


async def sync_model_derivations(
    pg_repo: PostgresRepository, neo4j_repo: Neo4jRepository, batch_size: int = 100
) -> None:
    """同步HFModel之间的DERIVED_FROM关系"""
    logger.info("开始同步模型之间的派生关系 (DERIVED_FROM)...")

    # 从hf_models表获取hf_model_id和hf_base_models
    # 确保在 load_postgres.py 中使用的列名是 hf_base_models
    query = """
    SELECT hf_model_id, hf_base_models 
    FROM hf_models 
    WHERE hf_base_models IS NOT NULL AND hf_base_models <> 'null' AND hf_base_models <> '[]'
    """

    try:
        links_to_process: List[Dict[str, str]] = []
        link_count = 0

        # Corrected call: Use keyword argument for batch_size
        async for record in pg_repo.fetch_data_cursor(
            query, batch_size=batch_size
        ):
            current_model_id = record.get("hf_model_id")
            base_models_data = record.get("hf_base_models") # This is likely a JSON string or already parsed by psycopg

            if not current_model_id or not base_models_data:
                continue

            parsed_base_models: Optional[Union[str, List[str]]] = None
            if isinstance(base_models_data, str): # If it's a JSON string from DB
                try:
                    parsed_base_models = json.loads(base_models_data)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        f"无法解析模型 {current_model_id} 的 hf_base_models JSON: {base_models_data}"
                    )
                    continue # Skip this record if parsing fails
            elif isinstance(base_models_data, (list, dict)): # If psycopg already parsed it
                 # Assuming if it's a dict, it's an error or unexpected format for base_models list/str
                 if isinstance(base_models_data, dict):
                    logger.warning(f"模型 {current_model_id} 的 hf_base_models 格式错误 (应为列表或字符串): {base_models_data}")
                    continue
                 parsed_base_models = base_models_data
            else:
                logger.warning(f"模型 {current_model_id} 的 hf_base_models 类型未知: {type(base_models_data)}")
                continue

            if not parsed_base_models:
                continue

            if isinstance(parsed_base_models, str):
                # Single base model ID
                links_to_process.append(
                    {"source_model_id": current_model_id, "base_model_id": parsed_base_models}
                )
            elif isinstance(parsed_base_models, list):
                # List of base model IDs
                for base_id in parsed_base_models:
                    if isinstance(base_id, str) and base_id.strip(): # Ensure it's a non-empty string
                        links_to_process.append(
                            {"source_model_id": current_model_id, "base_model_id": base_id.strip()}
                        )
                    else:
                        logger.debug(f"模型 {current_model_id} 的 hf_base_models 列表中包含无效条目: {base_id}")
            else:
                logger.warning(f"模型 {current_model_id} 解析后的 hf_base_models 类型无效: {type(parsed_base_models)}")

            # Batch processing
            if len(links_to_process) >= NEO4J_WRITE_BATCH_SIZE:
                try:
                    await neo4j_repo.link_models_derived_from_batch(links_to_process)
                    link_count += len(links_to_process)
                    logger.info(f"已同步 {link_count} 条模型派生关系...")
                except Exception as e:
                    logger.error(f"保存模型派生关系批次时出错: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    links_to_process = []

        # Process any remaining links
        if links_to_process:
            try:
                await neo4j_repo.link_models_derived_from_batch(links_to_process)
                link_count += len(links_to_process)
            except Exception as e:
                logger.error(f"保存最后一批模型派生关系时出错: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"模型派生关系同步完成，总计: {link_count} 条关系")

    except Exception as e:
        logger.error(f"获取模型派生关系时出错: {e}")
        logger.error(traceback.format_exc())


# --- Synchronization Runner ---
async def run_sync(
    pg_repo: Optional[PostgresRepository] = None,
    neo4j_repo: Optional[Neo4jRepository] = None,
) -> int:
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
            pg_pool = AsyncConnectionPool(conninfo=DATABASE_URL, open=True)
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

        # 1. Make sure Neo4j constraints exist - Changed to create_constraints_and_indexes
        await neo4j_repo.create_constraints_and_indexes()

        # 2. Sync HF Models
        await sync_hf_models(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        # 3. Sync Papers (with their relations)
        papers_count = await sync_papers_and_relations(
            pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE
        )

        # 4. Sync Model<->Paper links
        await sync_model_paper_links(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        # 5. Sync Model<->Model (DERIVED_FROM) links
        await sync_model_derivations(pg_repo, neo4j_repo, PG_FETCH_BATCH_SIZE)

        logger.info("Full synchronization completed successfully.")

        # 6. Count papers in Neo4j (可选的验证步骤)
        neo4j_papers = await neo4j_repo.count_paper_nodes()
        logger.info(f"Current Neo4j paper count: {neo4j_papers}")

        return papers_count  # 返回实际同步的论文数量
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
    """Main function to run the synchronization process."""
    pg_pool = None
    neo4j_driver = None
    pg_repo = None  # Initialize repo variable
    neo4j_repo = None  # Initialize repo variable
    total_papers_synced = 0

    try:
        # --- Initialize Connections ---
        logger.info(f"Initializing PostgreSQL pool for {DATABASE_URL}...")
        # Add assertion for DATABASE_URL
        assert DATABASE_URL is not None, "DATABASE_URL environment variable must be set"
        # Explicitly create the pool instance
        pg_pool = AsyncConnectionPool(conninfo=DATABASE_URL, open=True)
        logger.info("PostgreSQL pool initialized.")

        logger.info(f"Initializing Neo4j driver for {NEO4J_URI}...")
        # Add assertions for Neo4j connection details
        assert NEO4J_URI is not None, "NEO4J_URI environment variable must be set"
        assert NEO4J_USER is not None, "NEO4J_USER environment variable must be set"
        assert NEO4J_PASSWORD is not None, (
            "NEO4J_PASSWORD environment variable must be set"
        )
        # Explicitly create the driver instance
        neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        logger.info("Neo4j driver initialized.")

        # --- Create Repository Instances with initialized pool/driver ---
        # Pass the initialized pool and driver instances
        pg_repo = PostgresRepository(pool=pg_pool)
        neo4j_repo = Neo4jRepository(driver=neo4j_driver)
        logger.info("Repositories initialized.")

        # --- Optional: Reset Neo4j if requested ---
        if reset_neo4j:
            logger.warning("Resetting Neo4j database...")
            await neo4j_repo.reset_database()  # Assuming this method exists
            logger.info("Neo4j database reset complete.")
        else:
            logger.info("Skipping Neo4j database reset.")

        # --- Run Synchronization ---
        logger.info("--- Starting synchronization from PostgreSQL to Neo4j ---")
        total_papers_synced = await run_sync(pg_repo=pg_repo, neo4j_repo=neo4j_repo)

    except asyncpg.exceptions.CannotConnectNowError as pg_conn_err:
        logger.critical(
            f"FATAL: Could not connect to PostgreSQL at {DATABASE_URL}. Check if DB is running and accessible. Error: {pg_conn_err}"
        )
    except ConnectionRefusedError as neo4j_conn_err:
        logger.critical(
            f"FATAL: Could not connect to Neo4j at {NEO4J_URI}. Check if DB is running and accessible. Error: {neo4j_conn_err}"
        )
    except Exception as e:
        logger.critical(f"An unexpected error occurred during synchronization: {e}")
        # No need to import traceback here
        logger.critical(traceback.format_exc())
    finally:
        # --- Close Connections ---
        if pg_pool:
            logger.info("Closing PostgreSQL pool...")
            await pg_pool.close()
        if neo4j_driver:
            logger.info("Closing Neo4j driver...")
            await neo4j_driver.close()
        logger.info("Connections closed (or closing attempted).")
        # Log final paper count if sync was attempted
        if total_papers_synced > 0:
            logger.info(
                f"Final count of papers synced in this run: {total_papers_synced}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize data from PostgreSQL to Neo4j for AIGraphX."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the entire Neo4j database before starting synchronization.",
    )
    args = parser.parse_args()

    # Run the main asynchronous function
    try:
        asyncio.run(main(reset_neo4j=args.reset))
        logger.info("Script finished successfully.")
    except Exception as e:
        # Catch errors happening during asyncio.run() itself if any
        logger.critical(f"Script execution failed: {e}", exc_info=True)
        sys.exit(1)
