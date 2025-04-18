import logging
from typing import List, Optional, Dict, Any, Literal, Tuple
from neo4j import (
    AsyncDriver,
    AsyncSession,
    Query,
    AsyncTransaction,
    AsyncManagedTransaction,
)
import os
from dotenv import load_dotenv
from datetime import datetime, date  # Import datetime for Neo4j date handling
import traceback

# Keep loading .env for potential other uses or default values if driver isn't passed?
# Or remove if repo strictly relies on injected driver.
# For now, keeping it doesn't hurt.
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Default connection details might be removed if driver injection is enforced
# NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logger = logging.getLogger(__name__)


class Neo4jRepository:
    """Repository class for interacting with the Neo4j database via an injected driver."""

    def __init__(self, driver: AsyncDriver):
        """Initializes the repository with an externally managed async Neo4j driver."""
        if not driver:
            # Ensure a driver is actually passed
            logger.error("Neo4jRepository initialized without a valid AsyncDriver.")
            raise ValueError("AsyncDriver instance is required.")
        self.driver = driver
        logger.debug("Neo4jRepository initialized with provided driver.")

    async def create_constraints(self) -> None:
        """Creates necessary unique constraints and indexes in Neo4j."""
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot create constraints: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        # Updated list of constraints and indexes
        queries = [
            # Unique constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.pwc_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:HFModel) REQUIRE m.model_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Repository) REQUIRE r.url IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ar:Area) REQUIRE ar.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Framework) REQUIRE f.name IS UNIQUE;",
            # Indexes (add more as needed for performance)
            "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id_base);",
            "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title);",  # Keep title index
            "CREATE INDEX IF NOT EXISTS FOR (m:HFModel) ON (m.author);",  # Keep author index
            "CREATE INDEX IF NOT EXISTS FOR (e:Method) ON (e.name);",  # Assuming Method nodes are still planned/used elsewhere
        ]

        async with self.driver.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                    logger.info(f"Successfully applied Neo4j DDL: {query}")
                except Exception as e:
                    logger.error(f"Error applying Neo4j DDL: {query} - {e}")
                    # Ensure temporary raise is removed
                    # raise # Removed

    async def _execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Executes a single write query within a managed transaction."""
        if not self.driver or not hasattr(self.driver, "session"):  # Basic check
            logger.error("Neo4j driver not available or invalid in _execute_query")
            raise ConnectionError("Neo4j driver is not available.")

        async with self.driver.session() as session:
            try:
                # Use execute_write for automatic transaction management
                await session.execute_write(lambda tx: tx.run(query, parameters))
                # Reduced log level from info to debug for potentially frequent operations
                logger.debug(
                    f"Successfully executed write query: {str(query)[:100]}..."
                )
            except Exception as e:
                logger.error(f"Error executing Neo4j write query: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Parameters: {parameters}")
                raise  # Re-raise to indicate failure

    async def create_or_update_paper_node(
        self, pwc_id: str, title: Optional[str] = None
    ) -> None:
        """Creates or updates a Paper node identified by pwc_id."""
        query = (
            "MERGE (p:Paper {pwc_id: $pwc_id}) "
            "ON CREATE SET p.title = $title, p.created_at = timestamp() "
            "ON MATCH SET p.title = $title, p.updated_at = timestamp()"
        )
        parameters = {"pwc_id": pwc_id, "title": title or "N/A"}
        # Use the existing helper for single writes
        await self._execute_query(query, parameters)
        # Removed duplicate logger.info from here, _execute_query handles logging

    async def link_paper_to_entity(
        self, pwc_id: str, entity_label: str, entity_name: str, relationship: str
    ) -> None:
        """Creates a related entity node (if not exists) and links it to a paper within a single transaction."""
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error(
                "Neo4j driver not available or invalid in link_paper_to_entity"
            )
            raise ConnectionError("Neo4j driver is not available.")

        query_paper = (
            "MERGE (p:Paper {pwc_id: $pwc_id}) "
            "ON CREATE SET p.created_at = timestamp() "  # Simplified, title might be missing
            "ON MATCH SET p.updated_at = timestamp() "
            "RETURN p"
        )
        query_link = (
            f"MATCH (p:Paper {{pwc_id: $pwc_id}}) "
            f"MERGE (e:{entity_label} {{name: $entity_name}}) "
            f"ON CREATE SET e.created_at = timestamp() "
            f"MERGE (p)-[r:{relationship}]->(e) "
            f"ON CREATE SET r.created_at = timestamp()"
        )

        async with self.driver.session() as session:

            async def _link_tx(tx: AsyncManagedTransaction) -> None:
                # 1. Ensure Paper node exists
                await tx.run(query_paper, {"pwc_id": pwc_id})
                # 2. Create/Merge Entity and Link
                await tx.run(query_link, {"pwc_id": pwc_id, "entity_name": entity_name})

            try:
                await session.execute_write(_link_tx)
                logger.debug(
                    f"Linked Paper {pwc_id} -[{relationship}]-> {entity_label} {entity_name}"
                )
            except Exception as e:
                logger.error(
                    f"Error linking paper {pwc_id} to {entity_label} {entity_name}: {e}"
                )
                raise

    async def link_paper_to_task(self, pwc_id: str, task_name: str) -> None:
        """Links a paper to a Task node."""
        await self.link_paper_to_entity(pwc_id, "Task", task_name, "HAS_TASK")

    async def link_paper_to_dataset(self, pwc_id: str, dataset_name: str) -> None:
        """Links a paper to a Dataset node."""
        await self.link_paper_to_entity(pwc_id, "Dataset", dataset_name, "USES_DATASET")

    async def link_paper_to_method(self, pwc_id: str, method_name: str) -> None:
        """Links a paper to a Method node."""
        await self.link_paper_to_entity(pwc_id, "Method", method_name, "USES_METHOD")

    async def save_papers_batch(self, papers_data: List[Dict[str, Any]]) -> None:
        """
        Saves a batch of paper data to Neo4j using UNWIND, including related tasks, datasets, authors, area, and repositories/frameworks.
        Assumes input dictionaries contain necessary fields like 'pwc_id', 'arxiv_id_base', 'title', 'summary',
        'published_date', 'authors' (list), 'area', 'tasks', 'datasets', 'repositories' (list of dicts).
        Primarily merges Paper based on pwc_id. Handling for null pwc_id should be done by the caller.
        """
        if not papers_data:
            return

        # --- Add direct print for debugging --- #
        print("\n--- DEBUG: Entering Neo4j save_papers_batch ---")
        if papers_data:
            print(
                f"[Neo4j Save Papers DEBUG PRINT] Received batch size: {len(papers_data)}"
            )
            print(
                f"[Neo4j Save Papers DEBUG PRINT] First paper data received: {papers_data[0]}"
            )
            print(
                f"[Neo4j Save Papers DEBUG PRINT] First paper tasks: {papers_data[0].get('tasks')}"
            )
        else:
            print("[Neo4j Save Papers DEBUG PRINT] Received empty batch.")
        print("--- END DEBUG PRINT ---\n")
        # --- End direct print --- #

        # --- Add logging at the beginning --- #
        if papers_data:
            logger.debug(
                f"[Neo4j Save Papers] Received batch of {len(papers_data)}. Example paper pwc_id: {papers_data[0].get('pwc_id')}, tasks: {papers_data[0].get('tasks')}"
            )
        else:
            logger.debug("[Neo4j Save Papers] Received empty batch.")
        # --- End logging --- #

        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot save papers batch: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        # Restore the efficient UNWIND $batch query with all parts uncommented
        query = """
        UNWIND $batch AS paper_props

        // Merge Paper node - Using pwc_id as primary merge key
        MERGE (p:Paper {pwc_id: paper_props.pwc_id})
        ON CREATE SET
            p.arxiv_id_base = paper_props.arxiv_id_base,
            p.arxiv_id_versioned = paper_props.arxiv_id_versioned,
            p.title = paper_props.title,
            p.summary = paper_props.summary,
            p.published_date = CASE WHEN paper_props.published_date IS NOT NULL THEN date(paper_props.published_date) ELSE null END,
            p.area = paper_props.area,
            p.pwc_url = paper_props.pwc_url,
            p.pdf_url = paper_props.pdf_url,
            p.doi = paper_props.doi,
            p.primary_category = paper_props.primary_category,
            p.categories = paper_props.categories, // Restore list assignment
            p.created_at = timestamp()
        ON MATCH SET
            p.arxiv_id_base = CASE WHEN p.arxiv_id_base IS NULL THEN paper_props.arxiv_id_base ELSE p.arxiv_id_base END,
            p.arxiv_id_versioned = paper_props.arxiv_id_versioned,
            p.title = paper_props.title,
            p.summary = paper_props.summary,
            p.published_date = CASE WHEN paper_props.published_date IS NOT NULL THEN date(paper_props.published_date) ELSE p.published_date END,
            p.area = paper_props.area,
            p.pwc_url = paper_props.pwc_url,
            p.pdf_url = paper_props.pdf_url,
            p.doi = paper_props.doi,
            p.primary_category = paper_props.primary_category,
            p.categories = paper_props.categories, // Restore list assignment
            p.updated_at = timestamp()

        // Restore relationship processing
        // Merge Area relationship conditionally using FOREACH HACK
        FOREACH (ignoreMe IN CASE WHEN paper_props.area IS NOT NULL AND paper_props.area <> '' THEN [1] ELSE [] END |
            MERGE (ar:Area {name: paper_props.area})
            ON CREATE SET ar.created_at = timestamp()
            MERGE (p)-[r_area:HAS_AREA]->(ar)
            ON CREATE SET r_area.created_at = timestamp()
        )

        // Merge Author relationships
        FOREACH (author_name IN paper_props.authors | // Assumes authors is a list
            MERGE (a:Author {name: author_name})
            ON CREATE SET a.created_at = timestamp()
            MERGE (a)-[r_auth:AUTHORED]->(p) // Relationship from Author to Paper
            ON CREATE SET r_auth.created_at = timestamp()
        )

        // Merge Task relationships
        FOREACH (task_name IN paper_props.tasks | // Assumes tasks is a list
            MERGE (t:Task {name: task_name})
            ON CREATE SET t.created_at = timestamp()
            MERGE (p)-[r_task:HAS_TASK]->(t)
            ON CREATE SET r_task.created_at = timestamp()
        )

        // Merge Dataset relationships
        FOREACH (dataset_name IN paper_props.datasets | // Assumes datasets is a list
            MERGE (d:Dataset {name: dataset_name})
            ON CREATE SET d.created_at = timestamp()
            MERGE (p)-[r_data:USES_DATASET]->(d)
            ON CREATE SET r_data.created_at = timestamp()
        )

        // Merge Repository and Framework relationships
        FOREACH (repo_data IN paper_props.repositories | // Assumes repositories is a list of maps
            // Ensure repo_data and url are valid before merging Repo
            FOREACH (ignoreMe1 IN CASE WHEN repo_data IS NOT NULL AND repo_data.url IS NOT NULL AND repo_data.url <> '' THEN [1] ELSE [] END |
                MERGE (repo:Repository {url: repo_data.url})
                ON CREATE SET
                    repo.stars = repo_data.stars,
                    repo.is_official = repo_data.is_official,
                    repo.framework = repo_data.framework,
                    repo.created_at = timestamp()
                ON MATCH SET
                    repo.stars = repo_data.stars,
                    repo.is_official = repo_data.is_official,
                    repo.framework = repo_data.framework,
                    repo.updated_at = timestamp()
                MERGE (p)-[r_repo:HAS_REPOSITORY]->(repo)
                ON CREATE SET r_repo.created_at = timestamp()

                // Conditionally merge Framework and relationship inside the repo loop
                FOREACH (ignoreMe2 IN CASE WHEN repo_data.framework IS NOT NULL AND repo_data.framework <> '' THEN [1] ELSE [] END |
                    MERGE (f:Framework {name: repo_data.framework})
                    ON CREATE SET f.created_at = timestamp()
                    MERGE (repo)-[r_fw:USES_FRAMEWORK]->(f)
                    ON CREATE SET r_fw.created_at = timestamp()
                )
            )
        )
        """

        # Prepare batch data, ensuring necessary fields and types, including lists
        prepared_batch = []
        for paper in papers_data:
            if not paper.get(
                "pwc_id"
            ):  # This method still requires pwc_id for the initial MERGE
                logger.warning(
                    f"Skipping paper in batch save due to missing 'pwc_id': arxiv={paper.get('arxiv_id_base')}"
                )
                continue

            # Safely prepare repository data
            repositories = []
            if paper.get("repositories"):
                for repo in paper.get("repositories", []):
                    if repo and repo.get("url"):  # Ensure repo and url exist
                        repositories.append(
                            {
                                "url": repo["url"],
                                "stars": repo.get("stars"),  # Allow None
                                "is_official": repo.get(
                                    "is_official", False
                                ),  # Default to False
                                "framework": repo.get("framework"),  # Allow None
                            }
                        )

            # Restore list parameters in the dictionary passed to Cypher
            prepared_paper = {
                "pwc_id": paper["pwc_id"],
                "arxiv_id_base": paper.get("arxiv_id_base"),
                "arxiv_id_versioned": paper.get("arxiv_id_versioned"),
                "title": paper.get("title"),
                "summary": paper.get("summary"),
                "published_date": paper.get("published_date"),  # Pass as string
                "area": paper.get("area"),
                "pwc_url": paper.get("pwc_url"),
                "pdf_url": paper.get("pdf_url"),
                "doi": paper.get("doi"),
                "primary_category": paper.get("primary_category"),
                "authors": paper.get("authors") or [],
                "tasks": paper.get("tasks") or [],
                "datasets": paper.get("datasets") or [],
                "methods": paper.get("methods") or [],  # Include if still relevant
                "repositories": repositories,  # Use cleaned list
                "categories": paper.get("categories") or [],
            }
            prepared_batch.append(prepared_paper)

        if not prepared_batch:
            logger.info("No valid paper data (with pwc_id) found in batch to save.")
            return

        async def _run_batch_tx(tx: AsyncManagedTransaction) -> None:
            await tx.run(query, batch=prepared_batch)

        # Restore the single transaction execution using execute_write
        async with self.driver.session() as session:
            try:
                await session.execute_write(_run_batch_tx)
                logger.info(
                    f"Successfully processed batch of {len(prepared_batch)} papers (merged/updated with relations) in Neo4j."
                )
            except Exception as e:
                logger.error(
                    f"Error saving papers batch (with relations) to Neo4j: {e}"
                )
                # Consider logging details about the batch that failed
                raise

    async def save_hf_models_batch(self, models_data: List[Dict[str, Any]]) -> None:
        """Saves a batch of Hugging Face model data to Neo4j using MERGE."""
        if not models_data:
            return

        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot save HF models batch: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        # Prepare data for UNWIND
        params_list = []
        for model in models_data:
            params = {
                "model_id": model.get("model_id"),
                "author": model.get("author"),
                "sha": model.get("sha"),
                # Neo4j driver v5+ handles Python datetime, use datetime() for safety
                "last_modified": model.get("last_modified"),
                "tags": model.get("tags") or [],  # Ensure list
                "pipeline_tag": model.get("pipeline_tag"),
                "downloads": model.get("downloads"),
                "likes": model.get("likes"),
                "library_name": model.get("library_name"),
            }
            # Filter out models without a model_id and None values for properties
            # (MERGE SET with null might erase existing values depending on Cypher version/settings)
            if params["model_id"]:
                # Keep None values if Neo4j/Cypher handles them as expected (usually overwrites)
                # If you want to avoid overwriting with None, filter here:
                # params_list.append({k: v for k, v in params.items() if v is not None})
                params_list.append(params)
            else:
                logger.warning(
                    f"Skipping HF model data due to missing model_id: {model}"
                )

        if not params_list:
            logger.info("No valid HF model data found in batch to save.")
            return

        # Cypher query using UNWIND and MERGE
        query = """
        UNWIND $batch AS model_props
        MERGE (m:HFModel {model_id: model_props.model_id})
        ON CREATE SET
            m += model_props, // Set all properties provided in the map
            m.created_at = timestamp(),
            // Ensure last_modified is treated as datetime by Neo4j
            m.last_modified = CASE WHEN model_props.last_modified IS NOT NULL THEN datetime(model_props.last_modified) ELSE null END
        ON MATCH SET
            m += model_props, // Update all properties provided
            m.updated_at = timestamp(),
            // Ensure last_modified is treated as datetime by Neo4j
            m.last_modified = CASE WHEN model_props.last_modified IS NOT NULL THEN datetime(model_props.last_modified) ELSE null END
        """
        # The `m += model_props` syntax is a convenient way to set/update properties from a map.
        # We handle last_modified explicitly using datetime() for robustness.

        # --- ADDED: Merge Task and Relationship --- < FIX for test_save_hf_models_batch_integration >
        # Conditionally merge Task and relationship using FOREACH HACK if pipeline_tag exists
        query += """
        // Conditionally merge Task and relationship using FOREACH HACK if pipeline_tag exists
        FOREACH (ignoreMe IN CASE WHEN model_props.pipeline_tag IS NOT NULL AND model_props.pipeline_tag <> '' THEN [1] ELSE [] END |
            MERGE (t:Task {name: model_props.pipeline_tag})
            ON CREATE SET t.created_at = timestamp()
            MERGE (m)-[r_task:HAS_TASK]->(t)
            ON CREATE SET r_task.created_at = timestamp()
        )
        """

        async def _run_batch_tx(tx: AsyncManagedTransaction) -> None:
            await tx.run(query, batch=params_list)

        async with self.driver.session() as session:
            try:
                await session.execute_write(_run_batch_tx)
                logger.info(
                    f"Successfully saved batch of {len(params_list)} HF models to Neo4j."
                )
            except Exception as e:
                logger.error(f"Error saving HF models batch to Neo4j: {e}")
                # Consider logging details about the batch that failed
                raise

    async def count_paper_nodes(self) -> int:
        """Counts the total number of :Paper nodes in the Neo4j database."""
        query = "MATCH (p:Paper) RETURN count(p) AS count"
        count = -1
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Neo4j driver not available or invalid in count_paper_nodes")
            return count

        async def _count_papers_tx(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(query)
            record = await result.single()
            return record["count"] if record else 0

        async with self.driver.session() as session:
            try:
                count = await session.execute_read(_count_papers_tx)
                # Log level reduced to debug as this might be called frequently
                logger.debug(f"Found {count} :Paper nodes in Neo4j.")
            except Exception as e:
                logger.error(f"Error counting Paper nodes in Neo4j: {e}")
                count = -1
        return count

    async def count_hf_models(self) -> int:
        """Counts the total number of :HFModel nodes in the Neo4j database."""
        query = "MATCH (m:HFModel) RETURN count(m) AS count"
        count = -1
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Neo4j driver not available or invalid in count_hf_models")
            return count

        async def _count_models_tx(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(query)
            record = await result.single()
            return record["count"] if record else 0

        async with self.driver.session() as session:
            try:
                count = await session.execute_read(_count_models_tx)
                # Log level reduced to debug
                logger.debug(f"Found {count} :HFModel nodes in Neo4j.")
            except Exception as e:
                logger.error(f"Error counting HFModel nodes in Neo4j: {e}")
                count = -1
        return count

    async def get_paper_neighborhood(self, pwc_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the 1-hop graph neighborhood for a given paper ID from Neo4j.
        Returns data structured for the GraphData model.
        """
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error(
                "Neo4j driver not available or invalid in get_paper_neighborhood"
            )
            return None

        # Cypher query to get the center paper, its 1-hop neighbors, and relationships
        query = """
        MATCH (center:Paper {pwc_id: $pwc_id})
        OPTIONAL MATCH (center)-[r]-(neighbor)
        RETURN center,
               collect(DISTINCT r) as relationships,
               collect(DISTINCT neighbor) as neighbors
        """
        parameters = {"pwc_id": pwc_id}

        async def _get_neighborhood_tx(
            tx: AsyncManagedTransaction,
        ) -> Optional[Dict[str, Any]]:
            result = await tx.run(query, parameters)
            record = await result.single()
            if not record or not record["center"]:
                logger.warning(f"Paper with pwc_id {pwc_id} not found in Neo4j.")
                return None  # Paper itself not found

            center_node_data = record["center"]
            neighbor_nodes_data = record["neighbors"] if record["neighbors"] else []
            relationships_data = (
                record["relationships"] if record["relationships"] else []
            )

            nodes = []
            relationships = []

            processed_node_ids = set()

            # Process center node
            center_neo4j_id = (
                center_node_data.element_id
            )  # Or center_node_data.id based on driver version/usage
            node_props = dict(center_node_data)
            node_id = node_props.pop("pwc_id", None) or node_props.pop(
                "model_id", center_neo4j_id
            )  # Use actual ID if available
            node_label = (
                list(center_node_data.labels)[0]
                if center_node_data.labels
                else "Unknown"
            )
            node_title = node_props.get(
                "title", node_props.get("name", node_id)
            )  # Get a display label

            nodes.append(
                {
                    "id": node_id,
                    "label": node_title,
                    "type": node_label,
                    "properties": node_props,
                }
            )
            processed_node_ids.add(node_id)  # Use the actual ID

            # Process neighbor nodes
            for neighbor in neighbor_nodes_data:
                neighbor_neo4j_id = neighbor.element_id
                neighbor_props = dict(neighbor)
                neighbor_id = (
                    neighbor_props.pop("pwc_id", None)
                    or neighbor_props.pop("model_id", None)
                    or neighbor_props.pop("name", neighbor_neo4j_id)
                )
                neighbor_label = (
                    list(neighbor.labels)[0] if neighbor.labels else "Unknown"
                )
                neighbor_title = neighbor_props.get(
                    "title", neighbor_props.get("name", neighbor_id)
                )

                # Avoid adding duplicates if a neighbor was somehow fetched twice
                if neighbor_id not in processed_node_ids:
                    nodes.append(
                        {
                            "id": neighbor_id,
                            "label": neighbor_title,
                            "type": neighbor_label,
                            "properties": neighbor_props,
                        }
                    )
                    processed_node_ids.add(neighbor_id)

            # Process relationships
            for rel in relationships_data:
                source_node = rel.start_node
                target_node = rel.end_node

                # Extract actual IDs for source and target based on properties
                source_props = dict(source_node)
                target_props = dict(target_node)

                source_id = (
                    source_props.pop("pwc_id", None)
                    or source_props.pop("model_id", None)
                    or source_props.pop("name", source_node.element_id)
                )
                target_id = (
                    target_props.pop("pwc_id", None)
                    or target_props.pop("model_id", None)
                    or target_props.pop("name", target_node.element_id)
                )

                relationships.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "type": rel.type,
                        "properties": dict(rel),
                    }
                )

            return {"nodes": nodes, "relationships": relationships}

        try:
            async with self.driver.session() as session:
                graph_data = await session.execute_read(_get_neighborhood_tx)
                if graph_data:
                    logger.info(
                        f"Successfully fetched neighborhood for paper {pwc_id}."
                    )
                return graph_data
        except Exception as e:
            logger.exception(f"Error fetching neighborhood for paper {pwc_id}: {e}")
            return None

    # --- NEW Method: Link Models to Papers Batch ---
    async def link_model_to_paper_batch(self, links: List[Dict[str, Any]]) -> None:
        """
        Creates MENTIONS relationships between HFModels and Papers using UNWIND.
        Matches Papers using EITHER pwc_id OR arxiv_id_base (whichever is available).
        """
        if not links:
            return

        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot link model-paper batch: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        async def _run_link_batch_tx(tx: AsyncManagedTransaction) -> None:
            query = """
            UNWIND $batch AS link_data
            // Match HFModel using model_id (which should exist if passed)
            MATCH (m:HFModel {model_id: link_data.hf_model_id})

            // --- Match Paper using EITHER pwc_id OR arxiv_id_base ---
            OPTIONAL MATCH (p_pwc:Paper {pwc_id: link_data.pwc_id})
            OPTIONAL MATCH (p_arxiv:Paper {arxiv_id_base: link_data.arxiv_id_base})

            // Select the first non-null matched paper node
            WITH m, link_data, COALESCE(p_pwc, p_arxiv) AS p
            WHERE p IS NOT NULL // Ensure we found a paper node

            // Merge the relationship
            MERGE (m)-[r:MENTIONS]->(p)
            ON CREATE SET r.created_at = timestamp()
            RETURN count(*) // Return count of merged relationships
            """
            try:
                # Pass the original links list directly to the query
                result = await tx.run(query, parameters={"batch": links})
                summary = await result.consume()
                merged_count = summary.counters.relationships_created
                logger.info(
                    f"Successfully processed model-paper link batch. Relationships created/merged: {merged_count} (Note: only counts newly created)"
                )
            except Exception as e:
                logger.error(f"Error executing model-paper link batch query: {e}")
                # Log batch data causing error (first few items for brevity)
                logger.error(f"Batch data sample: {links[:5]}")
                raise  # Re-raise after logging context

        async with self.driver.session() as session:
            try:
                await session.execute_write(_run_link_batch_tx)
            except Exception:
                # Error already logged in _run_link_batch_tx, just re-raise
                raise

    # --- NEW Method: Save Papers by Arxiv ID Batch (for those without pwc_id) ---
    async def save_papers_by_arxiv_batch(
        self, papers_data: List[Dict[str, Any]]
    ) -> None:
        """
        Saves a batch of paper data to Neo4j using UNWIND, merging primarily based on arxiv_id_base.
        This is intended for papers lacking a pwc_id. It only saves basic properties and skips relationships.
        Assumes input dictionaries contain at least 'arxiv_id_base' and other basic properties.
        """
        if not papers_data:
            return

        if not self.driver or not hasattr(self.driver, "session"):
            logger.error(
                "Cannot save papers by arxiv batch: Neo4j driver not available."
            )
            raise ConnectionError("Neo4j driver is not available.")

        # Simplified UNWIND query focusing on arxiv_id_base merge and basic properties
        query = """
        UNWIND $batch AS paper_props

        // Merge Paper node - Using arxiv_id_base as the primary merge key
        MERGE (p:Paper {arxiv_id_base: paper_props.arxiv_id_base})
        ON CREATE SET
            // Set other known properties, relationships are skipped
            p.arxiv_id_versioned = paper_props.arxiv_id_versioned,
            p.title = paper_props.title,
            p.summary = paper_props.summary,
            p.published_date = CASE WHEN paper_props.published_date IS NOT NULL THEN date(paper_props.published_date) ELSE null END,
            p.area = paper_props.area, // Might be null if only from Arxiv
            p.pwc_url = paper_props.pwc_url, // Likely null
            p.pdf_url = paper_props.pdf_url,
            p.doi = paper_props.doi,
            p.primary_category = paper_props.primary_category,
            // Correct handling for potentially null/empty list
            p.categories = CASE WHEN paper_props.categories IS NOT NULL AND size(paper_props.categories) > 0 THEN paper_props.categories ELSE [] END,
            p.created_at = timestamp()
        ON MATCH SET
            // Update properties if needed, prefer existing non-null values?
            // Or simply update all like the pwc_id version?
            // Let's update all for consistency for now.
            p.arxiv_id_versioned = paper_props.arxiv_id_versioned,
            p.title = paper_props.title,
            p.summary = paper_props.summary,
            p.published_date = CASE WHEN paper_props.published_date IS NOT NULL THEN date(paper_props.published_date) ELSE p.published_date END,
            p.area = paper_props.area,
            p.pwc_url = paper_props.pwc_url,
            p.pdf_url = paper_props.pdf_url,
            p.doi = paper_props.doi,
            p.primary_category = paper_props.primary_category,
            // Correct handling for potentially null/empty list
            p.categories = CASE WHEN paper_props.categories IS NOT NULL AND size(paper_props.categories) > 0 THEN paper_props.categories ELSE [] END,
            p.updated_at = timestamp()
        """

        # Prepare batch data, ensuring arxiv_id_base exists
        prepared_batch = []
        for paper in papers_data:
            if not paper.get("arxiv_id_base"):
                logger.warning(
                    f"Skipping paper in arxiv batch save due to missing 'arxiv_id_base': title={paper.get('title')}"
                )
                continue

            prepared_paper = {
                "arxiv_id_base": paper["arxiv_id_base"],
                "arxiv_id_versioned": paper.get("arxiv_id_versioned"),
                "title": paper.get("title"),
                "summary": paper.get("summary"),
                "published_date": paper.get("published_date"),  # Pass as string
                "area": paper.get("area"),  # Might be null
                "pwc_url": paper.get("pwc_url"),  # Likely null
                "pdf_url": paper.get("pdf_url"),
                "doi": paper.get("doi"),
                "primary_category": paper.get("primary_category"),
                "categories": paper.get("categories") or [],
            }
            prepared_batch.append(prepared_paper)

        if not prepared_batch:
            logger.info(
                "No valid paper data (with arxiv_id_base) found in arxiv batch to save."
            )
            return

        async def _run_arxiv_batch_tx(tx: AsyncManagedTransaction) -> None:
            await tx.run(query, batch=prepared_batch)

        # Execute the transaction using execute_write
        async with self.driver.session() as session:
            try:
                await session.execute_write(_run_arxiv_batch_tx)
                logger.info(
                    f"Successfully processed batch of {len(prepared_batch)} papers (by arxiv_id, basic props) in Neo4j."
                )
            except Exception as e:
                logger.error(f"Error saving papers batch (by arxiv_id) to Neo4j: {e}")
                raise

    async def search_nodes(
        self,
        search_term: str,
        index_name: str,
        labels: List[str],
        limit: int = 10,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Performs a full-text search on nodes using a specified index.

        Args:
            search_term (str): The term to search for.
            index_name (str): The name of the full-text index to use.
            labels (List[str]): List of node labels to search within.
            limit (int, optional): Maximum number of results to return. Defaults to 10.
            skip (int, optional): Number of results to skip. Defaults to 0.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'node' and 'score'.

        Raises:
            ConnectionError: If the Neo4j driver is not available.
            Exception: If the database query fails.
        """
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error(f"Cannot search nodes: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        label_filter = "|".join(labels)  # Format labels for the query
        query = (
            f"CALL db.index.fulltext.queryNodes('{index_name}', $searchTerm) YIELD node, score "
            f"WHERE single(l IN labels(node) WHERE l IN $labelFilterList) "  # Ensure node has one of the target labels
            "RETURN node, score "
            "ORDER BY score DESC "  # Added explicit ordering
            "SKIP $skip LIMIT $limit"
        )

        parameters = {
            "searchTerm": search_term,
            "labelFilterList": labels,  # Pass the list for the WHERE clause
            "skip": skip,
            "limit": limit,
        }

        try:
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                # Corrected: result.data() is synchronous and returns a list
                records = result.data()
                logger.debug(
                    f"Neo4j search for '{search_term}' returned {len(records)} results."  # type: ignore[arg-type]
                )
                return records  # type: ignore[return-value]
        except Exception as e:
            logger.error(
                f"Error searching Neo4j with term '{search_term}' on index '{index_name}': {e}"
            )
            raise

    async def get_neighbors(
        self,
        node_label: str,
        node_prop: str,
        node_val: Any,
    ) -> List[Dict[str, Any]]:
        """
        Fetches the 1-hop neighbors for a given node.

        Args:
            node_label (str): The label of the starting node.
            node_prop (str): The property key of the starting node to match on.
            node_val (Any): The property value of the starting node to match on.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                represents a neighbor and contains:
                - 'node': Properties of the neighbor node.
                - 'relationship': Properties of the relationship.
                - 'direction': 'IN' or 'OUT' indicating relationship direction relative to the start node.
        """
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot get neighbors: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        # 修改查询，让数据库直接计算方向
        query = f"""
        MATCH (start:{node_label} {{{node_prop}: $node_val}})
        MATCH (start)-[r]-(neighbor)
        WHERE elementId(start) <> elementId(neighbor)
        RETURN 
            neighbor, 
            type(r) as rel_type, 
            properties(r) as rel_props, 
            CASE WHEN startNode(r) = start THEN 'OUT' ELSE 'IN' END as direction
        """
        params = {"node_val": node_val}

        results: List[Dict[str, Any]] = []
        async with self.driver.session() as session:
            try:
                result = await session.run(query, params)
                data: List[Dict[str, Any]] = await result.data()

                for record in data:
                    neighbor_node: Optional[Any] = record.get("neighbor")
                    rel_type: str = record.get("rel_type", "UNKNOWN")
                    rel_props: Dict[str, Any] = record.get("rel_props", {})
                    direction: str = record.get("direction", "UNKNOWN")

                    if not neighbor_node:
                        logger.warning(
                            f"Skipping neighbor record due to missing node data: {record}"
                        )
                        continue

                    # 提取节点属性
                    neighbor_props = (
                        dict(neighbor_node.items())
                        if hasattr(neighbor_node, "items")
                        and callable(neighbor_node.items)
                        else {}
                    )

                    # 日志输出用于调试
                    logger.info(
                        f"[GET_NEIGHBORS PROP DEBUG] 节点: {neighbor_props}, 关系类型: {rel_type}, 关系属性: {rel_props}, 方向: {direction}"
                    )

                    results.append(
                        {
                            "node": neighbor_props,
                            "relationship": {"type": rel_type, "properties": rel_props},
                            "direction": direction,
                        }
                    )

            except Exception as e:
                logger.error(
                    f'Error getting neighbors from Neo4j for {node_label} {node_prop}="{node_val}": {e}'
                )
                raise
        return results

    # --- NEW Method: Get Related Nodes ---
    async def get_related_nodes(
        self,
        start_node_label: str,
        start_node_prop: str,
        start_node_val: Any,
        relationship_type: str,
        target_node_label: str,
        direction: Literal["IN", "OUT", "BOTH"] = "BOTH",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Finds related nodes based on a starting node, relationship type, and target label.

        Args:
            start_node_label (str): The label of the start node.
            start_node_prop (str): The property name used to identify the start node (e.g., 'pwc_id', 'model_id').
            start_node_val (Any): The value of the identifying property for the start node.
            relationship_type (str): The type of relationship to search for.
            target_node_label (str): The label of the target node.
            direction (Literal['IN', 'OUT', 'BOTH'], optional): The direction of the relationship. Defaults to 'BOTH'.
            limit (int, optional): Maximum number of related nodes to return. Defaults to 25.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'target_node' properties
                                  and 'relationship' details (type, properties).

        Raises:
            ConnectionError: If the Neo4j driver is not available.
            ValueError: If the direction is invalid.
            Exception: If the database query fails.
        """
        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot get related nodes: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        # Validate direction
        if direction not in ["IN", "OUT", "BOTH"]:
            raise ValueError("Invalid direction. Must be 'IN', 'OUT', or 'BOTH'.")

        # Build the relationship pattern based on direction
        if direction == "OUT":
            rel_pattern = f"MATCH (start:{start_node_label} {{{start_node_prop}: $start_val}})-[r:`{relationship_type}`]->(target:{target_node_label})"
        elif direction == "IN":
            rel_pattern = f"MATCH (start:{start_node_label} {{{start_node_prop}: $start_val}})<-[r:`{relationship_type}`]-(target:{target_node_label})"
        elif direction == "BOTH":
            rel_pattern = f"MATCH (start:{start_node_label} {{{start_node_prop}: $start_val}})-[r:`{relationship_type}`]-(target:{target_node_label})"
        else:
            # Should be caught by validation, but as a safeguard
            raise ValueError(f"Invalid direction: {direction}")

        # Corrected Cypher query: RETURN DISTINCT target, r
        query = f"""
            {rel_pattern}
            RETURN DISTINCT target, r
            LIMIT $limit
            """
        parameters = {"start_val": start_node_val, "limit": limit}

        if not self.driver or not hasattr(self.driver, "session"):
            logger.error("Cannot get related nodes: Neo4j driver not available.")
            raise ConnectionError("Neo4j driver is not available.")

        # --- Transaction function to process results internally --- #
        async def _process_results_tx(
            tx: AsyncManagedTransaction,
        ) -> List[Dict[str, Any]]:
            processed_results: List[Dict[str, Any]] = []
            result_cursor = await tx.run(query, parameters)

            async for record in result_cursor:
                target_node = record.get("target")
                relationship = record.get("r")
                if target_node and relationship:
                    # Convert Node to dict safely
                    node_dict = {}
                    try:
                        if hasattr(target_node, "items") and callable(
                            target_node.items
                        ):
                            node_dict = dict(target_node.items())
                        else:
                            logger.warning(
                                f"DEBUG: Target node object type {type(target_node)} has no items() method."
                            )
                    except Exception as e:
                        logger.error(
                            f"DEBUG: Error converting target node items to dict: {e}",
                            exc_info=True,
                        )

                    # Convert Relationship to dict safely
                    rel_dict = {}
                    try:
                        if hasattr(relationship, "items") and callable(
                            relationship.items
                        ):
                            rel_dict = dict(relationship.items())
                        else:
                            logger.warning(
                                f"DEBUG: Relationship object type {type(relationship)} has no items() method."
                            )
                    except Exception as e:
                        logger.error(
                            f"DEBUG: Error converting relationship items to dict: {e}",
                            exc_info=True,
                        )

                    processed_results.append(
                        {
                            "target_node": node_dict,
                            "relationship": rel_dict,
                        }
                    )
            return processed_results

        # --- End transaction function --- #

        final_results: List[Dict[str, Any]] = []
        async with self.driver.session() as session:
            try:
                # Use execute_read with the processing function
                final_results = await session.execute_read(_process_results_tx)
                logger.debug(
                    f"Found {len(final_results)} related nodes for {start_node_label}[{start_node_prop}={start_node_val}] -[{relationship_type}]-> {target_node_label} (direction: {direction})"
                )

            except Exception as e:
                logger.error(
                    f"Error getting related nodes for {start_node_label}[{start_node_prop}={start_node_val}]: {e}"
                )
                logger.error(f"Query: {query}")
                logger.error(f"Parameters: {parameters}")
                raise

        return final_results

    # LINTER FIX: Remove unused _process_paper_results method (if truly unused)
    # async def _process_paper_results(self, result: Query) -> List[Dict[str, Any]]:
    #     # Example helper - adjust based on actual query structure
    #     # LINTER FIX: result (Query) does not have .data() directly
    #     # data = await result.data()
    #     # processed_results = []
    #     # for record in data:
    #     #     # process record
    #     #     pass
    #     # return processed_results
    #     pass # Assuming unused for now
