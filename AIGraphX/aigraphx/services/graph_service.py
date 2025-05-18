import logging
from typing import Optional, List, Dict, Any, Literal
import json  # Import json for potential decoding
from datetime import date  # Add date

# Import necessary components using CORRECT paths
from aigraphx.repositories.postgres_repo import PostgresRepository  # Corrected path
from aigraphx.repositories.neo4j_repo import (
    Neo4jRepository,
)  # Corrected path (assuming it moved too)
from aigraphx.models.graph import (
    PaperDetailResponse,
    GraphData,
    HFModelDetail,
    BasicPaperInfo,  # Import new model
    BasicTaskInfo,   # Import new model
)

logger = logging.getLogger(__name__)


class GraphService:
    """Provides services for retrieving graph-related data."""

    def __init__(
        self,
        pg_repo: PostgresRepository,
        neo4j_repo: Optional[Neo4jRepository],  # Accept Optional Neo4j repo
    ):
        """Initializes the service with necessary dependencies."""
        self.pg_repo = pg_repo
        self.neo4j_repo = neo4j_repo
        if neo4j_repo is None:
            logger.warning(
                "GraphService initialized without a Neo4j repository. Graph features will be unavailable."
            )

    async def get_paper_details(self, pwc_id: str) -> Optional[PaperDetailResponse]:
        """Retrieves detailed information for a specific paper using its PWC ID."""
        logger.info(f"Fetching details for paper: {pwc_id}")

        # 1. Get base details from PostgreSQL using the CORRECT method name
        paper_data_record = await self.pg_repo.get_paper_details_by_pwc_id(pwc_id)

        if not paper_data_record:
            logger.warning(f"Paper with pwc_id '{pwc_id}' not found in PostgreSQL.")
            return None

        # Ensure paper_data is a mutable dict for updates
        paper_data = dict(paper_data_record)

        # Initialize lists for related entities
        tasks_list: List[str] = []
        datasets_list: List[str] = []
        methods_list: List[str] = []
        repositories_list: List[Dict[str, Any]] = [] # Initialize here

        if self.neo4j_repo:
            try:
                logger.debug(
                    f"Fetching related entities from Neo4j for paper {pwc_id}..."
                )

                # Fetch Tasks
                task_nodes_data = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="HAS_TASK",
                    target_node_label="Task",
                    direction="OUT",
                    limit=50,
                )
                tasks_list = [
                    item.get("target_node", {}).get("name")
                    for item in task_nodes_data
                    if item.get("target_node", {}).get("name")
                ]

                # Fetch Datasets
                dataset_nodes_data = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="USES_DATASET",
                    target_node_label="Dataset",
                    direction="OUT",
                    limit=50,
                )
                datasets_list = [
                    item.get("target_node", {}).get("name")
                    for item in dataset_nodes_data
                    if item.get("target_node", {}).get("name")
                ]

                # Fetch Methods
                method_nodes_data = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="USES_METHOD",
                    target_node_label="Method",
                    direction="OUT",
                    limit=50,
                )
                methods_list = [
                    item.get("target_node", {}).get("name")
                    for item in method_nodes_data
                    if item.get("target_node", {}).get("name")
                ]
                
                # Fetch Repositories
                repository_related_data = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="HAS_REPOSITORY",
                    target_node_label="Repository",
                    direction="OUT",
                    limit=10,
                )
                temp_repo_list = []
                for data_item in repository_related_data: # Renamed variable to avoid conflict
                    target_node = data_item.get("target_node")
                    if isinstance(target_node, dict):
                        temp_repo_list.append(target_node)
                repositories_list = temp_repo_list

                logger.debug(
                    f"Retrieved relations for {pwc_id} from Neo4j: "
                    f"Tasks={len(tasks_list)}, Datasets={len(datasets_list)}, Methods={len(methods_list)}, Repositories={len(repositories_list)}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to fetch related entities from Neo4j for {pwc_id}: {e}",
                    exc_info=True,
                )

        paper_data["tasks"] = tasks_list
        paper_data["datasets"] = datasets_list
        paper_data["methods"] = methods_list
        paper_data["repositories"] = repositories_list

        pwc_id_val = paper_data.get("pwc_id")
        if not pwc_id_val or not isinstance(pwc_id_val, str):
            logger.error(
                f"Missing or invalid pwc_id in data returned for identifier: {pwc_id}. Found: {pwc_id_val}"
            )
            return None

        published_date_val = paper_data.get("published_date")
        published_date_obj: Optional[date] = None
        if isinstance(published_date_val, date):
            published_date_obj = published_date_val
        elif isinstance(published_date_val, str):
            try:
                published_date_obj = date.fromisoformat(published_date_val)
            except ValueError:
                logger.warning(
                    f"Invalid date format for published_date: {published_date_val}"
                )

        response = PaperDetailResponse(
            pwc_id=pwc_id_val,
            title=paper_data.get("title"),
            abstract=paper_data.get("summary"),
            arxiv_id=paper_data.get("arxiv_id_base"),
            url_abs=paper_data.get("pwc_url"),
            url_pdf=paper_data.get("pdf_url"),
            published_date=published_date_obj,
            authors=paper_data.get("authors", []),
            tasks=paper_data.get("tasks", []),
            datasets=paper_data.get("datasets", []),
            methods=paper_data.get("methods", []),
            repositories=paper_data.get("repositories", []),
            area=paper_data.get("area"),
            conference=paper_data.get("conference")
        )

        logger.info(
            f"Successfully retrieved details for paper: {pwc_id_val}"
        )
        return response

    async def get_paper_graph(self, pwc_id: str) -> Optional[GraphData]:
        logger.info(f"Fetching graph data for paper: {pwc_id}")
        if self.neo4j_repo is None:
            logger.error("Cannot fetch paper graph: Neo4j repository is not available.")
            return None
        try:
            graph_dict = await self.neo4j_repo.get_paper_neighborhood(pwc_id)
            if not graph_dict:
                logger.warning(f"No graph data found for paper {pwc_id} in Neo4j.")
                return None
            try:
                graph_data_model = GraphData(**graph_dict)
                logger.info(f"Successfully parsed graph data for paper {pwc_id}.")
                return graph_data_model
            except Exception as pydantic_error:
                logger.error(
                    f"Failed to validate graph data from Neo4j for paper {pwc_id}: {pydantic_error}"
                )
                logger.debug(f"Raw data from repo: {graph_dict}")
                return None
        except Exception as e:
            logger.exception(
                f"Error fetching graph data for paper {pwc_id} from Neo4j: {e}"
            )
            raise

    async def get_model_details(self, model_id: str) -> Optional[HFModelDetail]:
        logger.info(f"Fetching details for model: {model_id}")
        try:
            model_list = await self.pg_repo.get_hf_models_by_ids([model_id])
            if not model_list:
                logger.warning(f"No details found for model {model_id} in PostgreSQL.")
                return None

            model_data_pg = model_list[0]
            retrieved_model_id = model_data_pg.get("hf_model_id")
            if not retrieved_model_id or not isinstance(retrieved_model_id, str):
                logger.error(
                    f"Invalid or missing 'hf_model_id' in data for requested model_id '{model_id}'. Found: {retrieved_model_id}"
                )
                return None

            related_papers_info: List[BasicPaperInfo] = []
            related_tasks_info: List[BasicTaskInfo] = []

            if self.neo4j_repo:
                try:
                    # Fetch related papers (e.g., HFModel MENTIONS Paper)
                    paper_nodes_data = await self.neo4j_repo.get_related_nodes(
                        start_node_label="HFModel",
                        start_node_prop="modelId", # Property in Neo4j for HFModel
                        start_node_val=retrieved_model_id,
                        relationship_type="MENTIONS", # Relationship type
                        target_node_label="Paper",
                        direction="OUT", # Assuming HFModel -> MENTIONS -> Paper
                        limit=10,
                    )
                    for paper_node_item in paper_nodes_data:
                        paper_node = paper_node_item.get("target_node", {})
                        if paper_node.get("pwc_id") or paper_node.get("arxiv_id_base"): # Check for identifier
                            # Convert Neo4j date to Python date
                            neo4j_date = paper_node.get("published_date")
                            py_date: Optional[date] = None
                            if neo4j_date:
                                try:
                                    # neo4j.time.Date to datetime.date
                                    py_date = date(neo4j_date.year, neo4j_date.month, neo4j_date.day)
                                except AttributeError:
                                    logger.warning(f"Could not convert Neo4j date {neo4j_date} to datetime.date for paper related to model {retrieved_model_id}")
                                except TypeError: # Handle if neo4j_date is already a string or other unexpected type
                                    try:
                                        py_date = date.fromisoformat(str(neo4j_date))
                                    except (ValueError, TypeError):
                                         logger.warning(f"Could not parse date string {neo4j_date} from Neo4j to datetime.date for paper related to model {retrieved_model_id}")

                            related_papers_info.append(
                                BasicPaperInfo(
                                    pwc_id=paper_node.get("pwc_id"),
                                    arxiv_id=paper_node.get("arxiv_id_base"),
                                    title=paper_node.get("title"),
                                    published_date=py_date
                                )
                            )
                    
                    # Fetch related tasks from graph (e.g., HFModel HAS_TASK Task)
                    # This might be redundant if pipeline_tag is comprehensive, but provides graph context
                    task_nodes_data_graph = await self.neo4j_repo.get_related_nodes(
                        start_node_label="HFModel",
                        start_node_prop="modelId", # Property in Neo4j for HFModel
                        start_node_val=retrieved_model_id,
                        relationship_type="HAS_TASK", # Relationship type
                        target_node_label="Task",
                        direction="OUT", # Assuming HFModel -> HAS_TASK -> Task
                        limit=10,
                    )
                    for task_node_item in task_nodes_data_graph:
                        task_node = task_node_item.get("target_node", {})
                        if task_node.get("name"):
                             related_tasks_info.append(BasicTaskInfo(name=task_node.get("name")))

                except Exception as e:
                    logger.error(
                        f"Failed to fetch related graph entities from Neo4j for model {retrieved_model_id}: {e}",
                        exc_info=True,
                    )
            
            hf_dataset_links_raw = model_data_pg.get("hf_dataset_links")
            dataset_links_processed: Optional[List[str]] = None
            if isinstance(hf_dataset_links_raw, list):
                if all(isinstance(item, str) for item in hf_dataset_links_raw):
                    dataset_links_processed = hf_dataset_links_raw
                else:
                    logger.warning(f"hf_dataset_links for model {retrieved_model_id} is a list, but not all items are strings. Attempting conversion.")
                    dataset_links_processed = [str(item) for item in hf_dataset_links_raw]
            elif isinstance(hf_dataset_links_raw, str):
                try:
                    parsed_links = json.loads(hf_dataset_links_raw)
                    if isinstance(parsed_links, list):
                        if all(isinstance(item, str) for item in parsed_links):
                            dataset_links_processed = parsed_links
                        else:
                            logger.warning(f"Parsed hf_dataset_links for model {retrieved_model_id} is a list, but not all items are strings. Attempting conversion.")
                            dataset_links_processed = [str(item) for item in parsed_links]
                    else:
                        logger.warning(f"hf_dataset_links for model {retrieved_model_id} decoded to non-list: {type(parsed_links)}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode hf_dataset_links JSON for model {retrieved_model_id}: {hf_dataset_links_raw[:100]}...")
            elif hf_dataset_links_raw is not None:
                 logger.warning(f"hf_dataset_links for model {retrieved_model_id} is of unexpected type: {type(hf_dataset_links_raw)}")

            mapped_data = {
                "model_id": retrieved_model_id,
                "author": model_data_pg.get("hf_author"),
                "sha": model_data_pg.get("hf_sha"),
                "last_modified": model_data_pg.get("hf_last_modified"),
                "tags": model_data_pg.get("hf_tags"),
                "pipeline_tag": model_data_pg.get("hf_pipeline_tag"),
                "downloads": model_data_pg.get("hf_downloads"),
                "likes": model_data_pg.get("hf_likes"),
                "library_name": model_data_pg.get("hf_library_name"),
                "created_at": model_data_pg.get("created_at"),
                "updated_at": model_data_pg.get("updated_at"),
                "readme_content": model_data_pg.get("hf_readme_content"),
                "dataset_links": dataset_links_processed,
                "related_papers": related_papers_info if related_papers_info else None,
                "related_tasks_graph": related_tasks_info if related_tasks_info else None,
            }

            logger.debug(
                f"Mapped data for model {retrieved_model_id}: {mapped_data}"
            )
            return HFModelDetail(**mapped_data)

        except Exception as e:
            logger.exception(
                f"Error fetching details for model {model_id} from PostgreSQL: {e}"
            )
            raise

    async def get_related_entities(
        self,
        start_node_label: str,
        start_node_prop: str,
        start_node_val: Any,
        relationship_type: str,
        target_node_label: str,
        direction: Literal["IN", "OUT", "BOTH"] = "BOTH",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"Fetching related entities: start=({start_node_label}:{start_node_prop}={start_node_val}), "
            f"rel={relationship_type}({direction}), target={target_node_label}, limit={limit}"
        )
        if direction not in ["IN", "OUT", "BOTH"]:
            logger.error(
                f"Invalid direction provided: {direction}. Returning empty list."
            )
            return []
        if self.neo4j_repo is None:
            logger.error(
                "Cannot fetch related entities: Neo4j repository is not available."
            )
            return []
        try:
            related_nodes_props = await self.neo4j_repo.get_related_nodes(
                start_node_label=start_node_label,
                start_node_prop=start_node_prop,
                start_node_val=start_node_val,
                relationship_type=relationship_type,
                target_node_label=target_node_label,
                direction=direction,
                limit=limit,
            )
            return related_nodes_props
        except Exception as e:
            logger.exception(f"Error calling neo4j_repo.get_related_nodes: {e}")
            return []

    async def get_model_graph(self, model_id: str) -> Optional[GraphData]:
        """Fetches the neighborhood graph data for a specific Hugging Face model."""
        logger.info(f"Fetching graph data for model: {model_id}")
        if self.neo4j_repo is None:
            logger.error("Cannot fetch model graph: Neo4j repository is not available.")
            return None
        try:
            graph_dict = await self.neo4j_repo.get_model_neighborhood(model_id)
            if not graph_dict:
                logger.warning(f"No graph data found for model {model_id} in Neo4j.")
                return None
            try:
                # Ensure nodes and relationships are present before parsing
                if "nodes" not in graph_dict or "relationships" not in graph_dict:
                    logger.error(f"Graph data for model {model_id} is missing 'nodes' or 'relationships' key. Data: {graph_dict}")
                    return None

                graph_data_model = GraphData(**graph_dict)
                logger.info(f"Successfully parsed graph data for model {model_id}.")
                return graph_data_model
            except Exception as pydantic_error:
                logger.error(
                    f"Failed to validate graph data from Neo4j for model {model_id}: {pydantic_error}"
                )
                logger.debug(f"Raw data from repo for model {model_id}: {graph_dict}")
                return None
        except Exception as e:
            logger.exception(
                f"Error fetching graph data for model {model_id} from Neo4j: {e}"
            )
            # raise # Optionally re-raise, or return None as per current pattern
            return None
