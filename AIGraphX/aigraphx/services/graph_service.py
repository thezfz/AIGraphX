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
)  # Import the response model and new models

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
                task_nodes = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="HAS_TASK",  # Assuming this relationship
                    target_node_label="Task",
                    direction="OUT",  # Assuming Paper->Task
                    limit=50,  # Set a reasonable limit
                )
                tasks_list = [
                    node.get("target_node", {}).get("name") # Corrected: extract from target_node
                    for node in task_nodes
                    if node.get("target_node", {}).get("name")
                ]

                # Fetch Datasets
                dataset_nodes = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="USES_DATASET",  # Assuming this relationship
                    target_node_label="Dataset",
                    direction="OUT",  # Assuming Paper->Dataset
                    limit=50,
                )
                datasets_list = [
                    node.get("target_node", {}).get("name") # Corrected: extract from target_node
                    for node in dataset_nodes
                    if node.get("target_node", {}).get("name")
                ]

                # Fetch Methods
                method_nodes = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="USES_METHOD",  # Assuming this relationship
                    target_node_label="Method",
                    direction="OUT",  # Assuming Paper->Method
                    limit=50,
                )
                methods_list = [
                    node.get("target_node", {}).get("name") # Corrected: extract from target_node
                    for node in method_nodes
                    if node.get("target_node", {}).get("name")
                ]

                # Fetch Repositories (expecting detailed dictionaries)
                repository_related_data = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="HAS_REPOSITORY",
                    target_node_label="Repository",
                    direction="OUT",
                    limit=10, # Adjust limit as needed
                )
                # Corrected list comprehension for repositories_list
                temp_repo_list = []
                for data in repository_related_data:
                    target_node = data.get("target_node")
                    if isinstance(target_node, dict): # Ensure it's a dict and not None
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
                # Continue with PG data, lists will remain empty (as initialized)

        # Update paper_data with lists from Neo4j (or empty lists if failed/unavailable)
        paper_data["tasks"] = tasks_list
        paper_data["datasets"] = datasets_list
        paper_data["methods"] = methods_list
        paper_data["repositories"] = repositories_list

        # Add check for pwc_id before creating response object
        pwc_id_val = paper_data.get("pwc_id")
        if not pwc_id_val or not isinstance(pwc_id_val, str):
            logger.error(
                f"Missing or invalid pwc_id in data returned for identifier: {pwc_id}. Found: {pwc_id_val}"
            )
            # Depending on requirements, either return None or raise an error.
            # Raising an error highlights potential data inconsistency issues sooner.
            # Alternatively, adjust PaperDetailResponse model if None is acceptable.
            # Returning None might be preferred if the service layer should handle this gracefully.
            return None  # Returning None as service might handle this
            # raise ValueError(f"Missing or invalid pwc_id in fetched paper data for {pwc_id}")

        # 3. Construct the response model (using potentially updated paper_data)
        # Need to handle potential JSON string fields from PG if not auto-decoded
        # def _decode_json_field(field_data: Any) -> list[Any]:
        #     if isinstance(field_data, list):  # Already decoded
        #         return list(field_data)
        #     elif isinstance(field_data, str):
        #         try:
        #             return list(json.loads(field_data))
        #         except json.JSONDecodeError:
        #             logger.warning(f"Failed to decode JSON field: {field_data[:50]}...")
        #             return []
        #     return []  # Default to empty list for other types or None

        # Fix published_date conversion
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
        # If it's neither date nor valid string, published_date_obj remains None

        response = PaperDetailResponse(
            pwc_id=pwc_id_val,  # Use the validated pwc_id_val
            title=paper_data.get("title"),
            abstract=paper_data.get("summary"),
            arxiv_id=paper_data.get("arxiv_id_base"),
            url_abs=paper_data.get("pwc_url"),
            url_pdf=paper_data.get("pdf_url"),
            published_date=published_date_obj,
            authors=paper_data.get("authors", []), # Assuming psycopg handles JSONB to list
            tasks=paper_data.get("tasks", []),
            datasets=paper_data.get("datasets", []),
            methods=paper_data.get("methods", []),
            # frameworks and number_of_stars are removed from PaperDetailResponse
            repositories=paper_data.get("repositories", []), # Populate repositories
            area=paper_data.get("area"),
            conference=paper_data.get("conference") # Added conference
        )

        logger.info(
            f"Successfully retrieved details for paper: {pwc_id_val}"
        )  # Use validated ID
        return response

    async def get_paper_graph(self, pwc_id: str) -> Optional[GraphData]:
        """Retrieves the neighborhood graph for a given paper ID from Neo4j."""
        logger.info(f"Fetching graph data for paper: {pwc_id}")

        # Check if Neo4j repo is available
        if self.neo4j_repo is None:
            logger.error("Cannot fetch paper graph: Neo4j repository is not available.")
            # Consider raising HTTPException(503) for API layer
            return None

        try:
            # Call the repository method
            graph_dict = await self.neo4j_repo.get_paper_neighborhood(pwc_id)

            if not graph_dict:
                logger.warning(f"No graph data found for paper {pwc_id} in Neo4j.")
                return None

            # Validate and parse the dictionary into GraphData model
            try:
                graph_data_model = GraphData(**graph_dict)
                logger.info(f"Successfully parsed graph data for paper {pwc_id}.")
                return graph_data_model
            except (
                Exception
            ) as pydantic_error:  # Catch Pydantic validation errors specifically
                logger.error(
                    f"Failed to validate graph data from Neo4j for paper {pwc_id}: {pydantic_error}"
                )
                logger.debug(f"Raw data from repo: {graph_dict}")
                # Decide how to handle validation failure - return None or raise?
                # Returning None might hide data structure issues in Neo4j repo.
                # Raising might be better for debugging.
                # Let's return None for now, but consider raising.
                return None

        except Exception as e:
            logger.exception(
                f"Error fetching graph data for paper {pwc_id} from Neo4j: {e}"
            )
            # Re-raise the exception to be handled by the API layer (e.g., return 500)
            raise

    async def get_model_details(self, model_id: str) -> Optional[HFModelDetail]:
        """Retrieves details for a given HF model ID from PostgreSQL."""
        # This method does not depend on Neo4j
        logger.info(f"Fetching details for model: {model_id}")
        try:
            # Use the existing method in PostgresRepository
            model_list = await self.pg_repo.get_hf_models_by_ids([model_id])
            if not model_list:
                logger.warning(f"No details found for model {model_id} in PostgreSQL.")
                return None

            model_data = model_list[0]  # This is a dict from the database row

            # --- Validate hf_model_id before mapping ---
            retrieved_model_id = model_data.get("hf_model_id")
            if not retrieved_model_id or not isinstance(retrieved_model_id, str):
                logger.error(
                    f"Invalid or missing 'hf_model_id' in data for requested model_id '{model_id}'. Found: {retrieved_model_id}"
                )
                return None  # Cannot create details without a valid ID
            # --- End Validation ---

            # Manually map ALL database column names (with hf_ prefix)
            # to Pydantic model field names (without hf_ prefix).
            # Ensure hf_dataset_links is correctly handled (it might be JSON string or already parsed list/dict)
            hf_dataset_links_raw = model_data.get("hf_dataset_links")
            dataset_links_processed: Optional[List[str]] = None # Changed type to List[str]
            if isinstance(hf_dataset_links_raw, list): # Already a list
                # Ensure all elements are strings, if not, log warning or attempt conversion
                if all(isinstance(item, str) for item in hf_dataset_links_raw):
                    dataset_links_processed = hf_dataset_links_raw
                else:
                    logger.warning(f"hf_dataset_links for model {retrieved_model_id} is a list, but not all items are strings. Attempting conversion.")
                    dataset_links_processed = [str(item) for item in hf_dataset_links_raw]
            elif isinstance(hf_dataset_links_raw, str):
                try:
                    parsed_links = json.loads(hf_dataset_links_raw)
                    if isinstance(parsed_links, list):
                        # Ensure all elements of parsed_links are strings
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
                "model_id": retrieved_model_id,  # Use validated ID
                "author": model_data.get("hf_author"),
                "sha": model_data.get("hf_sha"),
                "last_modified": model_data.get("hf_last_modified"),
                "tags": model_data.get("hf_tags"), # Assumed to be list from PG if JSONB
                "pipeline_tag": model_data.get("hf_pipeline_tag"),
                "downloads": model_data.get("hf_downloads"),
                "likes": model_data.get("hf_likes"),
                "library_name": model_data.get("hf_library_name"),
                "created_at": model_data.get("created_at"),
                "updated_at": model_data.get("updated_at"),
                # Updated field names and use processed links
                "readme_content": model_data.get("hf_readme_content"),
                "dataset_links": dataset_links_processed,
            }

            logger.debug(
                f"Mapped data for model {retrieved_model_id}: {mapped_data}"
            )  # Log with actual ID

            # Convert to HFModelDetail Pydantic model using the explicitly mapped data
            # Mypy should now be satisfied as model_id is confirmed to be str
            return HFModelDetail(**mapped_data)

        except Exception as e:
            logger.exception(
                f"Error fetching details for model {model_id} from PostgreSQL: {e}"
            )
            raise

    # --- NEW Method: Get Related Entities ---
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
        """Retrieves properties of related entities based on relationship type and direction."""
        logger.info(
            f"Fetching related entities: start=({start_node_label}:{start_node_prop}={start_node_val}), "
            f"rel={relationship_type}({direction}), target={target_node_label}, limit={limit}"
        )

        # --- ADD Validation ---
        if direction not in ["IN", "OUT", "BOTH"]:
            logger.error(
                f"Invalid direction provided: {direction}. Returning empty list."
            )
            return []
        # --- END Validation ---

        if self.neo4j_repo is None:
            logger.error(
                "Cannot fetch related entities: Neo4j repository is not available."
            )
            # Returning empty list as service might be expected to gracefully handle unavailability
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
            # Log the exception that occurred in the repository
            logger.exception(f"Error calling neo4j_repo.get_related_nodes: {e}")
            # Re-raise the exception or return empty list based on desired error handling
            # For now, let's return empty list to the caller (e.g., API layer)
            # The API layer can then decide to return 500 or an empty result.
            return []  # 类型注解已指定返回List[Dict[str, Any]]，空列表是符合的
