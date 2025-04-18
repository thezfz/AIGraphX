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
        paper_data = await self.pg_repo.get_paper_details_by_pwc_id(pwc_id)

        if not paper_data:
            logger.warning(f"Paper with pwc_id '{pwc_id}' not found in PostgreSQL.")
            return None

        # Ensure paper_data is a mutable dict for updates
        paper_data = dict(paper_data)

        # 2. Get related entities from Neo4j if available
        # Initialize relation lists to empty, they will be populated from Neo4j if available
        paper_data["tasks"] = []
        paper_data["datasets"] = []
        paper_data["methods"] = []
        # Add others if needed (e.g., 'repositories', though that might be complex)

        if self.neo4j_repo:
            try:
                logger.debug(f"Fetching neighborhood from Neo4j for paper {pwc_id}...")
                graph_dict = await self.neo4j_repo.get_paper_neighborhood(pwc_id)

                if graph_dict and "nodes" in graph_dict:
                    logger.debug(
                        f"Parsing neighbors for paper {pwc_id} from Neo4j result."
                    )
                    related_tasks = []
                    related_datasets = []
                    related_methods = []
                    # Assuming 'nodes' is a list of dicts with 'type' and 'label' or 'properties.name'
                    for node in graph_dict["nodes"]:
                        node_type = node.get("type")
                        # Use node['label'] as primary, fallback to properties['name'] if label is generic like pwc_id
                        node_name = node.get(
                            "label", node.get("properties", {}).get("name")
                        )

                        if (
                            not node_name or node.get("id") == pwc_id
                        ):  # Skip self or nodes without a usable name
                            continue

                        if node_type == "Task":
                            related_tasks.append(node_name)
                        elif node_type == "Dataset":
                            related_datasets.append(node_name)
                        elif (
                            node_type == "Method"
                        ):  # Add support for methods if they exist as nodes
                            related_methods.append(node_name)
                        # Add other types like 'Repository' if needed, might need 'url' instead of 'name'

                    # Update paper_data with lists from Neo4j
                    paper_data["tasks"] = related_tasks
                    paper_data["datasets"] = related_datasets
                    paper_data["methods"] = related_methods
                    logger.debug(
                        f"Extracted relations for {pwc_id} from Neo4j: "
                        f"Tasks={len(related_tasks)}, Datasets={len(related_datasets)}, Methods={len(related_methods)}"
                    )
                else:
                    logger.debug(
                        f"No graph data or nodes found in Neo4j neighborhood for paper {pwc_id}."
                    )

            except Exception as e:
                logger.error(
                    f"Failed to fetch or parse relations from Neo4j for {pwc_id}: {e}",
                    exc_info=True,
                )
                # Decide if we should proceed with potentially incomplete PG data or return error?
                # For now, proceed with PG data, but log the error clearly.

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
        def _decode_json_field(field_data: Any) -> list[Any]:
            if isinstance(field_data, list):  # Already decoded
                return list(field_data)  # 显式转换为list
            elif isinstance(field_data, str):
                try:
                    return list(json.loads(field_data))  # 显式转换为list
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON field: {field_data[:50]}...")
                    return []
            return []  # Default to empty list for other types or None

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
            abstract=paper_data.get("abstract"),
            arxiv_id=paper_data.get("arxiv_id_base"),
            url_abs=paper_data.get("pwc_url"),
            url_pdf=paper_data.get("pdf_url"),
            published_date=published_date_obj,
            authors=_decode_json_field(paper_data.get("authors")),
            tasks=paper_data.get("tasks", []),
            datasets=paper_data.get("datasets", []),
            methods=paper_data.get("methods", []),
            frameworks=_decode_json_field(paper_data.get("frameworks")),
            number_of_stars=paper_data.get("number_of_stars"),
            area=paper_data.get("area"),
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
            # Note: get_hf_models_by_ids returns a list, we need one item
            model_list = await self.pg_repo.get_hf_models_by_ids([model_id])
            if not model_list:
                logger.warning(f"No details found for model {model_id} in PostgreSQL.")
                return None

            model_data = model_list[0]
            # Convert to HFModelDetail Pydantic model
            return HFModelDetail(**model_data)

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
