import logging
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import Optional, List, Dict, Any, Literal

# Core components and dependencies - Removed incorrect db imports
# from aigraphx.core.db import get_postgres_repo
# from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# Service layer
from aigraphx.services.graph_service import GraphService

# API Models
from aigraphx.models.graph import PaperDetailResponse, GraphData, HFModelDetail

# --- Import the correct dependency injection source ---
from aigraphx.api.v1 import dependencies as deps

# Router setup
router = APIRouter()
logger = logging.getLogger(__name__)

# --- Removed local/incorrect dependency function ---
# async def get_graph_service(pg_repo: PostgresRepository = Depends(get_postgres_repo)) -> GraphService:
#     return GraphService(pg_repo=pg_repo)


# --- API Endpoint ---
@router.get("/papers/{pwc_id}", response_model=PaperDetailResponse)
async def get_paper_details_endpoint(
    pwc_id: str = Path(
        ..., description="The PWC ID of the paper to retrieve.", min_length=1
    ),
    # Ensure Depends uses the correct source
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> PaperDetailResponse:
    """
    Retrieves detailed information about a specific paper by its PWC ID.
    """
    logger.info(f"Received request for paper details: pwc_id='{pwc_id}'")

    try:  # <--- Add try block
        details = await graph_service.get_paper_details(pwc_id=pwc_id)

        if details is None:
            raise HTTPException(
                status_code=404, detail=f"Paper with PWC ID '{pwc_id}' not found."
            )

        return details

    except HTTPException as http_exc:
        # Re-raise HTTPException (like the 404) directly
        raise http_exc
    except Exception as e:  # <--- Add except block for other exceptions
        # Log the unexpected error for debugging
        logger.exception(
            f"An unexpected error occurred while fetching details for paper '{pwc_id}': {e}"
        )
        # Raise a standard 500 error
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving details for paper '{pwc_id}'.",
        )


@router.get(
    "/papers/{pwc_id}/graph",
    response_model=GraphData,
    summary="Get graph neighborhood for a paper",
    description="Retrieves the graph neighborhood (related papers, models, concepts) for a given PWC paper ID.",
    tags=["Graph"],  # Ensure tag matches include_router
)
async def get_paper_graph_data(
    pwc_id: str = Path(
        ..., description="The PWC ID of the paper to get the graph for."
    ),
    # Ensure Depends uses the correct source
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> GraphData:
    logger.info(f"Received request for graph data for paper: {pwc_id}")
    try:
        graph_data = await graph_service.get_paper_graph(pwc_id)
        if graph_data is None:
            raise HTTPException(
                status_code=404, detail=f"Graph data not found for paper {pwc_id}"
            )
        return graph_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving graph data for paper {pwc_id}: {e}")
        # Consider specific exceptions for not found vs internal error
        raise HTTPException(
            status_code=500, detail="Internal server error retrieving graph data."
        )


@router.get(
    "/models/{model_id:path}",
    response_model=HFModelDetail,
    summary="Get details for a Hugging Face model",
    description="Retrieves detailed information for a specific Hugging Face model by its ID.",
    tags=["models", "Graph"],  # Add Graph tag if desired, consistent tag is important
)
async def get_hf_model_details(
    model_id: str = Path(
        ..., description="The Hugging Face model ID (e.g., 'google/flan-t5-base')."
    ),
    # Ensure Depends uses the correct source
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> HFModelDetail:
    logger.info(f"Received request for details for model: {model_id}")
    try:
        model_details = await graph_service.get_model_details(model_id)
        if model_details is None:
            raise HTTPException(
                status_code=404, detail=f"Model details not found for ID {model_id}"
            )
        return model_details
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving details for model {model_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error retrieving model details."
        )


@router.get(
    "/hf_models/{model_id:path}/graph",
    response_model=GraphData,
    summary="Get graph neighborhood for a Hugging Face model",
    description="Retrieves the graph neighborhood for a given Hugging Face model ID.",
    tags=["models", "Graph"],
)
async def get_hf_model_graph_data(
    model_id: str = Path(
        ..., description="The Hugging Face model ID (e.g., 'google/flan-t5-base') to get the graph for."
    ),
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> GraphData:
    logger.info(f"Received request for graph data for model: {model_id}")
    try:
        graph_data = await graph_service.get_model_graph(model_id)
        if graph_data is None:
            logger.warning(f"Graph data not found for model {model_id}. This could be due to the model not existing or an issue fetching its neighborhood.")
            raise HTTPException(
                status_code=404, detail=f"Graph data not found for model {model_id}"
            )
        return graph_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving graph data for model {model_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error retrieving model graph data."
        )


# --- NEW Endpoint: Get Related Entities ---
@router.get(
    "/related/{start_node_label}/{start_node_prop}/{start_node_val}",
    response_model=List[Dict[str, Any]],  # Basic response for now
    summary="Get related entities",
    description="Retrieves related entities connected by a specific relationship type and direction.",
    tags=["Graph"],
)
async def get_related_entities_endpoint(
    start_node_label: str = Path(
        ..., description="Label of the starting node (e.g., 'Paper', 'Task')."
    ),
    start_node_prop: str = Path(
        ...,
        description="Property name used to identify the start node (e.g., 'pwc_id', 'name').",
    ),
    start_node_val: str = Path(
        ..., description="Value of the identifying property for the start node."
    ),
    relationship_type: str = Query(
        ...,
        description="The type of relationship to traverse (e.g., 'HAS_TASK', 'AUTHORED').",
    ),
    target_node_label: str = Query(
        ...,
        description="The label of the target nodes to retrieve (e.g., 'Paper', 'Author').",
    ),
    direction: Literal["IN", "OUT", "BOTH"] = Query(
        "BOTH", description="Direction of the relationship relative to the start node."
    ),
    limit: int = Query(
        25, description="Maximum number of related entities to return.", ge=1, le=100
    ),
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> List[Dict[str, Any]]:
    logger.info(
        f"Received request for related entities: start={start_node_label}:{start_node_prop}={start_node_val}, "
        f"rel={relationship_type}, target={target_node_label}, dir={direction}, limit={limit}"
    )
    try:
        # Note: start_node_val from path is always string. Neo4jRepository might need type conversion
        # if the property is numeric, but for pwc_id/name it should be fine.
        related_entities = await graph_service.get_related_entities(
            start_node_label=start_node_label,
            start_node_prop=start_node_prop,
            start_node_val=start_node_val,
            relationship_type=relationship_type,
            target_node_label=target_node_label,
            direction=direction,
            limit=limit,
        )
        # Service returns empty list if Neo4j unavailable or on repo errors
        # No need to raise 404 here, empty list is a valid response.
        return related_entities
    except HTTPException as http_exc:
        raise http_exc  # Should not happen unless repo raises it?
    except Exception as e:
        logger.exception(f"Error retrieving related entities: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error retrieving related entities."
        )

@router.get(
    "/hf_models/{model_id:path}/radial_focus",
    response_model=GraphData,
    summary="Get 1-hop Radial Focus Graph for a Hugging Face Model",
    description="Fetches the 1-hop neighborhood graph for a specified Hugging Face model ID, optimized for a radial/focused display. Includes DERIVED_FROM models, MENTIONED papers, and HAS_TASK tasks.",
    tags=["Graph"]
)
async def get_model_radial_focus_graph_data(
    model_id: str = Path(..., description="The Hugging Face model ID (e.g., 'deepseek-ai/DeepSeek-R1') to get the focus graph for."),
    neo4j_repo: Neo4jRepository = Depends(deps.get_neo4j_repository)
) -> GraphData:
    logger.info(f"---> ENTERED get_model_radial_focus_graph_data for model_id: {model_id}")
    logger.info(f"Received request for radial focus graph for model_id: {model_id}")
    try:
        graph_data_dict = await neo4j_repo.get_radial_focus_graph(focus_model_id=model_id)
        
        if graph_data_dict is None:
            logger.warning(f"No radial focus graph data found for model {model_id}. Returning empty graph.")
            return GraphData(nodes=[], relationships=[]) 

        return GraphData(**graph_data_dict)
    except Exception as e:
        logger.error(f"Error in get_model_radial_focus_graph_data for {model_id}: {{e}}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving radial focus graph for model {model_id}.")
