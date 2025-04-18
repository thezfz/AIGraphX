import pytest
from unittest.mock import AsyncMock, patch
from starlette.testclient import TestClient  # Add this import

# from fastapi.testclient import TestClient # Remove commented import
from httpx import AsyncClient  # Import AsyncClient
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, date

# Import original dependency functions for override keys
from aigraphx.api.v1 import dependencies as deps
from aigraphx.services.graph_service import GraphService
from aigraphx.models.graph import (
    GraphData,
    Node,
    Relationship,
    HFModelDetail,
    PaperDetailResponse,
)

# from aigraphx.models.search import SearchResultItem # MOCK_PAPER_DETAIL is now PaperDetailResponse
# from aigraphx.services.search_service import SearchService # Not directly used in mocks anymore
from fastapi import FastAPI

# Mark all async tests in this module to use the default function-scoped event loop
pytestmark = pytest.mark.asyncio

# --- Mock Data (Use PaperDetailResponse for MOCK_PAPER_DETAIL) ---
MOCK_GRAPH_DATA = GraphData(
    nodes=[
        Node(id="pwc-1", label="Center Paper", type="Paper", properties={}),
        Node(id="pwc-2", label="Cited Paper", type="Paper", properties={}),
    ],
    relationships=[
        Relationship(source="pwc-1", target="pwc-2", type="CITES", properties={})
    ],
)

MOCK_MODEL_DETAIL = HFModelDetail(
    model_id="hf-model-1",
    author="TestAuthor",
    sha="abcdef123",
    last_modified=datetime.now(),
    tags=["test"],
    pipeline_tag="text-classification",
    downloads=100,
    likes=10,
    library_name="transformers",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

# Use PaperDetailResponse model for consistency
MOCK_PAPER_DETAIL = PaperDetailResponse(
    pwc_id="pwc-found",
    title="Mock Paper Detail",
    abstract="Mock abstract.",
    url_abs="http://example.com/abs",
    url_pdf="http://example.com/pdf",
    published_date=date(2023, 2, 15),
    authors=["Author A", "Author B"],
    tasks=["Task A"],
    datasets=["Dataset X"],
    methods=[],
    frameworks=[],
    number_of_stars=10,
    area="NLP",
)

MOCK_PAPER_NEIGHBORHOOD_DATA = {
    "nodes": [
        {"id": "pwc-found", "label": "Paper", "title": "Found Paper Title"},
        {"id": "pwc-related-1", "label": "Paper", "title": "Related Paper 1"},
    ],
    "edges": [{"from": "pwc-found", "to": "pwc-related-1", "label": "CITES"}],
}

# --- Test Cases ---


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_paper_graph_success(client: AsyncClient, test_app: FastAPI) -> None:
    """Test successful retrieval of paper graph data (overriding service)."""
    # --- 1. Create Mock Service ---
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_paper_graph = AsyncMock(return_value=MOCK_GRAPH_DATA)

    # --- 2. Apply Override ---
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. Execute API Call ---
        response = await client.get("/api/v1/graph/papers/pwc-1/graph")

        # --- 4. Assertions ---
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"][0]["id"] == "pwc-1"
        # assert data == MOCK_GRAPH_DATA.model_dump(mode='json') # More robust check

        # --- 5. Verify Mock Call ---
        mock_graph_service.get_paper_graph.assert_awaited_once_with("pwc-1")
    finally:
        # --- 6. Restore Overrides ---
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_paper_graph_not_found(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """Test case where paper graph data is not found (404) (overriding service)."""
    # --- 1. Create Mock Service ---
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_paper_graph.return_value = None

    # --- 2. Apply Override ---
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. Execute API Call ---
        response = await client.get("/api/v1/graph/papers/not-found-id/graph")

        # --- 4. Assertions ---
        assert response.status_code == 404
        assert "Graph data not found" in response.json()["detail"]

        # --- 5. Verify Mock Call ---
        mock_graph_service.get_paper_graph.assert_awaited_once_with("not-found-id")
    finally:
        # --- 6. Restore Overrides ---
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_paper_graph_error(client: AsyncClient, test_app: FastAPI) -> None:
    """Test error handling during paper graph retrieval (overriding service)."""
    # --- 1. Create Mock Service ---
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_paper_graph = AsyncMock(
        side_effect=Exception("Service Graph Error")
    )

    # --- 2. Apply Override ---
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. Execute API Call ---
        response = await client.get("/api/v1/graph/papers/error-id/graph")

        # --- 4. Assertions ---
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

        # --- 5. Verify Mock Call ---
        mock_graph_service.get_paper_graph.assert_awaited_once_with("error-id")
    finally:
        # --- 6. Restore Overrides ---
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_model_details_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """Test successful retrieval of model details (overriding service)."""
    # --- 1. Create Mock Service ---
    mock_graph_service = AsyncMock(spec=GraphService)

    # --- Configure Mock Return Value ---
    mock_return_data = HFModelDetail(  # Use the correct model
        model_id="hf-model-1",
        author="Mock Author",
        sha="mocksha123",
        last_modified=datetime.now(),
        tags=["tag1", "tag2"],
        pipeline_tag="text-classification",
        downloads=100,
        likes=10,
        library_name="transformers",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_graph_service.get_model_details.return_value = mock_return_data

    # --- 2. Apply Override ---
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. Execute API Call ---
        response = await client.get("/api/v1/graph/models/hf-model-1")

        # --- 4. Assertions ---
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "hf-model-1"
        assert data["author"] == "Mock Author"
        assert isinstance(data["last_modified"], str)

        # --- 5. Verify Mock Call ---
        mock_graph_service.get_model_details.assert_awaited_once_with("hf-model-1")
    finally:
        # --- 6. Restore Overrides ---
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_model_details_not_found(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """Test case where model details are not found (404) (overriding service)."""
    # --- 1. Create Mock Service ---
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_model_details.return_value = None

    # --- 2. Apply Override ---
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. Execute API Call ---
        response = await client.get("/api/v1/graph/models/not-found-model")

        # --- 4. Assertions ---
        assert response.status_code == 404
        assert "Model details not found" in response.json()["detail"]

        # --- 5. Verify Mock Call ---
        mock_graph_service.get_model_details.assert_awaited_once_with("not-found-model")
    finally:
        # --- 6. Restore Overrides ---
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_model_details_error(client: AsyncClient, test_app: FastAPI) -> None:
    """Test error handling during model detail retrieval (overriding service)."""
    # --- 1. Create Mock Service ---
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_model_details.side_effect = Exception("Service Model Error")

    # --- 2. Apply Override ---
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        # --- 3. Execute API Call ---
        response = await client.get("/api/v1/graph/models/error-model")

        # --- 4. Assertions ---
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

        # --- 5. Verify Mock Call ---
        mock_graph_service.get_model_details.assert_awaited_once_with("error-model")
    finally:
        # --- 6. Restore Overrides ---
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_paper_details_success(
    client: AsyncClient, test_app: FastAPI
) -> None:
    """Test successful retrieval of paper details (overriding GRAPH service dependency for this endpoint)."""
    mock_graph_service = AsyncMock(spec=GraphService)
    # Use the corrected MOCK_PAPER_DETAIL which is now PaperDetailResponse
    mock_graph_service.get_paper_details.return_value = MOCK_PAPER_DETAIL

    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)

    try:
        response = await client.get("/api/v1/graph/papers/pwc-found")
        assert response.status_code == 200
        assert response.json()["pwc_id"] == MOCK_PAPER_DETAIL.pwc_id
        mock_graph_service.get_paper_details.assert_awaited_once_with(
            pwc_id="pwc-found"
        )
    finally:
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_paper_details_not_found(
    client: AsyncClient, test_app: FastAPI
) -> None:
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_paper_details.return_value = None
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)
    try:
        response = await client.get("/api/v1/graph/papers/pwc-not-found")
        assert response.status_code == 404
        assert "Paper with PWC ID" in response.json()["detail"]
        mock_graph_service.get_paper_details.assert_awaited_once_with(
            pwc_id="pwc-not-found"
        )
    finally:
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


@pytest.mark.asyncio
# Add test_app fixture to signature
async def test_get_paper_details_service_error(
    client: AsyncClient, test_app: FastAPI
) -> None:
    mock_graph_service = AsyncMock(spec=GraphService)
    mock_graph_service.get_paper_details.side_effect = Exception(
        "Graph Service Internal Error"
    )
    # Use test_app for overrides
    original_overrides = test_app.dependency_overrides.copy()
    override_dict = {deps.get_graph_service: lambda: mock_graph_service}
    test_app.dependency_overrides.update(override_dict)
    try:
        response = await client.get("/api/v1/graph/papers/pwc-error")
        assert response.status_code == 500
        assert "An internal server error occurred" in response.json()["detail"]
        assert "pwc-error" in response.json()["detail"]
        mock_graph_service.get_paper_details.assert_awaited_once_with(
            pwc_id="pwc-error"
        )
    finally:
        # Use test_app for overrides
        test_app.dependency_overrides = original_overrides


# --- Tests for GET /related/{start_node_label}/{start_node_prop}/{start_node_val} ---


@pytest.mark.asyncio
async def test_get_related_entities_success(
    client: AsyncClient, mock_graph_service_fixture: AsyncMock
) -> None:
    """Test successfully retrieving related entities."""
    start_label = "Paper"
    start_prop = "pwc_id"
    start_val = "test-paper"
    rel_type = "HAS_TASK"
    target_label = "Task"
    direction = "OUT"
    limit = 5

    mock_return_data = [
        {"name": "Task A", "id": "task-a"},
        {"name": "Task B", "id": "task-b"},
    ]
    mock_graph_service_fixture.get_related_entities.return_value = mock_return_data

    url = f"/api/v1/graph/related/{start_label}/{start_prop}/{start_val}"
    params = {
        "relationship_type": rel_type,
        "target_node_label": target_label,
        "direction": direction,
        "limit": str(limit),
    }
    response = await client.get(url, params=params)

    assert response.status_code == 200
    assert response.json() == mock_return_data
    mock_graph_service_fixture.get_related_entities.assert_awaited_once_with(
        start_node_label=start_label,
        start_node_prop=start_prop,
        start_node_val=start_val,
        relationship_type=rel_type,
        target_node_label=target_label,
        direction=direction,
        limit=limit,
    )


@pytest.mark.asyncio
async def test_get_related_entities_not_found(
    client: AsyncClient, mock_graph_service_fixture: AsyncMock
) -> None:
    """Test retrieving related entities when service returns an empty list."""
    mock_graph_service_fixture.get_related_entities.return_value = []

    url = "/api/v1/graph/related/Paper/pwc_id/not-found-paper"
    params = {"relationship_type": "HAS_TASK", "target_node_label": "Task"}
    response = await client.get(url, params=params)

    assert response.status_code == 200
    assert response.json() == []
    mock_graph_service_fixture.get_related_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_related_entities_missing_params(client: AsyncClient) -> None:
    """Test request with missing required query parameters."""
    url = "/api/v1/graph/related/Paper/pwc_id/some-paper"
    # Missing relationship_type and target_node_label
    response = await client.get(url)
    # NOTE: Expecting 422 Unprocessable Entity as FastAPI validates query parameters
    # *before* dependency injection in this case.
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_related_entities_invalid_direction(client: AsyncClient) -> None:
    """Test request with invalid value for 'direction' parameter."""
    url = "/api/v1/graph/related/Paper/pwc_id/some-paper"
    params = {
        "relationship_type": "HAS_TASK",
        "target_node_label": "Task",
        "direction": "INVALID_DIRECTION",  # Incorrect value
    }
    response = await client.get(url, params=params)
    # NOTE: Expecting 422 Unprocessable Entity as FastAPI validates enum values
    # in query parameters *before* dependency injection.
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_related_entities_service_error(
    client: AsyncClient, mock_graph_service_fixture: AsyncMock
) -> None:
    """Test when the graph service raises an unexpected exception."""
    # Simulate the service raising an error (which the endpoint should catch)
    mock_graph_service_fixture.get_related_entities.side_effect = Exception(
        "Service layer crashed"
    )

    url = "/api/v1/graph/related/Paper/pwc_id/error-paper"
    params = {"relationship_type": "HAS_TASK", "target_node_label": "Task"}
    response = await client.get(url, params=params)

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]
    mock_graph_service_fixture.get_related_entities.assert_awaited_once()
