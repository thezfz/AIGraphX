from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date


# Define base models for Graph Data structure
class Node(BaseModel):
    """Represents a node in the graph."""

    id: str = Field(
        ..., description="Unique identifier for the node (e.g., pwc_id or model_id)"
    )
    label: Optional[str] = Field(
        None, description="Display label for the node (e.g., paper title or model name)"
    )
    type: str = Field(
        ..., description="Type of the node (e.g., 'Paper', 'HFModel', 'Concept')"
    )
    # Include other common properties if needed from Neo4j
    properties: Dict[str, Any] = Field(
        {}, description="Additional properties of the node."
    )


class Relationship(BaseModel):
    """Represents a relationship (edge) between two nodes."""

    source: str = Field(..., description="ID of the source node.")
    target: str = Field(..., description="ID of the target node.")
    type: str = Field(
        ...,
        description="Type of the relationship (e.g., 'CITES', 'RELATED_TO', 'USES_MODEL').",
    )
    # Include properties if relationships have them
    properties: Dict[str, Any] = Field(
        {}, description="Properties of the relationship."
    )


class GraphData(BaseModel):
    """Represents the data structure for graph visualization or analysis."""

    nodes: List[Node] = Field(
        ..., description="List of nodes in the graph neighborhood."
    )
    relationships: List[Relationship] = Field(
        ..., description="List of relationships connecting the nodes."
    )


# Define models related to graph data responses


class PaperDetailResponse(BaseModel):
    """Detailed information about a single paper."""

    pwc_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    url_abs: Optional[str] = None
    url_pdf: Optional[str] = None
    published_date: Optional[date] = None
    authors: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    methods: Optional[List[str]] = None
    datasets: Optional[List[str]] = None
    repositories: Optional[List[Dict[str, Any]]] = None
    area: Optional[str] = None
    conference: Optional[str] = None
    # Add neighborhood later if needed

    model_config = ConfigDict(from_attributes=True)


class HFModelDetail(BaseModel):
    """Detailed information for a Hugging Face model."""

    model_id: str = Field(..., description="The unique Hugging Face model ID.")
    author: Optional[str] = Field(None, description="The author or organization.")
    sha: Optional[str] = Field(
        None, description="The Git commit SHA associated with the model version."
    )
    last_modified: Optional[datetime] = Field(
        None, description="Timestamp of the last modification."
    )
    tags: Optional[List[str]] = Field(
        None, description="List of tags associated with the model."
    )
    pipeline_tag: Optional[str] = Field(
        None, description="The primary task pipeline tag."
    )
    downloads: Optional[int] = Field(None, description="Number of downloads.")
    likes: Optional[int] = Field(None, description="Number of likes.")
    library_name: Optional[str] = Field(
        None, description="The library associated with the model (e.g., transformers)."
    )
    readme_content: Optional[str] = Field(None, description="The README content of the model.")
    dataset_links: Optional[List[str]] = Field(None, description="Links to datasets used by the model.")
    created_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was created in the database."
    )
    updated_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was last updated in the database."
    )

    @field_validator("last_modified", mode="before")
    @classmethod
    def parse_last_modified(cls, v: Any) -> Optional[datetime]:
        if isinstance(v, str):
            try:
                # Handle 'Z' suffix for UTC
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                # Raise ValueError for Pydantic to catch as validation error
                raise ValueError(
                    f"Invalid datetime format for last_modified: '{v}'"
                ) from e
        elif isinstance(v, datetime):
            return v
        elif v is None:
            return None
        # Raise ValueError instead of TypeError for invalid types
        raise ValueError(
            "last_modified must be a valid ISO 8601 string or datetime object"
        )

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )  # Keep existing config


# --- Potentially add other models later ---
# class TaskDetailResponse(BaseModel): ...
# class DatasetDetailResponse(BaseModel): ...
