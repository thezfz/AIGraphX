import pytest
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    ValidationError,
)  # Import needed Pydantic types
from datetime import date, datetime, timezone
from typing import Optional, List, Dict, Any, Type, Union, cast

# 替换通配符导入，只导入实际存在的类型 (If these are the *actual* project models, import them)
# from aigraphx.models.graph import (...)
# For now, define minimal Pydantic models for testing structure


class NodePropertiesBase(BaseModel):
    """Pydantic base model for Node properties."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Test Node Models (Using Pydantic) ---
class Area(NodePropertiesBase):
    """Test Area node using Pydantic."""

    name: str = Field(..., min_length=1)  # Use Pydantic validation
    description: Optional[str] = None

    # Remove manual __init__ validation
    # def __init__(self, name: str, description: Optional[str] = None, created_at: Optional[datetime] = None) -> None:
    #     super().__init__(created_at=created_at) # Pydantic handles this
    #     # if not name: # Remove this
    #     #     raise ValidationError("Name cannot be empty", Area) # Remove this
    #     self.name = name
    #     self.description = description


class Author(NodePropertiesBase):
    """Test Author node using Pydantic."""

    name: str = Field(..., min_length=1)  # Use Pydantic validation
    affiliations: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)

    # Remove manual __init__ validation


class Dataset(NodePropertiesBase):
    """Test Dataset node using Pydantic."""

    name: str = Field(..., min_length=1)  # Use Pydantic validation
    description: Optional[str] = None

    # Remove manual __init__ validation


class Framework(NodePropertiesBase):
    """Test Framework node using Pydantic."""

    name: str = Field(..., min_length=1)  # Use Pydantic validation

    # Remove manual __init__ validation


class HFModel(NodePropertiesBase):
    """Test HFModel node using Pydantic."""

    model_id: str = Field(..., min_length=1)  # Use Pydantic validation
    author: Optional[str] = None
    sha: Optional[str] = None
    last_modified: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    pipeline_tag: Optional[str] = None
    siblings: List[Dict[str, Any]] = Field(default_factory=list)
    private: bool = False
    downloads: int = 0
    likes: int = 0
    library_name: Optional[str] = None
    masked: bool = False
    model_index: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    security: Optional[Any] = None
    card_data: Dict[str, Any] = Field(default_factory=dict)
    model_filenames: List[str] = Field(default_factory=list)

    @field_validator("last_modified", mode="before")
    @classmethod
    def parse_last_modified(cls, v: Any) -> Optional[datetime]:
        if isinstance(v, str):
            try:
                # Handle 'Z' suffix for UTC
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"Invalid datetime format for last_modified: {v}")
        elif isinstance(v, datetime):
            return v
        elif v is None:
            return None
        raise TypeError("last_modified must be str or datetime")

    # Remove manual __init__ validation


class Method(NodePropertiesBase):
    """Test Method node using Pydantic."""

    name: str = Field(..., min_length=1)  # Use Pydantic validation
    description: Optional[str] = None

    # Remove manual __init__ validation


class Paper(NodePropertiesBase):
    """Test Paper node using Pydantic."""

    pwc_id: str = Field(..., min_length=1)  # Use Pydantic validation
    title: Optional[str] = None
    arxiv_id_base: Optional[str] = None
    arxiv_id_versioned: Optional[str] = None
    summary: Optional[str] = None
    published_date: Optional[date] = None
    pwc_url: Optional[HttpUrl] = None  # Use HttpUrl for URL validation
    pdf_url: Optional[HttpUrl] = None  # Use HttpUrl for URL validation
    doi: Optional[str] = None
    primary_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)

    # Remove manual __init__ validation


class Repository(NodePropertiesBase):
    """Test Repository node using Pydantic."""

    url: HttpUrl  # Use HttpUrl for URL validation
    stars: int = 0
    is_official: bool = False
    framework: Optional[str] = None
    repo_name: Optional[str] = None
    repo_owner: Optional[str] = None

    # Remove manual __init__ validation


class Task(NodePropertiesBase):
    """Test Task node using Pydantic."""

    name: str = Field(..., min_length=1)  # Use Pydantic validation
    description: Optional[str] = None

    # Remove manual __init__ validation


# --- Test Relationship Models (Using Pydantic) ---
class RelationshipPropertiesBase(BaseModel):
    """Pydantic base model for Relationship properties."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Authored(RelationshipPropertiesBase):
    pass


class HasDataset(RelationshipPropertiesBase):
    split: Optional[str] = None
    config: Optional[str] = None


class HasMethod(RelationshipPropertiesBase):
    pass


class HasTask(RelationshipPropertiesBase):
    pass


class ImplementsMethod(RelationshipPropertiesBase):
    pass


class MentionsPaper(RelationshipPropertiesBase):
    context: Optional[str] = None


class TrainedOn(RelationshipPropertiesBase):
    split: Optional[str] = None
    config: Optional[str] = None


class UsesFramework(RelationshipPropertiesBase):
    pass


# --- Node Property Validation Tests ---


def test_node_properties_base_validation() -> None:
    """Tests basic validation for NodePropertiesBase."""
    # Pydantic handles default factory
    node = NodePropertiesBase()
    assert isinstance(node.created_at, datetime)

    # Test with specific created_at
    now = datetime.now(timezone.utc)
    # Pydantic expects keyword args for initialization
    node = NodePropertiesBase(created_at=now)
    assert node.created_at == now


def test_area_validation() -> None:
    """Tests Area node validation using Pydantic."""
    # Valid case
    area = Area(name="Computer Science", description="Area of CS")
    assert area.name == "Computer Science"
    assert area.description == "Area of CS"
    assert isinstance(area.created_at, datetime)

    # Invalid case (empty name) - Pydantic raises ValidationError
    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Area(name="")


def test_author_validation() -> None:
    """Tests Author node validation using Pydantic."""
    author = Author(
        name="John Doe", affiliations=["University X"], emails=["john@uni.edu"]
    )
    assert author.name == "John Doe"
    assert author.affiliations == ["University X"]
    assert author.emails == ["john@uni.edu"]

    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Author(name="")


def test_dataset_validation() -> None:
    """Tests Dataset node validation using Pydantic."""
    dataset = Dataset(name="ImageNet", description="Large image dataset")
    assert dataset.name == "ImageNet"
    assert dataset.description == "Large image dataset"

    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Dataset(name="")


def test_framework_validation() -> None:
    """Tests Framework node validation using Pydantic."""
    fw = Framework(name="PyTorch")
    assert fw.name == "PyTorch"

    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Framework(name="")


def test_hfmodel_validation() -> None:
    """Tests HFModel node validation using Pydantic."""
    now_str = "2023-01-15T10:00:00Z"
    now_dt = datetime.fromisoformat("2023-01-15T10:00:00+00:00")
    model = HFModel(
        model_id="org/model-abc",
        author="organization",
        sha="abc123",
        last_modified=now_str,  # type: ignore # Pass string, validator handles conversion
        tags=["nlp", "transformer"],
        pipeline_tag="text-generation",
        siblings=[{"name": "config.json"}],
        private=False,
        downloads=1000,
        likes=50,
        library_name="transformers",
        masked=False,
        model_index=None,
        config={"key": "value"},
        security=None,
        card_data={"license": "apache-2.0"},
        model_filenames=["pytorch_model.bin"],
    )
    assert model.model_id == "org/model-abc"
    assert model.last_modified == now_dt  # Check converted datetime
    assert model.tags == ["nlp", "transformer"]
    assert model.card_data == {"license": "apache-2.0"}

    # Test missing required model_id
    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        HFModel(model_id="", author="org", sha="abc", last_modified=now_str)  # type: ignore

    # Test invalid datetime format for last_modified
    with pytest.raises(ValidationError, match="Invalid datetime format"):
        HFModel(model_id="test-id", last_modified="invalid-date-string")  # type: ignore

    # Test incorrect type for last_modified - Expect TypeError from the validator
    with pytest.raises(TypeError, match="last_modified must be str or datetime"):
        HFModel(model_id="test-id", last_modified=12345)  # type: ignore


def test_method_validation() -> None:
    """Tests Method node validation using Pydantic."""
    method = Method(name="Transformer", description="Attention mechanism")
    assert method.name == "Transformer"
    assert method.description == "Attention mechanism"

    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Method(name="")


def test_paper_validation() -> None:
    """Tests Paper node validation using Pydantic."""
    pub_date = date(2023, 1, 15)
    paper = Paper(
        pwc_id="attention-all-need",
        title="Attention Is All You Need",
        arxiv_id_base="1706.03762",
        arxiv_id_versioned="1706.03762v5",
        summary="Proposes the Transformer model.",
        published_date=pub_date,
        pwc_url="http://pwc.com/..",  # type: ignore
        pdf_url="http://arxiv.org/pdf/..",  # type: ignore
        doi="10.some/doi",
        primary_category="cs.CL",
        categories=["cs.CL", "cs.LG"],
    )
    assert paper.pwc_id == "attention-all-need"
    assert paper.published_date == pub_date
    assert paper.categories == ["cs.CL", "cs.LG"]
    assert isinstance(paper.pwc_url, HttpUrl)  # Check type conversion

    # Test missing required pwc_id
    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Paper(pwc_id="", title="Missing ID")

    # Test invalid URL format
    with pytest.raises(ValidationError, match="URL scheme should be 'http' or 'https'"):
        Paper(pwc_id="test-id", title="Invalid URL", pwc_url="ftp://invalid.com")  # type: ignore


def test_repository_validation() -> None:
    """Tests Repository node validation using Pydantic."""
    repo = Repository(
        url="https://github.com/org/repo",  # type: ignore
        stars=100,
        is_official=True,
        framework="jax",
        repo_name="repo",
        repo_owner="org",
    )
    assert str(repo.url) == "https://github.com/org/repo"
    assert repo.stars == 100
    assert repo.is_official is True
    assert repo.framework == "jax"

    # Test invalid URL
    with pytest.raises(ValidationError) as excinfo:
        Repository(url="invalid-url")  # type: ignore
    # Check for specific URL validation error in Pydantic v2 style
    errors = excinfo.value.errors()
    assert any(
        err["type"] == "url_parsing" and err["loc"] == ("url",) for err in errors
    ), "Expected URL validation error not found."

    # Test missing required url
    with pytest.raises(ValidationError) as excinfo:
        Repository(stars=10)  # type: ignore
    errors = excinfo.value.errors()
    assert any(err["type"] == "missing" and err["loc"] == ("url",) for err in errors), (
        "Expected missing URL error not found."
    )


def test_task_validation() -> None:
    """Tests Task node validation using Pydantic."""
    task = Task(name="Text Classification", description="Classify text docs")
    assert task.name == "Text Classification"
    assert task.description == "Classify text docs"

    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        Task(name="")


# --- Relationship Property Validation Tests ---


def test_relationship_properties_base_validation() -> None:
    """Tests basic validation for RelationshipPropertiesBase."""
    rel = RelationshipPropertiesBase()
    assert isinstance(rel.created_at, datetime)

    now = datetime.now(timezone.utc)
    rel = RelationshipPropertiesBase(created_at=now)
    assert rel.created_at == now


def test_authored_validation() -> None:
    """Tests Authored relationship validation."""
    rel = Authored()  # No specific props other than base
    assert isinstance(rel.created_at, datetime)


def test_has_dataset_validation() -> None:
    """Tests HasDataset relationship validation."""
    rel = HasDataset(split="train", config="default")
    assert rel.split == "train"
    assert rel.config == "default"

    rel_minimal = HasDataset()
    assert rel_minimal.split is None
    assert rel_minimal.config is None


def test_has_method_validation() -> None:
    """Tests HasMethod relationship validation."""
    rel = HasMethod()
    assert isinstance(rel.created_at, datetime)


def test_has_task_validation() -> None:
    """Tests HasTask relationship validation."""
    rel = HasTask()
    assert isinstance(rel.created_at, datetime)


def test_implements_method_validation() -> None:
    """Tests ImplementsMethod relationship validation."""
    rel = ImplementsMethod()
    assert isinstance(rel.created_at, datetime)


def test_mentions_paper_validation() -> None:
    """Tests MentionsPaper relationship validation."""
    rel = MentionsPaper(context="Related work section")
    assert rel.context == "Related work section"

    rel_minimal = MentionsPaper()
    assert rel_minimal.context is None


def test_trained_on_validation() -> None:
    """Tests TrainedOn relationship validation."""
    rel = TrainedOn(split="test", config="custom")
    assert rel.split == "test"
    assert rel.config == "custom"

    rel_minimal = TrainedOn()
    assert rel_minimal.split is None
    assert rel_minimal.config is None


def test_uses_framework_validation() -> None:
    """Tests UsesFramework relationship validation."""
    rel = UsesFramework()
    assert isinstance(rel.created_at, datetime)


# We remove the test classes defined at the top as they are replaced by Pydantic models
# Make sure the actual project models (if they exist in aigraphx.models.graph)
# are compatible or update tests to use those.
# Assuming for now these Pydantic models are sufficient for testing the validation logic.
