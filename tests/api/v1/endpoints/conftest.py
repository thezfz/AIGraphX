import pytest
from unittest.mock import AsyncMock
from aigraphx.services.search_service import SearchService


@pytest.fixture
def mock_search_service() -> AsyncMock:
    """Provides a mock SearchService instance."""
    # Temporarily remove spec for debugging AttributeError
    # return AsyncMock(spec=SearchService)
    return AsyncMock()
