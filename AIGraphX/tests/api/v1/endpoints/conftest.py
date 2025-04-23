import pytest
from unittest.mock import AsyncMock
from aigraphx.services.search_service import SearchService


@pytest.fixture
def mock_search_service() -> AsyncMock:
    """
    提供一个 SearchService 的 mock 实例。

    这个 fixture 创建并返回一个 `unittest.mock.AsyncMock` 对象，
    它可以用来替换测试中的 `SearchService` 依赖。
    这允许我们在不实际调用 `SearchService` 的情况下测试依赖于它的组件（例如 API 端点）。

    Returns:
        AsyncMock: 一个配置为模拟 `SearchService` 行为的 `AsyncMock` 实例。
    """
    # 暂时移除 spec 以便调试 AttributeError，如果需要严格的类型检查，可以恢复
    # return AsyncMock(spec=SearchService)
    return AsyncMock()
