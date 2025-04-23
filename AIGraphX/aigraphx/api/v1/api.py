# -*- coding: utf-8 -*-
"""
API 版本 v1 的主路由器聚合文件

此文件定义了 API 版本 1 (`/api/v1`) 的主 `APIRouter` 实例 (`api_router`)。
它的主要作用是聚合来自 `endpoints` 子目录下各个模块的子路由器，并为它们添加路径前缀。
通过这种方式，可以将不同功能的 API 端点组织到不同的文件中，保持代码的模块化和可维护性。

最终，这个 `api_router` 会被包含到 `aigraphx/main.py` 中的 FastAPI 应用实例 `app` 中，
并添加 `/api/v1` 的总前缀。因此，此文件中添加的 `prefix` 会与 `/api/v1` 组合。
例如，`search_endpoints.router` 设置了 `prefix="/search"`，那么它内部定义的 `/papers` 路径
最终会对应到 `/api/v1/search/papers`。

与其他文件的交互:
*   **`fastapi.APIRouter`**: 用于创建路由器实例。
*   **`aigraphx.api.v1.endpoints.search`**: 导入 `search_endpoints.router`，包含了与搜索相关的 API 端点。
*   **`aigraphx.api.v1.endpoints.graph`**: 导入 `graph_endpoints.router`，包含了与图谱数据相关的 API 端点。
*   **`aigraphx.main`**: 此文件定义的 `api_router` 会被 `main.py` 导入并挂载到 FastAPI 应用实例 `app` 上。
"""

# 导入 FastAPI 的 APIRouter 类
from fastapi import APIRouter

# 从 endpoints 子目录导入各个功能的子路由器
# 使用 'as' 重命名导入的模块，以明确区分，避免命名冲突
from aigraphx.api.v1.endpoints import (
    search as search_endpoints,
)  # 重命名导入的搜索端点模块
from aigraphx.api.v1.endpoints import (
    graph as graph_endpoints,
)  # 重命名导入的图谱端点模块

# 注意：以下导入被注释掉了，说明论文和模型的具体详情获取可能已整合到其他端点（如 search 或 graph）
# 或尚未实现/需要调整。
# from aigraphx.api.v1.endpoints import papers as papers_api # 如果论文详情不在别处，则取消注释
# from aigraphx.api.v1.endpoints import models as models_api # 如果模型详情不在别处，则取消注释


# 创建 API 版本 1 的主路由器实例
api_router = APIRouter()

# 使用 include_router 方法将各个子路由器包含到主路由器中
# include_router 的参数:
#   - 第一个参数: 要包含的子路由器实例 (例如 search_endpoints.router)。
#   - prefix (可选): 为这个子路由器下的所有路径添加统一的前缀。
#     这里的 prefix 会与 main.py 中添加的 /api/v1 结合。
#   - tags (可选): 为这个子路由器下的所有端点添加标签，方便在 API 文档 (如 Swagger UI) 中分组显示。

# 包含搜索相关的端点。
# 设置 prefix="/search"，意味着 search_endpoints.router 中定义的路径（如 /papers/ 和 /models/）
# 会分别变为 /api/v1/search/papers 和 /api/v1/search/models。
# 打上 "Search" 标签。
api_router.include_router(search_endpoints.router, prefix="/search", tags=["Search"])

# 包含图谱相关的端点。
# 设置 prefix="/graph"，意味着 graph_endpoints.router 中定义的路径会变为 /api/v1/graph/...
# 打上 "Graph" 标签。
api_router.include_router(graph_endpoints.router, prefix="/graph", tags=["Graph"])

# 注释掉的部分：如果需要独立的论文/模型详情端点，需要取消注释并确保 endpoints 下有对应的文件和 router。
# 注意调整 prefix 和 tags。
# api_router.include_router(papers_api.router, prefix="/papers", tags=["Paper Details"])
# api_router.include_router(models_api.router, prefix="/models", tags=["Model Details"])
