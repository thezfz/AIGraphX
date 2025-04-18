from fastapi import APIRouter

# Import endpoint routers
from aigraphx.api.v1.endpoints import search as search_endpoints  # Renamed import

# from aigraphx.api.v1.endpoints import papers as papers_api # Remove if paper details are elsewhere
# from aigraphx.api.v1.endpoints import models as models_api # Remove if model details are elsewhere
from aigraphx.api.v1.endpoints import graph as graph_endpoints  # Renamed import

api_router = APIRouter()

# Include the search router with /search prefix. It defines /papers/ and /models/ internally.
api_router.include_router(search_endpoints.router, prefix="/search", tags=["Search"])

# Include the graph router. Assuming its paths are defined relative to root / or specific resources.
api_router.include_router(graph_endpoints.router, prefix="/graph", tags=["Graph"])

# Remove or adjust other includes if they conflict or are handled differently
# api_router.include_router(papers_api.router, prefix="/papers", tags=["Paper Details"])
# api_router.include_router(models_api.router, prefix="/models", tags=["Model Details"])
