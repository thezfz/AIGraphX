import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import Dict
from functools import partial  # Import partial

# Import lifespan manager and settings
from aigraphx.core.db import lifespan
from aigraphx.core.config import settings  # Import the settings object

# Import the main v1 API router
from aigraphx.api.v1.api import api_router as api_v1_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prepare the lifespan function with settings
# Use partial to fix the 'settings' argument for the lifespan context manager
lifespan_with_settings = partial(lifespan, settings=settings)

# Initialize FastAPI app with the prepared lifespan context manager
app = FastAPI(
    title="AIGraphX API",
    description="API for interacting with the AI Knowledge Graph.",
    version="1.0.0",
    lifespan=lifespan_with_settings,  # Attach the lifespan manager with settings
)

# CORS Configuration (adjust origins as needed for your frontend)
origins = [
    "http://localhost",
    "http://localhost:3000",  # Example frontend port
    # Add other origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# TODO: Add lifespan context manager later for resource initialization/cleanup
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Application startup: Initializing resources...")
#     # Initialize DB connections, load models etc. using core.db functions
#     await core.db.initialize_resources()

# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Application shutdown: Cleaning up resources...")
#     # Close DB connections etc.
#     await core.db.cleanup_resources()


# Simple root endpoint
@app.get("/")
async def read_root() -> Dict[str, str]:
    return {"message": "Welcome to AIGraphX API"}


# Health Check Endpoint
@app.get("/health", status_code=200, tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.
    Returns 200 OK with status "ok" if the server is running.
    """
    return {"status": "ok"}


# FIXED: Include the main v1 router with the /api/v1 prefix
app.include_router(api_v1_router, prefix="/api/v1")

# REMOVED: Direct inclusion of endpoint routers
# from aigraphx.api.v1.endpoints import search # Import the search router
# from aigraphx.api.v1.endpoints import graph # Import the graph router
# app.include_router(search.router, prefix="/api/v1", tags=["Search"])
# app.include_router(graph.router, prefix="/api/v1", tags=["Graph"])


# --- Run configuration for debugging (optional) ---
# Use uvicorn programmatically for more control if needed,
# otherwise run from terminal: uvicorn aigraphx.main:app --reload
if __name__ == "__main__":
    # Note: This is mainly for debugging. Use `uvicorn` command for production/dev.
    logger.info("Running FastAPI app with uvicorn (debug mode)...")
    uvicorn.run(
        "aigraphx.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
