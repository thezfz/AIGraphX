# aigraphx/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Import ConfigDict for type hint if needed, but pydantic-settings handles config internally
from pydantic import Field  # Field might be needed if we use alias

# Re-import BaseSettings and SettingsConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Try importing SettingsConfigDict for stricter typing if available
# from pydantic_settings import SettingsConfigDict # May not be directly available
from typing import Optional, List

# --- REMOVE Manual .env loading --- #
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# dotenv_path = os.path.join(project_root, ".env")
# loaded = load_dotenv(dotenv_path=dotenv_path)
# if not loaded:
#     logger.warning(f".env file not found or not loaded from path: {dotenv_path}")
# else:
#     logger.info(f".env file loaded from path: {dotenv_path}")
# --- End REMOVE --- #

# Check if running under pytest
IS_PYTEST = os.getenv("PYTEST_RUNNING") == "1"

# Re-calculate .env file path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dotenv_path = os.path.join(project_root, ".env")
# Add logging for path check
logger.info(f"Calculated .env path: {dotenv_path}")
logger.info(f"Does .env file exist at calculated path? {os.path.exists(dotenv_path)}")


class Settings(BaseSettings):
    # Explicitly configure .env file loading
    model_config = SettingsConfigDict(
        env_file=dotenv_path,
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra fields if needed, or use 'allow'
    )

    # General
    project_name: str = Field(default="AIGraphX", alias="PROJECT_NAME")
    api_v1_str: str = Field(default="/api/v1", alias="API_V1_STR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Database URLs (Primary)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    neo4j_uri: Optional[str] = Field(default=None, alias="NEO4J_URI")
    neo4j_username: Optional[str] = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: Optional[str] = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # Database Pool Sizes (Primary)
    pg_pool_min_size: int = Field(default=1, alias="PG_POOL_MIN_SIZE")
    pg_pool_max_size: int = Field(default=10, alias="PG_POOL_MAX_SIZE")

    # Embedding Settings
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2", alias="SENTENCE_TRANSFORMER_MODEL"
    )
    embedder_device: str = Field(default="cpu", alias="EMBEDDER_DEVICE")

    # Faiss Settings (Defaults) - Use Field with alias for env var mapping
    # Define the fields once
    faiss_index_path: str = Field(
        default="data/faiss_index.bin", alias="FAISS_INDEX_PATH"
    )
    faiss_mapping_path: str = Field(
        default="data/papers_faiss_ids.json", alias="FAISS_MAPPING_PATH"
    )
    models_faiss_index_path: str = Field(
        default="data/models_faiss.index", alias="MODELS_FAISS_INDEX_PATH"
    )
    models_faiss_mapping_path: str = Field(
        default="data/models_faiss_ids.json", alias="MODELS_FAISS_MAPPING_PATH"
    )
    # Batch size for building Faiss indexes
    build_faiss_batch_size: int = Field(default=128, alias="BUILD_FAISS_BATCH_SIZE")

    # API Keys (Load securely, avoid defaults in code)
    hf_api_key: Optional[str] = Field(default=None, alias="HF_API_KEY")
    github_api_key: Optional[str] = Field(default=None, alias="GITHUB_API_KEY")
    pwc_api_key: Optional[str] = Field(default=None, alias="PWC_API_KEY")

    # Add other settings as needed

    # Test specific overrides (will be applied *after* initial loading)
    # These fields allow loading TEST_* env vars into the settings object
    test_database_url: Optional[str] = Field(default=None, alias="TEST_DATABASE_URL")
    test_neo4j_uri: Optional[str] = Field(default=None, alias="TEST_NEO4J_URI")
    test_neo4j_password: Optional[str] = Field(
        default=None, alias="TEST_NEO4J_PASSWORD"
    )
    test_neo4j_database: Optional[str] = Field(
        default="neo4j", alias="TEST_NEO4J_DATABASE"
    )
    test_faiss_paper_index_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_PAPER_INDEX_PATH"
    )
    test_faiss_paper_mapping_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_PAPER_MAPPING_PATH"
    )
    test_faiss_model_index_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_MODEL_INDEX_PATH"
    )
    test_faiss_model_mapping_path: Optional[str] = Field(
        default=None, alias="TEST_FAISS_MODEL_MAPPING_PATH"
    )

    # --- REMOVED old class Config --- #


# Instantiate settings
settings = Settings()

# --- Log sensitive data presence/absence --- #
# (Keep existing logging checks)
if not settings.hf_api_key:
    logger.warning("HF_API_KEY not set in environment.")
if not settings.github_api_key:
    logger.warning("GITHUB_API_KEY not set in environment.")

# --- REMOVE duplicate definitions at the end --- #
# (Remove all variable definitions after `settings = Settings()`)
