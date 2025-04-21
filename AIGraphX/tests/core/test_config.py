import os
import pytest
from typing import Dict, Any, Callable, Optional, Generator
from unittest import mock  # Keep for patching load_dotenv if still needed
from pydantic import PostgresDsn, HttpUrl  # Import types used in Settings
from pydantic_settings import SettingsConfigDict

# Import the config module itself AND the Settings class directly
import aigraphx.core.config
from aigraphx.core.config import Settings

# Define default values as expected in config.py if env var is missing
DEFAULT_PG_USER = "aigraphx_user"
DEFAULT_PG_PASS = "aigraphx_password"
DEFAULT_PG_DB = "aigraphx"
DEFAULT_PG_HOST = "localhost"
DEFAULT_PG_PORT = "5432"
DEFAULT_NEO4J_URI = "neo4j://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_DB = "neo4j"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Use a dictionary for expected values in tests for easier management
DEFAULT_CONFIG: Dict[str, Any] = {
    "POSTGRES_USER": "aigraphx_user",
    "POSTGRES_PASSWORD": "aigraphx_password",
    "POSTGRES_DB": "aigraphx",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "DATABASE_URL": "postgresql://aigraphx_user:aigraphx_password@localhost:5432/aigraphx",
    "NEO4J_URI": "neo4j://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": None,  # 不验证实际密码
    "NEO4J_DATABASE": "neo4j",
    "PWC_API_KEY": None,  # 不验证实际API密钥
    "HUGGINGFACE_API_KEY": None,  # 不验证实际API密钥
    "SENTENCE_TRANSFORMER_MODEL": "all-MiniLM-L6-v2",
    "RELOAD": False,
    "TEST_DATABASE_URL": None,
}

# Fix: Reflect that DATABASE_URL is built from POSTGRES_*
# Removed NEO4J_DATABASE as it's not part of MOCK_ENV_CONFIG setup keys
MOCK_ENV_CONFIG: Dict[str, Any] = {
    # Define POSTGRES vars used to build DATABASE_URL
    "POSTGRES_USER": "test_pg_user",
    "POSTGRES_PASSWORD": "test_pg_pass",
    "POSTGRES_DB": "test_pg_db",
    "POSTGRES_HOST": "test_pg_host",
    "POSTGRES_PORT": "1234",
    # Expected DATABASE_URL built from above vars
    "DATABASE_URL": "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db",
    "NEO4J_URI": "neo4j://test_neo4j_host:7777",
    "NEO4J_USERNAME": "test_neo4j_user",  # Key used in config.py
    "NEO4J_PASSWORD": None,  # 不再使用硬编码密码
    "PWC_API_KEY": None,  # 不再使用硬编码API密钥
    "HUGGINGFACE_API_KEY": None,  # 不再使用硬编码API密钥
    "SENTENCE_TRANSFORMER_MODEL": "test-model",
    "RELOAD": True,
    "TEST_DATABASE_URL": "postgresql://test:pass@host/db",  # Assuming corrected format
}

# --- Test Data (Define expected values for the settings object attributes) ---

# Expected values when using environment variables
EXPECTED_SETTINGS_ENV = {
    "project_name": "AIGraphX",
    "api_v1_str": "/api/v1",
    "log_level": "INFO",
    "environment": "development",  # Ensure this is development
    "database_url": "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db",  # Built from PG_* vars
    "pg_pool_min_size": 1,
    "pg_pool_max_size": 10,
    "neo4j_uri": "neo4j://test_neo4j_host:7777",
    "neo4j_username": "test_neo4j_user",
    "neo4j_password": None,  # 不再校验实际密码
    "neo4j_database": "test_neo4j_db",
    "pwc_api_key": None,  # 不再校验实际API密钥
    "huggingface_api_key": None,  # 不再校验实际API密钥
    "sentence_transformer_model": "test-model",
    "embedder_device": "cpu",
    "faiss_index_path": "data/faiss_index.bin",
    "faiss_mapping_path": "data/papers_faiss_ids.json",
    "models_faiss_index_path": "data/models_faiss.index",
    "models_faiss_mapping_path": "data/models_faiss_ids.json",
    "reload": True,
    "test_database_url": "postgresql+psycopg://test:pass@host/db",  # Loaded from TEST_DATABASE_URL env var
    "build_faiss_batch_size": 128,
    # Add other test-specific or default fields if needed
    "test_neo4j_uri": None,  # Example: test_neo4j_uri will be None if not set
    "test_neo4j_password": None,
    "test_neo4j_database": None,
    "test_faiss_paper_index_path": None,
    "test_faiss_paper_mapping_path": None,
    "test_faiss_model_index_path": None,
    "test_faiss_model_mapping_path": None,
}

# Expected values when relying on defaults (no relevant env vars set)
EXPECTED_SETTINGS_DEFAULTS = {
    "project_name": "AIGraphX",
    "api_v1_str": "/api/v1",
    "log_level": "INFO",
    "environment": "development",  # Matches .env (or default if not in .env)
    # Value from .env
    "database_url": "postgresql://aigraphx_user:aigraphx_password@postgres:5432/aigraphx",
    "pg_pool_min_size": 1,
    "pg_pool_max_size": 10,
    # Value from .env (no password in URI)
    "neo4j_uri": "neo4j://neo4j:7687",
    "neo4j_username": "neo4j",  # Matches .env
    # Value from .env - REMOVING ACTUAL PASSWORD
    "neo4j_password": None,  # 不再校验实际密码
    "neo4j_database": "neo4j",  # Matches .env (or default)
    # Value from .env - REMOVING ACTUAL API KEYS
    "pwc_api_key": None,  # 不再校验实际API密钥
    # Value from .env - REMOVING ACTUAL API KEYS
    "hf_api_key": None,  # 不再校验实际API密钥
    "github_api_key": None,  # 不再校验实际API密钥
    "sentence_transformer_model": "all-MiniLM-L6-v2",  # Matches .env (or default)
    # Value from .env
    "embedder_device": "cuda",
    "faiss_index_path": "data/faiss_index.bin",  # Default, not in .env
    "faiss_mapping_path": "data/papers_faiss_ids.json",  # Default, not in .env
    # Value from .env
    "models_faiss_index_path": "data/models_faiss.index",
    # Value from .env
    "models_faiss_mapping_path": "data/models_faiss_ids.json",
    # Value from .env (RELOAD=true)
    "reload": True,
    # Value from .env (with 127.0.0.1)
    "test_database_url": "postgresql://aigraphx_test_user:aigraphx_test_password@127.0.0.1:5433/aigraphx_test",
    "build_faiss_batch_size": 128,  # Default, not in .env
    "workers_count": os.cpu_count(),  # Default, not in .env
    # Test values from .env
    "test_neo4j_uri": "neo4j://127.0.0.1:7688",
    "test_neo4j_password": None,  # 不再校验实际密码
    "test_neo4j_database": "neo4j",
    "test_faiss_paper_index_path": None,  # Default, not in .env
    "test_faiss_paper_mapping_path": None,  # Default, not in .env
    "test_faiss_model_index_path": None,  # Default, not in .env
    "test_faiss_model_mapping_path": None,  # Default, not in .env
}

# --- Fixtures (No longer need mock_getenv) ---

# --- Helper for mocking os.getenv --- #
MOCKED_ENV: Dict[str, Optional[str]] = {}
ORIGINAL_OS_GETENV = os.getenv


def mock_getenv(var_name: str, default: Optional[str] = None) -> Optional[str]:
    return MOCKED_ENV.get(var_name, default)


@pytest.fixture(autouse=True)
def patch_os_getenv(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Fixture to automatically mock os.getenv for all tests in this module."""
    # Clear the mock dictionary before each test
    MOCKED_ENV.clear()
    monkeypatch.setattr(os, "getenv", mock_getenv)
    yield  # Run the test
    # Restore original os.getenv after test (handled by monkeypatch)


# --- Tests (Modified Strategy) --- #


def test_settings_loading_with_env_vars_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test loading Settings and overriding SOME values with environment variables."""
    # 1. Define ONLY the environment variables to OVERRIDE .env values
    #    Explicitly set the target values, don't rely on building from components.
    env_vars_to_override = {
        "ENVIRONMENT": "testing_override_env",
        # Directly set the final DATABASE_URL we expect
        "DATABASE_URL": "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db",
        "NEO4J_URI": "neo4j://override_host:1111",
        "NEO4J_PASSWORD": "override_neo4j_pass",  # Example override
        "RELOAD": "false",
        "TEST_DATABASE_URL": "postgresql://override_test:pass@host/db",  # Example override
    }
    for k, v in env_vars_to_override.items():
        monkeypatch.setenv(k, v)

    # 2. Instantiate Settings directly (allow .env loading)
    settings_instance = Settings()

    # 3. Define expected values: Start with EXPECTED_SETTINGS_DEFAULTS (based on .env + class defaults)
    #    Then, update ONLY the values that were overridden by monkeypatch.
    expected_env = EXPECTED_SETTINGS_DEFAULTS.copy()
    expected_env["environment"] = "testing_override_env"
    expected_env["database_url"] = (
        "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db"
    )
    expected_env["neo4j_uri"] = "neo4j://override_host:1111"
    expected_env["neo4j_password"] = "override_neo4j_pass"
    expected_env["reload"] = False
    expected_env["test_database_url"] = "postgresql://override_test:pass@host/db"

    # 4. Assert against the expected mixed state
    # Iterate over the *actual* attributes of the settings instance
    # Use model_dump() which is standard in Pydantic v2
    actual_settings_dict = settings_instance.model_dump()

    # --- Add Debug Prints ---
    print("\n--- Override Test --- ")
    print("Actual keys:", sorted(actual_settings_dict.keys()))
    print("Expected keys:", sorted(expected_env.keys()))
    # --- End Debug Prints ---

    # 创建一个已知可以忽略比较的密钥列表
    ignore_keys = [
        "pwc_api_key",
        "hf_api_key",
        "github_api_key",
        "neo4j_password",
        "huggingface_api_key",
        "test_neo4j_password",
    ]

    for key, actual_value in actual_settings_dict.items():
        # 跳过API密钥和密码的比较
        if key in ignore_keys:
            continue

        # Check if the actual key exists in our expected dictionary
        assert key in expected_env, (
            f"Unexpected attribute '{key}' found in Settings object (value: {actual_value})"
        )
        expected_value = expected_env[key]

        # Compare values using the existing logic
        if (
            key in ["database_url", "neo4j_uri", "test_database_url", "test_neo4j_uri"]
            and expected_value is not None
        ):
            actual_value_str = str(actual_value) if actual_value else None
            expected_value_str = str(expected_value)
            assert actual_value_str == expected_value_str, (
                f"Settings key '{key}' mismatch. Expected: '{expected_value_str}', Got: '{actual_value_str}'"
            )
        elif isinstance(expected_value, PostgresDsn):
            assert str(actual_value) == str(expected_value), (
                f"Settings key '{key}' mismatch. Expected: '{str(expected_value)}', Got: '{str(actual_value)}'"
            )
        elif expected_value is None:
            assert actual_value is None, (
                f"Settings key '{key}' expected None, Got: '{actual_value}'"
            )
        else:
            # Special case for workers_count which might differ slightly
            if key == "workers_count":
                assert isinstance(actual_value, int) and actual_value > 0
            else:
                assert actual_value == expected_value, (
                    f"Settings key '{key}' mismatch. Expected: '{expected_value}', Got: '{actual_value}'"
                )


def test_settings_loading_defaults_with_env_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test loading Settings using .env file and class defaults when no env vars are set."""
    # --- Add Manual .env Check ---
    try:
        print("\n--- Manually checking .env ---")
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        dotenv_path = os.path.join(project_root, ".env")
        print(f"Checking path: {dotenv_path}")
        if os.path.exists(dotenv_path):
            with open(dotenv_path, "r", encoding="utf-8") as f:  # Specify encoding
                content = f.read()
                print(".env content sample (first 100 chars):", content[:100])
                # Try to find the key manually
                if "HF_API_KEY=" in content:
                    print("HF_API_KEY found in manual read.")
                else:
                    print("HF_API_KEY *NOT* found in manual read.")
        else:
            print(".env file NOT FOUND at expected path.")
    except Exception as e:
        print(f"Error manually reading .env: {e}")
    # --- End Manual .env Check ---

    # 1. REMOVE all monkeypatch.delenv calls. Let .env load naturally.

    # 2. Instantiate Settings directly (allow .env loading)
    settings_instance = Settings()

    # 3. Assert against the expected defaults (verified against .env + class defaults)
    # Iterate over the *actual* attributes of the settings instance
    actual_settings_dict = settings_instance.model_dump()

    # --- Add Debug Prints ---
    print("\n--- Defaults Test --- ")
    print("Actual keys:", sorted(actual_settings_dict.keys()))
    print("Expected default keys:", sorted(EXPECTED_SETTINGS_DEFAULTS.keys()))
    # --- End Debug Prints ---

    # 创建一个已知可以忽略比较的密钥列表
    ignore_keys = [
        "pwc_api_key",
        "hf_api_key",
        "github_api_key",
        "neo4j_password",
        "huggingface_api_key",
        "test_neo4j_password",
    ]

    # Verify EXPECTED_SETTINGS_DEFAULTS reflects the actual loaded state
    for key, actual_value in actual_settings_dict.items():
        # 跳过API密钥和密码的比较
        if key in ignore_keys:
            continue

        assert key in EXPECTED_SETTINGS_DEFAULTS, (
            f"Unexpected attribute '{key}' found in Settings object (value: {actual_value})"
        )
        expected_value = EXPECTED_SETTINGS_DEFAULTS[key]

        # Compare values using the existing logic
        if (
            key in ["database_url", "neo4j_uri", "test_database_url", "test_neo4j_uri"]
            and expected_value is not None
        ):
            actual_value_str = str(actual_value) if actual_value else None
            expected_value_str = str(expected_value)
            assert actual_value_str == expected_value_str, (
                f"Settings key '{key}' mismatch. Expected default: '{expected_value_str}', Got: '{actual_value_str}'"
            )
        elif isinstance(expected_value, (PostgresDsn, HttpUrl)):
            if expected_value is None:
                assert actual_value is None, (
                    f"Settings key '{key}' mismatch. Expected default: None, Got: '{actual_value}'"
                )
            else:
                assert str(actual_value) == str(expected_value), (
                    f"Settings key '{key}' mismatch. Expected default: '{str(expected_value)}', Got: '{str(actual_value)}'"
                )
        elif expected_value is None:
            assert actual_value is None, (
                f"Settings key '{key}' mismatch. Expected default: None, Got: '{actual_value}'"
            )
        else:
            # Special case for workers_count which might differ slightly
            if key == "workers_count":
                assert isinstance(actual_value, int) and actual_value > 0
            else:
                assert actual_value == expected_value, (
                    f"Settings key '{key}' mismatch. Expected default: '{expected_value}', Got: '{actual_value}'"
                )

    # This test implicitly relies on .env being loaded for the base DATABASE_URL if not overridden
    def run_case(env_setup: Dict[str, Optional[str]]) -> Settings:
        # Clear only the vars relevant to this test using monkeypatch
        # Using monkeypatch here is still valid for controlling specific test case inputs
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

        # Set env vars for this specific case
        for k, v in env_setup.items():
            if v is not None:
                monkeypatch.setenv(k, v)
            else:
                monkeypatch.delenv(k, raising=False)

        # Override model_config *specifically for this test* to prevent .env loading
        # and ensure we are testing the env var logic in isolation.
        original_config = Settings.model_config
        test_config = SettingsConfigDict(
            env_file="/tmp/nonexistent_for_correction_test.env",  # Use a unique non-existent path
            extra="ignore",
        )
        monkeypatch.setattr(aigraphx.core.config.Settings, "model_config", test_config)

        # Instantiate directly, allow .env loading
        settings = Settings()
        # monkeypatch automatically restores original_config here
        return settings

    # Case 1: TEST_DATABASE_URL with standard scheme
    env_1: Dict[str, Optional[str]] = {
        "DATABASE_URL": "postgresql://original:pass@host/db",
        "TEST_DATABASE_URL": "postgresql://user:pass@host/db",
    }
    settings_instance_1 = run_case(env_1)
    assert settings_instance_1.test_database_url is not None
    # Compare strings for test_database_url
    assert (
        str(settings_instance_1.test_database_url) == "postgresql://user:pass@host/db"
    )
    # Assert against the DATABASE_URL expected from .env or default
    # Since we patched model_config in run_case to block .env, and DATABASE_URL env var is unset in run_case,
    # the value should be the class default, which is None.
    # expected_db_url = EXPECTED_SETTINGS_DEFAULTS["database_url"]
    # assert settings_instance_1.database_url is not None
    # assert str(settings_instance_1.database_url) == str(expected_db_url)
    assert str(settings_instance_1.database_url) == "postgresql://original:pass@host/db"

    # Case 2: TEST_DATABASE_URL with +psycopg scheme
    env_2: Dict[str, Optional[str]] = {
        "DATABASE_URL": "postgresql://original2:pass@host/db",
        "TEST_DATABASE_URL": "postgresql+psycopg://user:pass@host/db",
    }
    settings_instance_2 = run_case(env_2)
    assert settings_instance_2.test_database_url is not None
    assert (
        str(settings_instance_2.test_database_url)
        == "postgresql+psycopg://user:pass@host/db"
    )
    assert (
        str(settings_instance_2.database_url) == "postgresql://original2:pass@host/db"
    )

    # Case 3: TEST_DATABASE_URL Variable not set
    env_3 = {
        "DATABASE_URL": "postgresql://original3:pass@host/db",
        "TEST_DATABASE_URL": None,
    }
    settings_instance_3 = run_case(env_3)
    assert settings_instance_3.test_database_url is None
    assert settings_instance_3.database_url is not None

    # Assert against the DATABASE_URL expected from .env or default
    # Ensure EXPECTED_SETTINGS_DEFAULTS["database_url"] is accurate!


#   expected_db_url = EXPECTED_SETTINGS_DEFAULTS["database_url"]
#   assert settings_instance_1.database_url is not None
#   assert str(settings_instance_1.database_url) == str(expected_db_url)

# Test dotenv loading (optional, might be tricky)
# You could mock load_dotenv itself or check logger output
# For simplicity, focusing on os.getenv logic for now.
