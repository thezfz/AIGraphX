# tests/test_neo4j_teardown.py
import pytest
import pytest_asyncio
import asyncio
import os
import logging
from neo4j import AsyncGraphDatabase, AsyncDriver, basic_auth
from typing import AsyncGenerator, Optional

# Basic logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Hardcode connection details (replace with your actual test env vars if different)
# These should match what's used in test_settings fixture
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI", "bolt://localhost:7688")
TEST_NEO4J_USER = os.getenv("TEST_NEO4J_USER", "neo4j")
TEST_NEO4J_PASSWORD = os.getenv(
    "TEST_NEO4J_PASSWORD", "testpassword"
)  # Replace with your actual test password if needed
TEST_NEO4J_DATABASE = os.getenv(
    "TEST_NEO4J_DATABASE", "neo4jtest"
)  # Replace with your actual test db name if needed

# Mark all tests in this file to use the session-scoped loop
# pytestmark = pytest.mark.asyncio(loop_scope="session") # REMOVED: Revert to default (function scope)


@pytest_asyncio.fixture(scope="function")
async def minimal_neo4j_driver() -> AsyncGenerator[AsyncDriver, None]:
    """Simplified Neo4j driver fixture for teardown testing."""
    logger.info("[minimal_neo4j_driver] Setting up...")
    driver: Optional[AsyncDriver] = None
    # Validate mandatory config
    if not all([TEST_NEO4J_URI, TEST_NEO4J_USER, TEST_NEO4J_PASSWORD]):
        logger.error(
            "Missing required Neo4j test connection environment variables: URI, USER, PASSWORD"
        )
        pytest.fail("Missing required Neo4j test connection environment variables.")

    # Database name can be optional for some operations, but required for session verification here
    if not TEST_NEO4J_DATABASE:
        logger.error(
            "Missing required Neo4j test connection environment variable: DATABASE"
        )
        pytest.fail("Missing required Neo4j test DATABASE environment variable.")

    try:
        # Use basic_auth for type safety
        auth = basic_auth(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD)
        driver = AsyncGraphDatabase.driver(TEST_NEO4J_URI, auth=auth)

        # Verify connection to the specific target database
        logger.info(
            f"[minimal_neo4j_driver] Verifying connection to DB '{TEST_NEO4J_DATABASE}'..."
        )
        async with driver.session(database=TEST_NEO4J_DATABASE) as session:
            await session.run("RETURN 1")
        logger.info(
            f"[minimal_neo4j_driver] Connection verified for DB '{TEST_NEO4J_DATABASE}'. Yielding driver."
        )
        yield driver
    except Exception as e:
        logger.error(
            f"[minimal_neo4j_driver] Failed to connect or verify Neo4j: {e}",
            exc_info=True,
        )
        if driver:
            try:
                # Try closing even if setup failed partially
                # Still using await here as we reverted the run_until_complete
                await driver.close()
            except Exception as close_err:
                logger.error(
                    f"[minimal_neo4j_driver] Error closing driver during setup failure cleanup: {close_err}",
                    exc_info=True,
                )
        pytest.fail(f"Failed to connect to Neo4j during setup: {e}")
    finally:
        if driver:
            logger.info("[minimal_neo4j_driver] Tearing down - closing Neo4j driver...")
            try:
                # --- REVERTED: Back to simple await ---
                await driver.close()
                # --------------------------------------
                logger.info("[minimal_neo4j_driver] Neo4j driver closed successfully.")
            except RuntimeError as e:
                # Log potential runtime errors during close
                logger.error(
                    f"[minimal_neo4j_driver] TEARDOWN Runtime error closing Neo4j driver: {e}",
                    exc_info=True,
                )
                # Do not fail test here
            except Exception as e:
                logger.error(
                    f"[minimal_neo4j_driver] Generic error closing Neo4j driver during teardown: {e}",
                    exc_info=True,
                )
                # Optionally re-raise to make test failure visible
                # raise


@pytest.mark.asyncio  # Explicitly mark the test
async def test_simple_neo4j_query(minimal_neo4j_driver: AsyncDriver) -> None:
    """A simple test that uses the Neo4j driver."""
    logger.info("[test_simple_neo4j_query] Running test...")
    async with minimal_neo4j_driver.session(database=TEST_NEO4J_DATABASE) as session:
        result = await session.run("RETURN datetime() as now")
        record = await result.single()
        assert record is not None
        assert "now" in record.keys()
        logger.info(
            f"[test_simple_neo4j_query] Query successful, got time: {record['now']}"
        )
    logger.info("[test_simple_neo4j_query] Test finished.")
