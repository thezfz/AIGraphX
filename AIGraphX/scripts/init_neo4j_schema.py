import asyncio
import logging
import os

# Removed dotenv import and call
# from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

# Adjust import path to ensure aigraphx modules can be found
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from aigraphx.repositories.neo4j_repo import Neo4jRepository
from aigraphx.core.config import settings # Using centralized settings

# Load environment variables from .env file if present
# settings object (Pydantic BaseSettings) should handle this automatically.
# load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Initializes the Neo4j schema by creating constraints and indexes."""
    logger.info("Starting Neo4j schema initialization...")

    if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
        logger.error("Neo4j connection details (URI, USER, PASSWORD) not found in settings. Exiting.")
        return

    neo4j_driver = None
    try:
        auth = (settings.neo4j_user, settings.neo4j_password)
        async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as neo4j_driver:
            logger.info(f"Connecting to Neo4j at {settings.neo4j_uri} with user {settings.neo4j_user}")
            
            # Check connectivity
            await neo4j_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")

            neo4j_repo = Neo4jRepository(driver=neo4j_driver, db_name=settings.neo4j_database)
            
            logger.info(f"Target Neo4j database: {settings.neo4j_database}")
            logger.info("Creating constraints and indexes...")
            await neo4j_repo.create_constraints_and_indexes()
            logger.info("Successfully created constraints and indexes.")

    except Exception as e:
        logger.error(f"An error occurred during Neo4j schema initialization: {e}", exc_info=True)
    finally:
        if neo4j_driver:
            await neo4j_driver.close()
            logger.info("Neo4j driver closed.")
        logger.info("Neo4j schema initialization process finished.")

if __name__ == "__main__":
    asyncio.run(main()) 