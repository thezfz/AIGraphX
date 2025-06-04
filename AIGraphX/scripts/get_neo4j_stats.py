import asyncio
import os
import logging
from neo4j import AsyncGraphDatabase, basic_auth, AsyncDriver
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Assumes .env file is in the parent directory of Backend or a common root
# Adjust the path if your .env file is located elsewhere relative to this script
# For example, if AIGraphX is the root and .env is there:
# dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
# If running scripts from AIGraphX/scripts and .env is in AIGraphX:
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env") 
load_dotenv(dotenv_path=dotenv_path)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


async def get_node_counts(driver: AsyncDriver) -> Dict[str, int]:
    """
    Fetches all node labels and their counts.
    """
    node_counts: Dict[str, int] = {}
    async with driver.session(database=NEO4J_DATABASE) as session:
        try:
            # Get all node labels
            result = await session.run("CALL db.labels() YIELD label")
            labels: List[str] = [record["label"] async for record in result]
            logger.info(f"Found node labels: {labels}")

            for label in labels:
                query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
                count_result = await session.run(query)
                record = await count_result.single()
                if record and record["count"] is not None:
                    node_counts[label] = record["count"]
                else:
                    node_counts[label] = 0
                logger.info(f"Count for label '{label}': {node_counts[label]}")
        except Exception as e:
            logger.error(f"Error getting node counts: {e}")
            raise
    return node_counts


async def get_relationship_counts(driver: AsyncDriver) -> Dict[str, int]:
    """
    Fetches all relationship types and their counts.
    """
    relationship_counts: Dict[str, int] = {}
    async with driver.session(database=NEO4J_DATABASE) as session:
        try:
            # Get all relationship types
            result = await session.run(
                "CALL db.relationshipTypes() YIELD relationshipType"
            )
            rel_types: List[str] = [record["relationshipType"] async for record in result]
            logger.info(f"Found relationship types: {rel_types}")

            for rel_type in rel_types:
                # Note: Using backticks for relationship types might be needed if they contain special characters,
                # but usually not required for standard types.
                query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS count"
                count_result = await session.run(query)
                record = await count_result.single()
                if record and record["count"] is not None:
                    relationship_counts[rel_type] = record["count"]
                else:
                    relationship_counts[rel_type] = 0
                logger.info(
                    f"Count for relationship type '{rel_type}': {relationship_counts[rel_type]}"
                )
        except Exception as e:
            logger.error(f"Error getting relationship counts: {e}")
            raise
    return relationship_counts


async def main() -> None:
    """
    Main function to connect to Neo4j and get statistics.
    """
    if not NEO4J_PASSWORD:
        logger.error(
            "NEO4J_PASSWORD environment variable not set. Please set it in your .env file."
        )
        return

    auth = basic_auth(NEO4J_USER, NEO4J_PASSWORD)
    driver: Optional[AsyncDriver] = None
    try:
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=auth)
        await driver.verify_connectivity()
        logger.info(f"Successfully connected to Neo4j at {NEO4J_URI}")

        logger.info("\\n--- Node Counts ---")
        node_counts = await get_node_counts(driver)
        if node_counts:
            for label, count in node_counts.items():
                print(f"Nodes with label '{label}': {count}")
            print(f"Total distinct node labels: {len(node_counts)}")
        else:
            print("No node labels found or error in fetching counts.")

        logger.info("\\n--- Relationship Counts ---")
        relationship_counts = await get_relationship_counts(driver)
        if relationship_counts:
            for rel_type, count in relationship_counts.items():
                print(f"Relationships of type '{rel_type}': {count}")
            print(f"Total distinct relationship types: {len(relationship_counts)}")
        else:
            print("No relationship types found or error in fetching counts.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if driver:
            await driver.close()
            logger.info("Neo4j connection closed.")


if __name__ == "__main__":
    asyncio.run(main()) 