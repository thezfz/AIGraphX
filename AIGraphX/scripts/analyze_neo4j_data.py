import asyncio
import os
from neo4j import AsyncGraphDatabase, basic_auth, AsyncDriver
from dotenv import load_dotenv
import logging
import traceback # Added for full traceback logging
from typing import List, Dict, Any, Coroutine

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the Backend directory
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env") 
if not load_dotenv(dotenv_path=dotenv_path):
    logger.warning(f"Could not load .env file from {dotenv_path}. Trying current directory or expecting environment variables.")
    if not load_dotenv(): # Try loading from current dir as a fallback
        logger.info("No .env file found in current directory either. Relying on pre-set environment variables.")


NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687") 
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

async def get_node_counts(driver: AsyncDriver) -> List[Dict[str, Any]]:
    """Counts nodes by label."""
    query = """
    MATCH (n)
    RETURN labels(n) AS node_labels, count(*) AS count
    ORDER BY count DESC
    """
    results: List[Dict[str, Any]] = []
    async with driver.session() as session:
        result_summary = await session.run(query)
        async for record in result_summary:
            label_str = ":".join(sorted(record["node_labels"])) 
            results.append({"label_set": label_str, "count": record["count"]})
    return results

async def get_relationship_counts(driver: AsyncDriver) -> List[Dict[str, Any]]:
    """Counts relationships by type."""
    query = """
    MATCH ()-[r]->()
    RETURN type(r) AS relationship_type, count(*) AS count
    ORDER BY count DESC
    """
    results: List[Dict[str, Any]] = []
    async with driver.session() as session:
        result_summary = await session.run(query)
        async for record in result_summary:
            results.append({"type": record["relationship_type"], "count": record["count"]})
    return results

async def get_degree_summary(driver: AsyncDriver, top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Gets a summary of node degrees (overall, in, out) for top N nodes.
       Tries to get a common identifier like 'name', 'modelId', 'pwc_id', or 'url'.
    """
    
    # Common identifier logic - COALESCE will pick the first non-null one
    identifier_expression = "COALESCE(n.name, n.modelId, n.pwc_id, n.arxiv_id_base, n.url, toString(elementId(n)))"

    query_overall = f"""
    MATCH (n)
    WITH n, COUNT {{ (n)--() }} AS degree
    WHERE degree > 0 // Only consider nodes with connections
    RETURN {identifier_expression} AS id, labels(n)[0] as label, degree
    ORDER BY degree DESC
    LIMIT {top_n}
    """
    query_in = f"""
    MATCH (n)
    WITH n, COUNT {{ (n)<--() }} AS in_degree
    WHERE in_degree > 0
    RETURN {identifier_expression} AS id, labels(n)[0] as label, in_degree
    ORDER BY in_degree DESC
    LIMIT {top_n}
    """
    query_out = f"""
    MATCH (n)
    WITH n, COUNT {{ (n)-->() }} AS out_degree
    WHERE out_degree > 0
    RETURN {identifier_expression} AS id, labels(n)[0] as label, out_degree
    ORDER BY out_degree DESC
    LIMIT {top_n}
    """
    degrees: Dict[str, List[Dict[str, Any]]] = {"overall": [], "in": [], "out": []}
    async with driver.session() as session:
        logger.info(f"Fetching top {top_n} nodes by overall degree...")
        res_overall = await session.run(query_overall)
        async for rec in res_overall:
            degrees["overall"].append({"id": rec["id"], "label": rec["label"], "degree": rec["degree"]})

        logger.info(f"Fetching top {top_n} nodes by in-degree...")
        res_in = await session.run(query_in)
        async for rec in res_in:
            degrees["in"].append({"id": rec["id"], "label": rec["label"], "degree": rec["in_degree"]})
        
        logger.info(f"Fetching top {top_n} nodes by out-degree...")
        res_out = await session.run(query_out)
        async for rec in res_out:
            degrees["out"].append({"id": rec["id"], "label": rec["label"], "degree": rec["out_degree"]})
            
    return degrees


async def main() -> None:
    logger.info(f"Attempting to connect to Neo4j URI: {NEO4J_URI} as user: {NEO4J_USER}")
    if not NEO4J_PASSWORD:
        logger.error("NEO4J_PASSWORD is not set. Please check your .env file or environment variables.")
        return

    driver: AsyncDriver = None  # Initialize driver to None for the finally block
    try:
        auth_details = basic_auth(NEO4J_USER, NEO4J_PASSWORD)
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=auth_details)
        await driver.verify_connectivity() # Verify connection
        logger.info("Successfully connected to Neo4j.")

        logger.info("--- Node Counts by Label Set ---")
        node_counts = await get_node_counts(driver)
        if node_counts:
            for item in node_counts:
                logger.info(f"  {item['label_set']}: {item['count']}")
        else:
            logger.info("  No nodes found or error fetching node counts.")

        logger.info("\n--- Relationship Counts by Type ---")
        rel_counts = await get_relationship_counts(driver)
        if rel_counts:
            for item in rel_counts:
                logger.info(f"  {item['type']}: {item['count']}")
        else:
            logger.info("  No relationships found or error fetching relationship counts.")

        logger.info("\n--- Node Degree Summary (Top 10) ---")
        degree_summary = await get_degree_summary(driver, top_n=10)
        
        logger.info("  Top 10 Overall Degree:")
        if degree_summary["overall"]:
            for node in degree_summary["overall"]:
                logger.info(f"    Node ID: {node.get('id', 'N/A')}, Label: {node.get('label', 'N/A')}, Degree: {node['degree']}")
        else:
            logger.info("    Could not fetch overall degree summary or no nodes with degree > 0.")
            
        logger.info("  Top 10 In-Degree:")
        if degree_summary["in"]:
            for node in degree_summary["in"]:
                 logger.info(f"    Node ID: {node.get('id', 'N/A')}, Label: {node.get('label', 'N/A')}, Degree: {node['degree']}")
        else:
            logger.info("    Could not fetch in-degree summary or no nodes with in-degree > 0.")

        logger.info("  Top 10 Out-Degree:")
        if degree_summary["out"]:
            for node in degree_summary["out"]:
                 logger.info(f"    Node ID: {node.get('id', 'N/A')}, Label: {node.get('label', 'N/A')}, Degree: {node['degree']}")
        else:
            logger.info("    Could not fetch out-degree summary or no nodes with out-degree > 0.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc()) # Log the full traceback
    finally:
        if driver:
            await driver.close()
            logger.info("Neo4j connection closed.")

if __name__ == "__main__":
    asyncio.run(main()) 