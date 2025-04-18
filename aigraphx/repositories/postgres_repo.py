from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Literal, cast
import traceback
import logging
from psycopg_pool import AsyncConnectionPool  # Import pool
from psycopg.rows import dict_row  # Import dict_row
import json
from datetime import date  # Add date
import asyncpg  # type: ignore[import-untyped]
from asyncpg import Record  # Add this import
from psycopg import sql  # Use psycopg3 for SQL composition
import numpy as np  # Import numpy
from aigraphx.models.paper import Paper  # Add import for Paper model
import psycopg  # Add missing import for psycopg.Error
from typing import get_args

logger = logging.getLogger(__name__)


class PostgresRepository:
    def __init__(self, pool: AsyncConnectionPool):  # Accept pool instead of URL
        """Initializes the repository with an async connection pool."""
        self.pool = pool
        self.logger = logger  # Use the module logger

    # Removed initialize, get_connection, release_connection
    # All methods will now use self.pool directly

    async def get_paper_details_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Fetches details for a single paper by its integer paper_id."""
        query = "SELECT * FROM papers WHERE paper_id = %s;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (paper_id,))
                    result = await cur.fetchone()
                    # Convert JSONB fields back if needed (psycopg usually handles this)
                    # if result and isinstance(result.get('authors'), str): result['authors'] = json.loads(result['authors'])
                    # if result and isinstance(result.get('categories'), str): result['categories'] = json.loads(result['categories'])
                    return result
        except Exception as e:
            self.logger.error(f"Error fetching paper with id {paper_id} from PG: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def get_papers_details_by_ids(
        self, paper_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Fetches details for multiple papers by their integer paper_ids."""
        if not paper_ids:
            return []
        query = "SELECT * FROM papers WHERE paper_id = ANY(%s);"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    # Pass the list directly to execute
                    await cur.execute(query, (paper_ids,))
                    results = await cur.fetchall()
                    # Convert JSONB fields back if needed (optional, depends on usage)
                    # for row in results:
                    #     if isinstance(row.get('authors'), str): row['authors'] = json.loads(row['authors'])
                    #     if isinstance(row.get('categories'), str): row['categories'] = json.loads(row['categories'])
                    return results
        except Exception as e:
            self.logger.error(f"Error fetching paper details by IDs from PG: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def get_all_papers_for_sync(self) -> List[Dict[str, Any]]:
        """Fetches all papers, primarily for tasks like building search indexes."""
        query = "SELECT paper_id, summary FROM papers WHERE summary IS NOT NULL AND summary != '';"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query)
                    return await cur.fetchall()
        except Exception as e:
            self.logger.error(f"Error fetching all papers for sync: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def count_papers(self) -> int:
        """Counts the total number of papers in the papers table."""
        query = "SELECT COUNT(*) FROM papers;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    result = await cur.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error counting papers in PG: {e}")
            self.logger.debug(traceback.format_exc())
            return 0

    async def search_papers_by_keyword(
        self,
        query: str,
        skip: int = 0,
        limit: int = 10,
        published_after: Optional[date] = None,
        published_before: Optional[date] = None,
        filter_area: Optional[str] = None,
        # Use Literal for specific valid column names
        sort_by: Optional[
            Literal["published_date", "title", "paper_id"]
        ] = "published_date",
        sort_order: Optional[Literal["asc", "desc"]] = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:  # UPDATED return type
        """Searches papers by keyword, returns paginated paper details (dict) and total count."""

        params: Dict[str, Any] = {"like_query": f"%{query}%"}
        where_clauses = ["(title ILIKE %(like_query)s OR summary ILIKE %(like_query)s)"]

        if published_after:
            where_clauses.append(f"published_date >= %(published_after)s")
            params["published_after"] = published_after
        if published_before:
            where_clauses.append(f"published_date <= %(published_before)s")
            params["published_before"] = published_before
        if filter_area:
            where_clauses.append(f"area = %(filter_area)s")
            params["filter_area"] = filter_area

        where_sql = " AND ".join(where_clauses)

        # Validate sort_by against the Literal type and ensure order_by_column is str
        order_by_column: str  # Declare type explicitly
        valid_sort_columns = get_args(Literal["published_date", "title", "paper_id"])

        if sort_by in valid_sort_columns:
            # Use cast to assure MyPy that sort_by is a valid string here
            order_by_column = cast(str, sort_by)
        else:
            # Default if sort_by is None or invalid
            if sort_by is not None:
                logger.warning(
                    f"Invalid sort_by value '{sort_by}' for papers keyword search. Defaulting to 'published_date'."
                )
            order_by_column = "published_date"

        order_direction = "DESC" if sort_order == "desc" else "ASC"
        order_by_sql = f'ORDER BY "{order_by_column}" {order_direction} NULLS LAST, paper_id {order_direction}'

        # Count query
        count_sql = f"SELECT COUNT(*) FROM papers WHERE {where_sql};"

        # Main selection query for paper details
        # Select all columns needed for SearchResultItem
        select_fields = (
            "paper_id, pwc_id, title, summary, pdf_url, published_date, authors, area"
        )
        params["offset"] = skip
        params["limit"] = limit
        select_sql = f"""
            SELECT {select_fields}
            FROM papers
            WHERE {where_sql}
            {order_by_sql}
            OFFSET %(offset)s
            LIMIT %(limit)s;
            """

        total_count = 0
        results: List[Dict[str, Any]] = []  # Initialize as list of dicts

        try:
            async with self.pool.connection() as conn:
                # Use dict_row cursor for both count and select
                async with conn.cursor(row_factory=dict_row) as cur:
                    # Execute count query first
                    await cur.execute(count_sql, params)
                    count_result = await cur.fetchone()
                    # Access count using key if dict_row is used
                    total_count = count_result["count"] if count_result else 0

                    # Execute main select query only if needed
                    if total_count > skip:
                        await cur.execute(select_sql, params)
                        results = await cur.fetchall()
                        # JSON decoding might happen automatically with dict_row/psycopg3
                        # but double-check if 'authors' remains string
                        # for row in results:
                        #    if isinstance(row.get('authors'), str):
                        #        try:
                        #            row['authors'] = json.loads(row['authors'])
                        #        except json.JSONDecodeError:
                        #            logger.warning(f"Could not decode authors JSON for paper_id {row.get('paper_id')}")
                        #            row['authors'] = None # Or []

            return results, total_count

        except Exception as e:
            logger.exception(
                f"Error searching papers by keyword '{query}' with filters/sort: {e}"
            )
            return [], 0

    async def get_hf_models_by_ids(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches details for multiple HF models by their model_ids."""
        if not model_ids:
            return []
        # Query needed columns from hf_models table
        query = "SELECT hf_model_id, author, hf_pipeline_tag FROM hf_models WHERE hf_model_id = ANY(%s);"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (model_ids,))
                    results = await cur.fetchall()
                    return results
        except Exception as e:
            self.logger.error(f"Error fetching HF model details by IDs from PG: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def search_models_by_keyword(
        self, query: str, limit: int = 10, skip: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Searches hf_models table by keyword across relevant fields."""
        search_term = f"%{query}%"
        # Use COALESCE for potentially null fields like pipeline_tag
        # Ensure correct column names are used
        where_clause = """
            model_id ILIKE %s OR 
            author ILIKE %s OR 
            COALESCE(pipeline_tag, '') ILIKE %s
        """
        params = [search_term] * 3  # Repeat search term for each ILIKE

        # Fields to select (adjust as needed)
        select_fields = "model_id, author, pipeline_tag, last_modified, tags, likes, downloads, library_name, sha"

        count_sql = f"""
            SELECT COUNT(*) 
            FROM hf_models 
            WHERE ({where_clause})
        """

        select_sql = f"""
            SELECT {select_fields} 
            FROM hf_models 
            WHERE ({where_clause}) 
            ORDER BY last_modified DESC 
            LIMIT %s OFFSET %s
        """

        # Add limit and offset to params
        query_params_paginated = params + [limit, skip]

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    # Get total count first
                    logger.debug(
                        f"Executing count SQL: {count_sql} with params: {params}"
                    )
                    await cur.execute(count_sql, params)
                    total_count_result = await cur.fetchone()
                    total_count = (
                        total_count_result["count"] if total_count_result else 0
                    )

                    if total_count == 0:
                        return [], 0

                    # Get paginated results
                    logger.debug(
                        f"Executing select SQL: {select_sql} with params: {query_params_paginated}"
                    )
                    await cur.execute(select_sql, query_params_paginated)
                    results = await cur.fetchall()
                    return results, total_count
        except Exception as e:
            logger.error(f"Error searching hf_models by keyword '{query}': {e}")
            # Consider logging the SQL and params here too if needed
            return [], 0

    async def count_hf_models(self) -> int:
        query = "SELECT COUNT(*) FROM hf_models;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    result = await cur.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error counting HF models in PG: {e}")
            self.logger.debug(traceback.format_exc())
            return 0

    async def get_all_hf_models_for_sync(
        self, batch_size: int = 1000
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Fetches all HF models in batches, yielding each batch as a list of dicts."""
        query = "SELECT * FROM hf_models;"  # Select all fields for sync
        offset = 0
        try:
            while True:
                batch_query = f"{query} ORDER BY hf_model_id LIMIT %s OFFSET %s;"  # Order for deterministic batches
                async with self.pool.connection() as conn:
                    async with conn.cursor(row_factory=dict_row) as cur:
                        await cur.execute(batch_query, (batch_size, offset))
                        batch = await cur.fetchall()
                        if not batch:
                            break
                        yield batch
                        offset += len(batch)
        except Exception as e:
            self.logger.error(f"Error fetching all hf_models for sync: {e}")
            self.logger.debug(traceback.format_exc())
            # Stop generation on error

    async def save_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Saves a single paper's data, handling potential JSON fields and conflicts."""
        # Prepare columns and values, handle JSON serialization
        # Note: psycopg handles most Python types including lists/dicts for JSONB
        cols = []
        vals = []
        excluded_updates = []
        for key, value in paper_data.items():
            # Basic validation/transformation (adjust as needed)
            if key == "authors" and not isinstance(value, list):
                value = []
            if key == "categories" and not isinstance(value, list):
                value = []
            # if isinstance(value, (dict, list)): # Psycopg handles this
            #     value = json.dumps(value)

            cols.append(key)
            vals.append(value)
            # For ON CONFLICT, exclude the primary key (pwc_id) from update set
            if key != "pwc_id":
                # Use f-string safely as key comes from dict keys
                excluded_updates.append(f"{key} = EXCLUDED.{key}")

        if not cols or "pwc_id" not in cols:
            self.logger.error("Cannot save paper: missing data or pwc_id.")
            return False

        # Construct query (Ensure table and column names are correct)
        # Using pwc_id for conflict resolution
        query = f"""
            INSERT INTO papers ({", ".join(cols)})
            VALUES ({", ".join(["%s"] * len(vals))})
            ON CONFLICT (pwc_id) DO UPDATE SET
                {", ".join(excluded_updates)},
                updated_at = CURRENT_TIMESTAMP;
        """
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, vals)
                    return cur.rowcount > 0  # Check if a row was inserted/updated
        except Exception as e:
            self.logger.error(
                f"Error saving paper with pwc_id {paper_data.get('pwc_id')} to PG: {e}"
            )
            # Consider logging paper_data partially for debugging
            self.logger.debug(traceback.format_exc())
            return False

    async def save_hf_models_batch(self, models_data: List[Dict[str, Any]]) -> None:
        """Saves a batch of HF models, updating existing ones based on model_id."""
        if not models_data:
            return

        # Prepare columns based on the first model (assume consistency)
        # Handle potential JSON fields if needed (e.g., tags)
        if not models_data[0]:
            return  # Handle empty dict case

        cols = list(models_data[0].keys())
        # Ensure model_id is present for ON CONFLICT
        if "model_id" not in cols:
            self.logger.error(
                "Cannot save HF models batch: 'model_id' missing in data."
            )
            return

        excluded_updates = []
        for key in cols:
            if key != "model_id":
                # Use f-string safely as key comes from dict keys
                excluded_updates.append(f"{key} = EXCLUDED.{key}")

        # Construct query (Ensure table and column names are correct)
        query = f"""
            INSERT INTO hf_models ({", ".join(cols)})
            VALUES %s
            ON CONFLICT (model_id) DO UPDATE SET
            {", ".join(excluded_updates)};
        """

        # Prepare data tuples, ensuring list fields are JSON strings
        data_tuples = []
        for model in models_data:
            row = []
            for col in cols:
                value = model.get(col)
                # FIX: Serialize list fields (like 'tags') to JSON strings
                if col == "tags" and isinstance(value, list):
                    row.append(json.dumps(value))
                else:
                    row.append(value)  # type: ignore[arg-type] # Allow None, db driver handles it
            data_tuples.append(tuple(row))

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(query, data_tuples)
                    self.logger.info(
                        f"Successfully saved/updated batch of {cur.rowcount} HF models."
                    )
        except Exception as e:
            self.logger.error(f"Error saving hf_models batch to PG: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def get_paper_details_by_pwc_id(
        self, pwc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetches details for a single paper by its PWC ID."""
        query = "SELECT * FROM papers WHERE pwc_id = %s;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (pwc_id,))
                    return await cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error fetching paper by pwc_id {pwc_id}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    # --- NEW METHOD ---
    async def fetch_data_cursor(
        self, query: str, params: Optional[tuple] = None, batch_size: int = 100
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fetches data using a server-side cursor asynchronously.

        Args:
            query (str): The SQL query to execute.
            params (Optional[tuple]): Parameters for the query.
            batch_size (int): Number of rows to fetch per batch.

        Yields:
            Dict[str, Any]: A dictionary representing a row from the result set.
        """
        try:
            async with self.pool.connection() as conn:
                # Use dict_row factory for easy access by column name
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params)
                    while True:
                        rows = await cur.fetchmany(batch_size)
                        if not rows:
                            break
                        for row in rows:
                            # Convert row (which might be a psycopg Row object) to dict
                            yield dict(row)
        except psycopg.Error as e:
            self.logger.error(
                f"Database error during fetch_data_cursor: {e}", exc_info=True
            )
            # Propagate the exception or handle it as needed
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error during fetch_data_cursor: {e}", exc_info=True
            )
            raise

    # --- New Method ---
    async def get_all_paper_ids_and_text(
        self, batch_size: int = 100
    ) -> AsyncGenerator[Tuple[int, str], None]:
        """Asynchronously generates tuples of (paper_id, text_content) for all papers.

        Uses summary if available, otherwise falls back to title.
        Filters out papers where both summary and title are NULL.

        Args:
            batch_size (int): How many rows to fetch from the database at a time.

        Yields:
            Tuple[int, str]: A tuple containing the paper_id and its text content.
        """
        query = """
            SELECT
                paper_id,
                COALESCE(summary, title) AS text_content
            FROM papers
            WHERE summary IS NOT NULL OR title IS NOT NULL
            ORDER BY paper_id; -- Optional: for deterministic order if needed
        """
        try:
            async for row in self.fetch_data_cursor(query, batch_size=batch_size):
                if "paper_id" in row and "text_content" in row and row["text_content"]:
                    yield (row["paper_id"], row["text_content"])
                else:
                    self.logger.warning(
                        f"Skipping row due to missing 'paper_id' or 'text_content': {row}"
                    )
        except Exception as e:
            self.logger.error(f"Error fetching paper IDs and text: {e}", exc_info=True)
            # Depending on requirements, you might want to raise the exception
            # or just log it and stop yielding.

    async def close(self) -> None:
        """Closes the connection pool (if applicable and managed by this instance)."""

    # --- NEW Method: fetch_one --- #
    async def fetch_one(
        self, query: str, params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Executes a query expected to return at most one row, returning it as a dict or None."""
        try:
            # Corrected: Use connection() for psycopg_pool
            async with self.pool.connection() as conn:
                # psycopg uses cursor() and fetchone()
                async with conn.cursor(
                    row_factory=dict_row
                ) as cur:  # Ensure dict_row for consistency
                    await cur.execute(query, params if params else ())
                    record = await cur.fetchone()  # Use fetchone()
                    return record if record else None  # dict_row already returns dict
        except Exception as e:
            logger.error(
                f"Error executing fetch_one query: {e}\nQuery: {query}\nParams: {params}"
            )
            logger.debug(traceback.format_exc())
            return None

    # --- End NEW Method --- #

    async def get_all_models_for_indexing(
        self,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Fetches all Hugging Face model IDs and their text representation (e.g., description or concatenated fields)
        from the database for indexing purposes. Yields batches.
        TODO: Define what constitutes the 'text' for a model (description? tags? combination?).
        For now, returning hf_model_id and description if available, else empty string.
        """
        # Corrected column name from model_id to hf_model_id
        # Corrected column name from tags to hf_tags
        query = """
            SELECT
                hf_model_id,
                COALESCE(hf_tags, '[]') || COALESCE(hf_pipeline_tag, '') AS text_representation
            FROM hf_models;
            """
        # Note: Using hf_tags + hf_pipeline_tag as a placeholder for text representation.
        # This might need refinement based on what should be indexed.

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    while True:
                        # Use fetchmany instead of fetch
                        rows = await cur.fetchmany(100)  # Fetch in batches
                        if not rows:
                            break
                        for row in rows:
                            # Yield hf_model_id and the constructed text
                            yield (
                                row[0],
                                row[1],
                            )  # Assuming row[0] is hf_model_id, row[1] is text
        except Exception as e:
            self.logger.error(f"Error fetching all models for indexing: {e}")
            self.logger.debug(traceback.format_exc())
            # Ensure the generator stops cleanly on error
            return

    async def upsert_paper(self, paper: Paper) -> Optional[int]:
        """
        Inserts or updates a paper record in the database based on pwc_id.

        Args:
            paper (Paper): The Paper object containing data to upsert.

        Returns:
            Optional[int]: The primary key (paper_id) of the upserted record, or None if failed.
        """
        # List of columns in the 'papers' table corresponding to Paper model fields
        # **Crucially, ensure these match the actual DB schema**
        columns = [
            "pwc_id",  # Use pwc_id as the likely unique identifier
            "title",
            "arxiv_id_base",  # Use the correct fields from Paper model
            "arxiv_id_versioned",
            "summary",
            "authors",  # Needs JSON encoding
            "categories",  # Needs JSON encoding
            "published_date",
            # "updated_date", # Assuming DB handles this automatically (e.g., trigger) or not needed from model
            # "source", # Assuming not in Paper model, remove or add if needed
            "doi",
            "pdf_url",  # Needs string conversion
            "pwc_url",  # Needs string conversion
            "pwc_title",  # Added from Paper model
            # "github_url", # Assuming not in Paper model
            "primary_category",
            # "journal_ref", # Assuming not in Paper model
            # "comment", # Assuming not in Paper model
            # "url_abs", # Assuming not in Paper model
            # "url_pdf", # Redundant with pdf_url?
            # "links", # Assuming not in Paper model
            "area",  # Added from Paper model
        ]

        # Prepare the INSERT statement with ON CONFLICT clause using pwc_id
        insert_sql = sql.SQL(
            """
            INSERT INTO papers ({fields})
            VALUES ({placeholders})
            ON CONFLICT (pwc_id) DO UPDATE SET
                {update_assignments}
            RETURNING paper_id;
            """
        ).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            # Dynamically create "col = EXCLUDED.col" assignments
            update_assignments=sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(
                    sql.Identifier(col), sql.Identifier(col)
                )
                for col in columns
                if col != "pwc_id"  # Don't update the conflict key itself
            ),
        )

        # Prepare parameter values from the Paper object, ensuring correct types
        values_tuple = (
            paper.pwc_id,  # Use pwc_id first
            paper.title,
            paper.arxiv_id_base,  # Use correct field
            paper.arxiv_id_versioned,  # Use correct field
            paper.summary,
            json.dumps(paper.authors) if paper.authors is not None else None,
            json.dumps(paper.categories) if paper.categories is not None else None,
            paper.published_date,
            # paper.updated_date, # If managed by DB
            # paper.source,
            paper.doi,
            str(paper.pdf_url) if paper.pdf_url else None,  # Convert HttpUrl to str
            str(paper.pwc_url) if paper.pwc_url else None,  # Convert HttpUrl to str
            paper.pwc_title,
            # paper.github_url,
            paper.primary_category,
            # paper.journal_ref,
            # paper.comment,
            # paper.url_abs,
            # paper.url_pdf,
            # paper.links,
            paper.area,
        )

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    self.logger.debug(
                        f"Executing upsert for pwc_id: {paper.pwc_id}"
                    )  # Log using pwc_id
                    # Log SQL only if necessary and safe
                    # self.logger.debug(f"SQL: {insert_sql.as_string(conn)}")
                    # self.logger.debug(f"Params: {values_tuple}")

                    # Execute the upsert
                    await cur.execute(insert_sql, values_tuple)
                    result = await cur.fetchone()

                    # Return the paper_id (primary key)
                    if result and result[0] is not None:
                        paper_id = int(result[0])
                        self.logger.info(
                            f"Successfully upserted paper (pwc_id={paper.pwc_id}), returned paper_id: {paper_id}"
                        )  # Log using pwc_id
                        return paper_id
                    else:
                        # This case might happen if RETURNING didn't work as expected or the row was deleted concurrently
                        self.logger.error(
                            f"Upsert for paper (pwc_id={paper.pwc_id}) did not return paper_id."
                        )  # Log using pwc_id
                        return None

        except Exception as e:
            # Log the specific paper data that caused the error for easier debugging
            try:
                # Use model_dump for Pydantic v2
                paper_data_dict = paper.model_dump(exclude_unset=True, mode="json")
            except Exception as dump_error:
                paper_data_dict = {"error": f"Could not dump paper model: {dump_error}"}
            # Log full traceback
            self.logger.error(
                f"Error upserting paper (pwc_id={paper.pwc_id or 'N/A'}) into PG: {e}",
                exc_info=True,
            )  # Log using pwc_id
            # Log relevant paper data
            self.logger.error(f"Failing paper data (dict): {paper_data_dict}")
            return None

    # --- START: Corrected implementations for specific relation fetching methods --- #

    async def get_tasks_for_papers(self, paper_ids: List[int]) -> Dict[int, List[str]]:
        """Fetches associated task names for a list of paper IDs."""
        if not paper_ids:
            return {}
        query = sql.SQL(
            """
            SELECT paper_id, task_name
            FROM pwc_tasks -- Corrected table name
            WHERE paper_id = ANY(%s);
            """
        )
        results_map: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    rows = await cur.fetchall()
                    for row in rows:
                        paper_id, item_value = row
                        if paper_id in results_map and item_value:
                            results_map[paper_id].append(item_value)
            return results_map
        except psycopg.Error as db_err:
            self.logger.error(
                f"Database error fetching tasks for papers: {db_err}", exc_info=True
            )
            return results_map
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching tasks for papers: {e}", exc_info=True
            )
            return results_map

    async def get_datasets_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """Fetches associated dataset names for a list of paper IDs."""
        if not paper_ids:
            return {}
        query = sql.SQL(
            """
            SELECT paper_id, dataset_name
            FROM pwc_datasets -- Corrected table name
            WHERE paper_id = ANY(%s);
            """
        )
        results_map: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    rows = await cur.fetchall()
                    for row in rows:
                        paper_id, item_value = row
                        if paper_id in results_map and item_value:
                            results_map[paper_id].append(item_value)
            return results_map
        except psycopg.Error as db_err:
            self.logger.error(
                f"Database error fetching datasets for papers: {db_err}", exc_info=True
            )
            return results_map
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching datasets for papers: {e}", exc_info=True
            )
            return results_map

    async def get_repositories_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """Fetches associated repository URLs for a list of paper IDs."""
        if not paper_ids:
            return {}
        query = sql.SQL(
            """
            SELECT paper_id, url -- Corrected column name
            FROM pwc_repositories -- Corrected table name
            WHERE paper_id = ANY(%s);
            """
        )
        results_map: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    rows = await cur.fetchall()
                    for row in rows:
                        paper_id, item_value = row
                        if paper_id in results_map and item_value:
                            results_map[paper_id].append(item_value)
            return results_map
        except psycopg.Error as db_err:
            self.logger.error(
                f"Database error fetching repositories for papers: {db_err}",
                exc_info=True,
            )
            return results_map
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching repositories for papers: {e}", exc_info=True
            )
            return results_map

    # --- END: Corrected implementations for specific relation fetching methods --- #
