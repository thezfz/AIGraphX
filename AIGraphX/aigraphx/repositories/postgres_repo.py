from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Literal, cast
import traceback
import logging
from psycopg_pool import AsyncConnectionPool  # Import pool
from psycopg.rows import dict_row  # Import dict_row
import json
from datetime import date  # Add date
from psycopg import sql  # Use psycopg3 for SQL composition
import numpy as np  # Import numpy
from aigraphx.models.paper import Paper  # Add import for Paper model
import psycopg  # Add missing import for psycopg.Error
from psycopg import errors as psycopg_errors # For specific error handling
from typing import get_args
import re # Added re for text preprocessing

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
                    # Convert DictRow to dict
                    return dict(result) if result else None
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
                    # Convert List[DictRow] to List[Dict]
                    return [dict(row) for row in results]
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
                    results = await cur.fetchall()
                    # Convert List[DictRow] to List[Dict]
                    return [dict(row) for row in results]
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
        filter_area: Optional[List[str]] = None,
        filter_authors: Optional[List[str]] = None,
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
        if filter_area and len(filter_area) > 0:
            where_clauses.append(f"area = ANY(%(filter_area)s)")
            params["filter_area"] = filter_area
        if filter_authors and len(filter_authors) > 0:
            author_conditions = []
            for i, author in enumerate(filter_authors):
                param_key = f"author_filter_{i}"
                author_conditions.append(f"authors::text ILIKE %({param_key})s")
                params[param_key] = f"%{author}%"
            if author_conditions:
                where_clauses.append(f"({' OR '.join(author_conditions)})")

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
            "paper_id, pwc_id, title, summary, pdf_url, published_date, authors, area, conference"
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
                        fetch_results = await cur.fetchall()
                        # Convert List[DictRow] to List[Dict]
                        results = [dict(row) for row in fetch_results]
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
        # Query ALL columns from hf_models table, not just selected ones
        query = "SELECT * FROM hf_models WHERE hf_model_id = ANY(%s);"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (model_ids,))
                    results = await cur.fetchall()
                    # Convert List[DictRow] to List[Dict]
                    return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error fetching HF model details by IDs from PG: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def search_models_by_keyword(
        self,
        query: str,
        limit: int = 10,
        skip: int = 0,
        sort_by: Optional[
            Literal["likes", "downloads", "last_modified"]
        ] = "last_modified",
        sort_order: Optional[Literal["asc", "desc"]] = "desc",
        pipeline_tag: Optional[str] = None,
        filter_library_name: Optional[str] = None,
        filter_tags: Optional[List[str]] = None,
        filter_author: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Searches hf_models table by keyword, with sorting and optional filtering."""
        search_term = f"%{query}%"
        params: Dict[str, Any] = {
            "search_term": search_term,
            "limit": limit,
            "offset": skip,
        }

        # WHERE conditions list
        where_conditions = [
            "(hf_model_id ILIKE %(search_term)s OR hf_author ILIKE %(search_term)s OR COALESCE(hf_pipeline_tag, '') ILIKE %(search_term)s)"
        ]
        if pipeline_tag:
            safe_pipeline_tag_key = "pipeline_tag"
            where_conditions.append(f"hf_pipeline_tag = %({safe_pipeline_tag_key})s")
            params[safe_pipeline_tag_key] = pipeline_tag

        if filter_library_name:
            safe_library_key = "filter_library"
            where_conditions.append(f"hf_library_name ILIKE %({safe_library_key})s")
            params[safe_library_key] = filter_library_name

        if filter_author:
            safe_author_key = "filter_author"
            where_conditions.append(f"hf_author ILIKE %({safe_author_key})s")
            params[safe_author_key] = f"%{filter_author}%"

        if filter_tags and len(filter_tags) > 0:
            safe_tags_key = "filter_tags"
            where_conditions.append(f"hf_tags @> %({safe_tags_key})s::jsonb")
            params[safe_tags_key] = json.dumps(filter_tags)

        final_where_clause = " AND ".join(where_conditions)

        # Fields to select - Added hf_readme_content and hf_dataset_links
        select_fields = "hf_model_id, hf_author, hf_pipeline_tag, hf_last_modified, hf_tags, hf_likes, hf_downloads, hf_library_name, hf_sha, hf_readme_content, hf_dataset_links"

        # --- Safely construct ORDER BY clause ---
        valid_sort_columns = {
            "likes": "hf_likes",
            "downloads": "hf_downloads",
            "last_modified": "hf_last_modified",
        }
        # Validate sort_by against allowed keys
        db_sort_column = valid_sort_columns.get(
            sort_by or "last_modified", "hf_last_modified"
        )
        # Validate sort_order
        db_sort_order = "DESC" if sort_order == "desc" else "ASC"
        # Construct the ORDER BY string safely (column name is validated)
        # Use double quotes for potential case sensitivity or reserved words
        order_by_sql_str = f'ORDER BY "{db_sort_column}" {db_sort_order} NULLS LAST, "hf_model_id" {db_sort_order}'
        # --- End Safe ORDER BY ---

        # Construct final SQL strings using f-strings
        # (Placeholders %(...)s are handled by psycopg execute)
        count_sql_str = f"""
            SELECT COUNT(*) 
            FROM hf_models 
            WHERE {final_where_clause}
        """

        select_sql_str = f"""
            SELECT {select_fields} 
            FROM hf_models 
            WHERE {final_where_clause} 
            {order_by_sql_str} 
            LIMIT %(limit)s OFFSET %(offset)s
        """

        total_count = 0
        results: List[Dict[str, Any]] = []

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    # Execute count query using the param dict
                    await cur.execute(count_sql_str, params)
                    count_result = await cur.fetchone()
                    total_count = count_result["count"] if count_result else 0
                    logger.debug(
                        f"[search_models_by_keyword] Total count: {total_count} for query '{query}' and filters {{\"pipeline_tag\": {pipeline_tag}}}"
                    )

                    # Execute main select query only if needed
                    if total_count > skip:
                        # Execute the built SQL string with the param dict
                        await cur.execute(select_sql_str, params)
                        results = await cur.fetchall()
                        logger.debug(
                            f"[search_models_by_keyword] Fetched {len(results)} rows for page."
                        )
                    else:
                        logger.debug(
                            "[search_models_by_keyword] Skip is >= total_count, not fetching rows."
                        )

            return results, total_count

        except psycopg.Error as db_err:
            logger.error(
                f"Database error during keyword model search: {db_err}", exc_info=True
            )
            return [], 0
        except Exception as e:
            logger.error(
                f"Unexpected error during keyword model search: {e}", exc_info=True
            )
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
        # 修复SQL语法错误，不要在查询中使用分号
        query = "SELECT * FROM hf_models"  # 移除末尾的分号
        offset = 0
        try:
            while True:
                # 正确的SQL格式
                batch_query = (
                    f"{query} ORDER BY hf_model_id LIMIT %s OFFSET %s"  # 移除末尾的分号
                )
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
        # 基本参数验证
        if not paper_data or "pwc_id" not in paper_data:
            self.logger.error("Cannot save paper: missing data or pwc_id.")
            return False

        try:
            # 准备列和值
            cols = []
            vals = []
            excluded_updates = []

            for key, value in paper_data.items():
                # 处理特殊字段
                if key in ["authors", "categories"] and value is not None:
                    # 确保列表类型被序列化为JSON
                    if isinstance(value, list):
                        value = json.dumps(value)

                cols.append(key)
                vals.append(value)

                # 对于ON CONFLICT，排除主键
                if key != "pwc_id":
                    excluded_updates.append(f"{key} = EXCLUDED.{key}")

            # 构建查询
            if not excluded_updates:
                # 如果只有pwc_id，至少添加一个更新字段
                excluded_updates = ["updated_at = CURRENT_TIMESTAMP"]

            query = f"""
                INSERT INTO papers ({", ".join(cols)})
                VALUES ({", ".join(["%s"] * len(vals))})
                ON CONFLICT (pwc_id) DO UPDATE SET
                    {", ".join(excluded_updates)};
            """

            self.logger.debug(f"Saving paper with pwc_id: {paper_data.get('pwc_id')}")

            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, vals)
                    # 返回True表示成功执行，无论是插入还是更新
                    return True
        except Exception as e:
            self.logger.error(
                f"Error saving paper with pwc_id {paper_data.get('pwc_id')} to PG: {e}"
            )
            self.logger.debug(traceback.format_exc())
            return False

    async def save_hf_models_batch(self, models_data: List[Dict[str, Any]]) -> None:
        """Saves a batch of HF models, updating existing ones based on hf_model_id."""
        if not models_data:
            return

        # Prepare columns based on the first model (assume consistency)
        # Handle potential JSON fields if needed (e.g., hf_tags)
        if not models_data[0]:
            return  # Handle empty dict case

        cols = list(models_data[0].keys())
        # Ensure hf_model_id is present for ON CONFLICT
        if "hf_model_id" not in cols:
            self.logger.error(
                "Cannot save HF models batch: 'hf_model_id' missing in data."
            )
            return

        excluded_updates = []
        for key in cols:
            if key != "hf_model_id":
                # Use f-string safely as key comes from dict keys
                excluded_updates.append(f"{key} = EXCLUDED.{key}")

        # Add updated_at to excluded_updates if not already present, to ensure it's always updated
        if "updated_at = EXCLUDED.updated_at" not in excluded_updates and "updated_at" in cols:
            pass # it's already there via EXCLUDED.updated_at
        elif not any("updated_at" in ex_up for ex_up in excluded_updates):
             excluded_updates.append("updated_at = NOW()")

        # 构建每行的VALUES部分
        placeholders = ", ".join(["%s"] * len(cols))

        # Construct query (Ensure table and column names are correct)
        query = f"""
            INSERT INTO hf_models ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT (hf_model_id) DO UPDATE SET
            {", ".join(excluded_updates)};
        """

        # Prepare data tuples, ensuring list fields are JSON strings
        data_tuples = []
        for model in models_data:
            row = []
            for col in cols:
                value = model.get(col)
                # FIX: Serialize list/dict fields (like 'hf_tags', 'hf_dataset_links') to JSON strings
                if col in ["hf_tags", "hf_dataset_links"] and isinstance(value, (list, dict)):
                    row.append(json.dumps(value))
                elif col == "hf_last_modified" and isinstance(value, str):
                    # Assuming hf_last_modified might come as string, parse to datetime if so
                    # This depends on the data source; if it's always datetime, this is not needed.
                    # For now, let's assume it's passed correctly as datetime or None.
                    row.append(value)
                else:
                    row.append(value)  # type: ignore[arg-type] # Allow None, db driver handles it
            data_tuples.append(tuple(row))

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Use executemany for batch operations
                    await cur.executemany(query, data_tuples)
                    self.logger.info(
                        f"Successfully saved/updated batch of {len(models_data)} HF models using executemany."
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
                    result = await cur.fetchone()
                    # Convert DictRow to dict
                    return dict(result) if result else None
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
        if hasattr(self, "pool") and self.pool is not None:
            try:
                await self.pool.close()
                self.logger.info("PostgreSQL connection pool closed successfully.")
            except Exception as e:
                self.logger.error(f"Error closing PostgreSQL connection pool: {e}")
                self.logger.debug(traceback.format_exc())

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

    def _preprocess_readme_content(self, text: Optional[str]) -> str:
        if not text:
            return ""
        
        # 1. Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. Remove Markdown fenced code blocks (and their content)
        text = re.sub(r'```[^\S\r\n]*[a-zA-Z0-9_+#]*\n.*?\n```', '', text, flags=re.DOTALL | re.MULTILINE)
        
        # 3. Remove Markdown inline code (and its content)
        text = re.sub(r'`.*?`', '', text)
        
        # 4. Remove Markdown images ![alt text](url)
        text = re.sub(r'!\[[^]]*\]\([^)]*\)', '', text)
        
        # 5. Remove Markdown links [link text](url) - keep link text
        text = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', text)
        
        # 6. Remove Markdown headings (e.g., # Heading, ## Heading)
        text = re.sub(r'^(#+\s+)', '', text, flags=re.MULTILINE)
        
        # 7. Remove Markdown list markers (*, -, +, 1.)
        text = re.sub(r'^[*\-\+]\s+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[*\-\+]\s*\d+\.\s+\s*', '', text, flags=re.MULTILINE)

        # 8. Remove Markdown bold and italic markers, keeping the text
        text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'\1\2', text, flags=re.DOTALL)
        text = re.sub(r'\*(.*?)\*|_(.*?)_', r'\1\2', text, flags=re.DOTALL)

        # 9. Remove horizontal rules (---, ***, ___)
        text = re.sub(r'^[*-_]{3,}\s*$', '', text, flags=re.MULTILINE)

        # 10. Remove blockquotes
        text = re.sub(r'^>\s?\s*', '', text, flags=re.MULTILINE)

        # Normalize whitespace: replace multiple spaces/newlines with a single space, then strip
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    async def get_all_models_for_indexing(
        self,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Fetches all Hugging Face model IDs and their text representation.
        hf_readme_content is preprocessed to remove markdown/HTML tags and code blocks.
        """
        query = """
            SELECT
                hf_model_id,
                hf_readme_content,    -- Raw README content
                hf_pipeline_tag,      -- Pipeline tag
                hf_tags               -- Raw tags (JSONB)
            FROM hf_models;
            """

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur: # Ensure dict_row is used
                    await cur.execute(query)
                    while True:
                        rows = await cur.fetchmany(100)
                        if not rows:
                            break
                        for row in rows:
                            hf_model_id = row.get("hf_model_id")
                            raw_readme = row.get("hf_readme_content")
                            pipeline_tag = row.get("hf_pipeline_tag") or ""
                            
                            tags_data = row.get("hf_tags") # Should be a list if JSONB is parsed by psycopg
                            tags_text = ""
                            if isinstance(tags_data, list):
                                tags_text = " ".join(filter(None, tags_data))
                            elif isinstance(tags_data, str): # Fallback if not parsed
                                try:
                                    parsed_tags = json.loads(tags_data)
                                    if isinstance(parsed_tags, list):
                                        tags_text = " ".join(filter(None, parsed_tags))
                                except json.JSONDecodeError:
                                    pass # Keep tags_text as ""
                            
                            processed_readme = self._preprocess_readme_content(raw_readme)
                            
                            # Concatenate parts for text representation
                            # Ensure no excessive spacing if some parts are empty
                            parts = [processed_readme, pipeline_tag, tags_text]
                            text_representation = " ".join(p for p in parts if p).strip() # Filter out empty strings before joining
                            
                            if hf_model_id:
                                # Yield even if text_representation is empty, as per prior logic
                                yield (hf_model_id, text_representation)
        except Exception as e:
            self.logger.error(f"Error fetching all models for indexing: {e}")
            self.logger.debug(traceback.format_exc())
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
            "conference", # Added conference
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
            paper.conference, # Added conference
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
            FROM pwc_tasks
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
            FROM pwc_datasets
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
    ) -> Dict[int, List[Dict[str, Any]]]: # Return type changed
        """Fetches associated repository details for a list of paper IDs."""
        if not paper_ids:
            return {}
        # Select all relevant columns from pwc_repositories
        query = sql.SQL(
            """
            SELECT paper_id, url, stars, is_official, framework, license, language
            FROM pwc_repositories
            WHERE paper_id = ANY(%s);
            """
        )
        # Initialize with empty lists
        results_map: Dict[int, List[Dict[str, Any]]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                # Use dict_row to get results as dictionaries directly
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (paper_ids,))
                    rows = await cur.fetchall()
                    for row_dict in rows: # row is already a dict due to dict_row
                        paper_id = row_dict.get("paper_id")
                        if paper_id in results_map:
                            # Create a dictionary for the repository, excluding paper_id itself
                            repo_details = {k: v for k, v in row_dict.items() if k != "paper_id"}
                            results_map[paper_id].append(repo_details)
            return results_map
        except psycopg.Error as db_err:
            self.logger.error(
                f"Database error fetching repositories for papers: {db_err}",
                exc_info=True,
            )
            return results_map # Return partially filled map on error
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching repositories for papers: {e}", exc_info=True
            )
            return results_map # Return partially filled map on error

    # --- END: Corrected implementations for specific relation fetching methods --- #

    async def get_unique_paper_areas(self) -> List[str]:
        """
        获取论文表中所有唯一的领域（area）值。

        返回:
            List[str]: 所有唯一的论文领域列表
        """
        query = """
            SELECT DISTINCT area 
            FROM papers 
            WHERE area IS NOT NULL AND area != ''
            ORDER BY area;
        """
        try:
            areas = []
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    results = await cur.fetchall()
                    # 结果是包含单个元素的元组列表，提取字符串值
                    areas = [row[0] for row in results if row[0]]

            self.logger.info(f"从数据库获取到 {len(areas)} 个唯一论文领域")
            return areas
        except Exception as e:
            self.logger.error(f"获取唯一论文领域列表时出错: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    # --- START: Add get_methods_for_papers --- #
    async def get_methods_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """Fetches related method names for a list of paper IDs."""
        if not paper_ids:
            return {}

        # Query directly from pwc_methods table
        query = sql.SQL(
            """
            SELECT paper_id, method_name 
            FROM pwc_methods
            WHERE paper_id = ANY(%s);
            """
        )
        result_map: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur: # Using dict_row
                    await cur.execute(query, (paper_ids,))
                    rows = await cur.fetchall() # Fetch all results
                    for row in rows:
                        paper_id_val = row.get("paper_id")
                        method_name = row.get("method_name")
                        if paper_id_val in result_map and method_name:
                            result_map[paper_id_val].append(method_name)
        except psycopg_errors.UndefinedTable: # Catch specific psycopg error
            logger.warning(
                "Table 'pwc_methods' does not exist. Cannot fetch methods."
            )
            return {} # Return empty map if table doesn't exist
        except psycopg.Error as db_err: # General psycopg error
            logger.error(f"Database error fetching methods for papers: {db_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error fetching methods for papers: {e}", exc_info=True)
            # Return potentially partial map on other errors, or an empty one if preferred
        return result_map

    # --- END: Add get_methods_for_papers --- #
