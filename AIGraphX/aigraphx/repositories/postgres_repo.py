# -*- coding: utf-8 -*-
"""
PostgresRepository - PostgreSQL 数据库交互模块

该模块定义了 `PostgresRepository` 类，封装了所有与 PostgreSQL 数据库相关的操作。
它使用 `psycopg` (版本3+) 异步库和连接池 (`psycopg_pool`) 来执行 SQL 查询，
处理论文 (papers)、Hugging Face 模型 (hf_models) 以及它们之间的关联数据。

主要功能:
- **连接管理**: 通过依赖注入接收并使用异步连接池 (`AsyncConnectionPool`)。
- **数据获取**:
    - 根据 ID 获取单篇或多篇论文/模型的详细信息。
    - 执行关键词搜索（支持过滤、排序、分页）。
    - 获取用于同步/索引构建的批量数据。
    - 获取关联数据（如论文的任务、数据集、代码库等）。
    - 获取唯一的论文领域列表。
- **数据写入/更新**:
    - 保存（upsert）单篇论文数据。
    - 批量保存 Hugging Face 模型数据。
- **通用查询**:
    - 提供基于游标的批量数据获取方法。
    - 提供获取单条记录的方法。
- **错误处理**: 记录数据库操作相关的错误和异常。
- **SQL 构建**: 使用 `psycopg.sql` 安全地构建动态 SQL 查询。

与其他文件的交互:
- **`aigraphx.core.db`**: 在应用启动时创建连接池，并通过依赖注入传递给 `PostgresRepository` 实例。
- **`aigraphx.services.search_service.SearchService`**: 依赖 `PostgresRepository` 获取关键词搜索结果和特定 ID 的数据详情。
- **`aigraphx.scripts.sync_papers` / `sync_models`**: 使用 `PostgresRepository` 来保存从外部源同步过来的数据。
- **`aigraphx.scripts.build_vector_index`**: 使用 `PostgresRepository` 获取需要建立索引的文本数据。
- **`aigraphx.models`**: 可能使用 Pydantic 模型（如 `Paper`）来验证或构造数据。
- **`psycopg_pool`, `psycopg`**: 底层的数据库驱动和连接池库。

注意事项:
- 所有与数据库交互的方法都应该是异步的 (`async def`)。
- 使用参数化查询或 `psycopg.sql` 来防止 SQL 注入。
- 错误处理应记录足够的信息以便调试。
- 对于可能返回大量数据的方法，考虑使用异步生成器 (`AsyncGenerator`) 或游标。
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Literal, cast
import traceback
import logging
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
import json
from datetime import date
import asyncpg
from asyncpg import Record
from psycopg import sql
import numpy as np
from aigraphx.models.paper import Paper
import psycopg
from typing import get_args

logger = logging.getLogger(__name__)


class PostgresRepository:
    """
    封装与 PostgreSQL 数据库交互操作的类。
    """

    def __init__(self, pool: AsyncConnectionPool):
        """
        初始化 PostgreSQL 仓库。

        Args:
            pool (AsyncConnectionPool): 一个已经初始化好的 psycopg 异步连接池实例。
                                         通过依赖注入传入，而不是在仓库内部创建。
        """
        self.pool = pool
        self.logger = logger
        logger.info("PostgresRepository initialized.")

    async def get_paper_details_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """
        根据论文的整数 ID (`paper_id`) 从 `papers` 表获取单篇论文的详细信息。

        Args:
            paper_id (int): 要查询的论文的整数 ID。

        Returns:
            Optional[Dict[str, Any]]: 包含论文所有字段的字典，如果未找到则返回 None。
                                      字段名对应数据库表的列名。
                                      对于 JSON/JSONB 字段（如 'authors'），psycopg 通常会自动解码为 Python 对象。
        """
        query = "SELECT * FROM papers WHERE paper_id = %s;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (paper_id,))
                    result = await cur.fetchone()
                    if result:
                        self.logger.debug(f"获取到 paper_id {paper_id} 的论文详情。")
                    else:
                        self.logger.debug(f"未找到 paper_id 为 {paper_id} 的论文。")
                    return result
        except psycopg.Error as db_err:
            self.logger.error(f"从 PG 获取 ID 为 {paper_id} 的论文时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return None
        except Exception as e:
            self.logger.error(f"获取 ID 为 {paper_id} 的论文时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def get_papers_details_by_ids(
        self, paper_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        根据论文的整数 ID 列表 (`paper_ids`) 从 `papers` 表获取多篇论文的详细信息。

        Args:
            paper_ids (List[int]): 要查询的论文的整数 ID 列表。

        Returns:
            List[Dict[str, Any]]: 包含多篇论文详细信息的字典列表。如果列表为空或查询出错，返回空列表。
        """
        if not paper_ids:
            return []
        query = "SELECT * FROM papers WHERE paper_id = ANY(%s);"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (paper_ids,))
                    results = await cur.fetchall()
                    self.logger.debug(f"获取到 {len(results)} 篇论文的详情。")
                    return results
        except psycopg.Error as db_err:
            self.logger.error(f"从 PG 按 ID 列表获取论文详情时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return []
        except Exception as e:
            self.logger.error(f"按 ID 列表获取论文详情时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def get_all_papers_for_sync(self) -> List[Dict[str, Any]]:
        """
        获取所有论文的特定字段（ID 和摘要），主要用于需要处理所有论文数据的后台任务，
        例如构建搜索索引（Faiss）或与其他系统同步。
        只选择必要的字段以减少内存消耗。
        只选择摘要不为空的论文。

        Returns:
            List[Dict[str, Any]]: 包含论文 ID 和摘要的字典列表。出错则返回空列表。
        """
        query = "SELECT paper_id, summary FROM papers WHERE summary IS NOT NULL AND summary != '';"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query)
                    results = await cur.fetchall()
                    self.logger.info(
                        f"获取到 {len(results)} 篇用于同步/索引的论文（ID 和摘要）。"
                    )
                    return results
        except psycopg.Error as db_err:
            self.logger.error(f"获取所有用于同步的论文时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return []
        except Exception as e:
            self.logger.error(f"获取所有用于同步的论文时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def count_papers(self) -> int:
        """
        统计 `papers` 表中的总记录数。

        Returns:
            int: 论文总数，如果出错则返回 0。
        """
        query = "SELECT COUNT(*) FROM papers;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    result = await cur.fetchone()
                    count = result[0] if result else 0
                    self.logger.info(f"数据库中的论文总数: {count}")
                    return count
        except psycopg.Error as db_err:
            self.logger.error(f"在 PG 中统计论文数量时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return 0
        except Exception as e:
            self.logger.error(f"统计论文数量时发生意外错误: {e}")
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
        sort_by: Optional[
            Literal["published_date", "title", "paper_id"]
        ] = "published_date",
        sort_order: Optional[Literal["asc", "desc"]] = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        根据关键词在论文的标题 (title) 或摘要 (summary) 中进行搜索。
        支持过滤、排序和分页。

        Args:
            query (str): 要搜索的关键词。
            skip (int): 要跳过的记录数（用于分页）。
            limit (int): 要返回的最大记录数。
            published_after (Optional[date]): 过滤发布日期在此日期之后（包含）的论文。
            published_before (Optional[date]): 过滤发布日期在此日期之前（包含）的论文。
            filter_area (Optional[List[str]]): 按领域过滤论文（精确匹配列表中的任何一个）。
            filter_authors (Optional[List[str]]): 按作者过滤论文（模糊匹配，包含任何一个作者即可）。
            sort_by (Optional[Literal["published_date", "title", "paper_id"]]): 用于排序的列名。
            sort_order (Optional[Literal["asc", "desc"]]): 排序方向。

        Returns:
            Tuple[List[Dict[str, Any]], int]: 一个元组，包含：
                - 匹配条件的论文详情字典列表。
                - 匹配条件的总论文数（忽略分页）。
        """
        self.logger.debug(
            f"按关键词搜索论文: '{query}', skip: {skip}, limit: {limit}, filters: ..."
        )

        # --- 构建 WHERE 子句和参数 --- #
        params: Dict[str, Any] = {"like_query": f"%{query}%"}
        # 初始条件：标题或摘要包含关键词（不区分大小写）
        where_clauses = ["(title ILIKE %(like_query)s OR summary ILIKE %(like_query)s)"]

        # 添加日期过滤条件
        if published_after:
            where_clauses.append(f"published_date >= %(published_after)s")
            params["published_after"] = published_after
        if published_before:
            where_clauses.append(f"published_date <= %(published_before)s")
            params["published_before"] = published_before
        # 添加领域过滤条件
        if filter_area and len(filter_area) > 0:
            where_clauses.append(f"area = ANY(%(filter_area)s)")
            params["filter_area"] = filter_area
        # 添加作者过滤条件 (模糊匹配，检查 authors JSON 数组)
        if filter_authors and len(filter_authors) > 0:
            author_conditions = []
            for i, author in enumerate(filter_authors):
                # 为每个作者创建一个唯一的参数键
                param_key = f"author_filter_{i}"
                # 使用 JSON 操作符或文本转换进行模糊匹配
                # 注意: authors::text ILIKE ... 效率可能不高，如果性能是问题，考虑其他方法
                author_conditions.append(f"authors::text ILIKE %({param_key})s")
                params[param_key] = f"%{author}%"
            if author_conditions:
                # 将多个作者条件用 OR 连接
                where_clauses.append(f"({' OR '.join(author_conditions)})")

        # 将所有 WHERE 条件用 AND 连接
        where_sql = " AND ".join(where_clauses)

        # --- 构建 ORDER BY 子句 --- #
        order_by_column: str
        valid_sort_columns = get_args(Literal["published_date", "title", "paper_id"])

        # 验证 sort_by 参数是否有效
        if sort_by in valid_sort_columns:
            order_by_column = cast(str, sort_by)
        else:
            # 如果 sort_by 无效或为 None，使用默认排序
            if sort_by is not None:
                self.logger.warning(
                    f"无效的 sort_by 值 '{sort_by}' 用于论文关键词搜索。默认使用 'published_date'。"
                )
            order_by_column = "published_date"

        # 确定排序方向
        order_direction = "DESC" if sort_order == "desc" else "ASC"
        # 构建完整的 ORDER BY 子句，添加 paper_id 作为第二排序键以确保稳定性
        # NULLS LAST 确保 NULL 值排在最后
        # 使用双引号包围列名以处理可能的保留字或大小写敏感性
        order_by_sql = f'ORDER BY "{order_by_column}" {order_direction} NULLS LAST, paper_id {order_direction}'

        # --- 构建 COUNT 查询 --- #
        # 查询匹配条件的总记录数
        count_sql = f"SELECT COUNT(*) FROM papers WHERE {where_sql};"

        # --- 构建 SELECT 查询 --- #
        # 选择 SearchResultItem 模型所需的所有列
        select_fields = (
            "paper_id, pwc_id, title, summary, pdf_url, published_date, authors, area"
        )
        # 添加分页参数
        params["offset"] = skip
        params["limit"] = limit
        # 构建最终的 SELECT 语句
        select_sql = f"""
            SELECT {select_fields}
            FROM papers
            WHERE {where_sql}
            {order_by_sql}
            OFFSET %(offset)s
            LIMIT %(limit)s;
            """

        total_count = 0
        results: List[Dict[str, Any]] = []

        try:
            async with self.pool.connection() as conn:
                # 首先执行 COUNT 查询获取总数
                async with conn.cursor() as cur_count:
                    await cur_count.execute(count_sql, params)
                    count_result = await cur_count.fetchone()
                    total_count = count_result[0] if count_result else 0
                    self.logger.debug(f"关键词搜索匹配总数: {total_count}")

                # 只有在总数大于0且请求的 limit 大于0时才执行 SELECT 查询
                if total_count > 0 and limit > 0:
                    async with conn.cursor(row_factory=dict_row) as cur_select:
                        await cur_select.execute(select_sql, params)
                        results = await cur_select.fetchall()
                        self.logger.debug(
                            f"关键词搜索获取到 {len(results)} 篇论文详情。"
                        )
                else:
                    self.logger.debug("由于总数为零或 limit 为零，跳过获取详情。")

        except psycopg.Error as db_err:
            self.logger.error(f"论文关键词搜索期间出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            # 出错时返回空列表和 0
            return [], 0
        except Exception as e:
            self.logger.error(f"论文关键词搜索期间发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return [], 0

        # 返回结果列表和总数
        return results, total_count

    async def get_hf_models_by_ids(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """
        根据 Hugging Face 模型 ID 列表从 `hf_models` 表获取模型详细信息。

        Args:
            model_ids (List[str]): Hugging Face 模型 ID 字符串列表。

        Returns:
            List[Dict[str, Any]]: 包含找到的模型详细信息的字典列表。
                                  如果输入列表为空或查询出错，返回空列表。
        """
        if not model_ids:
            self.logger.debug("get_hf_models_by_ids 接收到空列表。")
            return []

        # 使用 ANY 操作符进行批量查询
        query = "SELECT * FROM hf_models WHERE hf_model_id = ANY(%s);"

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (model_ids,))
                    results = await cur.fetchall()
                    self.logger.debug(f"按 ID 获取到 {len(results)} 个 HF 模型详情。")
                    # 可选: 处理 JSON 字段 (例如 'tags')
                    # for row in results:
                    #     if 'tags' in row and isinstance(row['tags'], str):
                    #         try: row['tags'] = json.loads(row['tags'])
                    #         except json.JSONDecodeError: logger.warning(...) ; row['tags'] = None
                    return results
        except psycopg.Error as db_err:
            self.logger.error(f"从 PG 按 ID 获取 HF 模型详情时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return []
        except Exception as e:
            self.logger.error(f"按 ID 获取 HF 模型详情时发生意外错误: {e}")
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
        # Add date filters for models as well
        published_after: Optional[date] = None,
        published_before: Optional[date] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        根据关键词在模型 ID (`hf_model_id`) 中进行搜索，并支持多种过滤条件。
        返回分页的模型详情和匹配总数。

        Args:
            query (str): 搜索关键词。
            limit (int): 返回的最大模型数量。
            skip (int): 跳过的模型数量（用于分页）。
            sort_by (Optional[Literal["likes", "downloads", "last_modified"]]): 排序依据的列。
            sort_order (Optional[Literal["asc", "desc"]]): 排序方向。
            pipeline_tag (Optional[str]): 按 pipeline tag 过滤。
            filter_library_name (Optional[str]): 按库名称过滤 (例如 'transformers')。
            filter_tags (Optional[List[str]]): 按标签过滤（需要包含列表中的 *所有* 标签）。
            filter_author (Optional[str]): 按作者/组织过滤（模糊匹配）。
            published_after (Optional[date]): 过滤最后修改日期在此之后（包含）的模型。
            published_before (Optional[date]): 过滤最后修改日期在此之前（包含）的模型。

        Returns:
            Tuple[List[Dict[str, Any]], int]: 一个元组，包含：
                - 匹配条件的模型详情字典列表。
                - 匹配条件的总模型数（忽略分页）。
        """
        self.logger.debug(
            f"按关键词搜索模型: '{query}', skip: {skip}, limit: {limit}, filters: ..."
        )

        # --- 构建 WHERE 子句和参数 --- #
        params: Dict[str, Any] = {"like_query": f"%{query}%"}
        # 初始条件：模型 ID 包含关键词（不区分大小写）
        # 未来可以扩展到搜索描述等字段
        where_clauses = ["hf_model_id ILIKE %(like_query)s"]

        # 添加过滤条件
        if pipeline_tag:
            where_clauses.append("hf_pipeline_tag = %(pipeline_tag)s")
            params["pipeline_tag"] = pipeline_tag
        if filter_library_name:
            # 转换为小写进行不区分大小写的比较
            where_clauses.append("LOWER(hf_library_name) = LOWER(%(library_name)s)")
            params["library_name"] = filter_library_name
        if filter_author:
            # 作者进行模糊匹配
            where_clauses.append("hf_author ILIKE %(author_filter)s")
            params["author_filter"] = f"%{filter_author}%"
        if filter_tags and len(filter_tags) > 0:
            # 使用 JSONB 的 @> 操作符，检查 hf_tags 数组是否包含 filter_tags 中的所有元素
            # 注意：filter_tags 需要是 JSON 数组格式的字符串或由驱动程序处理
            where_clauses.append("hf_tags @> %(filter_tags)s::jsonb")
            params["filter_tags"] = json.dumps(filter_tags)  # 显式转换为 JSON 字符串
        # 添加日期过滤 (作用于 hf_last_modified 列)
        if published_after:
            where_clauses.append("hf_last_modified >= %(published_after)s")
            params["published_after"] = published_after
        if published_before:
            # 如果需要包含当天，可以调整比较符或日期
            where_clauses.append("hf_last_modified <= %(published_before)s")
            params["published_before"] = published_before

        where_sql = " AND ".join(where_clauses)

        # --- 构建 ORDER BY 子句 --- #
        order_by_column: str
        valid_sort_columns = get_args(Literal["likes", "downloads", "last_modified"])
        # 映射 API 排序键到数据库列名
        db_sort_column_map = {
            "likes": "hf_likes",
            "downloads": "hf_downloads",
            "last_modified": "hf_last_modified",
        }

        # 验证 sort_by 并获取数据库列名
        if sort_by in valid_sort_columns:
            order_by_column = db_sort_column_map[cast(str, sort_by)]
        else:
            if sort_by is not None:
                self.logger.warning(
                    f"无效的 sort_by 值 '{sort_by}' 用于模型关键词搜索。默认使用 'last_modified'。"
                )
            order_by_column = "hf_last_modified"  # 默认排序

        order_direction = "DESC" if sort_order == "desc" else "ASC"
        # 添加 hf_model_id 作为第二排序键以确保稳定性
        order_by_sql = f'ORDER BY "{order_by_column}" {order_direction} NULLS LAST, hf_model_id {order_direction}'

        # --- 构建 COUNT 和 SELECT 查询 --- #
        count_sql = f"SELECT COUNT(*) FROM hf_models WHERE {where_sql};"
        # 选择 HFSearchResultItem 模型所需的所有列
        select_fields = (
            "hf_model_id, hf_author, hf_pipeline_tag, hf_library_name, hf_tags, "
            "hf_likes, hf_downloads, hf_last_modified"
        )
        params["offset"] = skip
        params["limit"] = limit
        select_sql = f"""
            SELECT {select_fields}
            FROM hf_models
            WHERE {where_sql}
            {order_by_sql}
            OFFSET %(offset)s
            LIMIT %(limit)s;
            """

        total_count = 0
        results: List[Dict[str, Any]] = []

        try:
            async with self.pool.connection() as conn:
                # 执行 COUNT 查询
                async with conn.cursor() as cur_count:
                    await cur_count.execute(count_sql, params)
                    count_result = await cur_count.fetchone()
                    total_count = count_result[0] if count_result else 0
                    self.logger.debug(f"模型关键词搜索匹配总数: {total_count}")

                # 执行 SELECT 查询
                if total_count > 0 and limit > 0:
                    async with conn.cursor(row_factory=dict_row) as cur_select:
                        await cur_select.execute(select_sql, params)
                        results = await cur_select.fetchall()
                        self.logger.debug(
                            f"模型关键词搜索获取到 {len(results)} 个详情。"
                        )
                else:
                    self.logger.debug("由于总数为零或 limit 为零，跳过获取详情。")

        except psycopg.Error as db_err:
            self.logger.error(f"模型关键词搜索期间出错: {db_err}")
            self.logger.error(f"失败的查询参数: {params}")  # 记录失败时的参数
            self.logger.debug(traceback.format_exc())
            return [], 0
        except Exception as e:
            self.logger.error(f"模型关键词搜索期间发生意外错误: {e}")
            self.logger.error(f"失败的查询参数: {params}")  # 记录失败时的参数
            self.logger.debug(traceback.format_exc())
            return [], 0

        return results, total_count

    async def count_hf_models(self) -> int:
        """
        统计 `hf_models` 表中的总记录数。

        Returns:
            int: 模型总数，如果出错则返回 0。
        """
        query = "SELECT COUNT(*) FROM hf_models;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    result = await cur.fetchone()
                    count = result[0] if result else 0
                    self.logger.info(f"数据库中的 HF 模型总数: {count}")
                    return count
        except psycopg.Error as db_err:
            self.logger.error(f"在 PG 中统计 HF 模型数量时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return 0
        except Exception as e:
            self.logger.error(f"统计 HF 模型数量时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return 0

    async def get_all_hf_models_for_sync(
        self, batch_size: int = 1000
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        分批从数据库获取所有 Hugging Face 模型，生成用于同步或索引任务所需的特定字段。
        选择必要的字段，例如 ID 和用于嵌入的文本字段。

        Args:
            batch_size (int): 每个批次获取的模型数量。

        Yields:
            AsyncGenerator[List[Dict[str, Any]], None]: 异步生成器，每次产生一个字典列表，
                每个字典包含一批模型的 'hf_model_id' 和其他可能的文本字段。
        """
        # 选择用于索引/嵌入的相关字段，例如模型 ID 和标签（作为文本）
        # 注意: hf_tags::text 将 JSONB 转换为文本表示
        query = sql.SQL(
            "SELECT hf_model_id, hf_tags::text FROM hf_models ORDER BY hf_model_id"
        )

        try:
            async with self.pool.connection() as conn:
                # 使用服务器端游标高效检索大量数据
                async with conn.cursor(row_factory=dict_row) as cur:
                    # 使用 stream() 方法，它返回一个异步迭代器
                    # 注意：psycopg 3.1+ 的 stream 不再直接接受 batch_size，
                    # 它会根据内部缓冲区大小进行流式处理。如果需要严格的批处理，
                    # 可以在 yield 之前手动收集 batch_size 的记录。
                    # 此处简化为直接迭代 stream 返回的记录。
                    record_batch: List[Dict[str, Any]] = []
                    async for record in cur.stream(query):
                        record_batch.append(record)
                        if len(record_batch) >= batch_size:
                            self.logger.debug(
                                f"生成一批 {len(record_batch)} 个 HF 模型用于同步。"
                            )
                            yield record_batch
                            record_batch = []  # 清空批次
                    # 处理最后一批不足 batch_size 的记录
                    if record_batch:
                        self.logger.debug(
                            f"生成最后一批 {len(record_batch)} 个 HF 模型用于同步。"
                        )
                        yield record_batch

                    self.logger.info("完成获取所有 HF 模型用于同步。")
        except psycopg.Error as db_err:
            self.logger.error(f"获取所有 HF 模型用于同步时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            # 异常会隐式停止生成器
        except Exception as e:
            self.logger.error(f"获取所有 HF 模型用于同步时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            # 异常会隐式停止生成器

    async def save_paper(self, paper_data: Dict[str, Any]) -> bool:
        """
        保存或更新单篇论文记录 (UPSERT)，使用命名占位符。
        假定 `paper_id` 是主键。

        Args:
            paper_data (Dict[str, Any]): 包含论文数据的字典。

        Returns:
            bool: 操作成功返回 True，否则返回 False。
        """
        if "paper_id" not in paper_data or paper_data["paper_id"] is None:
            self.logger.error("无法保存论文: 'paper_id' 缺失或为 None。")
            return False

        # 准备参数字典，确保 JSON 字段已序列化
        params = paper_data.copy()
        for key in ["authors", "categories"]:
            if key in params and isinstance(params[key], list):
                try:
                    params[key] = json.dumps(params[key])
                except TypeError:
                    self.logger.warning(
                        f"无法序列化字段 '{key}' (论文 {params.get('paper_id')}), 使用 null。"
                    )
                    params[key] = None

        columns = list(params.keys())
        # 使用 %(key)s 风格的命名占位符
        values_placeholders = sql.SQL(", ").join(
            sql.Placeholder(col) for col in columns
        )
        columns_sql = sql.SQL(", ").join(map(sql.Identifier, columns))
        # 构建 UPDATE SET 子句
        update_assignments = sql.SQL(", ").join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
            for col in columns
            if col != "paper_id"  # 不更新主键本身
        )

        # 构建完整的 UPSERT SQL
        query = sql.SQL("""
            INSERT INTO papers ({columns})
            VALUES ({values})
            ON CONFLICT (paper_id) DO UPDATE SET {updates}
            RETURNING paper_id; -- 返回 ID 以确认操作
        """).format(
            columns=columns_sql, values=values_placeholders, updates=update_assignments
        )

        paper_id_log = params.get("paper_id", "UNKNOWN")
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # 直接将参数字典传递给 execute
                    await cur.execute(query, params)
                    result = await cur.fetchone()
                    # 检查 RETURNING 子句是否返回了 ID
                    if result and result[0] is not None:
                        self.logger.debug(f"成功 Upsert 论文，ID: {result[0]}")
                        return True
                    else:
                        self.logger.warning(
                            f"Upsert 论文 {paper_id_log} 未返回有效的 ID。可能是冲突但无更新或配置问题。"
                        )
                        # 根据业务逻辑决定返回值，如果 DO NOTHING 且冲突，可能算成功
                        return False
        except psycopg.Error as db_err:
            self.logger.error(f"Upsert 论文 {paper_id_log} 时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return False
        except Exception as e:
            self.logger.error(f"Upsert 论文 {paper_id_log} 时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    async def save_hf_models_batch(self, models_data: List[Dict[str, Any]]) -> None:
        """
        批量保存或更新 Hugging Face 模型 (UPSERT)。
        逐条执行 UPSERT，使用命名占位符。

        Args:
            models_data (List[Dict[str, Any]]): 包含模型数据的字典列表。
        """
        if not models_data:
            return

        processed_count = 0
        error_count = 0

        # 准备 SQL 模板 (假设批次内 key 一致)
        sample_model = models_data[0]
        columns = list(sample_model.keys())
        if "hf_model_id" not in columns:
            self.logger.error("无法批量保存 HF 模型: 'hf_model_id' 缺失。")
            return

        values_placeholders = sql.SQL(", ").join(
            sql.Placeholder(col) for col in columns
        )
        columns_sql = sql.SQL(", ").join(map(sql.Identifier, columns))
        update_assignments = sql.SQL(", ").join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
            for col in columns
            if col != "hf_model_id"
        )

        query = sql.SQL("""
            INSERT INTO hf_models ({columns})
            VALUES ({values})
            ON CONFLICT (hf_model_id) DO UPDATE SET {updates};
            -- 通常批量操作不使用 RETURNING 以提高效率
        """).format(
            columns=columns_sql, values=values_placeholders, updates=update_assignments
        )

        async with self.pool.connection() as conn:
            # 在单个事务中处理整个批次
            async with conn.transaction():
                for model_data in models_data:
                    model_id_log = model_data.get("hf_model_id", "UNKNOWN")
                    try:
                        # 准备参数字典，序列化 JSON 字段
                        params = model_data.copy()
                        if "tags" in params and isinstance(params["tags"], list):
                            try:
                                params["tags"] = json.dumps(params["tags"])
                            except TypeError:
                                self.logger.warning(
                                    f"无法序列化 tags (模型 {model_id_log}), 使用 null。"
                                )
                                params["tags"] = None

                        # 执行单条 UPSERT
                        await conn.execute(query, params)
                        processed_count += 1
                    except psycopg.Error as db_err:
                        self.logger.error(
                            f"Upsert 模型 {model_id_log} 时发生 DB 错误: {db_err}"
                        )
                        self.logger.error(f"失败的模型数据: {params}")  # 记录失败的数据
                        error_count += 1
                        # 可选择: 在事务中遇到错误是否继续？通常建议回滚，但此处仅记录
                    except Exception as e:
                        self.logger.error(
                            f"Upsert 模型 {model_id_log} 时发生意外错误: {e}"
                        )
                        self.logger.error(f"失败的模型数据: {params}")  # 记录失败的数据
                        error_count += 1

        self.logger.info(
            f"完成 HF 模型批量保存。处理: {processed_count}, 错误: {error_count}"
        )

    async def get_paper_details_by_pwc_id(
        self, pwc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        根据 PapersWithCode ID (`pwc_id`) 获取论文详情。
        假定 `pwc_id` 在 `papers` 表中是唯一或已索引的列。

        Args:
            pwc_id (str): 要查询的 PapersWithCode ID。

        Returns:
            Optional[Dict[str, Any]]: 如果找到，返回包含论文详情的字典，否则返回 None。
        """
        if not pwc_id:
            return None

        query = "SELECT * FROM papers WHERE pwc_id = %s;"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, (pwc_id,))
                    result = await cur.fetchone()
                    if result:
                        self.logger.debug(f"获取到 pwc_id {pwc_id} 的论文详情。")
                    else:
                        self.logger.debug(f"未找到 pwc_id 为 {pwc_id} 的论文。")
                    return result
        except psycopg.Error as db_err:
            self.logger.error(f"从 PG 按 pwc_id {pwc_id} 获取论文时出错: {db_err}")
            self.logger.debug(traceback.format_exc())
            return None
        except Exception as e:
            self.logger.error(f"按 pwc_id {pwc_id} 获取论文时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    async def fetch_data_cursor(
        self, query: str, params: Optional[tuple] = None, batch_size: int = 100
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        通用方法：使用服务器端游标以流式方式获取数据。
        适用于需要处理大量结果集而不想一次性加载到内存中的场景。
        注意: `batch_size` 参数当前未被 `stream()` 直接使用。

        Args:
            query (str): 要执行的 SQL SELECT 查询语句。
            params (Optional[tuple]): 查询参数（如果需要）。
            batch_size (int): (当前未使用) 每次迭代从游标获取的行数。

        Yields:
            AsyncGenerator[Dict[str, Any], None]: 异步生成器，每次产生一个包含行数据的字典。
        """
        try:
            async with self.pool.connection() as conn:
                # 在事务中执行以支持服务器端游标 (虽然 stream 不一定需要)
                async with conn.transaction():
                    # 使用 stream 获取异步迭代器
                    async for record in conn.cursor(row_factory=dict_row).stream(
                        query, params
                    ):
                        yield record  # 产生单条记录字典
        except psycopg.Error as db_err:
            self.logger.error(f"游标获取数据期间发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            # 异常会隐式终止生成器
        except Exception as e:
            self.logger.error(f"游标获取数据期间发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            # 异常会隐式终止生成器

    async def get_all_paper_ids_and_text(
        self,
        batch_size: int = 100,  # batch_size 当前未使用
    ) -> AsyncGenerator[Tuple[int, str], None]:
        """
        获取所有论文的 ID 和用于索引的文本（标题 + 摘要），使用服务器端游标流式处理。

        Args:
            batch_size (int): (当前未使用) 每批获取的论文数量。

        Yields:
            AsyncGenerator[Tuple[int, str], None]: 异步生成器，每次产生一个元组 (paper_id, text_to_index)。
        """
        # 查询选择 paper_id 并连接 title 和 summary 作为索引文本
        # COALESCE 处理 NULL 值，避免连接结果为 NULL
        query = sql.SQL("""
            SELECT
                paper_id,
                COALESCE(title, '') || ' ' || COALESCE(summary, '') AS text_to_index
            FROM papers
            ORDER BY paper_id; -- 按 ID 排序以获得确定性顺序
        """)
        processed_count = 0
        try:
            async with self.pool.connection() as conn:
                async with conn.transaction():
                    # 使用默认元组游标和 stream
                    async for record in conn.cursor().stream(query):
                        paper_id, text = record
                        # 确保文本非空且包含有效字符
                        if text and text.strip():
                            yield (paper_id, text)
                            processed_count += 1
                            # 每处理 1000 条记录打印一次进度日志
                            if processed_count % 1000 == 0:
                                self.logger.debug(
                                    f"已处理 {processed_count} 篇论文用于索引..."
                                )
                        else:
                            self.logger.debug(
                                f"跳过论文 {paper_id}，因为 text_to_index 为空。"
                            )
            self.logger.info(f"完成获取 {processed_count} 篇论文的文本。")
        except psycopg.Error as db_err:
            self.logger.error(f"获取论文文本用于索引时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"获取论文文本用于索引时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())

    async def close(self) -> None:
        """
        优雅地关闭 PostgreSQL 连接池。
        通常在应用生命周期结束时调用。
        """
        if self.pool:
            self.logger.info("正在关闭 PostgreSQL 连接池...")
            try:
                await self.pool.close()
                self.logger.info("PostgreSQL 连接池已关闭。")
            except Exception as e:
                self.logger.error(f"关闭 PostgreSQL 连接池时出错: {e}", exc_info=True)

    async def fetch_one(
        self, query: str, params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        执行一个预期最多返回一行的查询，并将该行作为字典返回。

        Args:
            query (str): 要执行的 SQL 查询。
            params (Optional[tuple]): 查询参数 (如果需要)。

        Returns:
            Optional[Dict[str, Any]]: 包含结果行的字典；如果未找到行或发生错误，则返回 None。
        """
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params)
                    result = await cur.fetchone()
                    return result
        except psycopg.Error as db_err:
            self.logger.error(f"fetch_one 执行数据库错误: {db_err}")
            self.logger.debug(f"查询: {query}, 参数: {params}")
            self.logger.debug(traceback.format_exc())
            return None
        except Exception as e:
            self.logger.error(f"fetch_one 执行意外错误: {e}")
            self.logger.debug(f"查询: {query}, 参数: {params}")
            self.logger.debug(traceback.format_exc())
            return None

    async def get_all_models_for_indexing(
        self,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        获取所有 Hugging Face 模型的 ID 和用于索引的文本表示，使用服务器端游标流式处理。
        文本表示可以包括模型 ID、pipeline tag 和标签。

        Yields:
            AsyncGenerator[Tuple[str, str], None]: 异步生成器，每次产生 (model_id, text_to_index) 元组。
        """
        # 构建索引文本：连接模型 ID, pipeline tag (如果存在), 和所有标签 (转换为文本)
        query = sql.SQL("""
            SELECT
                hf_model_id,
                hf_model_id || ' ' || COALESCE(hf_pipeline_tag, '') || ' ' || COALESCE(hf_tags::text, '') AS text_to_index
            FROM hf_models
            WHERE hf_model_id IS NOT NULL AND hf_model_id != '' -- 确保 ID 有效
            ORDER BY hf_model_id; -- 确定性顺序
        """)
        processed_count = 0
        try:
            async with self.pool.connection() as conn:
                async with conn.transaction():
                    async for record in conn.cursor().stream(query):
                        model_id, text = record
                        if text and text.strip():
                            yield (model_id, text)
                            processed_count += 1
                            if processed_count % 1000 == 0:
                                self.logger.debug(
                                    f"已处理 {processed_count} 个模型用于索引..."
                                )
                        else:
                            self.logger.debug(
                                f"跳过模型 {model_id}，因为 text_to_index 为空。"
                            )
            self.logger.info(f"完成获取 {processed_count} 个模型的文本。")
        except psycopg.Error as db_err:
            self.logger.error(f"获取模型文本用于索引时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"获取模型文本用于索引时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())

    async def upsert_paper(self, paper: Paper) -> Optional[int]:
        """
        插入或更新单篇论文记录，使用命名占位符。
        基于 Pydantic 的 `Paper` 模型输入数据。
        冲突检测基于 `paper_id`。

        Args:
            paper (Paper): 包含论文数据的 Pydantic `Paper` 实例。

        Returns:
            Optional[int]: 成功插入或更新的论文的 `paper_id`，失败则返回 None。
        """
        # 从 Pydantic 模型导出字典
        paper_dict = paper.model_dump(exclude_unset=False)

        # 验证 paper_id 是否存在，它是冲突键
        if "paper_id" not in paper_dict or paper_dict["paper_id"] is None:
            self.logger.error("无法 Upsert 论文: 'paper_id' 缺失或为 None。")
            return None

        # 准备参数字典，序列化 JSON 字段
        params = paper_dict.copy()
        for key in ["authors", "categories"]:
            if key in params and isinstance(params[key], list):
                try:
                    params[key] = json.dumps(params[key])
                except TypeError:
                    self.logger.warning(
                        f"无法序列化字段 '{key}' (论文 {params.get('paper_id')}), 使用 null。"
                    )
                    params[key] = None

        # 获取所有有效的列名
        valid_columns = list(params.keys())
        if not valid_columns:
            self.logger.error("无法 Upsert 论文: 没有有效的列。")
            return None

        # 构建 SQL 查询
        values_placeholders = sql.SQL(", ").join(
            sql.Placeholder(col) for col in valid_columns
        )
        columns_sql = sql.SQL(", ").join(map(sql.Identifier, valid_columns))
        update_assignments = sql.SQL(", ").join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
            for col in valid_columns
            if col != "paper_id"  # 不更新冲突键
        )

        query = sql.SQL("""
            INSERT INTO papers ({columns})
            VALUES ({values})
            ON CONFLICT (paper_id) DO UPDATE SET {updates}
            RETURNING paper_id; -- 返回被操作行的 ID
        """).format(
            columns=columns_sql, values=values_placeholders, updates=update_assignments
        )

        paper_id_log = params.get("paper_id", "UNKNOWN")
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # 直接传递参数字典
                    await cur.execute(query, params)
                    result = await cur.fetchone()  # 获取 RETURNING 的结果
                    if result and result[0] is not None:
                        # 确保返回的是整数 ID
                        returned_id = cast(int, result[0])
                        self.logger.debug(f"成功 Upsert 论文，ID: {returned_id}")
                        return returned_id
                    else:
                        self.logger.warning(
                            f"Upsert 论文 {paper_id_log} 未返回有效的 ID。"
                        )
                        return None
        except psycopg.Error as db_err:  # 捕获特定数据库错误
            self.logger.error(f"Upsert 论文 {paper_id_log} 时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return None
        except Exception as e:  # 捕获其他所有意外错误
            self.logger.error(f"Upsert 论文 {paper_id_log} 时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    # --- 获取关联数据的方法 --- #

    async def get_tasks_for_papers(self, paper_ids: List[int]) -> Dict[int, List[str]]:
        """
        获取指定论文列表关联的任务名称。

        Args:
            paper_ids (List[int]): 论文 ID 列表。

        Returns:
            Dict[int, List[str]]: 字典，键为 paper_id，值为任务名称列表。
        """
        if not paper_ids:
            return {}
        # 假设关联表为 paper_tasks(paper_id, task_id) 和 tasks(task_id, task_name)
        query = """
            SELECT pt.paper_id, t.task_name
            FROM paper_tasks pt
            JOIN tasks t ON pt.task_id = t.task_id
            WHERE pt.paper_id = ANY(%s);
        """
        results: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    async for record in cur:
                        paper_id, task_name = record
                        if paper_id in results and task_name:
                            results[paper_id].append(task_name)
            self.logger.debug(f"获取到 {len(paper_ids)} 篇论文的任务信息。")
            return results
        except psycopg.Error as db_err:
            self.logger.error(f"获取论文任务时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return results  # 返回可能不完整的结果
        except Exception as e:
            self.logger.error(f"获取论文任务时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return results

    async def get_datasets_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """
        获取指定论文列表关联的数据集名称。

        Args:
            paper_ids (List[int]): 论文 ID 列表。

        Returns:
            Dict[int, List[str]]: 字典，键为 paper_id，值为数据集名称列表。
        """
        if not paper_ids:
            return {}
        # 假设关联表为 paper_datasets(paper_id, dataset_id) 和 datasets(dataset_id, dataset_name)
        query = """
            SELECT pd.paper_id, d.dataset_name
            FROM paper_datasets pd
            JOIN datasets d ON pd.dataset_id = d.dataset_id
            WHERE pd.paper_id = ANY(%s);
        """
        results: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    async for record in cur:
                        paper_id, dataset_name = record
                        if paper_id in results and dataset_name:
                            results[paper_id].append(dataset_name)
            self.logger.debug(f"获取到 {len(paper_ids)} 篇论文的数据集信息。")
            return results
        except psycopg.Error as db_err:
            self.logger.error(f"获取论文数据集时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return results
        except Exception as e:
            self.logger.error(f"获取论文数据集时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return results

    async def get_repositories_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """
        获取指定论文列表关联的代码库 URL。

        Args:
            paper_ids (List[int]): 论文 ID 列表。

        Returns:
            Dict[int, List[str]]: 字典，键为 paper_id，值为代码库 URL 列表。
        """
        if not paper_ids:
            return {}
        # 假设关联表为 paper_repositories(paper_id, repo_url)
        query = """
            SELECT paper_id, repo_url
            FROM paper_repositories
            WHERE paper_id = ANY(%s);
        """
        results: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    async for record in cur:
                        paper_id, repo_url = record
                        if paper_id in results and repo_url:
                            results[paper_id].append(repo_url)
            self.logger.debug(f"获取到 {len(paper_ids)} 篇论文的代码库信息。")
            return results
        except psycopg.Error as db_err:
            self.logger.error(f"获取论文代码库时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return results
        except Exception as e:
            self.logger.error(f"获取论文代码库时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return results

    async def get_unique_paper_areas(self) -> List[str]:
        """
        获取 `papers` 表中所有不重复的领域 (`area`) 列表，按字母排序。

        Returns:
            List[str]: 唯一的领域名称列表；出错则返回空列表。
        """
        query = "SELECT DISTINCT area FROM papers WHERE area IS NOT NULL AND area != '' ORDER BY area;"
        areas: List[str] = []
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    async for record in cur:
                        # 确保记录非空且是字符串
                        if record and isinstance(record[0], str):
                            areas.append(record[0])
            self.logger.info(f"获取到 {len(areas)} 个唯一的论文领域。")
            return areas
        except psycopg.Error as db_err:
            self.logger.error(f"获取唯一论文领域时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return []
        except Exception as e:
            self.logger.error(f"获取唯一论文领域时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    async def get_methods_for_papers(
        self, paper_ids: List[int]
    ) -> Dict[int, List[str]]:
        """
        获取指定论文列表关联的方法名称。

        Args:
            paper_ids (List[int]): 论文 ID 列表。

        Returns:
            Dict[int, List[str]]: 字典，键为 paper_id，值为方法名称列表。
        """
        if not paper_ids:
            return {}
        # 假设关联表为 paper_methods(paper_id, method_id) 和 methods(method_id, method_name)
        query = """
            SELECT pm.paper_id, m.method_name
            FROM paper_methods pm
            JOIN methods m ON pm.method_id = m.method_id
            WHERE pm.paper_id = ANY(%s);
        """
        results: Dict[int, List[str]] = {pid: [] for pid in paper_ids}
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (paper_ids,))
                    async for record in cur:
                        paper_id, method_name = record
                        if paper_id in results and method_name:
                            results[paper_id].append(method_name)
            self.logger.debug(f"获取到 {len(paper_ids)} 篇论文的方法信息。")
            return results
        except psycopg.Error as db_err:
            self.logger.error(f"获取论文方法时发生数据库错误: {db_err}")
            self.logger.debug(traceback.format_exc())
            return results
        except Exception as e:
            self.logger.error(f"获取论文方法时发生意外错误: {e}")
            self.logger.debug(traceback.format_exc())
            return results
