# -*- coding: utf-8 -*-
"""
文件目的：服务层测试的共享 Fixtures (tests/services/conftest.py)

本文件为 `tests/services/` 目录下的所有测试用例定义共享的 pytest fixtures。
主要目的是创建和配置 `SearchService` 及其所有依赖项（文本嵌入器、各种数据仓库）的
模拟 (Mock) 对象。这允许服务层的测试在隔离的环境中进行，专注于验证服务本身的业务逻辑，
而无需与真实的数据库、Faiss 索引或嵌入模型交互。

Fixtures 定义：
- `mock_embedder`: 提供一个模拟的 `TextEmbedder` 实例。
- `mock_faiss_paper_repo`: 提供一个模拟的 `FaissRepository` 实例 (用于论文)。
- `mock_faiss_model_repo`: 提供一个模拟的 `FaissRepository` 实例 (用于模型)。
- `mock_neo4j_repo`: 提供一个模拟的 `Neo4jRepository` 实例。
- `mock_pg_repo`: 提供一个模拟的 `PostgresRepository` 实例，配置了模拟数据和方法行为。
- `search_service`: 组合上述所有模拟依赖项，创建一个配置好的 `SearchService` 实例供测试使用。

模拟数据：
- 定义了一些 MOCK_PAPER_*_DETAIL_DICT 字典，用作 `mock_pg_repo` 返回的模拟论文详情。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock # 导入模拟工具，AsyncMock 用于异步，MagicMock 用于同步或通用
import numpy as np # 导入 numpy 用于创建模拟向量
from typing import List, Tuple, Dict, Optional, cast, Any, Literal # 导入类型提示
from datetime import date # 导入 date 用于模拟数据

# 导入需要被模拟或实例化的实际类
from aigraphx.services.search_service import SearchService
from aigraphx.vectorization.embedder import TextEmbedder
from aigraphx.repositories.faiss_repo import FaissRepository
from aigraphx.repositories.postgres_repo import PostgresRepository
from aigraphx.repositories.neo4j_repo import Neo4jRepository

# --- 模拟数据定义 ---
# 这些字典模拟了从 PostgresRepository 获取的单篇论文的详细信息结构。
# 它们将被用于配置 mock_pg_repo fixture 的返回值。
MOCK_PAPER_1_DETAIL_DICT = {
    "paper_id": 1,
    "pwc_id": "pwc-1",
    "title": "Paper 1",
    "summary": "Summary 1",
    "authors": ["Auth 1"],
    "published_date": date(2023, 1, 15), # 使用 date 对象
    "pdf_url": "url1",
    "area": "CV",
}
MOCK_PAPER_2_DETAIL_DICT = {
    "paper_id": 2,
    "pwc_id": "pwc-2",
    "title": "Paper 2",
    "summary": "Summary 2",
    "authors": ["Auth 2"],
    "published_date": date(2023, 2, 10),
    "pdf_url": "url2",
    "area": "NLP",
}
MOCK_PAPER_3_DETAIL_DICT = {
    "paper_id": 3,
    "pwc_id": "pwc-3",
    "title": "Paper 3",
    "summary": "Summary 3",
    "authors": ["Auth 3"],
    "published_date": date(2023, 1, 5),
    "pdf_url": "url3",
    "area": "CV",
}
# 模拟用于关键字搜索测试的论文详情
MOCK_PAPER_KEY_1_DETAIL_DICT = {
    "paper_id": 101,
    "pwc_id": "pwc-key-1",
    "title": "Keyword Paper 1",
    "summary": "Keyword Summary 1",
    "authors": ["Key Auth 1"],
    "published_date": date(2023, 2, 1),
    "pdf_url": "url_key1",
    "area": "NLP",
}
MOCK_PAPER_KEY_2_DETAIL_DICT = {
    "paper_id": 102,
    "pwc_id": "pwc-key-2",
    "title": "Keyword Paper 2",
    "summary": "Keyword Summary 2",
    "authors": ["Key Auth 2"],
    "published_date": date(2023, 2, 2),
    "pdf_url": "url_key2",
    "area": "CV",
}
# 更多模拟论文详情
MOCK_PAPER_4_DETAIL_DICT = {
    "paper_id": 4,
    "pwc_id": "pwc-test-4",
    "title": "Paper Z",
    "summary": "Summary Z",
    "authors": ["Auth Z"],
    "published_date": date(2023, 7, 7),
    "pdf_url": "http://example.com/4",
    "area": "CV",
}
MOCK_PAPER_5_DETAIL_DICT = {
    "paper_id": 5,
    "pwc_id": "pwc-test-5",
    "title": "Paper Y",
    "summary": "Summary Y",
    "authors": ["Auth Y"],
    "published_date": date(2023, 8, 8),
    "pdf_url": "http://example.com/5",
    "area": "CV",
}

# --- Fixtures 定义 ---

@pytest.fixture
def mock_embedder() -> MagicMock:
    """
    Pytest Fixture: 提供一个模拟的 `TextEmbedder` 实例。

    配置：
    - `embed` 方法返回一个固定的 384 维 numpy 向量。
    - `embed_batch` 方法返回一个包含 10 个相同固定向量的 numpy 数组。

    用途：
    用于测试服务层中需要调用文本嵌入器的逻辑，而无需实际加载和运行嵌入模型。

    Returns:
        MagicMock: 配置好的 `TextEmbedder` 模拟对象。
    """
    # 使用 MagicMock 创建同步模拟对象，并指定 spec 确保接口一致性
    mock = MagicMock(spec=TextEmbedder)
    # 配置 embed 方法的返回值
    mock.embed.return_value = np.array([0.1] * 384, dtype=np.float32)
    # 配置 embed_batch 方法的返回值
    mock.embed_batch = MagicMock(
        return_value=np.array([[0.1] * 384] * 10, dtype=np.float32) # 返回 10x384 的数组
    )
    return mock


@pytest.fixture
def mock_faiss_paper_repo() -> MagicMock:
    """
    Pytest Fixture: 提供一个模拟的 `FaissRepository` 实例 (用于论文)。

    配置：
    - `search_similar` 方法返回一个固定的包含 (paper_id, score) 元组的列表。
    - `is_ready` 方法返回 True。
    - `id_type` 属性设置为 "int"。

    用途：
    用于测试服务层中需要与论文 Faiss 索引交互的逻辑（如语义搜索），提供可预测的搜索结果。

    Returns:
        MagicMock: 配置好的用于论文的 `FaissRepository` 模拟对象。
    """
    mock = MagicMock(spec=FaissRepository)
    # 配置 search_similar 方法返回预定义的 (paper_id, score) 列表
    mock.search_similar.return_value = [(1, 0.1), (3, 0.3), (2, 0.5)]
    # 配置 is_ready 方法总是返回 True
    mock.is_ready = MagicMock(return_value=True)
    # 设置 id_type 属性，服务层可能会检查这个属性
    mock.id_type = "int"
    return mock


@pytest.fixture
def mock_faiss_model_repo() -> MagicMock:
    """
    Pytest Fixture: 提供一个模拟的 `FaissRepository` 实例 (用于模型)。

    配置：
    - `search_similar` 方法返回一个固定的包含 (model_id, score) 元组的列表。
    - `is_ready` 方法返回 True。
    - `id_type` 属性设置为 "str"。

    用途：
    用于测试服务层中需要与模型 Faiss 索引交互的逻辑，提供可预测的搜索结果。

    Returns:
        MagicMock: 配置好的用于模型的 `FaissRepository` 模拟对象。
    """
    mock = MagicMock(spec=FaissRepository)
    # 配置 search_similar 方法返回预定义的 (model_id, score) 列表
    mock.search_similar.return_value = [("org/model-a", 0.2), ("another/model-b", 0.4)]
    # 配置 is_ready 方法总是返回 True
    mock.is_ready = MagicMock(return_value=True)
    # 设置 id_type 属性
    mock.id_type = "str"
    return mock


@pytest.fixture
def mock_neo4j_repo() -> MagicMock:
    """
    Pytest Fixture: 提供一个模拟的 `Neo4jRepository` 实例。

    配置：
    - 使用 `AsyncMock` 因为 Neo4j 仓库的方法通常是异步的。
    - **注意:** 此处没有为具体方法配置返回值。如果 `SearchService` 或其他被测服务
      需要调用 `Neo4jRepository` 的方法（例如 `get_related_nodes`），则需要在此处
      或在具体的测试用例中为这些方法配置 `return_value` 或 `side_effect`。

    用途：
    提供一个符合 `Neo4jRepository` 接口的模拟对象，即使当前服务层测试可能不直接
    调用其方法，也能满足依赖注入的要求。

    Returns:
        AsyncMock: 一个基础的 `Neo4jRepository` 异步模拟对象。
    """
    mock = AsyncMock(spec=Neo4jRepository)
    # 根据需要取消注释并配置具体方法的模拟行为
    # mock.search_nodes.return_value = []
    # mock.get_neighbors.return_value = []
    # mock.get_related_nodes.return_value = []
    return mock


@pytest.fixture
def mock_pg_repo() -> MagicMock:
    """
    Pytest Fixture: 提供一个模拟的 `PostgresRepository` 实例。

    配置：
    - 使用 `AsyncMock` 因为 PG 仓库的方法是异步的。
    - 内部定义了一个 `paper_details_map` 用于存储模拟的论文详情数据。
    - `get_papers_details_by_ids` 方法被模拟：根据传入的 `paper_ids` 从 `paper_details_map`
      查找数据，并可选地根据传入的 `scores` 字典添加 "score" 字段。
    - `search_papers_by_keyword` 方法被模拟：
        - 它接收与真实方法类似的参数（查询词、分页、过滤、排序）。
        - 它对内部的 `paper_details_map` 执行简单的、内存中的过滤和排序来模拟数据库搜索。
        - **重要:** 这个模拟的搜索逻辑非常简化，仅用于提供可预测的、结构正确的返回数据
          （字典列表和总数），不能完全替代真实数据库的全文搜索、复杂过滤和排序行为。
        - 返回包含模拟结果列表和模拟总数的元组。
    - `get_hf_models_by_ids` 方法被模拟：返回预定义的 HF 模型详情字典列表。
    - `search_models_by_keyword` 方法被模拟：默认返回空列表和 0。

    用途：
    为服务层测试提供一个行为可预测的 PG 仓库模拟，使得测试可以验证服务层如何处理
    从 PG 获取的数据（如论文详情列表、关键字搜索结果）。

    Returns:
        AsyncMock: 配置好的 `PostgresRepository` 异步模拟对象。
    """
    mock = AsyncMock(spec=PostgresRepository)

    # 内部存储模拟论文详情
    mock.paper_details_map = {
        1: MOCK_PAPER_1_DETAIL_DICT,
        2: MOCK_PAPER_2_DETAIL_DICT,
        3: MOCK_PAPER_3_DETAIL_DICT,
        4: MOCK_PAPER_4_DETAIL_DICT,
        5: MOCK_PAPER_5_DETAIL_DICT,
        101: MOCK_PAPER_KEY_1_DETAIL_DICT,
        102: MOCK_PAPER_KEY_2_DETAIL_DICT,
        999: {"paper_id": 999, "pwc_id": "pwc-err-1", "title": "Error Paper"}, # 用于测试错误处理
    }

    # --- 模拟 get_papers_details_by_ids ---
    async def mock_get_details_by_ids(
        paper_ids: List[int], scores: Optional[Dict[int, float]] = None # 可选的分数映射
    ) -> List[Dict[str, Any]]:
        """模拟根据 ID 列表获取论文详情。"""
        results = []
        for pid in paper_ids:
            if pid in mock.paper_details_map:
                detail = mock.paper_details_map[pid].copy() # 复制字典以防意外修改
                # 如果提供了分数映射，则添加到结果字典中
                if scores and pid in scores:
                    detail["score"] = scores[pid]
                results.append(detail)
        return results
    # 将模拟函数设置为 get_papers_details_by_ids 的 side_effect
    mock.get_papers_details_by_ids.side_effect = mock_get_details_by_ids

    # --- 模拟 search_papers_by_keyword ---
    async def mock_search_papers_keyword_return_dicts(
        query: str,
        skip: int = 0,
        limit: int = 10,
        # 兼容旧的 date_from/date_to 和新的 published_after/published_before
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        published_after: Optional[date] = None,
        published_before: Optional[date] = None,
        # 兼容旧的 area 和新的 filter_area
        area: Optional[str] = None,
        filter_area: Optional[str] = None,
        sort_by: Optional[
            Literal["published_date", "title", "paper_id"] # 限制有效的排序字段
        ] = "published_date", # 默认排序字段
        sort_order: Optional[Literal["asc", "desc"]] = "desc", # 默认排序顺序
    ) -> Tuple[List[Dict[str, Any]], int]: # 返回结果列表和总数
        """模拟关键字搜索，返回字典列表和总数。"""
        # 处理兼容的参数名
        final_date_from = date_from or published_after
        final_date_to = date_to or published_before
        final_area = area or filter_area # 使用 filter_area 优先

        # 获取所有模拟论文详情
        all_paper_details = list(mock.paper_details_map.values())

        # --- 模拟过滤 ---
        filtered_details = all_paper_details
        if final_date_from:
            filtered_details = [
                p for p in filtered_details
                if p.get("published_date") and p.get("published_date") >= final_date_from
            ]
        if final_date_to:
            filtered_details = [
                p for p in filtered_details
                if p.get("published_date") and p.get("published_date") <= final_date_to
            ]
        if final_area:
            # 注意：这里假设 filter_area 是单个字符串，如果需要支持列表，逻辑需修改
            filtered_details = [p for p in filtered_details if p.get("area") == final_area]
        # 模拟简单的关键词匹配（标题或摘要包含查询词）
        if query:
            filtered_details = [
                p for p in filtered_details
                if query.lower() in p.get("title", "").lower()
                or query.lower() in p.get("summary", "").lower()
            ]

        # --- 模拟排序 ---
        reverse = sort_order == "desc" # 判断升序还是降序
        # 确定有效的排序键，如果无效则回退到默认值
        sort_key = sort_by if sort_by in ["published_date", "title", "paper_id"] else "published_date"

        # 定义用于排序的 key 函数，处理 None 值和类型差异
        def get_key(paper_dict: Dict[str, Any]) -> Any:
            val = paper_dict.get(sort_key)
            # 为不同类型提供可比较的值
            if isinstance(val, date): return val
            if isinstance(val, str): return val.lower()
            if isinstance(val, int): return val
            # 处理 None 或其他类型，提供默认值以避免排序错误
            if sort_key == "published_date": return date.min
            if sort_key == "title": return ""
            if sort_key == "paper_id": return -1
            return None # 其他未知情况

        try:
            # 执行排序，使用 paper_id 作为第二排序键以确保稳定性
            filtered_details.sort(
                key=lambda p: (get_key(p), p.get("paper_id", -1)), reverse=reverse
            )
        except TypeError as e:
            print(f"Sorting error in mock: {e}") # 记录可能的排序错误

        # --- 模拟分页 ---
        total = len(filtered_details) # 计算过滤后的总数
        paginated = filtered_details[skip : skip + limit] # 应用分页切片

        # 返回分页后的结果列表和总数
        return paginated, total

    # 将模拟搜索函数设置为 search_papers_by_keyword 的 side_effect
    mock.search_papers_by_keyword.side_effect = mock_search_papers_keyword_return_dicts

    # --- 模拟 get_hf_models_by_ids ---
    async def mock_get_hf_details(model_ids: List[str]) -> List[Dict[str, Any]]:
        """模拟根据 ID 列表获取 HF 模型详情。"""
        # 预定义的模型详情映射
        model_details_map = {
            "org/model-kw1": {"model_id": "org/model-kw1", "author": "Org", "pipeline_tag": "text-generation"},
            "org/model-kw3": {"model_id": "org/model-kw3", "author": "Org", "pipeline_tag": "text-generation"},
            "another/model-kw2": {"model_id": "another/model-kw2", "author": "Another", "pipeline_tag": "image-classification"},
            "org/model1": {"model_id": "org/model1", "author": "org", "likes": 100, "last_modified": "2023-01-01", "tags": ["tag1"], "pipeline_tag": "text", "downloads": 1000, "library_name": "transformers"},
            "user/model3": {"model_id": "user/model3", "author": "user", "likes": 50, "last_modified": "2023-01-03", "tags": ["tag1", "tag2"], "pipeline_tag": "image", "downloads": 500, "library_name": "diffusers"},
            "org/model2": {"model_id": "org/model2", "author": "org", "likes": 200, "last_modified": "2023-01-02", "tags": ["tag3"], "pipeline_tag": "text", "downloads": 2000, "library_name": "transformers"},
        }
        # 返回存在于映射中的模型详情
        return cast(
            List[Dict[str, Any]], # cast 用于类型提示
            [model_details_map[mid] for mid in model_ids if mid in model_details_map]
        )
    # 设置 side_effect
    mock.get_hf_models_by_ids.side_effect = mock_get_hf_details

    # --- 配置 search_models_by_keyword 的默认行为 ---
    # 默认情况下，模拟的模型关键字搜索返回空结果
    mock.search_models_by_keyword.return_value = ([], 0)

    return mock


@pytest.fixture
def search_service(
    # 这个 fixture 依赖于之前定义的所有模拟依赖项 fixtures
    mock_embedder: MagicMock,
    mock_faiss_paper_repo: MagicMock,
    mock_faiss_model_repo: MagicMock,
    mock_pg_repo: MagicMock,
    mock_neo4j_repo: MagicMock,
) -> SearchService:
    """
    Pytest Fixture: 创建并返回一个 `SearchService` 实例，其所有依赖项都被替换为模拟对象。

    参数：
        mock_embedder: 来自 `mock_embedder` fixture 的模拟 TextEmbedder。
        mock_faiss_paper_repo: 来自 `mock_faiss_paper_repo` fixture 的模拟 FaissRepository (论文)。
        mock_faiss_model_repo: 来自 `mock_faiss_model_repo` fixture 的模拟 FaissRepository (模型)。
        mock_pg_repo: 来自 `mock_pg_repo` fixture 的模拟 PostgresRepository。
        mock_neo4j_repo: 来自 `mock_neo4j_repo` fixture 的模拟 Neo4jRepository。

    用途：
    为服务层测试提供一个完全隔离的 `SearchService` 实例。测试函数可以直接使用这个
    `search_service` 实例来调用服务方法，并验证其逻辑，而无需担心底层依赖的实际状态。

    Returns:
        SearchService: 一个依赖项已被模拟的 `SearchService` 实例。
    """
    # 使用所有模拟对象初始化 SearchService
    service = SearchService(
        embedder=mock_embedder,
        faiss_repo_papers=mock_faiss_paper_repo,
        faiss_repo_models=mock_faiss_model_repo,
        pg_repo=mock_pg_repo,
        neo4j_repo=mock_neo4j_repo,
    )
    return service