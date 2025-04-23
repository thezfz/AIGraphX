# -----------------------------------------------------------------------------
# 文件名: aigraphx/api/v1/endpoints/graph.py
#
# 描述:
# 这个文件定义了与“图谱”相关的 API 端点。这里的“图谱”主要指
# 构成知识图谱的实体（如论文、模型、任务、作者等）及其关系。
# 这些端点允许前端或其他客户端通过 HTTP 请求来获取特定实体
# （如论文、Hugging Face 模型）的详细信息，以及与某个实体相关的
# 图谱邻域数据（即直接关联的其他实体）。
#
# 主要功能:
# - 定义 FastAPI 路由 (`APIRouter`) 来组织这些端点。
# - 提供获取特定论文详细信息的端点。
# - 提供获取特定论文相关图谱数据的端点。
# - 提供获取特定 Hugging Face 模型详细信息的端点。
# - 提供一个通用的、根据起点、关系类型和方向获取相关实体的端点。
#
# 交互:
# - 路由 (`router`) 会被包含在 `aigraphx/api/v1/api.py` 中，
#   最终被整合到 `aigraphx/main.py` 的主 FastAPI 应用实例中。
# - 端点函数依赖于 `GraphService` (通过 FastAPI 的依赖注入系统 `Depends`)
#   来处理业务逻辑。
# - `GraphService` 实例由 `aigraphx/api/v1/dependencies.py` 中的
#   `get_graph_service` 函数提供，这个函数负责组装 `GraphService`
#   所需的仓库实例（如 `PostgresRepository`, `Neo4jRepository`）。
# - 端点函数使用 Pydantic 模型 (来自 `aigraphx/models/graph.py`)
#   来定义请求和响应的数据结构，确保数据格式的正确性和一致性。
# - 使用 Python 内置的 `logging` 模块记录请求和潜在的错误信息。
# - 使用 FastAPI 的 `HTTPException` 来返回标准的 HTTP 错误响应
#   （如 404 Not Found, 500 Internal Server Error）。
# -----------------------------------------------------------------------------

# 导入标准库
import logging  # 用于记录日志信息，方便追踪和调试
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Literal,
)  # 用于类型提示，增强代码可读性和健壮性
# Optional: 表示值可以是指定类型或 None
# List: 表示列表类型
# Dict: 表示字典类型
# Any: 表示可以是任何类型
# Literal: 表示值必须是给定的几个常量之一

# 导入 FastAPI 相关组件
from fastapi import APIRouter, Depends, HTTPException, Path, Query
# APIRouter: 用于创建一组相关的 API 路由/端点
# Depends: FastAPI 的依赖注入标记，用于声明和获取依赖项（如服务实例）
# HTTPException: 用于在请求处理中抛出标准的 HTTP 错误响应
# Path: 用于声明和验证路径参数 (URL 中 {} 部分的值)
# Query: 用于声明和验证查询参数 (URL 中 ? 后面的键值对)

# -----------------------------------------------------------------------------
# 移除之前错误的导入注释，这些依赖现在通过 GraphService 获取
# from aigraphx.core.db import get_postgres_repo
# from aigraphx.repositories.postgres_repo import PostgresRepository
# from aigraphx.repositories.neo4j_repo import Neo4jRepository
# -----------------------------------------------------------------------------

# 导入服务层 (Service Layer)
# 服务层封装了业务逻辑，协调不同的数据仓库来完成特定任务
from aigraphx.services.graph_service import GraphService

# 导入 API 数据模型 (Pydantic Models)
# 这些模型定义了 API 端点预期接收的数据格式（请求体验证）和
# 返回的数据格式（响应体序列化）。
from aigraphx.models.graph import PaperDetailResponse, GraphData, HFModelDetail

# -----------------------------------------------------------------------------
# --- 导入正确的依赖注入来源 ---
# 这是非常关键的一步！项目的依赖注入管理策略要求所有核心依赖项
# （如服务、仓库）的获取逻辑都集中在 `dependencies.py` 中。
# 这里导入 `dependencies` 模块，并使用别名 `deps`，以便后续通过
# `Depends(deps.some_dependency_provider)` 的方式来获取依赖。
# 这样做可以确保依赖来源的统一，便于维护和测试。
# -----------------------------------------------------------------------------
from aigraphx.api.v1 import dependencies as deps

# --- 路由设置 ---
# 创建一个新的 APIRouter 实例。这个 router 将包含本文件中定义的所有 API 端点。
# 后续会在 `api.py` 中将这个 router 包含到 V1 版本的 API 路由中。
router = APIRouter()
# 获取一个 logger 实例，用于记录本模块相关的日志信息。
# `__name__` 会将 logger 的名称设置为模块的完整路径（例如 aigraphx.api.v1.endpoints.graph），
# 这有助于在日志中区分不同模块的输出。
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# --- 移除本地/不正确的依赖函数 ---
# 之前的版本可能在这里直接定义了获取 GraphService 的函数，但这违反了
# 依赖注入集中管理的原则。正确的做法是使用上面导入的 `deps.get_graph_service`。
# async def get_graph_service(pg_repo: PostgresRepository = Depends(get_postgres_repo)) -> GraphService:
#     return GraphService(pg_repo=pg_repo)
# -----------------------------------------------------------------------------


# --- API 端点: 获取论文详情 ---
# 使用 `@router.get` 装饰器将下面的异步函数注册为一个处理 GET 请求的 API 端点。
# 路径为 "/papers/{pwc_id}"，其中 `{pwc_id}` 是一个路径参数。
# `response_model=PaperDetailResponse` 指定了响应体应该符合 `PaperDetailResponse` 模型。
# FastAPI 会自动使用这个模型来序列化函数的返回值，并进行数据验证。
@router.get("/papers/{pwc_id}", response_model=PaperDetailResponse)
async def get_paper_details_endpoint(
    # 定义路径参数 `pwc_id`。
    # `Path(...)` 用于为路径参数添加描述、验证规则等。
    # `...` 表示这个参数是必需的。
    # `description` 提供了参数的说明，会显示在 API 文档中。
    # `min_length=1` 要求 ID 至少包含一个字符。
    pwc_id: str = Path(..., description="要检索的论文的 PWC ID。", min_length=1),
    # --- 确保 Depends 使用正确的来源 ---
    # 使用 `Depends(deps.get_graph_service)` 来声明对此端点的依赖：
    # 它需要一个 `GraphService` 的实例。FastAPI 会自动调用 `deps.get_graph_service`
    # 函数来获取这个实例，并将其作为参数传递给 `get_paper_details_endpoint`。
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> PaperDetailResponse:  # 函数的返回值类型提示也应与 response_model 一致
    """
    通过 PWC ID 检索特定论文的详细信息。

    Args:
        pwc_id (str): 从 URL 路径中提取的论文 PWC ID。
        graph_service (GraphService): 通过依赖注入提供的 GraphService 实例。

    Returns:
        PaperDetailResponse: 包含论文详细信息的 Pydantic 模型对象。

    Raises:
        HTTPException:
            - 404 Not Found: 如果找不到具有指定 PWC ID 的论文。
            - 500 Internal Server Error: 如果在处理过程中发生意外错误。
    """
    # 记录日志，说明收到了一个请求以及请求的参数
    logger.info(f"收到获取论文详情的请求: pwc_id='{pwc_id}'")

    # 使用 try...except 块来捕获处理过程中可能发生的异常
    try:
        # 调用注入的 graph_service 实例的 get_paper_details 方法来执行业务逻辑
        # `await` 用于等待异步操作完成
        details = await graph_service.get_paper_details(pwc_id=pwc_id)

        # 检查服务层返回的结果
        if details is None:
            # 如果返回 None，表示没有找到对应的论文，抛出 404 错误
            # HTTPException 会被 FastAPI 捕获并转换成标准的 HTTP 响应
            raise HTTPException(
                status_code=404, detail=f"未找到 PWC ID 为 '{pwc_id}' 的论文。"
            )

        # 如果找到了论文详情，直接返回 Pydantic 模型对象
        # FastAPI 会自动将其序列化为 JSON 响应
        return details

    except HTTPException as http_exc:
        # 如果捕获到的是 HTTPException (例如上面自己抛出的 404)，
        # 直接重新抛出，让 FastAPI 处理。
        raise http_exc
    except Exception as e:  # 捕获除了 HTTPException 之外的其他所有潜在异常
        # 记录非预期的错误日志，包含异常信息和堆栈跟踪，方便调试
        logger.exception(f"获取论文 '{pwc_id}' 详情时发生意外错误: {e}")
        # 向上层抛出标准的 500 内部服务器错误
        # 向客户端隐藏具体的错误细节，只提示服务器内部错误
        raise HTTPException(
            status_code=500,
            detail=f"检索论文 '{pwc_id}' 详情时发生内部服务器错误。",
        )


# --- API 端点: 获取论文相关的图谱数据 ---
@router.get(
    "/papers/{pwc_id}/graph",  # 路径包含论文 ID
    response_model=GraphData,  # 响应模型定义了图谱数据的结构
    summary="获取论文的图谱邻域",  # API 文档中的简短摘要
    description="为给定的 PWC 论文 ID 检索图谱邻域（相关的论文、模型、概念等）。",  # API 文档中的详细描述
    tags=[
        "Graph"
    ],  # API 文档中的标签，用于分组端点（确保标签与 api.py 中 include_router 时使用的标签匹配或相关）
)
async def get_paper_graph_data(
    pwc_id: str = Path(..., description="要获取图谱数据的论文的 PWC ID。"),
    # 依赖注入 GraphService
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> GraphData:  # 返回类型是 GraphData 模型
    """
    获取指定论文的图谱邻域数据。

    Args:
        pwc_id (str): 论文的 PWC ID。
        graph_service (GraphService): 注入的 GraphService 实例。

    Returns:
        GraphData: 包含节点和边的图谱数据模型。

    Raises:
        HTTPException:
            - 404 Not Found: 如果找不到指定论文的图谱数据。
            - 500 Internal Server Error: 如果检索过程中发生内部错误。
    """
    logger.info(f"收到获取论文图谱数据的请求: {pwc_id}")
    try:
        # 调用服务层获取图谱数据
        graph_data = await graph_service.get_paper_graph(pwc_id)
        # 检查是否成功获取数据
        if graph_data is None:
            # 未找到数据，抛出 404
            raise HTTPException(
                status_code=404, detail=f"未找到论文 {pwc_id} 的图谱数据"
            )
        # 返回图谱数据
        return graph_data
    except HTTPException as http_exc:
        # 重新抛出已知的 HTTP 异常
        raise http_exc
    except Exception as e:
        # 记录未知错误
        logger.exception(f"检索论文 {pwc_id} 的图谱数据时出错: {e}")
        # 考虑区分“未找到”和“内部错误”可以提供更精确的错误信息，但目前统一返回 500
        raise HTTPException(
            status_code=500, detail="检索图谱数据时发生内部服务器错误。"
        )


# --- API 端点: 获取 Hugging Face 模型详情 ---
@router.get(
    # 路径参数 `model_id:path` 允许 model_id 包含斜杠 '/'，这对于 Hugging Face ID 是必要的
    "/models/{model_id:path}",
    response_model=HFModelDetail,  # 响应模型
    summary="获取 Hugging Face 模型详情",
    description="通过 ID 检索特定 Hugging Face 模型的详细信息。",
    tags=[
        "models",
        "Graph",
    ],  # 可以包含多个标签，例如按资源类型 ("models") 和功能区域 ("Graph")
    # 保持标签的一致性对于 API 文档的组织很重要
)
async def get_hf_model_details(
    model_id: str = Path(
        ..., description="Hugging Face 模型 ID (例如 'google/flan-t5-base')。"
    ),
    # 依赖注入 GraphService
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> HFModelDetail:  # 返回类型是 HFModelDetail 模型
    """
    获取指定 Hugging Face 模型的详细信息。

    Args:
        model_id (str): Hugging Face 模型 ID。
        graph_service (GraphService): 注入的 GraphService 实例。

    Returns:
        HFModelDetail: 包含模型详细信息的 Pydantic 模型。

    Raises:
        HTTPException:
            - 404 Not Found: 如果找不到指定 ID 的模型。
            - 500 Internal Server Error: 如果检索过程中发生内部错误。
    """
    logger.info(f"收到获取模型详情的请求: {model_id}")
    try:
        # 调用服务层获取模型详情
        model_details = await graph_service.get_model_details(model_id)
        # 检查结果
        if model_details is None:
            # 未找到，抛出 404
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {model_id} 的模型详情"
            )
        # 返回模型详情
        return model_details
    except HTTPException as http_exc:
        # 重新抛出已知 HTTP 异常
        raise http_exc
    except Exception as e:
        # 记录未知错误
        logger.exception(f"检索模型 {model_id} 详情时出错: {e}")
        # 抛出 500 错误
        raise HTTPException(
            status_code=500, detail="检索模型详情时发生内部服务器错误。"
        )


# --- 新增端点: 获取相关实体 ---
# 这个端点提供了一个更通用的方式来查询图谱中的连接关系。
@router.get(
    # 路径包含起始节点的标签、用于识别的属性名和属性值
    "/related/{start_node_label}/{start_node_prop}/{start_node_val}",
    # 响应模型暂时定义为一个字典列表，每个字典代表一个相关实体。
    # 未来可以定义更具体的 Pydantic 模型来规范响应结构。
    response_model=List[Dict[str, Any]],
    summary="获取相关实体",
    description="根据指定的关系类型和方向，检索与起始节点相关的实体。",
    tags=["Graph"],
)
async def get_related_entities_endpoint(
    # --- 路径参数 ---
    start_node_label: str = Path(
        ..., description="起始节点的标签 (例如 'Paper', 'Task')。"
    ),
    start_node_prop: str = Path(
        ...,
        description="用于识别起始节点的属性名称 (例如 'pwc_id', 'name')。",
    ),
    start_node_val: str = Path(..., description="起始节点标识属性的值。"),
    # --- 查询参数 ---
    relationship_type: str = Query(
        ...,  # 必需的查询参数
        description="要遍历的关系类型 (例如 'HAS_TASK', 'AUTHORED')。",
    ),
    target_node_label: str = Query(
        ...,  # 必需的查询参数
        description="要检索的目标节点的标签 (例如 'Paper', 'Author')。",
    ),
    # 使用 Literal 类型限制 direction 参数只能是 "IN", "OUT", 或 "BOTH"
    direction: Literal["IN", "OUT", "BOTH"] = Query(
        "BOTH",  # 默认值为 "BOTH"
        description="相对于起始节点的关系方向。",
    ),
    limit: int = Query(
        25,  # 默认返回最多 25 个实体
        description="要返回的最大相关实体数量。",
        ge=1,  # 必须大于等于 1 (greater than or equal to)
        le=100,  # 必须小于等于 100 (less than or equal to)
    ),
    # --- 依赖注入 ---
    graph_service: GraphService = Depends(deps.get_graph_service),
) -> List[Dict[str, Any]]:  # 返回值是一个字典列表
    """
    根据起始节点、关系类型、方向和目标标签，检索相关的实体列表。

    Args:
        start_node_label (str): 起始节点的标签。
        start_node_prop (str): 用于查找起始节点的属性名。
        start_node_val (str): 用于查找起始节点的属性值。
        relationship_type (str): 要追踪的关系类型。
        target_node_label (str): 目标节点的标签。
        direction (Literal["IN", "OUT", "BOTH"]): 关系的方向。
        limit (int): 返回结果的最大数量。
        graph_service (GraphService): 注入的 GraphService 实例。

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个找到的相关实体及其属性。
                              如果 Neo4j 不可用或查询出错，服务层会返回空列表。

    Raises:
        HTTPException:
            - 500 Internal Server Error: 如果发生意外的内部错误。
    """
    # 记录详细的请求参数
    logger.info(
        f"收到获取相关实体的请求: start={start_node_label}:{start_node_prop}={start_node_val}, "
        f"rel={relationship_type}, target={target_node_label}, dir={direction}, limit={limit}"
    )
    try:
        # 注意: 从路径参数获取的 start_node_val 始终是字符串类型。
        # 如果 Neo4j 中的属性是数字类型，Neo4jRepository 层可能需要进行类型转换。
        # 但对于常见的标识符如 pwc_id 或 name，字符串类型通常是合适的。
        related_entities = await graph_service.get_related_entities(
            start_node_label=start_node_label,
            start_node_prop=start_node_prop,
            start_node_val=start_node_val,
            relationship_type=relationship_type,
            target_node_label=target_node_label,
            direction=direction,
            limit=limit,
        )
        # 服务层在 Neo4j 不可用或仓库层出错时会返回空列表。
        # 在这种情况下，返回空列表是一个有效的响应，因此这里不需要抛出 404。
        return related_entities
    except HTTPException as http_exc:
        # 正常情况下，服务层应该处理仓库层的错误并返回空列表，
        # 所以这里不太可能捕获到由服务层主动抛出的 HTTPException。
        # 但以防万一，还是重新抛出。
        raise http_exc
    except Exception as e:
        # 记录任何未预料到的错误
        logger.exception(f"检索相关实体时出错: {e}")
        # 抛出 500 错误
        raise HTTPException(
            status_code=500, detail="检索相关实体时发生内部服务器错误。"
        )
