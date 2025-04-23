# -*- coding: utf-8 -*-
"""
Graph Service - 图相关服务模块

这个文件定义了 `GraphService` 类，它扮演着业务逻辑层中处理与"图"相关的操作的角色。
这里的"图"主要指的是知识图谱中的实体（如论文、模型、任务、数据集）及其之间的关系。

主要功能:
1.  **获取论文详情**: 给定一个论文的 Papers with Code ID (pwc_id)，从不同的数据源（主要是 PostgreSQL 和 Neo4j）收集该论文的详细信息，并将它们整合成一个统一的响应对象。
2.  **获取论文的邻接图**: 给定一个论文的 pwc_id，从 Neo4j 图数据库中查询与该论文直接相关的其他实体（如相关的任务、数据集、方法等）以及它们之间的关系，返回一个表示这个局部图结构的数据。
3.  **获取模型详情**: 给定一个 Hugging Face 模型的 ID (model_id)，主要从 PostgreSQL 数据库中获取该模型的详细信息。
4.  **获取相关实体**: (新增) 提供一个更通用的方法，允许查询与给定起始节点通过特定关系类型连接的目标节点。

这个文件如何与其他部分协作:
- **依赖数据仓库 (`aigraphx.repositories.postgres_repo.PostgresRepository`, `aigraphx.repositories.neo4j_repo.Neo4jRepository`)**: `GraphService` 不直接访问数据库。它依赖于仓库层提供的类 (`PostgresRepository` 用于访问 PostgreSQL, `Neo4jRepository` 用于访问 Neo4j) 来执行实际的数据库查询操作。这种分层结构使得业务逻辑与数据访问细节解耦。
- **被 API 端点调用 (`aigraphx.api.v1.endpoints.graph.py`)**: FastAPI 的 API 端点（路由）会接收来自前端或用户的请求（例如请求获取某篇论文的详情），然后调用 `GraphService` 中相应的方法来处理这些请求。`GraphService` 处理完后将结果返回给 API 端点，端点再将其格式化并发送给客户端。
- **使用数据模型 (`aigraphx.models.graph`)**: `GraphService` 在处理数据和返回结果时，会使用在 `models` 目录下定义的 Pydantic 模型（如 `PaperDetailResponse`, `GraphData`, `HFModelDetail`）。这些模型定义了数据的结构和类型，有助于确保数据的一致性和正确性。
- **读取配置 (间接)**: 虽然 `GraphService` 不直接读取配置，但它依赖的仓库层 (`PostgresRepository`, `Neo4jRepository`) 在初始化时会需要数据库连接信息，这些信息通常来源于全局配置。
- **日志记录 (`logging`)**: 在执行操作和处理数据时，会使用 `logging` 模块记录重要的信息、警告或错误，便于追踪和调试。
"""

# 导入 logging 模块，用于记录日志
import logging

# 从 typing 模块导入类型提示，增强代码可读性和健壮性
# Optional: 表示一个值可以是指定类型或者 None
# List: 表示列表类型
# Dict: 表示字典类型
# Any: 表示可以是任何类型
# Literal: 表示一个变量只能是指定的几个字面值中的一个 (例如 Literal["IN", "OUT", "BOTH"])
from typing import Optional, List, Dict, Any, Literal

# 导入 json 模块，用于处理 JSON 格式的数据，例如将数据库中存储的 JSON 字符串解码为 Python 对象
import json

# 从 datetime 模块导入 date 类型，用于处理日期
from datetime import date

# 导入项目内部的其他模块和类
# 注意：这里使用了正确的相对路径或绝对路径来导入仓库和模型
# 从仓库层导入 PostgreSQL 数据库的访问类
from aigraphx.repositories.postgres_repo import PostgresRepository

# 从仓库层导入 Neo4j 图数据库的访问类
from aigraphx.repositories.neo4j_repo import (
    Neo4jRepository,
)

# 从数据模型层导入用于定义 API 响应结构和内部数据结构的 Pydantic 模型
from aigraphx.models.graph import (
    PaperDetailResponse,  # 用于封装论文详细信息的响应模型
    GraphData,  # 用于封装图数据（节点和边）的模型
    HFModelDetail,  # 用于封装 Hugging Face 模型详细信息的模型
)

# 获取当前模块的日志记录器实例
logger = logging.getLogger(__name__)


# 定义 GraphService 类
class GraphService:
    """
    提供用于检索图相关数据的服务。
    封装了与获取论文、模型及其关系图相关的业务逻辑。
    """

    # 初始化方法，创建 GraphService 实例时调用
    def __init__(
        self,
        pg_repo: PostgresRepository,  # 依赖注入：需要一个 PostgresRepository 实例
        neo4j_repo: Optional[
            Neo4jRepository
        ],  # 依赖注入：需要一个 Neo4jRepository 实例，但允许为 None
    ):
        """
        使用必要的依赖项（数据库仓库）初始化服务。

        Args:
            pg_repo (PostgresRepository): 用于访问 PostgreSQL 数据库的仓库实例。
            neo4j_repo (Optional[Neo4jRepository]): 用于访问 Neo4j 图数据库的仓库实例。
                                                   设置为 Optional 表示即使 Neo4j 不可用，服务也能初始化，
                                                   只是与图相关的功能会受限。
        """
        # 将传入的仓库实例赋值给服务实例的属性
        self.pg_repo = pg_repo
        self.neo4j_repo = neo4j_repo
        # 检查 Neo4j 仓库是否为 None
        if neo4j_repo is None:
            # 如果 Neo4j 仓库不可用，记录一条警告日志
            logger.warning("GraphService 初始化时没有 Neo4j 仓库。图相关功能将不可用。")

    # 定义一个异步方法 get_paper_details，用于获取论文详情
    async def get_paper_details(self, pwc_id: str) -> Optional[PaperDetailResponse]:
        """
        使用论文的 Papers with Code ID (pwc_id) 检索其详细信息。
        它会结合 PostgreSQL 中的基础数据和 Neo4j 中的关联信息（如果可用）。

        Args:
            pwc_id (str): 要查询的论文的 Papers with Code ID。

        Returns:
            Optional[PaperDetailResponse]: 包含论文详细信息的响应对象。
                                           如果找不到该论文或发生错误，则返回 None。
        """
        # 记录开始获取论文详情的日志
        logger.info(f"开始获取论文详情: {pwc_id}")

        # 1. 从 PostgreSQL 获取基础信息
        # 调用 PostgresRepository 的方法来根据 pwc_id 查询论文数据
        # 注意：这里使用了 `get_paper_details_by_pwc_id` 方法名，假设这是仓库中正确的方法
        paper_data_record = await self.pg_repo.get_paper_details_by_pwc_id(pwc_id)

        # 检查是否从 PostgreSQL 获取到了数据
        if not paper_data_record:
            # 如果没有找到记录，记录警告日志并返回 None
            logger.warning(f"在 PostgreSQL 中未找到 pwc_id 为 '{pwc_id}' 的论文。")
            return None

        # 将从数据库获取的记录（可能是只读的 Record 对象）转换为可变的字典
        # 这样我们后续可以方便地添加从 Neo4j 获取的信息
        paper_data = dict(paper_data_record)

        # 2. 从 Neo4j 获取关联实体（任务、数据集、方法）
        # 初始化用于存储关联实体名称的列表
        tasks_list: List[str] = []
        datasets_list: List[str] = []
        methods_list: List[str] = []

        # 检查 Neo4j 仓库是否可用
        if self.neo4j_repo:
            try:
                # 记录开始从 Neo4j 查询关联实体的调试日志
                logger.debug(f"开始从 Neo4j 获取论文 {pwc_id} 的关联实体...")

                # --- 获取关联的任务 (Tasks) ---
                # 调用 Neo4jRepository 的通用方法 get_related_nodes
                task_nodes = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",  # 起始节点的标签是 "Paper"
                    start_node_prop="pwc_id",  # 用于查找起始节点的属性是 "pwc_id"
                    start_node_val=pwc_id,  # 起始节点属性的值
                    relationship_type="HAS_TASK",  # 要遍历的关系类型（假设是 Paper -[HAS_TASK]-> Task）
                    target_node_label="Task",  # 目标节点的标签是 "Task"
                    direction="OUT",  # 关系方向是从 Paper 指向 Task (OUT)
                    limit=50,  # 限制返回最多 50 个任务，避免返回过多数据
                )
                # 从返回的节点列表中提取任务名称
                # node.get("properties", {}).get("name") 是一种安全的访问嵌套字典的方式
                # 它会处理 node 中没有 "properties" 键或 "properties" 中没有 "name" 键的情况，返回 None
                # `if node.get("properties", {}).get("name")` 确保只添加非空的名字
                tasks_list = [
                    node.get("properties", {}).get("name")
                    for node in task_nodes
                    if node.get("properties", {}).get("name")
                ]

                # --- 获取关联的数据集 (Datasets) ---
                # 逻辑与获取任务类似，只是关系类型和目标节点标签不同
                dataset_nodes = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="USES_DATASET",  # 假设关系是 Paper -[USES_DATASET]-> Dataset
                    target_node_label="Dataset",
                    direction="OUT",  # 假设方向是 Paper -> Dataset
                    limit=50,
                )
                datasets_list = [
                    node.get("properties", {}).get("name")
                    for node in dataset_nodes
                    if node.get("properties", {}).get("name")
                ]

                # --- 获取关联的方法 (Methods) ---
                # 逻辑与获取任务类似，只是关系类型和目标节点标签不同
                method_nodes = await self.neo4j_repo.get_related_nodes(
                    start_node_label="Paper",
                    start_node_prop="pwc_id",
                    start_node_val=pwc_id,
                    relationship_type="USES_METHOD",  # 假设关系是 Paper -[USES_METHOD]-> Method
                    target_node_label="Method",
                    direction="OUT",  # 假设方向是 Paper -> Method
                    limit=50,
                )
                methods_list = [
                    node.get("properties", {}).get("name")
                    for node in method_nodes
                    if node.get("properties", {}).get("name")
                ]

                # 记录从 Neo4j 获取到的关联实体的数量
                logger.debug(
                    f"成功从 Neo4j 获取论文 {pwc_id} 的关联信息: "
                    f"任务数={len(tasks_list)}, 数据集数={len(datasets_list)}, 方法数={len(methods_list)}"
                )

            # 捕获在与 Neo4j 交互过程中可能发生的任何异常
            except Exception as e:
                # 如果发生错误，记录错误日志（包括堆栈信息）
                logger.error(
                    f"从 Neo4j 获取论文 {pwc_id} 的关联实体失败: {e}",
                    exc_info=True,  # exc_info=True 会自动添加异常信息到日志
                )
                # 即使 Neo4j 查询失败，我们仍然可以继续处理来自 PostgreSQL 的数据
                # tasks_list, datasets_list, methods_list 会保持为空列表

        # 3. 将从 Neo4j 获取的列表（或空列表）更新到 paper_data 字典中
        paper_data["tasks"] = tasks_list
        paper_data["datasets"] = datasets_list
        paper_data["methods"] = methods_list

        # --- 在创建响应对象前，对关键字段进行验证 ---
        # 验证 pwc_id 是否存在且为字符串
        pwc_id_val = paper_data.get("pwc_id")
        if not pwc_id_val or not isinstance(pwc_id_val, str):
            # 如果 pwc_id 缺失或类型不正确，记录严重错误
            # 这通常表示数据源（PostgreSQL）中的数据存在问题
            logger.error(
                f"为标识符 {pwc_id} 返回的数据中缺少或包含无效的 pwc_id。找到的值: {pwc_id_val}"
            )
            # 根据业务需求，这里可以选择返回 None 让上层处理，或者抛出异常中断流程
            # 返回 None 可能更"优雅"，但可能隐藏数据问题
            # 抛出异常（如 ValueError）能更快地暴露问题
            # 目前选择返回 None
            return None
            # raise ValueError(f"为 {pwc_id} 获取的论文数据中缺少或包含无效的 pwc_id")

        # 4. 构建最终的响应模型对象 (PaperDetailResponse)
        # 需要处理 PostgreSQL 中可能以 JSON 字符串形式存储的字段（如作者、框架）
        # 定义一个内部辅助函数来解码这些字段
        def _decode_json_field(field_data: Any) -> list[Any]:
            """尝试将字段数据解码为列表。处理已经是列表或 JSON 字符串的情况。"""
            if isinstance(field_data, list):
                return list(field_data)
            elif isinstance(field_data, str):
                try:
                    return list(json.loads(field_data))
                except json.JSONDecodeError:
                    logger.warning(f"无法解码 JSON 字段: {field_data[:50]}...")
                    return []
            return []

        # --- 处理发布日期 (published_date) ---
        # 从 paper_data 获取原始的 published_date 值
        published_date_val = paper_data.get("published_date")
        published_date_obj: Optional[date] = None  # 初始化最终的 date 对象为 None
        # 如果原始值已经是 date 对象，直接使用
        if isinstance(published_date_val, date):
            published_date_obj = published_date_val
        # 如果原始值是字符串，尝试按 ISO 格式 (YYYY-MM-DD) 解析
        elif isinstance(published_date_val, str):
            try:
                published_date_obj = date.fromisoformat(published_date_val)
            except ValueError:
                # 如果字符串格式无效，记录警告，published_date_obj 保持 None
                logger.warning(f"published_date 的日期格式无效: {published_date_val}")
        # 如果原始值是其他类型 (如 None)，published_date_obj 保持 None

        # 使用准备好的数据创建 PaperDetailResponse Pydantic 模型实例
        # Pydantic 会自动进行类型检查和转换（如果可能）
        response = PaperDetailResponse(
            pwc_id=pwc_id_val,  # 使用验证过的 pwc_id
            title=paper_data.get("title"),  # 获取标题
            abstract=paper_data.get("summary"),  # 获取摘要 (字段名可能为 summary)
            arxiv_id=paper_data.get("arxiv_id_base"),  # 获取 arXiv ID (基础部分)
            url_abs=paper_data.get("pwc_url"),  # 获取 Papers with Code 页面 URL
            url_pdf=paper_data.get("pdf_url"),  # 获取 PDF 链接
            published_date=published_date_obj,  # 使用处理过的 date 对象
            authors=_decode_json_field(paper_data.get("authors")),  # 解码作者列表
            tasks=paper_data.get("tasks", []),  # 获取任务列表 (来自 Neo4j 或空列表)
            datasets=paper_data.get(
                "datasets", []
            ),  # 获取数据集列表 (来自 Neo4j 或空列表)
            methods=paper_data.get("methods", []),  # 获取方法列表 (来自 Neo4j 或空列表)
            frameworks=_decode_json_field(paper_data.get("frameworks")),  # 解码框架列表
            number_of_stars=paper_data.get("number_of_stars"),  # 获取 GitHub 星标数
            area=paper_data.get("area"),  # 获取研究领域
        )

        # 记录成功获取详情的日志，使用验证过的 ID
        logger.info(f"成功检索到论文的详细信息: {pwc_id_val}")
        # 返回构建好的响应对象
        return response

    # 定义一个异步方法 get_paper_graph，用于获取论文的邻接图数据
    async def get_paper_graph(self, pwc_id: str) -> Optional[GraphData]:
        """
        从 Neo4j 检索给定论文 ID 的邻接图（直接关联的节点和关系）。

        Args:
            pwc_id (str): 要查询其图邻域的论文的 Papers with Code ID。

        Returns:
            Optional[GraphData]: 一个包含节点和边列表的 GraphData 对象。
                                 如果 Neo4j 不可用、未找到图数据或解析数据时出错，则返回 None。
                                 如果在查询 Neo4j 时发生无法恢复的错误，则可能向上抛出异常。
        """
        # 记录开始获取图数据的日志
        logger.info(f"开始获取论文的图数据: {pwc_id}")

        # 检查 Neo4j 仓库是否可用
        if self.neo4j_repo is None:
            # 如果不可用，记录错误并返回 None
            logger.error("无法获取论文图数据：Neo4j 仓库不可用。")
            # 在 API 层，这可能应该转换为 HTTP 503 Service Unavailable 错误
            return None

        try:
            # 调用 Neo4jRepository 中的方法来获取图邻域数据
            # 假设仓库方法返回一个包含 'nodes' 和 'links' 键的字典
            graph_dict = await self.neo4j_repo.get_paper_neighborhood(pwc_id)

            # 检查是否成功获取到数据
            if not graph_dict:
                # 如果未找到数据，记录警告并返回 None
                logger.warning(f"在 Neo4j 中未找到论文 {pwc_id} 的图数据。")
                return None

            # --- 验证和解析从仓库返回的字典数据 ---
            try:
                # 尝试使用返回的字典 graph_dict 来创建 GraphData Pydantic 模型实例
                # Pydantic 会自动验证字典的结构和类型是否符合 GraphData 的定义
                # `**graph_dict` 是 Python 的解包语法，将字典的键值对作为关键字参数传递
                graph_data_model = GraphData(**graph_dict)
                # 如果成功创建模型实例，记录成功日志
                logger.info(f"成功解析论文 {pwc_id} 的图数据。")
                # 返回验证和解析后的 GraphData 对象
                return graph_data_model
            # 捕获在 Pydantic 验证或解析过程中可能发生的异常 (例如 ValidationError)
            except Exception as pydantic_error:
                # 如果验证失败，记录错误日志，包括 Pydantic 报告的错误信息
                logger.error(
                    f"无法验证来自 Neo4j 的论文 {pwc_id} 的图数据: {pydantic_error}"
                )
                # 同时记录从仓库获取到的原始数据，方便调试
                logger.debug(f"仓库返回的原始数据: {graph_dict}")
                # 决定如何处理验证失败：
                # 返回 None 比较简单，但可能隐藏 Neo4j 仓库返回数据结构的问题。
                # 抛出异常可以更快地暴露问题。
                # 这里选择返回 None，但标记了这是一个可以考虑修改的地方。
                return None

        # 捕获在调用 Neo4j 仓库方法过程中可能发生的其他异常（如连接错误）
        except Exception as e:
            # 如果发生异常，记录包含堆栈信息的错误日志
            logger.exception(f"从 Neo4j 获取论文 {pwc_id} 的图数据时出错: {e}")
            # 将异常重新抛出，让上层（通常是 API 端点）来处理这个错误
            # API 端点通常会将这类内部服务器错误转换为 HTTP 500 响应
            raise

    # 定义一个异步方法 get_model_details，用于获取 HF 模型详情
    async def get_model_details(self, model_id: str) -> Optional[HFModelDetail]:
        """
        从 PostgreSQL 检索给定 Hugging Face 模型 ID 的详细信息。
        这个方法目前不依赖 Neo4j。

        Args:
            model_id (str): 要查询的 Hugging Face 模型 ID (例如 "bert-base-uncased")。

        Returns:
            Optional[HFModelDetail]: 包含模型详细信息的 HFModelDetail 对象。
                                     如果找不到模型或发生错误，则返回 None。
        """

        # Define the helper function inside this method
        def _decode_json_field(field_data: Any) -> list[Any]:
            """尝试将字段数据解码为列表。处理已经是列表或 JSON 字符串的情况。"""
            if isinstance(field_data, list):
                return list(field_data)
            elif isinstance(field_data, str):
                try:
                    return list(json.loads(field_data))
                except json.JSONDecodeError:
                    logger.warning(f"无法解码 JSON 字段: {field_data[:50]}...")
                    return []
            return []

        # 这个操作不依赖 Neo4j，所以不需要检查 self.neo4j_repo
        logger.info(f"开始获取模型详情: {model_id}")
        try:
            # 调用 PostgresRepository 中的方法来根据模型 ID 列表获取模型数据
            # 即使我们只查一个 ID，仓库方法可能设计为接收列表，所以传入 [model_id]
            model_list = await self.pg_repo.get_hf_models_by_ids([model_id])
            # 检查返回的列表是否为空
            if not model_list:
                # 如果列表为空，说明数据库中没有这个模型的信息
                logger.warning(f"在 PostgreSQL 中未找到模型 {model_id} 的详细信息。")
                return None

            # 因为我们只查询了一个 ID，所以结果列表应该只包含一个元素（或为空）
            # 获取列表中的第一个元素，这应该是一个包含模型数据的字典
            model_data = model_list[0]

            # --- 在映射到 Pydantic 模型前，验证关键字段 ---
            # 获取从数据库返回的 hf_model_id
            retrieved_model_id = model_data.get("hf_model_id")
            # 检查 ID 是否存在且为字符串
            if not retrieved_model_id or not isinstance(retrieved_model_id, str):
                # 如果 ID 无效或缺失，记录严重错误
                logger.error(
                    f"为请求的模型 ID '{model_id}' 返回的数据中缺少或包含无效的 'hf_model_id'。找到的值: {retrieved_model_id}"
                )
                # 没有有效的 ID，无法创建详情对象，返回 None
                return None

            # --- 将数据库记录映射到 HFModelDetail Pydantic 模型 ---
            # Pydantic 会进行类型检查和转换
            # 注意：这里假设 HFModelDetail 的字段名与 model_data 字典中的键名匹配
            # 可能需要添加类似 _decode_json_field 的逻辑来处理 tags 或其他 JSON 存储的字段
            # 也需要处理日期/时间戳字段的转换
            # 例如，假设 last_modified 在数据库中是 datetime 对象
            last_modified_val = model_data.get("last_modified")
            # Pydantic 通常能自动处理 datetime 对象，但显式检查更安全
            if not isinstance(
                last_modified_val, (date, type(None))
            ):  # 允许 None 或 datetime.datetime
                # 如果类型不匹配，可能需要转换或记录警告
                logger.warning(
                    f"模型 {retrieved_model_id} 的 last_modified 字段类型非预期: {type(last_modified_val)}"
                )
                # 根据模型定义决定如何处理，这里假设 Pydantic 能处理或设为 None
                # last_modified_val = None # 或者尝试转换

            # 直接调用内部定义的函数，不需要 self.
            tags_list = _decode_json_field(model_data.get("tags"))

            model_details = HFModelDetail(
                model_id=retrieved_model_id,
                # 使用 .get() 提供默认值 None，防止因缺少键而报错
                author=model_data.get("author"),
                sha=model_data.get("sha"),
                last_modified=last_modified_val,
                tags=tags_list,  # 使用解码后的列表
                pipeline_tag=model_data.get("pipeline_tag"),
                downloads=model_data.get("downloads"),
                likes=model_data.get("likes"),
                library_name=model_data.get("library_name"),
                created_at=model_data.get("created_at"),
                updated_at=model_data.get("updated_at"),
            )

            # 记录成功获取模型详情的日志
            logger.info(f"成功检索到模型 {retrieved_model_id} 的详细信息。")
            # 返回创建好的模型详情对象
            return model_details

        # 捕获在与 PostgreSQL 交互过程中可能发生的异常
        except Exception as e:
            # 记录包含堆栈信息的错误日志
            logger.exception(f"获取模型 {model_id} 详细信息时出错: {e}")
            # 将异常重新抛出，由上层处理
            raise

    # 定义一个通用的异步方法 get_related_entities，用于获取相关节点
    async def get_related_entities(
        self,
        start_node_label: str,  # 起始节点的标签 (例如 "Paper", "Model")
        start_node_prop: str,  # 用于识别起始节点的属性名 (例如 "pwc_id", "hf_model_id")
        start_node_val: Any,  # 起始节点属性的值
        relationship_type: str,  # 要查询的关系类型 (例如 "HAS_TASK", "CITES")
        target_node_label: str,  # 目标节点的标签 (例如 "Task", "Paper")
        direction: Literal[
            "IN", "OUT", "BOTH"
        ] = "BOTH",  # 查询方向: "OUT" ->, "IN" <-, "BOTH" --
        limit: int = 25,  # 返回结果数量限制
    ) -> List[Dict[str, Any]]:
        """
        从 Neo4j 检索与给定起始节点通过特定关系连接的目标节点列表。
        这是一个更通用的图查询方法。

        Args:
            start_node_label (str): 起始节点的标签。
            start_node_prop (str): 用于查找起始节点的属性名称。
            start_node_val (Any): 起始节点属性的值。
            relationship_type (str): 要遍历的关系的类型。
            target_node_label (str): 目标节点的标签。
            direction (Literal["IN", "OUT", "BOTH"], optional): 关系的方向。
                                                               Defaults to "BOTH".
            limit (int, optional): 返回的最大节点数。Defaults to 25。

        Returns:
            List[Dict[str, Any]]: 目标节点属性字典的列表。
                                  如果 Neo4j 不可用或查询出错，则返回空列表。
                                  (注意：这里选择返回空列表而不是 None 或抛出异常，简化调用端的处理)
        """
        # 记录开始查询相关实体的日志
        logger.info(
            f"从 Neo4j 获取 '{start_node_label}' ({start_node_prop}={start_node_val}) "
            f"通过 '{relationship_type}' ({direction}) 连接的 '{target_node_label}' (limit={limit})"
        )

        # 检查 Neo4j 仓库是否可用
        if self.neo4j_repo is None:
            logger.error(f"无法获取相关实体：Neo4j 仓库不可用。")
            return []  # 返回空列表表示无法查询

        try:
            # 调用 Neo4jRepository 的 get_related_nodes 方法执行查询
            related_nodes = await self.neo4j_repo.get_related_nodes(
                start_node_label=start_node_label,
                start_node_prop=start_node_prop,
                start_node_val=start_node_val,
                relationship_type=relationship_type,
                target_node_label=target_node_label,
                direction=direction,
                limit=limit,
            )
            # 记录查询结果的数量
            logger.info(
                f"为 {start_node_val} 找到了 {len(related_nodes)} 个相关的 '{target_node_label}' 节点。"
            )
            # 返回查询到的节点列表（每个节点是一个包含其属性的字典）
            return related_nodes

        # 捕获查询过程中可能发生的异常
        except Exception as e:
            # 记录错误日志
            logger.exception(
                f"获取 '{start_node_label}' ({start_node_prop}={start_node_val}) 的相关实体时出错: {e}"
            )
            # 发生错误时返回空列表
            return []
