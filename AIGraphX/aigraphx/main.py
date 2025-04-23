# -*- coding: utf-8 -*-
"""
AIGraphX 后端 FastAPI 应用入口文件

此文件定义了 FastAPI 应用实例、配置了中间件（CORS、日志）、
注册了 API 路由、设置了全局异常处理器，并包含了应用的生命周期管理（lifespan）。
它是整个后端服务的启动点。

主要职责:
1.  **初始化 FastAPI 应用**: 创建 `FastAPI` 实例，设置标题、描述、版本和 `lifespan`。
2.  **配置中间件**:
    *   `CORSMiddleware`: 处理跨域资源共享，允许指定来源的前端访问。
    *   `DetailedLoggingMiddleware`: 自定义中间件，用于详细记录每个请求和响应的信息，包括时间、路径、参数、头部、请求体（如果适用）和状态码。
3.  **注册 API 路由**: 将 `aigraphx.api.v1.api.api_router` 挂载到 `/api/v1` 路径下，将所有 v1 版本的 API 端点整合进来。
4.  **定义全局异常处理器**:
    *   `global_exception_handler`: 捕获所有未被特定处理器处理的 `Exception`，记录详细错误信息和堆栈，并返回 500 内部服务器错误响应。
    *   `validation_exception_handler`: 捕获 FastAPI 的 `RequestValidationError`，记录验证失败的详细信息，并返回 422 Unprocessable Entity 错误响应。
5.  **定义基础端点**:
    *   `/`: 根路径，返回欢迎信息。
    *   `/health`: 健康检查端点，用于监控服务状态。
6.  **集成 Lifespan 管理**: 使用 `aigraphx.core.db.lifespan` 函数管理应用的启动和关闭事件，例如初始化数据库连接池、加载模型等。通过 `functools.partial` 将配置 `settings` 传入 `lifespan`。
7.  **启动服务 (通过 uvicorn)**: 文件末尾的 `if __name__ == "__main__":` 块允许直接运行此文件以启动 Uvicorn 开发服务器。通常在生产环境中会使用 Gunicorn + Uvicorn worker 或其他 ASGI 服务器。

与其他文件的交互:
*   **`aigraphx.logging_config`**: 导入 `setup_logging` 函数来初始化和配置全局日志记录器。
*   **`aigraphx.core.db`**: 导入 `lifespan` 函数来管理应用生命周期中的资源（如数据库连接池）。
*   **`aigraphx.core.config`**: 导入 `settings` 对象，包含了从环境变量或 `.env` 文件加载的应用配置。
*   **`aigraphx.api.v1.api`**: 导入 `api_router` (通常命名为 `api_v1_router`)，包含了所有 v1 版本的 API 端点。
*   **`fastapi`**: 核心框架，用于创建应用、定义路由、处理请求和响应。
*   **`uvicorn`**: ASGI 服务器，用于运行 FastAPI 应用。
*   **`starlette`**: FastAPI 底层依赖的 ASGI 框架，提供中间件、请求/响应对象等基础功能。
"""

# 导入标准库
import logging  # 用于日志记录
import os  # 用于与操作系统交互，例如获取环境变量
import traceback  # 用于获取和格式化异常堆栈信息
from typing import Dict, Callable, Any, Optional, Union, Awaitable  # 类型提示支持
from functools import partial  # 用于创建偏函数，方便给函数预设参数
import time  # 用于获取时间戳，计算处理时间
import json  # 用于处理 JSON 数据

# 导入项目内部模块
# 从 aigraphx.logging_config 模块导入 setup_logging 函数，用于设置日志记录器
from aigraphx.logging_config import setup_logging

# 初始化并获取配置好的日志记录器实例
# logger 将在整个应用中使用，记录运行信息、调试信息和错误
setup_logging()  # 先调用配置函数
logger = logging.getLogger("aigraphx.main")  # 然后获取 logger

# 从 aigraphx.core.db 模块导入 lifespan 函数
# lifespan 是一个异步上下文管理器，负责在应用启动时初始化资源（如数据库连接池），
# 并在应用关闭时清理资源。
from aigraphx.core.db import lifespan

# 从 aigraphx.core.config 模块导入 settings 对象
# settings 对象包含了通过 Pydantic Settings 加载的应用配置，例如数据库连接信息、API 密钥等
from aigraphx.core.config import settings

# 从 aigraphx.api.v1.api 模块导入 api_router，并重命名为 api_v1_router
# api_v1_router 是一个 FastAPI APIRouter 实例，聚合了所有 v1 版本的 API 端点
from aigraphx.api.v1.api import api_router as api_v1_router

# 导入 FastAPI 框架和相关组件
from fastapi import FastAPI, Request  # FastAPI 应用类和请求对象
from fastapi.middleware.cors import CORSMiddleware  # CORS 中间件，处理跨域请求
from fastapi.responses import JSONResponse, Response  # JSON 响应类和通用响应类
from starlette.responses import (
    Response as StarletteResponse,  # 从 Starlette 明确导入 Response，避免与 FastAPI 的 Response 混淆
)
from fastapi.exceptions import RequestValidationError  # 请求体验证错误异常类
import uvicorn  # ASGI 服务器，用于运行 FastAPI 应用

# 导入 Starlette 中间件基类，用于自定义中间件
from starlette.middleware.base import BaseHTTPMiddleware

# --- 应用初始化与 Lifespan ---

# 使用 functools.partial 创建一个新的 lifespan 函数 `lifespan_with_settings`
# 这个新函数在调用原始的 `lifespan` 函数时，会自动传入 `settings` 参数。
# 这样做可以避免在创建 FastAPI 应用时直接传递复杂的 lambda 函数。
lifespan_with_settings = partial(lifespan, settings=settings)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="AIGraphX API",  # 应用标题，会显示在 API 文档中
    description="用于与 AI 知识图谱交互的 API。",  # 应用描述
    version="1.0.0",  # 应用版本号
    # 设置应用的生命周期管理器
    # 当应用启动时，会进入 lifespan_with_settings 的 __aenter__ 方法
    # 当应用关闭时，会进入 lifespan_with_settings 的 __aexit__ 方法
    lifespan=lifespan_with_settings,
)

# --- 中间件配置 ---

# 配置 CORS (跨域资源共享)
# 允许来自指定源 (origins) 的前端应用访问 API
origins = [
    "http://localhost",  # 允许本地访问
    "http://localhost:3000",  # 常见的前端开发服务器端口
    "http://localhost:5173",  # Vite 前端开发服务器默认端口
    # 如果有其他部署环境或前端地址，需要在这里添加
    # 在生产环境中，应该更严格地限制允许的源
    # settings.FRONTEND_URL # 也可以从配置中读取允许的源
]

# 将 CORSMiddleware 添加到应用中
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源列表
    allow_credentials=True,  # 是否允许携带 cookies
    allow_methods=["*"],  # 允许所有 HTTP 方法 (GET, POST, PUT, DELETE 等)
    allow_headers=["*"],  # 允许所有请求头
)


# 自定义中间件：详细日志记录
# 这个中间件会拦截所有请求，记录详细的请求信息和响应信息，便于调试和监控
class DetailedLoggingMiddleware(BaseHTTPMiddleware):
    # dispatch 方法是中间件的核心，处理每个请求
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        拦截请求，记录详细信息，调用后续处理程序，记录响应信息。

        Args:
            request (Request): FastAPI 请求对象。
            call_next (Callable[[Request], Awaitable[Response]]):
                一个可调用对象，用于将请求传递给后续的处理程序（其他中间件或路径操作函数）。

        Returns:
            Response: FastAPI 响应对象。
        """
        # 为每个请求生成一个唯一的 ID (使用内存地址，简单但不保证全局唯一)
        request_id = id(request)
        # 记录请求开始时间
        start_time = time.time()

        # --- 记录请求信息 ---
        path = request.url.path  # 请求路径
        method = request.method  # 请求方法 (GET, POST, etc.)
        query_params = dict(request.query_params)  # 查询参数字典
        # 获取客户端 IP 和端口
        client = (
            f"{request.client.host}:{request.client.port}"
            if request.client
            else "Unknown"  # 如果无法获取客户端信息
        )

        # 使用 DEBUG 级别记录请求概要
        logger.debug(f"→ 请求开始 [{request_id}] {method} {path} 客户端: {client}")
        # 记录查询参数
        logger.debug(f"→ 查询参数 [{request_id}]: {query_params}")

        # 尝试记录请求体 (仅对 POST, PUT, PATCH 方法)
        request_body_text = None  # 用于后续可能的异常记录
        try:
            # 检查请求方法是否可能包含请求体
            if method in ["POST", "PUT", "PATCH"]:
                # 读取请求体内容 (注意：request.body() 只能读取一次)
                # 如果需要多次读取，需要先将 body 存储起来
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        # 尝试将请求体解码为 UTF-8 字符串
                        request_body_text = body_bytes.decode("utf-8")
                        if request_body_text:
                            try:
                                # 尝试将解码后的字符串解析为 JSON
                                json_body = json.loads(request_body_text)
                                logger.debug(
                                    f"→ 请求体 [{request_id}] (JSON): {json_body}"
                                )
                            except json.JSONDecodeError:
                                # 如果不是有效的 JSON，记录部分原始文本
                                logger.debug(
                                    f"→ 请求体 [{request_id}] (原始): {request_body_text[:1000]}{'...' if len(request_body_text) > 1000 else ''}"
                                )
                    except UnicodeDecodeError:
                        # 如果无法解码为 UTF-8 (可能是二进制数据)
                        logger.debug(
                            f"→ 请求体 [{request_id}]: <二进制数据，长度 {len(body_bytes)} 字节>"
                        )
                    # 将 body 重新包装，以便后续的处理程序可以读取
                    # Starlette/FastAPI 的 Request 对象在 body 被读取后需要特殊处理才能再次读取
                    # 这里通过创建一个新的 Request scope 来模拟，但更推荐的方式是
                    # 在中间件中读取 body 后存储在 request.state 中供后续使用，但这比较复杂
                    # 或者接受 request.body() 只能读一次的限制
                    # 此处简化，假设后续处理不需要再次读取原始 body
                    # 如果需要，参考 FastAPI 文档关于 request body 的处理
        except Exception as e:
            # 如果读取请求体时发生错误
            logger.warning(f"无法读取或记录请求体 [{request_id}]: {str(e)}")
            logger.debug(traceback.format_exc())  # 记录详细堆栈

        # 记录请求头
        headers = dict(request.headers)
        logger.debug(f"→ 请求头 [{request_id}]: {headers}")

        # --- 处理请求并获取响应 ---
        try:
            # 调用 call_next 将请求传递给下一个处理程序（可能是另一个中间件或最终的 API 端点）
            # await 等待响应返回
            response = await call_next(request)
            # 计算处理时间（毫秒）
            process_time = (time.time() - start_time) * 1000

            # --- 记录响应信息 ---
            status_code = response.status_code  # 响应状态码
            response_headers = dict(response.headers)  # 响应头字典

            # 根据状态码选择不同的日志级别
            log_msg = f"← 响应完成 [{request_id}] {method} {path} - 状态码: {status_code} - 耗时: {process_time:.2f}ms"
            if status_code >= 500:  # 服务器错误
                logger.error(log_msg)
                logger.error(f"← 响应头 [{request_id}]: {response_headers}")
                # 可以考虑记录响应体（如果适用且安全）
            elif status_code >= 400:  # 客户端错误
                logger.warning(log_msg)
                logger.warning(f"← 响应头 [{request_id}]: {response_headers}")
                # 可以考虑记录响应体
            else:  # 成功响应 (2xx, 3xx)
                logger.debug(log_msg)
                logger.debug(f"← 响应头 [{request_id}]: {response_headers}")

            # 返回原始响应
            return response

        # --- 处理请求过程中的异常 ---
        except Exception as e:
            # 如果在调用 call_next 或处理请求的过程中发生异常
            process_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            # 获取完整的异常堆栈信息
            stack_trace = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )

            # 使用 ERROR 级别记录异常信息
            logger.error(
                f"! 请求处理异常 [{request_id}] {method} {path} - 耗时: {process_time:.2f}ms"
            )
            logger.error(f"! 异常信息 [{request_id}]: {error_msg}")
            logger.error(f"! 异常堆栈 [{request_id}]:\\n{stack_trace}")

            # 注意：这里必须重新抛出异常！
            # 这样异常才能被 FastAPI 的异常处理器（如 global_exception_handler）捕获并处理，
            # 最终返回给客户端一个标准的错误响应。
            # 如果不重新抛出，客户端可能会收到一个不明确的连接错误。
            raise


# 将详细日志中间件添加到 FastAPI 应用中
# 中间件的执行顺序与添加顺序有关
app.add_middleware(DetailedLoggingMiddleware)


# --- 全局异常处理器 ---


# 注册一个全局异常处理器，用于捕获所有未被特定处理器处理的 Exception
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理所有未在路径操作函数或特定异常处理器中捕获的异常。
    记录详细错误信息，并返回一个标准的 500 JSON 错误响应。

    Args:
        request (Request): 发生异常时的请求对象。
        exc (Exception): 捕获到的异常对象。

    Returns:
        JSONResponse: 包含错误信息的 500 响应。
    """
    request_id = id(request)  # 获取请求 ID
    # 使用 logger.exception 可以自动记录异常类型、消息和堆栈跟踪
    logger.exception(
        f"全局异常处理器捕获到未处理异常 [{request_id}] "
        f"请求: {request.method} {request.url.path}"
    )

    # 尝试记录更多请求上下文信息，帮助调试
    try:
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path_params": dict(request.path_params),
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": f"{request.client.host}:{request.client.port}"
            if request.client
            else "Unknown",
            # 注意：不在此处记录 request_body，中间件已记录。
            # "request_body": request_body_text if 'request_body_text' in locals() else "<Body not read or error reading>"
        }
        logger.error(
            f"异常相关的请求上下文 [{request_id}]: {json.dumps(request_info, indent=2, default=str)}"
        )  # 使用 json.dumps 格式化输出
    except Exception as context_err:
        logger.error(f"记录异常请求上下文失败 [{request_id}]: {str(context_err)}")

    # 返回一个标准的 JSON 错误响应给客户端
    # status_code 设为 500 Internal Server Error
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部发生错误，请联系管理员或稍后重试。",  # 用户友好的错误信息
            "error_code": "INTERNAL_SERVER_ERROR",  # 可选的内部错误代码
            "request_id": str(request_id),  # 返回请求 ID，便于追踪日志
            # "message": str(exc), # 出于安全考虑，通常不直接暴露原始异常信息给最终用户
        },
    )


# 注册一个用于处理请求体验证错误的异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    处理由 Pydantic 模型验证失败或 FastAPI 路径/查询参数类型转换失败
    引发的 RequestValidationError。
    记录详细的验证错误，并返回一个标准的 422 JSON 错误响应。

    Args:
        request (Request): 发生验证错误的请求对象。
        exc (RequestValidationError): 捕获到的验证错误异常对象。

    Returns:
        JSONResponse: 包含验证错误详情的 422 响应。
    """
    request_id = id(request)  # 获取请求 ID
    # 获取 Pydantic 或 FastAPI 提供的详细错误信息列表
    # 每个错误包含位置 (loc)、错误消息 (msg) 和错误类型 (type)
    errors = exc.errors()

    # 使用 WARNING 或 ERROR 级别记录验证错误
    logger.warning(f"请求验证失败 [{request_id}] {request.method} {request.url.path}")
    # 记录详细的错误结构
    try:
        logger.warning(f"验证错误详情 [{request_id}]: {json.dumps(errors, indent=2)}")
    except TypeError:  # 处理无法序列化为 JSON 的情况
        logger.warning(f"验证错误详情 [{request_id}]: {errors}")

    # 尝试记录请求上下文
    try:
        context = {
            "query_params": dict(request.query_params),
            "path_params": dict(request.path_params),
            "headers": dict(request.headers),
            # 注意：不在此处记录 request_body，中间件已记录。
            # "request_body": request_body_text if 'request_body_text' in locals() else "<Body not read or error reading>"
        }
        logger.warning(
            f"验证错误相关的请求上下文 [{request_id}]: {json.dumps(context, indent=2, default=str)}"
        )
    except Exception as e:
        logger.warning(f"记录验证错误请求上下文失败 [{request_id}]: {str(e)}")

    # 返回一个标准的 JSON 错误响应给客户端
    # status_code 设为 422 Unprocessable Entity，表示服务器理解请求内容类型，
    # 并且请求语法正确，但无法处理包含的指令（因为验证失败）。
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数验证失败，请检查您的输入。",  # 用户友好的错误信息
            "errors": errors,  # 将详细的错误信息列表返回给客户端，方便前端调试
            "request_id": str(request_id),  # 返回请求 ID
        },
    )


# --- API 路由与基础端点 ---


# 定义根路径 ("/") 的 GET 请求处理函数
@app.get("/")
async def read_root() -> Dict[str, str]:
    """
    根端点，通常用于简单的连通性测试或返回应用的基本信息。
    """
    logger.debug("访问根端点 /")
    # 返回一个简单的 JSON 响应
    return {"message": "Welcome to AIGraphX API"}


# 定义健康检查路径 ("/health") 的 GET 请求处理函数
@app.get("/health", status_code=200, tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    提供一个基础的健康检查端点。
    如果应用能够正常响应此请求，通常表示 Web 服务器正在运行。
    更复杂的健康检查可能会检查数据库连接、外部服务可用性等。
    `tags=["Health"]` 用于在 OpenAPI 文档 (如 Swagger UI) 中对端点进行分组。
    """
    logger.debug("执行健康检查 /health")
    # 返回一个表示状态正常的 JSON 响应
    return {"status": "ok"}


# 包含 (挂载) v1 版本的 API 路由
# 所有定义在 api_v1_router 中的路径都会自动添加 `/api/v1` 前缀
# 例如，如果 api_v1_router 中定义了 `/search` 路径，
# 那么完整的访问路径将是 `/api/v1/search`
app.include_router(api_v1_router, prefix="/api/v1")


# --- 应用启动 (用于本地开发) ---

# 这个代码块只有在直接运行此 Python 文件时才会执行
# (例如，通过 `python aigraphx/main.py`)
# 在生产环境中，通常使用 ASGI 服务器命令行工具 (如 uvicorn 或 gunicorn) 来启动应用，
# 它们会导入 `app` 对象，而不是直接执行这个文件。
if __name__ == "__main__":
    # 记录启动信息
    logger.info(
        f"启动 Uvicorn 开发服务器 (配置: {getattr(settings, 'APP_ENV', 'unknown')})"
    )
    logger.info(
        f"访问地址: http://{getattr(settings, 'API_HOST', '0.0.0.0')}:{getattr(settings, 'API_PORT', 8000)}"
    )
    # 使用 uvicorn.run() 启动 ASGI 服务器
    uvicorn.run(
        "aigraphx.main:app",  # 指定要运行的应用实例 (模块路径:变量名)
        host=getattr(settings, "API_HOST", "0.0.0.0"),  # 使用 getattr 提供默认值
        port=getattr(settings, "API_PORT", 8000),  # 使用 getattr 提供默认值
        reload=getattr(
            settings, "API_RELOAD", True
        ),  # 使用 getattr 提供默认值 (开发时通常为 True)
        log_level=getattr(
            settings, "LOG_LEVEL", "info"
        ).lower(),  # 使用 getattr 提供默认值
        # 可以根据需要配置其他 uvicorn 参数
    )
