import logging
import os
import traceback
from typing import Dict, Callable, Any, Optional, Union, Awaitable
from functools import partial  # Import partial

# 导入并设置日志配置
from aigraphx.logging_config import setup_logging

# 初始化日志
logger = setup_logging()

# Import lifespan manager and settings
from aigraphx.core.db import lifespan
from aigraphx.core.config import settings  # Import the settings object

# Import the main v1 API router
from aigraphx.api.v1.api import api_router as api_v1_router

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.responses import Response as StarletteResponse  # 明确导入Starlette的Response
from fastapi.exceptions import RequestValidationError
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
import time
import json

# 准备具有设置的lifespan函数
lifespan_with_settings = partial(lifespan, settings=settings)

# 初始化FastAPI应用，附加lifespan上下文管理器
app = FastAPI(
    title="AIGraphX API",
    description="API for interacting with the AI Knowledge Graph.",
    version="1.0.0",
    lifespan=lifespan_with_settings,  # Attach the lifespan manager with settings
)

# CORS配置（根据前端需要调整origins）
origins = [
    "http://localhost",
    "http://localhost:3000",  # 前端端口示例
    "http://localhost:5173",  # Vite默认端口
    # 需要时添加其他源
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 详细日志中间件，记录请求和响应的完整信息
class DetailedLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = id(request)
        start_time = time.time()
        
        # 记录请求详情
        path = request.url.path
        method = request.method
        query_params = dict(request.query_params)
        client = f"{request.client.host}:{request.client.port}" if request.client else "Unknown"
        
        logger.debug(f"→ 请求开始 [{request_id}] {method} {path} 客户端: {client}")
        logger.debug(f"→ 查询参数 [{request_id}]: {query_params}")
        
        # 尝试记录请求体
        try:
            if method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    try:
                        # 尝试解析为JSON
                        body_text = body.decode('utf-8')
                        if body_text:
                            try:
                                json_body = json.loads(body_text)
                                logger.debug(f"→ 请求体 [{request_id}] (JSON): {json_body}")
                            except json.JSONDecodeError:
                                logger.debug(f"→ 请求体 [{request_id}] (原始): {body_text[:1000]}...")
                    except UnicodeDecodeError:
                        logger.debug(f"→ 请求体 [{request_id}]: <二进制数据，长度 {len(body)} 字节>")
        except Exception as e:
            logger.warning(f"无法读取请求体 [{request_id}]: {str(e)}")
        
        # 记录请求头
        headers = dict(request.headers)
        logger.debug(f"→ 请求头 [{request_id}]: {headers}")
        
        # 处理请求并捕获任何异常
        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            
            # 记录响应详情
            status_code = response.status_code
            response_headers = dict(response.headers)
            
            # 根据状态码使用不同的日志级别
            log_msg = f"← 响应完成 [{request_id}] {method} {path} - 状态码: {status_code} - 耗时: {process_time:.2f}ms"
            if status_code >= 500:
                logger.error(log_msg)
                logger.error(f"← 响应头 [{request_id}]: {response_headers}")
            elif status_code >= 400:
                logger.warning(log_msg)
                logger.warning(f"← 响应头 [{request_id}]: {response_headers}")
            else:
                logger.debug(log_msg)
                logger.debug(f"← 响应头 [{request_id}]: {response_headers}")
                
            return response
            
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            stack_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            
            logger.error(f"! 请求处理异常 [{request_id}] {method} {path} - 耗时: {process_time:.2f}ms")
            logger.error(f"! 异常信息 [{request_id}]: {error_msg}")
            logger.error(f"! 异常堆栈 [{request_id}]:\n{stack_trace}")
            
            # 重新抛出异常，由全局异常处理器处理
            raise

# 应用详细日志中间件
app.add_middleware(DetailedLoggingMiddleware)

# 全局异常处理器，捕获所有未处理的异常
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """处理所有未捕获的异常"""
    request_id = id(request)
    error_message = f"未处理的异常 [{request_id}]: {str(exc)}"
    error_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    
    # 使用logger.exception自动包含堆栈跟踪
    logger.exception(f"全局异常处理器捕获到异常 [{request_id}] {request.method} {request.url.path}")
    
    # 记录请求上下文信息以便调试
    try:
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path_params": dict(request.path_params),
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client": f"{request.client.host}:{request.client.port}" if request.client else "Unknown",
        }
        logger.error(f"请求上下文 [{request_id}]: {request_info}")
    except Exception as context_err:
        logger.error(f"记录请求上下文失败 [{request_id}]: {str(context_err)}")
    
    # 返回友好的错误响应
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误，请联系管理员",
            "request_id": str(request_id),
            "message": str(exc)
        }
    )

# 请求验证错误处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """处理请求验证错误"""
    request_id = id(request)
    errors = exc.errors()
    
    # 详细记录验证错误
    logger.error(f"请求验证错误 [{request_id}] {request.method} {request.url.path}")
    logger.error(f"验证错误详情 [{request_id}]: {errors}")
    
    # 记录请求上下文
    try:
        context = {
            "query_params": dict(request.query_params),
            "path_params": dict(request.path_params),
            "headers": dict(request.headers),
        }
        logger.error(f"请求上下文 [{request_id}]: {context}")
    except Exception as e:
        logger.error(f"记录请求上下文失败 [{request_id}]: {str(e)}")
    
    # 返回友好的错误响应
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数验证错误",
            "errors": errors,
            "request_id": str(request_id)
        }
    )

# 根端点
@app.get("/")
async def read_root() -> Dict[str, str]:
    """根端点，返回欢迎信息"""
    logger.debug("访问根端点")
    return {"message": "Welcome to AIGraphX API"}

# 健康检查端点
@app.get("/health", status_code=200, tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    基本健康检查端点。
    如果服务器正在运行，则返回200 OK和状态"ok"。
    """
    logger.debug("健康检查请求")
    return {"status": "ok"}

# 包含主v1路由器，前缀为/api/v1
app.include_router(api_v1_router, prefix="/api/v1")

# 运行配置（用于调试，可选）
if __name__ == "__main__":
    # 注意：这主要用于调试。生产/开发环境使用`uvicorn`命令。
    logger.info("使用uvicorn运行FastAPI应用（调试模式）...")
    uvicorn.run(
        "aigraphx.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,  # 启用开发环境的自动重新加载
        log_level="debug",
        access_log=True,
    )
