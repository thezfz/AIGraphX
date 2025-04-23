#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
enrich_existing_data.py - AIGraphX 数据丰富脚本

本脚本用于读取由 collect_data.py 生成的 JSON Lines (JSONL) 数据文件，
并尝试为其中的记录补充缺失的信息。主要目标是填充在初始收集中可能未能获取
或后续需要更新的字段。

主要功能:
1.  **读取现有数据:** 从指定的输入 JSONL 文件逐行读取数据记录。
2.  **检查缺失字段:** 检查每个记录中特定的字段是否缺失或为空，例如：
    *   Hugging Face 模型的 README 内容 (`hf_readme_content`)。
    *   与论文关联的 PWC 条目中的方法列表 (`methods`)。
    *   与论文关联的 PWC 条目中的会议信息 (`conference`)。
    *   PWC 代码库条目中的许可证 (`license`) 和主要编程语言 (`language`) 信息。
    *   (隐式) 从 HF 标签中提取数据集链接 (`hf_dataset_links`)，如果初始收集时逻辑不完善。
3.  **调用 API 补充信息:**
    *   如果 README 缺失，调用 Hugging Face Hub API (`hf_hub_download`) 获取。
    *   如果 PWC 方法列表缺失，调用 PWC API (`fetch_pwc_relation_list_enrich`) 获取。
    *   如果 PWC 会议信息缺失，调用 PWC API (`find_pwc_entry_by_arxiv_id_enrich`) 尝试获取（通过 ArXiv ID）。
    *   如果代码库的许可证或语言信息缺失，调用 GitHub API (`fetch_github_details_enrich`) 获取。
4.  **并发与重试:**
    *   使用 `asyncio` 和 `httpx` 实现高效的异步网络请求。
    *   复用 `collect_data.py` 中的 `tenacity` 重试配置 (`retry_config_http`) 处理 API 请求失败。
    *   复用 `collect_data.py` 中的速率限制器 (`hf_limiter`, `pwc_limiter`, `github_limiter`)。
5.  **更新与输出:**
    *   将补充了信息的记录（或未修改的记录）写入指定的输出 JSONL 文件。
6.  **检查点机制:**
    *   使用单独的检查点文件 (`data/enrich_checkpoint.txt`) 记录已处理的输入文件行号。
    *   允许脚本在中断后从上次停止的位置恢复处理，避免重复工作。
    *   支持通过命令行参数 `--reset` 强制从头开始处理。
7.  **配置与日志:**
    *   通过 `.env` 文件加载 API 密钥。
    *   提供独立的日志记录到控制台和文件 (`logs/enrich_data.log`)。

交互:
-   读取: 输入 JSONL 文件, `.env` 文件 (API 密钥), `data/enrich_checkpoint.txt` (检查点)。
-   写入: 输出 JSONL 文件, `data/enrich_checkpoint.txt` (更新检查点), `logs/enrich_data.log` (日志)。
-   外部调用: Hugging Face API, Papers with Code API, GitHub API (根据需要)。

运行方式:
`python scripts/enrich_existing_data.py [--input INPUT.jsonl] [--output OUTPUT.jsonl] [--reset]`
"""

# --- 标准库导入 ---
import asyncio  # 异步 I/O 框架
import os       # 操作系统交互，如路径操作、环境变量
import json     # JSON 数据处理
import logging  # 日志记录
import traceback # 异常堆栈跟踪
from typing import List, Dict, Any, Optional, Tuple, Set, TypedDict, Union # 类型提示
from datetime import datetime, timezone # 日期和时间处理
import sys      # 系统相关功能，如此处用于修改模块搜索路径和退出脚本
import re       # 正则表达式操作
import argparse # 命令行参数解析
from functools import partial  # 用于包装函数调用，特别是给 asyncio.to_thread 传递带关键字参数的函数

# --- 第三方库导入 (从 collect_data.py 复制相关部分) ---
from dotenv import load_dotenv  # 从 .env 文件加载环境变量
import httpx                    # 异步 HTTP 客户端
import tenacity                 # 重试逻辑库
from aiolimiter import AsyncLimiter # 异步速率限制器

# 导入 Hugging Face Hub 相关函数和异常类
from huggingface_hub import HfApi, hf_hub_download, ModelInfo
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# 调整 Python 模块搜索路径，以便能导入项目根目录下的其他模块 (如果需要)
# 获取当前脚本文件所在的目录 (__file__ 指向 enrich_existing_data.py)
# os.path.dirname(__file__) -> scripts 目录
# os.path.join(..., "..") -> 上一级目录，即 Backend 目录
# os.path.abspath(...) -> 获取 Backend 目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 检查项目根目录是否已在 Python 模块搜索路径 (sys.path) 中
if project_root not in sys.path:
    # 如果不在，则将其插入到搜索路径的开头 (index 0)，这样导入时会优先搜索此目录
    sys.path.insert(0, project_root)
    # 这通常是为了能够导入 aigraphx 包中的模块，但在此脚本中可能不是必需的，
    # 因为它主要复用了 collect_data.py 中的函数或逻辑。

# --- 配置加载 (与 collect_data.py 类似) ---
# 定位 .env 文件路径 (假设在 Backend 目录下)
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
# 加载环境变量
load_dotenv(dotenv_path=dotenv_path)

# 从环境变量获取 API 密钥
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")

# 定义默认的输入和输出文件路径
DEFAULT_INPUT_JSONL = "data/aigraphx_knowledge_data_v1.jsonl" # 假设这是 collect_data.py 的一个输出版本
DEFAULT_OUTPUT_JSONL = "data/aigraphx_knowledge_data_enriched.jsonl" # 丰富化后的数据输出文件
# 为丰富化过程定义单独的检查点文件
CHECKPOINT_FILE_ENRICH = "data/enrich_checkpoint.txt"
# 定义丰富化过程的检查点保存间隔（每处理多少行保存一次）
CHECKPOINT_INTERVAL_ENRICH = 100

# 定义丰富化过程的日志文件路径
LOG_FILE_ENRICH = "logs/enrich_data.log"
# 获取日志文件所在目录
LOG_DIR_ENRICH = os.path.dirname(LOG_FILE_ENRICH)

# API 端点 (与 collect_data.py 相同)
PWC_BASE_URL = "https://paperswithcode.com/api/v1/"
GITHUB_API_BASE_URL = "https://api.github.com/"

# 并发与速率限制 (可以根据需要调整，这里与 collect_data.py 保持一致)
# 限制每个记录内部同时进行的丰富化操作（例如同时查 PWC 和 GitHub）的数量，
# 但主要限制还是由下面的 API 限速器控制。
MAX_CONCURRENT_REQUESTS = 5
# 复用 collect_data.py 中的速率限制器定义
hf_limiter = AsyncLimiter(5, 1.0)
pwc_limiter = AsyncLimiter(2, 1.0)
github_limiter = AsyncLimiter(1, 1.0)

# --- 日志记录设置 (与 collect_data.py 类似，但使用不同的日志文件和记录器名称) ---
# 确保日志目录存在
os.makedirs(LOG_DIR_ENRICH, exist_ok=True)
# 配置基础日志设置
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
# 获取一个名为 'enrich_script' 的特定记录器实例
logger = logging.getLogger("enrich_script")
# (可选) 移除根记录器的处理器
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
# 配置控制台输出处理器
stream_handler_enrich = logging.StreamHandler()
stream_handler_enrich.setLevel(logging.INFO)
stream_formatter_enrich = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
stream_handler_enrich.setFormatter(stream_formatter_enrich)
logger.addHandler(stream_handler_enrich)
# 配置文件输出处理器
try:
    file_handler_enrich = logging.FileHandler(
        LOG_FILE_ENRICH, mode="a", encoding="utf-8" # 使用追加模式 'a'
    )
    file_handler_enrich.setLevel(logging.DEBUG)
    file_formatter_enrich = logging.Formatter(
        # 文件日志包含更详细的信息
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] [%(funcName)s] %(message)s"
    )
    file_handler_enrich.setFormatter(file_formatter_enrich)
    logger.addHandler(file_handler_enrich)
except Exception as e:
    logger.error(f"设置文件日志记录到 {LOG_FILE_ENRICH} 失败: {e}")

logger.info("丰富化脚本日志记录已配置。")

# --- Tenacity 重试配置 (直接复制 collect_data.py 的配置) ---
# 定义可重试的网络错误
RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
# 定义可重试的 HTTP 状态码
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# 定义 HTTP 请求的重试配置字典
retry_config_http = dict(
    stop=tenacity.stop_after_attempt(4), # 最多尝试 4 次
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) # 指数退避
    + tenacity.wait_random(0, 2), # 加随机抖动
    retry=( # 重试条件：网络错误 或 特定状态码的 HTTPStatusError
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING), # 重试前记录日志
    reraise=True, # 重试失败后重新抛出异常
)

# --- API 客户端初始化 (共享客户端) ---
# 创建一个共享的 httpx 异步客户端实例，用于 PWC 和 GitHub 的请求
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(15.0, read=60.0), # 设置超时
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20), # 设置连接限制
    http2=True, # 尝试启用 HTTP/2
)
# 定义共享的 GitHub 请求头
github_headers = {}
if GITHUB_TOKEN:
    github_headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
else:
    github_headers = {"Accept": "application/vnd.github.v3+json"}
# 注意：Hugging Face API 的调用通常通过 HfApi 实例进行，不需要在这里单独初始化客户端。


# --- 辅助函数 (复用/改编自 collect_data.py) ---

# 复用获取 PWC 关联列表的函数
# 直接在装饰器中定义重试参数以修复 mypy 类型错误
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            # 添加 hasattr 检查以增强类型安全性
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and hasattr(e, "response") # 检查 response 属性是否存在
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def fetch_pwc_relation_list_enrich(
    pwc_paper_id: str, relation: str
) -> List[Dict[str, Any]]:
    """
    (丰富化版本) 获取指定 PWC 论文 ID 的关联信息列表 (例如 'methods')。
    与 collect_data.py 中的版本基本相同，日志信息稍作调整以区分。

    Args:
        pwc_paper_id: PWC 论文 ID。
        relation: 要获取的关联类型 ('repositories', 'tasks', 'datasets', 'methods')。

    Returns:
        包含关联项目字典的列表。失败或未找到则返回空列表。
    """
    # 函数体与 collect_data.py 中的 fetch_pwc_relation_list 相同，
    # 但日志使用了本脚本的 logger 实例 (logger)，并可能添加了 "[Enrich]" 前缀。
    all_results = []
    current_url: Optional[str] = f"{PWC_BASE_URL}papers/{pwc_paper_id}/{relation}/"
    page_num = 1
    while current_url:
        async with pwc_limiter: # 应用 PWC 速率限制
            logger.debug(
                f"[Enrich] 正在获取 PWC {relation} (论文ID: {pwc_paper_id}) 从 {current_url}"
            )
            try:
                response = await http_client.get(current_url)
                response.raise_for_status() # 检查 HTTP 错误
                data = response.json()
                results = []
                next_url: Optional[str] = None
                # 处理直接列表和分页字典两种情况
                if isinstance(data, list):
                    results = data
                elif (
                    isinstance(data, dict)
                    and "results" in data
                    and isinstance(data["results"], list)
                ):
                    results = data["results"]
                    next_url = data.get("next")
                    if next_url == current_url: # 防止死循环
                        next_url = None
                else:
                    # 格式错误
                    logger.warning(
                        f"[Enrich] PWC {relation} (论文ID: {pwc_paper_id}) 的数据格式异常。得到: {type(data)}"
                    )
                    break # 中断此关联类型的获取
                all_results.extend(results) # 追加当前页结果
                current_url = next_url # 更新 URL
                if current_url:
                    page_num += 1
            except httpx.HTTPStatusError as e:
                # 记录 HTTP 错误并重新抛出，让 tenacity 处理
                logger.warning(
                    f"[Enrich] 获取 PWC {relation} (论文ID: {pwc_paper_id}) 时发生 HTTP 错误: {e.response.status_code}"
                )
                raise e
            except Exception as e:
                # 记录其他错误并重新抛出，让 tenacity 处理
                logger.warning(
                    f"[Enrich] 获取/解析 PWC {relation} (论文ID: {pwc_paper_id}) 时出错: {e}"
                )
                raise e
    # 记录最终获取结果数量
    logger.debug(
        f"[Enrich] 完成获取 PWC {relation} (论文ID: {pwc_paper_id})。找到 {len(all_results)} 个结果。"
    )
    return all_results


# 复用获取 GitHub 详细信息的函数
# 直接在装饰器中定义重试参数以修复 mypy 类型错误
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            # 添加 hasattr 检查以增强类型安全性
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and hasattr(e, "response") # 检查 response 属性是否存在
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def fetch_github_details_enrich(
    repo_url: str, follow_redirects: bool = True, max_redirects: int = 3
) -> Optional[Dict[str, Any]]:
    """
    (丰富化版本) 获取给定 GitHub 代码库 URL 的星标数、语言和许可证信息。
    与 collect_data.py 中的版本基本相同，日志信息稍作调整。

    Args:
        repo_url: GitHub 代码库 URL。
        follow_redirects: 是否跟随重定向。
        max_redirects: 最大重定向次数。

    Returns:
        包含 'stars', 'language', 'license' 的字典，或 None。
    """
    # 函数体与 collect_data.py 中的 fetch_github_details 相同，
    # 只是日志使用了本脚本的 logger 并添加了 "[Enrich]" 前缀。
    if not GITHUB_TOKEN: # 检查 Token
        return None
    if not repo_url: # 检查 URL
        return None
    current_api_url_to_fetch = None
    original_url = repo_url
    redirect_count = 0
    # --- URL 解析逻辑 (同 collect_data.py) ---
    try:
        if "github.com" in repo_url.lower():
            clean_url = repo_url.replace("https://", "").replace("http://", "").strip("/")
            if clean_url.endswith(".git"): clean_url = clean_url[:-4]
            parts = clean_url.split("/")
            if len(parts) >= 3 and parts[0].lower() == "github.com":
                owner, repo_name = parts[1], parts[2]
                if owner and repo_name: current_api_url_to_fetch = f"{GITHUB_API_BASE_URL}repos/{owner}/{repo_name}"
                else: return None
            else: return None
        elif "api.github.com/repositories/" in repo_url.lower():
            current_api_url_to_fetch = repo_url
        else: return None
    except Exception as parse_error:
        logger.warning(f"[Enrich] 解析初始 GitHub URL {repo_url} 时出错: {parse_error}")
        return None
    # --- 请求与重定向循环 (同 collect_data.py) ---
    while current_api_url_to_fetch and redirect_count <= max_redirects:
        async with github_limiter: # 应用 GitHub 速率限制
            logger.debug(f"[Enrich] 正在获取 GitHub 详细信息从 {current_api_url_to_fetch}")
            try:
                response = await http_client.get(
                    current_api_url_to_fetch, headers=github_headers, follow_redirects=False
                )
                if response.status_code == 200: # 成功
                    data = response.json()
                    stars = data.get("stargazers_count")
                    language = data.get("language")
                    license_info = data.get("license")
                    license_name = license_info.get("spdx_id") if license_info and isinstance(license_info, dict) else None
                    logger.debug(f"[Enrich] 获取到 {data.get('full_name')} 的详细信息: S={stars}, L={language}, Lic={license_name}")
                    return {"stars": stars, "language": language, "license": license_name}
                # --- 重定向和错误处理 (同 collect_data.py) ---
                elif follow_redirects and response.status_code in (301, 302, 307, 308) and "location" in response.headers:
                    redirect_count += 1
                    current_api_url_to_fetch = response.headers["location"]
                    logger.info(f"[Enrich] GitHub 重定向 ({response.status_code}) 到: {current_api_url_to_fetch}")
                    continue
                elif response.status_code == 404: logger.info(f"[Enrich] GitHub API 404 for {current_api_url_to_fetch}"); return None
                elif response.status_code == 403: logger.error(f"[Enrich] GitHub API 403 for {current_api_url_to_fetch}. Rate limit/perms? Check token."); return None
                elif response.status_code == 401: logger.error(f"[Enrich] GitHub API 401 for {current_api_url_to_fetch}. Check GITHUB_API_KEY."); return None
                else: response.raise_for_status() # 让 tenacity 处理可重试错误
            except Exception as e:
                logger.error(f"[Enrich] 从 {current_api_url_to_fetch} 获取/解析 GitHub 详细信息时出错: {e}")
                raise e # 重新抛出让 tenacity 处理
    # 超过最大重定向次数
    if redirect_count > max_redirects:
        logger.warning(f"[Enrich] 原始 URL {original_url} 超出最大重定向次数 {max_redirects}")
    return None # 最终失败


# 复用通过 ArXiv ID 查找 PWC 条目的函数，主要用于获取会议信息
# 直接在装饰器中定义重试参数以修复 mypy 类型错误
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            # 添加 hasattr 检查以增强类型安全性
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and hasattr(e, "response") # 检查 response 属性是否存在
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def find_pwc_entry_by_arxiv_id_enrich(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    (丰富化版本) 通过 ArXiv ID 查询 PWC 论文条目摘要信息。
    主要用于补充缺失的会议信息。

    Args:
        arxiv_id: ArXiv ID (查询时会去除版本号)。

    Returns:
        PWC 条目摘要信息的字典，或 None。
    """
    # 函数体与 collect_data.py 中的 find_pwc_entry_by_arxiv_id 相同，
    # 日志使用了本脚本的 logger 并添加了 "[Enrich]" 前缀。
    arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id) # 去除版本号
    async with pwc_limiter: # 应用 PWC 速率限制
        url = f"{PWC_BASE_URL}papers/"
        params = {"arxiv_id": arxiv_id_base}
        logger.debug(f"[Enrich] 正在为 ArXiv ID {arxiv_id_base} 查询 PWC")
        try:
            response = await http_client.get(url, params=params)
            response.raise_for_status() # 检查 HTTP 错误
            data = response.json()
            count = data.get("count")
            # 检查是否找到唯一条目
            if isinstance(count, int) and count == 1 and data.get("results"):
                result_entry = data["results"][0]
                if isinstance(result_entry, dict):
                    # 返回包含会议信息的摘要字典
                    return result_entry
                else:
                    return None # 格式错误
            else:
                return None # 未找到或找到多个
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None # 404 是预期情况
            logger.warning(f"[Enrich] 查询 PWC 时发生 HTTP 错误 ({arxiv_id_base}): {e.response.status_code}")
            raise e # 让 tenacity 处理可重试错误
        except Exception as e:
            logger.error(f"[Enrich] 查询 PWC 时出错 ({arxiv_id_base}): {e}")
            raise e # 让 tenacity 处理


# 获取 README 内容的独立函数
async def fetch_readme_content_enrich(model_id: str) -> Optional[str]:
    """
    (丰富化版本) 异步获取指定 Hugging Face 模型的 README.md 文件内容。
    使用了 hf_hub_download 并在后台线程运行。

    Args:
        model_id: Hugging Face 模型 ID。

    Returns:
        README 文件内容字符串，如果未找到或获取失败则返回 None。
    """
    logger.debug(f"[Enrich] 尝试获取模型 {model_id} 的 README")
    try:
        # 使用 functools.partial 包装 hf_hub_download 调用，以便传递关键字参数
        # 这样 asyncio.to_thread 可以正确调用它
        readme_path = await asyncio.to_thread(
            partial(
                hf_hub_download,
                repo_id=model_id,       # 模型 ID
                filename="README.md",   # 要下载的文件名
                token=HF_API_TOKEN,     # API Token
                repo_type="model",      # 仓库类型
                cache_dir=None,         # 不使用缓存（或指定缓存目录）
            )
        )
        # 读取下载的文件内容
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"[Enrich] 成功获取模型 {model_id} 的 README")
        return content
    except HfHubHTTPError as e: # 捕获 HF HTTP 错误
        status_code = e.response.status_code if hasattr(e, "response") else 0
        if status_code == 404: # 文件未找到
            logger.info(f"[Enrich] 模型 {model_id} 未找到 README.md。")
        else: # 其他 HTTP 错误
            logger.warning(f"[Enrich] 获取模型 {model_id} README 时发生 HTTP 错误 {status_code}: {e}")
        return None
    except Exception as e: # 捕获其他异常
        logger.warning(f"[Enrich] 获取/读取模型 {model_id} README 时出错: {e}")
        return None


# --- 丰富化过程的检查点管理 ---
def _save_checkpoint_enrich(line_count: int) -> None:
    """
    将已处理的行数保存到丰富化检查点文件。

    Args:
        line_count: 已成功处理并写入输出文件的输入文件行号。
    """
    try:
        # 确保存储检查点文件的目录存在
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_ENRICH), exist_ok=True)
        # 以写入模式打开检查点文件（会覆盖原有内容）
        with open(CHECKPOINT_FILE_ENRICH, "w") as f:
            # 将行数转换为字符串并写入文件
            f.write(str(line_count))
        # 记录调试日志
        logger.debug(f"[Enrich] 检查点已保存: 已处理 {line_count} 行。")
    except IOError as e:
        # 记录保存失败的错误
        logger.error(f"保存丰富化检查点失败: {e}")


def _load_checkpoint_enrich(reset_checkpoint: bool = False) -> int:
    """
    从丰富化检查点文件加载上次处理到的行号。

    Args:
        reset_checkpoint: 如果为 True，则在加载前删除现有的检查点文件，强制从头开始。

    Returns:
        上次成功处理的行号。如果检查点文件不存在、无效或被重置，则返回 0。
    """
    # 如果 reset_checkpoint 为 True 且检查点文件存在
    if reset_checkpoint and os.path.exists(CHECKPOINT_FILE_ENRICH):
        try:
            # 删除检查点文件
            os.remove(CHECKPOINT_FILE_ENRICH)
            logger.info("[Enrich] 已移除现有检查点。")
        except OSError as e:
            # 记录删除失败的错误
            logger.error(f"移除检查点失败: {e}")
    # 如果检查点文件不存在（或者已被删除）
    if not os.path.exists(CHECKPOINT_FILE_ENRICH):
        return 0 # 从第 0 行开始（即第一行）
    try:
        # 打开检查点文件读取行号
        with open(CHECKPOINT_FILE_ENRICH, "r") as f:
            # 读取文件内容，去除空白，转换为整数
            processed_count = int(f.read().strip())
        # 记录成功加载的检查点信息
        logger.info(f"[Enrich] 检查点已加载: 将从第 {processed_count + 1} 行开始处理。")
        return processed_count # 返回已处理的行号
    except (IOError, ValueError) as e: # 捕获读取错误或转换整数错误
        # 记录加载失败的信息，并返回 0 从头开始
        logger.error(f"加载丰富化检查点失败: {e}。将从第 0 行开始。")
        return 0


# --- 主要的丰富化逻辑 ---
async def enrich_record(record: Dict[str, Any]) -> bool:
    """
    对单个数据记录字典执行丰富化操作。
    会检查缺失字段并异步调用相应的 API 获取数据。

    Args:
        record: 从输入 JSONL 文件读取的单个记录字典。

    Returns:
        如果记录被修改（补充了信息），则返回 True；否则返回 False。
    """
    modified = False # 标记记录是否被修改
    # 获取记录中的模型 ID，如果不存在则无法进行丰富化
    model_id = record.get("hf_model_id")
    if not model_id:
        logger.warning("记录缺少 hf_model_id，无法进行丰富化。")
        return False

    # --- 初始化可能缺失的字段，确保它们存在且为列表类型 ---
    if "hf_dataset_links" not in record or not isinstance(record["hf_dataset_links"], list):
        record["hf_dataset_links"] = []
        modified = True # 如果字段原本不存在，标记为修改
    if "linked_papers" not in record or not isinstance(record["linked_papers"], list):
        record["linked_papers"] = []
        modified = True # 如果字段原本不存在，标记为修改


    # --- 从 hf_tags 提取数据集链接 (补充逻辑) ---
    # 这个逻辑可能在 collect_data.py 中已经存在，但在这里再次执行可以确保数据完整性
    # 或者处理早期 collect_data.py 版本生成的数据。
    hf_tags = record.get("hf_tags")
    if isinstance(hf_tags, list):
        initial_links_count = len(record["hf_dataset_links"]) # 记录初始链接数量
        # 使用集合存储现有链接以提高查找效率
        existing_links_set = set(record["hf_dataset_links"])

        # 遍历标签
        for tag in hf_tags:
            if isinstance(tag, str) and tag.startswith("dataset:"):
                # 提取数据集名称
                dataset_name = tag.split(":", 1)[1].strip()
                if dataset_name: # 确保名称不为空
                    # 构建数据集 URL
                    dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
                    # 如果 URL 不在现有链接集合中
                    if dataset_url not in existing_links_set:
                        record["hf_dataset_links"].append(dataset_url) # 添加到列表
                        existing_links_set.add(dataset_url) # 也添加到集合

        # 如果链接数量增加，标记为已修改
        if len(record["hf_dataset_links"]) > initial_links_count:
            modified = True
            logger.debug(
                f"[Enrich] 为 {model_id} 从 hf_tags 提取到数据集链接: {record['hf_dataset_links']}"
            )


    # --- 阶段 1: 收集顶层丰富化任务 (例如获取 README) ---
    hf_enrich_tasks = [] # 存储与 HF 模型本身相关的异步任务
    # 检查 README 内容是否缺失 (键不存在 或 值为 None)
    if record.get("hf_readme_content") is None:
        # 如果缺失，添加获取 README 的异步任务
        hf_enrich_tasks.append(fetch_readme_content_enrich(model_id))
    else:
        # 如果 README 已存在，添加一个"空"任务，它会立即完成并返回现有内容
        # 这使得后续 asyncio.gather 的索引保持一致
        # asyncio.sleep(0, result=...) 是一个技巧，创建一个立即完成并返回指定结果的协程
        hf_enrich_tasks.append(
            asyncio.sleep(0, result=record.get("hf_readme_content"))
        )

    # --- 阶段 2: 为每篇关联论文准备丰富化任务计划 ---
    # paper_enrichment_plan: 存储每个需要丰富化的论文及其对应的任务计划
    # 格式: [(paper_dict, task_dict), (paper_dict, task_dict), ...]
    #   paper_dict: 指向原始记录中论文信息字典的引用
    #   task_dict: 包含该论文需要执行的异步任务（或已存在的数据）
    paper_enrichment_plan = []

    # 遍历记录中的关联论文列表
    for paper in record.get("linked_papers", []):
        # 跳过无效的论文条目
        if not isinstance(paper, dict):
            continue
        # 获取论文关联的 PWC 条目信息
        pwc_entry = paper.get("pwc_entry")
        # 如果没有 PWC 条目，或者 PWC 条目格式不正确，则无法进行 PWC 相关丰富化
        if not pwc_entry or not isinstance(pwc_entry, dict):
            continue

        # 获取 PWC ID 和 ArXiv 基础 ID，用于后续查询
        pwc_id = pwc_entry.get("pwc_id")
        arxiv_id_base = paper.get("arxiv_id_base") # 需要 ArXiv ID 来查 PWC 会议信息

        # 初始化当前论文的任务字典
        tasks_for_this_paper: Dict[str, Any] = {
            "methods": None,           # 存储获取 PWC 方法的任务协程 或 已有的方法列表
            "conference_entry": None,  # 存储获取 PWC 摘要 (含会议) 的任务协程 或 None
            "repo_tasks": [],          # 存储获取 GitHub 仓库详细信息的任务协程列表
            "repos_to_update": [],     # 存储需要更新的原始仓库字典的引用列表
        }
        needs_paper_gather = False # 标记这篇论文是否需要执行异步任务

        # 3a. 检查并计划 PWC 方法 (Methods) 的获取
        # 如果 PWC ID 存在，且 'methods' 键不存在 或 其值为空列表/None
        if pwc_id and ("methods" not in pwc_entry or not pwc_entry.get("methods")):
            # 添加获取 PWC 方法列表的任务
            tasks_for_this_paper["methods"] = fetch_pwc_relation_list_enrich(pwc_id, "methods")
            needs_paper_gather = True # 标记需要异步执行
        else:
            # 如果方法已存在，直接存储现有数据
            tasks_for_this_paper["methods"] = pwc_entry.get("methods", [])

        # 3b. 检查并计划会议信息 (Conference) 的获取
        # 如果 PWC ID 存在，且 'conference' 键不存在
        if pwc_id and "conference" not in pwc_entry:
            # 需要 ArXiv ID 来反查 PWC 摘要信息以获取会议信息
            if arxiv_id_base:
                # 添加通过 ArXiv ID 查询 PWC 摘要信息的任务
                tasks_for_this_paper["conference_entry"] = find_pwc_entry_by_arxiv_id_enrich(arxiv_id_base)
                needs_paper_gather = True # 标记需要异步执行
            else:
                # 如果没有 ArXiv ID，无法查询会议信息
                logger.warning(f"[Enrich] 无法为 PWC ID {pwc_id} 获取会议信息，因为缺少 ArXiv ID。")
                tasks_for_this_paper["conference_entry"] = None # 存储 None 表示无法获取
        else:
            # 如果会议信息已存在，或无法获取，存储 None
            tasks_for_this_paper["conference_entry"] = None

        # 3c. 检查并计划代码库 (Repositories) 信息的获取 (许可证/语言)
        repo_tasks = []          # 存储 GitHub API 调用任务
        repos_to_update = []     # 存储需要更新的仓库字典引用
        # 遍历 PWC 条目中的代码库列表
        for repo in pwc_entry.get("repositories", []):
            # 跳过无效的代码库条目
            if not isinstance(repo, dict):
                continue
            repo_url = repo.get("url") # 获取 URL
            # 检查是否需要丰富化 (许可证 或 语言 缺失)
            needs_enrich = (
                "license" not in repo or repo.get("license") is None or
                "language" not in repo or repo.get("language") is None
            )

            # 如果是 GitHub URL 且需要丰富化
            if repo_url and "github.com" in repo_url.lower() and needs_enrich:
                # 添加获取 GitHub 详细信息的任务
                repo_tasks.append(fetch_github_details_enrich(repo_url))
                # 添加需要更新的仓库字典引用
                repos_to_update.append(repo)
                needs_paper_gather = True # 标记需要异步执行

        # 将代码库相关的任务和引用列表存入当前论文的任务字典
        tasks_for_this_paper["repo_tasks"] = repo_tasks
        tasks_for_this_paper["repos_to_update"] = repos_to_update

        # 如果这篇论文需要执行任何异步获取任务
        if needs_paper_gather:
            # 将论文引用和任务计划添加到总计划中
            paper_enrichment_plan.append((paper, tasks_for_this_paper))

    # --- 阶段 3: 并发执行所有收集到的任务 --- #
    # 执行与 HF 模型本身相关的任务 (目前只有 README 获取)
    hf_results = await asyncio.gather(*hf_enrich_tasks, return_exceptions=True)

    # 并发执行每篇论文的丰富化任务组
    # 外层 gather 针对不同的论文
    # 内层 gather 针对同一篇论文的 PWC 方法、PWC 会议、GitHub 仓库详情获取任务
    paper_results_list = await asyncio.gather(
        *[ # 解包列表，将每个内部 gather 任务作为参数传给外层 gather
            # 为每篇论文创建一个 gather 任务
            asyncio.gather(
                # 任务1: 获取 PWC 方法 (如果是协程则执行，否则用 sleep(0) 返回现有数据)
                plan[1]["methods"] if asyncio.iscoroutine(plan[1]["methods"]) else asyncio.sleep(0, result=plan[1]["methods"]),
                # 任务2: 获取 PWC 会议 (如果是协程则执行，否则用 sleep(0) 返回 None)
                plan[1]["conference_entry"] if asyncio.iscoroutine(plan[1]["conference_entry"]) else asyncio.sleep(0, result=None),
                # 任务3: gather 获取该论文所有需要更新的 GitHub 仓库详情
                asyncio.gather(*plan[1]["repo_tasks"], return_exceptions=True),
            )
            for plan in paper_enrichment_plan # 遍历每个论文的计划
        ],
        return_exceptions=True, # 外层 gather 也捕获异常
    )

    # --- 阶段 4: 处理所有任务的结果并更新原始记录字典 --- #

    # 处理 HF 任务的结果 (README)
    if hf_results: # 确保 hf_results 不是空列表
        readme_result = hf_results[0] # 获取第一个（也是唯一一个）任务的结果
        # 检查记录中 README 是否原本就缺失或为 None
        readme_was_missing_or_none = record.get("hf_readme_content") is None

        # 如果获取成功 (结果是字符串) 且 README 原本缺失
        if isinstance(readme_result, str) and readme_was_missing_or_none:
            # 更新记录中的 README 内容
            record["hf_readme_content"] = readme_result
            modified = True # 标记为已修改
            logger.info(f"[Enrich] 成功更新模型 {model_id} 的 README")
        # 如果尝试获取缺失的 README 但失败了 (结果不是字符串)
        elif readme_was_missing_or_none and not isinstance(readme_result, str):
            # 记录失败信息（区分异常和 API 返回 None）
            if isinstance(readme_result, Exception):
                logger.warning(f"[Enrich] 获取模型 {model_id} 的 HF README 失败。将设为 None。错误: {readme_result}")
            else: # fetch_readme_content_enrich 返回了 None (例如 404)
                logger.info(f"[Enrich] 未找到模型 {model_id} 的 README 或获取返回 None。将 hf_readme_content 设为 None。")
            # 显式将 README 设置为 None，表示已检查但无法获取
            record["hf_readme_content"] = None
            # 即使设置为 None，也认为是对记录的修改（因为键现在明确存在了）
            modified = True
        # 如果记录原本就有 README (readme_was_missing_or_none is False)
        # 并且 fetch placeholder 返回了 None (这种情况是当原始值为 null 时)
        # 则不需要做任何事，也不是修改。
        elif not readme_was_missing_or_none and record["hf_readme_content"] is None:
            pass
        # 其他情况：README 原本就存在且获取任务返回了相同内容，无需修改。

    # 处理每篇论文的丰富化结果
    # 检查计划和结果列表长度是否一致，防止出错
    if len(paper_results_list) != len(paper_enrichment_plan):
        logger.error("[Enrich] 论文丰富化计划与结果列表长度不匹配！")
    else:
        # 遍历每篇论文的结果
        for i, paper_overall_result in enumerate(paper_results_list):
            # 获取对应的原始论文字典引用和任务计划
            original_paper_dict, task_plan = paper_enrichment_plan[i]
            # 获取 PWC 条目字典的引用，用于更新
            pwc_entry_ref = original_paper_dict.get("pwc_entry")
            if not pwc_entry_ref: continue # 防御性检查

            # 如果获取这篇论文信息的整个 gather 失败了
            if isinstance(paper_overall_result, Exception):
                logger.warning(f"[Enrich] 收集论文 (PWC ID: {pwc_entry_ref.get('pwc_id')}) 结果时出错: {paper_overall_result}")
                continue

            # 检查内部结果结构是否正确（应该是包含3个元素的元组/列表）
            if not isinstance(paper_overall_result, (list, tuple)) or len(paper_overall_result) != 3:
                logger.error(f"[Enrich] 异常的内部结果结构: {paper_overall_result}")
                continue

            # 解包内部结果：PWC 方法结果, PWC 会议摘要结果, GitHub 仓库详情结果列表
            methods_res, conf_res, repo_res_list = paper_overall_result

            # 更新 PWC 方法
            # 检查是否真的执行了获取方法的任务 (而不是用的已有数据)
            if asyncio.iscoroutine(task_plan["methods"]):
                # 检查获取结果是否是列表
                if isinstance(methods_res, list):
                    # 提取方法名称列表
                    extracted_methods = [str(m.get("name")) for m in methods_res if m.get("name")]
                    # 更新 PWC 条目中的 'methods' 字段
                    pwc_entry_ref["methods"] = extracted_methods
                    # 如果提取到了方法，标记为修改
                    if extracted_methods: modified = True
                    logger.debug(f"[Enrich] 已更新 PWC ID {pwc_entry_ref.get('pwc_id')} 的方法")
                # 如果获取失败 (结果是异常)
                elif isinstance(methods_res, Exception):
                    logger.warning(f"[Enrich] 获取 PWC ID {pwc_entry_ref.get('pwc_id')} 的方法失败: {methods_res}")

            # 更新会议信息
            # 检查是否真的执行了获取会议信息的任务
            if asyncio.iscoroutine(task_plan["conference_entry"]):
                # 检查获取结果是否是字典 (PWC 摘要信息)
                if isinstance(conf_res, dict):
                    conf_name = conf_res.get("conference") # 提取会议名称
                    # 如果获取到了会议名称
                    if conf_name:
                        # 再次检查 PWC 条目中是否确实没有 'conference' 键 (防止重复写入)
                        if "conference" not in pwc_entry_ref:
                            pwc_entry_ref["conference"] = conf_name # 更新会议信息
                            modified = True # 标记为修改
                            logger.debug(f"[Enrich] 已更新 PWC ID {pwc_entry_ref.get('pwc_id')} 的会议信息")
                # 如果获取失败 (结果是异常)
                elif isinstance(conf_res, Exception):
                    logger.warning(f"[Enrich] 获取 PWC ID {pwc_entry_ref.get('pwc_id')} 的会议信息失败: {conf_res}")

            # 更新代码库信息 (许可证/语言)
            # 获取需要更新的仓库字典引用列表
            repos_to_update_list = task_plan.get("repos_to_update", [])
            # 检查 GitHub 仓库详情的 gather 结果是否是列表 (表示 gather 本身没出错)
            if isinstance(repo_res_list, list):
                # 检查结果列表长度是否与计划更新的仓库数量一致
                if len(repo_res_list) == len(repos_to_update_list):
                    # 遍历每个仓库的获取结果
                    for k, repo_detail_result in enumerate(repo_res_list):
                        # 获取对应的原始仓库字典引用
                        repo_dict_ref = repos_to_update_list[k]
                        # 如果获取成功 (结果是字典)
                        if isinstance(repo_detail_result, dict):
                            # 如果原始记录中 license 为 None 且获取到了新值
                            if repo_dict_ref.get("license") is None and repo_detail_result.get("license") is not None:
                                repo_dict_ref["license"] = repo_detail_result["license"]
                                modified = True # 标记为修改
                            # 如果原始记录中 language 为 None 且获取到了新值
                            if repo_dict_ref.get("language") is None and repo_detail_result.get("language") is not None:
                                repo_dict_ref["language"] = repo_detail_result["language"]
                                modified = True # 标记为修改
                        # 如果获取失败 (结果是异常)
                        elif isinstance(repo_detail_result, Exception):
                            logger.warning(f"[Enrich] 获取 GitHub 详细信息失败 ({repo_dict_ref.get('url')}): {repo_detail_result}")
                else:
                    # 仓库任务和结果数量不匹配
                    logger.error("[Enrich] 代码库任务与结果列表长度不匹配！")
            # 如果获取仓库详情的整个 gather 失败了
            elif isinstance(repo_res_list, Exception):
                logger.warning(f"[Enrich] 收集 PWC ID {pwc_entry_ref.get('pwc_id')} 的代码库详细信息时出错: {repo_res_list}")

    # 返回记录是否被修改的标志
    return modified


# 丰富化脚本的主异步函数
async def main_enrich(input_file: str, output_file: str, reset_checkpoint: bool) -> None:
    """
    丰富化脚本的主执行函数。
    负责打开文件、按行处理、调用 enrich_record、写入输出和管理检查点。

    Args:
        input_file: 输入 JSONL 文件路径。
        output_file: 输出 JSONL 文件路径。
        reset_checkpoint: 是否重置检查点。
    """
    logger.info(f"启动丰富化过程。")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"重置检查点: {reset_checkpoint}")

    # 加载检查点，获取起始处理行号
    start_line = _load_checkpoint_enrich(reset_checkpoint)
    # 初始化计数器
    processed_count = 0 # 本次运行实际处理的行数（跳过检查点之前的行）
    enriched_count = 0  # 本次运行修改（丰富化）的记录数
    error_count = 0     # 本次运行遇到的错误数
    # 记录最后成功处理并写入的输入文件行号，用于保存检查点
    last_line_num = start_line

    try:
        # 使用 'with' 语句同时打开输入文件和输出文件，确保文件会被正确关闭
        with (
            open(input_file, "r", encoding="utf-8") as infile, # 只读模式打开输入文件
            # 根据是否从头开始决定输出文件的打开模式 ('w' 写入/覆盖, 'a' 追加)
            open(output_file, "w" if start_line == 0 else "a", encoding="utf-8") as outfile,
        ):
            # 如果是以追加模式打开输出文件
            if start_line > 0:
                logger.info(f"将追加到现有输出文件: {output_file}")
                # 通常 'a' 模式会自动将写入位置定位到文件末尾，无需手动 seek

            current_line = 0 # 当前读取的输入文件行号（从 1 开始）
            # 逐行读取输入文件
            for line in infile:
                current_line += 1
                # 如果当前行号小于等于检查点记录的行号，则跳过
                if current_line <= start_line:
                    continue

                # 处理检查点之后的新行
                try:
                    # 解析 JSON 字符串为 Python 字典
                    record = json.loads(line)
                    # 检查解析结果是否为字典
                    if not isinstance(record, dict):
                        logger.warning(f"跳过第 {current_line} 行: 格式无效 (不是字典)。")
                        error_count += 1
                        continue # 继续处理下一行

                    logger.debug(f"正在处理第 {current_line} 行的记录 (HF ID: {record.get('hf_model_id')})")
                    # 调用核心丰富化函数处理记录
                    was_modified = await enrich_record(record)

                    # 将处理（可能已修改）后的记录写回输出文件
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_count += 1 # 增加已处理行数计数
                    # 更新最后成功处理的行号
                    last_line_num = current_line

                    # 如果记录被修改，增加丰富化计数
                    if was_modified:
                        enriched_count += 1

                    # 定期保存检查点
                    if processed_count % CHECKPOINT_INTERVAL_ENRICH == 0:
                        _save_checkpoint_enrich(last_line_num)

                # 捕获 JSON 解析错误
                except json.JSONDecodeError:
                    logger.error(f"跳过第 {current_line} 行: 无效的 JSON。")
                    error_count += 1
                # 捕获处理单个记录时发生的其他异常
                except Exception as e:
                    logger.error(
                        f"处理第 {current_line} 行的记录时出错: {e}",
                        exc_info=True, # 同时记录异常信息和堆栈跟踪
                    )
                    error_count += 1
                    # 在这里可以选择是停止脚本还是继续处理下一行，目前是继续

            # 循环结束后，保存最终的检查点
            # 仅当本次运行确实处理了新的行时才保存
            if last_line_num > start_line:
                _save_checkpoint_enrich(last_line_num)

    # 捕获文件未找到的错误
    except FileNotFoundError:
        logger.critical(f"输入文件未找到: {input_file}")
    # 捕获在文件读写或主循环中发生的其他未预料异常
    except Exception as e:
        logger.critical(f"丰富化过程中发生意外错误: {e}", exc_info=True)
    finally:
        # 确保共享的 HTTP 客户端被关闭
        if http_client and not http_client.is_closed:
            await http_client.aclose()
            logger.info("HTTP 客户端已关闭。")

        # 记录最终的总结信息
        logger.info("--- 丰富化过程结束 ---")
        logger.info(f"读取的行数 (检查点之后): {processed_count}")
        logger.info(f"被丰富化/修改的记录数: {enriched_count}")
        logger.info(f"遇到的错误数: {error_count}")
        logger.info(f"输出已写入: {output_file}")
        logger.info(f"最后处理的行号 (用于下次检查点): {last_line_num}")


# --- 脚本入口点 ---
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="使用缺失字段丰富现有的 AIGraphX 数据。")
    # 定义 --input 参数
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_JSONL, # 设置默认值
        help=f"输入 JSONL 文件路径 (默认: {DEFAULT_INPUT_JSONL})",
    )
    # 定义 --output 参数
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_JSONL, # 设置默认值
        help=f"输出丰富化后的 JSONL 文件路径 (默认: {DEFAULT_OUTPUT_JSONL})",
    )
    # 定义 --reset 参数 (开关类型)
    parser.add_argument(
        "--reset",
        action="store_true", # 表示此参数不需要值，出现即为 True
        help="忽略现有丰富化检查点，从输入文件的开头开始处理。",
    )
    # 解析命令行参数
    args = parser.parse_args()

    try:
        # 运行主异步函数，传入解析到的参数
        asyncio.run(main_enrich(args.input, args.output, args.reset))
    # 捕获键盘中断
    except KeyboardInterrupt:
        logger.info("用户中断了丰富化过程。")
    # 捕获其他未处理异常
    except Exception as e:
        logger.critical(f"脚本执行失败: {e}", exc_info=True)
        # 即使失败，也尝试关闭 HTTP 客户端
        try:
            if http_client and not http_client.is_closed:
                asyncio.run(http_client.aclose())
        except Exception:
            pass # 忽略关闭客户端时的错误
        # 以非零退出码退出
        sys.exit(1)
