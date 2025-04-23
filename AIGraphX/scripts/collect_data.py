# -*- coding: utf-8 -*-
"""
collect_data.py - AIGraphX 数据采集脚本

本脚本负责从多个关键在线数据源收集 AI 模型及其相关论文的元数据，并将收集到的信息
结构化地保存到 JSON Lines (JSONL) 文件中，作为 AIGraphX 系统数据处理流程的起点。

主要功能:
1.  **连接外部 API:**
    *   Hugging Face Hub: 获取 AI 模型列表和详细信息 (包括标签、下载量、点赞数、README 内容等)。
    *   ArXiv API: 根据模型标签中的 ArXiv ID，获取对应论文的元数据 (标题、作者、摘要、分类等)。
    *   Papers with Code (PWC) API: 根据 ArXiv ID 查询 PWC 平台，获取相关的任务、数据集、代码库等信息。
    *   GitHub API: (需要配置 Token) 获取 PWC 中链接的 GitHub 代码库的星标数、主要编程语言和许可证信息。
2.  **数据提取与关联:**
    *   从 Hugging Face 模型的标签中提取 ArXiv ID。
    *   将 Hugging Face 模型信息与通过 ArXiv ID 找到的论文信息关联。
    *   将论文信息与通过 ArXiv ID 在 PWC 找到的信息关联。
    *   将 PWC 中的代码库信息与通过 GitHub API 获取的星标等信息关联。
    *   从模型 README 中提取指向 Hugging Face 数据集的链接。
3.  **数据处理与格式化:**
    *   处理 API 请求中的各种异常情况 (如网络错误、速率限制、资源未找到等)。
    *   使用 `tenacity` 库实现 API 请求的自动重试。
    *   使用 `aiolimiter` 库对各个 API 的请求进行速率限制，避免触发服务端的限制。
    *   使用 `asyncio` 和 `httpx` 实现高效的异步网络请求。
    *   将每个 Hugging Face 模型及其关联的所有信息整合成一个 JSON 对象。
4.  **持久化存储:**
    *   将处理好的 JSON 对象逐行写入指定的 JSONL 文件 (`data/aigraphx_knowledge_data.jsonl`)。
    *   记录已成功处理的 Hugging Face 模型 ID 到一个单独的文件 (`data/processed_hf_model_ids.txt`)，
        以便脚本中断后可以从上次停止的地方继续，避免重复处理。
5.  **配置与日志:**
    *   通过 `.env` 文件加载 API 密钥等敏感配置。
    *   提供详细的日志记录，同时输出到控制台 (INFO 级别) 和日志文件 (`logs/collect_data.log`, DEBUG 级别)。
    *   允许通过命令行参数 `--sort-by` (点赞/下载量) 和 `--limit` (数量) 控制从 Hugging Face 获取的模型范围。

交互:
-   读取: `.env` 文件 (获取 API 密钥), `data/processed_hf_model_ids.txt` (获取已处理列表)。
-   写入: `data/aigraphx_knowledge_data.jsonl` (输出数据), `data/processed_hf_model_ids.txt` (更新已处理列表), `logs/collect_data.log` (记录日志)。
-   外部调用: Hugging Face API, ArXiv API, Papers with Code API, GitHub API。

运行方式:
通常通过命令行直接运行: `python scripts/collect_data.py [--sort-by likes/downloads] [--limit N]`
脚本依赖于项目根目录下的 `.env` 文件来获取必要的 API 密钥。
"""

# --- 标准库导入 ---
import asyncio  # 导入 asyncio 库，用于支持异步 I/O 操作，是本脚本并发执行网络请求的基础。
import os  # 导入 os 库，用于与操作系统交互，例如文件路径操作、环境变量读取、目录创建等。
import json  # 导入 json 库，用于处理 JSON 数据格式的序列化（对象转字符串）和反序列化（字符串转对象）。
import logging  # 导入 logging 库，用于记录程序运行过程中的信息、警告和错误。
import traceback  # 导入 traceback 库，用于获取和格式化异常的堆栈跟踪信息，方便调试。
import sys  # 导入 sys 库，提供对 Python 解释器使用或维护的变量和函数的访问，如此处用于在关键错误时退出脚本。
import re  # 导入 re 库，用于支持正则表达式操作，如此处用于从文本中提取特定模式的 ID（如 ArXiv ID）。
import argparse  # 导入 argparse 库，用于解析命令行参数，使脚本可以通过命令行接收配置，如排序方式和数量限制。
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Set,
    TypedDict,
    Union,
)  # 从 typing 模块导入类型提示相关的类，用于增强代码可读性和健壮性，方便静态类型检查。
    # List: 列表类型
    # Dict: 字典类型
    # Any: 任意类型
    # Optional: 表示类型可以是指定类型或 None
    # Tuple: 元组类型
    # Set: 集合类型
    # TypedDict: 定义具有固定键和类型的字典结构
    # Union: 表示类型可以是多种指定类型中的一种
from datetime import datetime, timezone, timedelta  # 从 datetime 模块导入日期和时间相关的类。
    # datetime: 表示日期和时间的对象。
    # timezone: 表示时区信息，如此处使用 UTC。
    # timedelta: 表示两个日期或时间之间的时间差。

# --- 第三方库导入 ---
from dotenv import load_dotenv  # 从 python-dotenv 库导入 load_dotenv 函数，用于从 .env 文件加载环境变量。
import httpx  # 导入 httpx 库，一个现代化的、支持异步的 HTTP 客户端，用于发送网络请求。
import tenacity  # 导入 tenacity 库，提供强大的重试机制，用于在函数调用失败时自动重试。
from aiolimiter import AsyncLimiter  # 从 aiolimiter 库导入 AsyncLimiter 类，用于异步环境下的速率限制。
import arxiv  # 导入 arxiv 库，一个用于访问 ArXiv API 的 Python 封装库。 # type: ignore[import-untyped] 忽略类型检查警告，因为 arxiv 库可能没有提供类型存根。
from huggingface_hub import (
    HfApi,
    ModelInfo,
    hf_hub_download,
)  # 从 huggingface_hub 库导入与 Hugging Face Hub 交互的类和函数。
    # HfApi: 用于与 HF Hub API 交互的主要客户端类。
    # ModelInfo: 用于表示 HF Hub 上模型信息的类。
    # hf_hub_download: 用于从 HF Hub 下载文件的函数。
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
)  # 从 huggingface_hub.utils 导入特定的异常类，用于更精确地处理 API 错误。
    # RepositoryNotFoundError: 当尝试访问不存在的仓库时抛出。
    # GatedRepoError: 当尝试访问需要授权的仓库但未提供足够权限时抛出。
    # HfHubHTTPError: HF Hub API 请求返回 HTTP 错误时抛出。
import aiohttp  # 导入 aiohttp 库，另一个支持异步的 HTTP 客户端库 (在本脚本当前版本中似乎未使用，可能是早期版本遗留或备用)。
import bs4  # 导入 bs4 (Beautiful Soup 4) 库，用于解析 HTML 和 XML 文档，提取数据。 # type: ignore[import-untyped] 忽略类型检查警告。 (在本脚本当前版本中似乎未使用)

# --- 配置加载 ---
# 定位 .env 文件路径，假设它在脚本所在目录的上一级目录 (即 Backend 目录)
# __file__ 是当前脚本的文件名 (collect_data.py)
# os.path.dirname(__file__) 获取脚本所在的目录 (scripts)
# os.path.join(...) 将目录和文件名或上级目录符号 ".." 拼接成完整路径
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
# 加载 .env 文件中的环境变量。如果文件不存在或变量已存在于系统环境中，则不会覆盖。
load_dotenv(dotenv_path=dotenv_path)

# --- API 密钥配置 (关键!) ---
# 从环境变量中获取 Hugging Face API 密钥。如果未设置，HF_API_TOKEN 将为 None。
# 这个 Token 用于认证对 Hugging Face Hub 的 API 请求，特别是访问私有模型或提高速率限制。
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
# 从环境变量中获取 GitHub API 密钥。如果未设置，GITHUB_TOKEN 将为 None。
# 这个 Token 用于认证对 GitHub API 的请求，主要用于获取代码库的星标数等信息，可以提高速率限制。
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")

# 检查 GitHub Token 是否存在，如果不存在则发出警告。
if not GITHUB_TOKEN:
    logging.warning(
        "环境变量中未找到 GITHUB_API_KEY。获取 GitHub 星标数的功能将被禁用。"
    )

# --- 数据采集参数 ---
# MAX_MODELS_TO_PROCESS = 15000 # 原来的固定值，现在改为通过命令行参数控制。
# MODELS_SORT_BY = "downloads" # 原来的固定值，现在改为通过命令行参数控制。
# 定义输出 JSONL 文件的路径。所有收集到的数据将以 JSON Lines 格式写入此文件。
OUTPUT_JSONL_FILE = "data/aigraphx_knowledge_data.jsonl"
# CHECKPOINT_FILE = "data/collection_checkpoint.txt" # 旧的检查点文件，已被 PROCESSED_IDS_FILE 替代。
# CHECKPOINT_INTERVAL = 50 # 旧的检查点保存间隔，已被 PROCESSED_IDS_FILE 的逻辑替代。
# --- 添加 ID 跟踪文件 ---
# 定义用于存储已成功处理的 Hugging Face 模型 ID 的文件路径。
PROCESSED_IDS_FILE = (
    "data/processed_hf_model_ids.txt"  # 存储成功处理的模型 ID
)
# 定义每处理多少个 *新* 模型后，就将当前已处理的 ID 集合保存到文件的间隔。
SAVE_PROCESSED_IDS_INTERVAL = 100  # 每处理 100 个新模型保存一次已处理 ID 集合

# 定义日志文件的路径。
LOG_FILE = "logs/collect_data.log"
# 获取日志文件所在的目录路径。
LOG_DIR = os.path.dirname(LOG_FILE)

# --- API 端点配置 ---
# 定义 Papers with Code (PWC) API 的基础 URL。
PWC_BASE_URL = "https://paperswithcode.com/api/v1/"
# 定义 GitHub REST API 的基础 URL。
GITHUB_API_BASE_URL = "https://api.github.com/"

# --- 并发与速率限制配置 ---
# 设置同时处理的最大模型数量。这限制了同时运行的 `process_single_model` 协程数量。
MAX_CONCURRENT_MODELS = 5
# 为 Hugging Face API 请求设置速率限制器：允许每秒最多 5 个请求。
hf_limiter = AsyncLimiter(5, 1.0)
# 为 ArXiv API 请求设置速率限制器：允许每 3 秒最多 1 个请求 (ArXiv 对请求频率要求严格)。
arxiv_limiter = AsyncLimiter(1, 3.0)
# 为 Papers with Code API 请求设置速率限制器：允许每秒最多 2 个请求。
pwc_limiter = AsyncLimiter(2, 1.0)
# 为 GitHub API 请求设置速率限制器：允许每秒最多 1 个请求 (官方速率限制较低，尤其对于未认证请求)。
github_limiter = AsyncLimiter(1, 1.0)  # 保持较低的速率

# --- 日志记录设置 (包含文件日志) ---
# 确保日志目录存在，如果不存在则创建。 `exist_ok=True` 表示如果目录已存在则不报错。
os.makedirs(LOG_DIR, exist_ok=True)

# 配置基本的日志记录设置。
# level=logging.DEBUG: 设置根日志记录器的级别为 DEBUG，这意味着所有级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）的日志都会被处理。
# format: 定义日志消息的格式字符串。
#   %(asctime)s: 日志记录创建的时间。
#   %(levelname)s: 日志记录的级别名称（例如 'INFO', 'WARNING'）。
#   %(funcName)s: 记录日志的函数名。
#   %(message)s: 实际的日志消息。
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
)
# 获取当前模块 (__name__ 通常是 'collect_data') 的日志记录器实例。
# 使用特定的记录器而不是根记录器，可以更精细地控制日志行为。
logger = logging.getLogger(__name__)

# (可选) 移除 basicConfig 可能添加的默认处理器，以避免重复日志。
# 这段代码被注释掉了，因为通常 basicConfig 只在首次调用时添加处理器。
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# 配置流处理器 (Stream Handler)，用于将日志输出到控制台。
stream_handler = logging.StreamHandler()
# 设置流处理器的日志级别为 INFO。这意味着只有 INFO, WARNING, ERROR, CRITICAL 级别的日志会输出到控制台。
stream_handler.setLevel(logging.INFO)
# 为流处理器创建一个格式化器，定义控制台输出的日志格式。
stream_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s"
)
# 将格式化器设置给流处理器。
stream_handler.setFormatter(stream_formatter)
# 将配置好的流处理器添加到我们模块的日志记录器中。
logger.addHandler(stream_handler)

# 配置文件处理器 (File Handler)，用于将日志写入文件。
try:
    # 创建文件处理器，指定日志文件路径、写入模式 ('a' 表示追加) 和编码 ('utf-8')。
    file_handler = logging.FileHandler(
        LOG_FILE, mode="a", encoding="utf-8"
    )
    # 设置文件处理器的日志级别为 DEBUG。所有级别的日志都会写入文件。
    file_handler.setLevel(logging.DEBUG)
    # 为文件处理器创建一个格式化器，定义文件输出的日志格式 (包含更详细的信息，如模块名和行号)。
    #   %(name)s: 记录器的名称。
    #   %(lineno)d: 记录日志的源代码行号。
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] [%(funcName)s] %(message)s"
    )
    # 将格式化器设置给文件处理器。
    file_handler.setFormatter(file_formatter)
    # 将配置好的文件处理器添加到我们模块的日志记录器中。
    logger.addHandler(file_handler)
except Exception as e:
    # 如果设置文件日志记录失败，则记录错误日志。
    logger.error(f"设置文件日志记录到 {LOG_FILE} 失败: {e}")

# (可选) 防止日志消息传播到根记录器，如果根记录器也有处理器，可能导致日志重复。
# logger.propagate = False

# 记录一条信息，表示日志记录已配置完成。
logger.info("日志记录已配置。控制台级别: INFO, 文件级别: DEBUG")

# --- Tenacity 重试配置 ---
# 定义一组可重试的网络相关异常。这些通常是暂时性的问题。
RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,  # 请求超时异常。
    httpx.NetworkError,  # 网络连接相关的通用异常 (包括 ConnectError, ReadError 等)。
    httpx.RemoteProtocolError, # 远程服务器协议错误。
)
# 定义一组值得重试的 HTTP 状态码。这些通常表示服务器暂时过载或不可用。
RETRYABLE_STATUS_CODES = {
    429,  # Too Many Requests (速率限制)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# 为 HTTP 请求定义一个重试策略配置字典。
# 注意：默认情况下，这不会重试 404 Not Found 或认证错误 (401/403)。
# 它也不会重试 ArXiv 库特定的错误，除非它们继承自常见的异常类型。
retry_config_http = dict(
    # stop: 定义停止重试的条件。这里设置为最多尝试 4 次 (1 次初始尝试 + 3 次重试)。
    stop=tenacity.stop_after_attempt(4),
    # wait: 定义每次重试之间的等待策略。
    #   tenacity.wait_exponential: 指数退避，等待时间随重试次数增加 (multiplier=1, min=2, max=20 秒)。
    #   tenacity.wait_random: 添加随机抖动 (0到2秒)，避免同时发起大量重试。
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20)
    + tenacity.wait_random(0, 2),
    # retry: 定义触发重试的条件。
    #   tenacity.retry_if_exception_type: 如果抛出的异常是指定类型之一 (RETRYABLE_NETWORK_ERRORS)，则重试。
    #   tenacity.retry_if_exception: 如果抛出的异常满足特定条件 (这里是 httpx.HTTPStatusError 且状态码在 RETRYABLE_STATUS_CODES 中)，则重试。
    #   `|` 表示逻辑或，满足任一条件即重试。
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS)
        | tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    # before_sleep: 在每次重试等待之前执行的操作。这里配置为记录一条警告日志。
    before_sleep=tenacity.before_sleep_log(
        logger, logging.WARNING
    ),
    # reraise: 如果所有重试都失败了，是否重新抛出最后一次捕获的异常。设置为 True 表示重新抛出。
    reraise=True,
)

# 为可能在线程中运行的同步调用定义一个更简单的重试配置。
# 这主要用于 `asyncio.to_thread` 包装的同步函数调用。
retry_config_sync = dict(
    # stop: 最多尝试 3 次。
    stop=tenacity.stop_after_attempt(3),
    # wait: 每次重试固定等待 2 秒。
    wait=tenacity.wait_fixed(2),
    # retry: 如果抛出任何 Exception 类型的异常，都进行重试。
    retry=tenacity.retry_if_exception_type(
        Exception
    ),
    # before_sleep: 重试前记录警告日志。
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    # reraise: 重试失败后重新抛出异常。
    reraise=True,
)


# --- 定义用于 PWC 数据的 TypedDict ---
# TypedDict 用于定义具有特定键和对应值类型的字典结构，增加代码的类型安全性。
# total=False 表示字典中的键不是必须全部存在的。
class PwcEntryData(TypedDict, total=False):
    """定义从 Papers with Code (PWC) API 获取的单个条目数据的结构。"""
    pwc_id: str                 # PWC 平台上的论文/条目 ID
    pwc_url: str                # PWC 平台上的论文/条目 URL
    title: Optional[str]        # 论文标题 (可能不存在)
    conference: Optional[str]   # 论文发表的会议或期刊 (可能不存在)
    tasks: List[str]            # 与论文相关的任务列表
    datasets: List[str]         # 与论文相关的数据集列表
    methods: List[str]          # 与论文相关的方法列表 (如果之前添加过)
    repositories: List[Dict[str, Any]] # 相关的代码库列表，每个代码库是一个字典，包含 url, stars 等信息
    error: str                  # 如果在获取 PWC 数据时发生错误，记录错误信息


class PaperData(TypedDict, total=False):
    """定义与单个论文相关的数据结构，整合了 ArXiv 和 PWC 的信息。"""
    arxiv_id_base: str          # ArXiv ID 的基础部分 (去掉版本号)
    arxiv_id_versioned: Optional[str] # 带版本号的 ArXiv ID (可能不存在)
    arxiv_metadata: Optional[Dict[str, Any]] # 从 ArXiv API 获取的元数据字典 (可能不存在)
    pwc_entry: Optional[Union[PwcEntryData, Dict[str, Any]]] # 关联的 PWC 条目数据，可以是定义的 PwcEntryData 类型或包含错误信息的字典 (可能不存在)


class ModelOutputData(TypedDict, total=False):
    """定义最终输出的单个模型及其关联信息的完整数据结构。"""
    hf_model_id: str            # Hugging Face 模型的 ID (例如 "bert-base-uncased")
    hf_author: Optional[str]    #模型的作者/组织 (例如 "google-bert")
    hf_sha: Optional[str]       # 模型仓库最后一次提交的 SHA 哈希值
    hf_last_modified: Optional[str] # 模型最后修改时间的 ISO 格式字符串
    hf_downloads: Optional[int] # 模型下载次数
    hf_likes: Optional[int]     # 模型点赞次数
    hf_tags: Optional[List[str]] # 模型的标签列表 (例如 "pytorch", "bert", "arxiv:xxxx.xxxxx")
    hf_pipeline_tag: Optional[str] # 模型的主要任务类型 (例如 "text-classification")
    hf_library_name: Optional[str] # 模型使用的主要库 (例如 "transformers")
    hf_readme_content: Optional[str] # 模型仓库中的 README.md 文件内容
    hf_dataset_links: List[str] # 从模型标签中提取的 Hugging Face 数据集链接列表
    processing_timestamp_utc: str # 开始处理该模型的时间戳 (UTC, ISO 格式)
    linked_papers: List[PaperData] # 与该模型关联的论文数据列表 (PaperData 结构)


# --- 基于 ID 的跟踪 ---
def _load_processed_ids() -> Set[str]:
    """
    从跟踪文件 (PROCESSED_IDS_FILE) 加载已成功处理的 Hugging Face 模型 ID 集合。
    如果文件不存在或读取失败，则返回一个空集合。

    Returns:
        一个包含已处理模型 ID 字符串的集合 (Set)。
    """
    processed_ids: Set[str] = set() # 初始化一个空集合
    # 检查记录已处理 ID 的文件是否存在
    if not os.path.exists(PROCESSED_IDS_FILE):
        logger.info(
            f"未找到已处理 ID 文件 ('{PROCESSED_IDS_FILE}')。将从头开始。"
        )
        return processed_ids # 文件不存在，返回空集
    try:
        # 如果文件存在，尝试以只读模式 ('r') 和 utf-8 编码打开文件
        with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
            # 逐行读取文件内容
            for line in f:
                # 去除行首尾的空白字符 (如换行符)
                model_id = line.strip()
                # 如果行内容不为空，则将其添加到集合中
                if model_id:
                    processed_ids.add(model_id)
        # 记录加载成功的 ID 数量
        logger.info(
            f"从 '{PROCESSED_IDS_FILE}' 加载了 {len(processed_ids)} 个已处理的模型 ID。"
        )
    except IOError as e:
        # 如果在读取文件过程中发生 I/O 错误，记录错误日志并返回空集
        logger.error(
            f"从 {PROCESSED_IDS_FILE} 加载已处理 ID 失败: {e}。将从头开始。"
        )
        return set()
    # 返回加载到的 ID 集合
    return processed_ids


def _save_processed_ids(ids_set: Set[str]) -> None:
    """
    将当前已处理的 Hugging Face 模型 ID 集合保存到跟踪文件 (PROCESSED_IDS_FILE)。
    每次保存都会覆盖文件的原始内容。

    Args:
        ids_set: 包含要保存的模型 ID 字符串的集合。
    """
    try:
        # 确保存储已处理 ID 文件的目录存在
        os.makedirs(os.path.dirname(PROCESSED_IDS_FILE), exist_ok=True)
        # 以写入模式 ('w') 和 utf-8 编码打开文件。写入模式会清空文件原有内容。
        with open(PROCESSED_IDS_FILE, "w", encoding="utf-8") as f:
            # 为了保持文件内容的一致性，先将集合转换为列表并排序
            for model_id in sorted(list(ids_set)):
                # 将每个 ID 写入文件，并在末尾添加换行符
                f.write(model_id + "\n")
        # 记录保存成功的 ID 数量 (使用 DEBUG 级别，因为这会频繁发生)
        logger.debug(
            f"已将 {len(ids_set)} 个已处理的模型 ID 保存到 '{PROCESSED_IDS_FILE}'。"
        )
    except IOError as e:
        # 如果在写入文件过程中发生 I/O 错误，记录错误日志
        logger.error(f"保存已处理 ID 到 {PROCESSED_IDS_FILE} 失败: {e}")


# --- API 客户端初始化 (模块级别) ---
# 初始化 Hugging Face Hub API 客户端实例，传入 API Token (如果存在)。
hf_api = HfApi(token=HF_API_TOKEN)
# 初始化 ArXiv API 客户端实例。
arxiv_client = arxiv.Client()
# 初始化 GitHub API 请求头字典。
github_headers = {}
# 如果 GitHub Token 存在，则将其添加到请求头中，用于认证。
if GITHUB_TOKEN:
    github_headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}", # Bearer 认证方式
        "Accept": "application/vnd.github.v3+json", # 指定接受 GitHub API v3 版本的 JSON 响应
    }
else:
    # 即使没有 Token (匿名请求)，也设置 Accept 头。
    github_headers = {"Accept": "application/vnd.github.v3+json"}

# 创建一个共享的 httpx 异步客户端实例。
# 在整个脚本的生命周期内复用同一个客户端实例可以提高效率（例如，连接复用）。
# 可以根据预期的负载和服务器行为调整超时和连接限制。
http_client = httpx.AsyncClient(
    # timeout: 设置超时时间。
    #   第一个参数 15.0 是连接超时时间（秒）。
    #   read=60.0 是读取超时时间（秒）。
    timeout=httpx.Timeout(15.0, read=60.0),
    # limits: 配置连接限制。
    #   max_connections=100: 允许的最大并发连接数。
    #   max_keepalive_connections=20: 允许保持活动状态（等待后续请求）的最大连接数。
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    # http2=True: 尝试启用 HTTP/2 协议（如果服务器支持），可以提高多路复用效率。
    http2=True,
)

# --- 辅助函数 (异步) ---

# 使用 tenacity 装饰器为函数添加重试逻辑。这里使用了更简单的 `retry_config_sync` 配置。
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3), # 最多尝试 3 次
    wait=tenacity.wait_fixed(2),        # 每次重试等待 2 秒
    retry=tenacity.retry_if_exception_type(Exception), # 捕获任何异常时重试
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING), # 重试前记录日志
    reraise=True, # 重试失败后重新抛出异常
)
async def fetch_hf_model_details(model_id: str) -> Optional[ModelInfo]:
    """
    异步获取指定 Hugging Face 模型的详细信息 (ModelInfo)。
    由于 `hf_api.model_info` 是同步函数，此函数使用 `asyncio.to_thread` 将其放入单独的线程中执行，
    以避免阻塞异步事件循环。应用了重试逻辑。

    Args:
        model_id: Hugging Face 模型的 ID。

    Returns:
        如果成功获取，返回 ModelInfo 对象；如果模型未找到或发生不可重试错误，返回 None。
        如果发生可重试错误且最终失败，会抛出异常。
    """
    # 使用 Hugging Face API 的速率限制器，确保请求不会过于频繁。
    async with hf_limiter:
        try:
            # 记录调试信息
            logger.debug(f"正在获取 HF 模型 {model_id} 的详细信息")
            # 使用 asyncio.to_thread 在后台线程中运行同步的 hf_api.model_info 方法。
            # lambda: hf_api.model_info(...) 创建一个无参数的 lambda 函数来调用目标方法。
            model_info = await asyncio.to_thread(
                hf_api.model_info, repo_id=model_id, token=HF_API_TOKEN
            )
            # 返回获取到的模型信息
            return model_info
        # 捕获特定的 HF Hub 异常：仓库未找到。
        except RepositoryNotFoundError:
            logger.warning(f"未找到 HF 模型仓库: {model_id}")
            return None # 模型不存在，返回 None
        # 捕获 HF Hub 的 HTTP 错误。
        except HfHubHTTPError as e:
            # --- 更新的 HfHubHTTPError 处理 ---
            # 尝试获取 HTTP 响应的状态码。
            status_code = (
                e.response.status_code
                if hasattr(e, "response") and hasattr(e.response, "status_code")
                else None
            )
            logger.error(
                f"获取 {model_id} 详细信息时发生 HF API HTTP 错误: Status {status_code} - {e}"
            )

            # 如果是 401 未授权错误，通常是 Token 问题，记录关键错误并直接返回 None (不重试)。
            if status_code == 401:
                logger.critical(
                    f"HF API 认证错误 (401)。请检查 HUGGINGFACE_API_KEY。"
                )
                return None

            # 定义应该由 tenacity 处理的可重试 HF 特定状态码。
            retryable_hf_status = {429, 500, 502, 503, 504}
            # 如果状态码是可重试的，记录警告并重新抛出异常，让 tenacity 的装饰器处理重试。
            if status_code in retryable_hf_status:
                logger.warning(
                    f"遇到可重试的 HF API 错误 (Status {status_code})。将重新抛出以供 tenacity 处理。"
                )
                raise e

            # 对于其他不可重试的 HF HTTP 错误 (例如 400 Bad Request, 403 Forbidden)，记录警告并返回 None。
            logger.warning(
                f"遇到不可重试的 HF API HTTP 错误 (Status {status_code})。将跳过模型详细信息获取。"
            )
            return None
            # --- 结束更新 ---
        # 捕获其他所有预料之外的异常。
        except Exception as e:
            logger.error(f"获取 HF 模型 {model_id} 详细信息时发生意外错误: {e}")
            # 记录完整的堆栈跟踪信息，方便调试。
            logger.debug(traceback.format_exc())
            # 重新抛出异常，让 tenacity (如果适用) 或上层调用者处理。
            raise e


def extract_arxiv_ids(tags: Optional[List[str]]) -> List[str]:
    """
    从 Hugging Face 模型标签列表中提取有效的 ArXiv ID。
    支持新旧两种格式，并去除版本号。

    Args:
        tags: 一个包含模型标签的字符串列表，可能为 None。

    Returns:
        一个包含提取到的、去除了版本号的 ArXiv ID 基础部分的字符串列表，按字母排序。
    """
    # 如果标签列表为空或为 None，直接返回空列表。
    if not tags:
        return []
    # 使用集合 (set) 来存储提取到的 ID，可以自动去重。
    arxiv_ids: Set[str] = set()
    # 编译正则表达式，用于匹配 ArXiv ID 格式的标签。
    # - `^arxiv:`: 匹配以 "arxiv:" 开头的字符串。
    # - `([\w.-]+(?:/\d{7})?(?:v\d+)?)`: 捕获组 1，匹配 ID 部分。
    #   - `[\w.-]+`: 匹配一个或多个字母、数字、下划线、点或连字符 (用于新格式 ID，如 2303.08774 或 cs.LG)。
    #   - `(?:/\d{7})?`: 非捕获组，可选地匹配斜杠后跟 7 个数字 (用于旧格式，如 hep-th/0207021)。 `?` 表示 0 次或 1 次。
    #   - `(?:v\d+)?`: 非捕获组，可选地匹配 'v' 后跟一个或多个数字 (版本号，如 v1, v2)。 `?` 表示 0 次或 1 次。
    # - `$`: 匹配字符串结尾。
    # - `re.IGNORECASE`: 忽略大小写匹配。
    pattern = re.compile(r"arxiv:([\w.-]+(?:/\d{7})?(?:v\d+)?)$", re.IGNORECASE)

    # 遍历标签列表中的每个标签。
    for tag in tags:
        # 尝试用正则表达式匹配标签。
        match = pattern.match(tag)
        # 如果匹配成功。
        if match:
            # 提取捕获组 1 中的内容，即 ArXiv ID (可能带版本号)。
            arxiv_id = match.group(1)
            # 进行简单的有效性检查：ID 中是否包含点号 '.' 或斜杠 '/'。
            # 这是为了过滤掉一些可能误匹配的简单标签 (如 "arxiv:bert")。
            if "." in arxiv_id or "/" in arxiv_id:
                # 使用正则表达式去除 ID 末尾可能存在的版本号 (v1, v2 等)。
                # `re.sub(r"v\d+$", "", arxiv_id)`: 查找末尾的 'v' 加数字并替换为空字符串。
                arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)
                # 将处理后的基础 ID 添加到集合中。
                arxiv_ids.add(arxiv_id_base)
            else:
                # 如果格式看起来不像有效的 ArXiv ID，记录一条调试信息。
                logger.debug(
                    f"跳过标签中可能无效的 ArXiv ID 格式: {tag}"
                )

    # 将集合转换为列表，并按字母顺序排序后返回。
    return sorted(list(arxiv_ids))


# 使用 tenacity 装饰器应用 HTTP 请求的重试逻辑 (`retry_config_http`)。
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS) |
        tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    异步获取指定 ArXiv ID 的论文元数据。
    使用 `arxiv.py` 库的 `Client.results()` 方法，并在后台线程中运行。应用了重试逻辑。

    Args:
        arxiv_id: ArXiv ID 字符串 (可以带版本号，但搜索时通常库会处理)。

    Returns:
        如果成功获取，返回包含论文元数据的字典；如果论文未找到或发生不可重试错误，返回 None。
        如果发生可重试错误且最终失败，会抛出异常。
    """
    # 使用 ArXiv API 的速率限制器。
    async with arxiv_limiter:
        try:
            logger.debug(f"正在使用 Client.results 获取 ArXiv ID {arxiv_id} 的元数据")
            # 创建一个 arxiv.Search 对象，指定要查询的 ID 列表和最大结果数。
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)

            # 使用全局初始化的 arxiv_client 实例执行搜索。
            # `arxiv_client.results(search)` 返回一个生成器 (generator)。
            # `asyncio.to_thread(list, ...)` 将这个生成器在后台线程中转换为列表。
            # 这是因为 `arxiv.py` 库本身可能包含阻塞 I/O 操作。
            results = await asyncio.to_thread(
                list, arxiv_client.results(search) # 使用 client.results(search)
            )

            # 检查是否找到了结果。
            if results:
                # 获取列表中的第一个结果 (因为我们设置了 max_results=1)。
                paper = results[0]
                # 从 paper 对象中提取需要的字段，构建一个字典。
                return {
                    # 从 entry_id (形如 'http://arxiv.org/abs/2303.08774v1') 中提取带版本的 ID。
                    "arxiv_id_versioned": paper.entry_id.split("/")[-1],
                    "title": paper.title, # 标题
                    "authors": [str(a) for a in paper.authors], # 作者列表 (转换为字符串)
                    "summary": paper.summary.replace("\n", " "), # 摘要 (替换换行符为空格)
                    # 发表日期 (如果存在，转换为 ISO 格式字符串)
                    "published_date": paper.published.isoformat() if paper.published else None,
                    # 更新日期 (如果存在，转换为 ISO 格式字符串)
                    "updated_date": paper.updated.isoformat() if paper.updated else None,
                    "pdf_url": paper.pdf_url, # PDF 链接
                    "doi": paper.doi, # DOI 号 (可能不存在)
                    "primary_category": paper.primary_category, # 主要分类
                    "categories": paper.categories, # 所有分类列表
                }
            else:
                # 如果没有找到结果，记录警告。
                logger.warning(
                    f"通过 arxiv.py API (Client.results) 未找到 ArXiv ID {arxiv_id}。"
                )
                return None # 返回 None
        except Exception as e:
            # 捕获在执行 arxiv.py 代码或线程转换过程中可能发生的任何异常。
            logger.error(
                f"获取 ArXiv ID {arxiv_id} (Client.results) 的元数据时出错: {e}"
            )
            # 记录完整的堆栈跟踪。
            logger.debug(traceback.format_exc())
            # 这里不重新抛出异常，而是返回 None。
            # Tenacity 的重试是基于 fetch_arxiv_metadata 函数本身的调用失败，
            # 而不是基于这里捕获的内部异常（除非内部异常导致函数提前返回 None 之外的值）。
            # 如果 tenacity 因为网络错误等原因重试后仍然失败，它会重新抛出异常，由上层调用者 (process_single_model) 处理。
            return None # 在重试或非重试错误后返回 None


# 应用 HTTP 请求的重试逻辑。
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS) |
        tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def find_pwc_entry_by_arxiv_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    通过 Papers with Code (PWC) API 查询与给定 ArXiv ID 关联的论文条目摘要信息。
    使用共享的 httpx 客户端发送 GET 请求。应用了重试逻辑。

    Args:
        arxiv_id: ArXiv ID 字符串 (可以带版本号，查询时会去除)。

    Returns:
        如果找到唯一的 PWC 条目，返回包含其摘要信息的字典；否则返回 None。
        如果发生可重试错误且最终失败，会抛出异常。
    """
    # 去除 ArXiv ID 中可能存在的版本号，因为 PWC API 通常使用基础 ID 查询。
    arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)
    # 使用 PWC API 的速率限制器。
    async with pwc_limiter:
        # 构建 PWC API 的查询 URL。
        url = f"{PWC_BASE_URL}papers/"
        # 设置查询参数，使用 `arxiv_id` 作为键。
        params = {"arxiv_id": arxiv_id_base}
        logger.debug(f"正在为 ArXiv ID {arxiv_id_base} (来自 {arxiv_id}) 查询 PWC")
        try:
            # 使用共享的 http_client 发送异步 GET 请求。
            response = await http_client.get(url, params=params)
            # 检查响应状态码，如果不是 2xx 成功状态，则抛出 HTTPStatusError 异常。
            response.raise_for_status()
            # 解析响应的 JSON 数据。
            data = response.json()
            # 获取响应中的 'count' 字段，表示找到的条目数量。
            count = data.get("count")

            # 检查是否找到了恰好一个结果。
            if isinstance(count, int) and count == 1 and data.get("results"):
                logger.info(f"为 ArXiv ID {arxiv_id_base} 找到了 1 个 PWC 条目")
                # 获取结果列表中的第一个条目。
                result_entry = data["results"][0]
                # 显式检查结果是否是一个字典。
                if isinstance(result_entry, dict):
                    return result_entry # 返回找到的条目字典
                else:
                    logger.warning(
                        f"ArXiv ID {arxiv_id_base} 的 PWC 条目不是字典类型: {type(result_entry)}"
                    )
                    return None # 格式不符，返回 None
            # 如果找到 0 个结果。
            elif isinstance(count, int) and count == 0:
                logger.info(f"未找到 ArXiv ID {arxiv_id_base} 的 PWC 条目")
                return None # 未找到，返回 None
            # 其他情况：找到多个结果，或响应结构不符合预期。
            else:
                logger.warning(
                    f"为 ArXiv ID {arxiv_id_base} 找到了 {count} 个 PWC 条目或响应异常。将跳过。"
                )
                return None # 结果不唯一或异常，返回 None
        # 捕获 HTTP 状态错误。
        except httpx.HTTPStatusError as e:
            # 如果是 404 Not Found 错误，可能是因为 ArXiv ID 格式错误或 PWC 未收录。
            if e.response.status_code == 404:
                logger.info(f"PWC API 对 ArXiv ID 查询返回 404: {arxiv_id_base}")
                return None # 404 通常不重试，直接返回 None
            # 其他 HTTP 错误（例如 429, 5xx）会由 tenacity 根据 retry_config_http 处理。
            logger.warning(
                f"查询 PWC 出错 ({arxiv_id_base}): {e.response.status_code} - {e}"
            )
            raise e # 重新抛出，让 tenacity 处理重试
        # 捕获 JSON 解析错误或其他异常。
        except (json.JSONDecodeError, Exception) as e:
            logger.error(
                f"查询 PWC 或解析响应时出错 ({arxiv_id_base}): {e}"
            )
            logger.debug(traceback.format_exc())
            raise e # 重新抛出，让 tenacity 处理重试


# 应用 HTTP 请求的重试逻辑。
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS) |
        tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_pwc_relation_list(
    pwc_paper_id: str, relation: str
) -> List[Dict[str, Any]]:
    """
    获取指定 PWC 论文 ID 的关联信息列表 (例如 'repositories', 'tasks', 'datasets')。
    处理 PWC API 可能返回直接列表或分页字典两种格式的情况。应用了重试逻辑。

    Args:
        pwc_paper_id: Papers with Code 平台上的论文 ID。
        relation: 要获取的关联类型 ('repositories', 'tasks', 'datasets', 'methods')。

    Returns:
        一个包含关联项目字典的列表。如果获取失败或未找到，返回空列表。
        如果发生可重试错误且最终失败，会抛出异常。
    """
    all_results: List[Dict[str, Any]] = [] # 初始化用于存储所有结果的列表
    # 构建初始请求的 URL
    current_url: Optional[str] = f"{PWC_BASE_URL}papers/{pwc_paper_id}/{relation}/"
    page_num = 1 # 页码，用于日志记录

    # 循环处理分页，只要 current_url 不是 None 就继续
    while current_url:
        # 使用 PWC API 的速率限制器
        async with pwc_limiter:
            logger.debug(
                f"正在获取 PWC {relation} (论文ID: {pwc_paper_id}, "
                f"页码: {'初始' if page_num == 1 else page_num}) 从 {current_url}"
            )
            try:
                # 发送 GET 请求
                response = await http_client.get(current_url)
                response.raise_for_status() # 检查 HTTP 错误
                data = response.json() # 解析 JSON 响应

                # --- 新增: 处理列表和字典两种响应格式 ---
                results: List[Dict[str, Any]] = [] # 当前页的结果
                next_url: Optional[str] = None # 下一页的 URL

                # 情况 1: API 直接返回一个列表 (旧版或某些端点的行为)
                if isinstance(data, list):
                    logger.debug(
                        f"PWC {relation} (论文ID: {pwc_paper_id}) 返回了直接列表。"
                    )
                    results = data
                    # 如果是直接列表，认为没有下一页
                # 情况 2: API 返回一个包含 'results' 列表和 'next' URL 的字典 (常见的分页格式)
                elif (
                    isinstance(data, dict)
                    and "results" in data
                    and isinstance(data["results"], list)
                ):
                    logger.debug(
                        f"PWC {relation} (论文ID: {pwc_paper_id}) 返回了分页字典。"
                    )
                    results = data["results"]
                    # 获取下一页的 URL，如果不存在则为 None
                    next_url = data.get("next")
                    # 防止无限循环的保护：如果下一页 URL 与当前 URL 相同，则中断
                    if next_url == current_url:
                        logger.warning(
                            f"PWC {relation} 的 'next' URL 与当前 URL 相同，中断循环: {current_url}"
                        )
                        next_url = None
                # 情况 3: 响应格式不符合预期
                else:
                    logger.warning(
                        f"PWC {relation} (论文ID: {pwc_paper_id}) 的数据格式异常。"
                        f"预期是列表或包含 'results' 的字典，实际得到: {type(data)}"
                    )
                    # 如果格式错误，停止处理此关联类型
                    break

                # 将当前页的结果追加到总结果列表中
                all_results.extend(results)

                # 准备下一次迭代：将 current_url 更新为 next_url (如果为 None 则循环结束)
                current_url = next_url
                if current_url:
                    page_num += 1 # 如果有下一页，页码加 1

            # 捕获 HTTP 状态错误
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"获取 PWC {relation} 页面 (论文ID: {pwc_paper_id}) 时发生 HTTP 错误: {e.response.status_code}"
                )
                # 让 tenacity 根据状态码处理重试
                raise e # 重新抛出
            # 捕获 JSON 解析错误或其他异常
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"获取或解析 PWC {relation} 页面 (论文ID: {pwc_paper_id}) 时出错: {e}"
                )
                logger.debug(traceback.format_exc())
                # 让 tenacity 根据异常类型处理重试
                raise e # 重新抛出

    # 循环结束后 (正常结束或因错误中断)
    # 检查是否正常完成但没有结果
    if not all_results and current_url is None:
        logger.info(
            f"完成获取 PWC {relation} (论文ID: {pwc_paper_id})。找到 0 个结果。"
        )
    # 检查是否因为错误中断循环 (current_url 在重试后仍然失败)
    elif current_url:
        logger.error(
            f"在重试后未能获取 PWC {relation} (论文ID: {pwc_paper_id}) 的所有页面。"
        )

    # 返回收集到的所有结果
    return all_results


# 应用 HTTP 请求的重试逻辑。
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
    retry=(
        tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS) |
        tenacity.retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code in RETRYABLE_STATUS_CODES
        )
    ),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def fetch_github_details(
    repo_url: str, follow_redirects: bool = True, max_redirects: int = 3
) -> Optional[Dict[str, Any]]:
    """
    通过 GitHub API 获取给定代码库 URL 的星标数、主要语言和许可证信息。
    需要配置 GITHUB_API_KEY 环境变量才能工作。
    可选地处理 HTTP 301/302 重定向。应用了重试逻辑。

    Args:
        repo_url: GitHub 代码库的 URL (例如 'https://github.com/owner/repo')。
        follow_redirects: 是否跟随重定向。默认为 True。
        max_redirects: 允许的最大重定向次数。默认为 3。

    Returns:
        如果成功获取，返回包含 'stars', 'language', 'license' 的字典；
        如果 URL 无效、未找到仓库、API 密钥无效或发生不可重试错误，返回 None。
        如果发生可重试错误且最终失败，会抛出异常。
    """
    # 如果没有配置 GitHub Token，则直接跳过并返回 None。
    if not GITHUB_TOKEN:
        logger.warning("GitHub token 不可用，跳过详细信息获取。")
        return None
    # 如果传入的 URL 为空，也直接返回 None。
    if not repo_url:
        return None

    current_api_url_to_fetch: Optional[str] = None # 当前要请求的 GitHub API URL
    original_url = repo_url # 保存原始 URL 用于日志记录
    redirect_count = 0 # 重定向计数器

    # --- 初始 URL 解析 ---
    # 尝试从用户提供的各种格式的 URL 中提取 GitHub API 需要的 owner/repo 格式。
    try:
        # 标准 GitHub 网页 URL
        if "github.com" in repo_url.lower():
            # 清理 URL，去除协议头和末尾的斜杠、.git 后缀
            clean_url = (
                repo_url.replace("https://", "").replace("http://", "").strip("/")
            )
            if clean_url.endswith(".git"):
                clean_url = clean_url[:-4]
            # 按斜杠分割 URL 路径
            parts = clean_url.split("/")
            # 检查是否是 github.com/owner/repo 的形式
            if len(parts) >= 3 and parts[0].lower() == "github.com":
                owner, repo_name = parts[1], parts[2]
                # 确保 owner 和 repo_name 都有效
                if owner and repo_name:
                    # 构建对应的 GitHub API URL
                    current_api_url_to_fetch = (
                        f"{GITHUB_API_BASE_URL}repos/{owner}/{repo_name}"
                    )
                else:
                    logger.debug(
                        f"无法从初始 URL 解析 owner/repo: {repo_url}"
                    )
                    return None
            else:
                logger.debug(
                    f"初始 URL 格式无法识别为 github.com/owner/repo: {repo_url}"
                )
                return None # 格式不符
        # GitHub API URL (例如重定向后的 URL)
        elif "api.github.com/repositories/" in repo_url.lower():
            current_api_url_to_fetch = repo_url # 直接使用
        # 其他无法识别的 URL
        else:
            logger.debug(
                f"URL 似乎不是标准的 GitHub 仓库 URL: {repo_url}"
            )
            return None
    except Exception as parse_error:
        logger.warning(f"解析初始 GitHub URL {repo_url} 时出错: {parse_error}")
        return None # 解析出错

    # --- 循环处理请求和重定向 ---
    # 只要当前有有效的 API URL 且未超过最大重定向次数，就继续循环
    while current_api_url_to_fetch and redirect_count <= max_redirects:
        # 使用 GitHub API 的速率限制器
        async with github_limiter:
            logger.debug(
                f"正在从 {current_api_url_to_fetch} 获取 GitHub 详细信息 (尝试: {redirect_count + 1})"
            )
            try:
                # 发送异步 GET 请求到 GitHub API
                # follow_redirects=False: 禁止 httpx 自动处理重定向，我们需要手动处理以跟踪 API URL 的变化
                response = await http_client.get(
                    current_api_url_to_fetch,
                    headers=github_headers, # 使用之前定义的包含 Token 的请求头
                    follow_redirects=False,
                )

                # 情况 1: 请求成功 (状态码 200 OK)
                if response.status_code == 200:
                    data = response.json() # 解析 JSON 响应
                    # 提取所需信息
                    stars = data.get("stargazers_count") # 星标数
                    language = data.get("language") # 主要语言
                    license_info = data.get("license") # 许可证信息字典 (可能为 None)
                    # 安全地访问许可证字典中的 spdx_id (许可证的标准标识符)
                    license_name = None
                    if license_info and isinstance(license_info, dict):
                        license_name = license_info.get("spdx_id")

                    # 获取仓库全名 (owner/repo) 用于日志
                    owner_repo_name = data.get("full_name", "[未知仓库]")
                    # 构建日志信息
                    log_prefix = f"成功获取 {owner_repo_name} 的详细信息: Stars={stars}, Lang={language}, License={license_name}"
                    # 如果发生过重定向，在日志中包含原始 URL
                    log_suffix = (
                        f" (原始 URL: {original_url})" if redirect_count > 0 else ""
                    )
                    logger.info(log_prefix + log_suffix)
                    # 返回包含提取信息的字典
                    return {
                        "stars": stars if isinstance(stars, int) else None,
                        "language": language if isinstance(language, str) else None,
                        "license": license_name if isinstance(license_name, str) else None,
                    }

                # 情况 2: 处理重定向 (状态码 301, 302, 307, 308)
                elif (
                    follow_redirects # 检查是否允许跟随重定向
                    and response.status_code in (301, 302, 307, 308) # 检查是否是重定向状态码
                    and "location" in response.headers # 检查响应头中是否有 Location 字段
                ):
                    redirect_count += 1 # 增加重定向计数
                    redirect_url = response.headers["location"] # 获取重定向的目标 URL
                    logger.info(
                        f"GitHub 请求 {current_api_url_to_fetch} 被重定向 ({response.status_code}) 到: {redirect_url}。"
                        f"正在跟随重定向 ({redirect_count}/{max_redirects})."
                    )
                    # 更新当前要请求的 API URL 为重定向后的 URL，进入下一次循环
                    current_api_url_to_fetch = redirect_url
                    continue # 继续下一次 while 循环

                # 情况 3: 处理特定错误状态码
                elif response.status_code == 404: # Not Found
                    logger.info(
                        f"GitHub API 对 URL 返回 404: {current_api_url_to_fetch}"
                    )
                    return None # 仓库不存在，返回 None
                elif response.status_code == 403: # Forbidden
                    # 特别检查是否是速率限制导致的 403
                    if (
                        "X-RateLimit-Remaining" in response.headers
                        and response.headers["X-RateLimit-Remaining"] == "0"
                    ):
                        # 尝试获取速率限制重置时间
                        reset_time_unix = int(response.headers.get("X-RateLimit-Reset", "0"))
                        reset_time_dt = (
                            datetime.fromtimestamp(reset_time_unix, tz=timezone.utc)
                            if reset_time_unix else "未知"
                        )
                        logger.error(
                            f"GitHub API 速率限制超出 (403) URL: {current_api_url_to_fetch}。"
                            f"限制将在 {reset_time_dt} 重置。停止对此仓库的重试。"
                        )
                    else:
                        # 其他 403 错误可能是权限问题
                        logger.error(
                            f"GitHub API 禁止访问 (403) URL: {current_api_url_to_fetch}。"
                            f"请检查 Token 权限。停止对此仓库的重试。"
                        )
                    return None # 403 错误通常不重试，返回 None
                elif response.status_code == 401: # Unauthorized
                    logger.error(
                        f"GitHub API 未授权 (401) URL: {current_api_url_to_fetch}。"
                        f"请检查 GITHUB_API_KEY。"
                    )
                    return None # 认证失败，返回 None
                # 情况 4: 其他 HTTP 错误 (例如 429, 5xx)
                else:
                    logger.warning(
                        f"获取 GitHub 数据时发生 HTTP 错误 {response.status_code} 从 {current_api_url_to_fetch}。"
                    )
                    # 抛出异常，让 tenacity 处理（如果状态码可重试）
                    response.raise_for_status()

            # 捕获 JSON 解析错误或其他在 try 块内发生的异常
            except (json.JSONDecodeError, Exception) as e:
                logger.error(
                    f"从 {current_api_url_to_fetch} 获取/解析 GitHub 详细信息时出错: {e}"
                )
                logger.debug(traceback.format_exc())
                # 重新抛出异常，让 tenacity 处理
                raise e

    # 如果循环因为超过最大重定向次数而结束
    if redirect_count > max_redirects:
        logger.warning(
            f"原始 URL: {original_url} 超出最大重定向次数 ({max_redirects})"
        )

    # 如果循环正常结束但没有成功返回，或者因为错误中断，最终返回 None
    return None


async def process_single_model(model_id: str) -> Optional[ModelOutputData]:
    """
    处理单个 Hugging Face 模型的主要流程函数。
    协调调用其他辅助函数来获取模型、论文、代码库等信息，并将它们整合在一起。

    Args:
        model_id: 要处理的 Hugging Face 模型 ID。

    Returns:
        如果处理成功，返回包含所有收集到的信息的 ModelOutputData 字典；
        如果处理过程中发生关键错误（例如无法获取 HF 模型基础信息），返回 None。
    """
    logger.info(f"--- 开始处理模型: {model_id} ---")
    # 记录处理开始的时间戳 (UTC)
    processing_start_time = datetime.now(timezone.utc)
    # 初始化输出数据结构
    output_data: Optional[ModelOutputData] = None

    try:
        # 1. 获取 Hugging Face 模型详细信息
        hf_model_info = await fetch_hf_model_details(model_id)
        # 如果无法获取到基础的 HF 模型信息，则认为无法继续处理，记录错误并返回 None
        if not hf_model_info:
            logger.error(
                f"未能获取模型 {model_id} 所需的 HF 详细信息。将跳过此模型。"
            )
            return None

        # --- 1a. 获取 HF README 内容 ---
        hf_readme_content: Optional[str] = None # 初始化 README 内容变量
        try:
            # 使用 huggingface_hub 提供的 hf_hub_download 函数下载 README.md 文件。
            # 这个函数是同步的，所以用 asyncio.to_thread 在后台线程执行。
            # 使用 lambda 传递关键字参数给后台线程中的函数。
            # token=HF_API_TOKEN: 传入 API Token (如果需要访问私有仓库)。
            # repo_type="model": 指定仓库类型为模型。
            # cache_dir=None: 不使用本地缓存 (每次都尝试下载最新)。
            readme_path = await asyncio.to_thread(
                lambda: hf_hub_download(
                    repo_id=model_id,
                    filename="README.md",
                    token=HF_API_TOKEN,
                    repo_type="model",
                    cache_dir=None,
                    # ignore_patterns=None, # ignore_patterns 用于列出文件，而非下载单个文件
                )
            )
            # 下载成功后，读取文件内容。使用 utf-8 编码。
            with open(readme_path, "r", encoding="utf-8") as f:
                hf_readme_content = f.read()
            logger.info(f"成功获取模型 {model_id} 的 README")
        # 捕获 HF Hub 的 HTTP 错误
        except HfHubHTTPError as e:
            # 特别处理 README.md 文件不存在的情况 (404 错误)
            status_code = e.response.status_code if hasattr(e, "response") else 0
            if status_code == 404:
                logger.info(f"模型 {model_id} 未找到 README.md 文件。")
            else:
                # 其他 HTTP 错误记录警告
                logger.warning(
                    f"获取模型 {model_id} 的 README 时发生 HTTP 错误 {status_code}: {e}"
                )
        # 捕获其他下载或读取文件时可能发生的异常
        except Exception as e:
            logger.warning(f"获取或读取模型 {model_id} 的 README 时出错: {e}")
            logger.debug(traceback.format_exc())
        # --- 结束 HF README 获取 ---

        # --- 1b. 提取 HF 数据集链接 ---
        hf_dataset_links: List[str] = [] # 初始化数据集链接列表
        # 检查模型信息中是否有标签
        if hf_model_info.tags:
            # 定义匹配 'dataset:<dataset_id>' 格式标签的正则表达式
            dataset_pattern = re.compile(r"^dataset:([\w/-]+)$", re.IGNORECASE)
            # 遍历所有标签
            for tag in hf_model_info.tags:
                match = dataset_pattern.match(tag)
                # 如果标签匹配成功
                if match:
                    dataset_id = match.group(1) # 提取数据集 ID
                    # 构建标准的 Hugging Face 数据集 URL
                    dataset_url = f"https://huggingface.co/datasets/{dataset_id}"
                    hf_dataset_links.append(dataset_url) # 添加到列表中
            if hf_dataset_links:
                logger.info(
                    f"为模型 {model_id} 提取到数据集链接: {hf_dataset_links}"
                )
        # --- 结束 HF 数据集链接提取 ---

        # 初始化输出字典，填入已获取的 Hugging Face 模型信息
        output_data = {
            "hf_model_id": hf_model_info.id,
            "hf_author": hf_model_info.author,
            "hf_sha": hf_model_info.sha,
            # 将最后修改时间转换为 ISO 格式字符串 (如果存在)
            "hf_last_modified": hf_model_info.lastModified.isoformat() if hf_model_info.lastModified else None,
            "hf_downloads": hf_model_info.downloads,
            "hf_likes": hf_model_info.likes,
            "hf_tags": hf_model_info.tags,
            "hf_pipeline_tag": hf_model_info.pipeline_tag,
            "hf_library_name": hf_model_info.library_name,
            "processing_timestamp_utc": processing_start_time.isoformat(), # 记录处理开始时间
            "hf_readme_content": hf_readme_content, # 添加 README 内容
            "hf_dataset_links": hf_dataset_links, # 添加数据集链接
            "linked_papers": [], # 初始化关联论文列表为空
        }

        # 2. 从 HF 标签中提取 ArXiv ID
        arxiv_ids = extract_arxiv_ids(hf_model_info.tags)
        # 如果没有找到 ArXiv ID，记录信息并直接返回已有的 HF 数据
        if not arxiv_ids:
            logger.info(f"在模型 {model_id} 的标签中未找到有效的 ArXiv ID。")
            return output_data # 即使没有论文，也返回模型数据

        logger.info(f"为模型 {model_id} 找到 {len(arxiv_ids)} 个 ArXiv ID: {arxiv_ids}")

        # 3. 遍历处理找到的每个 ArXiv ID
        for arxiv_id in arxiv_ids:
            # 初始化当前论文的数据结构
            paper_data: PaperData = {
                "arxiv_id_base": re.sub(r"v\d+$", "", arxiv_id), # 存储基础 ID (去版本号)
                "arxiv_metadata": None, # 初始化 ArXiv 元数据为 None
                "pwc_entry": None, # 初始化 PWC 条目为 None
            }

            try:
                # 3a. 获取 ArXiv 元数据
                arxiv_meta = await fetch_arxiv_metadata(arxiv_id)
                if arxiv_meta:
                    # 如果成功获取，更新 paper_data
                    paper_data["arxiv_metadata"] = arxiv_meta
                    # 同时更新带版本号的 ArXiv ID (如果元数据中有)
                    paper_data["arxiv_id_versioned"] = arxiv_meta.get(
                        "arxiv_id_versioned", paper_data["arxiv_id_base"] # Fallback 到基础 ID
                    )
                else:
                    # 如果获取 ArXiv 元数据失败，将当前不完整的 paper_data 添加到结果中，
                    # 并跳过后续的 PWC 查询，继续处理下一个 ArXiv ID。
                    if output_data: # 确保 output_data 不是 None
                        output_data["linked_papers"].append(paper_data)
                    continue # 跳到下一个 arxiv_id

                # 3b. 使用基础 ArXiv ID 查询 PWC 摘要信息
                pwc_entry_summary = await find_pwc_entry_by_arxiv_id(
                    paper_data["arxiv_id_base"] # 使用基础 ID 查询
                )
                # 如果没有找到对应的 PWC 条目
                if not pwc_entry_summary:
                    # 将只包含 ArXiv 信息的 paper_data 添加到结果中
                    if output_data: # 确保 output_data 不是 None
                        output_data["linked_papers"].append(paper_data)
                    continue # 跳到下一个 arxiv_id

                # 从 PWC 摘要信息中获取 PWC 论文 ID
                pwc_paper_id = pwc_entry_summary.get("id")
                # --- 从 PWC 摘要中获取会议信息 ---
                pwc_conference = pwc_entry_summary.get("conference")
                # 如果 PWC 条目中缺少 'id' 字段，则无法继续获取详细信息
                if not pwc_paper_id:
                    logger.warning(
                        f"为 {paper_data['arxiv_id_base']} 找到的 PWC 条目缺少 'id' 字段。"
                    )
                    # 添加包含 ArXiv 信息和部分 PWC 摘要信息的 paper_data
                    if output_data: # 确保 output_data 不是 None
                        output_data["linked_papers"].append(paper_data)
                    continue # 跳到下一个 arxiv_id

                # 3c. 并发获取 PWC 详细信息 (代码库、任务、数据集、方法)
                # 创建一个字典，键是关联类型，值是调用 fetch_pwc_relation_list 的协程对象
                pwc_details_tasks = {
                    "repositories": fetch_pwc_relation_list(pwc_paper_id, "repositories"),
                    "tasks": fetch_pwc_relation_list(pwc_paper_id, "tasks"),
                    "datasets": fetch_pwc_relation_list(pwc_paper_id, "datasets"),
                    "methods": fetch_pwc_relation_list(pwc_paper_id, "methods"),
                }
                # 使用 asyncio.gather 并发执行所有获取 PWC 关联信息的任务
                # return_exceptions=True 表示即使某个任务失败，gather 也会等待所有任务完成，
                # 并将异常对象作为结果返回，而不是直接抛出异常中断 gather。
                pwc_details_results = await asyncio.gather(
                    *pwc_details_tasks.values(), return_exceptions=True
                )
                # 将任务键和结果重新组合成字典
                pwc_details = dict(zip(pwc_details_tasks.keys(), pwc_details_results))

                # 检查每个关联信息的获取结果
                fetched_repos = []
                fetched_tasks = []
                fetched_datasets = []
                fetched_methods = []
                # 处理代码库结果
                if isinstance(pwc_details["repositories"], list):
                    fetched_repos = pwc_details["repositories"]
                else:
                    # 如果结果是异常或其他类型，记录警告
                    logger.warning(
                        f"未能获取 PWC 代码库 (论文ID: {pwc_paper_id}): {pwc_details['repositories']}"
                    )
                # 处理任务结果
                if isinstance(pwc_details["tasks"], list):
                    fetched_tasks = pwc_details["tasks"]
                else:
                    logger.warning(
                        f"未能获取 PWC 任务 (论文ID: {pwc_paper_id}): {pwc_details['tasks']}"
                    )
                # 处理数据集结果
                if isinstance(pwc_details["datasets"], list):
                    fetched_datasets = pwc_details["datasets"]
                else:
                    logger.warning(
                        f"未能获取 PWC 数据集 (论文ID: {pwc_paper_id}): {pwc_details['datasets']}"
                    )
                # 处理方法结果
                if isinstance(pwc_details["methods"], list):
                    fetched_methods = pwc_details["methods"]
                else:
                    logger.warning(
                        f"未能获取 PWC 方法 (论文ID: {pwc_paper_id}): {pwc_details['methods']}"
                    )

                # --- 3d. 并发获取 PWC 代码库的 GitHub 详细信息 ---
                processed_repos = [] # 初始化处理后的代码库列表
                # 如果从 PWC 获取到了代码库列表
                if fetched_repos:
                    detail_fetch_tasks = [] # 存储获取 GitHub 详细信息的协程
                    valid_repo_data = [] # 存储对应的原始 PWC 代码库字典，用于后续合并信息
                    # 遍历从 PWC 获取的每个代码库字典
                    for repo in fetched_repos:
                        repo_url = repo.get("url") # 获取代码库 URL
                        # 检查 URL 是否存在且看起来是 GitHub URL
                        if repo_url and "github.com" in repo_url.lower():
                            # 如果是 GitHub URL，创建获取其详细信息的协程任务
                            detail_fetch_tasks.append(fetch_github_details(repo_url))
                            # 将原始 repo 字典保存起来
                            valid_repo_data.append(repo)
                        else:
                            # 如果不是 GitHub URL 或 URL 不存在，直接添加到处理结果中，
                            # GitHub 相关字段 (stars, license, language) 设为 None。
                            processed_repos.append(
                                {
                                    "url": repo_url,
                                    "stars": None,
                                    "is_official": repo.get("is_official"), # 保留 PWC 的 is_official 字段
                                    "framework": repo.get("framework"), # 保留 PWC 的 framework 字段
                                    "license": None,
                                    "language": None,
                                }
                            )

                    # 如果存在需要获取 GitHub 详细信息的任务
                    if detail_fetch_tasks:
                        # 并发执行所有获取 GitHub 详细信息的任务
                        repo_details_results = await asyncio.gather(
                            *detail_fetch_tasks, return_exceptions=True
                        )

                        # 遍历 GitHub 详细信息的结果和对应的原始 PWC repo 字典
                        for repo_meta, details_result in zip(
                            valid_repo_data, repo_details_results
                        ):
                            stars = None
                            license_name = None
                            language = None
                            # 如果获取 GitHub 详细信息成功 (返回的是字典)
                            if isinstance(details_result, dict):
                                stars = details_result.get("stars")
                                license_name = details_result.get("license")
                                language = details_result.get("language")
                            # 如果获取 GitHub 详细信息失败 (返回的是异常)
                            elif isinstance(details_result, Exception):
                                logger.warning(
                                    f"未能获取 GitHub 详细信息 ({repo_meta.get('url')}): {details_result}"
                                )
                            # 其他情况 (例如 fetch_github_details 返回 None)

                            # 将 PWC 的信息和从 GitHub 获取的信息合并，添加到处理结果列表
                            processed_repos.append(
                                {
                                    "url": repo_meta.get("url"),
                                    "stars": stars,
                                    "is_official": repo_meta.get("is_official"),
                                    "framework": repo_meta.get("framework"),
                                    "license": license_name,
                                    "language": language,
                                }
                            )
                else:
                    # 如果 PWC 没有列出任何代码库
                    logger.info(
                        f"PWC 条目 (ID: {pwc_paper_id}) 未列出代码库"
                    )
                # --- 结束 GitHub 详细信息获取 ---

                # 组装完整的 PWC 条目数据 (包含之前获取的 conference 信息)
                paper_data["pwc_entry"] = {
                    "pwc_id": pwc_paper_id,
                    "pwc_url": f"https://paperswithcode.com/paper/{pwc_paper_id}", # 构建 PWC 页面 URL
                    "title": pwc_entry_summary.get("title"), # PWC 上的标题
                    "conference": pwc_conference, # PWC 上的会议信息
                    # 提取任务名称列表 (过滤掉 None 值)
                    "tasks": [
                        str(task.get("name")) for task in fetched_tasks if task.get("name") is not None
                    ],
                    # 提取数据集名称列表 (过滤掉 None 值)
                    "datasets": [
                        str(dataset.get("name")) for dataset in fetched_datasets if dataset.get("name") is not None
                    ],
                    # 提取方法名称列表 (过滤掉 None 值)
                    "methods": [
                        str(method.get("name")) for method in fetched_methods if method.get("name") is not None
                    ],
                    # 添加处理后的代码库列表 (包含 GitHub 信息)
                    "repositories": processed_repos,
                }

                # 将完整填充的 paper_data 添加到模型的关联论文列表中
                if output_data: # 确保 output_data 不是 None
                    output_data["linked_papers"].append(paper_data)

            # 捕获在处理单个 ArXiv ID 过程中发生的任何未预料异常
            except Exception as e:
                logger.error(
                    f"处理模型 {model_id} 的 ArXiv ID {arxiv_id} 时发生意外错误: {e}"
                )
                logger.error(traceback.format_exc()) # 记录完整堆栈
                # 即使发生错误，也尝试将包含错误信息的 paper_data 添加到结果中
                if "pwc_entry" not in paper_data:
                    # 如果 PWC 部分还没创建，则创建一个带 error 字段的字典
                    paper_data["pwc_entry"] = {"error": str(e)}
                elif isinstance(paper_data["pwc_entry"], dict):
                     # 如果 pwc_entry 已经是字典，添加或更新 error 字段
                    paper_data["pwc_entry"]["error"] = str(e)

                if output_data: # 确保 output_data 不是 None
                    output_data["linked_papers"].append(paper_data)
                # 继续处理下一个 ArXiv ID (不因为单个论文处理失败而中断整个模型处理)

        # 完成对模型所有关联 ArXiv ID 的处理
        logger.info(f"--- 完成处理模型: {model_id} ---")
        # 返回包含模型及其所有（成功或部分成功）关联论文信息的字典
        return output_data

    # 捕获在 process_single_model 函数顶层发生的任何未预料异常
    except Exception as e:
        logger.critical(f"处理模型 {model_id} 过程中发生严重错误: {e}")
        logger.critical(traceback.format_exc())
        # 即使发生严重错误，如果 output_data 已经被部分填充（例如，HF 信息已获取），
        # 仍然尝试返回它，以便至少保存部分信息。
        # 如果 HF 信息获取失败，output_data 会是 None，这里也会返回 None。
        return output_data


# --- 新函数：获取目标模型 ID 列表 (可复用) ---
async def fetch_target_model_ids(limit: int, sort_by: str) -> List[str]:
    """
    从 Hugging Face Hub 获取模型 ID 列表，根据指定的排序方式和数量限制。
    应用了重试逻辑来处理获取列表时的网络问题。

    Args:
        limit: 要获取的最大模型数量。
        sort_by: 排序依据 ('downloads' 或 'likes')。

    Returns:
        一个包含模型 ID 字符串的列表。如果获取失败，返回空列表。
    """
    logger.info(f"正在从 HF Hub 获取前 {limit} 个模型 ID，按 {sort_by} 排序...")
    target_ids: List[str] = [] # 初始化目标 ID 列表
    try:
        # 使用 tenacity 装饰器为内部的 API 调用函数添加重试逻辑
        @tenacity.retry(
            stop=tenacity.stop_after_attempt(4),
            wait=tenacity.wait_exponential(multiplier=1, min=2, max=20) + tenacity.wait_random(0, 2),
            retry=(
                tenacity.retry_if_exception_type(RETRYABLE_NETWORK_ERRORS) |
                tenacity.retry_if_exception(
                    lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code in RETRYABLE_STATUS_CODES
                )
            ),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def get_models_list() -> List[ModelInfo]:
            """内部异步函数，负责调用同步的 hf_api.list_models"""
            # 在后台线程中运行同步的 hf_api.list_models 方法
            loop = asyncio.get_running_loop() # 获取当前事件循环
            models_iterator = await loop.run_in_executor(
                None, # 使用默认的线程池执行器
                lambda: hf_api.list_models(
                    full=False,     # full=False 表示只需要模型的基本信息（主要是 ID），不需要完整配置
                    sort=sort_by,   # 按指定的字段排序
                    direction=-1,   # direction=-1 表示降序排列（最多下载/点赞的在前）
                    limit=limit,    # 直接在 API 调用中应用数量限制
                    # fetch_config=False 已移除，list_models 没有此参数
                ),
            )
            # hf_api.list_models 返回的是一个迭代器，将其转换为列表
            return list(models_iterator)

        # 调用内部函数获取模型列表
        models = await get_models_list()
        # 从 ModelInfo 对象列表中提取模型 ID
        target_ids = [model.id for model in models if model.id]
        logger.info(f"成功从 HF Hub 获取了 {len(target_ids)} 个模型 ID。")

    # 捕获 tenacity 重试多次后仍然失败的错误
    except tenacity.RetryError as e:
        logger.error(
            f"多次重试后仍无法从 HF Hub 获取模型列表: {e}。将返回空列表。"
        )
    # 捕获其他 HF Hub HTTP 错误或意外异常
    except (HfHubHTTPError, Exception) as e:
        logger.error(
            f"从 HF Hub 获取模型列表时发生意外错误: {e}。将返回空列表。"
        )
        logger.debug(traceback.format_exc()) # 记录详细堆栈

    # 返回获取到的（或空的）模型 ID 列表
    return target_ids


# --- 主流程协调 ---
async def main() -> None:
    """脚本的主入口和执行流程控制函数。"""
    # --- 参数解析 ---
    # 创建 ArgumentParser 对象，用于定义和解析命令行参数。
    parser = argparse.ArgumentParser(
        description="收集 Hugging Face 模型数据及其关联的论文和 GitHub 星标。", # 脚本描述
        # formatter_class 指定如何显示帮助信息，ArgumentDefaultsHelpFormatter 会显示参数的默认值。
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 添加 '--sort-by' 参数
    parser.add_argument(
        "--sort-by",
        type=str, # 参数类型为字符串
        choices=["likes", "downloads"], # 允许的值为 'likes' 或 'downloads'
        default="likes", # 默认值为 'likes' (按点赞量排序)
        help="从 Hugging Face Hub 获取模型时按 'likes' 或 'downloads' 排序。", # 参数的帮助说明
    )
    # 添加 '--limit' 参数
    parser.add_argument(
        "--limit",
        type=int, # 参数类型为整数
        default=5000, # 默认获取前 5000 个模型
        help="基于排序从 Hugging Face Hub 获取的最大模型数量。", # 参数的帮助说明
    )
    # 解析命令行传入的参数
    args = parser.parse_args()
    # 将解析到的值赋给变量
    sort_by = args.sort_by
    limit = args.limit
    # --- 结束参数解析 ---

    # 加载之前已经成功处理过的模型 ID 集合
    # 这个集合会在处理过程中被直接更新
    processed_ids_set = _load_processed_ids()
    logger.info(
        # 更新日志，包含命令行参数信息
        f"启动数据收集运行。目标：按 {sort_by} 排序的前 {limit} 个模型。"
    )
    logger.info(f"已加载 {len(processed_ids_set)} 个先前处理过的模型 ID。")

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_JSONL_FILE)
    try:
        # 创建目录，包括任何必要的父目录。如果目录已存在则不报错。
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"已确保输出目录存在: {output_dir}")
    except OSError as e:
        # 如果创建目录失败，记录严重错误并退出脚本。
        logger.critical(
            f"创建输出目录 {output_dir} 失败: {e}。正在退出。"
        )
        sys.exit(1) # 退出码 1 表示异常退出

    # 决定文件写入模式。由于使用 ID 跟踪，总是以追加模式 ('a') 打开文件。
    # 可以选择性地添加一个命令行标志来强制覆盖文件并清空已处理 ID 列表。
    write_mode = "a"
    logger.info(f"输出文件 '{OUTPUT_JSONL_FILE}' 将以追加模式打开。")
    # 确保用于存储已处理 ID 的数据目录也存在（这一步在 _save_processed_ids 函数内部处理）

    # 获取本次运行的目标模型 ID 列表
    target_model_ids = await fetch_target_model_ids(
        limit,  # 使用命令行参数 limit
        sort_by, # 使用命令行参数 sort_by
    )
    # 如果无法获取目标 ID 列表（例如 HF Hub API 访问失败）
    if not target_model_ids:
        logger.warning(
            # 更新日志，包含命令行参数信息
            f"无法获取目标模型 ID (按 {sort_by} 排序的前 {limit} 个)。正在退出。"
        )
        await close_http_client_safely() # 确保关闭 HTTP 客户端
        return # 结束 main 函数

    # 从目标列表中筛选出本次运行需要处理的新模型 ID
    # （即目标 ID 不在已处理 ID 集合中的那些）
    models_to_run = [mid for mid in target_model_ids if mid not in processed_ids_set]
    logger.info(
        # 更新日志，包含命令行参数信息
        f"从 HF 获取的目标模型 (按 {sort_by} 排序的前 {limit} 个): {len(target_model_ids)}"
    )
    logger.info(f"已处理的模型数量: {len(processed_ids_set)}")
    logger.info(f"本次运行需要处理的模型数量: {len(models_to_run)}")

    # 如果没有需要处理的新模型，则记录信息并退出
    if not models_to_run:
        logger.info(
            "根据当前目标列表和已处理 ID 文件，没有需要处理的新模型。"
        )
        await close_http_client_safely() # 关闭 HTTP 客户端
        return # 结束 main 函数

    # 初始化计数器
    attempted_this_run = 0 # 本次运行尝试处理的模型数量
    successful_saves_this_run = 0 # 本次运行成功保存的模型数量

    # 创建一个 asyncio.Semaphore，用于限制同时运行的 process_single_model 协程数量
    process_semaphore = asyncio.Semaphore(MAX_CONCURRENT_MODELS)
    # 创建一个列表，用于存储将要并发执行的任务（协程）
    tasks = []
    # 遍历需要处理的新模型 ID 列表
    for model_id in models_to_run:

        # 定义一个内部的异步函数，用于包装 process_single_model 的调用
        # 这个函数会先获取信号量，然后执行处理，确保并发数受控
        async def process_with_semaphore(mid: str) -> Optional[ModelOutputData]:
            # 声明 nonlocal，表示要修改外部 main 函数作用域中的 attempted_this_run 变量
            nonlocal attempted_this_run
            # 异步获取信号量，如果信号量计数已满，则在此等待直到有空闲
            async with process_semaphore:
                attempted_this_run += 1 # 增加尝试计数
                # 调用核心处理函数并返回结果
                return await process_single_model(mid)

        # 将包装后的协程添加到任务列表中
        tasks.append(process_with_semaphore(model_id))

    # 处理任务并写入结果
    try:
        # 以追加模式 ('a') 和 utf-8 编码打开输出 JSONL 文件
        with open(OUTPUT_JSONL_FILE, "a", encoding="utf-8") as f:
            # 使用 asyncio.as_completed 按完成顺序迭代处理任务列表中的协程
            # enumerate 从 1 开始计数，用于日志记录
            for i, future in enumerate(asyncio.as_completed(tasks), 1):
                # 旧的计数方式被移除: current_total_processed = processed_count + i
                try:
                    # 等待下一个完成的任务，并获取其结果
                    result_data = await future

                    # 检查结果是否有效（是一个包含 'hf_model_id' 的字典）
                    if (
                        result_data
                        and isinstance(result_data, dict)
                        and "hf_model_id" in result_data
                    ):
                        hf_model_id = result_data["hf_model_id"]

                        # --- 立即保存逻辑 ---
                        # 1. 将结果字典转换为 JSON 字符串并写入文件，添加换行符
                        #    ensure_ascii=False 确保非 ASCII 字符（如中文）正确写入，而不是被转义
                        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                        # 立刻将缓冲区的内容刷新（写入）到磁盘文件，确保数据不丢失
                        f.flush()

                        # 2. 将成功处理的模型 ID 添加到内存中的集合
                        processed_ids_set.add(hf_model_id)

                        # 3. 立刻将更新后的 ID 集合保存回文件
                        _save_processed_ids(processed_ids_set)
                        # --- 结束立即保存逻辑 ---

                        # 增加成功保存计数
                        successful_saves_this_run += 1
                        # 记录保存成功的日志信息
                        logger.info(
                            f"已保存: {hf_model_id} ({successful_saves_this_run} saved this run / {attempted_this_run} attempted / {len(processed_ids_set)} total unique)"
                        )

                    # 如果 process_single_model 明确返回 None (例如，获取 HF 基础信息失败)
                    elif result_data is None:
                        logger.warning(
                            f"跳过保存模型尝试 {attempted_this_run} (返回 None)。目前总共唯一处理: {len(processed_ids_set)}"
                        )
                    # 如果返回了非预期的结果类型
                    else:
                        logger.error(
                            f"process_single_model 返回了意外的结果类型 (尝试 {attempted_this_run}): {type(result_data)}。目前总共唯一处理: {len(processed_ids_set)}"
                        )

                # 捕获等待任务完成或处理单个任务结果时发生的异常
                except Exception as e:
                    # 注意：attempted_this_run 在进入 process_with_semaphore 时已增加
                    logger.error(
                        f"等待或处理模型 future 时出错 (尝试 {attempted_this_run}): {e}"
                    )
                    logger.error(traceback.format_exc()) # 记录完整堆栈
                    # 如果处理失败，则不将 ID 添加到 processed_ids_set，也不保存

    # 捕获文件打开或写入时的 I/O 错误
    except IOError as e:
        logger.critical(
            f"严重错误: 打开或写入输出文件 {OUTPUT_JSONL_FILE} 失败: {e}"
        )
    # 捕获主处理循环中其他未预料的异常
    except Exception as e:
        logger.critical(
            f"严重错误: 主处理循环中发生意外错误: {e}"
        )
        logger.critical(traceback.format_exc())
    finally:
        # --- 最终保存与清理 ---
        # 在程序结束前（无论是正常结束还是异常退出），再次保存已处理 ID 集合。
        # 这是一种保险措施，尽管立即保存逻辑减少了其必要性。
        logger.info(
            "正在保存最终的已处理模型 ID 集合 (如果未发生错误，则为冗余操作)..."
        )
        _save_processed_ids(processed_ids_set)

        # 优雅地关闭共享的 httpx 客户端
        await close_http_client_safely()

        # 记录最终的总结信息
        logger.info("--- 数据收集运行结束 ---")
        logger.info(
            f"本次运行尝试处理了 {attempted_this_run} 个新模型。"
        )
        logger.info(
            f"成功收集并保存了 {successful_saves_this_run} 个新模型的数据。"
        )
        logger.info(
            f"所有运行累计处理的唯一模型总数 (记录在 {PROCESSED_IDS_FILE}): {len(processed_ids_set)}"
        )


async def close_http_client_safely() -> None:
    """安全地关闭共享的 httpx 异步客户端（如果它存在且尚未关闭）。"""
    logger.info("正在关闭 HTTP 客户端...")
    # 检查全局变量中是否存在 http_client，它是否是 AsyncClient 实例，以及它是否尚未关闭
    if (
        "http_client" in globals() # 检查变量是否存在
        and isinstance(http_client, httpx.AsyncClient) # 检查类型
        and not http_client.is_closed # 检查是否已关闭
    ):
        try:
            # 异步关闭客户端
            await http_client.aclose()
            logger.info("HTTP 客户端已关闭。")
        except Exception as close_e:
            # 如果关闭过程中发生错误，记录日志
            logger.error(f"关闭 HTTP 客户端时出错: {close_e}")
    else:
        # 如果客户端不存在、类型不对或已关闭，则记录相应信息
        logger.info("HTTP 客户端已关闭或未初始化。")


# --- 脚本入口 ---
# 当脚本被直接运行时（而不是作为模块导入时），__name__ 的值是 "__main__"
if __name__ == "__main__":
    try:
        # 运行主异步函数 main()
        asyncio.run(main())
    # 捕获用户通过 Ctrl+C 发送的键盘中断信号
    except KeyboardInterrupt:
        logger.info(
            "用户中断了收集过程。正在尝试保存最终状态并关闭客户端。"
        )
        # 在键盘中断时尝试进行优雅关闭
        # 状态保存现在主要在 main 函数的 finally 块中处理
        # 这里确保客户端被关闭
        asyncio.run(close_http_client_safely())
    # 捕获在脚本执行过程中未被处理的其他所有严重异常
    except Exception as e:
        logger.critical(f"脚本执行过程中发生未处理的严重异常: {e}")
        logger.critical(traceback.format_exc())
        # 即使发生严重错误，也尝试关闭客户端
        asyncio.run(close_http_client_safely())
        # 以非零退出码退出脚本，表示执行失败
        sys.exit(1)