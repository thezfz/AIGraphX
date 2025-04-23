# -*- coding: utf-8 -*-
"""
文件目的：测试 AIGraphX 项目的核心配置加载逻辑。

本测试文件 (`test_config.py`) 专注于验证位于 `aigraphx/core/config.py` 文件中的配置加载机制，
特别是 `Settings` 类（基于 Pydantic-Settings）的行为。

目标是确保配置项（如数据库连接字符串、API 密钥、模型名称等）能够正确地从以下来源加载，
并遵循正确的优先级顺序：
1. 环境变量 (最高优先级)
2. `.env` 文件中定义的值
3. `Settings` 类中定义的默认值 (最低优先级)

主要交互：
- 导入 `os` 模块：用于与环境变量交互（虽然主要通过 `monkeypatch` 模拟）。
- 导入 `pytest` 和 `pytest.MonkeyPatch`：`pytest` 是测试框架，`MonkeyPatch` 用于在测试期间安全地修改或模拟环境变量和模块行为。
- 导入 `typing`：用于类型提示，增强代码可读性和健壮性。
- 导入 `unittest.mock`：用于更复杂的模拟场景（如果需要）。
- 导入 `Pydantic` 相关类型：如 `PostgresDsn`, `HttpUrl`，因为 `Settings` 类中使用了这些类型。
- 导入 `pydantic_settings.SettingsConfigDict`：用于 Pydantic v2 中的配置。
- 导入被测试的模块和类：`aigraphx.core.config` 模块和 `aigraphx.core.config.Settings` 类。
- 定义常量和字典：`DEFAULT_PG_*`, `DEFAULT_CONFIG`, `MOCK_ENV_CONFIG`, `EXPECTED_SETTINGS_*` 等，用于存储预期的默认值、模拟的环境变量值以及测试断言的期望结果。
- 定义 Fixtures：特别是 `patch_os_getenv`，它使用 `monkeypatch` 来模拟 `os.getenv` 函数，使得测试可以独立于实际运行环境进行，确保测试的隔离性和可重复性。
- 编写测试函数 (`test_*`)：
    - `test_settings_loading_with_env_vars_override`: 测试当设置了环境变量时，它们是否能成功覆盖 `.env` 文件或 `Settings` 类中的默认值。
    - `test_settings_loading_defaults_with_env_file`: 测试在没有设置覆盖环境变量的情况下，`Settings` 是否能正确地从 `.env` 文件和类定义中加载默认值。
    - 测试 `_correct_postgres_driver` 辅助函数（如果适用，虽然当前代码已移除直接调用，但逻辑内嵌在 Pydantic 模型中）。

这些测试对于确保应用程序在不同环境（开发、测试、生产）中都能获取到正确的配置至关重要，从而保证数据库连接、外部服务访问等关键功能的正常运行。
测试还特别关注了敏感信息（如密码、API 密钥）的处理，确保它们不被硬编码，并通过配置加载。
"""

import os # 导入 Python 内置的 os 模块，用于与操作系统交互，特别是访问环境变量。
import pytest # 导入 pytest 测试框架。
from typing import Dict, Any, Callable, Optional, Generator # 从 typing 模块导入类型提示工具，用于提高代码可读性和进行静态类型检查。
from unittest import mock # 导入 unittest.mock 模块，如果需要进行更复杂的模拟操作时使用。
from pydantic import PostgresDsn, HttpUrl # 从 Pydantic 库导入特定的数据类型，用于验证配置项（如数据库连接字符串和 URL）。
from pydantic_settings import SettingsConfigDict # 从 pydantic-settings 导入用于配置 Settings 类的字典类型（Pydantic V2 方式）。

# 导入被测试的配置模块本身，以及模块中定义的 Settings 类
import aigraphx.core.config # 导入整个配置模块
from aigraphx.core.config import Settings # 直接导入 Settings 类，方便实例化和测试

# 定义一些常量，代表如果在环境变量或 .env 文件中找不到对应项时，config.py 中应使用的默认值
# 这些常量主要用于在测试中设置期望值，确保与 config.py 中的定义一致。
DEFAULT_PG_USER = "aigraphx_user"
DEFAULT_PG_PASS = "aigraphx_password"
DEFAULT_PG_DB = "aigraphx"
DEFAULT_PG_HOST = "localhost"
DEFAULT_PG_PORT = "5432"
DEFAULT_NEO4J_URI = "neo4j://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_DB = "neo4j" # Neo4j 默认数据库名
DEFAULT_MODEL = "all-MiniLM-L6-v2" # 默认使用的句子转换器模型

# 使用字典来存储预期的默认配置值，方便在测试中断言
# 注意：这里的 DATABASE_URL 是根据上面的 POSTGRES_* 默认值手动构建的，用于对比。
# Pydantic Settings 类会自动处理这种构建。
DEFAULT_CONFIG: Dict[str, Any] = {
    "POSTGRES_USER": "aigraphx_user",
    "POSTGRES_PASSWORD": "aigraphx_password",
    "POSTGRES_DB": "aigraphx",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "DATABASE_URL": "postgresql://aigraphx_user:aigraphx_password@localhost:5432/aigraphx", # 预期从 PG_* 构建的 URL
    "NEO4J_URI": "neo4j://localhost:7687",
    "NEO4J_USERNAME": "neo4j", # 注意 config.py 使用 NEO4J_USERNAME
    "NEO4J_PASSWORD": None,  # 测试中不验证实际密码的值，设为 None
    "NEO4J_DATABASE": "neo4j", # Neo4j 数据库名
    "PWC_API_KEY": None,  # 不验证实际 API 密钥
    "HUGGINGFACE_API_KEY": None,  # 不验证实际 API 密钥
    "SENTENCE_TRANSFORMER_MODEL": "all-MiniLM-L6-v2",
    "RELOAD": False, # Uvicorn 是否自动重载，默认为 False
    "TEST_DATABASE_URL": None, # 测试数据库 URL，默认为 None
}

# 修正：这个字典现在反映了 DATABASE_URL 是由 POSTGRES_* 变量构建的预期结果。
# 移除了 NEO4J_DATABASE，因为它不是 MOCK_ENV_CONFIG 直接设置的键。
# 这个字典用于模拟环境变量，测试 Settings 是否能正确加载这些模拟值。
MOCK_ENV_CONFIG: Dict[str, Any] = {
    # 定义用于构建 DATABASE_URL 的 PostgreSQL 环境变量
    "POSTGRES_USER": "test_pg_user",
    "POSTGRES_PASSWORD": "test_pg_pass",
    "POSTGRES_DB": "test_pg_db",
    "POSTGRES_HOST": "test_pg_host",
    "POSTGRES_PORT": "1234",
    # 预期 Settings 类根据上面的 PG_* 变量计算出的 DATABASE_URL
    "DATABASE_URL": "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db",
    "NEO4J_URI": "neo4j://test_neo4j_host:7777",
    "NEO4J_USERNAME": "test_neo4j_user",  # config.py Settings 类实际使用的环境变量键
    "NEO4J_PASSWORD": None,  # 不在测试中硬编码密码
    "PWC_API_KEY": None,  # 不在测试中硬编码 API 密钥
    "HUGGINGFACE_API_KEY": None,  # 不在测试中硬编码 API 密钥
    "SENTENCE_TRANSFORMER_MODEL": "test-model",
    "RELOAD": True, # 模拟设置为 True
    # 假设 TEST_DATABASE_URL 的环境变量格式是正确的 DSN 字符串
    "TEST_DATABASE_URL": "postgresql://test:pass@host/db",
}

# --- 测试数据 (定义 settings 对象属性的预期值) ---

# 当使用环境变量覆盖时，Settings 实例预期包含的值
EXPECTED_SETTINGS_ENV = {
    "project_name": "AIGraphX", # Pydantic 模型中定义的默认值
    "api_v1_str": "/api/v1", # Pydantic 模型中定义的默认值
    "log_level": "INFO", # Pydantic 模型中定义的默认值
    "environment": "development", # 需要确保这个值是 'development'，除非被 env 覆盖
    "database_url": "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db", # 从 MOCK_ENV_CONFIG 的 PG_* 构建
    "pg_pool_min_size": 1, # Pydantic 模型中定义的默认值
    "pg_pool_max_size": 10, # Pydantic 模型中定义的默认值
    "neo4j_uri": "neo4j://test_neo4j_host:7777", # 从 MOCK_ENV_CONFIG 加载
    "neo4j_username": "test_neo4j_user", # 从 MOCK_ENV_CONFIG 加载
    "neo4j_password": None,  # 不校验实际密码
    "neo4j_database": "test_neo4j_db", # 需要环境变量 NEO4J_DATABASE 设置
    "pwc_api_key": None,  # 不校验实际 API 密钥
    "huggingface_api_key": None,  # 不校验实际 API 密钥
    "sentence_transformer_model": "test-model", # 从 MOCK_ENV_CONFIG 加载
    "embedder_device": "cpu", # Pydantic 模型中定义的默认值
    "faiss_index_path": "data/faiss_index.bin", # Pydantic 模型中定义的默认值
    "faiss_mapping_path": "data/papers_faiss_ids.json", # Pydantic 模型中定义的默认值
    "models_faiss_index_path": "data/models_faiss.index", # Pydantic 模型中定义的默认值
    "models_faiss_mapping_path": "data/models_faiss_ids.json", # Pydantic 模型中定义的默认值
    "reload": True, # 从 MOCK_ENV_CONFIG 加载
    # TEST_DATABASE_URL 应从环境变量加载，并由 Pydantic 添加驱动部分
    "test_database_url": "postgresql+psycopg://test:pass@host/db",
    "build_faiss_batch_size": 128, # Pydantic 模型中定义的默认值
    # 如果测试环境没有设置相应的 test 变量，则这些字段应为 None 或其默认值
    "test_neo4j_uri": None,
    "test_neo4j_password": None,
    "test_neo4j_database": None,
    "test_faiss_paper_index_path": None,
    "test_faiss_paper_mapping_path": None,
    "test_faiss_model_index_path": None,
    "test_faiss_model_mapping_path": None,
}

# 当只依赖 .env 文件和类定义默认值时，Settings 实例预期包含的值
# !! 重要 !! 这个字典的值需要手动验证，确保它准确反映了项目根目录下 .env 文件的内容
# 以及 Settings 类中定义的默认值。
# 这里的值是基于示例 .env 文件推断的，实际测试时需要与你的 .env 文件匹配。
EXPECTED_SETTINGS_DEFAULTS = {
    "project_name": "AIGraphX", # 类默认值
    "api_v1_str": "/api/v1", # 类默认值
    "log_level": "INFO", # 类默认值
    "environment": "development", # 假设 .env 中是 development 或类默认值
    # 预期从 .env 文件加载的值
    "database_url": "postgresql://aigraphx_user:aigraphx_password@postgres:5432/aigraphx",
    "pg_pool_min_size": 1, # 类默认值
    "pg_pool_max_size": 10, # 类默认值
    # 预期从 .env 文件加载的值 (注意 URI 中通常不包含密码)
    "neo4j_uri": "neo4j://neo4j:7687",
    "neo4j_username": "neo4j", # 预期从 .env 加载
    # 预期从 .env 文件加载的值 - 不再校验实际密码
    "neo4j_password": None,
    "neo4j_database": "neo4j", # 预期从 .env 或类默认值加载
    # 预期从 .env 加载 - 不再校验实际 API 密钥
    "pwc_api_key": None,
    "hf_api_key": None, # Hugging Face Key (注意 Settings 类字段名)
    "github_api_key": None, # GitHub Key
    "sentence_transformer_model": "all-MiniLM-L6-v2", # 预期从 .env 或类默认值加载
    # 预期从 .env 加载
    "embedder_device": "cuda",
    "faiss_index_path": "data/faiss_index.bin", # 类默认值 (假设 .env 未定义)
    "faiss_mapping_path": "data/papers_faiss_ids.json", # 类默认值 (假设 .env 未定义)
    # 预期从 .env 加载
    "models_faiss_index_path": "data/models_faiss.index",
    "models_faiss_mapping_path": "data/models_faiss_ids.json",
    # 预期从 .env 加载 (假设 RELOAD=true)
    "reload": True,
    # 预期从 .env 加载 (注意主机名可能是 127.0.0.1)
    "test_database_url": "postgresql://aigraphx_test_user:aigraphx_test_password@127.0.0.1:5433/aigraphx_test",
    "build_faiss_batch_size": 128, # 类默认值 (假设 .env 未定义)
    "workers_count": os.cpu_count(), # 类默认值 (使用 os.cpu_count())
    # 预期从 .env 加载的测试 Neo4j 配置
    "test_neo4j_uri": "neo4j://127.0.0.1:7688",
    "test_neo4j_password": None, # 不校验密码
    "test_neo4j_database": "neo4j",
    # 测试 Faiss 路径，如果 .env 未定义，则使用类默认值 (None)
    "test_faiss_paper_index_path": None,
    "test_faiss_paper_mapping_path": None,
    "test_faiss_model_index_path": None,
    "test_faiss_model_mapping_path": None,
}

# --- Fixtures ---
# 不再需要 mock_getenv fixture，因为我们将直接使用 monkeypatch 来设置/取消设置环境变量。

# --- 用于模拟 os.getenv 的辅助函数 ---
# MOCKED_ENV: Dict[str, Optional[str]] = {} # 存储模拟的环境变量
# ORIGINAL_OS_GETENV = os.getenv # 保存原始的 os.getenv 函数

# def mock_getenv(var_name: str, default: Optional[str] = None) -> Optional[str]:
#     """模拟的 os.getenv 函数，从 MOCKED_ENV 字典返回值。"""
#     return MOCKED_ENV.get(var_name, default)

# @pytest.fixture(autouse=True) # autouse=True 表示这个 fixture 会自动应用于模块中的所有测试函数
# def patch_os_getenv(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
#     """
#     Pytest fixture，自动模拟 os.getenv 函数。
#     它使用 monkeypatch 将 os.getenv 替换为我们定义的 mock_getenv。
#     在每个测试运行前，它会清空 MOCKED_ENV，确保测试隔离。
#     测试结束后，monkeypatch 会自动恢复原始的 os.getenv 函数。
#     """
#     # 在每个测试开始前清空模拟环境字典
#     MOCKED_ENV.clear()
#     # 使用 monkeypatch 将 os.getenv 替换为 mock_getenv
#     monkeypatch.setattr(os, "getenv", mock_getenv)
#     yield # yield 语句表示执行测试函数本身
#     # 测试结束后，monkeypatch 会自动恢复 os.getenv

# --- 测试 (修改后的策略) ---
# 新策略：直接使用 monkeypatch.setenv() 和 monkeypatch.delenv() 控制环境变量，
# 并直接实例化 Settings 类，让 Pydantic-Settings 处理加载逻辑（包括 .env 文件）。


def test_settings_loading_with_env_vars_override(
    monkeypatch: pytest.MonkeyPatch, # 请求 pytest 的 monkeypatch fixture
) -> None:
    """
    测试场景：加载 Settings，并使用环境变量覆盖 `.env` 文件或类定义中的部分值。
    目标：验证环境变量具有最高优先级。
    """
    # 1. 定义需要通过环境变量覆盖的值
    #    只定义那些我们想要改变的值。Pydantic-Settings 会优先使用这些值。
    #    直接设置最终期望的 URL，而不是组件。
    env_vars_to_override = {
        "ENVIRONMENT": "testing_override_env", # 覆盖环境标识
        "DATABASE_URL": "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db", # 直接覆盖数据库 URL
        "NEO4J_URI": "neo4j://override_host:1111", # 覆盖 Neo4j URI
        "NEO4J_PASSWORD": "override_neo4j_pass", # 覆盖 Neo4j 密码 (仅用于测试覆盖逻辑)
        "RELOAD": "false", # 覆盖 reload 标志 (注意环境变量通常是字符串)
        "TEST_DATABASE_URL": "postgresql://override_test:pass@host/db", # 覆盖测试数据库 URL
    }
    # 使用 monkeypatch 设置这些环境变量
    for k, v in env_vars_to_override.items():
        monkeypatch.setenv(k, v)

    # 2. 直接实例化 Settings 类
    #    Pydantic-Settings 会自动尝试加载 .env 文件（如果存在且未在 Settings.model_config 中禁用），
    #    然后应用环境变量覆盖。
    settings_instance = Settings()

    # 3. 定义期望的结果状态
    #    首先，复制一份基于 .env 和类默认值的期望状态 (EXPECTED_SETTINGS_DEFAULTS)。
    #    然后，手动更新那些我们通过环境变量覆盖了的字段。
    expected_env = EXPECTED_SETTINGS_DEFAULTS.copy()
    expected_env["environment"] = "testing_override_env" # 应用覆盖
    expected_env["database_url"] = (
        "postgresql://test_pg_user:test_pg_pass@test_pg_host:1234/test_pg_db" # 应用覆盖
    )
    expected_env["neo4j_uri"] = "neo4j://override_host:1111" # 应用覆盖
    # 注意：虽然我们用 monkeypatch 设置了 NEO4J_PASSWORD，但在断言时通常会忽略它。
    # 但如果需要精确测试覆盖，可以在这里设置预期值，并调整下面的忽略逻辑。
    # expected_env["neo4j_password"] = "override_neo4j_pass"
    expected_env["reload"] = False # 应用覆盖 (Pydantic 会将 "false" 转为 False)
    expected_env["test_database_url"] = "postgresql://override_test:pass@host/db" # 应用覆盖

    # 4. 断言实际加载的 Settings 与期望状态一致
    #    使用 Pydantic v2 的 model_dump() 获取实例的字典表示
    actual_settings_dict = settings_instance.model_dump()

    # --- 添加调试打印 ---
    # 这些 print 语句有助于在测试失败时，快速比较实际值和期望值的键集合，找出差异。
    print("\n--- Override Test --- ")
    print("Actual keys (实际加载的键):", sorted(actual_settings_dict.keys()))
    print("Expected keys (期望的键):", sorted(expected_env.keys()))
    # --- 结束调试打印 ---

    # 创建一个包含不应进行值比较的敏感键（如 API 密钥、密码）的列表
    ignore_keys = [
        "pwc_api_key",
        "hf_api_key", # Settings 类中使用的字段名
        "github_api_key",
        "neo4j_password",
        "huggingface_api_key", # 环境变量名，可能也存在于 dump 中，以防万一
        "test_neo4j_password",
    ]

    # 遍历实际加载的配置项
    for key, actual_value in actual_settings_dict.items():
        # 如果键在忽略列表中，则跳过值的比较
        if key in ignore_keys:
            continue

        # 检查实际加载的键是否存在于我们的期望字典中
        # 如果实际加载了一个我们未预期的键，测试应该失败
        assert key in expected_env, (
            f"在 Settings 对象中发现了未预期的属性 '{key}' (值: {actual_value})"
        )
        # 获取该键的期望值
        expected_value = expected_env[key]

        # --- 值比较逻辑 ---
        # 对于 URL 类型的字段，并且期望值不为 None 时，进行字符串比较
        # Pydantic V2 会将 URL 字符串解析为特定类型，所以转换为字符串比较更可靠
        if (
            key in ["database_url", "neo4j_uri", "test_database_url", "test_neo4j_uri"]
            and expected_value is not None
        ):
            actual_value_str = str(actual_value) if actual_value else None
            expected_value_str = str(expected_value)
            assert actual_value_str == expected_value_str, (
                f"配置键 '{key}' 不匹配。预期: '{expected_value_str}', 实际: '{actual_value_str}'"
            )
        # 如果期望值是 PostgresDsn 类型，也进行字符串比较
        elif isinstance(expected_value, PostgresDsn):
            assert str(actual_value) == str(expected_value), (
                f"配置键 '{key}' 不匹配。预期: '{str(expected_value)}', 实际: '{str(actual_value)}'"
            )
        # 如果期望值是 None，断言实际值也是 None
        elif expected_value is None:
            assert actual_value is None, (
                f"配置键 '{key}' 预期为 None, 实际: '{actual_value}'"
            )
        # 其他情况，直接比较值
        else:
            # 特殊处理 workers_count，它可能因运行环境的 CPU 核心数而异
            if key == "workers_count":
                # 只检查它是否是正整数
                assert isinstance(actual_value, int) and actual_value > 0, f"workers_count 应为正整数，实际: {actual_value}"
            else:
                # 普通的值比较
                assert actual_value == expected_value, (
                    f"配置键 '{key}' 不匹配。预期: '{expected_value}', 实际: '{actual_value}'"
                )


def test_settings_loading_defaults_with_env_file(
    monkeypatch: pytest.MonkeyPatch, # 请求 monkeypatch fixture，虽然此测试不主动设置 env
) -> None:
    """
    测试场景：加载 Settings，不设置任何覆盖环境变量。
    目标：验证 Settings 能否正确地从 `.env` 文件（如果存在）和类定义中加载默认值。
    依赖：此测试的准确性依赖于 `EXPECTED_SETTINGS_DEFAULTS` 字典正确反映了 `.env` 文件和类定义。
    """
    # --- 手动检查 .env 文件内容（用于调试） ---
    # 这部分代码尝试读取项目根目录下的 .env 文件，并打印部分内容。
    # 这有助于在测试失败时，确认 .env 文件是否按预期存在，以及其内容是否与 EXPECTED_SETTINGS_DEFAULTS 匹配。
    try:
        print("\n--- 手动检查 .env 文件 ---")
        # 获取项目根目录（假设测试文件在 Backend/tests/core/ 下）
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        dotenv_path = os.path.join(project_root, ".env") # 构建 .env 文件路径
        print(f"检查路径: {dotenv_path}")
        if os.path.exists(dotenv_path): # 检查文件是否存在
            # 使用 utf-8 编码打开文件
            with open(dotenv_path, "r", encoding="utf-8") as f:
                content = f.read() # 读取文件内容
                print(".env 文件内容示例 (前 100 字符):", content[:100])
                # 尝试手动查找某个键，以确认文件内容是否符合预期
                if "HF_API_KEY=" in content:
                    print("在手动读取中找到了 HF_API_KEY。")
                else:
                    print("在手动读取中 *未* 找到 HF_API_KEY。")
        else:
            print(".env 文件在预期路径未找到。")
    except Exception as e:
        print(f"手动读取 .env 文件时出错: {e}")
    # --- 结束手动检查 .env 文件 ---

    # 1. 不使用 monkeypatch 设置或删除任何环境变量。
    #    让 Pydantic-Settings 自然地加载 .env 文件（如果存在）。

    # 2. 直接实例化 Settings 类
    settings_instance = Settings()

    # 3. 断言加载结果与预期的默认值 (EXPECTED_SETTINGS_DEFAULTS) 一致
    actual_settings_dict = settings_instance.model_dump()

    # --- 添加调试打印 ---
    print("\n--- Defaults Test (默认值测试) --- ")
    print("Actual keys (实际加载的键):", sorted(actual_settings_dict.keys()))
    print("Expected default keys (预期的默认键):", sorted(EXPECTED_SETTINGS_DEFAULTS.keys()))
    # 打印一些关键的实际加载值，用于调试
    print("Actual DATABASE_URL:", actual_settings_dict.get("database_url"))
    print("Actual TEST_DATABASE_URL:", actual_settings_dict.get("test_database_url"))
    print("Actual NEO4J_URI:", actual_settings_dict.get("neo4j_uri"))
    # --- 结束调试打印 ---

    # 同样，定义需要忽略比较的敏感键
    ignore_keys = [
        "pwc_api_key",
        "hf_api_key",
        "github_api_key",
        "neo4j_password",
        "huggingface_api_key",
        "test_neo4j_password",
    ]

    # 验证 EXPECTED_SETTINGS_DEFAULTS 是否准确反映了加载的状态
    for key, actual_value in actual_settings_dict.items():
        # 跳过敏感键
        if key in ignore_keys:
            continue

        # 确保实际加载的键在我们的预期默认值字典中
        assert key in EXPECTED_SETTINGS_DEFAULTS, (
            f"在 Settings 对象中发现了未预期的属性 '{key}' (值: {actual_value})"
        )
        # 获取预期的默认值
        expected_value = EXPECTED_SETTINGS_DEFAULTS[key]

        # --- 值比较逻辑 (与上一个测试类似) ---
        if (
            key in ["database_url", "neo4j_uri", "test_database_url", "test_neo4j_uri"]
            and expected_value is not None
        ):
            actual_value_str = str(actual_value) if actual_value else None
            expected_value_str = str(expected_value)
            assert actual_value_str == expected_value_str, (
                f"配置键 '{key}' 不匹配。预期默认值: '{expected_value_str}', 实际: '{actual_value_str}'"
            )
        # 处理 Pydantic 的特殊 URL 或 DSN 类型
        elif isinstance(expected_value, (PostgresDsn, HttpUrl)):
            if expected_value is None: # 如果期望是 None
                 assert actual_value is None, (
                    f"配置键 '{key}' 不匹配。预期默认值: None, 实际: '{actual_value}'"
                 )
            else: # 否则比较字符串形式
                assert str(actual_value) == str(expected_value), (
                    f"配置键 '{key}' 不匹配。预期默认值: '{str(expected_value)}', 实际: '{str(actual_value)}'"
                )
        # 如果期望值为 None
        elif expected_value is None:
            assert actual_value is None, (
                f"配置键 '{key}' 不匹配。预期默认值: None, 实际: '{actual_value}'"
            )
        # 其他普通类型的值比较
        else:
            # 特殊处理 workers_count
            if key == "workers_count":
                assert isinstance(actual_value, int) and actual_value > 0, f"workers_count 应为正整数，实际: {actual_value}"
            else:
                assert actual_value == expected_value, (
                    f"配置键 '{key}' 不匹配。预期默认值: '{expected_value}', 实际: '{actual_value}'"
                )

# --- 以下部分似乎是一个独立的测试逻辑，用于测试 DATABASE_URL 和 TEST_DATABASE_URL 的修正 ---
# 它定义了一个内部函数 run_case 来运行不同环境变量设置下的场景。

# def test_postgres_driver_correction_logic(monkeypatch: pytest.MonkeyPatch) -> None:
#     """
#     测试 Pydantic Settings 在处理 DATABASE_URL 和 TEST_DATABASE_URL 时，
#     是否能正确处理不同的 PostgreSQL 连接方案（带或不带 +psycopg）。
#     这个测试现在修改为在 run_case 内部阻止 .env 加载，以隔离测试环境变量逻辑。
#     """

    def run_case(env_setup: Dict[str, Optional[str]]) -> Settings:
        """
        辅助函数，用于设置特定的环境变量组合，实例化 Settings，并返回实例。
        关键：此函数内部使用 monkeypatch 修改 Settings.model_config 来阻止加载 .env 文件，
              目的是隔离测试环境变量对 URL 处理的影响。
        """
        # 使用 monkeypatch 清除可能影响此测试的环境变量
        # raising=False 表示如果变量不存在，不要抛出错误
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

        # 根据传入的 env_setup 字典设置环境变量
        for k, v in env_setup.items():
            if v is not None:
                monkeypatch.setenv(k, v) # 设置环境变量
            else:
                monkeypatch.delenv(k, raising=False) # 删除环境变量

        # !!! 核心修改：临时覆盖 Settings 的 model_config 以阻止加载 .env 文件 !!!
        # 保存原始的 model_config
        original_config = Settings.model_config
        # 创建一个指向不存在的 .env 文件的测试配置
        test_config = SettingsConfigDict(
            env_file="/tmp/nonexistent_for_correction_test.env", # 使用一个唯一的、不存在的路径
            extra="ignore", # 忽略未定义的字段
        )
        # 使用 monkeypatch 将 Settings.model_config 替换为我们的测试配置
        monkeypatch.setattr(aigraphx.core.config.Settings, "model_config", test_config)

        # 实例化 Settings。此时它不会加载 .env 文件。
        settings = Settings()
        # monkeypatch 会在 run_case 函数退出时自动恢复原始的 model_config
        return settings

    # --- 测试用例 ---

    # 用例 1: TEST_DATABASE_URL 使用标准的 postgresql:// 方案
    env_1: Dict[str, Optional[str]] = {
        "DATABASE_URL": "postgresql://original:pass@host/db", # 设置一个基础 URL
        "TEST_DATABASE_URL": "postgresql://user:pass@host/db", # 测试 URL
    }
    settings_instance_1 = run_case(env_1) # 运行测试用例
    # 断言 test_database_url 不为 None
    assert settings_instance_1.test_database_url is not None
    # Pydantic V2 会保持原始方案，因此直接比较字符串
    assert (
        str(settings_instance_1.test_database_url) == "postgresql://user:pass@host/db"
    )
    # 断言 database_url 被正确加载（来自 env_1 设置）
    assert str(settings_instance_1.database_url) == "postgresql://original:pass@host/db"


    # 用例 2: TEST_DATABASE_URL 使用 postgresql+psycopg:// 方案
    env_2: Dict[str, Optional[str]] = {
        "DATABASE_URL": "postgresql://original2:pass@host/db",
        "TEST_DATABASE_URL": "postgresql+psycopg://user:pass@host/db", # 使用带 +psycopg 的方案
    }
    settings_instance_2 = run_case(env_2)
    assert settings_instance_2.test_database_url is not None
    # Pydantic V2 同样会保持原始方案
    assert (
        str(settings_instance_2.test_database_url)
        == "postgresql+psycopg://user:pass@host/db"
    )
    # 断言 database_url 被正确加载
    assert (
        str(settings_instance_2.database_url) == "postgresql://original2:pass@host/db"
    )

    # 用例 3: TEST_DATABASE_URL 环境变量未设置
    env_3: Dict[str, Optional[str]] = {
        "DATABASE_URL": "postgresql://original3:pass@host/db",
        "TEST_DATABASE_URL": None, # 明确表示不设置此环境变量
    }
    settings_instance_3 = run_case(env_3)
    # 断言 test_database_url 应该为 None (因为 Settings 类中默认是 None)
    assert settings_instance_3.test_database_url is None
    # 断言 database_url 仍然从环境变量加载
    assert str(settings_instance_3.database_url) == "postgresql://original3:pass@host/db"


# 之前用于测试旧的 getenv 逻辑的注释，现在可以移除或保留作为历史参考
# expected_db_url = EXPECTED_SETTINGS_DEFAULTS["database_url"]
# assert settings_instance_1.database_url is not None
# assert str(settings_instance_1.database_url) == str(expected_db_url)

# 测试 dotenv 加载的注释 (可选, 可能比较复杂)
# 可以模拟 load_dotenv 本身或检查日志输出
# 为简单起见，目前主要关注 os.getenv 逻辑（通过 monkeypatch 控制）。