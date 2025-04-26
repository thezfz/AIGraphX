# -*- coding: utf-8 -*-
"""
脚本目的：将指定项目目录下的代码文件转换为 Markdown 文件，
         并保持原始目录结构，用于导出笔记。
"""

import os
from pathlib import Path
import logging

# --- 配置区域 ---

# 源项目目录 (你的后端项目路径)
SOURCE_DIR = Path("/home/thezfz/MVP/AIGraphX")

# 输出 Markdown 文件的目标目录 (建议放在源目录之外)
TARGET_DIR = Path("/home/thezfz/MVP/Backend_Markdown_Notes")

# 需要转换的文件扩展名集合 (小写)
# 添加或删除你认为需要包含/排除的扩展名
CODE_EXTENSIONS = {
    ".py",  # Python 文件
    ".ini",  # INI 配置文件
    ".toml",  # TOML 配置文件
    ".yml",  # YAML 配置文件
    ".yaml",  # YAML 配置文件
    ".sh",  # Shell 脚本
    ".sql",  # SQL 文件
    ".md",  # Markdown 文件 (也转换，保留格式)
    ".txt",  # 普通文本文件
    # 可以添加其他如 .js, .ts, .html, .css 等，如果需要的话
}

# 需要转换的特定文件名 (无论扩展名如何，例如 Dockerfile/Containerfile)
INCLUDE_FILES = {
    "Containerfile",
    "Dockerfile",
    "compose.yml",  # 也包含 compose 文件
    "compose.test.yml",
}

# 需要完全排除的目录名称集合
EXCLUDE_DIRS = {
    ".git",  # Git 仓库元数据
    "__pycache__",  # Python 字节码缓存
    ".venv",  # 虚拟环境 (常用名称)
    "env",  # 虚拟环境 (常用名称)
    "node_modules",  # Node.js 依赖 (如果存在)
    "logs",  # 日志文件目录
    "data",  # 运行时数据目录 (如 Faiss 索引)
    "test_data",  # 测试使用的静态数据
    "build",  # Python 构建目录
    "dist",  # Python 打包目录
    ".mypy_cache",  # MyPy 缓存
    ".pytest_cache",  # Pytest 缓存
    ".ruff_cache",  # Ruff 缓存
    "htmlcov",  # Coverage 报告目录
    # 可以添加其他需要排除的目录，例如 '.vscode', '.idea'
}

# 需要排除的特定文件名集合
EXCLUDE_FILES = {
    ".env",  # 包含敏感信息的环境变量文件
    ".env.example",  # 环境变量示例文件 (可以选择包含)
    # 可以添加其他需要排除的特定文件
}

# 文件扩展名到 Markdown 代码块语言标识符的映射
# 用于在 Markdown 中正确高亮代码
LANG_MAP = {
    ".py": "python",
    ".ini": "ini",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "bash",
    ".sql": "sql",
    "Containerfile": "dockerfile",  # 特定文件名映射
    "Dockerfile": "dockerfile",
    "compose.yml": "yaml",
    "compose.test.yml": "yaml",
    ".md": "markdown",
    ".txt": "text",
}

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- 主函数 ---
def convert_to_markdown() -> None:
    """
    遍历源目录，将符合条件的代码文件转换为 Markdown 文件，
    输出到目标目录，并保持目录结构。
    """
    if not SOURCE_DIR.is_dir():
        logging.error(f"错误：源目录 '{SOURCE_DIR}' 不存在或不是一个目录。")
        return

    # 创建目标目录 (如果不存在)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"开始转换 '{SOURCE_DIR}' 到 '{TARGET_DIR}'")

    converted_count = 0
    skipped_count = 0

    # 使用 rglob('*') 递归遍历所有文件和目录
    for source_path in SOURCE_DIR.rglob("*"):
        try:
            relative_path = source_path.relative_to(SOURCE_DIR)

            # --- 1. 排除目录 ---
            # 检查路径的任何部分是否是需要排除的目录名
            is_excluded_dir = False
            for part in relative_path.parts:
                if part in EXCLUDE_DIRS:
                    is_excluded_dir = True
                    break
            if is_excluded_dir:
                # 如果是目录，跳过其内容；如果是文件，直接跳过
                if source_path.is_dir():
                    # logging.debug(f"Skipping excluded directory and its contents: {relative_path}")
                    pass  # os.walk 的 prune 更直接，但 rglob 需要手动跳过文件
                else:
                    # logging.debug(f"Skipping file within excluded directory: {relative_path}")
                    skipped_count += 1
                continue  # 跳过当前项

            # --- 2. 处理文件 ---
            if source_path.is_file():
                # 2a. 排除特定文件
                if source_path.name in EXCLUDE_FILES:
                    logging.info(f"跳过排除的文件: {relative_path}")
                    skipped_count += 1
                    continue

                # 2b. 检查是否为需要转换的文件类型
                is_code_ext = source_path.suffix.lower() in CODE_EXTENSIONS
                is_included_filename = source_path.name in INCLUDE_FILES

                if not (is_code_ext or is_included_filename):
                    # logging.debug(f"Skipping non-code file: {relative_path}")
                    skipped_count += 1
                    continue

                # --- 3. 读取和检查内容 ---
                try:
                    content = source_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as read_err:
                    logging.warning(f"无法读取文件 {relative_path}: {read_err}. 跳过。")
                    skipped_count += 1
                    continue

                # 跳过空文件或只包含空白字符的文件
                if not content.strip():
                    logging.info(f"跳过空文件: {relative_path}")
                    skipped_count += 1
                    continue

                # --- 4. 构造目标路径和内容 ---
                # 目标路径保持相对结构，扩展名改为 .md
                target_path = TARGET_DIR / relative_path.with_suffix(".md")

                # 确保目标文件的父目录存在
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # 获取 Markdown 代码块的语言标识符
                lang = LANG_MAP.get(source_path.suffix.lower())  # 优先使用扩展名
                if source_path.name in LANG_MAP:  # 如果文件名有特定映射，则覆盖
                    lang = LANG_MAP[source_path.name]
                if not lang:  # 如果没有找到映射，则留空
                    lang = ""

                # 构建 Markdown 内容
                md_content = f"# 原始路径: {relative_path}\n\n"  # 添加原始路径作为标题
                md_content += f"```{lang}\n"  # 开始代码块，指定语言
                md_content += content  # 插入文件内容
                md_content += "\n```\n"  # 结束代码块

                # --- 5. 写入 Markdown 文件 ---
                try:
                    target_path.write_text(md_content, encoding="utf-8")
                    logging.info(
                        f"已转换: {relative_path} -> {target_path.relative_to(TARGET_DIR)}"
                    )
                    converted_count += 1
                except Exception as write_err:
                    logging.error(
                        f"写入 Markdown 文件 {target_path} 时出错: {write_err}"
                    )
                    skipped_count += 1

        except Exception as e:
            logging.error(f"处理路径 {source_path} 时发生意外错误: {e}", exc_info=True)
            skipped_count += 1

    logging.info(
        f"转换完成。成功转换 {converted_count} 个文件，跳过 {skipped_count} 个文件/目录。"
    )
    logging.info(f"Markdown 笔记已保存至: {TARGET_DIR.resolve()}")


# --- 脚本入口 ---
if __name__ == "__main__":
    convert_to_markdown()
