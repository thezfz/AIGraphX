# -*- coding: utf-8 -*-
"""
Alembic 迁移文件：为多个表添加新字段。

此迁移在初始表结构的基础上，为 hf_models, papers, pwc_repositories, pwc_tasks, pwc_datasets 表
添加了新的列，以存储更多的信息，如 README 内容、数据集链接、会议信息、代码仓库详情、任务/数据集摘要等。

Revision ID: 8a6d489340fe
Revises: 23d0b64741be (此迁移基于上一个创建初始表的迁移)
Create Date: 2025-04-23 01:22:27.675560 (注意：这个日期看起来是未来的，可能是笔误)
"""

from typing import Sequence, Union  # 导入类型提示

from alembic import op  # 导入 Alembic 操作对象
import sqlalchemy as sa  # 导入 SQLAlchemy，用于定义列类型 (例如 sa.TEXT(), sa.VARCHAR(), sa.ARRAY())


# revision identifiers, used by Alembic.
revision: str = "8a6d489340fe"  # 当前迁移的 ID
down_revision: Union[str, None] = "23d0b64741be"  # 指向它所基于的前一个迁移 ID
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    定义将数据库模式升级到此版本的操作。
    执行 `alembic upgrade head` 或 `alembic upgrade 8a6d489340fe` 时调用。
    主要是添加新的列。
    """
    # ### 手动编写的 Alembic 命令 ###
    print(  # 在 Alembic 运行时输出信息，告知正在进行的操作
        "正在应用升级：为 readme, datasets, conference, repo details, abstracts 添加字段"
    )

    # --- 为 hf_models 表添加列 ---
    # 添加 hf_readme_content 列，类型为 TEXT，允许为空
    op.add_column("hf_models", sa.Column("hf_readme_content", sa.TEXT(), nullable=True))
    # 添加 hf_dataset_links 列，类型为 TEXT 数组 (PostgreSQL 支持)，允许为空
    op.add_column(
        "hf_models", sa.Column("hf_dataset_links", sa.ARRAY(sa.TEXT()), nullable=True)
    )
    print("已向 hf_models 表添加列。")

    # --- 为 papers 表添加列 ---
    # 添加 conference 列，类型为 VARCHAR(255)，允许为空
    op.add_column(
        "papers", sa.Column("conference", sa.VARCHAR(length=255), nullable=True)
    )
    print("已向 papers 表添加列。")

    # --- 为 pwc_repositories 表添加列 ---
    # 添加 license 列，类型为 VARCHAR(100)，允许为空
    op.add_column(
        "pwc_repositories", sa.Column("license", sa.VARCHAR(length=100), nullable=True)
    )
    # 添加 language 列，类型为 VARCHAR(100)，允许为空
    op.add_column(
        "pwc_repositories", sa.Column("language", sa.VARCHAR(length=100), nullable=True)
    )
    print("已向 pwc_repositories 表添加列。")

    # --- 为 pwc_tasks 表添加列 ---
    # 添加 task_abstract 列，类型为 TEXT，允许为空
    op.add_column("pwc_tasks", sa.Column("task_abstract", sa.TEXT(), nullable=True))
    print("已向 pwc_tasks 表添加列。")

    # --- 为 pwc_datasets 表添加列 ---
    # 添加 dataset_abstract 列，类型为 TEXT，允许为空
    op.add_column(
        "pwc_datasets", sa.Column("dataset_abstract", sa.TEXT(), nullable=True)
    )
    print("已向 pwc_datasets 表添加列。")

    print("升级完成。")
    # ### Alembic 命令结束 ###


def downgrade() -> None:
    """
    定义将数据库模式从此版本降级（回滚）的操作。
    执行 `alembic downgrade 8a6d489340fe` 时调用。
    主要是删除在 `upgrade` 中添加的列。
    """
    # ### 手动编写的 Alembic 命令 ###
    print(  # 输出降级信息
        "正在应用降级：移除 readme, datasets, conference, repo details, abstracts 的字段"
    )

    # --- 删除列 ---
    # 删除操作通常与添加操作的顺序无关，但按表的反向顺序删除也无妨。
    op.drop_column("pwc_datasets", "dataset_abstract")
    print("已从 pwc_datasets 表删除列。")

    op.drop_column("pwc_tasks", "task_abstract")
    print("已从 pwc_tasks 表删除列。")

    op.drop_column("pwc_repositories", "language")
    op.drop_column("pwc_repositories", "license")
    print("已从 pwc_repositories 表删除列。")

    op.drop_column("papers", "conference")
    print("已从 papers 表删除列。")

    op.drop_column("hf_models", "hf_dataset_links")
    op.drop_column("hf_models", "hf_readme_content")
    print("已从 hf_models 表删除列。")

    print("降级完成。")
    # ### Alembic 命令结束 ###
