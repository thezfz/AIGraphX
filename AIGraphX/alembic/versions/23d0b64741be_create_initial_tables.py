# -*- coding: utf-8 -*-
"""
Alembic 迁移文件：创建初始数据库表结构。

此文件定义了数据库的初始模式，包括创建所有核心表（如 hf_models, papers, model_paper_links 等）、
它们各自的列、数据类型、约束（主键、外键、唯一约束、检查约束），以及必要的索引和触发器。

这是项目数据库模式演进的起点。

Revision ID: 23d0b64741be
Revises: None (这是第一个迁移)
Create Date: 2024-08-04 17:59:14.276074
"""

from typing import Sequence, Union # 导入类型提示

from alembic import op # 导入 Alembic 的操作对象 (op)，用于执行数据库模式更改命令
import sqlalchemy as sa # 导入 SQLAlchemy 库，Alembic 使用它来定义列类型等 (虽然这里主要用原生 SQL)


# revision identifiers, used by Alembic.
revision: str = "23d0b64741be" # 当前迁移的唯一标识符
down_revision: Union[str, None] = None # 指向此迁移所基于的上一个迁移 ID (对于第一个迁移是 None)
branch_labels: Union[str, Sequence[str], None] = None # 用于支持多分支迁移历史 (通常为 None)
depends_on: Union[str, Sequence[str], None] = None # 用于声明此迁移依赖于其他分支的迁移 (通常为 None)


def upgrade() -> None:
    """
    定义将数据库模式升级到此版本的操作。
    执行 `alembic upgrade head` 或 `alembic upgrade 23d0b64741be` 时会调用此函数。
    """
    # ### Alembic 自动生成的命令（可能已手动调整） ###
    # 使用 op.execute 执行原生 SQL 语句来创建表和索引。
    # IF NOT EXISTS 确保如果表已存在（例如手动创建或之前迁移失败），命令不会出错。

    # --- 创建 hf_models 表 ---
    # 存储 Hugging Face 模型元数据
    op.execute("""
    CREATE TABLE IF NOT EXISTS hf_models (
        hf_model_id VARCHAR(255) PRIMARY KEY, -- 模型 ID，主键
        hf_author VARCHAR(255),               -- 作者
        hf_sha VARCHAR(64),                   -- Git SHA
        hf_last_modified TIMESTAMPTZ,         -- 最后修改时间 (带时区)
        hf_downloads INTEGER,                 -- 下载量
        hf_likes INTEGER,                     -- 点赞数
        hf_tags JSONB,                        -- 标签 (使用 JSONB 类型存储)
        hf_pipeline_tag VARCHAR(100),         -- Pipeline 标签
        hf_library_name VARCHAR(100),         -- 库名称
        created_at TIMESTAMPTZ DEFAULT NOW(), -- 创建时间，默认为当前时间
        updated_at TIMESTAMPTZ DEFAULT NOW()  -- 更新时间，默认为当前时间
    );
    """)
    print("Created hf_models table.")

    # --- 创建 papers 表 ---
    # 存储论文元数据
    # !!! 注意：添加了 categories TEXT 列（原注释说明，但实际使用了 TEXT，可能存储 JSON 字符串）!!!
    op.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        paper_id SERIAL PRIMARY KEY,               -- 论文内部 ID，自增主键
        pwc_id VARCHAR(255) UNIQUE,                -- PapersWithCode ID，唯一
        arxiv_id_base VARCHAR(50) UNIQUE,          -- arXiv ID (不带版本)，唯一
        arxiv_id_versioned VARCHAR(60),            -- arXiv ID (带版本)
        title TEXT,                                -- 标题
        authors JSONB,                             -- 作者列表 (JSONB)
        summary TEXT,                              -- 摘要
        published_date DATE,                       -- 发表日期
        updated_date TIMESTAMPTZ,                  -- 更新日期 (来自 PWC 或其他来源)
        pdf_url TEXT,                              -- PDF 链接
        doi VARCHAR(255),                          -- DOI
        primary_category VARCHAR(50),              -- 主要分类
        categories TEXT,                           -- 分类列表 (存储为 JSON 字符串)
        pwc_title TEXT,                            -- PWC 上的标题
        pwc_url TEXT,                              -- PWC 链接
        area VARCHAR(50),                          -- 研究领域
        created_at TIMESTAMPTZ DEFAULT NOW(),      -- 创建时间
        updated_at TIMESTAMPTZ DEFAULT NOW(),      -- 更新时间 (将被触发器自动更新)
        -- 添加检查约束，确保 pwc_id 或 arxiv_id_base 至少有一个不为空
        CONSTRAINT chk_paper_identifier CHECK (pwc_id IS NOT NULL OR arxiv_id_base IS NOT NULL)
    );
    """)
    print("Created papers table.")
    # --- 为 papers 表创建索引 ---
    # 提高基于 pwc_id, arxiv_id_base, area, published_date 的查询性能
    op.execute("CREATE INDEX IF NOT EXISTS idx_papers_pwc_id ON papers(pwc_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id_base ON papers(arxiv_id_base);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_papers_area ON papers(area);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date);")
    print("Created indexes on papers table.")

    # --- 创建 model_paper_links 表 ---
    # 存储 Hugging Face 模型和论文之间的关联关系 (多对多)
    op.execute("""
    CREATE TABLE IF NOT EXISTS model_paper_links (
        link_id SERIAL PRIMARY KEY,                   -- 关联 ID，自增主键
        -- 外键关联到 hf_models 表，ON DELETE CASCADE 表示如果模型被删除，关联记录也自动删除
        hf_model_id VARCHAR(255) NOT NULL REFERENCES hf_models(hf_model_id) ON DELETE CASCADE,
        -- 外键关联到 papers 表，ON DELETE CASCADE 表示如果论文被删除，关联记录也自动删除
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ DEFAULT NOW(),      -- 创建时间
        -- 唯一约束，确保同一对模型和论文只能关联一次
        UNIQUE (hf_model_id, paper_id)
    );
    """)
    print("Created model_paper_links table.")
    # --- 为 model_paper_links 表创建索引 ---
    # 提高基于模型 ID 或论文 ID 查询关联的性能
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_paper_links_model ON model_paper_links(hf_model_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_paper_links_paper ON model_paper_links(paper_id);")
    print("Created indexes on model_paper_links table.")

    # --- 创建 pwc_tasks 表 ---
    # 存储论文相关的任务信息 (来自 PapersWithCode)
    op.execute("""
    CREATE TABLE IF NOT EXISTS pwc_tasks (
        task_id SERIAL PRIMARY KEY,                    -- 任务关联 ID，自增主键
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE, -- 外键关联论文
        task_name VARCHAR(255) NOT NULL,               -- 任务名称
        created_at TIMESTAMPTZ DEFAULT NOW(),      -- 创建时间
        -- 唯一约束，确保一篇论文下的同一个任务名称只记录一次
        UNIQUE (paper_id, task_name)
    );
    """)
    print("Created pwc_tasks table.")
    # --- 为 pwc_tasks 表创建索引 ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_tasks_paper ON pwc_tasks(paper_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_tasks_name ON pwc_tasks(task_name);")
    print("Created indexes on pwc_tasks table.")

    # --- 创建 pwc_datasets 表 ---
    # 存储论文相关的数据集信息 (来自 PapersWithCode)
    op.execute("""
    CREATE TABLE IF NOT EXISTS pwc_datasets (
        dataset_id SERIAL PRIMARY KEY,                 -- 数据集关联 ID，自增主键
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE, -- 外键关联论文
        dataset_name VARCHAR(255) NOT NULL,            -- 数据集名称
        created_at TIMESTAMPTZ DEFAULT NOW(),       -- 创建时间
        -- 唯一约束，确保一篇论文下的同一个数据集名称只记录一次
        UNIQUE (paper_id, dataset_name)
    );
    """)
    print("Created pwc_datasets table.")
    # --- 为 pwc_datasets 表创建索引 ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_datasets_paper ON pwc_datasets(paper_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_datasets_name ON pwc_datasets(dataset_name);")
    print("Created indexes on pwc_datasets table.")

    # --- 创建 pwc_methods 表 ---
    # 存储论文相关的方法信息 (来自 PapersWithCode)
    op.execute("""
    CREATE TABLE IF NOT EXISTS pwc_methods (
        method_id SERIAL PRIMARY KEY,                  -- 方法关联 ID，自增主键
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE, -- 外键关联论文
        method_name VARCHAR(255) NOT NULL,             -- 方法名称
        created_at TIMESTAMPTZ DEFAULT NOW(),        -- 创建时间
        -- 唯一约束，确保一篇论文下的同一个方法名称只记录一次
        UNIQUE (paper_id, method_name)
    );
    """)
    print("Created pwc_methods table.")
    # --- 为 pwc_methods 表创建索引 ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_methods_paper ON pwc_methods(paper_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_methods_name ON pwc_methods(method_name);")
    print("Created indexes on pwc_methods table.")

    # --- 创建 pwc_repositories 表 ---
    # 存储论文相关的代码仓库信息 (来自 PapersWithCode)
    op.execute("""
    CREATE TABLE IF NOT EXISTS pwc_repositories (
        repo_id SERIAL PRIMARY KEY,                    -- 仓库关联 ID，自增主键
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE, -- 外键关联论文
        url TEXT,                                      -- 仓库 URL
        stars INTEGER,                                 -- 星标数
        is_official BOOLEAN,                           -- 是否官方仓库
        framework VARCHAR(100),                        -- 使用的框架
        created_at TIMESTAMPTZ DEFAULT NOW(),        -- 创建时间
        -- 唯一约束，确保一篇论文下的同一个仓库 URL 只记录一次
        UNIQUE (paper_id, url)
    );
    """)
    print("Created pwc_repositories table.")
    # --- 为 pwc_repositories 表创建索引 ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_repositories_paper ON pwc_repositories(paper_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pwc_repositories_url ON pwc_repositories(url);") # TEXT 列也可以创建索引
    print("Created indexes on pwc_repositories table.")

    # --- 创建触发器函数 ---
    # 这个 PostgreSQL 函数会在被触发时，自动将 `updated_at` 列更新为当前时间
    op.execute("""
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
       -- 将 NEW 记录（即被更新的行）的 updated_at 列设置为当前时间
       NEW.updated_at = NOW();
       -- 返回修改后的行
       RETURN NEW;
    END;
    $$ language 'plpgsql';
    """)
    print("Created update_updated_at_column trigger function.")

    # --- 将触发器应用到需要自动更新 updated_at 的表 ---
    # 为 hf_models 表创建触发器，在每次更新 (BEFORE UPDATE) 行数据时，执行上面的函数
    op.execute("""
    CREATE TRIGGER update_hf_models_updated_at
    BEFORE UPDATE ON hf_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    # 为 papers 表创建触发器
    op.execute("""
    CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    print("Applied update_updated_at triggers to tables.")

    # ### Alembic 命令结束 ###


def downgrade() -> None:
    """
    定义将数据库模式从此版本降级（回滚）的操作。
    执行 `alembic downgrade 23d0b64741be` （如果这是 head）或 `alembic downgrade <previous_revision>` 时会调用。
    操作应与 `upgrade` 中的操作顺序相反。
    """
    # ### Alembic 自动生成的命令（可能已手动调整） ###

    # --- 首先删除触发器 ---
    # 必须在删除函数或表之前删除依赖于它们的触发器
    op.execute("DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;")
    op.execute("DROP TRIGGER IF EXISTS update_hf_models_updated_at ON hf_models;")
    print("Dropped update_updated_at triggers.")

    # --- 然后删除触发器函数 ---
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
    print("Dropped update_updated_at_column trigger function.")

    # --- 删除表 ---
    # 以与创建大致相反的顺序删除，或者考虑外键依赖关系。
    # 使用 CASCADE 可以自动删除依赖于此表的对象（如外键、索引），简化降级操作，但需谨慎。
    op.execute("DROP TABLE IF EXISTS pwc_repositories CASCADE;")
    print("Dropped pwc_repositories table.")
    op.execute("DROP TABLE IF EXISTS pwc_datasets CASCADE;")
    print("Dropped pwc_datasets table.")
    op.execute("DROP TABLE IF EXISTS pwc_methods CASCADE;")
    print("Dropped pwc_methods table.")
    op.execute("DROP TABLE IF EXISTS pwc_tasks CASCADE;")
    print("Dropped pwc_tasks table.")
    op.execute("DROP TABLE IF EXISTS model_paper_links CASCADE;")
    print("Dropped model_paper_links table.")
    op.execute("DROP TABLE IF EXISTS papers CASCADE;")
    print("Dropped papers table.")
    op.execute("DROP TABLE IF EXISTS hf_models CASCADE;")
    print("Dropped hf_models table.")

    # 注意：与表关联的索引通常在删除表时会自动删除。
    # 除非索引是独立创建的，否则通常不需要显式的 DROP INDEX 命令。

    # ### Alembic 命令结束 ###