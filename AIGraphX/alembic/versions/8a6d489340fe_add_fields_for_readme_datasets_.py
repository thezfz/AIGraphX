"""Add fields for readme, datasets, conference, repo details, abstracts

Revision ID: 8a6d489340fe
Revises: 23d0b64741be
Create Date: 2025-04-23 01:22:27.675560

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8a6d489340fe"
down_revision: Union[str, None] = "23d0b64741be"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands manually written ###
    print(
        "Applying upgrade: Add fields for readme, datasets, conference, repo details, abstracts"
    )

    # Add column to hf_models
    op.add_column("hf_models", sa.Column("hf_readme_content", sa.TEXT(), nullable=True))
    # PostgreSQL uses TEXT[] for string arrays. Adjust if using a different DB.
    op.add_column(
        "hf_models", sa.Column("hf_dataset_links", sa.JSONB(), nullable=True)
    )
    print("Added columns to hf_models.")

    # Add column to papers
    op.add_column(
        "papers", sa.Column("conference", sa.VARCHAR(length=255), nullable=True)
    )
    print("Added column to papers.")

    # Add columns to pwc_repositories
    op.add_column(
        "pwc_repositories", sa.Column("license", sa.VARCHAR(length=100), nullable=True)
    )
    op.add_column(
        "pwc_repositories", sa.Column("language", sa.VARCHAR(length=100), nullable=True)
    )
    print("Added columns to pwc_repositories.")

    # Add column to pwc_tasks
    # op.add_column("pwc_tasks", sa.Column("task_abstract", sa.TEXT(), nullable=True))
    # print("Added column to pwc_tasks.")

    # Add column to pwc_datasets
    # op.add_column(
    #     "pwc_datasets", sa.Column("dataset_abstract", sa.TEXT(), nullable=True)
    # )
    # print("Added column to pwc_datasets.")

    print("Upgrade completed.")
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands manually written ###
    print(
        "Applying downgrade: Remove fields for readme, datasets, conference, repo details, abstracts"
    )

    # Drop columns (in reverse order of table modification if needed, but generally order doesn't matter for columns)
    # op.drop_column("pwc_datasets", "dataset_abstract")
    # print("Dropped column from pwc_datasets.")

    # op.drop_column("pwc_tasks", "task_abstract")
    # print("Dropped column from pwc_tasks.")

    op.drop_column("pwc_repositories", "language")
    op.drop_column("pwc_repositories", "license")
    print("Dropped columns from pwc_repositories.")

    op.drop_column("papers", "conference")
    print("Dropped column from papers.")

    op.drop_column("hf_models", "hf_dataset_links")
    op.drop_column("hf_models", "hf_readme_content")
    print("Dropped columns from hf_models.")

    print("Downgrade completed.")
    # ### end Alembic commands ###
