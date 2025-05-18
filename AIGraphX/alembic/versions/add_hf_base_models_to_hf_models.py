"""add_hf_base_models_to_hf_models

Revision ID: <NEW_REVISION_ID>  # Replace with the actual new revision ID
Revises: 8a6d489340fe
Create Date: <YYYY-MM-DD HH:MM:SS.ffffff> # Replace with actual creation date

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql # Import postgresql dialect



# revision identifiers, used by Alembic.
revision: str = "<NEW_REVISION_ID>" # Replace with the actual new revision ID
down_revision: Union[str, None] = "8a6d489340fe" # Points to the previous migration
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    print("Applying upgrade: Add hf_base_models column to hf_models table")
    op.add_column(
        "hf_models",
        sa.Column("hf_base_models", postgresql.JSONB(), nullable=True) # Use postgresql.JSONB()
    )
    print("Added hf_base_models column to hf_models.")
    print("Upgrade completed.")


def downgrade() -> None:
    """Downgrade schema."""
    print("Applying downgrade: Remove hf_base_models column from hf_models table")
    op.drop_column("hf_models", "hf_base_models")
    print("Dropped hf_base_models column from hf_models.")
    print("Downgrade completed.")
