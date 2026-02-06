"""Rename name_hn to name_hi in master_icons

Revision ID: rename_hn_to_hi
Revises: 
Create Date: 2026-02-05

"""
from alembic import op

# revision identifiers
revision = 'rename_hn_to_hi'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename column from name_hn to name_hi
    op.execute('ALTER TABLE kissan_cv.master_icons RENAME COLUMN name_hn TO name_hi')
    pass


def downgrade() -> None:
    # Revert: rename back from name_hi to name_hn
    op.execute('ALTER TABLE kissan_cv.master_icons RENAME COLUMN name_hi TO name_hn')
