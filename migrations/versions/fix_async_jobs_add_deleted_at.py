"""add deleted_at to async_jobs

Revision ID: fix_async_schema
Revises: job_status_table
Create Date: 2026-01-14 17:55:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fix_async_schema'
down_revision = 'job_status_table'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('async_jobs', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True), schema='kissan_cv')


def downgrade() -> None:
    op.drop_column('async_jobs', 'deleted_at', schema='kissan_cv')
