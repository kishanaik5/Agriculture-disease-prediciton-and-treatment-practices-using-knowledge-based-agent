"""increase report_id length

Revision ID: fix_report_id_len
Revises: fix_async_schema
Create Date: 2026-01-14 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fix_report_id_len'
down_revision = 'fix_async_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Increase length to 100
    op.alter_column('async_jobs', 'report_id',
               existing_type=sa.String(length=50),
               type_=sa.String(length=100),
               existing_nullable=True,
               schema='kissan_cv')


def downgrade() -> None:
    op.alter_column('async_jobs', 'report_id',
               existing_type=sa.String(length=100),
               type_=sa.String(length=50),
               existing_nullable=True,
               schema='kissan_cv')
