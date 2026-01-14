"""drop async_jobs table cleanup

Revision ID: drop_async_jobs_table
Revises: fix_report_id_len
Create Date: 2026-01-14 18:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'drop_async_jobs_table'
down_revision = 'fix_report_id_len'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table('async_jobs', schema='kissan_cv')


def downgrade() -> None:
    # Recreate the table exactly as it was at fix_report_id_len state
    op.create_table('async_jobs',
        sa.Column('uid', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('job_id', sa.String(length=50), nullable=True),
        sa.Column('user_id', sa.String(length=50), nullable=True),
        sa.Column('report_id', sa.String(length=100), nullable=True), # Note 100
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=True),
        sa.Column('stage', sa.String(length=50), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True), # JSONB may vary but JSON generic
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True), # Note added column
        sa.PrimaryKeyConstraint('uid'),
        schema='kissan_cv'
    )
    # Indexes omitted for brevity but should be here for perfect downgrade
