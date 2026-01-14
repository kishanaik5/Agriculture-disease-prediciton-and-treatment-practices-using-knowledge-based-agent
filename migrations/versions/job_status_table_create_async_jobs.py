"""create async_jobs table

Revision ID: job_status_table
Revises: f123456789ab, ra1234567890, cu1234567890
Create Date: 2026-01-14 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'job_status_table'
down_revision = ('f123456789ab', 'ra1234567890', 'cu1234567890')
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('async_jobs',
        sa.Column('uid', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('job_id', sa.String(length=50), nullable=True),
        sa.Column('user_id', sa.String(length=50), nullable=True),
        sa.Column('report_id', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=True),
        sa.Column('stage', sa.String(length=50), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.PrimaryKeyConstraint('uid'),
        schema='kissan_cv'
    )
    op.create_index(op.f('ix_kissan_cv_async_jobs_job_id'), 'async_jobs', ['job_id'], unique=True, schema='kissan_cv')
    op.create_index(op.f('ix_kissan_cv_async_jobs_report_id'), 'async_jobs', ['report_id'], unique=False, schema='kissan_cv')
    op.create_index(op.f('ix_kissan_cv_async_jobs_user_id'), 'async_jobs', ['user_id'], unique=False, schema='kissan_cv')


def downgrade() -> None:
    op.drop_index(op.f('ix_kissan_cv_async_jobs_user_id'), table_name='async_jobs', schema='kissan_cv')
    op.drop_index(op.f('ix_kissan_cv_async_jobs_report_id'), table_name='async_jobs', schema='kissan_cv')
    op.drop_index(op.f('ix_kissan_cv_async_jobs_job_id'), table_name='async_jobs', schema='kissan_cv')
    op.drop_table('async_jobs', schema='kissan_cv')
