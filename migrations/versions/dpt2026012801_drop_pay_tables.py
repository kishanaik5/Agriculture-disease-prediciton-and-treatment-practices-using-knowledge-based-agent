"""drop payment tables

Revision ID: drop_pay_tables_2026
Revises: drop_async_jobs_cleanup
Create Date: 2026-01-28 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'dpt2026012801'
down_revision = 'drop_async_jobs_table' # Ensure this matches the actual latest
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Drop tables if they exist
    op.execute("DROP TABLE IF EXISTS kissan_cv.transaction_table CASCADE;")
    op.execute("DROP TABLE IF EXISTS kissan_cv.report_amount CASCADE;")


def downgrade() -> None:
    # Re-create tables (simplified schema for restore)
    op.create_table('report_amount',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('uid', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        schema='kissan_cv'
    )
    op.create_table('transaction_table',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('order_id', sa.String(length=100), nullable=False),
        sa.Column('payment_status', sa.String(length=20), nullable=True),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('uid', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        schema='kissan_cv'
    )
