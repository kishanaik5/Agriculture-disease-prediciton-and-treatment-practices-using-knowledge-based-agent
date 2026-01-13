"""add_coupon_used_column

Revision ID: cu1234567890
Revises: mi1234567890
Create Date: 2026-01-13 17:50:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'cu1234567890'
down_revision = 'mi1234567890'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('transaction_table', sa.Column('coupon_used', sa.String(length=50), nullable=True), schema='kissan_cv')

def downgrade():
    op.drop_column('transaction_table', 'coupon_used', schema='kissan_cv')
