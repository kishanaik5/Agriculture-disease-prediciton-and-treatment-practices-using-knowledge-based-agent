"""rename_category_to_category_type

Revision ID: cat123456789
Revises: beef12345678
Create Date: 2026-01-12 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'cat123456789'
down_revision = 'beef12345678'
branch_labels = None
depends_on = None

def upgrade():
    op.alter_column('translated_analysis_report', 'category', new_column_name='category_type', schema='kissan_cv')

def downgrade():
    op.alter_column('translated_analysis_report', 'category_type', new_column_name='category', schema='kissan_cv')
