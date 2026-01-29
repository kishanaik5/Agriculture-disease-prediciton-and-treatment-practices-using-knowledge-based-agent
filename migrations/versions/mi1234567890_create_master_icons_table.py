"""create_master_icons_table

Revision ID: mi1234567890
Revises: ra1234567890
Create Date: 2026-01-13 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'mi1234567890'
down_revision = 'ra1234567890'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('master_icons',
        sa.Column('uid', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.Column('category_id', sa.String(length=50), nullable=True),
        sa.Column('category_type', sa.String(length=50), nullable=True),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('name_en', sa.String(length=255), nullable=True),
        sa.Column('name_kn', sa.String(length=255), nullable=True),
        sa.Column('name_hn', sa.String(length=255), nullable=True),
        
        sa.PrimaryKeyConstraint('uid'),
        sa.UniqueConstraint('category_id', name='uq_master_icons_category_id'),
        schema='kissan_cv'
    )
    op.create_index(op.f('ix_kissan_cv_master_icons_category_id'), 'master_icons', ['category_id'], unique=True, schema='kissan_cv')
    op.create_index(op.f('ix_kissan_cv_master_icons_category_type'), 'master_icons', ['category_type'], unique=False, schema='kissan_cv')

def downgrade():
    op.drop_index(op.f('ix_kissan_cv_master_icons_category_type'), table_name='master_icons', schema='kissan_cv')
    op.drop_index(op.f('ix_kissan_cv_master_icons_category_id'), table_name='master_icons', schema='kissan_cv')
    op.drop_table('master_icons', schema='kissan_cv')
