"""create_knowledge_base_table

Revision ID: kb1234567890
Revises: cat123456789
Create Date: 2026-01-13 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = 'kb1234567890'
down_revision = 'cat123456789'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('knowledge_base',
        sa.Column('uid', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('item_name', sa.String(length=255), nullable=False),
        sa.Column('disease_name', sa.String(length=255), nullable=False),
        sa.Column('scientific_name', sa.String(length=255), nullable=True),
        sa.Column('treatment', sa.Text(), nullable=True),
        sa.Column('language', sa.String(length=10), server_default='en', nullable=True),
        
        sa.PrimaryKeyConstraint('uid'),
        schema='kissan_cv'
    )
    op.create_index(op.f('ix_kissan_cv_knowledge_base_category'), 'knowledge_base', ['category'], unique=False, schema='kissan_cv')
    op.create_index(op.f('ix_kissan_cv_knowledge_base_item_name'), 'knowledge_base', ['item_name'], unique=False, schema='kissan_cv')
    op.create_index(op.f('ix_kissan_cv_knowledge_base_disease_name'), 'knowledge_base', ['disease_name'], unique=False, schema='kissan_cv')

def downgrade():
    op.drop_index(op.f('ix_kissan_cv_knowledge_base_disease_name'), table_name='knowledge_base', schema='kissan_cv')
    op.drop_index(op.f('ix_kissan_cv_knowledge_base_item_name'), table_name='knowledge_base', schema='kissan_cv')
    op.drop_index(op.f('ix_kissan_cv_knowledge_base_category'), table_name='knowledge_base', schema='kissan_cv')
    op.drop_table('knowledge_base', schema='kissan_cv')
