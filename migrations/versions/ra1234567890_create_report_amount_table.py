"""create_report_amount_table

Revision ID: ra1234567890
Revises: kb1234567890
Create Date: 2026-01-13 14:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = 'ra1234567890'
down_revision = 'kb1234567890'
branch_labels = None
depends_on = None

def upgrade():
    # Create table
    op.create_table('report_amount',
        sa.Column('uid', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        
        sa.PrimaryKeyConstraint('uid'),
        sa.UniqueConstraint('category', name='uq_report_amount_category'),
        schema='kissan_cv'
    )
    op.create_index(op.f('ix_kissan_cv_report_amount_category'), 'report_amount', ['category'], unique=True, schema='kissan_cv')

    # Insert initial data
    op.execute("INSERT INTO kissan_cv.report_amount (uid, category, amount, created_at) VALUES ('" + str(uuid.uuid4()) + "', 'crop', 9.0, now())")
    op.execute("INSERT INTO kissan_cv.report_amount (uid, category, amount, created_at) VALUES ('" + str(uuid.uuid4()) + "', 'fruit', 10.0, now())")
    op.execute("INSERT INTO kissan_cv.report_amount (uid, category, amount, created_at) VALUES ('" + str(uuid.uuid4()) + "', 'vegetable', 11.0, now())")

def downgrade():
    op.drop_index(op.f('ix_kissan_cv_report_amount_category'), table_name='report_amount', schema='kissan_cv')
    op.drop_table('report_amount', schema='kissan_cv')
