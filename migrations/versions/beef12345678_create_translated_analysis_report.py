"""create translated_analysis_report table

Revision ID: beef12345678
Revises: f123456789ab
Create Date: 2026-01-12 13:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'beef12345678'
down_revision = '003_standardize_scientific_names'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'translated_analysis_report',
        sa.Column('uid', sa.String(), nullable=False), # BaseSchema PK
        sa.Column('report_uid', sa.String(), nullable=True), # Logical Reference to original report ID
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('category', sa.String(length=20), nullable=False),
        sa.Column('item_name', sa.String(length=255), nullable=True),
        sa.Column('disease_name', sa.String(length=255), nullable=True),
        sa.Column('scientific_name', sa.String(length=255), nullable=True),
        sa.Column('severity', sa.Text(), nullable=True),
        sa.Column('grade', sa.String(length=100), nullable=True),
        sa.Column('treatment', sa.Text(), nullable=True),
        sa.Column('analysis_raw', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('original_image_url', sa.Text(), nullable=True),
        sa.Column('bbox_image_url', sa.Text(), nullable=True),
        sa.Column('order_id', sa.String(length=100), nullable=True),
        sa.Column('payment_status', sa.String(length=20), server_default='PENDING', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('uid'),
        schema='kissan_cv'
    )
    op.create_index(op.f('ix_kissan_cv_translated_analysis_report_user_id'), 'translated_analysis_report', ['user_id'], unique=False, schema='kissan_cv')
    op.create_index(op.f('ix_kissan_cv_translated_analysis_report_report_uid'), 'translated_analysis_report', ['report_uid'], unique=False, schema='kissan_cv')


def downgrade():
    op.drop_index(op.f('ix_kissan_cv_translated_analysis_report_report_uid'), table_name='translated_analysis_report', schema='kissan_cv')
    op.drop_index(op.f('ix_kissan_cv_translated_analysis_report_user_id'), table_name='translated_analysis_report', schema='kissan_cv')
    op.drop_table('translated_analysis_report', schema='kissan_cv')
