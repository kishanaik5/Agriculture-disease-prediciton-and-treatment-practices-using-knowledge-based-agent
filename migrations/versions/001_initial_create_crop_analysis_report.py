"""create crop_analysis_report table

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-27 17:20:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create crop_analysis_report table
    op.create_table('crop_analysis_report',
    sa.Column('uid', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('user_id', sa.String(length=50), nullable=False),
    sa.Column('user_input_crop', sa.String(length=255), nullable=True),
    sa.Column('language', sa.String(length=10), nullable=True),
    sa.Column('is_mixed_cropping', sa.Boolean(), nullable=True),
    sa.Column('acres_of_land', sa.String(length=50), nullable=True),
    sa.Column('detected_crop', sa.String(length=255), nullable=True),
    sa.Column('detected_disease', sa.String(length=255), nullable=True),
    sa.Column('pathogen_type', sa.String(length=100), nullable=True),
    sa.Column('severity', sa.Text(), nullable=True),
    sa.Column('treatment', sa.Text(), nullable=True),
    sa.Column('analysis_raw', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('original_image_url', sa.Text(), nullable=True),
    sa.Column('bbox_image_url', sa.Text(), nullable=True),
    sa.Column('report_url', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('uid')
    )
    op.create_index(op.f('idx_crop_analysis_report_created_at'), 'crop_analysis_report', ['created_at'], unique=False)
    op.create_index(op.f('idx_crop_analysis_report_user_id'), 'crop_analysis_report', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('idx_crop_analysis_report_user_id'), table_name='crop_analysis_report')
    op.drop_index(op.f('idx_crop_analysis_report_created_at'), table_name='crop_analysis_report')
    op.drop_table('crop_analysis_report')
