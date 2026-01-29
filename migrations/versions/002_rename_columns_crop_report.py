"""rename_columns_crop_report

Revision ID: 002_rename_columns_crop_report
Revises: 001_initial_create_crop_analysis_report
Create Date: 2026-01-09 15:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
from app.config import init_settings

# revision identifiers, used by Alembic.
revision = '002_rename_columns_crop_report'
down_revision = 'dropcropcols01'
branch_labels = None
depends_on = None

settings = init_settings()
schema = settings.DB_SCHEMA

def upgrade() -> None:
    # 1. Rename columns
    op.alter_column('crop_analysis_report', 'detected_disease', new_column_name='disease_name', schema=schema)
    op.alter_column('crop_analysis_report', 'detected_crop', new_column_name='crop_name', schema=schema)
    
    # 2. Drop user_input_crop
    # Note: Data in user_input_crop will be lost. 
    # If preservation was needed, we would update crop_name from user_input_crop where crop_name is null before dropping.
    op.drop_column('crop_analysis_report', 'user_input_crop', schema=schema)


def downgrade() -> None:
    # 1. Add user_input_crop back
    op.add_column('crop_analysis_report', sa.Column('user_input_crop', sa.String(255), nullable=True), schema=schema)
    
    # 2. Rename columns back
    op.alter_column('crop_analysis_report', 'disease_name', new_column_name='detected_disease', schema=schema)
    op.alter_column('crop_analysis_report', 'crop_name', new_column_name='detected_crop', schema=schema)
