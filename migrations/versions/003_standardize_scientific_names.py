"""standardize_scientific_names

Revision ID: 003_standardize_scientific_names
Revises: 002_rename_columns_crop_report
Create Date: 2026-01-09 16:35:00.000000

"""
from alembic import op
import sqlalchemy as sa
from app.config import init_settings

# revision identifiers, used by Alembic.
revision = '003_standardize_scientific_names'
down_revision = '002_rename_columns_crop_report'
branch_labels = None
depends_on = None

settings = init_settings()
schema = settings.DB_SCHEMA

def upgrade() -> None:
    # 1. crop_analysis_report: pathogen_type -> scientific_name
    op.alter_column('crop_analysis_report', 'pathogen_type', new_column_name='scientific_name', schema=schema)
    
    # 2. fruit_analysis_report: pathogen_scientific_name -> scientific_name
    op.alter_column('fruit_analysis_report', 'pathogen_scientific_name', new_column_name='scientific_name', schema=schema)


def downgrade() -> None:
    # 1. Reverse crop_analysis_report
    op.alter_column('crop_analysis_report', 'scientific_name', new_column_name='pathogen_type', schema=schema)
    
    # 2. Reverse fruit_analysis_report
    op.alter_column('fruit_analysis_report', 'scientific_name', new_column_name='pathogen_scientific_name', schema=schema)
