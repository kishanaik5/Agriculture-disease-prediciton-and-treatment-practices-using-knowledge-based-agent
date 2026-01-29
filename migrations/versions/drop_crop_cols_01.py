"""drop_crop_cols

Revision ID: dropcropcols01
Revises: 123456789abc
Create Date: 2026-01-07 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dropcropcols01'
down_revision: Union[str, None] = '123456789abc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('crop_analysis_report', 'is_mixed_cropping', schema='kissan_cv')
    op.drop_column('crop_analysis_report', 'acres_of_land', schema='kissan_cv')


def downgrade() -> None:
    op.add_column('crop_analysis_report', sa.Column('is_mixed_cropping', sa.Boolean(), nullable=True), schema='kissan_cv')
    op.add_column('crop_analysis_report', sa.Column('acres_of_land', sa.String(length=50), nullable=True), schema='kissan_cv')
