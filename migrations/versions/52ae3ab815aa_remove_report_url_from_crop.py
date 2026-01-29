"""remove_report_url_from_crop

Revision ID: 52ae3ab815aa
Revises: a43552fd5dae
Create Date: 2026-01-02 14:19:21.285462

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52ae3ab815aa'
down_revision: Union[str, None] = 'a43552fd5dae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('crop_analysis_report', 'report_url', schema='kissan_cv')


def downgrade() -> None:
    op.add_column('crop_analysis_report', sa.Column('report_url', sa.Text(), nullable=True), schema='kissan_cv')
