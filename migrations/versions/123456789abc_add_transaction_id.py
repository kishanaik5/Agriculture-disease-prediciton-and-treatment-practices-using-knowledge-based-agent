"""add_transaction_id

Revision ID: 123456789abc
Revises: f123456789ab
Create Date: 2026-01-07 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '123456789abc'
down_revision: Union[str, None] = 'f123456789ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('transaction_table', sa.Column('transaction_id', sa.String(), nullable=True), schema='kissan_cv')


def downgrade() -> None:
    op.drop_column('transaction_table', 'transaction_id', schema='kissan_cv')
