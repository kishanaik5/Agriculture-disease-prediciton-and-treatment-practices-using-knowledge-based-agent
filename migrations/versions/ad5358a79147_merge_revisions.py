"""merge revisions

Revision ID: ad5358a79147
Revises: 6d0694028d22, drop_pay_tables_2026
Create Date: 2026-01-28 10:18:11.995017

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ad5358a79147'
down_revision: Union[str, None] = ('6d0694028d22', 'drop_async_jobs_table')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
