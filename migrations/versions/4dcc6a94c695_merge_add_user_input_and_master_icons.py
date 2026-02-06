"""merge_add_user_input_and_master_icons

Revision ID: 4dcc6a94c695
Revises: add_master_icons_languages, 20260205_add_user_input_item
Create Date: 2026-02-05 18:17:31.256715

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4dcc6a94c695'
down_revision: Union[str, None] = ('add_master_icons_languages', '20260205_add_user_input_item')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
