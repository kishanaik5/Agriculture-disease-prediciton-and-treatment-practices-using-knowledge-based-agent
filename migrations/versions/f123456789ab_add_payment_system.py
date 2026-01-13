"""add_payment_system

Revision ID: f123456789ab
Revises: cf1a7ac1336f
Create Date: 2026-01-05 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f123456789ab'
down_revision = 'cf1a7ac1336f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create Orders Table
    op.create_table('orders',
        sa.Column('uid', sa.String(), nullable=False),
        sa.Column('cf_order_id', sa.String(), nullable=True),
        sa.Column('cf_payment_session_id', sa.String(), nullable=True),
        sa.Column('upid', sa.String(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'SUCCESS', 'FAILED', 'CANCELLED', 'EXPIRED', name='order_status_enum', schema='kissan_cv'), nullable=False),
        sa.Column('total_cost', sa.Numeric(), nullable=False),
        sa.Column('discount', sa.Numeric(), nullable=False, server_default='0'),
        sa.Column('order_type', sa.Enum('ONE_TIME', 'SUBSCRIPTION', name='order_type_enum', schema='kissan_cv'), nullable=False),
        sa.Column('context_id', sa.String(), nullable=False),
        sa.Column('context_type', sa.String(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('confirmed_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('uid'),
        schema='kissan_cv'
    )

    # 2. Create Payments Table
    op.create_table('payments',
        sa.Column('uid', sa.String(), nullable=False),
        sa.Column('order_uid', sa.String(), nullable=False),
        sa.Column('cf_payment_id', sa.String(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'SUCCESS', 'FAILED', 'USER_DROPPED', 'CANCELLED', 'REFUNDED', name='payment_status_enum', schema='kissan_cv'), nullable=False),
        sa.Column('amount', sa.Numeric(), nullable=False),
        sa.Column('payment_method', sa.Enum('UPI', 'CARD', 'NETBANKING', 'WALLET', name='payment_method_enum', schema='kissan_cv'), nullable=False),
        sa.Column('transaction_id', sa.String(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('attempted_at', sa.DateTime(), nullable=True),
        sa.Column('paid_at', sa.DateTime(), nullable=True),
        sa.Column('refunded_at', sa.DateTime(), nullable=True),
        sa.Column('failed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('uid'),
        sa.ForeignKeyConstraint(['order_uid'], ['kissan_cv.orders.uid'], ),
        sa.UniqueConstraint('transaction_id'),
        schema='kissan_cv'
    )

    # 4. Add Columns to Analysis Tables
    for table in ['crop_analysis_report', 'fruit_analysis_report', 'vegetable_analysis_report']:
        op.add_column(table, sa.Column('order_id', sa.String(length=100), nullable=True), schema='kissan_cv')
        op.add_column(table, sa.Column('payment_status', sa.String(length=20), nullable=True, server_default='PENDING'), schema='kissan_cv')
        op.create_index(op.f(f'ix_kissan_cv_{table}_order_id'), table, ['order_id'], unique=False, schema='kissan_cv')

    # 5. Drop unused columns from transaction_table
    op.drop_column('transaction_table', 'currency', schema='kissan_cv')


def downgrade() -> None:
    # Drop Columns
    for table in ['crop_analysis_report', 'fruit_analysis_report', 'vegetable_analysis_report']:
        op.drop_index(op.f(f'ix_kissan_cv_{table}_order_id'), table_name=table, schema='kissan_cv')
        op.drop_column(table, 'payment_status', schema='kissan_cv')
        op.drop_column(table, 'order_id', schema='kissan_cv')

    # Drop Tables
    op.drop_table('payments', schema='kissan_cv')
    op.drop_table('orders', schema='kissan_cv')
