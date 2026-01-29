import sqlalchemy as db
from sqlalchemy.orm import relationship

from SharedBackend.managers import BaseSchema, GenericManager

import app.utils.constants as C

# association_table = db.Table(
#     "order_coupons",
#     BaseSchema.metadata,
#     db.Column("order_uid", db.String, db.ForeignKey("orders.uid"), primary_key=True),
#     db.Column("coupon_uid", db.String, db.ForeignKey("coupons.uid"), primary_key=True),
# )


class OrderSchema(BaseSchema):
    __tablename__ = "orders"
    __table_args__ = {'schema': 'kissan_cv'}

    cf_order_id = db.Column(db.String, info={"flags": {"post:exclude", "patch:exclude"}})
    cf_payment_session_id = db.Column(db.String, info={"flags": {"post:exclude", "patch:exclude"}})
    upid = db.Column(db.String, nullable=False, info={"flags": {"post:exclude", "patch:exclude"}})
    status = db.Column(db.Enum(*C.ORDER_STATUS, name="order_status_enum"), default="PENDING", nullable=False,
                       info={"flags": {"post:exclude", "patch:exclude"}})
    total_cost = db.Column(db.Numeric, nullable=False)
    discount = db.Column(db.Numeric, nullable=False, default=0, info={"flags": {"post:exclude", "patch:exclude"}})
    order_type = db.Column(db.Enum(*C.ORDER_TYPE, name="order_type_enum"), nullable=False)
    context_id = db.Column(db.String, nullable=False)
    context_type = db.Column(db.String)
    meta = db.Column(db.JSON, default={}, nullable=False)

    confirmed_at = db.Column(db.DateTime, info={"flags": {"post:exclude", "patch:exclude"}})
    completed_at = db.Column(db.DateTime, info={"flags": {"post:exclude", "patch:exclude"}})
    cancelled_at = db.Column(db.DateTime, info={"flags": {"post:exclude", "patch:exclude"}})

    payments = relationship(
        "PaymentSchema",
        foreign_keys="PaymentSchema.order_uid",
        backref="order",
        uselist=True,
    )

    # coupons = relationship(
    #     "CouponSchema",
    #     secondary=association_table,
    #     backref="orders",
    #     uselist=True,
    # )

    @property
    def final_cost(self):
        return max(self.total_cost - self.discount, 0)


from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

class OrderManager(GenericManager[OrderSchema]):
    def __init__(self, engine):
        self.engine = engine
        try:
            super().__init__(engine)
        except Exception:
            pass # In case GenericManager init is strict or different

    async def get(self, uid: str):
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
            result = await session.execute(select(OrderSchema).where(OrderSchema.uid == uid))
            return result.scalars().first()

    async def create(self, obj_in):
        # Handle Pydantic model
        data = obj_in.model_dump() if hasattr(obj_in, 'model_dump') else obj_in
        # Create instance
        obj = OrderSchema(**data)
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
            session.add(obj)
            await session.commit()
            return obj

    async def update(self, uid: str, obj_in: dict):
        async with AsyncSession(self.engine, expire_on_commit=False) as session:
            stmt = update(OrderSchema).where(OrderSchema.uid == uid).values(**obj_in)
            await session.execute(stmt)
            await session.commit()


__all__ = [
    "OrderSchema", "OrderManager",
]
