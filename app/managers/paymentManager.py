import sqlalchemy as db

from SharedBackend.managers import BaseSchema, GenericManager

import app.utils.constants as C


class PaymentSchema(BaseSchema):
    __tablename__ = "payments"
    __table_args__ = {'schema': 'kissan_cv'}

    order_uid = db.Column(db.String, db.ForeignKey("orders.uid"), nullable=False, info={"flags": {"patch:exclude"}})
    cf_payment_id = db.Column(db.String, info={"flags": {"post:exclude", "patch:exclude"}})
    status = db.Column(db.Enum(*C.PAYMENT_STATUS, name="payment_status_enum"), default="PENDING", nullable=False,
                       info={"flags": {"post:exclude", "patch:exclude"}})
    amount = db.Column(db.Numeric, nullable=False, info={"flags": {"patch:exclude", "post:exclude"}})
    payment_method = db.Column(db.Enum(*C.PAYMENT_METHOD, name="payment_method_enum"), nullable=False)
    transaction_id = db.Column(db.String, unique=True)
    meta = db.Column(db.JSON)

    attempted_at = db.Column(db.DateTime)
    paid_at = db.Column(db.DateTime)
    refunded_at = db.Column(db.DateTime)
    failed_at = db.Column(db.DateTime)


class PaymentManager(GenericManager[PaymentSchema]):
    pass


__all__ = [
    "PaymentSchema", "PaymentManager",
]
