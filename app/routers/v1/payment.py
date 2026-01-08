from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import httpx
import hmac
import hashlib
import base64
import json
from datetime import datetime
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import init_settings
from app.database import engine, get_db
from app.models.scan import PaymentTransaction, AnalysisReport, FruitAnalysis, VegetableAnalysis

router = APIRouter(prefix="/payments", tags=["Payments"])
settings = init_settings()

class OrderCreateRequest(BaseModel):
    user_id: str = Field(..., description="User ID (UPID)")
    amount: float = 1.0
    mobile_number: str
    analysis_report_uid: str = Field(..., description="UID of the report (Context ID)")
    analysis_type: str = Field(..., description="Type of analysis (crop, fruit, vegetable)")
    extra_meta: Dict[str, Any] = Field(default_factory=dict, description="Additional meta info like crop_name")

class OrderResponse(BaseModel):
    order_id: str
    payment_link: Optional[str] = None
    payment_links: Optional[Dict[str, str]] = Field(None, description="All UPI deep links (gpay, phonepe, web, etc)")
    status: str
    transaction_id: Optional[str] = None

# Helper to map analysis_type to context_type (for external API only)
def get_context_type(analysis_type: str) -> str:
    mapping = {
        "crop": "crop_analysis_report",
        "fruit": "fruit_analysis_report",
        "vegetable": "vegetable_analysis_report"
    }
    return mapping.get(analysis_type.lower(), f"{analysis_type}_report")

@router.post("/create-order", response_model=OrderResponse)
async def create_order(payload: OrderCreateRequest):
    async with httpx.AsyncClient() as client:
        try:
            # Prepare Meta
            meta_data = {
                "phone": payload.mobile_number,
                "analysis_report_uid": payload.analysis_report_uid,
                **payload.extra_meta
            }

            # 1. Create Order in Payment Service
            order_payload = {
                "total_cost": payload.amount,
                "order_type": "ONE_TIME",
                "context_id": payload.analysis_report_uid,
                "context_type": get_context_type(payload.analysis_type),
                "meta": meta_data
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": settings.PAYMENT_API_KEY,
                "x-upid": payload.user_id
            }

            resp = await client.post(
                f"{settings.PAYMENT_BASE_URL}/api/v1/orders/",
                json=order_payload,
                headers=headers
            )
            resp.raise_for_status()
            
            order_data = resp.json()
            order_uid = order_data["uid"] # This is the internal UID from payment service
            cf_order_id = order_data.get("cf_order_id")
            
            # 2. Save to Local PaymentTransaction Table
            async for db_session in get_db():
                try:
                    # Check if transaction exists
                    stmt = select(PaymentTransaction).where(PaymentTransaction.order_id == order_uid)
                    result = await db_session.execute(stmt)
                    existing_tx = result.scalars().first()

                    if existing_tx:
                        existing_tx.amount = payload.amount
                        existing_tx.transaction_id = cf_order_id
                    else:
                        new_tx = PaymentTransaction(
                            user_id=payload.user_id,
                            order_id=order_uid,
                            payment_status=order_data.get("status", "PENDING"),
                            amount=payload.amount,
                            analysis_type=payload.analysis_type,
                            analysis_report_uid=payload.analysis_report_uid,
                            transaction_id=cf_order_id,
                        )
                        db_session.add(new_tx)
                    
                    await db_session.commit()
                finally:
                    await db_session.close()
                break

            # 3. Generate Payment Link
            payment_payload = {
                "payment": {
                    "order_uid": order_uid,
                    "payment_method": "UPI"
                },
                "method": "upi"
            }
            
            pay_resp = await client.post(
                f"{settings.PAYMENT_BASE_URL}/api/v1/payments/",
                json=payment_payload,
                headers={"x-api-key": settings.PAYMENT_API_KEY}
            )
            pay_resp.raise_for_status()
            pay_data = pay_resp.json()
            
            link_data = pay_data.get("data", {})
            raw_payload = link_data.get("payload", {})
            
            # Extract main web link for backward compatibility
            payment_link = link_data.get("data", {}).get("url")
            if not payment_link:
                payment_link = raw_payload.get("web")
            
            return OrderResponse(
                order_id=order_uid,
                payment_link=payment_link,
                payment_links=raw_payload, # Return full dict of links
                status="PENDING",
                transaction_id=cf_order_id
            )

        except httpx.HTTPStatusError as e:
            print(f"External API Error: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Payment Service Error: {e.response.text}")
        except Exception as e:
            print(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Order Creation Failed: {str(e)}")


@router.get("/verify/{order_id}")
async def verify_payment(order_id: str):
    """
    Verify payment status from external service and update local PaymentTransaction.
    """
    async with httpx.AsyncClient() as client:
        try:
            # 1. Fetch Status from External Service
            resp = await client.get(
                f"{settings.PAYMENT_BASE_URL}/api/v1/orders/{order_id}",
                headers={"x-api-key": settings.PAYMENT_API_KEY}
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "PENDING")
            
            # 2. Update Local PaymentTransaction Table
            async for db_session in get_db():
                try:
                    stmt = select(PaymentTransaction).where(PaymentTransaction.order_id == order_id)
                    result = await db_session.execute(stmt)
                    tx = result.scalars().first()

                    if tx:
                        tx.payment_status = status
                        if status == 'SUCCESS':
                            tx.payment_success_at = datetime.utcnow()
                        elif status == 'FAILED':
                            # Optionally handle any failure specific logic here
                            # Ensuring success date is not set if failed
                            tx.payment_success_at = None
                        await db_session.commit()
                        
                        await db_session.commit()
                        
                        # Sync Report Status (User Request)
                        # If payment is success, update the respective report table immediately
                        if status == 'SUCCESS' and tx.analysis_report_uid:
                            target_model = None
                            atype = tx.analysis_type.lower() if tx.analysis_type else ""
                            
                            if atype in ["crop", "crops"]:
                                target_model = AnalysisReport
                            elif atype in ["fruit", "fruits"]:
                                target_model = FruitAnalysis
                            elif atype in ["vegetable", "vegetables"]:
                                target_model = VegetableAnalysis
                                
                            if target_model:
                                # Fetch and Update
                                r_stmt = select(target_model).where(target_model.uid == tx.analysis_report_uid)
                                r_res = await db_session.execute(r_stmt)
                                report_rec = r_res.scalars().first()
                                if report_rec:
                                    report_rec.payment_status = 'SUCCESS'
                                    await db_session.commit()
                                    await db_session.refresh(report_rec)

                        
                finally:
                    await db_session.close()
                break

            return {"order_id": order_id, "status": status}
            
        except Exception as e:
             print(f"Verify failed: {e}")
             raise HTTPException(status_code=500, detail=f"Verification Failed: {str(e)}")

# Webhook remains same as before (optional for now as verify endpoint is primary requested)
