from app.dependencies import require_auth
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, literal_column, union_all, func
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.models.scan import AnalysisReport, FruitAnalysis, VegetableAnalysis, TranslatedAnalysisReport, MasterIcon
# from models.payment import PaymentTransaction # Removed incorrect import
from app.schemas.scan import ScanResponse, AnalysisResult, CropItem
from app.services.gemini import gemini_service
from app.services.s3 import s3_service_public
from app.services.image import image_service
from app.services.knowledge import knowledge_service
import uuid
import logging
import io
import json
import os
import httpx
import random
import string
from datetime import datetime
from app.config import get_settings
from typing import Optional, Any, Dict
import asyncio  # For retry sleep logic
from app.exceptions import (
    SubscriptionLimitExceeded,
    SubscriptionInvalid,
    SubscriptionServiceUnavailable
)

from zoneinfo import ZoneInfo
from app.database import SessionLocal
from SharedBackend.managers.base import ListModel
from pydantic import BaseModel
# from services.redis_manager import task_manager # Removed Redis dependency

logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE FLAG: Bounding Box Generation
# =============================================================================
# Set to True to enable bounding box generation and upload to S3
# Set to False to disable bounding box generation (bbox_image_url will be NULL)
# 
# 📍 PRODUCTION TOGGLE: Change this line to enable/disable bbox generation
ENABLE_BOUNDING_BOX_GENERATION = True  # TESTING: Disabled for verification
# =============================================================================

class BytesUploadFile:
    def __init__(self, data: bytes, filename: str, content_type: str = "image/jpeg"):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        return self._data

router = APIRouter()



from app.models.scan import MasterIcon
from app.constants.languages import SUPPORTED_LANGUAGE_CODES, DEFAULT_LANGUAGE

@router.get("/All", tags=["Show Details"])
async def get_all_items(
    language: str = Query(DEFAULT_LANGUAGE, description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or, pa, ur, ne"), 
    name: Optional[str] = None, 
    category: Optional[str] = None, 
    category_id: Optional[str] = None, 
    db: AsyncSession = Depends(get_db)
):
    """
    Get all supported items from MasterIcon DB.
    
    Returns master icons with names in the requested language.
    If the requested language is not available for an item, falls back to English.
    
    **Supported languages:**
    - en: English (default)
    - kn: Kannada
    - hn: Hindi
    - ta: Tamil
    - te: Telugu
    - ml: Malayalam
    - mr: Marathi
    - gu: Gujarati
    - bn: Bengali
    - or: Odia
    - pa: Punjabi
    - ur: Urdu
    - ne: Nepali
    """
    # Validate language
    lang = language.lower()
    if lang not in SUPPORTED_LANGUAGE_CODES:
        lang = DEFAULT_LANGUAGE
    
    stmt = select(MasterIcon)
    if category:
        # DB stores "crop", "fruit", "vegetable"
        # Input might be plural
        cat_search = category.lower()
        if cat_search.endswith('s') and cat_search != "crops": # vegetables, fruits
            cat_search = cat_search[:-1]
        elif cat_search == "crops":
            cat_search = "crop"
            
        stmt = stmt.where(MasterIcon.category_type == cat_search)
        
    if category_id:
        stmt = stmt.where(MasterIcon.category_id == category_id)
        
    if name:
        # Search in the requested language field or English
        if lang == 'en':
            stmt = stmt.where(MasterIcon.name_en.ilike(f"%{name}%"))
        elif lang == 'kn':
            stmt = stmt.where(MasterIcon.name_kn.ilike(f"%{name}%"))
        elif lang == 'hn':
            stmt = stmt.where(MasterIcon.name_hn.ilike(f"%{name}%"))
        elif lang == 'ta':
            stmt = stmt.where(MasterIcon.name_ta.ilike(f"%{name}%"))
        elif lang == 'te':
            stmt = stmt.where(MasterIcon.name_te.ilike(f"%{name}%"))
        elif lang == 'ml':
            stmt = stmt.where(MasterIcon.name_ml.ilike(f"%{name}%"))
        elif lang == 'mr':
            stmt = stmt.where(MasterIcon.name_mr.ilike(f"%{name}%"))
        elif lang == 'gu':
            stmt = stmt.where(MasterIcon.name_gu.ilike(f"%{name}%"))
        elif lang == 'bn':
            stmt = stmt.where(MasterIcon.name_bn.ilike(f"%{name}%"))
        elif lang == 'or':
            stmt = stmt.where(MasterIcon.name_or.ilike(f"%{name}%"))
        elif lang == 'pa':
            stmt = stmt.where(MasterIcon.name_pa.ilike(f"%{name}%"))
        elif lang == 'ur':
            stmt = stmt.where(MasterIcon.name_ur.ilike(f"%{name}%"))
        elif lang == 'ne':
            stmt = stmt.where(MasterIcon.name_ne.ilike(f"%{name}%"))
        else:
            stmt = stmt.where(MasterIcon.name_en.ilike(f"%{name}%"))
        
    res = await db.execute(stmt)
    items = res.scalars().all()
    
    # Format Response
    resp_list = []
    for i in items:
        # Get name in requested language, fallback to English if not available
        display_name = i.name_en  # Default fallback
        
        if lang == 'kn' and i.name_kn:
            display_name = i.name_kn
        elif lang == 'hn' and i.name_hn:
            display_name = i.name_hn
        elif lang == 'ta' and i.name_ta:
            display_name = i.name_ta
        elif lang == 'te' and i.name_te:
            display_name = i.name_te
        elif lang == 'ml' and i.name_ml:
            display_name = i.name_ml
        elif lang == 'mr' and i.name_mr:
            display_name = i.name_mr
        elif lang == 'gu' and i.name_gu:
            display_name = i.name_gu
        elif lang == 'bn' and i.name_bn:
            display_name = i.name_bn
        elif lang == 'or' and i.name_or:
            display_name = i.name_or
        elif lang == 'pa' and i.name_pa:
            display_name = i.name_pa
        elif lang == 'ur' and i.name_ur:
            display_name = i.name_ur
        elif lang == 'ne' and i.name_ne:
            display_name = i.name_ne
            
        resp_list.append({
            "Name": display_name,
            "image": i.url,
            "category": i.category_type,
            "category_id": i.category_id
        })
        
    return resp_list

    return resp_list


# --- Pydantic Models for All Reports ---
class ReportSummary(BaseModel):
    user_id: str
    report_id: str
    category: str
    created_at: datetime
    severity: Optional[str] = None
    disease_name: Optional[str] = None
    status: Optional[str] = None
    crop_name: Optional[str] = None

@router.get("/all_reports", tags=["Show Details"], response_model=ListModel[ReportSummary])
async def get_all_reports(
    user_id: str = Depends(require_auth),
    language: str = "en",
    category: Optional[str] = None, # Optional
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get generic list of reports with summary details.
    """
    cat = None
    if category:
        cat = category.lower().strip()
        if cat.endswith('s') and cat != "crops": cat = cat[:-1]
        elif cat == "crops": cat = "crop"
    
    # We need to construct a specific query to get detailed fields
    # Fields: user_id, report_id, category, created_at, severity, disease_name, status
    
    items = []
    total_count = 0
    
    if language == 'en':
        # UNION approach for English (Original Tables)
        # We need to select columns in same order
        
        # Helper to build subquery for a table
        def build_select(model, cat_name):
            return select(
                model.user_id,
                model.uid.label("report_id"),
                literal_column(f"'{cat_name}'").label("category"),
                model.created_at,
                model.severity,
                model.disease_name,
                model.status,
                getattr(model, 'crop_name', getattr(model, 'fruit_name', getattr(model, 'vegetable_name', None))).label("crop_name")
            ).where(
                model.user_id == user_id,
                model.status.in_(['PROCESSING', 'SUCCESS'])
            )

        queries = []
        if cat:
            if cat == 'crop': queries.append(build_select(AnalysisReport, 'crop'))
            elif cat == 'fruit': queries.append(build_select(FruitAnalysis, 'fruit'))
            elif cat == 'vegetable': queries.append(build_select(VegetableAnalysis, 'vegetable'))
        else:
            queries.append(build_select(AnalysisReport, 'crop'))
            queries.append(build_select(FruitAnalysis, 'fruit'))
            queries.append(build_select(VegetableAnalysis, 'vegetable'))
            
        if not queries:
            return ListModel(items=[], count=0)
            
        from sqlalchemy import union_all, func
        
        if len(queries) > 1:
            u_stmt = union_all(*queries).subquery()
        else:
            u_stmt = queries[0].subquery()
            
        # Total Count
        # count_stmt = select(func.count()).select_from(u_stmt)
        # c_res = await db.execute(count_stmt)
        # total_count = c_res.scalar()
        
        # Paginated Items
        # NOTE: SQLAlchemy Union pagination can be tricky. 
        # Better to select fields from the subquery
        final_stmt = select(
            u_stmt.c.user_id,
            u_stmt.c.report_id,
            u_stmt.c.category,
            u_stmt.c.created_at,
            u_stmt.c.severity,
            u_stmt.c.disease_name,
            u_stmt.c.severity,
            u_stmt.c.disease_name,
            u_stmt.c.status,
            u_stmt.c.crop_name
        ).order_by(u_stmt.c.created_at.desc()).limit(limit).offset(offset)
        
        # Count query separate? Union count is expensive but standard
        # For simplicity, let's fetch count if possible or just paginated items?
        # User requested "count".
        
        # Optimized count: 
        # count_stmt = select(func.count()).select_from(u_stmt) # Invalid for Union subquery sometimes?
        # Let's try simple count first
        c_res = await db.execute(select(func.count()).select_from(u_stmt))
        total_count = c_res.scalar()

        res = await db.execute(final_stmt)
        rows = res.all()
        
        for r in rows:
            items.append(ReportSummary(
                user_id=r.user_id,
                report_id=r.report_id,
                category=r.category,
                created_at=r.created_at,
                severity=r.severity,
                disease_name=r.disease_name,
                status=r.status,
                crop_name=r.crop_name
            ))
            
    else:
        # 3. Translation
        target_lang = report.desired_language_output or "en"
        
        # Force translation update
        if target_lang != "en":
             try:
                 await translation_service.translate_and_store_report(
                     db, report.uid, report.analysis_raw, target_lang, cat
                 )
             except Exception as e:
                 logger.error(f"Translation failed: {e}")
                 # We don't fail the whole report, just log
                 
        # 4. Final Success
        report.status = "SUCCESS"
        # Commit happens in caller or we do it here?
        # Caller (generate_report_item) commits to save subscription confirmation too.
        # But let's set it here to be safe.
        # report.status = "SUCCESS" 
        # await db.commit() 
        # Translated Reports
        # Table: TranslatedAnalysisReport
        # Fields mapping:
        # report_id -> report_uid (Universal ID)
        # category -> category_type
        
        base_stmt = select(TranslatedAnalysisReport).where(
            TranslatedAnalysisReport.user_id == user_id,
            TranslatedAnalysisReport.language == language,
            TranslatedAnalysisReport.status.in_(['PROCESSING', 'SUCCESS'])
        )
        
        if cat:
            base_stmt = base_stmt.where(TranslatedAnalysisReport.category_type == cat)
            
        # Count
        from sqlalchemy import func
        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        c_res = await db.execute(count_stmt)
        total_count = c_res.scalar()
        
        # Items
        stmt = base_stmt.order_by(TranslatedAnalysisReport.created_at.desc()).limit(limit).offset(offset)
        res = await db.execute(stmt)
        rows = res.scalars().all()
        
        for r in rows:
            items.append(ReportSummary(
                user_id=r.user_id,
                report_id=r.report_uid, # Universal ID per user request
                category=r.category_type,
                created_at=r.created_at,
                severity=r.severity,
                disease_name=r.disease_name,
                status=r.status,
                crop_name=r.item_name
            ))

    return ListModel[ReportSummary](items=items, count=total_count)



@router.post("/add_icon", tags=["Admin Section"])
async def add_icon(
    category_type: str = Form(..., description="crop, fruit, vegetable"),
    name_en: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload image to S3 (public) and add to MasterIcon table.
    """
    # 1. Determine Path
    cat_folder_map = {
        "crop": "crops",
        "fruit": "fruit",
        "vegetable": "vegetables"
    }
    
    folder = cat_folder_map.get(category_type.lower())
    if not folder:
        raise HTTPException(status_code=400, detail="Invalid category_type. Use crop, fruit, or vegetable.")
        
    # 2. Upload
    try:
        ext = file.filename.split('.')[-1]
        if not ext: ext = "jpg"
        
        # Sanitize filename from name_en
        clean_name = name_en.replace(" ", "_").strip()
        filename = f"{clean_name}.{ext}"
        
        content = await file.read()
        f_wrapper = BytesUploadFile(content, filename, file.content_type)
        
        # Using existing public upload logic, but we want specific folder?
        # s3_service_public.upload_file uses "dev/original/" by default prefix logic in my head? 
        # No, checking usage: upload_file(file, prefix)
        
        url = await s3_service_public.upload_file(file=f_wrapper, prefix=f"{folder}/")
        
        # If url has prefix like dev/original, we might want to ensure it maps to the correct folder user expects.
        # But s3_service_public.upload_file logic appends the filename to prefix. 
        # Check if s3_service_public allows custom path fully.
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    # 3. Generate category_id
    # Find last id for this category
    stmt = select(MasterIcon.category_id).where(MasterIcon.category_type == category_type).order_by(MasterIcon.category_id.desc()).limit(1)
    res = await db.execute(stmt)
    last_id_str = res.scalar_one_or_none()
    
    if last_id_str:
        # format: crop_001
        try:
            prefix, num = last_id_str.split('_')
            new_num = int(num) + 1
            new_id = f"{prefix}_{new_num:03d}"
        except:
            new_id = f"{category_type}_999" # Fallback
    else:
        new_id = f"{category_type}_001"
        
    # 4. Save to DB
    new_icon = MasterIcon(
        category_id=new_id,
        category_type=category_type,
        url=url,
        name_en=name_en
    )
    db.add(new_icon)
    await db.commit()
    await db.refresh(new_icon)
    
    return {
        "message": "Icon added successfully",
        "data": {
            "category_id": new_icon.category_id,
            "name": new_icon.name_en,
            "url": new_icon.url
        }
    }




# =============================================================================
# Subscription Verification with Improved Error Handling
# =============================================================================

async def verify_consumption(
    user_id: str, 
    subscription_id: str, 
    action: str = "scan", 
    quantity: int = 1,
    max_retries: int = 2,
    timeout: float = 5.0
) -> bool:
    """
    Validates if user can consume subscription quota with proper error handling.
    
    Args:
        user_id: User's unique ID
        subscription_id: Subscription identifier
        action: Action type (default: "scan")
        quantity: Number of units to consume (default: 1)
        max_retries: Number of retry attempts for transient failures (default: 2)
        timeout: Request timeout in seconds (default: 5.0)
    
    Returns:
        True if consumption is allowed
    
    Raises:
        SubscriptionLimitExceeded: User quota exhausted (403)
        SubscriptionInvalid: Invalid or expired subscription_id (400)
        SubscriptionServiceUnavailable: Service down/timeout (503)
    """
    settings = get_settings()
    # Sanitize URL to prevent "Invalid character in header" errors
    base_url = settings.SUBSCRIPTION_BASE_URL.strip().rstrip('/')
    url = f"{base_url}/api/v1/usage/consume/verify"
    
    payload = {
        "action": action,
        "subscription_id": subscription_id,
        "quantity": quantity
    }
    
    headers = {
        "x-upid": user_id.strip() if user_id else "",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, 
                    json=payload, 
                    headers=headers, 
                    timeout=timeout
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    allowed = data.get("allowed", False)
                    
                    if not allowed:
                        # Service returned "not allowed" - quota exceeded
                        reason = data.get("reason", "Subscription limit exceeded")
                        logger.warning(f"Subscription limit exceeded for user {user_id}: {reason}")
                        raise SubscriptionLimitExceeded(reason)
                    
                    # Success - consumption allowed
                    return True
                    
                elif resp.status_code == 400:
                    # Invalid subscription_id - don't retry
                    error_detail = resp.json() if resp.content else {"detail": "Invalid subscription ID"}
                    logger.error(f"Invalid subscription ID: {subscription_id}")
                    exc = SubscriptionInvalid()
                    exc.status_code = resp.status_code
                    exc.response = error_detail
                    raise exc
                    
                elif resp.status_code == 404:
                    # Subscription not found - don't retry
                    error_detail = resp.json() if resp.content else {"detail": "Subscription not found"}
                    logger.error(f"Subscription not found: {subscription_id}")
                    exc = SubscriptionInvalid()
                    exc.status_code = resp.status_code
                    exc.response = error_detail
                    raise exc
                    
                elif resp.status_code == 480:
                    # Custom subscription service error - pass through
                    error_detail = resp.json() if resp.content else {"detail": "Subscription error"}
                    logger.error(f"Subscription service returned 480: {error_detail}")
                    exc = SubscriptionInvalid()
                    exc.status_code = resp.status_code
                    exc.response = error_detail
                    raise exc
                    
                elif resp.status_code >= 500:
                    # Server error - pass through with exact error
                    error_detail = resp.json() if resp.content else {"detail": "Subscription service error"}
                    logger.error(f"Subscription service error {resp.status_code}: {error_detail}")
                    exc = SubscriptionServiceUnavailable()
                    exc.status_code = resp.status_code
                    exc.response = error_detail
                    raise exc
                    
                else:
                    # Other errors (e.g., 401, 403 from service itself) - retry once
                    logger.warning(f"Unexpected status {resp.status_code}: {resp.text}, retry {attempt + 1}/{max_retries}")
                    last_error = SubscriptionServiceUnavailable(f"Subscription service returned unexpected response.")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                    
        except httpx.TimeoutException as e:
            logger.warning(f"Subscription verify timeout (attempt {attempt + 1}/{max_retries}): {str(e)}")
            last_error = SubscriptionServiceUnavailable("Subscription service is taking too long to respond. Please try again.")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            
        except httpx.NetworkError as e:
            logger.warning(f"Network error connecting to subscription service (attempt {attempt + 1}/{max_retries}): {str(e)}")
            last_error = SubscriptionServiceUnavailable("Cannot connect to subscription service. Please check your network.")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            continue
        
        except httpx.RequestError as e:
            # Catches invalid URL, invalid headers, etc.
            logger.error(f"HTTP request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            last_error = SubscriptionServiceUnavailable(f"Error communicating with subscription service: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
            continue
            
        except (SubscriptionLimitExceeded, SubscriptionInvalid):
            # Don't retry these - they're deterministic
            raise
    
    # All retries failed
    if last_error:
        raise last_error
    raise SubscriptionServiceUnavailable("Subscription verification failed after multiple attempts. Please try again later.")

# Confirms consumption
# Confirms consumption
async def confirm_consumption(user_id: str, subscription_id: str, action: str = "scan", quantity: int = 1) -> bool:
    settings = get_settings()
    url = f"{settings.SUBSCRIPTION_BASE_URL}/api/v1/usage/consume/confirm"
    
    payload = {
        "action": action,
        "subscription_id": subscription_id,
        "quantity": quantity
    }
    
    headers = {
        "x-upid": user_id,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                # { "status": "consumed", "new_value": 1 }
                return True
            else:
                logger.error(f"Subscription Confirm Failed: {resp.status_code} {resp.text}")
                return False
    except Exception as e:
        logger.error(f"Subscription Confirm Exception: {e}")

@router.get("/report/item", tags=["Report Generation"])
async def generate_report_item(
    report_id: str = Query(...),
    category: str = Query(...),
    language: str = Query("en"),
    user_id: str = Depends(require_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """
    Merged endpoint for Status + Generate + Details.
    1. If Status=PROCESSING -> Starts Async Generation -> Returns minimal response.
    2. If Status=GENERATING/TRANSLATION_AWAITING -> Returns minimal processing response.
    3. If Status=SUCCESS -> Returns Full Report Details.
    4. If Status=FAILED -> Returns minimal failed response.
    """
    cat = category.lower().strip()
    if cat.endswith('s') and cat != "crops": 
        cat = cat[:-1]
    elif cat == "crops": 
        cat = "crop"
    
    lang = language.lower().strip()
    
    # 1. Fetch Report
    report_model = None
    if cat == "crop": 
        report_model = AnalysisReport
    elif cat == "fruit": 
        report_model = FruitAnalysis
    elif cat == "vegetable": 
        report_model = VegetableAnalysis
    
    if not report_model: 
        raise HTTPException(status_code=400, detail="Invalid Category")
    
    stmt = select(report_model).where(report_model.uid == report_id)
    res = await db.execute(stmt)
    report = res.scalars().first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Update desired language if provided and different
    if lang and lang != report.desired_language_output:
        report.desired_language_output = lang
        await db.commit()
    
    # 2. Get current status
    current_status = report.status 
    if not current_status: 
        current_status = "PROCESSING"
    
    # 3. Logic Branching
    
    # Case A: Needs Generation (PROCESSING - Initial State)
    if current_status == "PROCESSING":
        # Update to GENERATING
        report.status = "GENERATING"
        await db.commit()
        
        # Trigger Background Worker
        async def background_wrapper(report_uid, cat, lang, uid):
            async with SessionLocal() as session:
                # Re-fetch report inside session
                model_inner = None
                if cat == "crop": 
                    model_inner = AnalysisReport
                elif cat == "fruit": 
                    model_inner = FruitAnalysis
                elif cat == "vegetable": 
                    model_inner = VegetableAnalysis
                
                stmt_inner = select(model_inner).where(model_inner.uid == report_uid)
                res_inner = await session.execute(stmt_inner)
                report_inner = res_inner.scalars().first()
                
                if report_inner:
                    try:
                        await _core_process_generation(session, report_inner, cat, lang, uid, None)
                    except Exception as e:
                        logger.error(f"Background Generation Failed: {e}")
                        report_inner.status = "FAILED"
                        await session.commit()
        
        background_tasks.add_task(background_wrapper, report.uid, cat, lang, user_id)
        
        # Return minimal response
        details = {
            "qa": "DONE",
            "generation": "PROCESSING",
            "translation": "NA" if lang == "en" else "AWAITING"
        }
        
        return {
            "user_id": user_id,
            "report_id": report.uid,
            "language": report.desired_language_output or lang,
            "details": details,
            "status": "PROCESSING"
        }

    # Case B: Currently Generating
    if current_status == "GENERATING":
        details = {
            "qa": "DONE",
            "generation": "PROCESSING",
            "translation": "NA" if lang == "en" else "AWAITING"
        }
        
        return {
            "user_id": user_id,
            "report_id": report.uid,
            "language": report.desired_language_output or lang,
            "details": details,
            "status": "PROCESSING"
        }
    
    # Case C: Translation Awaiting
    if current_status == "TRANSLATION_AWAITING":
        details = {
            "qa": "DONE",
            "generation": "DONE",
            "translation": "NA" if lang == "en" else "PROCESSING"
        }
        
        return {
            "user_id": user_id,
            "report_id": report.uid,
            "language": report.desired_language_output or lang,
            "details": details,
            "status": "PROCESSING"
        }
        
    # Case D: Failed
    if current_status == "FAILED":
        details = {
            "qa": "DONE",
            "generation": "FAILED",
            "translation": "NA"
        }
        
        return {
            "user_id": user_id,
            "report_id": report.uid,
            "language": lang,
            "details": details,
            "status": "FAILED"
        }

    # Case E: Success (English or Translation)
    if current_status == "SUCCESS":
        # Query available translations
        stmt_all_trans = select(TranslatedAnalysisReport.language).where(
            TranslatedAnalysisReport.report_uid == report.uid,
            TranslatedAnalysisReport.status == "SUCCESS"
        )
        res_all_trans = await db.execute(stmt_all_trans)
        available_translations = [row[0] for row in res_all_trans.fetchall()]
        
        # Build available_lang dict
        available_lang_dict = {"en": True}
        for lang_code in available_translations:
            available_lang_dict[lang_code] = True
        
        # If requesting English, return full English details
        if lang == 'en':
            return {
                "id": report.uid,
                "user_id": report.user_id,
                "created_at": report.created_at,
                "disease_detected": report.disease_name,
                "original_image_url": report.original_image_url,
                "bbox_image_url": report.bbox_image_url,
                "user_input_crop": getattr(report, 'crop_name', getattr(report, 'fruit_name', getattr(report, 'vegetable_name', ''))),
                "category": cat,
                "language": "en",
                "kb_treatment": report.treatment,
                "analysis_raw": report.analysis_raw,
                "available_lang": available_lang_dict,
                "status": "SUCCESS"
            }
        
        else:
            # Case F: Valid Report, Requesting Translation
            from app.services.translation_service import translation_service
            
            t_stmt = select(TranslatedAnalysisReport).where(
                TranslatedAnalysisReport.report_uid == report_id,
                TranslatedAnalysisReport.language == lang
            )
            t_res = await db.execute(t_stmt)
            t_report = t_res.scalars().first()
            
            # Get translation status
            t_status = t_report.status if t_report else None
            
            # Self-healing: If status is NULL but we have data, treat as SUCCESS
            if t_report and not t_status and t_report.disease_name:
                t_status = "SUCCESS"
            
            if not t_status or t_status == "PENDING":
                # Trigger background translation
                async def bg_translation(r_uid, u_id, l):
                    async with SessionLocal() as session:
                        try:
                            await translation_service.get_or_create_translation(session, r_uid, u_id, l)
                        except Exception as e:
                            logger.error(f"Background Translation Failed: {e}")
                
                background_tasks.add_task(bg_translation, report.uid, user_id, lang)
                
                return {
                    "user_id": user_id,
                    "report_id": report.uid,
                    "language": report.desired_language_output or lang,
                    "details": {
                        "qa": "DONE",
                        "generation": "SUCCESS",
                        "translation": "AWAITING"
                    },
                    "status": "PROCESSING"
                }

            elif t_status == "PROCESSING" or t_status == "AWAITING":
                return {
                    "user_id": user_id,
                    "report_id": report.uid,
                    "language": report.desired_language_output or lang,
                    "details": {
                        "qa": "DONE",
                        "generation": "SUCCESS",
                        "translation": "PROCESSING"
                    },
                    "status": "PROCESSING"
                }
            
            elif t_status == "FAILED":
                # Retry translation in background
                async def bg_translation(r_uid, u_id, l):
                    async with SessionLocal() as session:
                        try:
                            await translation_service.get_or_create_translation(session, r_uid, u_id, l)
                        except Exception as e:
                            logger.error(f"Background Translation Failed: {e}")
                
                background_tasks.add_task(bg_translation, report.uid, user_id, lang)
                
                return {
                    "user_id": user_id,
                    "report_id": report.uid,
                    "language": report.desired_language_output or lang,
                    "details": {
                        "qa": "DONE",
                        "generation": "SUCCESS",
                        "translation": "PROCESSING"
                    },
                    "status": "PROCESSING"
                }
                
            elif t_status == "SUCCESS":
                # Return Translated Full Response
                return {
                    "id": t_report.report_uid,
                    "user_id": t_report.user_id,
                    "created_at": t_report.created_at,
                    "disease_detected": t_report.disease_name,
                    "original_image_url": t_report.original_image_url,
                    "bbox_image_url": t_report.bbox_image_url,
                    "user_input_crop": t_report.item_name,
                    "category": cat,
                    "language": lang,
                    "kb_treatment": t_report.treatment,
                    "analysis_raw": t_report.analysis_raw,
                    "available_lang": available_lang_dict,
                    "status": "SUCCESS"
                }

    return {"status": "UNKNOWN"}


@router.post("/qa/scan", response_model=ScanResponse, tags=["QA_Scan"])
async def analyze_qa(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    category_id: str = Form(...),
    language: str = Form("en", description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or or"), # Supported: English, Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Gujarati, Bengali, Odia
    subscription_id: Optional[str] = Form(None), # Added subscription_id with fallback
    db: AsyncSession = Depends(get_db)
):
    """
    Unified Scan Flow (Subscription Based):
    1. Verify Subscription (Verify endpoint).
    2. QA Scan (Gemini Flash).
    3. If Healthy -> Return Result (Status SUCCESS).
    4. If Unhealthy -> 
       - Deep Scan (Gemini Pro/Flash + BBox).
       - Create Report (Status SUCCESS).
       - Confirm Subscription (Confirm endpoint).
       - Return Full Result.
    """
    # 0. Resolve Subscription ID
    if not subscription_id:
        # Fallback to random if not provided (per user request "keep random value as of now")
        subscription_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    # 1. Verify Subscription with improved error handling
    subscription_quota_exceeded = False
    subscription_message = None
    
    try:
        await verify_consumption(user_id, subscription_id, action="scan")
    except SubscriptionLimitExceeded as e:
        # User has exceeded their quota - return 200 with specific status
        subscription_quota_exceeded = True
        subscription_message = str(e) or "Subscription limit exceeded. Please upgrade your plan."
        logger.warning(f"Subscription quota exceeded for user {user_id}: {subscription_message}")
    except SubscriptionInvalid as e:
        # Invalid subscription_id or not found - pass through subscription service response
        status_code = getattr(e, 'status_code', 400)
        response = getattr(e, 'response', {"detail": str(e) or "Invalid subscription"})
        raise HTTPException(
            status_code=status_code,
            detail=response.get('detail') if isinstance(response, dict) else response
        )
    except SubscriptionServiceUnavailable as e:
        # Service is down or timing out - pass through subscription service response
        status_code = getattr(e, 'status_code', 503)
        response = getattr(e, 'response', {"detail": str(e) or "Subscription service unavailable"})
        raise HTTPException(
            status_code=status_code,
            detail=response.get('detail') if isinstance(response, dict) else response
        )
    
    # If quota exceeded, return early with 200 OK but specific status
    if subscription_quota_exceeded:
        return {
            "status": "quota_exceeded",
            "message": subscription_message,
            "upgrade_required": True,
            "user_id": user_id,
            "subscription_id": subscription_id
        }

    # 2. Lookup Category & Crop Name
    stmt = select(MasterIcon).where(MasterIcon.category_id == category_id)
    res = await db.execute(stmt)
    icon_entry = res.scalars().first()
    
    if not icon_entry:
        raise HTTPException(status_code=400, detail="Invalid category_id")
        
    category = icon_entry.category_type.lower() # crop, fruit, vegetable
    
    # Standardize category string for checks
    cat = category.lower()
    if cat.endswith('s') and cat != "crops": cat = cat[:-1]
    elif cat == "crops": cat = "crop"

    crop_name = icon_entry.name_en 
        
    # 3. Read Image Bytes
    image_bytes = await file.read()
    
    # 4. QA Analysis (Flash)
    qa_result = {}
    
    try:
        if cat == "crop":
            qa_result = await gemini_service.analyze_crop_qa(image_bytes, crop_name, language, file.content_type)
        elif cat == "fruit":
            qa_result = await gemini_service.analyze_fruit_qa(image_bytes, crop_name, language, file.content_type)
        elif cat == "vegetable":
            qa_result = await gemini_service.analyze_vegetable_qa(image_bytes, crop_name, language, file.content_type)
        else:
             raise HTTPException(status_code=400, detail="Invalid category")
    except HTTPException as e:
        logger.error(f"QA Scan AI Model Error: {e.detail}")
        raise HTTPException(status_code=400, detail="AI Model not found or unavailable")
    except Exception as e:
        logger.error(f"QA Scan failed: {e}")
        raise HTTPException(status_code=400, detail=f"AI Service failed: {str(e)}")
        
    # 5. Validation Check
    if not qa_result.get("is_valid_crop", True):
        detected = qa_result.get("detected_object_name", "Unknown")
        msg = f"The uploaded image appears to be '{detected}', but you selected '{crop_name}'. Please upload a valid image."
        if language == 'kn':
            msg = f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರವು '{detected}' ಎನಿಸುತ್ತದೆ, ಆದರೆ ನೀವು '{crop_name}' ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ಸರಿಯಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
        raise HTTPException(status_code=400, detail=msg)
        
    # 6. Health Check
    is_healthy = qa_result.get("is_healthy", False)
    disease_name = qa_result.get("disease_name", "No Disease")
    
    if is_healthy:
        # RETURN IMMEDIATELY - No Confirmation needed (Free check?)
        msg = "Your crop is healthy! No disease detected."
        if language == 'kn':
            msg = "ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ! ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ."
            
        return ScanResponse(
            id="0",
            user_id=user_id,
            created_at=datetime.now(),
            disease_detected=msg,
            user_input_crop=crop_name,
            language=language,
            original_image_url=None, 
            analysis_raw=qa_result,
            category=cat,
            status="SUCCESS"
        )
        
    # 7. Unhealthy -> Full Analysis Cycle
    
    # Upload S3
    filename_uuid = str(uuid.uuid4())
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

    # Create Initial DB Record
    current_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    report = None
    
    if cat == "crop":
        report = AnalysisReport(
            user_id=user_id, crop_name=crop_name, language="en", desired_language_output=language,
            original_image_url=original_url, disease_name=disease_name, 
            status="PROCESSING", created_at=current_time,
            order_id=subscription_id 
        )
    elif cat == "fruit":
        report = FruitAnalysis(
            user_id=user_id, fruit_name=crop_name, language="en", desired_language_output=language,
            original_image_url=original_url, disease_name=disease_name, 
            status="PROCESSING", created_at=current_time,
            order_id=subscription_id
        )
    elif cat == "vegetable":
        report = VegetableAnalysis(
            user_id=user_id, vegetable_name=crop_name, language="en", desired_language_output=language,
            original_image_url=original_url, disease_name=disease_name, 
            status="PROCESSING", created_at=current_time,
            order_id=subscription_id
        )

    if report:
        db.add(report)
        await db.commit()
        await db.refresh(report)

        # REVERTED TO 2-STEP FLOW:
        # Return acknowledgement immediately. User triggers /report/item manually.
        return ScanResponse(
            id=report.uid,
            user_id=user_id,
            created_at=current_time,
            disease_detected=disease_name,
            user_input_crop=crop_name,
            language=language,
            original_image_url=original_url, 
            analysis_raw=qa_result, # Partial QA result
            category=cat,
            status="PROCESSING" # Awaiting manual trigger
        )

    return ScanResponse(id="0", user_id=user_id, created_at=current_time, disease_detected="Error", user_input_crop=crop_name, language=language)

# REFACTOR: Removed AsyncJob helper. Status is now derived from data.
async def _update_job_status(db: AsyncSession, job_id: str, status: str, details: dict = None):
    pass # No-op

# Helper to confirm consumption inside background task
async def _confirm_background_consumption(user_id: str, sub_id: str):
    if sub_id:
        await confirm_consumption(user_id, sub_id)

async def _core_process_generation(
    db: AsyncSession,
    report: Any,
    cat: str,
    lang: str,
    user_id: str,
    task_id: Optional[str] = None
) -> dict:
    # Re-fetch report in current session (essential for background tasks)
    if report:
        report_model = type(report)
        # Ensure we have the uid
        if hasattr(report, 'uid'):
            r_stmt = select(report_model).where(report_model.uid == report.uid)
            r_res = await db.execute(r_stmt)
            report = r_res.scalars().first()
            if not report:
                raise Exception(f"Report {report.uid} not found in background session")

    
    def build_treatment_text(analysis):
        mgmt = analysis.get("management", {})
        parts = []
        if mgmt.get("organic_practices"):
            parts.append("Organic Practices:")
            parts.extend([f"- {p}" for p in mgmt["organic_practices"]])
        if mgmt.get("chemical_practices"):
            if parts: parts.append("")
            parts.append("Chemical Practices:")
            parts.extend([f"- {p}" for p in mgmt["chemical_practices"]])
        if mgmt.get("consumption_safety"):
            if parts: parts.append("")
            parts.append(f"Consumption Safety: {mgmt['consumption_safety']}")
        return "\n".join(parts)

    # Initialize Status
    state = {
        "qa": "DONE",

        "generating_report": "PENDING",
        "translating": "PENDING",
        "progress": 30,
        "stage": "Starting Generation"
    }
    
    # 1. Update Status to GENERATING
    report.status = "GENERATING"
    await db.commit()

    
    if task_id:
        await _update_job_status(db, task_id, "processing", state)

    try:
        # Download Image
        try:
            key = report.original_image_url.split(".com/")[-1]
            s3_obj = s3_service_public.s3_client.get_object(Bucket=s3_service_public.bucket_name, Key=key)
            image_bytes = s3_obj['Body'].read()
        except Exception as e:
            msg = f"Failed to retrieve image: {e}"
            logger.error(msg)
            report.status = "FAILED"
            await db.commit()
            if task_id: await _update_job_status(db, task_id, "failed", {"error": "Image Fetch Failed", **state})
            raise HTTPException(status_code=500, detail=msg)

        if task_id:
            state["progress"] = 30
            state["stage"] = "AI Analysis"
            await _update_job_status(db, task_id, "processing", state)

        # Full Analysis
        full_analysis = {}
        bbox_list = []
        
        try:
            if cat in ["crop", "crops"]:
                 crop_name = report.crop_name
                 full_analysis = await gemini_service.analyze_crop(image_bytes, crop_name, "en", "image/jpeg")
                 disease = full_analysis.get("disease_info", {}).get("disease_name")
                 if disease and disease != "Unknown":
                     bbox_list = await gemini_service.generate_bbox_crop(image_bytes, disease, "image/jpeg")
                     
            elif cat in ["fruit", "fruits"]:
                 crop_name = report.fruit_name
                 full_analysis = await gemini_service.analyze_fruit(image_bytes, crop_name, "en", "image/jpeg")
                 disease = full_analysis.get("diagnosis", {}).get("disease_name")
                 if disease:
                     bbox_list = await gemini_service.generate_bbox_fruit(image_bytes, disease, "image/jpeg")
                     
            elif cat in ["vegetable", "vegetables"]:
                 crop_name = report.vegetable_name
                 full_analysis = await gemini_service.analyze_vegetable(image_bytes, crop_name, "en", "image/jpeg")
                 disease = full_analysis.get("disease_info", {}).get("disease_name")
                 if disease:
                     bbox_list = await gemini_service.generate_bbox_vegetable(image_bytes, disease, "image/jpeg")
                     
        except Exception as e:
            logger.error(f"Full Analysis Failed: {e}")
            report.status = "FAILED" 
            await db.commit()
            
            state["generating_report"] = "FAILED"
            if task_id: await _update_job_status(db, task_id, "failed", {"error": f"Analysis Failed: {e}", **state})
            raise HTTPException(status_code=500, detail="Analysis Failed")

        # Process BBox Image (controlled by feature flag)
        processed_url = None
        if ENABLE_BOUNDING_BOX_GENERATION and bbox_list:
            try:
                logger.info(f"🎯 Bounding box generation ENABLED - Processing {len(bbox_list)} boxes")
                p_bytes = image_service.draw_bounding_boxes(image_bytes, bbox_list)
                new_key = f"processed_{uuid.uuid4()}.jpg"
                p_wrapper = BytesUploadFile(p_bytes, new_key, "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=p_wrapper, prefix="dev/processed/")
                logger.info(f"✅ Bounding box image uploaded: {processed_url}")
            except Exception as e:
                logger.error(f"BBox draw/upload failed: {e}")
        elif not ENABLE_BOUNDING_BOX_GENERATION:
            logger.info(f"⏭️  Bounding box generation DISABLED - Skipping bbox processing")
        else:
            logger.info(f"⏭️  No bounding boxes detected - Skipping bbox processing")

        if task_id:
            state["progress"] = 60
            state["stage"] = "Database Update"
            await _update_job_status(db, task_id, "processing", state)

        # Update DB (Base English Report)
        report.analysis_raw = full_analysis
        report.bbox_image_url = processed_url  # Will be NULL if feature is disabled
        report.desired_language_output = lang
        
        # 2. Update Status to TRANSLATION_AWAITING
        report.status = "TRANSLATION_AWAITING"
        await db.commit()

        DEFAULT_TREATMENT = "Consult a doctor"

        if cat in ["crop", "crops"]:
             d_info = full_analysis.get("disease_info", {})
             p_info = full_analysis.get("plant_info", {})
             d_name = d_info.get("disease_name", "Unknown")
             d_sci = d_info.get("scientific_name", "")
             
             kb_text = await knowledge_service.get_treatment(report.crop_name, d_name, category="crop", db=db, scientific_name=d_sci, language="en")
             final_treatment = kb_text if (kb_text and len(kb_text) > 5) else DEFAULT_TREATMENT

             report.crop_name = p_info.get("common_name", report.crop_name)
             report.disease_name = d_name
             report.scientific_name = d_info.get("pathogen_type") or d_sci
             report.severity = str(d_info.get("severity") or d_info.get("severity_stage") or "")
             report.treatment = final_treatment
             
        elif cat in ["fruit", "fruits"]:
             diag = full_analysis.get("diagnosis", {})
             market = full_analysis.get("market_quality", {})
             d_name = diag.get("disease_name", "Unknown")
             d_sci = diag.get("scientific_name", "")
             
             kb_text = await knowledge_service.get_treatment(report.fruit_name, d_name, category="fruit", db=db, scientific_name=d_sci, language="en")
             final_treatment = kb_text if (kb_text and len(kb_text.strip()) > 5) else DEFAULT_TREATMENT
                
             report.disease_name = d_name
             report.scientific_name = d_sci
             report.severity = diag.get("severity_stage")
             report.grade = market.get("grade")
             report.treatment = final_treatment
             
        elif cat in ["vegetable", "vegetables"]:
             d_info = full_analysis.get("disease_info") or {}
             market = full_analysis.get("marketability") or {}
             d_name = d_info.get("disease_name", "Unknown")
             d_sci = d_info.get("scientific_name", "")
             
             kb_text = await knowledge_service.get_treatment(report.vegetable_name, d_name, category="vegetable", db=db, scientific_name=d_sci, language="en")
             final_treatment = kb_text if (kb_text and len(kb_text.strip()) > 5) else DEFAULT_TREATMENT
             
             report.disease_name = d_name
             report.scientific_name = d_sci
             report.severity = d_info.get("severity") or d_info.get("stage")
             report.grade = market.get("export_grade")
             report.treatment = final_treatment

        report.payment_status = None
        if lang == 'en':
            report.status = 'SUCCESS' # Successful generation
            # Confirm Consumption Here
            await _confirm_background_consumption(user_id, report.order_id)
        else:
            report.status = 'TRANSLATION_AWAITING'
        await db.commit()

        await db.refresh(report)

        # Base Response
        final_response_args = {
            "id": report.uid,
            "user_id": user_id,
            "created_at": report.created_at,
            "disease_detected": report.disease_name,
            "user_input_crop": report.crop_name if cat == 'crop' else (report.vegetable_name if cat == 'vegetable' else report.fruit_name),
            "language": "en",
            "kb_treatment": report.treatment,
            "analysis_raw": full_analysis,
            "original_image_url": report.original_image_url,
            "bbox_image_url": processed_url,
            "available_lang": {"en": True, lang: True} if lang != 'en' else {"en": True},
            "category": cat.rstrip('s') if cat not in ['crop', 'crops'] else 'crop' 
        }

        if task_id:
            state["generating_report"] = "DONE"
            state["progress"] = 70
            state["stage"] = "Report Saved"
            await _update_job_status(db, task_id, "processing", state)

        if lang != 'en':
            if task_id:
                state["translating"] = "PROCESSING"
                state["progress"] = 80
                state["stage"] = f"Translating to {lang}"
                await _update_job_status(db, task_id, "processing", state)
            
            from app.services.translation_service import translation_service
            try:
                translated_report = await translation_service.get_or_create_translation(db, report.uid, user_id, lang)
                
                final_response_args.update({
                    "id": translated_report.report_uid, 
                    "disease_detected": translated_report.disease_name,
                    "user_input_crop": translated_report.item_name,
                    "language": translated_report.language,
                    "kb_treatment": translated_report.treatment,
                    "analysis_raw": translated_report.analysis_raw,
                    "bbox_image_url": translated_report.bbox_image_url 
                })
                
                # Mark Parent Report as SUCCESS
                report.status = 'SUCCESS'
                await db.commit()
                # Confirm Consumption Here (if translation was the blocking step)
                await _confirm_background_consumption(user_id, report.order_id)


                if task_id:
                    state["translating"] = "DONE"
                    state["progress"] = 100
                    state["stage"] = "Completed"
                    await _update_job_status(db, task_id, "completed", state)
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                state["translating"] = "FAILED"
                if task_id: await _update_job_status(db, task_id, "failed", {"error": f"Translation Failed: {e}", **state})
                print(f"CRITICAL: Translation Service Failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
        else:
            if task_id:
                state["translating"] = "SKIPPED"
                state["progress"] = 100
                state["stage"] = "Completed"
                await _update_job_status(db, task_id, "completed", state)

        return final_response_args
    except HTTPException:
        raise # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Core process generation failed for task {task_id}: {e}")
        if task_id:
            await _update_job_status(db, task_id, "failed", {"error": str(e), **state})
        raise HTTPException(status_code=500, detail="Internal server error during report generation.")


async def background_generation_worker(
    task_id: str,
    report: Any,
    cat: str,
    lang: str,
    user_id: str
):
    """Background worker for async generation"""
    # Create fresh session for the background task
    async for db_session in get_db():
        try:
            await _core_process_generation(db_session, report, cat, lang, user_id, task_id)
        except Exception as e:
            logger.error(f"Background Generation Failed for task {task_id}: {e}")
            await _update_job_status(db_session, task_id, "failed", {"error": str(e)})
        finally:
            await db_session.close()


@router.post("/report/generate", response_model=ScanResponse)
async def generate_full_report(
    user_id: str = Depends(require_auth),
    report_id: str = Form(...),
    category: str = Form(...), 
    language: str = Form(..., description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or or"), 
    db: AsyncSession = Depends(get_db)
):
    """
    Stage 2: Full Analysis (Sync).
    Calls core logic.
    """
    cat = category.lower().strip()
    lang = language.lower().strip()
    
    logger.info(f"🚀 Generate Report Request (Sync) | ID: {report_id} | Cat: {cat} | Lang: {lang}")

    # 1. Verify Payment
    stmt = select(PaymentTransaction).where(
        PaymentTransaction.analysis_report_uid == report_id,
        PaymentTransaction.payment_status == 'SUCCESS'
    )
    result = await db.execute(stmt)
    tx = result.scalars().first()
    
    if not tx:
        raise HTTPException(status_code=402, detail="Payment Required or Not Success")

    # 2. Get Partial Report
    report_model = None
    if cat in ["crop", "crops"]: report_model = AnalysisReport
    elif cat in ["fruit", "fruits"]: report_model = FruitAnalysis
    elif cat in ["vegetable", "vegetables"]: report_model = VegetableAnalysis
    
    if not report_model: raise HTTPException(status_code=400, detail="Invalid Category")
    
    r_stmt = select(report_model).where(report_model.uid == report_id)
    r_res = await db.execute(r_stmt)
    report = r_res.scalars().first()
    
    if not report or not report.original_image_url:
        raise HTTPException(status_code=404, detail="Report or Image not found")

    # Call Core
    result = await _core_process_generation(db, report, cat, lang, user_id, None)
    return ScanResponse(**result)
    
@router.get("/report/item", tags=["Generate Report"])
async def get_report_item(
    report_id: str,
    category: str,
    user_id: str = Depends(require_auth),
    language: str = "en",
    background_tasks: BackgroundTasks = BackgroundTasks(), # Default
    db: AsyncSession = Depends(get_db)
):
    """
    Merged endpoint for Status + Generate + Details.
    1. Checks Payment.
    2. If Status=PENDING -> Starts Async Generation -> Returns PROCESSING.
    3. If Status=PROCESSING -> Returns PROCESSING.
    4. If Status=SUCCESS -> Returns Report Details.
    5. If Status=FAILED -> Returns Failed Message.
    """
    cat = category.lower().strip()
    lang = language.lower().strip()
    
    # 1. Fetch Report & Payment Check
    report_model = None
    if cat in ["crop", "crops"]: report_model = AnalysisReport
    elif cat in ["fruit", "fruits"]: report_model = FruitAnalysis
    elif cat in ["vegetable", "vegetables"]: report_model = VegetableAnalysis
    
    if not report_model: raise HTTPException(status_code=400, detail="Invalid Category")
    
    # Check payment via transaction or report? Report usually has payment_status logic synced.
    # Let's check report directly first.
    stmt = select(report_model).where(report_model.uid == report_id)
    res = await db.execute(stmt)
    report = res.scalars().first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    if report.payment_status != 'SUCCESS':
         # Return partial response indicating pending payment??
         # User requirement: "if the payment is done... update status... processing"
         # If payment NOT done, just return current state
         return {
             "user_id": user_id,
             "report_id": report_id,
             "language": report.desired_language_output or "en",
             "details": {
                 "qa": "DONE",
                 "payment": "PENDING",
                 "generation": "PENDING",
                 "translation": "NA"
             },
             "status": "PENDING_PAYMENT"
         }

    # Payment IS Success
    current_status = report.status 
    if not current_status: current_status = "PENDING" # Fallback
    
    # 2. Logic Branching
    
    # Case A: Needs Generation (PENDING)
    # Case A: Needs Generation (PENDING)
    if current_status == "PENDING":
        # Update to PROCESSING & Desired Language
        report.status = "PROCESSING"
        # Only update desired language if provided and different? 
        # User said "fetch from desired_language column", implies it might be set.
        # But if user requests a specific language now, we should probably respect it for the future?
        # Safe to update it.
        if lang: report.desired_language_output = lang
        
        await db.commit()
        
        # Trigger Background Worker
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            background_generation_worker,
            task_id=task_id,
            report=report,
            cat=cat,
            lang="en", # Worker always generates English first
            user_id=user_id
        )
        
        return {
             "user_id": user_id,
             "report_id": report_id,
             "language": report.desired_language_output or "en",
             "details": {
                 "qa": "DONE",
                 "payment": "SUCCESS",
                 "generation": "PROCESSING",
                 "translation": "AWAITING"
             },
             "status": "PROCESSING"
        }

    # Case B: Currently Processing
    if current_status == "PROCESSING":
        return {
             "user_id": user_id,
             "report_id": report_id,
             "language": report.desired_language_output or "en",
             "details": {
                 "qa": "DONE",
                 "payment": "SUCCESS",
                 "generation": "PROCESSING",
                 "translation": "AWAITING"
             },
             "status": "PROCESSING"
        }
        
    # Case C: Failed
    if current_status == "FAILED":
         return {
             "user_id": user_id,
             "report_id": report_id,
             "language": lang,
             "details": {
                 "qa": "DONE",
                 "payment": "SUCCESS",
                 "generation": "FAILED",
                 "translation": "NA"
             },
             "status": "FAILED"
        }

    # Case D: Success (English)
    if current_status == "SUCCESS":
        # If requesting English, return full details
        if lang == 'en':
             # Reuse logic to format response? Or Build manual?
             # Build manually as per user sample
             return {
                  "id": report.uid,
                  "user_id": report.user_id,
                  "created_at": report.created_at,
                  "disease_detected": report.disease_name,
                  "original_image_url": report.original_image_url,
                  "bbox_image_url": report.bbox_image_url,
                  "user_input_crop": report.crop_name if cat == 'crop' else (report.vegetable_name if cat == 'vegetable' else report.fruit_name),
                  "category": cat.rstrip('s') if cat != 'crop' else 'crop',
                  "language": "en",
                  "kb_treatment": report.treatment,
                  "analysis_raw": report.analysis_raw,
                  "available_lang": {"en": True, "kn": True}, # dynamic?
                  "status": "SUCCESS"
             }
        
        else:
            # Case E: Valid Report, Requesting Translation (e.g. 'kn')
            # Check Translation Table
            
            t_stmt = select(TranslatedAnalysisReport).where(
                TranslatedAnalysisReport.report_uid == report_id,
                TranslatedAnalysisReport.language == lang
            )
            t_res = await db.execute(t_stmt)
            t_report = t_res.scalars().first()
            
            # If no record or status is NULL/PENDING -> Check content or Trigger Translation
            t_status = t_report.status if t_report else "PENDING"
            if not t_status: t_status = "PENDING"
            
            # Self-healing: If status is PENDING/NULL but we have data, treat as SUCCESS
            if t_report and t_status == "PENDING" and t_report.disease_name:
                t_status = "SUCCESS"
            
            if t_status == "PENDING":
                # Create/Get & Update to PROCESSING
                from app.services.translation_service import translation_service
                
                # We need to make sure we trigger it.
                # If record doesn't exist, Create it as PROCESSING
                if not t_report:
                    # Logic is inside translation service usually, but we want async trigger?
                    # Trigger worker? We don't have a dedicated worker for just translation exposed here easily.
                    # We can use the core logic or call a new helper.
                    
                    # For now, let's treat it as AWAITING/PROCESSING trigger
                    # To keep it simple, I will run translation logic in background task wrapper
                    
                    # First, create placeholder if not exists to lock it?
                    # Or just return AWAITING and background task creates it.
                     pass 
                
                # Let's trigger a background task for translation
                async def bg_translation(r_uid, u_id, l, c):
                     async for s in get_db():
                         try:
                             from app.services.translation_service import translation_service
                             # This helper does the heavy lifting: checks or creates
                             # We might need to update status manually if service doesn't
                             # translation_service.get_or_create_translation usually awaits.
                             
                             # Let's assume we call it sync here for now or wrap it properly?
                             # Logic:
                             # 1. Create entry with PROCESSING
                             # 2. Call API
                             # 3. Update to SUCCESS
                             pass 
                         finally:
                             await s.close()

                # Trigger via imported service? 
                # Re-using the background_generation_worker flow is complex as it regenerates english.
                
                # Simplified: Just trigger logic in background
                background_tasks.add_task(
                    _run_translation_bg,
                    report_uid=report_id,
                    user_id=user_id,
                    lang=lang,
                    category=cat
                )
                
                return {
                    "user_id": user_id,
                    "report_id": report_id,
                    "language": report.desired_language_output, # Original logic? User said "fetch from desired_language column... not translated"
                    "details": {
                        "qa": "DONE",
                        "payment": "SUCCESS",
                        "generation": "SUCCESS",
                        "translation": "AWAITING" 
                    },
                    "status": "PROCESSING"
                }

            elif t_status == "PROCESSING":
                 return {
                    "user_id": user_id,
                    "report_id": report_id,
                    "language": report.desired_language_output,
                    "details": {
                        "qa": "DONE",
                        "payment": "SUCCESS",
                        "generation": "SUCCESS",
                        "translation": "PROCESSING" 
                    },
                    "status": "PROCESSING"
                }
            
            elif t_status == "FAILED":
                 return {
                    "user_id": user_id,
                    "report_id": report_id,
                    "language": lang,
                    "details": {
                        "qa": "DONE",
                        "payment": "SUCCESS",
                        "generation": "SUCCESS",
                        "translation": "FAILED" 
                    },
                    "status": "FAILED"
                }
                
            elif t_status == "SUCCESS":
                # Return Translated Body
                return {
                  "id": t_report.report_uid,
                  "user_id": t_report.user_id,
                  "created_at": t_report.created_at,
                  "disease_detected": t_report.disease_name,
                  "original_image_url": t_report.original_image_url,
                  "bbox_image_url": t_report.bbox_image_url,
                  "user_input_crop": t_report.item_name,
                  "category": cat.rstrip('s') if cat != 'crop' else 'crop',
                  "language": lang,
                  "kb_treatment": t_report.treatment,
                  "analysis_raw": t_report.analysis_raw,
                  "available_lang": {"en": True, "kn": True}, 
                  "status": "SUCCESS"
                }

    return {"status": "UNKNOWN"}

async def _run_translation_bg(report_uid: str, user_id: str, lang: str, category: str):
    """Helper to run translation in background"""
    from app.services.translation_service import translation_service
    async for db in get_db():
        try:
             # Check/Set Processing?
             # translation_service.get_or_create_translation handles logic. 
             # We need to ensure it updates STATUS column in translated table.
             # We might need to modify translation_service too? 
             # For now assume it does or we wrap it.
             await translation_service.get_or_create_translation(db, report_uid, user_id, lang)
        except Exception as e:
            logger.error(f"BG Translation failed: {e}")
            # Try to update status to FAILED in translated table?
        finally:
            await db.close()






async def analyze_crop(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # New Inputs
    user_id: str = Depends(require_auth),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or or"),
    db: AsyncSession = Depends(get_db)
):
    """
    Workflow:
    1. Upload Original -> S3 Private.
    2. Analyze (Gemini 2.5 Flash).
    3. Healthy Check: If "No visible disease", return "Healthy" (Skip DB).
    4. If Disease:
       - Generate BBox (Gemini 3.0 Pro Preview).
       - Draw Boxes -> Local Save + S3 Public.
       - Save to DB (AnalysisReport) -> Get Serial ID.
    5. Return Response.
    """
    
    # 1. Read Image
    image_bytes = await file.read()
    filename_uuid = str(uuid.uuid4())
    filename = f"{filename_uuid}_{file.filename}"
    
    # 2. Analyze Image
    # Standard Crop Scan -> Plant/Leaf Analysis
    logger.info(f"Crop Analysis Request for: {crop_name}")
    try:
        # Default to standard plant analysis for "crops"
        analysis_dict = await gemini_service.analyze_crop(image_bytes, crop_name, language=language, mime_type=file.content_type)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI Service failed to analyze image: {str(e)}")

    # VALIDATION CHECK
    if not analysis_dict.get("is_valid_crop", True):
         detected = analysis_dict.get("detected_object_name", "Unknown")
         msg = f"The uploaded image appears to be '{detected}', but you selected '{crop_name}'. Please upload a valid image."
         if language == 'kn':
             msg = f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರವು '{detected}' ಎನಿಸುತ್ತದೆ, ಆದರೆ ನೀವು '{crop_name}' ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ಸರಿಯಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
         raise HTTPException(status_code=400, detail=msg)

    # 3. VALIDATION CHECK
    plant_info = analysis_dict.get("plant_info", {})
    sci_name = plant_info.get("scientific_name", "")
    comm_name = plant_info.get("common_name", "")

    # Check for Mismatch or Invalid Image
    if "MISMATCH" in sci_name or "INVALID_IMAGE" in comm_name or "INVALID_IMAGE" in sci_name:
        # Localized Error Message
        error_msg = "Please choose the correct crop and upload the image correctly."
        if language == 'kn':
            error_msg = "ದಯವಿಟ್ಟು ಸರಿಯಾದ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ಮತ್ತು ಚಿತ್ರವನ್ನು ಸರಿಯಾಗಿ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
            
        raise HTTPException(status_code=400, detail=error_msg)

    # 4. Upload Original (Public Bucket)
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed but analysis valid. Proceeding. Error: {e}")
        original_url = None

    # 5. Process Logic
    disease_info = analysis_dict.get("disease_info", {})
    disease_name = disease_info.get("common_name", "Unknown")
    
    # KB Lookup
    kb_text = ""
    dis_sci_name = disease_info.get("scientific_name", "")
    if disease_name and disease_name.lower() != "unknown":
        kb_text = await knowledge_service.get_treatment(crop_name, disease_name, category="crop", db=db, scientific_name=dis_sci_name, language=language)
    
    # Healthy Heuristic
    lower_name = str(disease_name).lower().strip()
    is_healthy = (
        lower_name in ["none", "healthy", "n/a", "", "no visible disease", "none detected", "ಆರೋಗ್ಯಕರ", "ಯಾವುದೇ ರೋಗವಿಲ್ಲ"] 
        or "no disease" in lower_name
        or "healthy" in lower_name
        or "none detected" in lower_name
    )
    
    processed_url = None
    report_url = None 
    
    # Bounding Box Logic
    if not is_healthy and disease_name:
        try:
            logger.info(f"🎯 Generating bounding boxes for disease: {disease_name}")
            boxes = await gemini_service.generate_bbox_crop(image_bytes, disease_name, mime_type=file.content_type)
            
            if boxes:
                logger.info(f"✅ Generated {len(boxes)} bounding boxes")
                processed_image_bytes = image_service.draw_bounding_boxes(image_bytes, boxes)
                
                # Upload processed
                processed_wrapper = BytesUploadFile(processed_image_bytes, f"processed_{filename}", "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=processed_wrapper, prefix="dev/processed/")
            else:
                logger.warning(f"⚠️ No bounding boxes generated for {disease_name}. Model returned empty list.")
        except Exception as e:
             logger.error(f"❌ Bounding Box stage failed: {e}", exc_info=True)

    # 6. Save to DB & Return
    return await _save_crop_and_respond(
        db, user_id, crop_name, language,
        original_url, processed_url, disease_name, kb_text, analysis_dict, is_healthy, plant_info, disease_info
    )

async def analyze_vegetable(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    db: AsyncSession = Depends(get_db)
):
    """
    Dedicated endpoint for Vegetable Analysis.
    Use this for vegetables like Tomato, Chillies, etc. when inspecting produce quality/disease.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    filename_uuid = str(uuid.uuid4())
    filename = f"{filename_uuid}_{file.filename}"

    # 1. Analyze Vegetable
    try:
        analysis_dict = await gemini_service.analyze_vegetable(image_bytes, crop_name, language=language, mime_type=file.content_type)
    except Exception as e:
       logger.error(f"Veg Analysis failed: {e}")
       raise HTTPException(status_code=500, detail=f"AI Service failed: {str(e)}")

    # Validation Check
    if not analysis_dict.get("is_valid_crop", True):
         detected = analysis_dict.get("detected_object_name", "Unknown")
         msg = f"The uploaded image appears to be '{detected}', but you selected '{crop_name}'. Please upload a valid image."
         if language == 'kn':
             msg = f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರವು '{detected}' ಎನಿಸುತ್ತದೆ, ಆದರೆ ನೀವು '{crop_name}' ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ಸರಿಯಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
         raise HTTPException(status_code=400, detail=msg)

    # 2. Upload Original
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed but analysis valid. Proceeding. Error: {e}")
        original_url = None

    # 3. Process Logic
    # Schema: disease_info
    disease_info = analysis_dict.get("disease_info", {})
    disease_name = disease_info.get("common_name", "Unknown")
    
    # Healthy?
    lower_name = str(disease_name).lower().strip()
    is_healthy = (
        lower_name in ["none", "healthy", "n/a", "", "no visible disease", "none detected", "ಆರೋಗ್ಯಕರ", "ಯಾವುದೇ ರೋಗವಿಲ್ಲ"] 
        or "no disease" in lower_name
        or "healthy" in lower_name
        or "none detected" in lower_name
    )

    # KB Lookup (Try to match disease name if possible)
    kb_text = ""
    dis_sci_name = disease_info.get("scientific_name", "")
    
    if not is_healthy and disease_name and disease_name.lower() != "unknown":
         kb_text = await knowledge_service.get_treatment(crop_name, disease_name, category="vegetable", db=db, scientific_name=dis_sci_name, language=language)

    processed_url = None

    # BBox
    if not is_healthy and disease_name:
        try:
             boxes = await gemini_service.generate_bbox_vegetable(image_bytes, disease_name, mime_type=file.content_type)
             if boxes:
                logger.info(f"✅ Generated {len(boxes)} bounding boxes for vegetable")
                processed_image_bytes = image_service.draw_bounding_boxes(image_bytes, boxes)
                processed_wrapper = BytesUploadFile(processed_image_bytes, f"processed_{filename}", "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=processed_wrapper, prefix="dev/processed/")
             else:
                logger.warning(f"⚠️ No bounding boxes generated for {disease_name}. Model returned empty list.")
        except Exception as e:
            logger.error(f"❌ Veg BBox failed: {e}", exc_info=True)

    plant_info = analysis_dict.get("plant_info", {})

    return await _save_vegetable_and_respond(
        db=db,
        user_id=user_id,
        crop_name=crop_name,
        original_url=original_url,
        processed_url=processed_url,
        analysis_raw=analysis_dict,
        is_healthy=is_healthy,
        language=language,
        kb_text=kb_text
    )



async def analyze_fruit(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    db: AsyncSession = Depends(get_db)
):
    """
    Dedicated endpoint for Fruit Analysis.
    Use this for fruits like Apple, Pomegranate, etc.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    filename_uuid = str(uuid.uuid4())
    filename = f"{filename_uuid}_{file.filename}"

    # 1. Analyze Fruit
    try:
        analysis_dict = await gemini_service.analyze_fruit(image_bytes, crop_name, language=language, mime_type=file.content_type)
    except Exception as e:
       logger.error(f"Fruit Analysis failed: {e}")
       raise HTTPException(status_code=500, detail=f"AI Service failed: {str(e)}")

    # Validation Check
    if not analysis_dict.get("is_valid_crop", True):
         detected = analysis_dict.get("detected_object_name", "Unknown")
         msg = f"The uploaded image appears to be '{detected}', but you selected '{crop_name}'. Please upload a valid image."
         if language == 'kn':
             msg = f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರವು '{detected}' ಎನಿಸುತ್ತದೆ, ಆದರೆ ನೀವು '{crop_name}' ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ಸರಿಯಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
         raise HTTPException(status_code=400, detail=msg)

    # 2. Upload Original
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed but analysis valid. Proceeding. Error: {e}")
        original_url = None

    # 3. Process Logic
    # Schema: diagnosis (different from veg/crop)
    diagnosis = analysis_dict.get("diagnosis", {}) # Use diagnosis for fruit
    disease_name = diagnosis.get("disease_name", "Unknown") # Changed from common_name
    
    # Healthy?
    lower_name = str(disease_name).lower().strip()
    is_healthy = (
        lower_name in ["none", "healthy", "n/a", "", "no visible disease", "none detected", "ಆರೋಗ್ಯಕರ", "ಯಾವುದೇ ರೋಗವಿಲ್ಲ"] 
        or "no disease" in lower_name
        or "healthy" in lower_name
        or "none detected" in lower_name
    )

    # KB Lookup
    kb_text = ""
    sci_name = diagnosis.get("pathogen_scientific_name", "")
    
    if not is_healthy and disease_name and disease_name.lower() != "unknown":
         kb_text = await knowledge_service.get_treatment(crop_name, disease_name, db=db, scientific_name=sci_name, language=language)
    
    processed_url = None

    # BBox
    if not is_healthy and disease_name:
        try:
             # Use general bbox or specific produce bbox tool
             boxes = await gemini_service.generate_bbox_fruit(image_bytes, disease_name, mime_type=file.content_type)
             if boxes:
                logger.info(f"✅ Generated {len(boxes)} bounding boxes for fruit")
                processed_image_bytes = image_service.draw_bounding_boxes(image_bytes, boxes)
                processed_wrapper = BytesUploadFile(processed_image_bytes, f"processed_{filename}", "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=processed_wrapper, prefix="dev/processed/")
             else:
                logger.warning(f"⚠️ No bounding boxes generated for {disease_name}. Model returned empty list.")
        except Exception as e:
            logger.error(f"❌ Fruit BBox failed: {e}", exc_info=True)

    return await _save_fruit_and_respond(
        db=db,
        user_id=user_id,
        crop_name=crop_name,
        original_url=original_url,
        processed_url=processed_url,
        analysis_raw=analysis_dict,
        is_healthy=is_healthy,
        language=language,
        kb_text=kb_text
    )

@router.post("/{category}_scan", response_model=ScanResponse, tags=["Generate Report"])
async def analyze_dynamic(
    category: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    db: AsyncSession = Depends(get_db)
):
    """
    Universal endpoint for crop/fruit/vegetable scans via path parameter.
    
    Category values:
    - 'crop' -> Invokes crop scan logic
    - 'fruit' -> Invokes fruit scan logic
    - 'vegetable' -> Invokes vegetable scan logic
    """
    cat = category.lower()
    
    # Dispatch to specific handlers
    if cat in ["crop", "crops"]:
        return await analyze_crop(
            background_tasks=background_tasks,
            file=file,
            user_id=user_id,
            crop_name=crop_name,
            language=language,
            db=db
        )
        
    elif cat in ["fruit", "fruits"]:
        # analyze_fruit does not take background_tasks
        return await analyze_fruit(
            file=file,
            user_id=user_id,
            crop_name=crop_name,
            language=language,
            db=db
        )
        
    elif cat in ["vegetable", "vegetables"]:
        # analyze_vegetable does not take background_tasks
        return await analyze_vegetable(
            file=file,
            user_id=user_id,
            crop_name=crop_name,
            language=language,
            db=db
        )
    
    # Fallback for invalid category
    raise HTTPException(status_code=400, detail=f"Invalid category '{category}'. Must be 'crop', 'fruit', or 'vegetable'.")

async def _save_crop_and_respond(
    db: AsyncSession, user_id: str, crop_name: str, language: str,
    original_url: Optional[str], processed_url: Optional[str], disease_name: str, kb_text: str, analysis_raw: dict, is_healthy: bool,
    plant_info: dict, disease_info: dict
) -> ScanResponse:
    current_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    scan_id = "0"
    
    if not is_healthy:
        db_report = AnalysisReport(
            user_id=user_id,
            crop_name=crop_name, # Changed from user_input_crop, maps to crop_name
            language=language,
            # detected_crop=plant_info.get("common_name"), # Removed as detected_crop -> crop_name which we just set
            disease_name=disease_name, # Renamed from detected_disease
            scientific_name=disease_info.get("pathogen_type"),
            severity=str(disease_info.get("severity") or disease_info.get("severity_stage")),
            treatment=kb_text,
            analysis_raw=analysis_raw,
            created_at=current_time,
            original_image_url=original_url,
            bbox_image_url=processed_url
        )
        db.add(db_report)
        await db.commit()
        await db.refresh(db_report)
        scan_id = db_report.uid

    return ScanResponse(
        id=scan_id,
        user_id=user_id,
        created_at=current_time,
        disease_detected="No disease detected" if is_healthy else disease_name,
        user_input_crop=crop_name,
        language=language,
        kb_treatment=kb_text,
        analysis_raw=analysis_raw,
        original_image_url=original_url,
        bbox_image_url=processed_url
    )

async def _save_fruit_and_respond(
    db: AsyncSession, user_id: str, crop_name: str, original_url: Optional[str], processed_url: Optional[str], 
    analysis_raw: dict, is_healthy: bool, language: str, kb_text: str = ""
) -> ScanResponse:
    current_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    
    # Check JSON structure
    fruit_info = analysis_raw.get("fruit_info", {})
    diagnosis = analysis_raw.get("diagnosis", {})
    market = analysis_raw.get("market_quality", {})
    shelf_life = analysis_raw.get("shelf_life", {})
    management = analysis_raw.get("management", {})
    
    disease_name = diagnosis.get("disease_name") or "Unknown"
    
    scan_id = "0"
    
    # Construct combined treatment text
    treatment_list = []
    if management.get("organic_practices"):
        treatment_list.append("Organic Practices:")
        treatment_list.extend([f"- {p}" for p in management["organic_practices"]])
    if management.get("chemical_practices"):
        if treatment_list: treatment_list.append("")
        treatment_list.append("Chemical Practices:")
        treatment_list.extend([f"- {p}" for p in management["chemical_practices"]])
    if management.get("consumption_safety"):
        if treatment_list: treatment_list.append("")
        treatment_list.append(f"Consumption Safety: {management['consumption_safety']}")
    
    treatment_text = "\n".join(treatment_list)
    
    # Prioritize KB text if available and valid
    final_treatment = treatment_text
    if kb_text and len(kb_text.strip()) > 10 and "no treatment" not in kb_text.lower():
        final_treatment = kb_text

    if not is_healthy:
        db_report = FruitAnalysis(
            user_id=user_id,
            fruit_name=fruit_info.get("name"),
            language=language,
            disease_name=disease_name,
            scientific_name=diagnosis.get("pathogen_scientific_name"),
            severity=diagnosis.get("severity_stage"), # Using stage as severity
            grade=market.get("grade"),
            treatment=final_treatment,
            analysis_raw=analysis_raw,
            created_at=current_time,
            original_image_url=original_url,
            bbox_image_url=processed_url
        )
        db.add(db_report)
        await db.commit()
        await db.refresh(db_report)
        scan_id = db_report.uid

    return ScanResponse(
        id=scan_id,
        user_id=user_id,
        created_at=current_time,
        disease_detected="No disease detected" if is_healthy else disease_name,
        user_input_crop=crop_name,
        language=language,
        kb_treatment=final_treatment,
        analysis_raw=analysis_raw,
        original_image_url=original_url,
        bbox_image_url=processed_url
    )

async def _save_vegetable_and_respond(
    db: AsyncSession, user_id: str, crop_name: str, original_url: Optional[str], processed_url: Optional[str],
    analysis_raw: dict, is_healthy: bool, language: str, kb_text: str = ""
) -> ScanResponse:
    current_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    
    plant_info = analysis_raw.get("plant_info", {})
    disease_info = analysis_raw.get("disease_info", {})
    market = analysis_raw.get("marketability", {})
    shelf_life = analysis_raw.get("shelf_life_prediction", {})
    management = analysis_raw.get("management", {})

    disease_name = disease_info.get("common_name", "Unknown")
    
    scan_id = "0"
    
    # Construct combined treatment text for Veg
    treatment_list = []
    if management.get("organic_practices"):
        treatment_list.append("Organic Practices:")
        treatment_list.extend([f"- {p}" for p in management["organic_practices"]])
    if management.get("chemical_practices"):
        if treatment_list: treatment_list.append("")
        treatment_list.append("Chemical Practices:")
        treatment_list.extend([f"- {p}" for p in management["chemical_practices"]])
    if management.get("consumption_safety"):
        if treatment_list: treatment_list.append("")
        treatment_list.append(f"Consumption Safety: {management['consumption_safety']}")
        
    treatment_text = "\n".join(treatment_list)

    # Prioritize KB text for Veg
    final_treatment = treatment_text
    if kb_text and len(kb_text.strip()) > 10 and "no treatment" not in kb_text.lower():
        final_treatment = kb_text

    if not is_healthy:
        db_report = VegetableAnalysis(
            user_id=user_id,
            vegetable_name=plant_info.get("common_name"),
            language=language,
            disease_name=disease_name,
            scientific_name=disease_info.get("scientific_name"),
            severity=disease_info.get("severity") or disease_info.get("stage"),
            grade=market.get("export_grade"),
            treatment=final_treatment,
            analysis_raw=analysis_raw,
            created_at=current_time,
            original_image_url=original_url,
            bbox_image_url=processed_url
        )
        db.add(db_report)
        await db.commit()
        await db.refresh(db_report)
        scan_id = db_report.uid 

    return ScanResponse(
        id=scan_id,
        user_id=user_id,
        created_at=current_time,
        disease_detected="No disease detected" if is_healthy else disease_name,
        user_input_crop=crop_name,
        language=language,
        kb_treatment=final_treatment,
        analysis_raw=analysis_raw,
        original_image_url=original_url,
        bbox_image_url=processed_url
    )
