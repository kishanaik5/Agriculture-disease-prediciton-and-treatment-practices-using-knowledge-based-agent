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
from datetime import datetime, timezone
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
    - hi: Hindi
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
        elif lang == 'hi':
            stmt = stmt.where(MasterIcon.name_hi.ilike(f"%{name}%"))
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
        elif lang == 'hi' and i.name_hi:
            display_name = i.name_hi
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
    user_input_item: Optional[str] = None

@router.get("/all_reports", tags=["Show Details"], response_model=ListModel[ReportSummary])
async def get_all_reports(
    user_id: str = Depends(require_auth),
    language: Optional[str] = None,  # Optional - if None, retrieve all languages
    category: Optional[str] = None, # Optional
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get generic list of reports with summary details.
    If language is not provided, returns reports from all languages.
    """
    cat = None
    if category:
        cat = category.lower().strip()
        if cat.endswith('s') and cat != "crops": cat = cat[:-1]
        elif cat == "crops": cat = "crop"
    
    # We need to construct a specific query to get detailed fields
    # Fields: user_id, report_id, category, created_at, severity, disease_name, status, crop_name, user_input_item
    
    items = []
    total_count = 0
    
    # Query TranslatedAnalysisReport table only
    stmt = select(
        TranslatedAnalysisReport.user_id,
        TranslatedAnalysisReport.report_uid.label("report_id"),
        TranslatedAnalysisReport.category_type.label("category"),
        TranslatedAnalysisReport.created_at,
        TranslatedAnalysisReport.severity,
        TranslatedAnalysisReport.disease_name,
        TranslatedAnalysisReport.status,
        TranslatedAnalysisReport.item_name.label("crop_name"),
        TranslatedAnalysisReport.user_input_item
    ).where(
        TranslatedAnalysisReport.user_id == user_id,
        TranslatedAnalysisReport.status.in_(['PROCESSING', 'SUCCESS'])
    )
    
    # Apply language filter only if language is provided
    if language:
        stmt = stmt.where(TranslatedAnalysisReport.language == language)
    
    # Apply category filter if provided
    if cat:
        stmt = stmt.where(TranslatedAnalysisReport.category_type == cat)
    
    # Get total count
    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(stmt.subquery())
    c_res = await db.execute(count_stmt)
    total_count = c_res.scalar()
    
    # Get paginated items
    stmt = stmt.order_by(TranslatedAnalysisReport.created_at.desc()).limit(limit).offset(offset)
    res = await db.execute(stmt)
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
            crop_name=r.crop_name,
            user_input_item=r.user_input_item
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
    """
    settings = get_settings()
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
                resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if not data.get("allowed", False):
                         # Service returned "not allowed" - quota exceeded
                        reason = data.get("reason", "Subscription limit exceeded")
                        logger.warning(f"Subscription limit exceeded for user {user_id}: {reason}")
                        raise SubscriptionLimitExceeded(reason)
                    return True
                    
                elif resp.status_code == 400:
                    raise SubscriptionInvalid()
                elif resp.status_code == 404:
                    raise SubscriptionInvalid()
                elif resp.status_code >= 500:
                    raise SubscriptionServiceUnavailable()
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                    
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
                continue
        except Exception as e:
            logger.error(f"Subscription Verify Error: {e}")
            pass
            
    # Default to True in dev if service fails? No, safe fail.
    # But for now, let's allow if service is down to avoid blocking dev work?
    # raise SubscriptionServiceUnavailable("Subscription service unavailable")
    # TEMPORARY: Allow if service fails/unreachable for dev convenience if needed. 
    # But sticking to strict logic:
    raise SubscriptionServiceUnavailable("Service unavailable")

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




@router.post("/qa/scan", response_model=ScanResponse, tags=["QA_Scan"])
async def analyze_qa(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    category_id: str = Form(...),
    language: str = Form("en", description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or or"),
    subscription_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    QA Scan - Now unified with standard flow.
    """
    # 0. Validate Subscription (As requested)
    if subscription_id:
        await verify_consumption(user_id, subscription_id, action="scan")

    # 1. Lookup Category
    stmt = select(MasterIcon).where(MasterIcon.category_id == category_id)
    res = await db.execute(stmt)
    icon = res.scalars().first()
    
    if not icon:
        raise HTTPException(status_code=400, detail="Invalid category_id")
        
    crop_name_en = icon.name_en
    # Localized
    crop_name_localized = getattr(icon, f"name_{language}", icon.name_en) 
    if not crop_name_localized: crop_name_localized = icon.name_en
    
    cat_type = icon.category_type # e.g. crop, fruit, vegetable
    
    # 2. Upload
    image_bytes = await file.read()
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Image upload failed")
        
    # 3. Create PENDING Report
    new_uid = f"reports_{uuid.uuid4()}"
    
    new_report = TranslatedAnalysisReport(
        report_uid=new_uid,
        user_id=user_id,
        language=language,
        category_type=cat_type, # inferred from ID
        user_input_item=crop_name_localized, 
        item_name=crop_name_en, 
        original_image_url=original_url,
        status="PENDING",
        order_id=subscription_id,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    
    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)

    # PROCESS SYNCHRONOUSLY (User requested: "let qa scan take time")
    # We await the background task logic here
    try:
        # QA Scan uses is_qa=True for "QA Prompt"
        await _core_process_generation_v2(db, new_uid, cat_type, language, user_id, is_qa=True)
        # Refresh to get updated fields
        await db.refresh(new_report)
    except Exception as e:
        logger.error(f"Sync QA Analysis Failed: {e}")
        # Even if failed, we return the report state (which should be FAILED)
        await db.refresh(new_report)

    return ScanResponse(
        id=new_uid,
        user_id=user_id,
        created_at=new_report.created_at,
        disease_detected=new_report.disease_name or "Healthy", # Handle Healthy case
        user_input_crop=new_report.user_input_item,
        language=language,
        category=cat_type,
        kb_treatment=new_report.treatment or "",
        analysis_raw=new_report.analysis_raw or {},
        original_image_url=original_url,
        bbox_image_url=new_report.bbox_image_url,
        status=new_report.status
    )
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
    is_qa: bool = False, # Added for QA Scan
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
@router.get("/report/item", tags=["Report Generation"])
async def get_report_item(
    report_id: str = Query(...),
    category: str = Query(...),
    # language: str = Query("en"), # REMOVED: Language inferred from stored report
    user_id: str = Depends(require_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """
    Merged endpoint for Status + Generate + Details.
    Now operates primarily on TranslatedAnalysisReport.
    """
    cat = category.lower().strip()
    if cat.endswith('s') and cat != "crops": 
        cat = cat[:-1]
    elif cat == "crops": 
        cat = "crop"
    
    # 1. Fetch Report (Try Translated First - The new single source)
    stmt = select(TranslatedAnalysisReport).where(TranslatedAnalysisReport.report_uid == report_id)
    res = await db.execute(stmt)
    report = res.scalars().first()
    
    # Fallback to Legacy Tables if not found (for old reports)
    if not report:
        return await _handle_legacy_report(report_id, cat, user_id, db)

    # 2. Verify user ownership
    if report.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You don't have access to this report")

    # 3. Get current status
    current_status = report.status 
    if not current_status: 
        current_status = "PROCESSING"
    
    lang = report.language or "en"
    
    # 3. Logic Branching
    
    # Case A: Needs Generation (PENDING/PROCESSING/QA_SUCCESS)
    if current_status in ["PENDING", "PROCESSING", "QA_SUCCESS"]:
        # Update to PROCESSING if needed
        is_upgrade = False
        if current_status == "PENDING" or current_status == "QA_SUCCESS":
             # If QA_SUCCESS, we are UPGRADING to Full Report
             if current_status == "QA_SUCCESS": is_upgrade = True
             report.status = "PROCESSING"
             await db.commit()
        
        # Determine if we need to trigger generation
        # If it's PENDING/PROCESSING/QA_SUCCESS, we assume generation is running or needs to start.
        # But since we use background tasks, we should check if we need to re-trigger.
        # For simplicity, if it is PENDING or was QA_SUCCESS, we trigger.
        
        # Trigger Background Worker
        async def background_wrapper_new(r_uid, u_id, l, c):
             async for s in get_db():
                 try:
                     # Always run Full Analysis (is_qa=False) in background upgrade
                     await _core_process_generation_v2(s, r_uid, c, l, u_id, is_qa=False)
                 except Exception as e:
                     logger.error(f"background_wrapper_new failed: {e}")
                 finally:
                     await s.close()

        # Triggering indiscriminately might duplicate work if not careful.
        # Ideally, we trigger ONLY if we just created it (in POST).
        # But POST is asyncfire? POST creates PENDING. GET triggers? 
        # User said "in the qa... generate the report...". 
        # Let's assume GET triggers if it's PENDING.
        if current_status == "PENDING" or is_upgrade:
             background_tasks.add_task(background_wrapper_new, report_id, user_id, lang, cat)
             report.status = "PROCESSING" # Mark as processing to avoid double trigger
             await db.commit()
        
        details = {
            "qa": "DONE",
            "generation": "PROCESSING"
        }
        
        return {
            "user_id": user_id,
            "report_id": report.report_uid,
            "language": lang,
            "details": details,
            "status": "PROCESSING",
             "user_input_crop": report.user_input_item or report.item_name
        }

    # Case B: Failed
    if current_status == "FAILED":
        return {
            "user_id": user_id,
            "report_id": report.report_uid,
            "language": lang,
            "details": {"generation": "FAILED"},
            "status": "FAILED"
        }

    # Case C: Success
    if current_status == "SUCCESS":
        return {
            "id": report.report_uid,
            "user_id": report.user_id,
            "created_at": report.created_at,
            "disease_detected": report.disease_name,
            "user_input_crop": report.user_input_item or report.item_name,
            "category": cat,
            "language": lang,
            "kb_treatment": report.treatment,
            "analysis_raw": report.analysis_raw,
            "original_image_url": report.original_image_url,
            "bbox_image_url": report.bbox_image_url,
            "status": "SUCCESS"
        }

    return {"status": "UNKNOWN"}


async def _handle_legacy_report(report_id, cat, user_id, db):
    """Fallback for reading old reports from specific tables"""
    # Priority search: Requested category first, then others
    models_map = {
        "crop": AnalysisReport,
        "fruit": FruitAnalysis,
        "vegetable": VegetableAnalysis
    }
    
    # Order of search
    search_order = []
    if cat in models_map:
        search_order.append((cat, models_map[cat]))
    
    for c, m in models_map.items():
        if c != cat:
            search_order.append((c, m))
            
    found_report = None
    found_cat = None
    
    for category_key, report_model in search_order:
        res = await db.execute(select(report_model).where(report_model.uid == report_id))
        report = res.scalars().first()
        if report:
            found_report = report
            found_cat = category_key
            break
            
    if not found_report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Verify user ownership for legacy reports
    if found_report.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You don't have access to this report")
    
    # Return formatted legacy report
    return {
        "id": found_report.uid,
        "user_id": found_report.user_id,
        "created_at": found_report.created_at,
        "disease_detected": found_report.disease_name,
        "user_input_crop": getattr(found_report, "crop_name", None) or getattr(found_report, "fruit_name", None) or getattr(found_report, "vegetable_name", None),
        "language": "en",
        "category": found_cat, # Return actual category
        "kb_treatment": found_report.treatment,
        "analysis_raw": found_report.analysis_raw,
        "original_image_url": found_report.original_image_url,
        "bbox_image_url": found_report.bbox_image_url,
        "status": found_report.status
    }

async def _core_process_generation_v2(db, report_uid, cat, lang, user_id, is_qa: bool = False):
    """
    Optimized Generation:
    1. Fetch PENDING record.
    2. Analyze (EN).
    3. KB (EN).
    4. Translate (if lang!=en).
    5. Update Record.
    """
    stmt = select(TranslatedAnalysisReport).where(TranslatedAnalysisReport.report_uid == report_uid)
    res = await db.execute(stmt)
    report = res.scalars().first()
    
    if not report or report.status == "SUCCESS": return
    
    cat = cat.lower() # Normalize category to avoid case mismatch logic errors

    try:
        # 1. Download Image
        key = report.original_image_url.split(".com/")[-1]
        s3_obj = s3_service_public.s3_client.get_object(Bucket=s3_service_public.bucket_name, Key=key)
        image_bytes = s3_obj['Body'].read()
        
        # 2. Analyze (English Context ALWAYS first for best accuracy)
        analysis_raw = {}
        bbox_list = []
        user_input = report.item_name # Always use English Name (item_name) for Analysis/KB
        user_display_name = report.user_input_item # Localized name for display
        
        # Initialize variables to avoid UnboundLocalError
        disease_name = "Unknown"
        sci_name = ""
        severity = ""
        grade = None
        
        # Analyze based on category
        if cat == "crop":
             if is_qa:
                 analysis_raw = await gemini_service.analyze_crop_qa(image_bytes, user_input, "en", "image/jpeg")
                 disease_name = analysis_raw.get("disease_name") or "Unknown"
                 sci_name = "" # QA doesn't give sci name usually
                 severity = ""
             else:
                 analysis_raw = await gemini_service.analyze_crop(image_bytes, user_input, "en", "image/jpeg")
                 d_info = analysis_raw.get("disease_info", {})
                 disease_name = d_info.get("common_name", d_info.get("disease_name", "Unknown"))
                 sci_name = d_info.get("scientific_name", "")
                 severity = str(d_info.get("severity") or "")
             
        elif cat == "fruit":
             if is_qa:
                  analysis_raw = await gemini_service.analyze_fruit_qa(image_bytes, user_input, "en", "image/jpeg")
                  disease_name = analysis_raw.get("disease_name") or "Unknown"
                  sci_name = ""
                  severity = ""
             else:
                  analysis_raw = await gemini_service.analyze_fruit(image_bytes, user_input, "en", "image/jpeg")
                  diag = analysis_raw.get("diagnosis", {})
                  market = analysis_raw.get("market_quality", {})
                  disease_name = diag.get("disease_name", "Unknown")
                  sci_name = diag.get("scientific_name", "")
                  severity = diag.get("severity_stage")
                  grade = market.get("grade")
             
        elif cat == "vegetable":
             if is_qa:
                  analysis_raw = await gemini_service.analyze_vegetable_qa(image_bytes, user_input, "en", "image/jpeg")
                  disease_name = analysis_raw.get("disease_name") or "Unknown"
                  sci_name = ""
                  severity = ""
             else:
                  analysis_raw = await gemini_service.analyze_vegetable(image_bytes, user_input, "en", "image/jpeg")
                  d_info = analysis_raw.get("disease_info", {})
                  market = analysis_raw.get("marketability", {})
                  disease_name = d_info.get("common_name", d_info.get("disease_name", "Unknown"))
                  sci_name = d_info.get("scientific_name", "")
                  severity = d_info.get("severity")
                  grade = market.get("export_grade")
        
        # 3. KB Lookup (English)
        kb_text = await knowledge_service.get_treatment(user_input, disease_name, category=cat, db=db, scientific_name=sci_name, language="en")
        
        if not kb_text:
             kb_text = "Consult a Doctor"
        
        # 4. BBox Generation (If disease)
        processed_url = None
        is_healthy = False
        lower_name = str(disease_name).lower()
        if "healthy" in lower_name or "none" in lower_name: is_healthy = True
        
        if not is_healthy and ENABLE_BOUNDING_BOX_GENERATION:
             # calls gemini bbox...
             pass # Skip for brevity or implement if needed
             
        # 5. Prepare Data for Saving
        # If target lang is NOT English, we translate the whole bundle
        final_disease = disease_name
        final_treatment = kb_text
        final_user_input = user_input
        final_analysis = analysis_raw
        
        if lang != "en":
              from app.services.translation_service import translation_service
              # Construct data dict
              data_to_trans = {
                  "item_name": user_input,
                  "disease_name": disease_name,
                  "scientific_name": sci_name,
                  "severity": severity,
                  "treatment": kb_text,
                  "analysis_raw": analysis_raw
              }
              translated = await translation_service.translate_dictionary(data_to_trans, lang)
              
              final_disease = translated.get("disease_name")
              final_treatment = translated.get("treatment")
              final_user_input = translated.get("item_name")
              final_analysis = translated.get("analysis_raw")
              # Scientific name usually stays latin/english but can be transliterated
              
        # 6. Update Record
        report.disease_name = final_disease
        report.scientific_name = sci_name  # Scientific names stay in Latin/English
        report.severity = severity
        if grade:  # Only set grade if it exists (for fruits/vegetables)
            report.grade = grade
        report.treatment = final_treatment
        report.analysis_raw = final_analysis
        report.status = "QA_SUCCESS" if is_qa else "SUCCESS"
        # report.user_input_item = final_user_input # Keep original input or updated? User said "user_input_item should have that item value in that desired language"
        if lang != 'en':
             report.item_name = final_user_input # item_name is usually the translated one
        
        await db.commit()
        
        # 7. Confirm Consumption
        await _confirm_background_consumption(user_id, report.order_id)
    
    except Exception as e:
        logger.error(f"Core V2 Failed: {e}")
        report.status = "FAILED"
        await db.commit()

# Stub for BBox helper
async def _process_bbox(image_bytes, disease_name):
    pass 


async def analyze_crop(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    category_id: str = Form(...),
    language: str = Form("en", description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or or"),
    subscription_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Optimized Workflow:
    1. Validate Subscription.
    2. Lookup Category ID -> Name.
    3. Upload Original -> S3.
    4. Create PENDING record.
    5. Return PENDING response.
    """
    # 0. Validate Subscription
    if subscription_id:
        await verify_consumption(user_id, subscription_id, action="scan")
    
    # 1. Lookup Category Name
    stmt = select(MasterIcon).where(MasterIcon.category_id == category_id)
    res = await db.execute(stmt)
    icon = res.scalars().first()
    
    if not icon:
        raise HTTPException(status_code=400, detail="Invalid category_id")
    
    crop_name_en = icon.name_en # Canonical English Name
    
    # Get Localized Name
    # Fallback to English if localized not available
    crop_name_localized = getattr(icon, f"name_{language}", icon.name_en) 
    if not crop_name_localized: crop_name_localized = icon.name_en
    
    # 2. Read & Upload Image
    image_bytes = await file.read()
    
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Image upload failed")
        
    # 3. Create PENDING Report
    new_uid = f"reports_{uuid.uuid4()}"
    
    new_report = TranslatedAnalysisReport(
        report_uid=new_uid,
        user_id=user_id,
        language=language,
        category_type="crop",
        user_input_item=crop_name_localized, # Store localized name for user
        item_name=crop_name_en, # Store English name for logic
        original_image_url=original_url,
        status="PENDING",
        order_id=subscription_id, # Store subscription_id for consumption later
        created_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    
    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)
    
    # 4. Return PENDING Response
    return ScanResponse(
        id=new_uid,
        user_id=user_id,
        created_at=new_report.created_at,
        disease_detected="Pending",
        user_input_crop=crop_name_localized,
        language=language,
        kb_treatment="",
        analysis_raw={},
        original_image_url=original_url,
        bbox_image_url=None,
        status="PENDING"
    )

async def analyze_vegetable(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    category_id: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    subscription_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Optimized Vegetable Scan
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 0. Validate Subscription
    if subscription_id:
        await verify_consumption(user_id, subscription_id, action="scan")

    # 1. Lookup Name
    stmt = select(MasterIcon).where(MasterIcon.category_id == category_id)
    res = await db.execute(stmt)
    icon = res.scalars().first()
    if not icon: raise HTTPException(status_code=400, detail="Invalid category_id")
    
    crop_name_en = icon.name_en
    # Localized Name
    crop_name_localized = getattr(icon, f"name_{language}", icon.name_en) 
    if not crop_name_localized: crop_name_localized = icon.name_en

    # 2. Read & Upload
    image_bytes = await file.read()
    
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Image upload failed")

    # 3. Create PENDING Report
    new_uid = f"reports_{uuid.uuid4()}"
    
    new_report = TranslatedAnalysisReport(
        report_uid=new_uid,
        user_id=user_id,
        language=language,
        category_type="vegetable",
        user_input_item=crop_name_localized,
        item_name=crop_name_en,
        original_image_url=original_url,
        status="PENDING",
        order_id=subscription_id,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    
    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)
    
    return ScanResponse(
        id=new_uid,
        user_id=user_id,
        created_at=new_report.created_at,
        disease_detected="Pending",
        user_input_crop=crop_name_localized,
        language=language,
        kb_treatment="",
        analysis_raw={},
        original_image_url=original_url,
        bbox_image_url=None,
        status="PENDING"
    )

async def analyze_fruit(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    category_id: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    subscription_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Optimized Fruit Scan
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 0. Validate Subscription
    if subscription_id:
        await verify_consumption(user_id, subscription_id, action="scan")

    # 1. Lookup Name
    stmt = select(MasterIcon).where(MasterIcon.category_id == category_id)
    res = await db.execute(stmt)
    icon = res.scalars().first()
    if not icon: raise HTTPException(status_code=400, detail="Invalid category_id")
    
    crop_name_en = icon.name_en
    # Localized Name
    crop_name_localized = getattr(icon, f"name_{language}", icon.name_en) 
    if not crop_name_localized: crop_name_localized = icon.name_en

    # 2. Read & Upload
    image_bytes = await file.read()
    
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Image upload failed")

    # 3. Create PENDING Report
    new_uid = f"reports_{uuid.uuid4()}"
    
    new_report = TranslatedAnalysisReport(
        report_uid=new_uid,
        user_id=user_id,
        language=language,
        category_type="fruit",
        user_input_item=crop_name_localized,
        item_name=crop_name_en,
        original_image_url=original_url,
        status="PENDING",
        order_id=subscription_id,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    
    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)
    
    return ScanResponse(
        id=new_uid,
        user_id=user_id,
        created_at=new_report.created_at,
        disease_detected="Pending",
        user_input_crop=crop_name_localized,
        language=language,
        kb_treatment="",
        analysis_raw={},
        original_image_url=original_url,
        bbox_image_url=None,
        status="PENDING"
    )

@router.post("/{category}_scan", response_model=ScanResponse, tags=["Generate Report"])
async def analyze_dynamic(
    category: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
    category_id: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    subscription_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Universal endpoint for crop/fruit/vegetable scans via path parameter.
    """
    cat = category.lower()
    
    # Dispatch to specific handlers
    if cat in ["crop", "crops"]:
        return await analyze_crop(
            background_tasks=background_tasks,
            file=file,
            user_id=user_id,
            category_id=category_id,
            language=language,
            subscription_id=subscription_id,
            db=db
        )
        
    elif cat in ["fruit", "fruits"]:
        return await analyze_fruit(
            file=file,
            user_id=user_id,
            category_id=category_id,
            language=language,
            subscription_id=subscription_id,
            db=db
        )
        
    elif cat in ["vegetable", "vegetables"]:
        return await analyze_vegetable(
            file=file,
            user_id=user_id,
            category_id=category_id,
            language=language,
            subscription_id=subscription_id,
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
