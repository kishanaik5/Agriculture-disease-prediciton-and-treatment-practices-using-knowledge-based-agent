from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.models.scan import AnalysisReport, FruitAnalysis, VegetableAnalysis, TranslatedAnalysisReport, MasterIcon, ReportAmount, PaymentTransaction
# from app.models.payment import PaymentTransaction # Removed incorrect import
from app.schemas.scan import ScanResponse, AnalysisResult, CropItem
from app.services.gemini import gemini_service
from app.services.s3 import s3_service_public, s3_service_private
from app.services.image import image_service
import uuid
import logging
import io
import json
import os
from datetime import datetime
from typing import Optional, Any, Dict
from app.services.knowledge import knowledge_service
from zoneinfo import ZoneInfo
from app.database import SessionLocal
# from app.services.redis_manager import task_manager # Removed Redis dependency

logger = logging.getLogger(__name__)

class BytesUploadFile:
    def __init__(self, data: bytes, filename: str, content_type: str = "image/jpeg"):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        return self._data

router = APIRouter()



from app.models.scan import MasterIcon

@router.get("/All", tags=["Show Details"])
async def get_all_items(language: str = "en", name: Optional[str] = None, category: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """
    Get all supported items from MasterIcon DB.
    """
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
        
    if name:
        # Search in relevant language or english?
        # User said "manually make a changes for the language", so maybe search in en?
        stmt = stmt.where(MasterIcon.name_en.ilike(f"%{name}%"))
        
    res = await db.execute(stmt)
    items = res.scalars().all()
    
    # Format Response
    resp_list = []
    for i in items:
        # Select name based on lang
        display_name = i.name_en
        if language == 'kn' and i.name_kn:
            display_name = i.name_kn
        elif language == 'hi' and i.name_hn:
            display_name = i.name_hn
            
        resp_list.append({
            "Name": display_name,
            "image": i.url,
            "category": i.category_type,
            "category_id": i.category_id
        })
        
    return resp_list

    return resp_list


@router.get("/all_reports", tags=["Show Details"])
async def get_all_reports(
    user_id: str,
    language: str, # Mandatory
    category: Optional[str] = None, # Optional
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all report IDs for a user, filtered by language and optionally category.
    Results are ordered by created_at DESC.
    """
    stmt = None
    
    # Normalize category if present
    cat = None
    if category:
        cat = category.lower().strip()
        if cat.endswith('s') and cat != "crops": 
            cat = cat[:-1]
        elif cat == "crops":
            cat = "crop"
    
    if language == "en":
        if cat:
            # Specific Category
            model = None
            if cat == "crop":
                model = AnalysisReport
            elif cat == "fruit":
                model = FruitAnalysis
            elif cat == "vegetable":
                model = VegetableAnalysis
            else:
                return []
                
            stmt = select(model.uid).where(model.user_id == user_id).order_by(model.created_at.desc())
        else:
            # UNION ALL of all 3 tables
            # We select uid and created_at to order by created_at
            
            s1 = select(AnalysisReport.uid, AnalysisReport.created_at).where(AnalysisReport.user_id == user_id)
            s2 = select(FruitAnalysis.uid, FruitAnalysis.created_at).where(FruitAnalysis.user_id == user_id)
            s3 = select(VegetableAnalysis.uid, VegetableAnalysis.created_at).where(VegetableAnalysis.user_id == user_id)
            
            from sqlalchemy import union_all, literal_column
            
            # Combine
            u_stmt = union_all(s1, s2, s3).subquery()
            
            # Select from union and order
            stmt = select(u_stmt.c.uid).order_by(u_stmt.c.created_at.desc())
            
    else:
        # Translated Analysis Report
        stmt = select(TranslatedAnalysisReport.uid).where(
            TranslatedAnalysisReport.user_id == user_id,
            TranslatedAnalysisReport.language == language
        )
        
        if cat:
            stmt = stmt.where(TranslatedAnalysisReport.category_type == cat)
            
        stmt = stmt.order_by(TranslatedAnalysisReport.created_at.desc())
        
    # Apply Limit and Offset
    stmt = stmt.limit(limit).offset(offset)
    
    res = await db.execute(stmt)
    report_ids = res.scalars().all()
    
    return report_ids


@router.get("/get_price", tags=["Show Details"])
async def get_price(category: str, db: AsyncSession = Depends(get_db)):
    """
    Get price for a specific category.
    """
    cat = category.lower().strip()
    if cat.endswith('s') and cat != "crops": cat = cat[:-1] # simple normalization
    
    stmt = select(ReportAmount).where(ReportAmount.category == cat)
    res = await db.execute(stmt)
    item = res.scalars().first()
    
    if not item:
        return {"category": cat, "amount": 0.0} # Default or error? Return 0 as default.
        
    return {"category": item.category, "amount": item.amount}
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

@router.post("/set_category_report_price", tags=["Admin Section"])
async def set_category_report_price(
    category: str = Form(...),
    amount: float = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Set or Update report price for a category.
    """
    cat = category.lower().strip()
    if cat.endswith('s') and cat != "crops": cat = cat[:-1]

    # Check existing
    stmt = select(ReportAmount).where(ReportAmount.category == cat)
    res = await db.execute(stmt)
    existing = res.scalars().first()
    
    if existing:
        existing.amount = amount
    else:
        new_price = ReportAmount(category=cat, amount=amount)
        db.add(new_price)
        
    await db.commit()
    return {"message": "Price updated", "category": cat, "amount": amount}

@router.post("/qa/scan", response_model=ScanResponse, tags=["QA_Scan"])
async def analyze_qa(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    crop_name: str = Form(...),
    category: str = Form(...), # crop, fruit, vegetable
    language: str = Form("en"), # Added language support
    db: AsyncSession = Depends(get_db)
):
    """
    Stage 1: QA / Validity Check.
    - Uploads to S3.
    - Checks if valid crop and if healthy (Gemini Flash).
    - If HEALTHY -> Returns immediately (No DB, No Output ID).
    - If UNHEALTHY -> Creates DB record (PENDING), Returns Report ID.
    - If INVALID -> Error.
    """
    # language = "en" # Use input
    
    # 1. Read Image Bytes
    image_bytes = await file.read()
    
    # 2. QA Analysis (Flash) - Analyze BEFORE upload to save costs
    cat = category.lower()
    qa_result = {}
    
    try:
        if cat in ["crop", "crops"]:
            qa_result = await gemini_service.analyze_crop_qa(image_bytes, crop_name, language, file.content_type)
        elif cat in ["fruit", "fruits"]:
            qa_result = await gemini_service.analyze_fruit_qa(image_bytes, crop_name, language, file.content_type)
        elif cat in ["vegetable", "vegetables"]:
            qa_result = await gemini_service.analyze_vegetable_qa(image_bytes, crop_name, language, file.content_type)
        else:
             raise HTTPException(status_code=400, detail="Invalid category")
    except Exception as e:
        logger.error(f"QA Scan failed: {e}")
        raise HTTPException(status_code=500, detail="AI Service failed")
        
    # 3. Validation
    if not qa_result.get("is_valid_crop", True):
        detected = qa_result.get("detected_object_name", "Unknown")
        msg = f"The uploaded image appears to be '{detected}', but you selected '{crop_name}'. Please upload a valid image."
        if language == 'kn':
            msg = f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರವು '{detected}' ಎನಿಸುತ್ತದೆ, ಆದರೆ ನೀವು '{crop_name}' ಆಯ್ಕೆ ಮಾಡಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ಸರಿಯಾದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
        raise HTTPException(status_code=400, detail=msg)
        
    # 4. Health Check
    is_healthy = qa_result.get("is_healthy", False)
    disease_name = qa_result.get("disease_name", "No Disease")
    
    if is_healthy:
        # RETURN IMMEDIATELY - NO DB - NO S3
        msg = "Your crop is healthy! No disease detected."
        if language == 'kn':
            msg = "ನಿಮ್ಮ ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ! ಯಾವುದೇ ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ."
            
        return ScanResponse(
            id="0",
            user_id=user_id,
            created_at=datetime.now(),
            disease_detected=msg, # Friendly message in disease field or separate? Using field for now.
            user_input_crop=crop_name,
            language=language,
            original_image_url=None, # Skipped S3
            analysis_raw=qa_result,
            category=category.lower().rstrip('s') if category.lower() not in ['crop', 'crops'] else 'crop' 
        )
        
    # 5. Unhealthy -> Upload to S3 & Create Partial DB Record
    
    # Upload S3
    filename_uuid = str(uuid.uuid4())
    # Reset cursor if needed, but we have bytes
    try:
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        # If upload fails, we might still want to return something or fail? 
        # Failing safe for now.
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

    # Create record with PENDING payment
    current_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    scan_id = "0"
    
    if cat in ["crop", "crops"]:
        report = AnalysisReport(
            user_id=user_id,
            crop_name=crop_name, # Renamed from user_input_crop, maps to crop_name column
            language="en",
            desired_language_output=language,
            original_image_url=original_url,
            disease_name=disease_name, 
            payment_status="PENDING",
            created_at=current_time 
        )
        db.add(report)
        await db.commit()
        await db.refresh(report)
        scan_id = report.uid
        
    elif cat in ["fruit", "fruits"]:
        report = FruitAnalysis(
            user_id=user_id,
            fruit_name=crop_name,
            language="en",
            desired_language_output=language,
            original_image_url=original_url,
            disease_name=disease_name,
            payment_status="PENDING",
             created_at=current_time
        )
        db.add(report)
        await db.commit()
        await db.refresh(report)
        scan_id = report.uid
        
    elif cat in ["vegetable", "vegetables"]:
        report = VegetableAnalysis(
             user_id=user_id,
             vegetable_name=crop_name,
             language="en",
             desired_language_output=language,
             original_image_url=original_url,
             disease_name=disease_name,
             payment_status="PENDING",
             created_at=current_time
        )
        db.add(report)
        await db.commit()
        await db.refresh(report)
        scan_id = report.uid

    # Return with ID so frontend can pay
    return ScanResponse(
        id=scan_id,
        user_id=user_id,
        created_at=current_time,
        disease_detected=disease_name,
        user_input_crop=crop_name,
        language=language,
        original_image_url=original_url,
        category=cat.rstrip('s') if cat not in ['crop', 'crops'] else 'crop' 
    )

# REFACTOR: Removed AsyncJob helper. Status is now derived from data.
async def _update_job_status(db: AsyncSession, job_id: str, status: str, details: dict = None):
    pass # No-op

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
        "payment": "SUCCESS",
        "generating_report": "PENDING",
        "translating": "PENDING",
        "progress": 30,
        "stage": "Starting Generation"
    }
    
    if task_id:
        await _update_job_status(db, task_id, "processing", state)

    try:
        # Download Image
        try:
            key = report.original_image_url.split(".com/")[-1]
            s3_obj = s3_service_public.s3_client.get_object(Bucket=s3_service_public.bucket_name, Key=key)
            image_bytes = s3_obj['Body'].read()
        except Exception as e:
            logger.error(f"Failed to retrieve image: {e}")
            state["generating_report"] = "FAILED"
            if task_id: await _update_job_status(db, task_id, "failed", {"error": "Image Fetch Failed", **state})
            raise HTTPException(status_code=500, detail=f"Could not retrieve image: {e}")

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
            state["generating_report"] = "FAILED"
            if task_id: await _update_job_status(db, task_id, "failed", {"error": f"Analysis Failed: {e}", **state})
            raise HTTPException(status_code=500, detail="Analysis Failed")

        # Process BBox Image
        processed_url = None
        if bbox_list:
            try:
                p_bytes = image_service.draw_bounding_boxes(image_bytes, bbox_list)
                new_key = f"processed_{uuid.uuid4()}.jpg"
                p_wrapper = BytesUploadFile(p_bytes, new_key, "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=p_wrapper, prefix="dev/processed/")
            except Exception as e:
                logger.error(f"BBox draw/upload failed: {e}")

        if task_id:
            state["progress"] = 60
            state["stage"] = "Database Update"
            await _update_job_status(db, task_id, "processing", state)

        # Update DB (Base English Report)
        report.analysis_raw = full_analysis
        report.bbox_image_url = processed_url
        report.desired_language_output = lang

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

        report.payment_status = 'SUCCESS'
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
    user_id: str = Form(...),
    report_id: str = Form(...),
    category: str = Form(...), 
    language: str = Form(..., description="Language code: en or kn"), 
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
    
@router.post("/report/generate_async", tags=["Generate Report"])
async def generate_report_async(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    report_id: str = Form(...),
    category: str = Form(...), 
    language: str = Form(..., description="Language code: en or kn"), 
    db: AsyncSession = Depends(get_db)
):
    """
    Stage 2: Full Analysis (Async).
    Returns task_id immediately.
    """
    cat = category.lower().strip()
    lang = language.lower().strip()
    
    logger.info(f"🚀 Generate Report Request (Async) | ID: {report_id} | Cat: {cat} | Lang: {lang}")

    # 1. Verify Payment (Fail fast)
    stmt = select(PaymentTransaction).where(
        PaymentTransaction.analysis_report_uid == report_id,
        PaymentTransaction.payment_status == 'SUCCESS'
    )
    result = await db.execute(stmt)
    tx = result.scalars().first()
    
    if not tx:
        raise HTTPException(status_code=402, detail="Payment Required or Not Success")
        
    # Check if report exists
    report_model = None
    if cat in ["crop", "crops"]: report_model = AnalysisReport
    elif cat in ["fruit", "fruits"]: report_model = FruitAnalysis
    elif cat in ["vegetable", "vegetables"]: report_model = VegetableAnalysis
    
    if not report_model: raise HTTPException(status_code=400, detail="Invalid Category")

    # Get Partial Report to pass to worker
    r_stmt = select(report_model).where(report_model.uid == report_id)
    r_res = await db.execute(r_stmt)
    report = r_res.scalars().first()
    
    if not report or not report.original_image_url:
        raise HTTPException(status_code=404, detail="Report or Image not found")
    
    # 2. Queue Task
    task_id = str(uuid.uuid4())
    
    # Create AsyncJob Record - REMOVED per user request
    # new_job = AsyncJob(...)
    # db.add(new_job)
    # await db.commit()
    
    background_tasks.add_task(
        background_generation_worker,
        task_id=task_id,
        report=report,
        cat=cat,
        lang=lang,
        user_id=user_id
    )
    
    return {
        "task_id": task_id,
        "status": "processing", # Always processing initially
        "message": "Report generation started. Poll /report/status with user_id & report_id.",
        "user_id": user_id,
        "report_id": report_id,
        "language": lang
    }

@router.get("/report/status", tags=["Generate Report"])
async def get_generation_status(
    user_id: str,
    report_id: str,
    language: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """"Check async generation status by user/report context."""
    # 1. Identify Report Type
    # We can try to guess from prefix or just query Transaction first
    # report_id usually starts with "crop_analysis_report" etc?
    
    # Check Transaction
    stmt = select(PaymentTransaction).where(
        PaymentTransaction.analysis_report_uid == report_id
    )
    res = await db.execute(stmt)
    tx = res.scalars().first()
    
    if not tx:
         raise HTTPException(status_code=404, detail="Transaction not found for this report")
         
    payment_status = tx.payment_status # SUCCESS / PENDING
    
    if payment_status != 'SUCCESS':
        return {
            "status": "pending_payment",
            "progress": 0,
            "details": {"payment": "PENDING"}
        }

    # 2. Check Analysis Report
    cat = tx.analysis_type # "crop", "fruit", "vegetable"
    if not cat: cat = "crop" # default
    
    model = None
    if "fruit" in cat.lower(): model = FruitAnalysis
    elif "veg" in cat.lower(): model = VegetableAnalysis
    else: model = AnalysisReport
    
    r_stmt = select(model).where(model.uid == report_id)
    r_res = await db.execute(r_stmt)
    report = r_res.scalars().first()
    
    if not report:
         # Should not happen if Tx exists, but maybe partial?
         return {"status": "processing", "progress": 10, "details": "Report record initialization"}
         
    # 3. Check Language
    current_lang = language or "en"
    
    if current_lang == "en":
        # Check if EN analysis is done
        if report.analysis_raw and report.bbox_image_url:
             return {
                 "status": "completed",
                 "progress": 100,
                 "stage": "Completed",
                 "details": {"qa": "DONE", "payment": "SUCCESS", "generation": "DONE"}
             }
        else:
             return {
                 "status": "processing",
                 "progress": 50,
                 "stage": "AI Analysis",
                 "details": {"qa": "DONE", "payment": "SUCCESS", "generation": "PROCESSING"}
             }
    else:
        # Check Translation
        # User requirement: if translation table has same report_id and user_id then status success
        t_stmt = select(TranslatedAnalysisReport).where(
            TranslatedAnalysisReport.report_uid == report_id,
            TranslatedAnalysisReport.language == current_lang
            # User ID check implied by report_uid uniqueness, but can add if column exists
        )
        t_res = await db.execute(t_stmt)
        translated = t_res.scalars().first()
        
        if translated:
             return {
                 "status": "completed",
                 "progress": 100,
                 "stage": "Completed",
                 "details": {"qa": "DONE", "payment": "SUCCESS", "generation": "DONE", "translation": "DONE"}
             }
        else:
             return {
                 "status": "processing", 
                 "progress": 75,
                 "stage": "Translating",
                 "details": {"qa": "DONE", "payment": "SUCCESS", "generation": "DONE", "translation": "PROCESSING"}
             }
    
@router.get("/report/details", response_model=ScanResponse, tags=["Generate Report"])
async def get_report_details(
    user_id: str,
    report_id: str,
    language: str = 'en', 
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve report details.
    - If language == en: Checks Base Report table (payment_status='SUCCESS').
    - If language != en: Checks Translation table (payment_status='SUCCESS').
    """
    lang = language.lower()
    
    # 1. Determine Category (Use Transaction for efficient lookup)
    stmt = select(PaymentTransaction).where(
        PaymentTransaction.analysis_report_uid == report_id,
        PaymentTransaction.user_id == user_id
    )
    result = await db.execute(stmt)
    tx = result.scalars().first()
    
    # If no transaction found, we can't easily find the report category without trying all 3 tables or asking user.
    # Assuming valid flow implies transaction exists.
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction record not found")

    category_type = tx.analysis_type
    
    # RESPONSE ARGS INITIALIZATION
    resp_args = {
        "user_id": user_id, 
        "language": lang,
        "category": category_type,
        "available_lang": {"en": True}
    }

    # 2. LOGIC BRANCHING
    if lang == 'en':
        # Fetch Base Report
        target_table = None
        if category_type == 'crop': target_table = AnalysisReport
        elif category_type == 'fruit': target_table = FruitAnalysis
        elif category_type == 'vegetable': target_table = VegetableAnalysis
        
        if not target_table:
             raise HTTPException(status_code=400, detail="Unknown Category Type")

        r_stmt = select(target_table).where(
            target_table.uid == report_id,
            target_table.user_id == user_id,
            target_table.payment_status == 'SUCCESS'
        )
        r_res = await db.execute(r_stmt)
        report = r_res.scalars().first()
        
        if not report:
             # Check if it exists but PENDING?
             # User requested "search and give report IF payment status is SUCCESS... else record not found"
             # So 404 is appropriate if not SUCCESS.
             raise HTTPException(status_code=404, detail="Report not found or Payment Pending")
             
        # Populate Response
        resp_args.update({
            "id": report.uid,
            "created_at": report.created_at,
            "analysis_raw": report.analysis_raw,
            "original_image_url": report.original_image_url,
            "bbox_image_url": report.bbox_image_url,
            "kb_treatment": report.treatment
        })
        
        # Category specific fields
        if category_type == 'crop':
            resp_args["user_input_crop"] = report.crop_name
            resp_args["disease_detected"] = report.disease_name
        elif category_type == 'fruit':
            resp_args["user_input_crop"] = report.fruit_name
            resp_args["disease_detected"] = report.disease_name
        elif category_type == 'vegetable':
            resp_args["user_input_crop"] = report.vegetable_name
            resp_args["disease_detected"] = report.disease_name
            
        # Available Lang Logic (Check Translations)
        t_stmt = select(TranslatedAnalysisReport.language).where(TranslatedAnalysisReport.report_uid == report_id)
        t_res = await db.execute(t_stmt)
        for existing_lang in t_res.scalars().all():
            resp_args["available_lang"][existing_lang] = True

        # Check intent if processing
        desired = getattr(report, "desired_language_output", "en")
        if desired and desired != "en" and desired not in resp_args["available_lang"]:
            resp_args["available_lang"][desired] = "PROCESSING"

    else:
        # Fetch Translated Report Directly
        t_stmt = select(TranslatedAnalysisReport).where(
            TranslatedAnalysisReport.report_uid == report_id,
            TranslatedAnalysisReport.language == lang,
            TranslatedAnalysisReport.user_id == user_id,
            TranslatedAnalysisReport.payment_status == 'SUCCESS'
        )
        t_res = await db.execute(t_stmt)
        report = t_res.scalars().first()
        
        if not report:
             # Check if processing?
             # User explicitly said: "see if ... matching language ... and payment_status SUCCESS"
             # So if not found/success, return 404.
             # We might want to check the base report to see if "PROCESSING" is valid?
             # But request says "record not found as response".
             raise HTTPException(status_code=404, detail="Translated Report not found")
             
        resp_args.update({
            "id": report.report_uid, # Maintain original ID reference? Or return translated record ID? Usually API returns resource ID.
            # But the 'id' field in response model usually matches report_id? 
            # Let's keep report_uid (Original) or report.uid (Translation)?
            # Model definition says id is Alias.
            # I'll use report.report_uid which maps to the Requested ID.
             "created_at": report.created_at,
             "analysis_raw": report.analysis_raw,
             "original_image_url": report.original_image_url,
             "bbox_image_url": report.bbox_image_url,
             "kb_treatment": report.treatment,
             "disease_detected": report.disease_name,
             "user_input_crop": report.item_name
        })
        
        # Populate available lang for Translation response too?
        # Usually good practice.
        t_stmt_all = select(TranslatedAnalysisReport.language).where(TranslatedAnalysisReport.report_uid == report_id)
        tr = await db.execute(t_stmt_all)
        for el in tr.scalars().all():
            resp_args["available_lang"][el] = True

    return ScanResponse(**resp_args)



async def analyze_crop(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # New Inputs
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
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
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
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
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
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
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
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
