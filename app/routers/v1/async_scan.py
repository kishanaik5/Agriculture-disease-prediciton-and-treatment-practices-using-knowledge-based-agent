from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db, SessionLocal
from app.models.scan import AnalysisReport, FruitAnalysis, VegetableAnalysis
# REMOVED AsyncJob, Redis
from sqlalchemy import select
from app.schemas.scan import ScanResponse, AnalysisResult
from app.services.gemini import gemini_service
from app.services.s3 import s3_service_public, s3_service_private
from app.services.image import image_service
# from app.services.redis_manager import task_manager # Removed
import uuid
import logging
import io
import json
import os
from datetime import datetime
from typing import Optional
from app.services.knowledge import knowledge_service

logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE FLAG: Bounding Box Generation
# =============================================================================
# Set to True to enable bounding box generation and upload to S3
# Set to False to disable bounding box generation (bbox_image_url will be NULL)
# 
# 📍 PRODUCTION TOGGLE: Change this line to enable/disable bbox generation
ENABLE_BOUNDING_BOX_GENERATION = False  # TESTING: Disabled for verification
# =============================================================================

class BytesUploadFile:
    def __init__(self, data: bytes, filename: str, content_type: str = "image/jpeg"):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        return self._data

router = APIRouter()

async def process_crop_scan_async(
    task_id: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    user_id: str,
    crop_name: str,
    language: str,
    is_mixed_cropping: bool,
    acres_of_land: Optional[str]
):
    """Background task to process crop scan"""
    args = locals()
    
    db = SessionLocal()
    try:
        # await task_manager.set_task_status(task_id, "processing", {"progress": 10})
        # Helper to update AsyncJob
        async def _update_status(status, details):
            stmt = select(AsyncJob).where(AsyncJob.job_id == task_id)
            res = await db.execute(stmt)
            job = res.scalars().first()
            if job:
                job.status = status
                if details: 
                    job.details = details
                    if "progress" in details: job.progress = details["progress"]
                await db.commit()
                
        await _update_status("processing", {"progress": 10})
        
        # 1. Analyze Image
        category = knowledge_service.get_crop_category(crop_name)
        is_produce = category in ["Fruits", "Vegetable"]
        
        if is_produce:
            analysis_dict = await gemini_service.analyze_vegetable(image_bytes, crop_name, language=language, mime_type=content_type)
        else:
            analysis_dict = await gemini_service.analyze_crop(image_bytes, crop_name, language=language, mime_type=content_type)
            
        await _update_status("processing", {"progress": 30})
        
        # 2. Validation
        plant_info = analysis_dict.get("plant_info", {})
        sci_name = plant_info.get("scientific_name", "")
        comm_name = plant_info.get("common_name", "")

        if "MISMATCH" in sci_name or "INVALID_IMAGE" in comm_name or "INVALID_IMAGE" in sci_name:
            error_msg = "Please choose the correct crop and upload the image correctly."
            if language == 'kn':
                error_msg = "ದಯವಿಟ್ಟು ಸರಿಯಾದ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ಮತ್ತು ಚಿತ್ರವನ್ನು ಸರಿಯಾಗಿ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
            
            await _update_status("failed", {"error": error_msg})
            return

        await _update_status("processing", {"progress": 50})
        
        # 3. Upload Original
        file_wrapper = BytesUploadFile(image_bytes, filename, content_type)
        # CHANGED: Use Public Bucket for original images as per new requirement
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
        
        await _update_status("processing", {"progress": 60})
        
        # 4. Process disease if detected
        disease_info = analysis_dict.get("disease_info", {})
        disease_name = disease_info.get("disease_name", "Unknown")
        
        kb_text = ""
        if disease_name and disease_name.lower() != "unknown":
            kb_text = await knowledge_service.get_treatment(crop_name, disease_name, db=db, language=language)
        
        lower_name = str(disease_name).lower().strip()
        is_healthy = (
            lower_name in ["none", "healthy", "n/a", "", "no visible disease", "none detected", "ಆರೋಗ್ಯಕರ", "ಯಾವುದೇ ರೋಗವಿಲ್ಲ"] 
            or "no disease" in lower_name
            or "healthy" in lower_name
            or "none detected" in lower_name
        )
        
        processed_url = None
        
        # Bounding boxes for diseased crops (controlled by feature flag)
        if ENABLE_BOUNDING_BOX_GENERATION and not is_healthy and disease_name:
            await _update_status("processing", {"progress": 70})
            
            if is_produce:
                boxes = await gemini_service.generate_bbox_vegetable(image_bytes, disease_name, mime_type=content_type)
            else:
                boxes = await gemini_service.generate_bbox_crop(image_bytes, disease_name, mime_type=content_type)
            
            if boxes:
                logger.info(f"🎯 Bounding box generation ENABLED - Processing {len(boxes)} boxes")
                processed_image_bytes = image_service.draw_bounding_boxes(image_bytes, boxes)
                
                timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
                local_filename = f"{user_id}-{timestamp_str}.jpg"
                local_path = os.path.join("bb_images", local_filename)
                
                with open(local_path, "wb") as f:
                    f.write(processed_image_bytes)
                
                processed_wrapper = BytesUploadFile(processed_image_bytes, f"processed_{filename}", "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=processed_wrapper, prefix="dev/processed/")
                logger.info(f"✅ Bounding box image uploaded: {processed_url}")
        elif not ENABLE_BOUNDING_BOX_GENERATION:
            logger.info(f"⏭️  Bounding box generation DISABLED - Skipping bbox processing")
        
        await _update_status("processing", {"progress": 90})
        
        # 5. Save to DB
        current_time = datetime.now()
        
        response_data = {
            "user_id": user_id,
            "created_at": current_time,
            "original_image_url": original_url,
            "bbox_image_url": processed_url,
            "disease_detected": disease_name,
            "user_input_crop": crop_name,
            "language": language,
            "kb_treatment": kb_text,
            "analysis_raw": analysis_dict
        }

        if not is_healthy:
            # Determine correct table based on category from knowledge service
            cat_check = category.lower().rstrip('s') # fruit, vegetable, crop
            if cat_check == "fruits": cat_check = "fruit" # handle 'fruits' -> 'fruit' logic specifically if needed

            if cat_check == "fruit":
                 db_report = FruitAnalysis(
                    user_id=user_id,
                    fruit_name=crop_name, # Logic maps crop_name to fruit_name
                    language="en",
                    desired_language_output=language,
                    disease_name=disease_name,
                    scientific_name=disease_info.get("pathogen_scientific_name") or disease_info.get("scientific_name"),
                    severity=str(analysis_dict.get("diagnosis", {}).get("severity_stage") or analysis_dict.get("disease_info", {}).get("severity")), 
                    grade=analysis_dict.get("market_quality", {}).get("grade"),
                    treatment=kb_text,
                    analysis_raw=analysis_dict,
                    created_at=current_time,
                    original_image_url=original_url,
                    bbox_image_url=processed_url,
                    payment_status="PENDING"
                 )
            elif cat_check == "vegetable":
                 db_report = VegetableAnalysis(
                    user_id=user_id,
                    vegetable_name=crop_name,
                    language="en",
                    desired_language_output=language,
                    disease_name=disease_name,
                    scientific_name=disease_info.get("scientific_name"),
                    severity=str(analysis_dict.get("disease_info", {}).get("severity")),
                    treatment=kb_text,
                    analysis_raw=analysis_dict,
                    created_at=current_time,
                    original_image_url=original_url,
                    bbox_image_url=processed_url,
                    payment_status="PENDING"
                 )
            else:
                db_report = AnalysisReport(
                    user_id=user_id,
                    crop_name=crop_name, 
                    language="en",
                    desired_language_output=language,
                    disease_name=disease_name, 
                    scientific_name=disease_info.get("pathogen_type"),
                    severity=str(analysis_dict.get("disease_info", {}).get("severity")),
                    treatment=kb_text,
                    analysis_raw=analysis_dict,
                    created_at=current_time,
                    original_image_url=original_url,
                    bbox_image_url=processed_url,
                    payment_status="PENDING"
                )
            
            db.add(db_report)
            await db.commit()
            await db.refresh(db_report)

            # Translation Logic (Async)
            if language != "en":
                from app.services.translation_service import translation_service
                try:
                    # We are in sync context wrapper (run_in_threadpool potentially or native async). 
                    # create_task is safer if we want parallel, but here we want result for task status.
                    async_args = {
                        "db": db, 
                        "report_id": db_report.uid, 
                        "user_id": user_id, 
                        "target_language": language
                    }
                    translated_report = await translation_service.get_or_create_translation(**async_args)
                    
                    # Update response data with translation
                    response_data.update({
                        "disease_detected": translated_report.disease_name,
                        "user_input_crop": translated_report.item_name,
                        "language": translated_report.language,
                        "kb_treatment": translated_report.treatment,
                        "analysis_raw": translated_report.analysis_raw
                    })
                except Exception as e:
                    logger.error(f"Async Translation failed: {e}")
                    # Don't fail the whole task, just return English with warning?
                    # Or set error? User prefers completed.
                    pass
            
            response_data["id"] = db_report.uid
        else:
            response_data["id"] = "0"
        
        await _update_status("completed", {"progress": 100, "result": response_data})
        
    except Exception as e:
        logger.error(f"Background processing failed for task {task_id}: {e}")
        await _update_status("failed", {"error": str(e)})
    finally:
        await db.close()


@router.post("/crop_scan_async")
async def analyze_crop_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en, kn, hn, ta, te, ml, mr, gu, bn, or or"),
    is_mixed_cropping: bool = Form(False),
    acres_of_land: Optional[str] = Form(None)
):
    raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use /api/v1/report/generate_async instead.")
    # Legacy code removed to prevent dependency errors


@router.get("/crop_scan_status/{task_id}")
async def get_scan_status(task_id: str):
    raise HTTPException(status_code=410, detail="This endpoint is deprecated.")
