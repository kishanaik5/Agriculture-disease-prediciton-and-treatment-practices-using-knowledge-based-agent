from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.scan import AnalysisReport
from app.schemas.scan import ScanResponse, AnalysisResult
from app.services.gemini import gemini_service
from app.services.s3 import s3_service_public, s3_service_private
from app.services.image import image_service
from app.services.redis_manager import task_manager
import uuid
import logging
import io
import json
import os
from datetime import datetime
from typing import Optional
from app.services.knowledge import knowledge_service

logger = logging.getLogger(__name__)

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
    acres_of_land: Optional[str],
    db: AsyncSession
):
    """Background task to process crop scan"""
    try:
        await task_manager.set_task_status(task_id, "processing", {"progress": 10})
        
        # 1. Analyze Image
        analysis_dict = await gemini_service.analyze_image(image_bytes, crop_name, language=language, mime_type=content_type)
        await task_manager.set_task_status(task_id, "processing", {"progress": 30})
        
        # 2. Validation
        plant_info = analysis_dict.get("plant_info", {})
        sci_name = plant_info.get("scientific_name", "")
        comm_name = plant_info.get("common_name", "")

        if "MISMATCH" in sci_name or "INVALID_IMAGE" in comm_name or "INVALID_IMAGE" in sci_name:
            error_msg = "Please choose the correct crop and upload the image correctly."
            if language == 'kn':
                error_msg = "ದಯವಿಟ್ಟು ಸರಿಯಾದ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ ಮತ್ತು ಚಿತ್ರವನ್ನು ಸರಿಯಾಗಿ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
            
            await task_manager.set_task_status(task_id, "failed", {"error": error_msg})
            return

        await task_manager.set_task_status(task_id, "processing", {"progress": 50})
        
        # 3. Upload Original
        file_wrapper = BytesUploadFile(image_bytes, filename, content_type)
        # CHANGED: Use Public Bucket for original images as per new requirement
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
        
        await task_manager.set_task_status(task_id, "processing", {"progress": 60})
        
        # 4. Process disease if detected
        disease_info = analysis_dict.get("disease_info", {})
        disease_name = disease_info.get("common_name", "Unknown")
        
        kb_text = ""
        if disease_name and disease_name.lower() != "unknown":
            kb_text = knowledge_service.get_treatment(crop_name, disease_name, language=language)
        
        lower_name = str(disease_name).lower().strip()
        is_healthy = (
            lower_name in ["none", "healthy", "n/a", "", "no visible disease", "none detected", "ಆರೋಗ್ಯಕರ", "ಯಾವುದೇ ರೋಗವಿಲ್ಲ"] 
            or "no disease" in lower_name
            or "healthy" in lower_name
            or "none detected" in lower_name
        )
        
        processed_url = None
        
        # Bounding boxes for diseased crops
        if not is_healthy and disease_name:
            await task_manager.set_task_status(task_id, "processing", {"progress": 70})
            
            boxes = await gemini_service.generate_bounding_boxes(image_bytes, disease_name, mime_type=content_type)
            
            if boxes:
                processed_image_bytes = image_service.draw_bounding_boxes(image_bytes, boxes)
                
                timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
                local_filename = f"{user_id}-{timestamp_str}.jpg"
                local_path = os.path.join("bb_images", local_filename)
                
                with open(local_path, "wb") as f:
                    f.write(processed_image_bytes)
                
                processed_wrapper = BytesUploadFile(processed_image_bytes, f"processed_{filename}", "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=processed_wrapper, prefix="dev/processed/")
        
        await task_manager.set_task_status(task_id, "processing", {"progress": 90})
        
        # 5. Save to DB
        current_time = datetime.now()
        
        response_data = {
            "user_id": user_id,
            "created_at": current_time,
            "original_image_url": original_url,
            "bbox_image_url": processed_url,
            "report_url": None,
            "disease_detected": disease_name,
            "user_input_crop": crop_name,
            "language": language,
            "kb_treatment": kb_text,
            "analysis_raw": analysis_dict
        }

        if not is_healthy:
            db_report = AnalysisReport(
                user_id=user_id,
                user_input_crop=crop_name,
                language=language,
                is_mixed_cropping=is_mixed_cropping,
                acres_of_land=acres_of_land,
                detected_crop=plant_info.get("common_name"),
                detected_disease=disease_name,
                pathogen_type=disease_info.get("pathogen_type"),
                severity=str(analysis_dict.get("disease_info", {}).get("severity")),
                treatment=kb_text,
                analysis_raw=analysis_dict,
                created_at=current_time,
                original_image_url=original_url,
                bbox_image_url=processed_url,
                report_url=None
            )
            
            db.add(db_report)
            await db.commit()
            await db.refresh(db_report)
            
            response_data["id"] = db_report.uid
        else:
            response_data["id"] = "0"
        
        await task_manager.set_task_status(task_id, "completed", {"progress": 100, "result": response_data})
        
    except Exception as e:
        logger.error(f"Background processing failed for task {task_id}: {e}")
        await task_manager.set_task_status(task_id, "failed", {"error": str(e)})


@router.post("/crop_scan_async")
async def analyze_crop_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    is_mixed_cropping: bool = Form(False),
    acres_of_land: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Async version of crop_scan endpoint.
    Returns immediately with a task_id.
    Client polls /crop_scan_status/{task_id} for results.
    
    No timeout issues - processes in background.
    """
    task_id = str(uuid.uuid4())
    
    # Read image immediately
    image_bytes = await file.read()
    
    # Initialize task in Redis
    await task_manager.set_task_status(task_id, "queued", {"progress": 0})
    
    # Start background task
    background_tasks.add_task(
        process_crop_scan_async,
        task_id=task_id,
        image_bytes=image_bytes,
        filename=file.filename,
        content_type=file.content_type,
        user_id=user_id,
        crop_name=crop_name,
        language=language,
        is_mixed_cropping=is_mixed_cropping,
        acres_of_land=acres_of_land,
        db=db
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Processing started. Poll /api/v1/crop_scan_status/{task_id} for results.",
        "poll_interval_seconds": 3
    }


@router.get("/crop_scan_status/{task_id}")
async def get_scan_status(task_id: str):
    """
    Check the status of an async crop scan task.
    
    Response status values:
    - queued: Task is waiting to start
    - processing: Task is being processed (check progress field)
    - completed: Task finished successfully (check result field)
    - failed: Task failed (check error field)
    """
    task_data = await task_manager.get_task_status(task_id)
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    
    return task_data
