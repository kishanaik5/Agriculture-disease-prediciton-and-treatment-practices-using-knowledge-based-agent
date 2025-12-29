from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.scan import AnalysisReport
from app.schemas.scan import ScanResponse, AnalysisResult
from app.services.gemini import gemini_service
from app.services.s3 import s3_service_public, s3_service_private
from app.services.image import image_service
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

@router.get("/crops", response_model=list[str])
async def get_crops(language: str = "en"):
    """
    Get list of supported crops from Knowledge Base.
    """
    return knowledge_service.get_unique_crops(language=language)



@router.post("/crop_scan", response_model=ScanResponse)
async def analyze_crop(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # New Inputs
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    is_mixed_cropping: bool = Form(False),
    acres_of_land: Optional[str] = Form(None),
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
    
    # 2. Analyze FIRST to validate crop (as per user request)
    try:
        analysis_dict = await gemini_service.analyze_image(image_bytes, crop_name, language=language, mime_type=file.content_type)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI Service failed to analyze image: {str(e)}")

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

    # 4. Upload Original (Only if valid)
    try:
        # Wrap bytes for the new S3Service which expects an UploadFile-like object
        file_wrapper = BytesUploadFile(image_bytes, file.filename, file.content_type)
        # CHANGED: Use Public Bucket for original images as per new requirement
        # Path: dev/original/{filename}
        original_url = await s3_service_public.upload_file(file=file_wrapper, prefix="dev/original/")
    except Exception as e:
        logger.error(f"S3 Upload failed but analysis valid. Proceeding. Error: {e}")
        original_url = None



    # 4. Process Logic
    disease_info = analysis_dict.get("disease_info", {})
    disease_name = disease_info.get("common_name", "Unknown")
    
    # KB Lookup
    kb_text = ""
    if disease_name and disease_name.lower() != "unknown":
        kb_text = knowledge_service.get_treatment(crop_name, disease_name, language=language)
    
    # Healthy Heuristic
    lower_name = str(disease_name).lower().strip()
    is_healthy = (
        lower_name in ["none", "healthy", "n/a", "", "no visible disease", "none detected", "ಆರೋಗ್ಯಕರ", "ಯಾವುದೇ ರೋಗವಿಲ್ಲ"] 
        or "no disease" in lower_name
        or "healthy" in lower_name
        or "none detected" in lower_name
    )
    
    processed_url = None
    report_url = None # Placeholder for PDF report
    
    # Bounding Box Logic (Only if NOT healthy)
    if not is_healthy and disease_name:
        try:
            logger.info(f"🎯 Generating bounding boxes for disease: {disease_name}")
            boxes = await gemini_service.generate_bounding_boxes(image_bytes, disease_name, mime_type=file.content_type)
            
            if boxes:
                logger.info(f"✅ Generated {len(boxes)} bounding boxes")
                # Draw boxes
                processed_image_bytes = image_service.draw_bounding_boxes(image_bytes, boxes)
                
                # Save LOCALLY (using user_id + formatted timestamp as requested roughly)
                # "users_xxx-timestamp.jpg"
                timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
                local_filename = f"{user_id}-{timestamp_str}.jpg"
                local_path = os.path.join("bb_images", local_filename)
                
                with open(local_path, "wb") as f:
                    f.write(processed_image_bytes)
                
                logger.info(f"💾 Saved processed image to {local_path}")
                
                # Upload to S3 Public (using dev/processed/ prefix)
                processed_wrapper = BytesUploadFile(processed_image_bytes, f"processed_{file.filename}", "image/jpeg")
                processed_url = await s3_service_public.upload_file(file=processed_wrapper, prefix="dev/processed/")
                logger.info(f"☁️ Uploaded bbox image to S3: {processed_url}")
            else:
                logger.warning(f"⚠️ No bounding boxes generated for {disease_name}. Model returned empty list.")
                
        except Exception as e:
            logger.error(f"❌ Bounding Box stage failed: {e}", exc_info=True)
            # Non-blocking error - continue without bbox

    # 5. Save to DB (Only if NOT healthy)
    current_time = datetime.now()
    
    plant_info = analysis_dict.get("plant_info", {})
    
    # Prepare common response data
    response_data = {
        "user_id": user_id,
        "created_at": current_time,
        "original_image_url": original_url,
        "bbox_image_url": processed_url,
        "report_url": report_url,
        "disease_detected": disease_name,
        "user_input_crop": crop_name,
        "language": language,
        "kb_treatment": kb_text,
        "analysis_raw": analysis_dict
    }

    if not is_healthy:
        # Create DB Entry
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
            report_url=report_url
        )
        
        db.add(db_report)
        await db.commit()
        await db.refresh(db_report)
        
        # Use the real serial ID (UID)
        response_data["id"] = db_report.uid
    else:
        # If healthy, skip DB.
        # Assign a temporary ID (0 or -1) to satisfy schema since it won't be in DB.
        response_data["id"] = "0" 

    return ScanResponse(**response_data)
