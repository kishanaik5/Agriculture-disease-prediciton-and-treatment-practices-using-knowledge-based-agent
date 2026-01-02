from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.scan import AnalysisReport, FruitAnalysis, VegetableAnalysis
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
from typing import Optional
from app.services.knowledge import knowledge_service
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

class BytesUploadFile:
    def __init__(self, data: bytes, filename: str, content_type: str = "image/jpeg"):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        return self._data

router = APIRouter()

@router.get("/crops", response_model=list[CropItem])
async def get_crops(language: str = "en"):
    """
    Get list of supported crops (field crops only) with icons.
    """
    return knowledge_service.get_crops_with_icons(language=language, category_filter="crop")

@router.get("/fruits", response_model=list[CropItem])
async def get_fruits(language: str = "en"):
    """
    Get list of supported fruits.
    """
    return knowledge_service.get_crops_with_icons(language=language, category_filter="fruit")

@router.get("/vegetables", response_model=list[CropItem])
async def get_vegetables(language: str = "en"):
    """
    Get list of supported vegetables.
    """
    return knowledge_service.get_crops_with_icons(language=language, category_filter="vegetable")



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
        kb_text = knowledge_service.get_treatment(crop_name, disease_name, scientific_name=dis_sci_name, language=language)
    
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
        db, user_id, crop_name, language, is_mixed_cropping, acres_of_land,
        original_url, processed_url, disease_name, kb_text, analysis_dict, is_healthy, plant_info, disease_info
    )

@router.post("/vegetable_scan", response_model=ScanResponse)
async def analyze_vegetable(
    file: UploadFile = File(...),
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    is_mixed_cropping: bool = Form(False),
    acres_of_land: Optional[str] = Form(None),
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
         kb_text = knowledge_service.get_treatment(crop_name, disease_name, scientific_name=dis_sci_name, language=language)

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


@router.post("/fruit_scan", response_model=ScanResponse)
async def analyze_fruit(
    file: UploadFile = File(...),
    user_id: str = Form(..., description="User ID string (e.g. users_xxx)"),
    crop_name: str = Form(...),
    language: str = Form("en", description="Language code: en or kn"),
    is_mixed_cropping: bool = Form(False),
    acres_of_land: Optional[str] = Form(None),
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
         kb_text = knowledge_service.get_treatment(crop_name, disease_name, scientific_name=sci_name, language=language)
    
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

@router.post("/{category}_scan", response_model=ScanResponse)
async def analyze_dynamic(
    category: str,
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
            is_mixed_cropping=is_mixed_cropping,
            acres_of_land=acres_of_land,
            db=db
        )
        
    elif cat in ["fruit", "fruits"]:
        # analyze_fruit does not take background_tasks
        return await analyze_fruit(
            file=file,
            user_id=user_id,
            crop_name=crop_name,
            language=language,
            is_mixed_cropping=is_mixed_cropping,
            acres_of_land=acres_of_land,
            db=db
        )
        
    elif cat in ["vegetable", "vegetables"]:
        # analyze_vegetable does not take background_tasks
        return await analyze_vegetable(
            file=file,
            user_id=user_id,
            crop_name=crop_name,
            language=language,
            is_mixed_cropping=is_mixed_cropping,
            acres_of_land=acres_of_land,
            db=db
        )
    
    # Fallback for invalid category
    raise HTTPException(status_code=400, detail=f"Invalid category '{category}'. Must be 'crop', 'fruit', or 'vegetable'.")

async def _save_crop_and_respond(
    db: AsyncSession, user_id: str, crop_name: str, language: str, is_mixed_cropping: bool, acres_of_land: Optional[str],
    original_url: Optional[str], processed_url: Optional[str], disease_name: str, kb_text: str, analysis_raw: dict, is_healthy: bool,
    plant_info: dict, disease_info: dict
) -> ScanResponse:
    current_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    scan_id = "0"
    
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
            pathogen_scientific_name=diagnosis.get("pathogen_scientific_name"),
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
