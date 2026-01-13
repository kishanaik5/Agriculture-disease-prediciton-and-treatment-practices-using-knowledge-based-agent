from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.config import settings
from app.database import get_db
from app.models.scan import AnalysisReport, FruitAnalysis, VegetableAnalysis, TranslatedAnalysisReport
from app.services.gemini import gemini_service
import logging

router = APIRouter(
    prefix="/translation",
    tags=["Translation"]
)

logger = logging.getLogger(__name__)

class TranslateRequest(BaseModel):
    user_id: str
    report_id: str
    language: str

from sqlalchemy.ext.asyncio import AsyncSession

@router.post("/translate_report")
async def translate_report(request: TranslateRequest, db: AsyncSession = Depends(get_db)):
    """
    Translates an existing analysis report into the target language.
    If translation exists, returns it.
    If not, generates it using Gemini Flash 2.0 and stores it.
    """
    user_id = request.user_id
    report_id = request.report_id
    lang = request.language.lower()
    
    if lang == 'en':
        raise HTTPException(status_code=400, detail="Default language is English. Use GET /report for English.")

    # 1. Check if translation already exists
    stmt = select(TranslatedAnalysisReport).where(
        TranslatedAnalysisReport.report_uid == report_id,
        TranslatedAnalysisReport.language == lang
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    
    if existing:
        return {
            "id": existing.report_uid,
            "user_id": existing.user_id,
            "created_at": existing.created_at,
            "disease_detected": existing.disease_name,
            "original_image_url": existing.original_image_url,
            "bbox_image_url": existing.bbox_image_url,
            "user_input_crop": existing.item_name,
            "category": existing.category_type,
            "language": existing.language,
            "kb_treatment": existing.treatment,
            "analysis_raw": existing.analysis_raw
        }

    # 2. Find the original report in one of the tables
    # We try all 3 tables. In efficient design we'd know the category, but here we search.
    # Order of likelihood: Crop -> Vegetable -> Fruit or any.
    
    # Try Crop
    crop_stmt = select(AnalysisReport).where(AnalysisReport.uid == report_id)
    res = await db.execute(crop_stmt)
    original = res.scalar_one_or_none()
    category = 'crop'
    
    if not original:
        # Try Fruit
        fruit_stmt = select(FruitAnalysis).where(FruitAnalysis.uid == report_id)
        res = await db.execute(fruit_stmt)
        original = res.scalar_one_or_none()
        category = 'fruit'

    if not original:
        # Try Vegetable
        veg_stmt = select(VegetableAnalysis).where(VegetableAnalysis.uid == report_id)
        res = await db.execute(veg_stmt)
        original = res.scalar_one_or_none()
        category = 'vegetable'
        
    if not original:
        raise HTTPException(status_code=404, detail="Original report not found")

    # 3. Construct data for translation
    # Normalize fields across models
    data_to_translate = {
        "item_name": getattr(original, 'crop_name', None) or getattr(original, 'fruit_name', None) or getattr(original, 'vegetable_name', None),
        "disease_name": original.disease_name,
        "scientific_name": original.scientific_name,
        "severity": original.severity,
        "grade": getattr(original, 'grade', None),
        "treatment": original.treatment,
        "analysis_raw": original.analysis_raw
    }
    
    # 4. Perform Translation
    translated_data = await gemini_service.translate_report_content(data_to_translate, lang)
    
    if not translated_data:
        raise HTTPException(status_code=500, detail="Translation generation failed")

    # 5. Save to DB
    new_report = TranslatedAnalysisReport(
        report_uid=original.uid,
        user_id=user_id,
        language=lang,
        category_type=category,
        item_name=translated_data.get("item_name"),
        disease_name=translated_data.get("disease_name"),
        scientific_name=translated_data.get("scientific_name"), # Should remain latin/english ideally
        severity=translated_data.get("severity"),
        grade=translated_data.get("grade"),
        treatment=translated_data.get("treatment"),
        analysis_raw=translated_data.get("analysis_raw"),
        original_image_url=original.original_image_url,
        bbox_image_url=original.bbox_image_url,
        order_id=original.order_id,
        payment_status=original.payment_status
    )
    
    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)
    
    # 6. Lookup KB Treatment for Translated Report
    from app.services.knowledge import knowledge_service
    
    # We use the TRANSLATED disease name and item name to find localized treatment if possible
    # But usually KB is keyed by English or has separate CSVs.
    # knowledge_service.get_treatment handles language='kn' by looking up in kn csv.
    
    kb_treatment = await knowledge_service.get_treatment(
        crop=new_report.item_name, 
        disease=new_report.disease_name, 
        category=category, 
        db=db, 
        scientific_name=new_report.scientific_name, 
        language=lang
    )
    
    # Update treatment if found
    if kb_treatment:
        new_report.treatment = kb_treatment
        await db.commit()
        await db.refresh(new_report)

    return {
        "id": new_report.report_uid,
        "user_id": new_report.user_id,
        "created_at": new_report.created_at,
        "disease_detected": new_report.disease_name,
        "original_image_url": new_report.original_image_url,
        "bbox_image_url": new_report.bbox_image_url,
        "user_input_crop": new_report.item_name,
        "category": new_report.category_type,
        "language": new_report.language,
        "kb_treatment": new_report.treatment,
        "analysis_raw": new_report.analysis_raw
    }

@router.get("/report")
async def get_report(report_id: str, language: str = 'en', db: AsyncSession = Depends(get_db)):
    """
    Retrieves a report in the specified language.
    Default 'en' fetches from original tables.
    Other languages fetch from translated_analysis_report.
    """
    lang = language.lower()
    
    if lang == 'en':
        # Search original tables
        # Try Crop
        res = await db.execute(select(AnalysisReport).where(AnalysisReport.uid == report_id))
        report = res.scalar_one_or_none()
        report = res.scalar_one_or_none()
        if report: 
             return {
                "id": report.uid,
                "user_id": report.user_id,
                "created_at": report.created_at,
                "disease_detected": report.disease_name,
                "original_image_url": report.original_image_url,
                "bbox_image_url": report.bbox_image_url,
                "user_input_crop": report.crop_name,
                "category": "crop",
                "language": report.language,
                "kb_treatment": report.treatment,
                "analysis_raw": report.analysis_raw
             }
        
        # Try Fruit
        res = await db.execute(select(FruitAnalysis).where(FruitAnalysis.uid == report_id))
        report = res.scalar_one_or_none()
        if report: 
             return {
                "id": report.uid,
                "user_id": report.user_id,
                "created_at": report.created_at,
                "disease_detected": report.disease_name,
                "original_image_url": report.original_image_url,
                "bbox_image_url": report.bbox_image_url,
                "user_input_crop": report.fruit_name,
                "category": "fruit",
                "language": report.language,
                "kb_treatment": report.treatment,
                "analysis_raw": report.analysis_raw
             }

        # Try Vegetable
        res = await db.execute(select(VegetableAnalysis).where(VegetableAnalysis.uid == report_id))
        report = res.scalar_one_or_none()
        report = res.scalar_one_or_none()
        if report: 
             return {
                "id": report.uid,
                "user_id": report.user_id,
                "created_at": report.created_at,
                "disease_detected": report.disease_name,
                "original_image_url": report.original_image_url,
                "bbox_image_url": report.bbox_image_url,
                "user_input_crop": report.vegetable_name,
                "category": "vegetable",
                "language": report.language,
                "kb_treatment": report.treatment,
                "analysis_raw": report.analysis_raw
             }
        
        raise HTTPException(status_code=404, detail="Report not found")
    
    else:
        # Fetch translated
        stmt = select(TranslatedAnalysisReport).where(
            TranslatedAnalysisReport.report_uid == report_id,
            TranslatedAnalysisReport.language == lang
        )
        res = await db.execute(stmt)
        report = res.scalar_one_or_none()
        
        if not report:
            raise HTTPException(status_code=404, detail=f"Translation for language '{lang}' not found. Please generate it first using POST /translate_report.")
            
        return {
            "id": report.report_uid,
            "user_id": report.user_id,
            "created_at": report.created_at,
            "disease_detected": report.disease_name,
            "original_image_url": report.original_image_url,
            "bbox_image_url": report.bbox_image_url,
            "user_input_crop": report.item_name,
            "category": report.category_type,
            "language": report.language,
            "kb_treatment": report.treatment,
            "analysis_raw": report.analysis_raw
        }
