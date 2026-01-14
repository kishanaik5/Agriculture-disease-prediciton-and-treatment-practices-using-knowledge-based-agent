from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from app.models.scan import AnalysisReport, FruitAnalysis, VegetableAnalysis, TranslatedAnalysisReport
from app.services.gemini import gemini_service
from app.services.knowledge import knowledge_service
import logging

logger = logging.getLogger(__name__)

class TranslationService:
    async def get_or_create_translation(self, db: AsyncSession, report_id: str, user_id: str, target_lang: str):
        """
        Ensures a translation exists for the given report_id in target_lang.
        Returns the TranslatedAnalysisReport object.
        """
        lang = target_lang.lower()
        
        # 1. Check if translation already exists
        stmt = select(TranslatedAnalysisReport).where(
            TranslatedAnalysisReport.report_uid == report_id,
            TranslatedAnalysisReport.language == lang
        )
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            return existing

        # 2. Find the original report
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
            raise HTTPException(status_code=404, detail="Original report not found for translation")

        # 3. Construct data for translation
        data_to_translate = {
            "item_name": getattr(original, 'crop_name', None) or getattr(original, 'fruit_name', None) or getattr(original, 'vegetable_name', None),
            "disease_name": original.disease_name,
            "scientific_name": getattr(original, 'scientific_name', None),
            "severity": getattr(original, 'severity', None),
            "grade": getattr(original, 'grade', None),
            "treatment": getattr(original, 'treatment', None),
            "analysis_raw": original.analysis_raw
        }
        
        # 4. Perform Translation (Using Gemini Service)
        try:
            translated_data = await gemini_service.translate_report_content(data_to_translate, lang)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
        
        if not translated_data:
            raise HTTPException(status_code=500, detail="Translation generation returned empty")

        # 5. Save to DB
        new_report = TranslatedAnalysisReport(
            report_uid=original.uid,
            user_id=user_id,
            language=lang,
            category_type=category,
            item_name=translated_data.get("item_name"),
            disease_name=translated_data.get("disease_name"),
            scientific_name=translated_data.get("scientific_name"),
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
        
        # 6. Lookup KB Treatment (Post-Translation Enhancement)
        # Using the translated terms to find localized KB entry
        kb_treatment = await knowledge_service.get_treatment(
            crop=new_report.item_name, 
            disease=new_report.disease_name, 
            category=category, 
            db=db, 
            scientific_name=new_report.scientific_name, 
            language=lang
        )
        
        if kb_treatment:
            new_report.treatment = kb_treatment
            await db.commit()
            await db.refresh(new_report)
            
        return new_report
        
    async def get_translated_report(self, db: AsyncSession, report_id: str, language: str):
        stmt = select(TranslatedAnalysisReport).where(
            TranslatedAnalysisReport.report_uid == report_id,
            TranslatedAnalysisReport.language == language
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

translation_service = TranslationService()
