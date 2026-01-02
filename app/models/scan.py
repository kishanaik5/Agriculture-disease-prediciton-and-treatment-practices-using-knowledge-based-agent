from sqlalchemy import Column, String, Boolean, Text, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB
from SharedBackend.managers.base import BaseSchema
from app.config import init_settings

settings = init_settings()

class AnalysisReport(BaseSchema):
    __tablename__ = "crop_analysis_report"
    __table_args__ = {'schema': settings.DB_SCHEMA}
    
    user_id = Column(String(50), nullable=False, index=True) 
    user_input_crop = Column(String(255), nullable=True)
    language = Column(String(10), default="en") 
    is_mixed_cropping = Column(Boolean, nullable=True)
    acres_of_land = Column(String(50), nullable=True)
    
    detected_crop = Column(String(255), nullable=True)
    detected_disease = Column(String(255), nullable=True)
    pathogen_type = Column(String(100), nullable=True)
    severity = Column(Text, nullable=True)
    treatment = Column(Text, nullable=True) 
    
    analysis_raw = Column(JSONB, nullable=True)
    
    original_image_url = Column(Text, nullable=True)
    bbox_image_url = Column(Text, nullable=True)


class FruitAnalysis(BaseSchema):
    __tablename__ = "fruit_analysis_report"
    __table_args__ = {'schema': settings.DB_SCHEMA}

    user_id = Column(String(50), nullable=False, index=True)
    fruit_name = Column(String(255), nullable=True)
    language = Column(String(10), default="en")
    
    # Diagnosis
    disease_name = Column(String(255), nullable=True)
    pathogen_scientific_name = Column(String(255), nullable=True)
    severity = Column(Text, nullable=True)
    grade = Column(String(100), nullable=True)
    treatment = Column(Text, nullable=True)
    
    # Raw JSON
    analysis_raw = Column(JSONB, nullable=True)
    
    # Media
    original_image_url = Column(Text, nullable=True)
    bbox_image_url = Column(Text, nullable=True)


class VegetableAnalysis(BaseSchema):
    __tablename__ = "vegetable_analysis_report"
    __table_args__ = {'schema': settings.DB_SCHEMA}

    user_id = Column(String(50), nullable=False, index=True)
    vegetable_name = Column(String(255), nullable=True)
    language = Column(String(10), default="en")
    
    # Diagnosis
    disease_name = Column(String(255), nullable=True)
    scientific_name = Column(String(255), nullable=True)
    severity = Column(Text, nullable=True)
    grade = Column(String(100), nullable=True)
    treatment = Column(Text, nullable=True)

    analysis_raw = Column(JSONB, nullable=True)
    
    original_image_url = Column(Text, nullable=True)
    bbox_image_url = Column(Text, nullable=True)
