from sqlalchemy import Column, String, Boolean, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from SharedBackend.managers.base import BaseSchema

class AnalysisReport(BaseSchema):
    __tablename__ = "analysis_reports"
    
    # BaseSchema provides: uid (str), created_at, updated_at, deleted_at
    
    user_id = Column(String(50), nullable=False, index=True) # users_xxxxxxxx
    
    # Inputs
    user_input_crop = Column(String(255), nullable=True)
    language = Column(String(10), default="en") # en or kn
    is_mixed_cropping = Column(Boolean, nullable=True)
    acres_of_land = Column(String(50), nullable=True)
    
    # Detection Results
    detected_crop = Column(String(255), nullable=True)
    detected_disease = Column(String(255), nullable=True)
    pathogen_type = Column(String(100), nullable=True)
    severity = Column(Text, nullable=True)
    treatment = Column(Text, nullable=True) # KB Treatment
    
    # Data
    analysis_raw = Column(JSONB, nullable=True)
    
    # URLs
    original_image_url = Column(Text, nullable=True)
    bbox_image_url = Column(Text, nullable=True)
    report_url = Column(Text, nullable=True)
