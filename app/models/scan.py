from sqlalchemy import Column, String, Boolean, Text, Integer, Float, DateTime
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
    
    detected_crop = Column(String(255), nullable=True)
    detected_disease = Column(String(255), nullable=True)
    pathogen_type = Column(String(100), nullable=True)
    severity = Column(Text, nullable=True)
    treatment = Column(Text, nullable=True) 
    
    analysis_raw = Column(JSONB, nullable=True)
    
    original_image_url = Column(Text, nullable=True)
    bbox_image_url = Column(Text, nullable=True)

    # Payment Info
    order_id = Column(String(100), nullable=True, index=True)
    payment_status = Column(String(20), default="PENDING", nullable=True)


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

    # Payment Info
    order_id = Column(String(100), nullable=True, index=True)
    payment_status = Column(String(20), default="PENDING", nullable=True)


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

    # Payment Info
    order_id = Column(String(100), nullable=True, index=True)
    payment_status = Column(String(20), default="PENDING", nullable=True)


class PaymentTransaction(BaseSchema):
    __tablename__ = "transaction_table"
    __table_args__ = {'schema': settings.DB_SCHEMA}

    user_id = Column(String(50), nullable=False, index=True)
    order_id = Column(String(100), unique=True, nullable=False, index=True)
    payment_status = Column(String(20), default="PENDING") # PENDING, SUCCESS
    amount = Column(Float, nullable=False, default=1.0)
    # currency = Column(String(3), default="INR")
    
    # Context
    analysis_type = Column(String(20)) # fruit, vegetable, crop
    analysis_report_uid = Column(String, nullable=True, index=True) # Logical Foreign Key to respective report table
    
    transaction_id = Column(String, nullable=True) # Stores cf_order_id
    
    payment_success_at = Column(DateTime(timezone=True), nullable=True)

