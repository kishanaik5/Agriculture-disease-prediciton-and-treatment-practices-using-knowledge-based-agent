from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Gemini JSON Structure (Proposed by User) ---

class PlantInfo(BaseModel):
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None

class DiseaseInfo(BaseModel):
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    pathogen_type: Optional[str] = None
    cause: Optional[str] = None
    symptoms: Optional[str] = None
    disease_spread: Optional[str] = None
    severity: Optional[str] = None

class Management(BaseModel):
    organic_practices: List[str] = []
    chemical_practices: List[str] = []

class AnalysisResult(BaseModel):
    plant_info: PlantInfo
    disease_info: DiseaseInfo
    management: Management

# --- API Response ---

class CropItem(BaseModel):
    Name: str
    image: str
    category: str

class ScanResponse(BaseModel):
    id: str # Serial ID / UUID
    user_id: str
    created_at: datetime
    

    disease_detected: Optional[str] = None
    
    # URLs
    original_image_url: Optional[str] = None
    bbox_image_url: Optional[str] = None
    report_url: Optional[str] = None
    
    # inputs which might be echoed back
    user_input_crop: Optional[str] = None
    language: Optional[str] = "en"
    
    # Full Analysis
    kb_treatment: Optional[str] = Field(None, description="Verified treatment from local Knowledge Base")
    analysis_raw: Optional[AnalysisResult] = None

    class Config:
        from_attributes = True
