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
    # produce specific
    stage: Optional[str] = None
    affected_percentage: Optional[str] = None

class Marketability(BaseModel):
    export_grade: Optional[str] = None
    market_suitability: Optional[str] = None
    reasoning: Optional[str] = None

class ShelfLifePrediction(BaseModel):
    stability_score: Optional[float] = None
    days_to_rot_10C: Optional[float] = None
    days_to_rot_25C: Optional[float] = None
    shipping_recommendation: Optional[str] = None

class Management(BaseModel):
    organic_practices: List[str] = []
    chemical_practices: List[str] = []
    consumption_safety: Optional[str] = None

class AnalysisResult(BaseModel):
    # Flexible schema to support Crops, Fruits, and Vegetables scans
    plant_info: Optional[Dict[str, Any]] = None
    fruit_info: Optional[Dict[str, Any]] = None # Fruit specific
    
    # Diagnosis / Disease
    disease_info: Optional[Dict[str, Any]] = None
    diagnosis: Optional[Dict[str, Any]] = None # Fruit specific
    
    # Market & Shelf Life
    marketability: Optional[Dict[str, Any]] = None
    market_quality: Optional[Dict[str, Any]] = None # Fruit specific
    shelf_life_prediction: Optional[Dict[str, Any]] = None
    shelf_life: Optional[Dict[str, Any]] = None # Fruit specific
    
    # Management / Recommendations
    management: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None # Fruit specific
    
    model_config = {"extra": "allow", "exclude_none": True}

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
    
    # inputs which might be echoed back
    user_input_crop: Optional[str] = None
    language: Optional[str] = "en"
    
    # Full Analysis
    kb_treatment: Optional[str] = Field(None, description="Verified treatment from local Knowledge Base")
    analysis_raw: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True, "exclude_none": True}
