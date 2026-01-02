from google import genai
from google.genai import types
from app.config import settings
import json
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        # Initialize the new GenAI Client
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Models configuration
        self.flash_model = settings.GEMINI_MODEL_FLASH
        self.pro_model = settings.GEMINI_MODEL_PRO
        self.bbox_model_name = settings.GEMINI_MODEL_BBOX
        
        # System instruction for BBox model
        self.bbox_instruction = "You are a precise scientific annotator. You never group objects; you identify every individual instance separately using [ymin, xmin, ymax, xmax]."

    async def analyze_crop(self, image_data: bytes, crop_name: str, language: str = "en", mime_type: str = "image/jpeg") -> dict:
        """
        Analyzes the crop image using Gemini Flash with 5 retries, falling back to Pro on failure.
        """
        
        lang_target = "KANNADA" if language.lower() in ['kn', 'kannada'] else "ENGLISH"

        prompt = f"""
        Role: You are an expert Plant Pathologist and Agricultural Advisory System.

        Task:
        Analyze the attached image of a plant leaf to perform a comprehensive diagnosis and advisory generation.
        
        [LANGUAGE INSTRUCTION]: 
        - JSON KEYS must remain exactly as defined (English)
        - JSON VALUES should be in {lang_target}
        - Scientific names should remain in Latin. For physiological disorders, use the English technical term (e.g., "Calcium Deficiency").

        [CONTEXT]:
        The user claims this is a "{crop_name}" plant.
        
        Primary Objectives:
        
        [VALIDATION CHECK]:
        1. Identify the object in the image.
        2. Compare it with the user's claim: "{crop_name}".
        3. If the image is NOT "{crop_name}" (or closely related), set "is_valid_crop" to false and "detected_object_name" to what you see.
        4. If it IS valid, set "is_valid_crop" to true.

        1. Identify the Plant accurately.
           - CRITICAL CHECK: Does the image match the user's claim of "{crop_name}"? 
           - If the image is CLEARLY a different plant, flag this in the "scientific_name" field as "MISMATCH: Detected [Actual Plant] vs User Claim [User Plant]".
           - If it is non-plant material, return "INVALID_IMAGE" in the common_name.
        2. Identify the Disease. If no disease is visible, return "Healthy".
        3. Explain the CAUSE of the disease (how and why it occurs).
        4. Explain disease spread (seed, soil, wind, rain, insects).
        5. Provide integrated disease management:
        - Organic & biological practices
        - Chemical practices (active ingredients only)


        Output Rules:
        - STRICT JSON ONLY
        - No markdown
        - No extra text

        JSON FORMAT:
        {{
            "is_valid_crop": true,
            "detected_object_name": "",
            "plant_info": {{
                "common_name": "",
                "scientific_name": ""
            }},
            "disease_info": {{
                "common_name": "",
                "scientific_name": "",
                "pathogen_type": "",
                "cause": "",
                "symptoms": "",
                "disease_spread": "",
                "severity": ""
            }},
            "management": {{
                "organic_practices": [],
                "chemical_practices": []
            }}
        }}
        """

        # Retry logic for Flash: 5 attempts
        max_retries = 5
        
        # Prepare content part (image)
        content_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        
        # Config for JSON response
        json_config = types.GenerateContentConfig(response_mime_type="application/json")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} with Gemini Flash ({self.flash_model})...")
                
                flash_response = await self.client.aio.models.generate_content(
                    model=self.flash_model,
                    contents=[prompt, content_part],
                    config=json_config
                )

                if flash_response.text:
                    try:
                        return json.loads(flash_response.text)
                    except json.JSONDecodeError:
                        logger.warning("Flash returned invalid JSON, retrying...")
                        continue
            except Exception as e:
                logger.warning(f"Flash attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1)) # Simple backoff
        
        # Fallback to Pro
        logger.warning(f"All Flash attempts failed. Switching to Gemini Pro ({self.pro_model})...")
        try:
            pro_response = await self.client.aio.models.generate_content(
                model=self.pro_model,
                contents=[prompt, content_part],
                config=json_config
            )
            return json.loads(pro_response.text)
        except Exception as e2:
            logger.error(f"Gemini Pro also failed: {e2}")
            raise e2



    async def analyze_vegetable(self, image_data: bytes, crop_name: str, language: str = "en", mime_type: str = "image/jpeg") -> dict:
        """
        Specialized analysis for Produce (Vegetables/Fruits).
        """
        lang_target = "KANNADA" if language.lower() in ['kn', 'kannada'] else "ENGLISH"

        prompt = f"""
        Role: You are an expert Plant Pathologist, Post-Harvest Food Safety Specialist, and Agricultural Quality Grader.

        Task:
        Analyze the attached image. Determine if it shows a **Growing Plant** (leaf, stem, vine) or a **Harvested Vegetable/Fruit** (fruit, root, tuber) and perform a comprehensive diagnosis and commercial assessment.
        
        [LANGUAGE INSTRUCTION]: 
        - JSON KEYS must remain exactly as defined (English)
        - JSON VALUES should be in {lang_target}
        - Scientific names should remain in Latin. For physiological disorders, use the English technical term (e.g., "Calcium Deficiency").

        [CONTEXT]:
        User claims this is: {crop_name}

        Primary Objectives:
        
        [VALIDATION CHECK]:
        1. Identify the object (Growing Plant vs Harvested).
        2. Compare with user's claim: "{crop_name}".
        3. If mismatch (completely different crop), set "is_valid_crop": false.

        1. Identify the Plant/Vegetable Name.
        2. Identify the Condition.
           - If Growing Plant: Identify the disease, pest, or deficiency.
           - If Harvested Vegetable: Identify the rot, mold, physical defect, or physiological disorder.
           - If healthy, return "Healthy".
        3. Disease Metrics:
           - Estimate the **Stage** of the disease (Early, Mid, Late).
           - Estimate the **Affected Percentage** of the visible surface area.
        4. Explain the CAUSE (Pathogen/Pest for plants; Storage/Physiological for produce).
        5. Explain SPREAD risk (Field transmission for plants; Bin/Storage cross-contamination for produce).
        6. Harvest "Marketability" Assessment (Crucial for Harvested Crops):
           - Assign an **export_grade** based on aesthetic standards (e.g., "Class I (Retail)", "Class II (Processing/Juice)", "Reject/Cull").
           - Determine financial suitability (e.g., "Suitable for Export," "Local Market Only," "Processing Only").
        7. Post-Harvest "Shelf-Life" Predictor (Crucial for Harvested Crops):
           - Assign a **stability_score** (0-100) based on tissue integrity and spore activity.
           - Estimate **Days_to_Rot** at specific storage temperatures ($10^\\circ C$ vs $25^\\circ C$).
        8. Provide Context-Aware Management:
           - For Growing Plants: Provide field treatment (Organic & Chemical).
           - For Harvested Vegetables: Provide storage advice and consumption safety.

        Output Rules:
        - STRICT JSON ONLY.
        - No markdown code blocks (```json).
        - No extra text or conversational filler.
        - If the image is a Growing Plant, set marketability and shelf_life fields to null.

        JSON FORMAT:
        {{
          "is_valid_crop": true,
          "detected_object_name": "",
          "plant_info": {{
            "common_name": ""
          }},
          "disease_info": {{
            "common_name": "",
            "scientific_name": "",
            "pathogen_type": "",
            "cause": "",
            "symptoms": "",
            "disease_spread": "",
            "stage": "", 
            "affected_percentage": "",
            "severity": ""
          }},
          "marketability": {{
            "export_grade": "",
            "market_suitability": "",
            "reasoning": ""
          }},
          "shelf_life_prediction": {{
            "stability_score": 0,
            "days_to_rot_10C": 0,
            "days_to_rot_25C": 0,
            "shipping_recommendation": ""
          }},
          "management": {{
            "organic_practices": [],
            "chemical_practices": [],
            "consumption_safety": ""
          }}
        }} 
        """
        return await self._generate_response(prompt, image_data, mime_type)
    



    async def analyze_fruit(self, image_data: bytes, crop_name: str, language: str = "en", mime_type: str = "image/jpeg") -> dict:
        """
        Specialized analysis for Fruits.
        """
        lang_target = "KANNADA" if language.lower() in ['kn', 'kannada'] else "ENGLISH"

        prompt = f"""
        Role: You are an expert Pomologist (Fruit Specialist) and Post-Harvest Food Safety Specialist.
        Task:

        Analyze the attached image of a Fruit and perform a comprehensive diagnosis covering health status, disease condition, and commercial quality.
        
        [LANGUAGE INSTRUCTION]: 
        - JSON KEYS must remain exactly as defined (English)
        - JSON VALUES should be in {lang_target}
        - Scientific names (pathogen_scientific_name) should remain in Latin. For physiological disorders, use the English technical term (e.g., "Calcium Deficiency" or "Blossom End Rot").

        [CONTEXT]:
        User claims this is: {crop_name}

        Primary Objectives:
        
        [VALIDATION CHECK]:
        1. Identify object.
        2. Compare with user's claim: "{crop_name}".
        3. If mismatch, set "is_valid_crop": false.

        Primary Objectives:

        Identify the Fruit Name.
        Identify the Condition:
        Disease (Fungal, Bacterial, Viral),
        Pest damage,
        Physiological disorder (e.g., sunscald, cracking),
        Post-harvest rot.
        If no abnormality is detected, return "Healthy".
        
        Disease Metrics:
        Estimate the Stage (Early, Mid, Late/Rotting).
        Estimate the Affected Percentage of the visible surface area.
        Explain the Cause, identifying the pathogen (Latin scientific name) or environmental/storage factor.
        
        Marketability Assessment:
        Assign an export_grade (e.g., "Extra Class", "Class I", "Class II", "Reject/Cull").
        Determine Market Suitability (e.g., "Export Quality", "Local Market", "Juice/Processing", "Discard").
        
        Shelf-Life Prediction:
        Assign a stability_score (0–100) based on firmness, tissue breakdown, and rot progression.
        Estimate Days_to_Rot at:
        Room temperature (25°C),
        Cold storage (10°C).
        
        Management & Safety:
        Recommend Organic and Chemical Management Practices (fungicides or storage adjustments).
        Assess Edibility (e.g., "Safe", "Trim and Eat", "Toxic/Discard").
        
        Output Rules:

        STRICT JSON ONLY
        No markdown code blocks
        No conversational or explanatory text
        JSON FORMAT:
        {{
        "is_valid_crop": true,
        "detected_object_name": "",
        "fruit_info": {{
        
        "name": ""
        
        }},
        
        "diagnosis": {{
        
        "disease_name": "",
        
        "pathogen_scientific_name": "",
        
        "type": "",
        
        "symptoms_description": "",
        
        "severity_stage": "",
        
        "affected_area_percentage": ""
        
        }},
        
        "market_quality": {{
        
        "grade": "",
        
        "suitability": "",
        
        "reasoning": ""
        
        }},
        
        "shelf_life": {{
        
        "stability_score_0_to_100": 0,
        
        "days_remaining_room_temp": 0,
        
        "days_remaining_cold_storage": 0
        
        }},
        
        "management": {{
        
        "organic_practices": [],
        
        "chemical_practices": [],
        
        "consumption_safety": ""
        
        }}
        
        }}
        """
        return await self._generate_response(prompt, image_data, mime_type)

    async def _generate_response(self, prompt: str, image_data: bytes, mime_type: str) -> dict:
        """Helper to run Gemini generation with retries"""
        max_retries = 3
        content_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        json_config = types.GenerateContentConfig(response_mime_type="application/json")
        
        # Try Flash
        for attempt in range(max_retries):
            try:
                flash_response = await self.client.aio.models.generate_content(
                    model=self.flash_model,
                    contents=[prompt, content_part],
                    config=json_config
                )
                if flash_response.text:
                    try:
                        return json.loads(flash_response.text)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.warning(f"Flash attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        # Fallback to Pro
        try:
            pro_response = await self.client.aio.models.generate_content(
                model=self.pro_model,
                contents=[prompt, content_part],
                config=json_config
            )
            return json.loads(pro_response.text)
        except Exception as e2:
            logger.error(f"Pro failed: {e2}")
            raise e2

    async def generate_bbox_crop(self, image_data: bytes, disease_name: str, mime_type: str = "image/jpeg") -> list:
        """
        Standard BBox for Field Crops (Leaf/Plant).
        """
        prompt = f"""
        Analyze this plant leaf image. Identify the regions affected by {disease_name}.
        
        INSTRUCTIONS:
        1. **Draw Bounding Boxes:** Generate a bounding box for EACH distinct lesion or affected area.
        2. **Precision:** The box must be TIGHT around the actual disease symptoms.
        3. **Coordinate System:** You must use a scale of 0 to 1000 for [ymin, xmin, ymax, xmax].

        Return the result as a valid JSON object strictly in this format:
        {{
          "detections": [
            {{
              "label": "{disease_name}",
              "box_2d": [ymin, xmin, ymax, xmax] 
            }}
          ]
        }}
        """
        return await self._generate_bbox_response(prompt, image_data, mime_type)

    async def generate_bbox_vegetable(self, image_data: bytes, disease_name: str, mime_type: str = "image/jpeg") -> list:
        """
        Specialized BBox for Produce (Vegetables/Fruits).
        """
        prompt = f"""
        Analyze this plant image for any signs of pathology (disease, rot, pests, or physical defects).
        
        VISUALIZATION RULES:
        1. **Granular Detection:** Detect **individual** distinct areas of disease (e.g., specific spots on a leaf, rot on a fruit, or mold patches).
        2. **Segmentation:** Do NOT draw a single large box around the whole vegetable/leaf.
        3. **Fragmentation:** If a disease forms a large curve or ring (like blight or rot), **break it into multiple smaller, overlapping boxes** that tightly hug the affected tissue.
        4. **Precision:** Ignore healthy tissue and background. Only box the actual defect.

        Return JSON: 
        {{ "detections": [ {{"label": "Specific Disease Name (e.g. Late Blight, Scab)", "box_2d": [ymin, xmin, ymax, xmax]}} ] }}
        """
        return await self._generate_bbox_response(prompt, image_data, mime_type)

    async def generate_bbox_fruit(self, image_data: bytes, disease_name: str, mime_type: str = "image/jpeg") -> list:
        """
        Specialized BBox for Fruits.
        """
        prompt = f"""
        Act as an expert plant pathologist and computer vision system.
        Analyze the image and detect any plant diseases, pests, rot, or physical defects.

        INSTRUCTIONS:
        1. **Identify the Disease:** Diagnose the specific issue .
        2. **Draw Bounding Boxes:** Generate a bounding box for EACH distinct lesion.
        3. **Precision:** The box must be TIGHT around the actual rot/lesion.
        4. **Coordinate System:** You must use a scale of 0 to 1000 for [ymin, xmin, ymax, xmax].

        Return the result as a valid JSON object strictly in this format:
        {{
          "detections": [
            {{
              "label": "Name of Disease",
              "box_2d": [ymin, xmin, ymax, xmax] 
            }}
          ]
        }}
        """
        return await self._generate_bbox_response(prompt, image_data, mime_type)

    async def _generate_bbox_response(self, prompt: str, image_data: bytes, mime_type: str) -> list:
        content_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.bbox_model_name,
                contents=[prompt, content_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=30000 
                )
            )
            logger.info(f"RAW BBOX RESPONSE: {response.text}")
            
            try:
                data = json.loads(response.text)
                detections = data.get("detections", [])
                boxes = [d.get("box_2d") for d in detections if d.get("box_2d")]
                return boxes
            except json.JSONDecodeError:
                return []
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

gemini_service = GeminiService()