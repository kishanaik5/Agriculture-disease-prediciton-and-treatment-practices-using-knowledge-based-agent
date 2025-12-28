from google import genai
from google.genai import types
from app.config import settings
import json
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

    async def analyze_image(self, image_data: bytes, crop_name: str, language: str = "en", mime_type: str = "image/jpeg") -> dict:
        """
        Analyzes the crop image using Gemini Flash with 5 retries, falling back to Pro on failure.
        """
        
        lang_instruction = "Respond entirely in KANNADA." if language.lower() in ['kn', 'kannada'] else "Respond in ENGLISH."

        prompt = f"""
        Role: You are an expert Plant Pathologist and Agricultural Advisory System.

        Task:
        Analyze the attached image of a plant leaf to perform a comprehensive diagnosis and advisory generation.
        
        [LANGUAGE INSTRUCTION]: {lang_instruction}

        [CONTEXT]:
        The user claims this is a "{crop_name}" plant.
        
        Primary Objectives:
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

    async def generate_bounding_boxes(self, image_data: bytes, disease_name: str, mime_type: str = "image/jpeg") -> list:
        target = disease_name if disease_name and disease_name.lower() not in ["no visible disease","healthy", "none", "n/a"] else "necrotic lesions"

        # --- ATOMIC DETECTION PROMPT ---
        prompt = f"""
        [TASK] Detect EVERY individual {target} lesion on this leaf.
        [GROUNDING RULES]
        1. Use a 1000x1000 grid.
        2. ATOMIC DETECTION: Do NOT group spots. Every distinct lesion MUST have its own tiny box.
        3. TIGHTNESS: Boxes must tightly enclose only the dark center of each spot.
        4. COORDINATE ORDER: You MUST return boxes ONLY as [ymin, xmin, ymax, xmax].
           - ymin: vertical start from top.
           - xmin: horizontal start from left.
           - ymax: vertical end.
           - xmax: horizontal end.
        
        [OUTPUT] JSON only: {{"lesions": [{{"box_2d": [ymin, xmin, ymax, xmax]}}]}} 
        """
        
        # Prepare content part (image)
        content_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.bbox_model_name,
                contents=[prompt, content_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=30000,
                    system_instruction=self.bbox_instruction
                )
            )
            
            # --- CRITICAL FOR AUTOMATION: LOG THE RAW RESPONSE ---
            # This lets you see the numbers in your terminal/server logs
            logger.info(f"RAW BBOX RESPONSE: {response.text}")
            
            try:
                data = json.loads(response.text)
                return [item["box_2d"] for item in data.get("lesions", [])]
            except json.JSONDecodeError:
                logger.warning("JSON Decode failed (likely truncated), attempting Regex fallback...")
                import re
                # Pattern for [ymin, xmin, ymax, xmax]
                pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
                matches = re.findall(pattern, response.text)
                boxes = []
                for m in matches:
                    boxes.append([int(m[0]), int(m[1]), int(m[2]), int(m[3])])
                
                logger.info(f"Regex recovered {len(boxes)} boxes from truncated response.")
                return boxes
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

gemini_service = GeminiService()