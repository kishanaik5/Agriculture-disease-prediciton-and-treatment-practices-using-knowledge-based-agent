import csv
import logging
from typing import List, Optional, Dict
import os

logger = logging.getLogger(__name__)

class KnowledgeService:
    def __init__(self, 
                 csv_path_en: str = "knowledge_based_folder/crop_kb_data/Crop_kb_data_en.csv",
                 csv_path_kn: str = "knowledge_based_folder/crop_kb_data/Crop_kb_data_kn.csv"):
        
        self.data_en: List[Dict[str, str]] = []
        self.data_kn: List[Dict[str, str]] = []
        
        # Load All KB Data
        self._load_kb_files()

        # Load Icons dynamically from folders
        self.crop_items = []
        self._load_all_icons("icons_folder")

    def get_kn_name(self, eng_name: str) -> str:
        """Resolve English Crop Name to Kannada Name using loaded icons data."""
        s = eng_name.lower().strip()
        for item in self.crop_items:
            # item["Name"] is usually Title Case, convert to lower
            if item["Name"].lower() == s:
                return item.get("kannada_name", "")
        return ""

    def _resolve_path(self, path: str) -> str:
        if not os.path.exists(path):
             project_root = os.getcwd() 
             possible_path = os.path.join(project_root, path)
             if os.path.exists(possible_path):
                 return possible_path
        return path

    def _load_data(self, path: str, lang: str):
        try:
            if not os.path.exists(path):
                logger.warning(f"Knowledge Base CSV ({lang}) not found at: {path}")
                return

            with open(path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    cleaned_fieldnames = [name.strip() for name in reader.fieldnames]
                    reader.fieldnames = cleaned_fieldnames
                
                rows = [row for row in reader]
                if lang == 'en':
                    self.data_en.extend(rows)
                else:
                    self.data_kn.extend(rows)
                
            logger.info(f"Loaded {len(rows)} records for language: {lang}")
            
        except Exception as e:
            logger.error(f"Failed to load Knowledge CSV ({lang}): {e}")

    async def get_treatment(self, crop: str, disease: str, category: str, db: Optional[object] = None, scientific_name: str = None, language: str = 'en') -> str:
        """
        Search for a treatment.
        Language: 'en' or 'kn' (Kannada)
        scientific_name: Pathogen scientific name (optional)
        db: AsyncSession (required for 'en' lookup)
        """
        import re
        from sqlalchemy import select, func, or_
        from app.models.scan import KnowledgeBase

        # Normalize inputs
        crop_input = crop.strip()
        disease_input = disease.strip()
        
        # Normalize category
        cat_search = category.lower()
        if cat_search.endswith('s') and cat_search != "crops": 
            cat_search = cat_search[:-1]
        elif cat_search == "crops":
            cat_search = "crop"

        if language == 'en':
            if not db:
                logger.warning("DB session not provided for English KB lookup. Returning empty.")
                return ""
            
            try:
                # Multi-stage KB Lookup Strategy:
                # Stage 1: Category + Item Name + Scientific Name (Pattern Match)
                # Stage 2: Category + Item Name + Disease Name (Exact Match)
                # Stage 3: Category + Item Name + Disease Name (Pattern Match)
                
                # Stage 1: Try scientific name with pattern matching (ILIKE)
                if scientific_name and len(scientific_name) > 2:
                    stmt = select(KnowledgeBase.treatment).where(
                        func.lower(KnowledgeBase.category) == cat_search,
                        func.lower(KnowledgeBase.item_name) == crop_input.lower(),
                        KnowledgeBase.scientific_name.ilike(f"%{scientific_name}%")
                    )
                    result = await db.execute(stmt)
                    treatment = result.scalar_one_or_none()
                    
                    if treatment:
                        logger.info(f"KB Match (Stage 1 - Scientific Pattern): {crop_input} + {scientific_name}")
                        return treatment
                
                # Stage 2: Try exact disease name match
                stmt = select(KnowledgeBase.treatment).where(
                    func.lower(KnowledgeBase.category) == cat_search,
                    func.lower(KnowledgeBase.item_name) == crop_input.lower(),
                    func.lower(KnowledgeBase.disease_name) == disease_input.lower()
                )
                result = await db.execute(stmt)
                treatment = result.scalar_one_or_none()
                
                if treatment:
                    logger.info(f"KB Match (Stage 2 - Disease Exact): {crop_input} + {disease_input}")
                    return treatment
                
                # Stage 3: Try disease name pattern matching (more lenient)
                stmt = select(KnowledgeBase.treatment).where(
                    func.lower(KnowledgeBase.category) == cat_search,
                    func.lower(KnowledgeBase.item_name) == crop_input.lower(),
                    KnowledgeBase.disease_name.ilike(f"%{disease_input}%")
                )
                result = await db.execute(stmt)
                treatment = result.scalar_one_or_none()
                
                if treatment:
                    logger.info(f"KB Match (Stage 3 - Disease Pattern): {crop_input} + {disease_input}")
                    return treatment
                
                logger.warning(f"KB No Match: {cat_search}/{crop_input}/{disease_input}/{scientific_name}")
                return ""
                
            except Exception as e:
                logger.error(f"DB Lookup failed: {e}")
                return ""

        # Kannada Logic (Memory-based fallback)
        dataset = self.data_kn
        if not dataset:
            return ""

        # Regex patterns for strict matching (case-insensitive)
        try:
            crop_pattern = re.compile(f"^{re.escape(crop_input)}$", re.IGNORECASE)
            disease_pattern = re.compile(f"^{re.escape(disease_input)}$", re.IGNORECASE)
        except re.error as e:
            logger.error(f"Regex compilation failed: {e}")
            return ""

        # Column names
        plant_keys = ['Plant Common Name', 'Fruit Common Name', 'Vegetable Common Name', 'Crop Common Name']

        # 1. Resolve English Crop Name to Kannada Name
        kn_crop_name = self.get_kn_name(crop)
        
        candidates = []
        if kn_crop_name:
            kn_crop_pattern = re.compile(f"^{re.escape(kn_crop_name.strip())}$", re.IGNORECASE)
            for row in dataset:
                match = False
                for key in plant_keys:
                    val = row.get(key, '').strip()
                    if val and kn_crop_pattern.match(val):
                        match = True
                        break
                if match:
                    candidates.append(row)
        
        if not candidates:
            # Fallback search if mapping failed
            candidates = dataset

        # 2. Search for Disease in candidates
        # High Confidence: Scientific Name check first
        if scientific_name:
            sci_pattern = re.compile(f".*{re.escape(scientific_name.strip())}.*", re.IGNORECASE)
            for row in candidates:
                row_sci = row.get('Scientific name', '')
                if row_sci and sci_pattern.match(row_sci):
                    t = row.get('Treatment Methods', '')
                    if t: return t

        # 3. Disease Name Match
        for row in candidates:
            row_disease = row.get('Disease Name (Type)', '').strip()
            if row_disease and disease_pattern.match(row_disease):
                return row.get('Treatment Methods', '')

        return ""

    def get_unique_crops(self, language: str = 'en') -> List[str]:
        """
        Get unique crop names from the loaded CSV.
        """
        dataset = self.data_kn if language == 'kn' else self.data_en
        if not dataset:
            return []
            
        crops = set()
        for row in dataset:
            name = row.get('Plant Common Name', '').strip()
            if name:
                crops.add(name)
        
    def _load_all_icons(self, root_folder: str):
        """
        Scan subfolders of root_folder.
        Category = Subfolder Name (e.g. 'fruits' -> 'Fruits')
        Load all CSVs in that subfolder.
        """
        resolved_root = self._resolve_path(root_folder)
        if not os.path.exists(resolved_root):
            logger.warning(f"Icons root folder not found: {resolved_root}")
            return

        # Iterate over subdirectories in icons_folder
        # e.g. crops, fruits, vegetables
        for item in os.listdir(resolved_root):
            category_path = os.path.join(resolved_root, item)
            
            if os.path.isdir(category_path):
                # Normalize Category Name (lowercase, singular)
                # "crops" -> "crop", "fruits" -> "fruit", "vegetables" -> "vegetable"
                raw_cat = item.lower()
                if raw_cat.endswith('s'):
                    category_name = raw_cat[:-1] 
                else:
                    category_name = raw_cat
                
                # Explicit overrides if needed
                if category_name not in ["crop", "fruit", "vegetable"]:
                     # Fallback or keep as is? 
                     # If folder is "other", it becomes "other".
                     pass
                
                # Find CSV files in this category folder
                for file_name in os.listdir(category_path):
                    if file_name.endswith(".csv"):
                        self._process_icon_csv(os.path.join(category_path, file_name), category_name)

    def _process_icon_csv(self, csv_path: str, category: str):
        try:
            with open(csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_name = row.get("crop_name", "").lower().replace("_crop", "")
                    url = row.get("url", "")
                    kannada_name = row.get("kannada_name", "")
                    
                    if raw_name:
                        # Format Name: "green_chilli" -> "Green Chilli"
                        if raw_name == "ladies_finger":
                            display_name = "Ladies Finger (Okra)"
                        elif raw_name == "pepper_belly":
                            display_name = "Pepper Bell"
                        else:
                            display_name = raw_name.replace("_", " ").title()
                        
                        self.crop_items.append({
                            "Name": display_name,
                            "kannada_name": kannada_name,
                            "image": url,
                            "category": category
                        })
        except Exception as e:
            logger.error(f"Failed to load icons csv {csv_path}: {e}")

    def _load_kb_files(self):
        """Loads all KB CSVs from configured paths"""
        # Crops
        self._load_data("knowledge_based_folder/crop_kb_data/Crop_kb_data_en.csv", 'en')
        self._load_data("knowledge_based_folder/crop_kb_data/Crop_kb_data_kn.csv", 'kn')
        
        # Fruits
        self._load_data("knowledge_based_folder/fruits_kb_data/Fruit_kb_data_en.csv", 'en')
        self._load_data("knowledge_based_folder/fruits_kb_data/Fruit_kb_data_kn.csv", 'kn')
        
        # Vegetables
        self._load_data("knowledge_based_folder/vegetables_kb_data/Vegetable_kb_data_en.csv", 'en')
        self._load_data("knowledge_based_folder/vegetables_kb_data/Vegetable_kb_data_kn.csv", 'kn')

    def get_crops_with_icons(self, language: str = 'en', category_filter: str = None, name_filter: str = None) -> List[Dict[str, str]]:
        """
        Return list of crops with Name, image, category.
        Optional: Filter by category (case-insensitive) or name (case-insensitive substring).
        """
        result = []
        for item in self.crop_items:
            # Filter by category if provided
            if category_filter:
                # "Crops" vs "Fruits" vs "Vegetable"
                # Normalize both to safe check
                if item["category"].lower() != category_filter.lower():
                    continue

            # Select name based on language
            name = item["kannada_name"] if language == 'kn' and item.get("kannada_name") else item["Name"]
            
            # Filter by name if provided
            if name_filter:
                if name_filter.lower() not in name.lower():
                    continue
            
            result.append({
                "Name": name,
                "image": item["image"],
                "category": item["category"]
            })
            
        # Sort by Name
        result.sort(key=lambda x: x["Name"])
        return result

    def get_crop_category(self, crop_name: str) -> Optional[str]:
        """
        Identify the category of a crop based on loaded icons list.
        Returns: "Fruits", "Vegetable", "Crops" or None.
        """
        if not crop_name:
            return None
            
        search = crop_name.lower().strip()
        
        for item in self.crop_items:
            # Check English Name (e.g. "Green Chilli" vs "green chilli")
            if item["Name"].lower() == search:
                return item["category"]
            
            # Check against raw internal keys if needed or kannada?
            # Assuming main input is English Common Name.
            
        return None

# Global Instance
knowledge_service = KnowledgeService()