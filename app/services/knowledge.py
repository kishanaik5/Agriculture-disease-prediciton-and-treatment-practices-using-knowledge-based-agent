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

    def get_treatment(self, crop: str, disease: str, scientific_name: str = None, language: str = 'en') -> str:
        """
        Search for a treatment.
        Language: 'en' or 'kn' (Kannada)
        scientific_name: Pathogen scientific name (optional, used for precise lookup in Kannada)
        """
        # Select dataset
        dataset = self.data_kn if language == 'kn' else self.data_en
        
        if not dataset:
            return ""

        # Normalize inputs
        crop_low = crop.lower().strip()
        disease_low = disease.lower().strip()
        
        # 1. search candidates
        candidates = []
        
        if language == 'en':
            # English Logic (Match Crop then Disease)
            candidates = [d for d in dataset if crop_low in d.get('Plant Common Name', '').lower()]
            if not candidates:
                # Fallback: Search all if crop mismatch (e.g. "Pepper Bell" vs "Capsicum")
                candidates = dataset
        else:
            # Kannada Logic
            # Try to resolve English Crop Name to Kannada Name
            kn_crop_name = self.get_kn_name(crop)
            
            # Filter by Crop Name (Kannada) if possible
            if kn_crop_name:
                kn_crop_low = kn_crop_name.strip()
                # Check Plant, Fruit, or Vegetable Common Name columns
                candidates = []
                for row in dataset:
                    # Check all possible crop name columns
                    row_crop = (row.get('Plant Common Name') or row.get('Fruit Common Name') or row.get('Vegetable Common Name') or row.get('Crop Common Name') or '').strip()
                    if kn_crop_low in row_crop or row_crop in kn_crop_low:
                        candidates.append(row)
            
            if not candidates:
                # If no crop match (or mapping failed), search entire dataset 
                # (Risk: False positives, but better than nothing)
                candidates = dataset

            # High Confidence Lookup: Scientific Name (Pathogen)
            # Users often use specific scientific names for diseases.
            if scientific_name:
                sci_low = scientific_name.lower().strip()
                for row in candidates:
                    row_sci = row.get('Scientific name', '').lower()
                    # Clean the row_sci (remove parens etc if needed, but 'in' check catches it)
                    if sci_low and (sci_low in row_sci or row_sci in sci_low):
                        treatment = row.get('Treatment Methods', '')
                        if treatment:
                            return treatment

        # 2. Search for Disease in candidates (Name Match)
        best_match = None
        max_score = 0
        
        for row in candidates:
            # English CSV: "Early Blight"
            # Kannada CSV: ".... (Early Blight)"
            row_disease = row.get('Disease Name (Type)', '').lower()
            
            # Direct usage of substring check
            # For Kannada, 'row_disease' contains '... (disease_low) ...'
            if disease_low in row_disease or row_disease in disease_low:
                # Strong match
                return row.get('Treatment Methods', '')

            # Basic word overlap score
            query_words = set(disease_low.split())
            row_words = set(row_disease.replace('(', ' ').replace(')', ' ').split())
            intersection = query_words.intersection(row_words)
            
            if intersection:
                score = len(intersection)
                if score > max_score:
                    max_score = score
                    best_match = row
        
        # Treatment Methods key update
        if best_match and max_score > 0:
            return best_match.get('Treatment Methods', '')
            
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

    def get_crops_with_icons(self, language: str = 'en', category_filter: str = None) -> List[Dict[str, str]]:
        """
        Return list of crops with Name, image, category.
        Optional: Filter by category (case-insensitive).
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
            
            result.append({
                "Name": name,
                "image": item["image"],
                "category": item["category"]
            })
            
        # Sort by Name
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