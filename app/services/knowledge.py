import csv
import logging
from typing import List, Optional, Dict
import os

logger = logging.getLogger(__name__)

class KnowledgeService:
    def __init__(self, 
                 csv_path_en: str = "knowledge_based_folder/crop_kb_data/Crop_kb_data_en.csv",
                 csv_path_kn: str = "knowledge_based_folder/crop_kb_data/Crop_kb_data_kn.csv"):
        
        self.csv_path_en = self._resolve_path(csv_path_en)
        self.csv_path_kn = self._resolve_path(csv_path_kn)
        
        self.data_en: List[Dict[str, str]] = []
        self.data_kn: List[Dict[str, str]] = []
        
        self._load_data(self.csv_path_en, 'en')
        self._load_data(self.csv_path_kn, 'kn')

        # Load Icons dynamically from folders
        self.crop_items = []
        self._load_all_icons("icons_folder")

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
                    self.data_en = rows
                else:
                    self.data_kn = rows
                
            logger.info(f"Loaded {len(rows)} records for language: {lang}")
            
        except Exception as e:
            logger.error(f"Failed to load Knowledge CSV ({lang}): {e}")

    def get_treatment(self, crop: str, disease: str, language: str = 'en') -> str:
        """
        Search for a treatment.
        Language: 'en' or 'kn' (Kannada)
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
                candidates = dataset
        else:
            # Kannada Logic
            # Issue: Crop Name in CSV is Kannada (e.g. ಟೊಮೆಟೊ), but input is English (e.g. Tomato).
            # Strategy: Skip Crop match if we can't match it, or check if we can matching substring?
            # Better: Trust the Disease Match significantly more because it has English in parens.
            # We will search the ENTIRE Kannada dataset for the English Disease Name.
            candidates = dataset

        # 2. Search for Disease in candidates
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
                return row.get('Treatment Methods (Human-Friendly Guide)', '')

            # Basic word overlap score
            query_words = set(disease_low.split())
            row_words = set(row_disease.replace('(', ' ').replace(')', ' ').split())
            intersection = query_words.intersection(row_words)
            
            if intersection:
                score = len(intersection)
                if score > max_score:
                    max_score = score
                    best_match = row

        if best_match and max_score > 0:
            return best_match.get('Treatment Methods (Human-Friendly Guide)', '')
            
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
                # Category Name (capitalize)
                category_name = item.title() # "crops" -> "Crops"
                
                # Special handling if user wants singular/plural mapping?
                # User's example: "crops/fruits/vegetable"
                # If folder is "vegetables", title() gives "Vegetables". 
                # User used "vegetable" (singular). Let's stick to title case of folder for now unless mapped.
                if item.lower() == "vegetables":
                    category_name = "Vegetable" # Matching user's sample output style
                
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

    def get_crops_with_icons(self, language: str = 'en') -> List[Dict[str, str]]:
        """
        Return list of crops with Name, image, category
        """
        result = []
        for item in self.crop_items:
            # Select name based on language
            name = item["kannada_name"] if language == 'kn' and item.get("kannada_name") else item["Name"]
            
            result.append({
                "Name": name,
                "image": item["image"],
                "category": item["category"]
            })
            
        # Sort by Name
        result.sort(key=lambda x: x["Name"])
        return result

# Global Instance
knowledge_service = KnowledgeService()