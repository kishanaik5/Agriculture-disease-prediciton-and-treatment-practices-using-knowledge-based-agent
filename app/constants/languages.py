"""
Supported languages configuration for the Kisaan CV Service.
"""

# Supported language codes and their full names
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'kn': 'Kannada',
    'hn': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'gu': 'Gu jarati',
    'bn': 'Bengali',
    'or': 'Odia'
}

# Just the language codes for quick validation
SUPPORTED_LANGUAGE_CODES = list(SUPPORTED_LANGUAGES.keys())

# Default language
DEFAULT_LANGUAGE = 'en'

# Language validation helper
def is_supported_language(language_code: str) -> bool:
    """Check if a language code is supported"""
    return language_code.lower() in SUPPORTED_LANGUAGE_CODES

def get_language_name(language_code: str) -> str:
    """Get full language name from code"""
    return SUPPORTED_LANGUAGES.get(language_code.lower(), 'Unknown')
