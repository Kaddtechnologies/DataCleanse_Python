"""
Text Processing Utilities
------------------------
This module contains functions for processing and normalizing text data,
which is a critical step in the deduplication process.
"""

import re
from typing import Any, Set

# --- Abbreviation Mapping ---
# Dictionary mappings for common abbreviations in company names and addresses
COMPANY_ABBR = {
    "corporation": ["corp", "corp.", "co", "co.", "inc", "inc.", "incorporated", "company", "llc", "ltd", "limited"],
    "international": ["intl", "int'l", "int"],
    "brothers": ["bros", "bros."],
    "industries": ["ind", "ind.", "indus"],
    "manufacturing": ["mfg", "mfg.", "manuf"],
    "services": ["svc", "svc.", "svcs", "svcs."],
    "associates": ["assoc", "assoc.", "assc", "assc."],
    "systems": ["sys", "sys."],
    "solutions": ["sol", "sol.", "soln", "soln."],
    "technologies": ["tech", "tech.", "techn"],
    "products": ["prod", "prod.", "prods"],
    "enterprises": ["ent", "ent.", "enterp"],
}

ADDRESS_ABBR = {
    "street": ["st", "st."],
    "avenue": ["ave", "ave."],
    "boulevard": ["blvd", "blvd."],
    "road": ["rd", "rd."],
    "drive": ["dr", "dr."],
    "lane": ["ln", "ln."],
    "place": ["pl", "pl."],
    "court": ["ct", "ct."],
    "suite": ["ste", "ste.", "suite"],
    "apartment": ["apt", "apt.", "apartment", "unit", "unit #", "# "],
    "building": ["bldg", "bldg.", "building"],
    "floor": ["fl", "fl.", "floor"],
    "north": ["n", "n."],
    "south": ["s", "s."],
    "east": ["e", "e."],
    "west": ["w", "w."],
}

def normalize(text: Any) -> str:
    """
    Enhanced normalization with business-specific rules.
    
    This function normalizes text by:
    1. Converting to lowercase
    2. Removing prefixes like "the" 
    3. Removing special characters
    4. Standardizing company abbreviations to their full forms
    5. Standardizing address abbreviations to their full forms
    
    Args:
        text: The text to normalize (can be any type, will be converted to str)
        
    Returns:
        str: The normalized text
    """
    if text is None:
        return ""
    
    # Basic normalization
    text = str(text).lower().strip()
    
    # Replace specific patterns before general cleaning
    # Handle common prefixes like "the" that don't add value for matching
    text = re.sub(r"^the\s+", "", text)
    
    # Handle special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Keep spaces
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces to single
    
    # Normalize company abbreviations
    words = text.split()
    normalized_words = []
    
    for word in words:
        # Skip very short words or numbers
        if len(word) <= 1 or word.isdigit():
            normalized_words.append(word)
            continue
            
        # Check if word is in any abbreviation list
        normalized = word
        for full_form, abbrs in COMPANY_ABBR.items():
            if word in abbrs or word == full_form:
                normalized = full_form  # Standardize to full form
                break
                
        for full_form, abbrs in ADDRESS_ABBR.items():
            if word in abbrs or word == full_form:
                normalized = full_form  # Standardize to full form
                break
                
        normalized_words.append(normalized)
    
    return " ".join(normalized_words).strip()

def generate_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Generate character n-grams from text.
    
    N-grams are continuous sequences of n characters from a string, useful for
    fuzzy matching when exact matches aren't possible.
    
    Args:
        text: The input text to generate n-grams from
        n: The size of each n-gram (default: 3)
        
    Returns:
        Set[str]: A set of n-grams
    """
    if not text or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1)) 