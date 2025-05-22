"""
Fuzzy Matching Utilities
-----------------------
This module contains functions for comparing strings using various fuzzy matching algorithms.
These functions are crucial for identifying potential duplicates where text may not match exactly.
"""

import jellyfish  # pip install jellyfish
from typing import Dict, Optional, Tuple
from thefuzz import fuzz as _fuzz

def calculate_similarity_scores(a: Optional[str], b: Optional[str]) -> Dict[str, int]:
    """
    Calculate multiple similarity scores between two strings using different algorithms.
    
    This function compares two strings using various fuzzy matching techniques:
    - Token set ratio: Order-independent comparison of token sets
    - Token sort ratio: Order-dependent comparison after sorting tokens
    - Partial ratio: Best partial string matches
    - Jaro-Winkler similarity: Good for names with typos
    - Phonetic matching: Sounds-alike comparison
    
    Args:
        a: First string to compare
        b: Second string to compare
        
    Returns:
        Dict[str, int]: Dictionary of similarity scores by method name
    """
    if a is None or b is None:
        return {"token_set": 0, "token_sort": 0, "partial": 0, "jaro_winkler": 0, "phonetic": 0}
    
    a_str, b_str = str(a), str(b)
    
    # Calculate various fuzzy matching scores
    token_set = _fuzz.token_set_ratio(a_str, b_str)
    token_sort = _fuzz.token_sort_ratio(a_str, b_str)
    partial = _fuzz.partial_ratio(a_str, b_str)
    
    # Jaro-Winkler similarity (good for names)
    jaro_winkler = int(jellyfish.jaro_winkler_similarity(a_str, b_str) * 100)
    
    # Phonetic matching
    phonetic_match = 0
    try:
        a_metaphone = jellyfish.metaphone(a_str)
        b_metaphone = jellyfish.metaphone(b_str)
        if a_metaphone and b_metaphone:
            phonetic_match = 100 if a_metaphone == b_metaphone else int(jellyfish.jaro_winkler_similarity(a_metaphone, b_metaphone) * 100)
    except:
        # Handle potential errors in phonetic encoding
        pass
    
    return {
        "token_set": token_set,
        "token_sort": token_sort,
        "partial": partial,
        "jaro_winkler": jaro_winkler,
        "phonetic": phonetic_match
    }

def get_best_similarity_score(scores: Dict[str, int]) -> Tuple[int, str]:
    """
    Get the best similarity score from multiple metrics as a weighted average.
    
    Args:
        scores: Dictionary of similarity scores by method name
        
    Returns:
        Tuple[int, str]: The weighted average score and the name of the best individual method
    """
    # Weighted average of scores, giving more weight to certain metrics
    weights = {
        "token_set": 0.3,
        "token_sort": 0.2,
        "partial": 0.2,
        "jaro_winkler": 0.2,
        "phonetic": 0.1
    }
    
    weighted_sum = sum(score * weights[metric] for metric, score in scores.items())
    
    # Find which method gave the highest individual score
    best_method = max(scores.items(), key=lambda x: x[1])[0] if scores else "none"
    
    return round(weighted_sum), best_method

def neo_token_set_ratio(a: Optional[str], b: Optional[str]) -> Tuple[int, str]:
    """
    Legacy function maintained for compatibility. Calculates similarity with weighted metrics.
    
    Args:
        a: First string to compare
        b: Second string to compare
        
    Returns:
        Tuple[int, str]: The weighted similarity score and the name of the best method
    """
    if a is None or b is None:
        return 0, "none"
    
    scores = calculate_similarity_scores(a, b)
    return get_best_similarity_score(scores) 