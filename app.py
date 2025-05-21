from __future__ import annotations

import io
import json
import re
import itertools
import uuid
import collections
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set

import pandas as pd
import jellyfish  # pip install jellyfish
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator # field_validator for newer Pydantic

# Assuming thefuzz is installed: pip install thefuzz python-levenshtein
from thefuzz import fuzz as _fuzz

# --- Pydantic Models (defined above, copied here for self-containment if run separately) ---
class DeduplicationColumnMap(BaseModel):
    customer_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    tpi: Optional[str] = None

    @field_validator('customer_name') # Example: ensure customer_name is provided if it's critical
    def customer_name_must_be_provided(cls, v: Optional[str], values: Any) -> Optional[str]:
        # If customer_name is absolutely essential for your logic, you can enforce it.
        # For broader applicability, the main logic will check for None.
        # if v is None:
        #     raise ValueError("customer_name mapping is required for deduplication.")
        return v

class DuplicateRecordDetail(BaseModel):
    Row: int
    Name: Optional[str] = None
    Address: Optional[str] = None
    City: Optional[str] = None  # Added City field
    Country: Optional[str] = None  # Added Country field
    TPI: Optional[str] = None  # Added TPI field
    Name_score: Optional[int] = None
    Addr_score: Optional[int] = None
    City_score: Optional[int] = None
    Country_score: Optional[int] = None
    TPI_score: Optional[int] = None
    Overall_score: int
    IsLowConfidence: bool
    BlockType: Optional[str] = None  # Which blocking strategy found this match
    MatchMethod: Optional[str] = None  # Which matching method was most influential (fuzzy, phonetic, etc.)
    BestNameMatchMethod: Optional[str] = None  # Which method gave the best name match
    BestAddrMatchMethod: Optional[str] = None  # Which method gave the best address match
    LLM_conf: Optional[float] = None
    uid: str

class MasterRecord(BaseModel):
    MasterRow: int
    MasterName: Optional[str] = None
    MasterAddress: Optional[str] = None
    MasterCity: Optional[str] = None  # Added MasterCity field
    MasterCountry: Optional[str] = None  # Added MasterCountry field
    MasterTPI: Optional[str] = None  # Added MasterTPI field
    DuplicateCount: int
    AvgSimilarity: float
    IsLowConfidenceGroup: bool
    Duplicates: list[DuplicateRecordDetail]
    master_uid: str

class DeduplicationStats(BaseModel):
    high_confidence_duplicates_groups: int # Count of master groups
    medium_confidence_duplicates_groups: int # Count of master groups
    low_confidence_duplicates_groups: int # Count of master groups where at least one duplicate is low confidence
    block_stats: Dict[str, Any]
    total_master_records_with_duplicates: int
    total_potential_duplicate_records: int


class DeduplicationResponse(BaseModel):
    message: str
    results: Optional[Dict[str, Any]] = None # Make results optional for error cases
    error: Optional[str] = None


# --- Abbreviation Mapping ---
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

# --- Fuzzy Matching Helpers ---
def calculate_similarity_scores(a: Optional[str], b: Optional[str]) -> Dict[str, int]:
    """Calculate multiple similarity scores between two strings."""
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
    """Get the best similarity score from multiple metrics and which method gave the best score."""
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
    """Legacy function maintained for compatibility."""
    if a is None or b is None:
        return 0, "none"
    
    scores = calculate_similarity_scores(a, b)
    return get_best_similarity_score(scores)

# --- Text Normalization Helper ---
def normalize(text: Any) -> str:
    """Enhanced normalization with business-specific rules."""
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
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

# --- Core Deduplication Logic ---
def build_duplicate_df(df: pd.DataFrame, col_map: DeduplicationColumnMap) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enhanced deduplication with multiple blocking strategies and improved similarity metrics.
    """
    # Validate that essential columns from col_map exist in df.columns
    mapped_cols_from_user_file = [val for val in col_map.model_dump().values() if val is not None]
    for user_col_name in mapped_cols_from_user_file:
        if user_col_name not in df.columns:
            raise ValueError(f"Mapped column '{user_col_name}' not found in the uploaded file. Available columns: {df.columns.tolist()}")

    # Create a working copy with only the necessary mapped columns + original index
    # Ensure we only try to select columns that were actually mapped by the user
    cols_to_select = [user_col_name for user_col_name in mapped_cols_from_user_file if user_col_name in df.columns]
    
    if not cols_to_select:
        raise ValueError("No valid columns were mapped or found in the uploaded file for deduplication.")

    work = df[cols_to_select].copy()
    work["ExcelRow"] = df.index # Keep original row index (0-based)

    # Normalize relevant fields
    # These are the internal logical names we use
    logical_fields_for_norm = ["customer_name", "address", "city", "country", "tpi"]
    for logical_field in logical_fields_for_norm:
        user_column_name = getattr(col_map, logical_field, None)
        if user_column_name and user_column_name in work.columns:
            work[f"{logical_field}_norm"] = work[user_column_name].apply(normalize)
            
            # Generate phonetic codes for name fields
            if logical_field == "customer_name":
                # Metaphone encoding (better for company names)
                work["customer_name_metaphone"] = work[user_column_name].apply(
                    lambda x: jellyfish.metaphone(str(x)) if pd.notna(x) else ""
                )
                # Soundex encoding (alternative phonetic algorithm)
                work["customer_name_soundex"] = work[user_column_name].apply(
                    lambda x: jellyfish.soundex(str(x)) if pd.notna(x) else ""
                )
        else:
            # Ensure the normalized column exists even if not mapped, to prevent KeyErrors later
            work[f"{logical_field}_norm"] = ""
            if logical_field == "customer_name":
                work["customer_name_metaphone"] = ""
                work["customer_name_soundex"] = ""

    # --- MULTIPLE BLOCKING STRATEGIES ---
    # We'll create several different blocking schemes and combine their results
    
    # 1. Traditional prefix blocking (first 4 chars of name + first char of city)
    blocks_prefix: dict[str, list[int]] = {}
    # 2. Phonetic blocking using Metaphone
    blocks_metaphone: dict[str, list[int]] = {}
    # 3. Soundex blocking
    blocks_soundex: dict[str, list[int]] = {}
    # 4. N-gram blocking (character 3-grams from name)
    blocks_ngram: dict[str, list[int]] = {}
    
    # Check if columns exist
    customer_name_norm_col_exists = "customer_name_norm" in work.columns and col_map.customer_name is not None
    city_norm_col_exists = "city_norm" in work.columns and col_map.city is not None
    
    # Track all potential comparison pairs to avoid duplicates
    all_comparison_pairs = set()
    
    # Generate blocks using different strategies
    for i, row in work.iterrows(): # i is the original DataFrame index here
        # 1. Prefix blocking
        name_prefix = row["customer_name_norm"][:4] if customer_name_norm_col_exists and pd.notna(row["customer_name_norm"]) and len(row["customer_name_norm"]) >= 4 else "xxxx"
        city_prefix = row["city_norm"][0] if city_norm_col_exists and pd.notna(row["city_norm"]) and row["city_norm"] else "y"
        
        # Ensure name_prefix is not empty if customer_name_norm was empty or too short
        if not name_prefix: name_prefix = "xxxx" # Default prefix if name is empty/short
        
        blocks_prefix.setdefault(f"{name_prefix}_{city_prefix}", []).append(i)
        
        # 2. Metaphone blocking (if available)
        if customer_name_norm_col_exists and pd.notna(row["customer_name_metaphone"]) and row["customer_name_metaphone"]:
            metaphone_key = row["customer_name_metaphone"]
            blocks_metaphone.setdefault(metaphone_key, []).append(i)
            
        # 3. Soundex blocking (if available)
        if customer_name_norm_col_exists and pd.notna(row["customer_name_soundex"]) and row["customer_name_soundex"]:
            soundex_key = row["customer_name_soundex"]
            blocks_soundex.setdefault(soundex_key, []).append(i)
            
        # 4. N-gram blocking
        if customer_name_norm_col_exists and pd.notna(row["customer_name_norm"]) and row["customer_name_norm"]:
            # Generate character 3-grams from the name
            ngrams = generate_ngrams(row["customer_name_norm"], 3)
            for ngram in ngrams:
                blocks_ngram.setdefault(ngram, []).append(i)

    # Combine block statistics
    total_blocks = len(blocks_prefix) + len(blocks_metaphone) + len(blocks_soundex) + len(blocks_ngram)
    all_block_sizes = list(map(len, blocks_prefix.values())) + list(map(len, blocks_metaphone.values())) + \
                      list(map(len, blocks_soundex.values())) + list(map(len, blocks_ngram.values()))
    
    block_stats = {
        "total_blocks": total_blocks,
        "max_block_size": max(all_block_sizes) if all_block_sizes else 0,
        "avg_block_size": (sum(all_block_sizes) / len(all_block_sizes)) if all_block_sizes else 0,
        "records_in_blocks": sum(all_block_sizes),
        "prefix_blocks": len(blocks_prefix),
        "metaphone_blocks": len(blocks_metaphone),
        "soundex_blocks": len(blocks_soundex),
        "ngram_blocks": len(blocks_ngram)
    }

    master_records_dict: dict[int, dict] = {} # Key is master record's original index

    # Function to process comparison pairs from a block
    def process_block_comparisons(block_indices, block_type):
        nonlocal all_comparison_pairs
        
        if len(block_indices) < 2:  # Skip blocks with less than 2 records
            return
            
        for i1_idx, i2_idx in itertools.combinations(block_indices, 2):
            # Skip if we've already compared this pair
            if (i1_idx, i2_idx) in all_comparison_pairs or (i2_idx, i1_idx) in all_comparison_pairs:
                continue
                
            # Add to tracked pairs
            all_comparison_pairs.add((i1_idx, i2_idx))
            
            r1 = work.loc[i1_idx]
            r2 = work.loc[i2_idx]

            # --- ENHANCED SIMILARITY CALCULATION ---
            
            # Name similarity using multiple metrics
            name_scores = {"token_set": 0, "token_sort": 0, "partial": 0, "jaro_winkler": 0, "phonetic": 0}
            if col_map.customer_name and col_map.customer_name in r1 and col_map.customer_name in r2:
                name_scores = calculate_similarity_scores(r1[col_map.customer_name], r2[col_map.customer_name])
            
            # Get best name similarity score and method
            name_s, best_name_method = get_best_similarity_score(name_scores)
            
            # If primary focus is name, and name similarity is too low, skip
            if col_map.customer_name and name_s < 70:  # Threshold for name similarity
                continue

            # Address similarity using multiple metrics
            addr_scores = {"token_set": 0, "token_sort": 0, "partial": 0, "jaro_winkler": 0, "phonetic": 0}
            if col_map.address and col_map.address in r1 and col_map.address in r2:
                addr_scores = calculate_similarity_scores(r1[col_map.address], r2[col_map.address])
            
            addr_s, best_addr_method = get_best_similarity_score(addr_scores)
            
            # City similarity
            city_s = 0
            best_city_method = "none"
            if col_map.city and col_map.city in r1 and col_map.city in r2:
                city_scores = calculate_similarity_scores(r1[col_map.city], r2[col_map.city])
                city_s, best_city_method = get_best_similarity_score(city_scores)
            
            # Country similarity
            country_s = 0
            best_country_method = "none"
            if col_map.country and col_map.country in r1 and col_map.country in r2:
                country_scores = calculate_similarity_scores(r1[col_map.country], r2[col_map.country])
                country_s, best_country_method = get_best_similarity_score(country_scores)
            
            # TPI similarity (if available)
            tpi_s = 0
            best_tpi_method = "none"
            if col_map.tpi and col_map.tpi in r1 and col_map.tpi in r2:
                tpi_scores = calculate_similarity_scores(r1[col_map.tpi], r2[col_map.tpi])
                tpi_s, best_tpi_method = get_best_similarity_score(tpi_scores)
            
            # Calculate weighted overall similarity
            weights = {
                "name": 0.4,  # Name is most important
                "address": 0.3,  # Address is second most important
                "city": 0.1,
                "country": 0.1,
                "tpi": 0.1
            }
            
            weighted_scores = []
            if col_map.customer_name: weighted_scores.append(("name", name_s))
            if col_map.address: weighted_scores.append(("address", addr_s))
            if col_map.city: weighted_scores.append(("city", city_s))
            if col_map.country: weighted_scores.append(("country", country_s))
            if col_map.tpi: weighted_scores.append(("tpi", tpi_s))
            
            if not weighted_scores:
                continue
                
            # Calculate weighted average
            overall = round(sum(weights[field] * score for field, score in weighted_scores) /
                           sum(weights[field] for field, score in weighted_scores))
            
            # Adjust threshold based on block type - we can be more lenient with phonetic blocks
            threshold = 65 if block_type in ["metaphone", "soundex"] else 70
            
            if overall < threshold:  # Overall threshold
                continue

            # Determine which field had the highest influence on the match
            field_scores = [
                ("name", name_s, best_name_method),
                ("address", addr_s, best_addr_method),
                ("city", city_s, best_city_method),
                ("country", country_s, best_country_method)
            ]
            
            # Find the field with the highest score
            best_field = max(field_scores, key=lambda x: x[1] if x[1] is not None else 0)
            match_method = f"{best_field[0]}_{best_field[2]}"
            
            # Construct duplicate record detail with enhanced information
            dup_detail = {
                "Row": int(r2["ExcelRow"]) + 2,  # 1-based for Excel +1 for header
                "Name": str(r2[col_map.customer_name]) if col_map.customer_name and col_map.customer_name in r2 else None,
                "Address": str(r2[col_map.address]) if col_map.address and col_map.address in r2 else None,
                "City": str(r2[col_map.city]) if col_map.city and col_map.city in r2 else None,
                "Country": str(r2[col_map.country]) if col_map.country and col_map.country in r2 else None,
                "TPI": str(r2[col_map.tpi]) if col_map.tpi and col_map.tpi in r2 else None,
                "Name_score": name_s if col_map.customer_name else None,
                "Addr_score": addr_s if col_map.address else None,
                "City_score": city_s if col_map.city else None,
                "Country_score": country_s if col_map.country else None,
                "TPI_score": tpi_s if col_map.tpi else None,
                "Overall_score": overall,
                "IsLowConfidence": overall < 90,  # True if score is below high confidence threshold
                "BlockType": block_type,  # Track which blocking strategy found this match
                "MatchMethod": match_method,  # Which field and method had the highest influence
                "BestNameMatchMethod": best_name_method if name_s > 0 else None,
                "BestAddrMatchMethod": best_addr_method if addr_s > 0 else None,
                "LLM_conf": None,  # Placeholder, not used
                "uid": str(uuid.uuid4())
            }

            master_row_excel_num = int(r1["ExcelRow"]) + 2
            
            if master_row_excel_num not in master_records_dict:
                master_records_dict[master_row_excel_num] = {
                    "MasterRow": master_row_excel_num,
                    "MasterName": str(r1[col_map.customer_name]) if col_map.customer_name and col_map.customer_name in r1 else None,
                    "MasterAddress": str(r1[col_map.address]) if col_map.address and col_map.address in r1 else None,
                    "MasterCity": str(r1[col_map.city]) if col_map.city and col_map.city in r1 else None,
                    "MasterCountry": str(r1[col_map.country]) if col_map.country and col_map.country in r1 else None,
                    "MasterTPI": str(r1[col_map.tpi]) if col_map.tpi and col_map.tpi in r1 else None,
                    "Duplicates": [],
                    "master_uid": str(uuid.uuid4())
                }
            master_records_dict[master_row_excel_num]["Duplicates"].append(dup_detail)
    
    # Process each blocking strategy
    for block_indices in blocks_prefix.values():
        process_block_comparisons(block_indices, "prefix")
        
    for block_indices in blocks_metaphone.values():
        process_block_comparisons(block_indices, "metaphone")
        
    for block_indices in blocks_soundex.values():
        process_block_comparisons(block_indices, "soundex")
        
    # For n-gram blocks, we need to be more selective as they can be very large
    for ngram, block_indices in blocks_ngram.items():
        # Only process n-gram blocks of reasonable size to avoid combinatorial explosion
        if 2 <= len(block_indices) <= 50:  # Skip very large blocks
            process_block_comparisons(block_indices, "ngram")
    
    # --- TRANSITIVE CLOSURE ---
    # Apply transitive closure to find complete duplicate clusters
    # If A matches B and B matches C, then A should be in the same cluster as C
    
    # First, build a graph of relationships
    duplicate_graph = collections.defaultdict(set)
    for master_row, master_data in master_records_dict.items():
        for dup in master_data["Duplicates"]:
            duplicate_graph[master_row].add(dup["Row"])
            duplicate_graph[dup["Row"]].add(master_row)
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    for node in duplicate_graph:
        if node in visited:
            continue
            
        # BFS to find all connected nodes
        cluster = []
        queue = collections.deque([node])
        visited.add(node)
        
        while queue:
            current = queue.popleft()
            cluster.append(current)
            
            for neighbor in duplicate_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        clusters.append(sorted(cluster))
    
    # Rebuild master_records_dict based on clusters
    new_master_records = {}
    
    for cluster in clusters:
        if not cluster:  # Skip empty clusters
            continue
            
        # Use the smallest row number as the master
        master_row = min(cluster)
        
        # Create a new master record
        if master_row in master_records_dict:
            # Use existing master record data
            new_master_records[master_row] = master_records_dict[master_row].copy()
            new_master_records[master_row]["Duplicates"] = []
        else:
            # Create a new master record
            r1 = work.loc[master_row - 2]  # Convert back to 0-based index
            new_master_records[master_row] = {
                "MasterRow": master_row,
                "MasterName": str(r1[col_map.customer_name]) if col_map.customer_name and col_map.customer_name in r1 else None,
                "MasterAddress": str(r1[col_map.address]) if col_map.address and col_map.address in r1 else None,
                "MasterCity": str(r1[col_map.city]) if col_map.city and col_map.city in r1 else None,
                "MasterCountry": str(r1[col_map.country]) if col_map.country and col_map.country in r1 else None,
                "MasterTPI": str(r1[col_map.tpi]) if col_map.tpi and col_map.tpi in r1 else None,
                "Duplicates": [],
                "master_uid": str(uuid.uuid4())
            }
        
        # Add all other rows in the cluster as duplicates
        for row in cluster:
            if row == master_row:
                continue
                
            r2 = work.loc[row - 2]  # Convert back to 0-based index
            
            # Calculate similarity scores for this pair
            name_s = 0
            best_name_method = "none"
            if col_map.customer_name and col_map.customer_name in r1 and col_map.customer_name in r2:
                name_scores = calculate_similarity_scores(r1[col_map.customer_name], r2[col_map.customer_name])
                name_s, best_name_method = get_best_similarity_score(name_scores)
                
            addr_s = 0
            best_addr_method = "none"
            if col_map.address and col_map.address in r1 and col_map.address in r2:
                addr_scores = calculate_similarity_scores(r1[col_map.address], r2[col_map.address])
                addr_s, best_addr_method = get_best_similarity_score(addr_scores)
                
            # City similarity
            city_s = 0
            best_city_method = "none"
            if col_map.city and col_map.city in r1 and col_map.city in r2:
                city_scores = calculate_similarity_scores(r1[col_map.city], r2[col_map.city])
                city_s, best_city_method = get_best_similarity_score(city_scores)
            
            # Country similarity
            country_s = 0
            best_country_method = "none"
            if col_map.country and col_map.country in r1 and col_map.country in r2:
                country_scores = calculate_similarity_scores(r1[col_map.country], r2[col_map.country])
                country_s, best_country_method = get_best_similarity_score(country_scores)
                
            # Calculate overall similarity
            scores_present = []
            if col_map.customer_name: scores_present.append(name_s)
            if col_map.address: scores_present.append(addr_s)
            
            overall = round(sum(scores_present) / len(scores_present)) if scores_present else 0
            
            # Determine which field had the highest influence on the match
            field_scores = [
                ("name", name_s, best_name_method),
                ("address", addr_s, best_addr_method),
                ("city", city_s, best_city_method),
                ("country", country_s, best_country_method)
            ]
            
            # Find the field with the highest score
            best_field = max(field_scores, key=lambda x: x[1] if x[1] is not None else 0)
            match_method = f"{best_field[0]}_{best_field[2]}"
            
            dup_detail = {
                "Row": row,
                "Name": str(r2[col_map.customer_name]) if col_map.customer_name and col_map.customer_name in r2 else None,
                "Address": str(r2[col_map.address]) if col_map.address and col_map.address in r2 else None,
                "City": str(r2[col_map.city]) if col_map.city and col_map.city in r2 else None,
                "Country": str(r2[col_map.country]) if col_map.country and col_map.country in r2 else None,
                "TPI": str(r2[col_map.tpi]) if col_map.tpi and col_map.tpi in r2 else None,
                "Name_score": name_s if col_map.customer_name else None,
                "Addr_score": addr_s if col_map.address else None,
                "City_score": city_s if col_map.city else None,
                "Country_score": country_s if col_map.country else None,
                "Overall_score": overall,
                "IsLowConfidence": overall < 90,
                "BlockType": "transitive",  # Indicate this was found via transitive closure
                "MatchMethod": match_method,  # Which field and method had the highest influence
                "BestNameMatchMethod": best_name_method if name_s > 0 else None,
                "BestAddrMatchMethod": best_addr_method if addr_s > 0 else None,
                "LLM_conf": None,
                "uid": str(uuid.uuid4())
            }
            
            new_master_records[master_row]["Duplicates"].append(dup_detail)
    
    # Replace the original master_records_dict with the new one
    master_records_dict = new_master_records

    # Convert dictionary of master records to a list of MasterRecord-like dicts
    masters_list = []
    for m_data in master_records_dict.values():
        sims = [d["Overall_score"] for d in m_data["Duplicates"]]
        avg_sim = round(sum(sims) / len(sims)) if sims else 0
        
        masters_list.append({
            "MasterRow": m_data["MasterRow"],
            "MasterName": m_data["MasterName"],
            "MasterAddress": m_data["MasterAddress"],
            "MasterCity": m_data.get("MasterCity"),  # Use get() to handle missing keys
            "MasterCountry": m_data.get("MasterCountry"),
            "MasterTPI": m_data.get("MasterTPI"),
            "DuplicateCount": len(m_data["Duplicates"]),
            "AvgSimilarity": avg_sim,
            "IsLowConfidenceGroup": any(d["IsLowConfidence"] for d in m_data["Duplicates"]),
            "Duplicates": m_data["Duplicates"],
            "master_uid": m_data["master_uid"]
        })
    
    # Create DataFrame from the list of master records
    # Sort by average similarity to bring more likely duplicates to the top
    results_df = pd.DataFrame(masters_list)
    if not results_df.empty:
        results_df = results_df.sort_values("AvgSimilarity", ascending=False).reset_index(drop=True)
        
    return results_df, block_stats


# --- FastAPI App Setup ---
app = FastAPI(
    title="Simplified Duplicate Finder API",
    description="Accepts a file and column mappings to find duplicates.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoint ---
@app.post("/deduplicate/", response_model=DeduplicationResponse)
async def run_deduplication(
    file: UploadFile = File(..., description="CSV or XLSX file to be deduplicated."),
    column_map_json: str = Form(..., description="JSON string of DeduplicationColumnMap Pydantic model.")
):
    """
    Processes an uploaded file with specified column mappings to find duplicates.
    """
    try:
        # Parse the column mapping JSON string
        try:
            column_map_data = json.loads(column_map_json)
            column_map = DeduplicationColumnMap(**column_map_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for column mapping.")
        except Exception as e: # Catches Pydantic validation errors
            raise HTTPException(status_code=400, detail=f"Invalid column mapping data: {str(e)}")

        # Read file content
        content = await file.read()
        
        # Determine file type and read into DataFrame
        if file.filename.endswith(".csv"):
            try:
                # Basic CSV read, consider adding encoding detection if needed
                df_raw = pd.read_csv(io.BytesIO(content), dtype=str, na_filter=False, keep_default_na=False)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        elif file.filename.endswith((".xls", ".xlsx")):
            try:
                df_raw = pd.read_excel(io.BytesIO(content), dtype=str, na_filter=False, keep_default_na=False)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV or Excel.")

        if df_raw.empty:
            return DeduplicationResponse(message="Uploaded file is empty.", error="File empty")

        # Run deduplication logic
        try:
            dup_df, block_stats_dict = build_duplicate_df(df_raw, column_map)
        except ValueError as ve: # Catch specific errors from build_duplicate_df
             return DeduplicationResponse(message=f"Deduplication error: {str(ve)}", error=str(ve))


        # Prepare statistics
        # These counts are based on "master" groups of duplicates
        high_conf_groups = dup_df[dup_df["AvgSimilarity"] >= 98] if not dup_df.empty else pd.DataFrame()
        medium_conf_groups = dup_df[
            (dup_df["AvgSimilarity"] < 98) & (dup_df["AvgSimilarity"] >= 90)
        ] if not dup_df.empty else pd.DataFrame()
        
        # Low confidence groups are those where any duplicate has IsLowConfidence = True (overall < 90)
        # Or, more simply, groups with AvgSimilarity < 90, as IsLowConfidenceGroup is already calculated
        low_conf_groups = dup_df[dup_df["IsLowConfidenceGroup"]] if not dup_df.empty else pd.DataFrame()
        
        total_potential_duplicate_records_count = 0
        if not dup_df.empty:
            total_potential_duplicate_records_count = dup_df['DuplicateCount'].sum()


        stats = DeduplicationStats(
            high_confidence_duplicates_groups=len(high_conf_groups),
            medium_confidence_duplicates_groups=len(medium_conf_groups),
            low_confidence_duplicates_groups=len(low_conf_groups), # Count of groups containing at least one low confidence duplicate
            block_stats=block_stats_dict,
            total_master_records_with_duplicates=len(dup_df),
            total_potential_duplicate_records=total_potential_duplicate_records_count
        )

        # Convert DataFrame to list of dicts for the response
        # This should align with the MasterRecord Pydantic model
        duplicates_list = dup_df.to_dict(orient="records") if not dup_df.empty else []
        
        # Validate with Pydantic models before sending (optional, good for debugging)
        # validated_duplicates = [MasterRecord(**item) for item in duplicates_list]
        # validated_stats = DeduplicationStats(**stats.model_dump())


        # Filter out duplicates that are less than 100% confidence (AvgSimilarity < 100)
        potential_duplicates = [
            record for record in duplicates_list
            if record["AvgSimilarity"] < 100 or record["IsLowConfidenceGroup"]
        ]

        # Add KPI metrics that match the Streamlit app naming convention
        kpi_metrics = {
            "auto_merge": len(high_conf_groups),  # AvgSimilarity >= 98
            "needs_review": len(medium_conf_groups),  # 90 <= AvgSimilarity < 98
            "needs_ai": len(low_conf_groups),  # IsLowConfidenceGroup = True
            "total_blocks": block_stats_dict.get("total_blocks", 0)
        }

        return DeduplicationResponse(
            message="Deduplication process completed.",
            results={
                "duplicate_group_count": len(duplicates_list), # Number of master records with duplicates
                "total_potential_duplicates": stats.total_potential_duplicate_records, # Sum of all individual duplicates found
                "duplicates": duplicates_list, # List of MasterRecord-like dicts
                "potential_duplicates": potential_duplicates, # Duplicates with less than 100% confidence
                "kpi_metrics": kpi_metrics, # KPI metrics matching Streamlit app
                "stats": stats.model_dump(), # Convert Pydantic model to dict
                "column_map": column_map.model_dump() # Include the column mapping in the response
            }
        )

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        # Log the error in a real application: import logging; logging.exception("Deduplication error")
        return DeduplicationResponse(message=f"An unexpected error occurred: {str(e)}", error=str(e))


@app.get("/")
async def root():
    return {"message": "Simplified Duplicate Finder API is running. Use the /deduplicate/ endpoint to process files."}

# To run this app:
# 1. Save as app.py (or similar)
# 2. Install dependencies: pip install fastapi uvicorn pandas "python-multipart" "thefuzz[speedup]" openpyxl
#    (openpyxl for .xlsx, python-levenshtein is included in thefuzz[speedup] for better performance)
# 3. Run with uvicorn: uvicorn app:app --reload
