"""
Implementation with prefix, metaphone, and soundex blocking strategies
"""
import pandas as pd
import itertools
import uuid
import jellyfish
from typing import Dict, Any, Tuple, Set

def build_duplicate_df(df: pd.DataFrame, col_map: Dict[str, str], name_threshold: int = 70, overall_threshold: int = 70) -> Tuple[int, int]:
    """
    Deduplication with prefix, metaphone, and soundex blocking strategies
    Returns: (master_count, duplicate_count)
    """
    # Create a working copy with only the necessary mapped columns
    work = df[[col for col in col_map.values() if col]].copy().reset_index(drop=False)
    work = work.rename(columns={"index": "ExcelRow"})
    
    # Normalize relevant fields
    for field in ["customer_name", "address", "city", "country"]:
        if col_map[field]:
            work[f"{field}_norm"] = work[col_map[field]].apply(normalize)
    
    # Generate phonetic codes for name fields
    if col_map["customer_name"]:
        # Metaphone encoding (better for company names)
        work["customer_name_metaphone"] = work[col_map["customer_name"]].apply(
            lambda x: jellyfish.metaphone(str(x)) if pd.notna(x) else ""
        )
        # Soundex encoding (alternative phonetic algorithm)
        work["customer_name_soundex"] = work[col_map["customer_name"]].apply(
            lambda x: jellyfish.soundex(str(x)) if pd.notna(x) else ""
        )
    
    # --- BLOCKING STRATEGIES ---
    # 1. Traditional prefix blocking (first 4 chars of name + first char of city)
    blocks_prefix = {}
    # 2. Phonetic blocking using Metaphone
    blocks_metaphone = {}
    # 3. Soundex blocking
    blocks_soundex = {}
    
    # Track all potential comparison pairs to avoid duplicates
    all_comparison_pairs = set()
    
    # Generate blocks using different strategies
    for i, row in work.iterrows():
        # 1. Prefix blocking
        name_prefix = row["customer_name_norm"][:4] if len(row["customer_name_norm"]) >= 4 else row["customer_name_norm"]
        city_prefix = row["city_norm"][0] if col_map["city"] and len(row["city_norm"]) > 0 else ""
        block_key = f"{name_prefix}_{city_prefix}"
        
        if block_key not in blocks_prefix:
            blocks_prefix[block_key] = []
        blocks_prefix[block_key].append(i)
        
        # 2. Metaphone blocking (if available)
        if pd.notna(row["customer_name_metaphone"]) and row["customer_name_metaphone"]:
            metaphone_key = row["customer_name_metaphone"]
            if metaphone_key not in blocks_metaphone:
                blocks_metaphone[metaphone_key] = []
            blocks_metaphone[metaphone_key].append(i)
            
        # 3. Soundex blocking (if available)
        if pd.notna(row["customer_name_soundex"]) and row["customer_name_soundex"]:
            soundex_key = row["customer_name_soundex"]
            if soundex_key not in blocks_soundex:
                blocks_soundex[soundex_key] = []
            blocks_soundex[soundex_key].append(i)
    
    # Find duplicates
    master_records_dict = {}
    
    # Function to process comparison pairs from a block
    def process_block_comparisons(block_indices, block_type):
        nonlocal all_comparison_pairs, master_records_dict
        
        if len(block_indices) < 2:  # Skip blocks with less than 2 records
            return
            
        for i1, i2 in itertools.combinations(block_indices, 2):
            # Skip if we've already compared this pair
            if (i1, i2) in all_comparison_pairs or (i2, i1) in all_comparison_pairs:
                continue
                
            # Add to tracked pairs
            all_comparison_pairs.add((i1, i2))
            
            r1 = work.loc[i1]
            r2 = work.loc[i2]
            
            # Calculate name similarity
            name_s = neo_token_set_ratio(r1[col_map["customer_name"]], r2[col_map["customer_name"]])
            
            # Skip if name similarity is too low
            if name_s < name_threshold:
                continue
            
            # Calculate address similarity
            addr_s = neo_token_set_ratio(r1[col_map["address"]], r2[col_map["address"]]) if col_map["address"] else 0
            
            # Calculate overall similarity - simplified to match duplicate_finder_app.py
            overall = round((name_s + addr_s) / 2)
            
            # Skip if overall similarity is too low
            if overall < overall_threshold:
                continue
            
            # Add to master records
            master_row = int(r1["ExcelRow"]) + 2  # +2 for 1-based Excel row and header
            if master_row not in master_records_dict:
                master_records_dict[master_row] = {
                    "duplicates": []
                }
            
            # Add duplicate
            dup = {
                "Row": int(r2["ExcelRow"]) + 2,  # +2 for 1-based Excel row and header
                "BlockType": block_type,
                "Overall": overall
            }
            master_records_dict[master_row]["duplicates"].append(dup)
    
    # Process blocks from each strategy
    for block_indices in blocks_prefix.values():
        process_block_comparisons(block_indices, "prefix")
    
    for block_indices in blocks_metaphone.values():
        process_block_comparisons(block_indices, "metaphone")
        
    for block_indices in blocks_soundex.values():
        process_block_comparisons(block_indices, "soundex")
    
    # Count master records and duplicates
    master_count = len(master_records_dict)
    duplicate_count = sum(len(m["duplicates"]) for m in master_records_dict.values())
    
    return master_count, duplicate_count

def normalize(text):
    """Normalize text for comparison"""
    import re
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def neo_token_set_ratio(a, b):
    """Calculate similarity between two strings"""
    from thefuzz import fuzz
    return fuzz.token_set_ratio(a, b)