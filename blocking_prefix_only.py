"""
Implementation with only prefix blocking strategy
"""
import pandas as pd
import itertools
import uuid
from typing import Dict, Any, Tuple

def build_duplicate_df(df: pd.DataFrame, col_map: Dict[str, str], name_threshold: int = 70, overall_threshold: int = 70) -> Tuple[int, int]:
    """
    Deduplication with only prefix blocking strategy
    Returns: (master_count, duplicate_count)
    """
    # Create a working copy with only the necessary mapped columns
    work = df[[col for col in col_map.values() if col]].copy().reset_index(drop=False)
    work = work.rename(columns={"index": "ExcelRow"})
    
    # Normalize relevant fields
    for field in ["customer_name", "address", "city", "country"]:
        if col_map[field]:
            work[f"{field}_norm"] = work[col_map[field]].apply(normalize)
    
    # --- ONLY PREFIX BLOCKING STRATEGY ---
    blocks = {}
    
    for i, row in work.iterrows():
        name_prefix = row["customer_name_norm"][:4] if len(row["customer_name_norm"]) >= 4 else row["customer_name_norm"]
        city_prefix = row["city_norm"][0] if col_map["city"] and len(row["city_norm"]) > 0 else ""
        block_key = f"{name_prefix}_{city_prefix}"
        
        if block_key not in blocks:
            blocks[block_key] = []
        blocks[block_key].append(i)
    
    # Find duplicates
    master_records_dict = {}
    
    for block_indices in blocks.values():
        for i1, i2 in itertools.combinations(block_indices, 2):
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
                "Overall": overall
            }
            master_records_dict[master_row]["duplicates"].append(dup)
    
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