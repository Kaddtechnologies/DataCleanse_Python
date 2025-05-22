"""
Core Deduplication Logic
-----------------------
This module contains the core logic for finding duplicate records in a dataset.
It implements multiple blocking strategies and similarity calculations.
"""

import uuid
import itertools
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Set

from src.models.data_models import DeduplicationColumnMap
from src.utils.text_processing import normalize, generate_ngrams
from src.utils.fuzzy_matching import neo_token_set_ratio, calculate_similarity_scores, get_best_similarity_score
from src.utils.ai_scoring import apply_ai_confidence_scoring

def build_duplicate_df(
    df: pd.DataFrame,
    col_map: DeduplicationColumnMap,
    use_prefix: bool = True,
    use_metaphone: bool = False,
    use_soundex: bool = False,
    use_ngram: bool = False,
    use_ai: bool = False,
    name_threshold: int = 70,
    overall_threshold: int = 70,
    max_records: int = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enhanced deduplication with configurable blocking strategies.
    
    This function processes a DataFrame to find potential duplicate records using
    different blocking strategies to efficiently compare similar records. It supports
    multiple matching techniques and can use AI to enhance confidence scoring.
    
    Args:
        df: Input DataFrame containing records to deduplicate
        col_map: Mapping between logical column names and actual DataFrame columns
        use_prefix: Whether to use prefix blocking (first 4 chars of name + first char of city)
        use_metaphone: Whether to use metaphone blocking (phonetic encoding)
        use_soundex: Whether to use soundex blocking (phonetic encoding)
        use_ngram: Whether to use n-gram blocking (character n-grams)
        use_ai: Whether to use AI for confidence scoring
        name_threshold: Minimum threshold for name similarity (0-100)
        overall_threshold: Minimum threshold for overall similarity (0-100)
        max_records: Maximum number of master records to process (for testing)
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame of master records with duplicates and block statistics
        
    Raises:
        ValueError: If mapped columns are not found in the DataFrame
    """
    # Validate that essential columns from col_map exist in df.columns
    mapped_cols_from_user_file = [val for val in col_map.model_dump().values() if val is not None]
    for user_col_name in mapped_cols_from_user_file:
        if user_col_name not in df.columns:
            raise ValueError(f"Mapped column '{user_col_name}' not found in the uploaded file. Available columns: {df.columns.tolist()}")

    # Create a working copy with only the necessary mapped columns + original index
    cols_to_select = [user_col_name for user_col_name in mapped_cols_from_user_file if user_col_name in df.columns]
    
    if not cols_to_select:
        raise ValueError("No valid columns were mapped or found in the uploaded file for deduplication.")

    work = df[cols_to_select].copy()
    work["ExcelRow"] = df.index  # Keep original row index (0-based)

    # Normalize relevant fields
    logical_fields_for_norm = ["customer_name", "address", "city", "country", "tpi"]
    for logical_field in logical_fields_for_norm:
        user_column_name = getattr(col_map, logical_field, None)
        if user_column_name and user_column_name in work.columns:
            work[f"{logical_field}_norm"] = work[user_column_name].apply(normalize)
            
            # Generate phonetic codes for name fields if needed
            if logical_field == "customer_name" and (use_metaphone or use_soundex):
                if use_metaphone:
                    # Metaphone encoding (better for company names)
                    work["customer_name_metaphone"] = work[user_column_name].apply(
                        lambda x: jellyfish.metaphone(str(x)) if pd.notna(x) else ""
                    )
                if use_soundex:
                    # Soundex encoding (alternative phonetic algorithm)
                    work["customer_name_soundex"] = work[user_column_name].apply(
                        lambda x: jellyfish.soundex(str(x)) if pd.notna(x) else ""
                    )
        else:
            # Ensure the normalized column exists even if not mapped
            work[f"{logical_field}_norm"] = ""
            if logical_field == "customer_name":
                if use_metaphone:
                    work["customer_name_metaphone"] = ""
                if use_soundex:
                    work["customer_name_soundex"] = ""

    # --- MULTIPLE BLOCKING STRATEGIES ---
    # 1. Traditional prefix blocking (first 4 chars of name + first char of city)
    blocks_prefix = {}
    # 2. Phonetic blocking using Metaphone
    blocks_metaphone = {}
    # 3. Soundex blocking
    blocks_soundex = {}
    # 4. N-gram blocking (character 3-grams from name)
    blocks_ngram = {}
    
    # Track all potential comparison pairs to avoid duplicates
    all_comparison_pairs = set()
    
    # Generate blocks using different strategies
    for i, row in work.iterrows():
        # 1. Prefix blocking
        if use_prefix:
            name_prefix = row["customer_name_norm"][:4] if len(row["customer_name_norm"]) >= 4 else row["customer_name_norm"]
            city_prefix = row["city_norm"][0] if len(row["city_norm"]) > 0 else ""
            block_key = f"{name_prefix}_{city_prefix}"
            
            if block_key not in blocks_prefix:
                blocks_prefix[block_key] = []
            blocks_prefix[block_key].append(i)
        
        # 2. Metaphone blocking
        if use_metaphone and "customer_name_metaphone" in work.columns:
            if pd.notna(row["customer_name_metaphone"]) and row["customer_name_metaphone"]:
                metaphone_key = row["customer_name_metaphone"]
                if metaphone_key not in blocks_metaphone:
                    blocks_metaphone[metaphone_key] = []
                blocks_metaphone[metaphone_key].append(i)
        
        # 3. Soundex blocking
        if use_soundex and "customer_name_soundex" in work.columns:
            if pd.notna(row["customer_name_soundex"]) and row["customer_name_soundex"]:
                soundex_key = row["customer_name_soundex"]
                if soundex_key not in blocks_soundex:
                    blocks_soundex[soundex_key] = []
                blocks_soundex[soundex_key].append(i)
        
        # 4. N-gram blocking
        if use_ngram:
            if len(row["customer_name_norm"]) > 0:
                # Generate character 3-grams from the name
                ngrams = generate_ngrams(row["customer_name_norm"], 3)
                for ngram in ngrams:
                    if ngram not in blocks_ngram:
                        blocks_ngram[ngram] = []
                    blocks_ngram[ngram].append(i)
    
    # Block statistics
    block_stats = {
        "total_blocks": len(blocks_prefix) + len(blocks_metaphone) + len(blocks_soundex) + len(blocks_ngram),
        "max_block_size": max(
            [max(map(len, blocks.values())) if blocks else 0 
             for blocks in [blocks_prefix, blocks_metaphone, blocks_soundex, blocks_ngram]
             if blocks]
        ) if any([blocks_prefix, blocks_metaphone, blocks_soundex, blocks_ngram]) else 0,
        "avg_block_size": (
            sum(map(len, blocks_prefix.values())) + 
            sum(map(len, blocks_metaphone.values())) + 
            sum(map(len, blocks_soundex.values())) + 
            sum(map(len, blocks_ngram.values()))
        ) / (
            len(blocks_prefix) + len(blocks_metaphone) + len(blocks_soundex) + len(blocks_ngram)
        ) if (len(blocks_prefix) + len(blocks_metaphone) + len(blocks_soundex) + len(blocks_ngram)) > 0 else 0,
        "records_in_blocks": (
            sum(map(len, blocks_prefix.values())) + 
            sum(map(len, blocks_metaphone.values())) + 
            sum(map(len, blocks_soundex.values())) + 
            sum(map(len, blocks_ngram.values()))
        ),
        "prefix_blocks": len(blocks_prefix),
        "metaphone_blocks": len(blocks_metaphone),
        "soundex_blocks": len(blocks_soundex),
        "ngram_blocks": len(blocks_ngram)
    }
    
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
            name_s = 0
            if col_map.customer_name:
                name_scores = calculate_similarity_scores(r1[col_map.customer_name], r2[col_map.customer_name])
                name_s, best_name_method = get_best_similarity_score(name_scores)
            
            # Skip if name similarity is too low
            if name_s < name_threshold:
                continue
            
            # Calculate address similarity
            addr_s = 0
            best_addr_method = "none"
            if col_map.address:
                addr_scores = calculate_similarity_scores(r1[col_map.address], r2[col_map.address])
                addr_s, best_addr_method = get_best_similarity_score(addr_scores)
            
            # Calculate city similarity
            city_s = 0
            best_city_method = "none"
            if col_map.city and col_map.city in r1 and col_map.city in r2:
                city_scores = calculate_similarity_scores(r1[col_map.city], r2[col_map.city])
                city_s, best_city_method = get_best_similarity_score(city_scores)
            
            # Calculate country similarity
            country_s = 0
            best_country_method = "none"
            if col_map.country and col_map.country in r1 and col_map.country in r2:
                country_scores = calculate_similarity_scores(r1[col_map.country], r2[col_map.country])
                country_s, best_country_method = get_best_similarity_score(country_scores)
            
            # Calculate TPI similarity
            tpi_s = 0
            best_tpi_method = "none"
            if col_map.tpi and col_map.tpi in r1 and col_map.tpi in r2:
                tpi_scores = calculate_similarity_scores(r1[col_map.tpi], r2[col_map.tpi])
                tpi_s, best_tpi_method = get_best_similarity_score(tpi_scores)
            
            # Calculate overall similarity - simplified to match duplicate_finder_app.py
            # Just average the name and address scores
            overall = round((name_s + addr_s) / 2)
            
            # Skip if overall similarity is too low
            if overall < overall_threshold:
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
            
            # Construct duplicate record detail
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
                "LLM_conf": None,  # Will be filled in by AI if enabled
                "uid": str(uuid.uuid4())
            }
            
            master_row_excel_num = int(r1["ExcelRow"]) + 2  # 1-based for Excel +1 for header
            
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
            
            # Add duplicate to master record
            master_records_dict[master_row_excel_num]["Duplicates"].append(dup_detail)
    
    # Process blocks from each strategy
    if use_prefix:
        for block_indices in blocks_prefix.values():
            process_block_comparisons(block_indices, "prefix")
    
    if use_metaphone:
        for block_indices in blocks_metaphone.values():
            process_block_comparisons(block_indices, "metaphone")
    
    if use_soundex:
        for block_indices in blocks_soundex.values():
            process_block_comparisons(block_indices, "soundex")
    
    if use_ngram:
        # For n-gram blocks, we need to be more selective as they can be very large
        for ngram, block_indices in blocks_ngram.items():
            # Only process n-gram blocks of reasonable size to avoid combinatorial explosion
            if 2 <= len(block_indices) <= 50:  # Skip very large blocks
                process_block_comparisons(block_indices, "ngram")
    
    # Convert dictionary of master records to a list of MasterRecord-like dicts
    masters_list = []
    for m_data in master_records_dict.values():
        sims = [d["Overall_score"] for d in m_data["Duplicates"]]
        avg_sim = round(sum(sims) / len(sims)) if sims else 0
        
        masters_list.append({
            "MasterRow": m_data["MasterRow"],
            "MasterName": m_data["MasterName"],
            "MasterAddress": m_data["MasterAddress"],
            "MasterCity": m_data.get("MasterCity"),
            "MasterCountry": m_data.get("MasterCountry"),
            "MasterTPI": m_data.get("MasterTPI"),
            "DuplicateCount": len(m_data["Duplicates"]),
            "AvgSimilarity": avg_sim,
            "IsLowConfidenceGroup": any(d["IsLowConfidence"] for d in m_data["Duplicates"]),
            "Duplicates": m_data["Duplicates"],
            "master_uid": m_data["master_uid"]
        })
    
    # Limit the number of master records if max_records is specified
    if max_records is not None and max_records > 0:
        print(f"Limiting to {max_records} master records for testing")
        masters_list = masters_list[:max_records]
    
    # Apply AI confidence scoring if enabled
    if use_ai and masters_list:
        masters_list = apply_ai_confidence_scoring(masters_list)
    
    # Create DataFrame from the list of master records
    # Sort by average similarity to bring more likely duplicates to the top
    results_df = pd.DataFrame(masters_list)
    if not results_df.empty:
        results_df = results_df.sort_values("AvgSimilarity", ascending=False).reset_index(drop=True)
    
    return results_df, block_stats 