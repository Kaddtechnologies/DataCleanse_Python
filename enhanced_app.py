from __future__ import annotations

import io
import json
import re
import itertools
import uuid
import collections
import os
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set

import pandas as pd
import jellyfish  # pip install jellyfish
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator # field_validator for newer Pydantic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assuming thefuzz is installed: pip install thefuzz python-levenshtein
from thefuzz import fuzz as _fuzz

# --- Pydantic Models ---
class DeduplicationColumnMap(BaseModel):
    customer_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    tpi: Optional[str] = None

    @field_validator('customer_name')
    def customer_name_must_be_provided(cls, v: Optional[str], values: Any) -> Optional[str]:
        return v

class DuplicateRecordDetail(BaseModel):
    Row: int
    Name: Optional[str] = None
    Address: Optional[str] = None
    City: Optional[str] = None
    Country: Optional[str] = None
    TPI: Optional[str] = None
    Name_score: Optional[int] = None
    Addr_score: Optional[int] = None
    City_score: Optional[int] = None
    Country_score: Optional[int] = None
    TPI_score: Optional[int] = None
    Overall_score: int
    IsLowConfidence: bool
    BlockType: Optional[str] = None
    MatchMethod: Optional[str] = None
    BestNameMatchMethod: Optional[str] = None
    BestAddrMatchMethod: Optional[str] = None
    LLM_conf: Optional[float] = None
    uid: str
class MasterRecord(BaseModel):
    MasterRow: int
    MasterName: Optional[str] = None
    MasterAddress: Optional[str] = None
    MasterCity: Optional[str] = None
    MasterCountry: Optional[str] = None
    MasterTPI: Optional[str] = None
    DuplicateCount: int
    AvgSimilarity: float
    IsLowConfidenceGroup: bool
    Duplicates: list[DuplicateRecordDetail]
    master_uid: str

class DeduplicationStats(BaseModel):
    high_confidence_duplicates_groups: int
    medium_confidence_duplicates_groups: int
    low_confidence_duplicates_groups: int
    block_stats: Dict[str, Any]
    total_master_records_with_duplicates: int
    total_potential_duplicate_records: int

class DeduplicationResponse(BaseModel):
    message: str
    results: Optional[Dict[str, Any]] = None
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

# --- AI Confidence Scoring ---
import asyncio
import aiohttp

async def apply_ai_confidence_scoring_async(master_records: List[Dict]) -> List[Dict]:
    """
    Apply AI-based confidence scoring to the master records asynchronously.
    Uses OpenAI's API to evaluate the confidence of each duplicate match.
    Processes all records and allows AI to update confidence levels based on its judgment.
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OpenAI API key not found. Skipping AI confidence scoring.")
        return master_records
    
    try:
        # Process all records, but prioritize those with uncertain confidence
        # Uncertain confidence: overall score between 70-90
        records_for_ai = []
        record_indices = []
        
        for i, master in enumerate(master_records):
            # Check if any duplicates have uncertain confidence scores
            uncertain_dups = [
                dup for dup in master['Duplicates']
                if 70 <= dup['Overall_score'] <= 90
            ]
            
            if uncertain_dups:
                records_for_ai.append(master)
                record_indices.append(i)
        
        if not records_for_ai:
            print("No records found that need AI scoring.")
            return master_records
            
        print(f"Found {len(records_for_ai)} master records with duplicates for AI scoring.")
        
        # Process in smaller batches to avoid exceeding token limits
        batch_size = 3  # Process 3 master records at a time
        
        # Create batches
        batches = []
        for batch_idx in range(0, len(records_for_ai), batch_size):
            batch = records_for_ai[batch_idx:batch_idx+batch_size]
            batches.append((batch_idx, batch))
        
        # Process batches asynchronously
        async def process_batch(batch_idx, batch):
            # Prepare data for the AI
            entries = []
            for master in batch:
                # Add master record
                master_entry = f"Master: {master['MasterName']} | {master['MasterAddress']}"
                entries.append(master_entry)
                
                # Add all duplicates, with focus on uncertain ones
                for dup in master['Duplicates']:
                    confidence_level = "uncertain" if 70 <= dup['Overall_score'] <= 90 else "high" if dup['Overall_score'] > 90 else "low"
                    dup_entry = f"Duplicate: {dup['Name']} | {dup['Address']} | Score: {dup['Overall_score']} | Confidence: {confidence_level}"
                    entries.append(dup_entry)
            
            # Create prompt for OpenAI - focused on evaluating and updating confidence
            prompt = """
            You are an AI data deduplication and scoring assistant with expertise in evaluating duplicate records.
            
            Given a list of data entries with potential duplicates, your task is to:
            1. For each duplicate entry, evaluate how likely it is a true duplicate of the master record.
            2. Assign a confidence score (between 0 and 1) that represents your assessment.
            3. Pay special attention to entries marked as "uncertain" confidence, but evaluate all entries.
            4. Consider name similarity, address similarity, and any other relevant factors.
            5. If you believe the existing score is incorrect (too high or too low), your score should reflect your best judgment.
            
            Guidelines for confidence scoring:
            - 0.9-1.0: Definite duplicate (nearly identical records)
            - 0.8-0.9: Very likely duplicate (minor variations but clearly the same entity)
            - 0.7-0.8: Probable duplicate (some differences but likely the same entity)
            - 0.5-0.7: Possible duplicate (significant differences, but could be the same entity)
            - 0.0-0.5: Unlikely to be a duplicate (major differences, likely different entities)
            
            Process the following data entries:
            
            {}
            
            Please output a JSON array with objects like:
            [
              {{
                "entry": "<duplicate entry>",
                "confidence": <float between 0 and 1>,
                "reasoning": "<brief explanation for your confidence score>"
              }},
              ...
            ]
            """.format("\n".join(entries))
            
            print(f"Processing batch {batch_idx//batch_size + 1}/{len(batches)}...")
            
            # Call OpenAI API with a timeout
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4.1-nano",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2
                        },
                        timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            ai_content = result["choices"][0]["message"]["content"]
                            
                            try:
                                # Parse AI response
                                ai_scores = json.loads(ai_content)
                                
                                # Update confidence scores in the filtered master records
                                score_index = 0
                                for master_idx, master in enumerate(batch):
                                    for dup_idx, dup in enumerate(master['Duplicates']):
                                        if score_index < len(ai_scores):
                                            # Add AI confidence score
                                            # Update the original master_records list using the stored indices
                                            original_idx = record_indices[batch_idx + master_idx]
                                            ai_conf = ai_scores[score_index]['confidence']
                                            reasoning = ai_scores[score_index].get('reasoning', 'No reasoning provided')
                                            
                                            # Update the confidence score
                                            master_records[original_idx]['Duplicates'][dup_idx]['LLM_conf'] = ai_conf
                                            master_records[original_idx]['Duplicates'][dup_idx]['LLM_reasoning'] = reasoning
                                            
                                            # Log the AI confidence score
                                            print(f"AI confidence for {dup['Name']}: {ai_conf} - {reasoning[:50]}...")
                                            
                                            score_index += 1
                                
                                return True
                            except Exception as e:
                                print(f"Error parsing AI response: {str(e)}")
                                return False
                        else:
                            response_text = await response.text()
                            print(f"Error calling OpenAI API: {response.status} - {response_text}")
                            return False
            except asyncio.TimeoutError:
                print(f"Timeout when calling OpenAI API for batch {batch_idx//batch_size + 1}. Skipping this batch.")
                return False
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                return False
        
        # Process all batches concurrently
        tasks = []
        for batch_idx, batch in batches:
            task = asyncio.create_task(process_batch(batch_idx, batch))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful batches
        successful_batches = sum(1 for result in results if result is True)
        print(f"AI confidence scoring completed for {successful_batches}/{len(batches)} batches.")
    
    except Exception as e:
        print(f"Error in AI confidence scoring: {str(e)}")
    
    return master_records

# Synchronous wrapper for the async function
def apply_ai_confidence_scoring(master_records: List[Dict]) -> List[Dict]:
    """
    Synchronous wrapper for the asynchronous AI confidence scoring function.
    """
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Warning: Event loop is already running. Using synchronous processing instead.")
            # Fall back to synchronous processing
            return process_records_synchronously(master_records)
        else:
            # Use the existing loop
            return loop.run_until_complete(apply_ai_confidence_scoring_async(master_records))
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(apply_ai_confidence_scoring_async(master_records))
        finally:
            loop.close()

def process_records_synchronously(master_records: List[Dict]) -> List[Dict]:
    """
    Process records synchronously as a fallback when asyncio can't be used.
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OpenAI API key not found. Skipping AI confidence scoring.")
        return master_records
    
    try:
        # Process all records, but prioritize those with uncertain confidence
        # Uncertain confidence: overall score between 70-90
        records_for_ai = []
        record_indices = []
        
        for i, master in enumerate(master_records):
            # Check if any duplicates have uncertain confidence scores
            uncertain_dups = [
                dup for dup in master['Duplicates']
                if 70 <= dup['Overall_score'] <= 90
            ]
            
            if uncertain_dups:
                records_for_ai.append(master)
                record_indices.append(i)
        
        if not records_for_ai:
            print("No records found that need AI scoring.")
            return master_records
            
        print(f"Found {len(records_for_ai)} master records with duplicates for AI scoring (synchronous mode).")
        
        # Process in smaller batches to avoid exceeding token limits
        batch_size = 3  # Process 3 master records at a time
        for batch_idx in range(0, len(records_for_ai), batch_size):
            batch = records_for_ai[batch_idx:batch_idx+batch_size]
            
            # Prepare data for the AI
            entries = []
            for master in batch:
                # Add master record
                master_entry = f"Master: {master['MasterName']} | {master['MasterAddress']}"
                entries.append(master_entry)
                
                # Add all duplicates, with focus on uncertain ones
                for dup in master['Duplicates']:
                    confidence_level = "uncertain" if 70 <= dup['Overall_score'] <= 90 else "high" if dup['Overall_score'] > 90 else "low"
                    dup_entry = f"Duplicate: {dup['Name']} | {dup['Address']} | Score: {dup['Overall_score']} | Confidence: {confidence_level}"
                    entries.append(dup_entry)
            
            # Create prompt for OpenAI - focused on evaluating and updating confidence
            prompt = """
            You are an AI data deduplication and scoring assistant with expertise in evaluating duplicate records.
            
            Given a list of data entries with potential duplicates, your task is to:
            1. For each duplicate entry, evaluate how likely it is a true duplicate of the master record.
            2. Assign a confidence score (between 0 and 1) that represents your assessment.
            3. Pay special attention to entries marked as "uncertain" confidence, but evaluate all entries.
            4. Consider name similarity, address similarity, and any other relevant factors.
            5. If you believe the existing score is incorrect (too high or too low), your score should reflect your best judgment.
            
            Guidelines for confidence scoring:
            - 0.9-1.0: Definite duplicate (nearly identical records)
            - 0.8-0.9: Very likely duplicate (minor variations but clearly the same entity)
            - 0.7-0.8: Probable duplicate (some differences but likely the same entity)
            - 0.5-0.7: Possible duplicate (significant differences, but could be the same entity)
            - 0.0-0.5: Unlikely to be a duplicate (major differences, likely different entities)
            
            Process the following data entries:
            
            {}
            
            Please output a JSON array with objects like:
            [
              {{
                "entry": "<duplicate entry>",
                "confidence": <float between 0 and 1>,
                "reasoning": "<brief explanation for your confidence score>"
              }},
              ...
            ]
            """.format("\n".join(entries))
            
            print(f"Processing batch {batch_idx//batch_size + 1}/{(len(records_for_ai) + batch_size - 1)//batch_size}...")
            
            # Call OpenAI API with a timeout
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4.1-nano",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2
                    },
                    timeout=30  # 30 second timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_content = result["choices"][0]["message"]["content"]
                    
                    try:
                        # Parse AI response
                        ai_scores = json.loads(ai_content)
                        
                        # Update confidence scores in the filtered master records
                        score_index = 0
                        for master_idx, master in enumerate(batch):
                            for dup_idx, dup in enumerate(master['Duplicates']):
                                if score_index < len(ai_scores):
                                    # Add AI confidence score
                                    # Update the original master_records list using the stored indices
                                    original_idx = record_indices[batch_idx + master_idx]
                                    ai_conf = ai_scores[score_index]['confidence']
                                    reasoning = ai_scores[score_index].get('reasoning', 'No reasoning provided')
                                    
                                    # Update the confidence score
                                    master_records[original_idx]['Duplicates'][dup_idx]['LLM_conf'] = ai_conf
                                    master_records[original_idx]['Duplicates'][dup_idx]['LLM_reasoning'] = reasoning
                                    
                                    # Log the AI confidence score
                                    print(f"AI confidence for {dup['Name']}: {ai_conf} - {reasoning[:50]}...")
                                    
                                    score_index += 1
                    except Exception as e:
                        print(f"Error parsing AI response: {str(e)}")
                else:
                    print(f"Error calling OpenAI API: {response.status_code} - {response.text}")
            except requests.exceptions.Timeout:
                print(f"Timeout when calling OpenAI API for batch {batch_idx//batch_size + 1}. Skipping this batch.")
            except Exception as e:
                print(f"Error in API call: {str(e)}")
    
    except Exception as e:
        print(f"Error in AI confidence scoring: {str(e)}")
    
    print(f"AI confidence scoring completed for {len(records_for_ai)} master records.")
    return master_records

# --- Core Deduplication Logic ---
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
    
    Parameters:
    - df: Input DataFrame
    - col_map: Column mapping
    - use_prefix: Whether to use prefix blocking
    - use_metaphone: Whether to use metaphone blocking
    - use_soundex: Whether to use soundex blocking
    - use_ngram: Whether to use n-gram blocking
    - use_ai: Whether to use AI for confidence scoring
    - name_threshold: Threshold for name similarity
    - overall_threshold: Threshold for overall similarity
    
    Returns:
    - DataFrame of master records with duplicates
    - Block statistics
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

# --- FastAPI App Setup ---
app = FastAPI(
    title="Enhanced Duplicate Finder API",
    description="Accepts a file and column mappings to find duplicates with configurable blocking strategies.",
    version="0.2.0"
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
    column_map_json: str = Form(..., description="JSON string of DeduplicationColumnMap Pydantic model."),
    use_prefix: bool = Query(True, description="Use prefix blocking strategy"),
    use_metaphone: bool = Query(False, description="Use metaphone blocking strategy"),
    use_soundex: bool = Query(False, description="Use soundex blocking strategy"),
    use_ngram: bool = Query(False, description="Use n-gram blocking strategy"),
    use_ai: bool = Query(False, description="Use AI for confidence scoring"),
    name_threshold: int = Query(70, description="Threshold for name similarity (0-100)"),
    overall_threshold: int = Query(70, description="Threshold for overall similarity (0-100)"),
    max_records: int = Query(None, description="Maximum number of master records to process (for testing)")
):
    """
    Processes an uploaded file with specified column mappings to find duplicates.
    Allows configuring which blocking strategies to use and whether to apply AI confidence scoring.
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

        # If no blocking strategy is selected, default to prefix
        if not any([use_prefix, use_metaphone, use_soundex, use_ngram]):
            use_prefix = True
            
        # Run deduplication logic with selected blocking strategies
        try:
            dup_df, block_stats_dict = build_duplicate_df(
                df_raw,
                column_map,
                use_prefix=use_prefix,
                use_metaphone=use_metaphone,
                use_soundex=use_soundex,
                use_ngram=use_ngram,
                use_ai=use_ai,
                name_threshold=name_threshold,
                overall_threshold=overall_threshold,
                max_records=max_records
            )
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
        
        # Filter out duplicates that are less than 100% confidence (AvgSimilarity < 100)
        potential_duplicates = [
            record for record in duplicates_list
            if record["AvgSimilarity"] < 100 or record["IsLowConfidenceGroup"]
        ]

        # Add KPI metrics that match the Streamlit app naming convention
        # Count groups by confidence level based on AvgSimilarity
        high_conf_count = 0
        medium_conf_count = 0
        low_conf_count = 0
        
        for group in duplicates_list:
            avg_sim = group.get("AvgSimilarity", 0)
            if avg_sim >= 90:
                high_conf_count += 1
            elif avg_sim >= 70:
                medium_conf_count += 1
            else:
                low_conf_count += 1
        
        kpi_metrics = {
            "auto_merge": high_conf_count,  # AvgSimilarity >= 90
            "needs_review": medium_conf_count,  # 70 <= AvgSimilarity < 90
            "needs_ai": low_conf_count,  # AvgSimilarity < 70
            "total_blocks": block_stats_dict.get("total_blocks", 0),
            "blocking_strategies": {
                "prefix": use_prefix,
                "metaphone": use_metaphone,
                "soundex": use_soundex,
                "ngram": use_ngram
            },
            "ai_enabled": use_ai
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
                "column_map": column_map.model_dump(), # Include the column mapping in the response
                "blocking_config": {
                    "use_prefix": use_prefix,
                    "use_metaphone": use_metaphone,
                    "use_soundex": use_soundex,
                    "use_ngram": use_ngram,
                    "use_ai": use_ai,
                    "name_threshold": name_threshold,
                    "overall_threshold": overall_threshold
                }
            }
        )

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        # Log the error in a real application: import logging; logging.exception("Deduplication error")
        import traceback
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        error_traceback = traceback.format_exc()
        
        # Get the file name and line number where the exception occurred
        fname = traceback.extract_tb(exc_tb)[-1][0]
        line_no = traceback.extract_tb(exc_tb)[-1][1]
        
        error_message = f"Error at {fname}:{line_no} - {str(e)}"
        print(f"Detailed error: {error_traceback}")
        
        return DeduplicationResponse(message=f"An unexpected error occurred: {error_message}", error=error_message)


@app.get("/")
async def root():
    return {
        "message": "Enhanced Duplicate Finder API is running. Use the /deduplicate/ endpoint to process files.",
        "features": {
            "blocking_strategies": ["prefix", "metaphone", "soundex", "ngram"],
            "ai_confidence_scoring": "Available with use_ai=True parameter"
        }
    }

# To run this app:
# 1. Save as enhanced_app.py
# 2. Install dependencies: pip install fastapi uvicorn pandas "python-multipart" "thefuzz[speedup]" openpyxl jellyfish python-dotenv requests
# 3. Run with uvicorn: uvicorn enhanced_app:app --reload