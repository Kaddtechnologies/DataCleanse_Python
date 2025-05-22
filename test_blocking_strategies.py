"""
Master script to test different blocking strategies and report results
"""
import pandas as pd
import json
from tabulate import tabulate
import time
import requests

# Import all implementation modules
import blocking_prefix_only
import blocking_prefix_metaphone
import blocking_prefix_metaphone_soundex
import blocking_soundex_only
import blocking_ngram_only
import blocking_prefix_AI
import blocking_all_strategies

# Column mapping - match the original duplicate_finder_app.py
column_map = {
    "customer_name": "Name 1",
    "address": "Street",
    "city": "City",
    "country": "Country",
    "tpi": "TPI-Nummer"
}

# Thresholds
name_threshold = 70
overall_threshold = 70

def test_local_implementations():
    """Test all local implementations and report results"""
    print("\n=== TESTING LOCAL IMPLEMENTATIONS ===")
    
    # Load the sample data
    df = pd.read_csv('sample_data.csv')
    print(f"Loaded {len(df)} rows from sample_data.csv")
    
    # Define implementations to test
    implementations = [
        ("Prefix Only", blocking_prefix_only),
        ("Prefix + Metaphone", blocking_prefix_metaphone),
        ("Prefix + Metaphone + Soundex", blocking_prefix_metaphone_soundex),
        ("Soundex Only", blocking_soundex_only),
        ("N-gram Only", blocking_ngram_only),
        ("Prefix + AI", blocking_prefix_AI),
        ("All Strategies", blocking_all_strategies)
    ]
    
    # Run each implementation and collect results
    results = []
    
    for name, implementation in implementations:
        start_time = time.time()
        master_count, duplicate_count = implementation.build_duplicate_df(
            df, column_map, name_threshold, overall_threshold
        )
        elapsed_time = time.time() - start_time
        
        results.append([
            name,
            master_count,
            duplicate_count,
            f"{elapsed_time:.2f}s"
        ])
    
    # Display results in a table
    print("\n=== LOCAL IMPLEMENTATION RESULTS ===")
    print(tabulate(results, headers=["Strategy", "Master Records", "Duplicates", "Time"]))

def test_api():
    """Test the API endpoint and report results"""
    print("\n=== TESTING API ENDPOINT ===")
    
    url = "http://localhost:8000/deduplicate/"
    
    # Create the multipart form data
    files = {
        'file': ('sample_data.csv', open('sample_data.csv', 'rb'), 'text/csv'),
        'column_map_json': (None, json.dumps(column_map), 'application/json')
    }
    
    # Parameters
    params = {
        "name_threshold": name_threshold,
        "overall_threshold": overall_threshold
    }
    
    try:
        # Send the request
        print("Sending request to API...")
        start_time = time.time()
        response = requests.post(url, files=files, params=params)
        elapsed_time = time.time() - start_time
        
        # Print the response status code
        print(f"Status Code: {response.status_code}")
        
        # Parse the JSON response
        data = response.json()
        
        if 'results' in data and data['results']:
            results = data['results']
            stats = results.get('stats', {})
            
            master_count = stats.get('total_master_records_with_duplicates', 0)
            duplicate_count = stats.get('total_potential_duplicate_records', 0)
            
            # Print block statistics
            block_stats = stats.get('block_stats', {})
            print("\n=== API BLOCKING STATISTICS ===")
            print(f"Total blocks: {block_stats.get('total_blocks', 0)}")
            print(f"Prefix blocks: {block_stats.get('prefix_blocks', 0)}")
            print(f"Metaphone blocks: {block_stats.get('metaphone_blocks', 0)}")
            print(f"Soundex blocks: {block_stats.get('soundex_blocks', 0)}")
            print(f"N-gram blocks: {block_stats.get('ngram_blocks', 0)}")
            
            # Add API results to the comparison table
            api_results = [["API Endpoint", master_count, duplicate_count, f"{elapsed_time:.2f}s"]]
            print("\n=== API RESULTS ===")
            print(tabulate(api_results, headers=["Strategy", "Master Records", "Duplicates", "Time"]))
        else:
            print(f"Error: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        print("Skipping API test.")

def run_simplified_implementation():
    """Run the simplified implementation from test_fixed_api.py"""
    print("\n=== RUNNING SIMPLIFIED IMPLEMENTATION ===")
    
    # Load the sample data
    df = pd.read_csv('sample_data.csv')
    print(f"Loaded {len(df)} rows from sample_data.csv")
    
    # Create a working copy with only the necessary mapped columns
    work = df[[col for col in column_map.values() if col]].copy().reset_index(drop=False)
    work = work.rename(columns={"index": "ExcelRow"})
    
    # Normalize relevant fields
    for field in ["customer_name", "address", "city", "country"]:
        if column_map[field]:
            work[f"{field}_norm"] = work[column_map[field]].apply(normalize)
    
    # Create blocks based on name prefix and city prefix
    blocks = {}
    for i, row in work.iterrows():
        name_prefix = row["customer_name_norm"][:4] if len(row["customer_name_norm"]) >= 4 else row["customer_name_norm"]
        city_prefix = row["city_norm"][0] if column_map["city"] and len(row["city_norm"]) > 0 else ""
        block_key = f"{name_prefix}_{city_prefix}"
        if block_key not in blocks:
            blocks[block_key] = []
        blocks[block_key].append(i)
    
    print(f"Created {len(blocks)} blocks")
    block_sizes = [len(block) for block in blocks.values()]
    print(f"Max block size: {max(block_sizes) if block_sizes else 0}")
    print(f"Avg block size: {sum(block_sizes)/len(blocks) if blocks else 0:.2f}")
    
    # Find duplicates
    master_records = {}
    total_comparisons = 0
    total_duplicates = 0
    
    start_time = time.time()
    
    for block_indices in blocks.values():
        for i1, i2 in [(a, b) for idx, a in enumerate(block_indices) for b in block_indices[idx+1:]]:
            total_comparisons += 1
            r1, r2 = work.loc[i1], work.loc[i2]
            
            # Calculate name similarity
            name_s = neo_token_set_ratio(r1[column_map["customer_name"]], r2[column_map["customer_name"]])
            if name_s < name_threshold:
                continue
                
            # Calculate address similarity
            addr_s = neo_token_set_ratio(r1[column_map["address"]], r2[column_map["address"]]) if column_map["address"] else 0
            
            # Calculate overall similarity
            overall = round((name_s + addr_s) / 2)
            if overall < overall_threshold:
                continue
                
            # Add to master records
            master_row = int(r1["ExcelRow"]) + 2  # +2 for 1-based Excel row and header
            if master_row not in master_records:
                master_records[master_row] = {
                    "duplicates": []
                }
            
            # Add duplicate
            dup = {
                "Row": int(r2["ExcelRow"]) + 2  # +2 for 1-based Excel row and header
            }
            master_records[master_row]["duplicates"].append(dup)
            total_duplicates += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"Made {total_comparisons} comparisons and found {total_duplicates} duplicates")
    print(f"Found {len(master_records)} master records with duplicates")
    print(f"Execution time: {elapsed_time:.2f}s")
    
    # Add simplified implementation results to the comparison table
    simplified_results = [["Simplified Implementation", len(master_records), total_duplicates, f"{elapsed_time:.2f}s"]]
    print("\n=== SIMPLIFIED IMPLEMENTATION RESULTS ===")
    print(tabulate(simplified_results, headers=["Strategy", "Master Records", "Duplicates", "Time"]))

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

if __name__ == "__main__":
    print("=== DEDUPLICATION BLOCKING STRATEGY COMPARISON ===")
    print(f"Name Threshold: {name_threshold}")
    print(f"Overall Threshold: {overall_threshold}")
    
    # Run all tests
    test_local_implementations()
    test_api()
    run_simplified_implementation()
    
    print("\n=== SUMMARY ===")
    print("If any strategy returns 0 duplicates while others find duplicates,")
    print("that strategy is likely causing the issue in the API.")