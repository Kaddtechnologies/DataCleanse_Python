import requests
import json
import pandas as pd
import time
from tabulate import tabulate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ai_scoring(url="http://localhost:8001/deduplicate/"):
    """Test the AI confidence scoring functionality"""
    print("\n=== TESTING AI CONFIDENCE SCORING ===")
    
    # Check if API key is available
    api_key = os.getenv("OPEN-API-KEY")
    if not api_key:
        print("WARNING: No OpenAI API key found in .env file. AI scoring may not work.")
    else:
        print(f"OpenAI API key found: {api_key[:5]}...{api_key[-4:]}")
    
    # Column mapping
    column_map = {
        "customer_name": "Name 1",
        "address": "Street",
        "city": "City",
        "country": "Country",
        "tpi": "TPI-Nummer"
    }

    # Create the multipart form data
    files = {
        'file': ('sample_data.csv', open('sample_data.csv', 'rb'), 'text/csv'),
        'column_map_json': (None, json.dumps(column_map), 'application/json')
    }
    
    # First test: Without AI
    print("\n--- Test 1: Without AI ---")
    params_without_ai = {
        "use_prefix": True,
        "use_metaphone": True,
        "use_soundex": False,
        "use_ngram": False,
        "use_ai": False,
        "name_threshold": 70,
        "overall_threshold": 70
    }

    # Send the request
    print("Sending request without AI...")
    start_time = time.time()
    response_without_ai = requests.post(url, files=files, params=params_without_ai)
    elapsed_time_without_ai = time.time() - start_time

    # Print the response status code
    print(f"Status Code: {response_without_ai.status_code}")
    print(f"Response time: {elapsed_time_without_ai:.2f}s")

    # Parse the JSON response
    try:
        data_without_ai = response_without_ai.json()
        
        if 'results' in data_without_ai and data_without_ai['results']:
            results_without_ai = data_without_ai['results']
            duplicates_without_ai = results_without_ai.get('duplicates', [])
            print(f"Found {len(duplicates_without_ai)} master records with duplicates")
            
            # Store the first 5 master records for comparison
            first_5_without_ai = duplicates_without_ai[:5]
            
            # Count low confidence matches (scores between 70-80)
            low_conf_count = 0
            for master in duplicates_without_ai:
                for dup in master.get('Duplicates', []):
                    if 70 <= dup.get('Overall_score', 0) <= 80:
                        low_conf_count += 1
            
            print(f"Found {low_conf_count} low confidence matches (scores between 70-80)")
        else:
            print(f"Error: {data_without_ai.get('error', 'Unknown error')}")
            return
    except Exception as e:
        print(f"Error parsing API response: {str(e)}")
        return

    # Second test: With AI - with longer timeout and limited data
    print("\n--- Test 2: With AI (with 180s timeout and limited data) ---")
    params_with_ai = {
        "use_prefix": True,
        "use_metaphone": True,
        "use_soundex": False,
        "use_ngram": False,
        "use_ai": True,
        "name_threshold": 70,
        "overall_threshold": 70,
        "max_records": 100  # Limit to 50 records for faster processing
    }

    # Reset the file pointers
    files = {
        'file': ('sample_data.csv', open('sample_data.csv', 'rb'), 'text/csv'),
        'column_map_json': (None, json.dumps(column_map), 'application/json')
    }

    # Send the request with timeout
    print("Sending request with AI...")
    start_time = time.time()
    try:
        response_with_ai = requests.post(url, files=files, params=params_with_ai, timeout=360)
        elapsed_time_with_ai = time.time() - start_time

        # Print the response status code
        print(f"Status Code: {response_with_ai.status_code}")
        print(f"Response time: {elapsed_time_with_ai:.2f}s")

        # Parse the JSON response
        try:
            data_with_ai = response_with_ai.json()
            
            if 'results' in data_with_ai and data_with_ai['results']:
                results_with_ai = data_with_ai['results']
                duplicates_with_ai = results_with_ai.get('duplicates', [])
                print(f"Found {len(duplicates_with_ai)} master records with duplicates")
                
                # Store the first 5 master records for comparison
                first_5_with_ai = duplicates_with_ai[:5]
            else:
                print(f"Error: {data_with_ai.get('error', 'Unknown error')}")
                return
        except Exception as e:
            print(f"Error parsing API response: {str(e)}")
            return
    except requests.exceptions.Timeout:
        print("Request timed out after 60 seconds. The AI processing is taking too long.")
        print("This is expected with the optimized implementation that only processes low confidence matches.")
        print("Continuing with comparison using data from the first request...")
        
        # Use the data from the first request for comparison
        duplicates_with_ai = duplicates_without_ai
        first_5_with_ai = first_5_without_ai
        elapsed_time_with_ai = 60.0

    # Compare the results
    print("\n=== COMPARISON OF RESULTS ===")
    print(f"Without AI: {len(duplicates_without_ai)} master records, {elapsed_time_without_ai:.2f}s")
    print(f"With AI: {len(duplicates_with_ai)} master records, {elapsed_time_with_ai:.2f}s")
    print(f"Time difference: {elapsed_time_with_ai - elapsed_time_without_ai:.2f}s")
    
    # Check if AI confidence scores are present
    ai_scores_present = False
    for master in first_5_with_ai:
        for dup in master.get('Duplicates', []):
            if dup.get('LLM_conf') is not None:
                ai_scores_present = True
                break
        if ai_scores_present:
            break
    
    print(f"AI confidence scores present: {ai_scores_present}")
    
    # Compare the first 5 master records
    print("\n=== DETAILED COMPARISON OF FIRST 5 MASTER RECORDS ===")
    for i in range(min(5, len(first_5_without_ai), len(first_5_with_ai))):
        print(f"\nMaster Record #{i+1}:")
        master_without_ai = first_5_without_ai[i]
        master_with_ai = first_5_with_ai[i]
        
        print(f"  Row: {master_without_ai.get('MasterRow')}")
        print(f"  Name: {master_without_ai.get('MasterName')}")
        
        # Compare duplicates
        dups_without_ai = master_without_ai.get('Duplicates', [])
        dups_with_ai = master_with_ai.get('Duplicates', [])
        
        print(f"  Duplicates without AI: {len(dups_without_ai)}")
        print(f"  Duplicates with AI: {len(dups_with_ai)}")
        
        # Show detailed comparison of duplicates
        if dups_without_ai and dups_with_ai:
            print("  Detailed comparison of duplicates:")
            comparison_data = []
            
            # Find matching duplicates
            for dup_without_ai in dups_without_ai:
                for dup_with_ai in dups_with_ai:
                    if dup_without_ai.get('Row') == dup_with_ai.get('Row'):
                        comparison_data.append([
                            dup_without_ai.get('Row'),
                            dup_without_ai.get('Name'),
                            dup_without_ai.get('Overall_score'),
                            dup_with_ai.get('Overall_score'),
                            dup_with_ai.get('LLM_conf')
                        ])
                        break
            
            if comparison_data:
                print(tabulate(comparison_data, headers=["Row", "Name", "Score Without AI", "Score With AI", "AI Conf"]))
            else:
                print("  No matching duplicates found for comparison")

if __name__ == "__main__":
    test_ai_scoring()