"""
Test script for the Duplicate Finder API
This script demonstrates how to use the API endpoints
"""
import requests
import json
import time
import os
import sys

# Configuration
API_URL = "http://localhost:8000"
TEST_FILE = "sample_data.csv"  # Replace with your test file

def print_step(step_number, description):
    """Print a formatted step header"""
    print(f"\n{'='*80}")
    print(f"STEP {step_number}: {description}")
    print(f"{'='*80}")

def check_api_running():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to API at {API_URL}")
        return False

def upload_file(file_path):
    """Upload a file to the API"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"Uploading file: {file_path}")
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        response = requests.post(f"{API_URL}/upload", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ File uploaded successfully")
        print(f"   File ID: {result['file_id']}")
        print(f"   Row count: {result['row_count']}")
        print(f"   Detected columns: {json.dumps(result['detected_columns'], indent=2)}")
        return result
    else:
        print(f"❌ Failed to upload file: {response.text}")
        return None

def set_column_mapping(file_id, column_map):
    """Set column mapping for the file"""
    print(f"Setting column mapping for file {file_id}")
    response = requests.post(
        f"{API_URL}/files/{file_id}/column-mapping",
        json=column_map
    )
    
    if response.status_code == 200:
        print(f"✅ Column mapping set successfully")
        return True
    else:
        print(f"❌ Failed to set column mapping: {response.text}")
        return False

def run_deduplication(file_id):
    """Run deduplication on the file"""
    print(f"Running deduplication for file {file_id}")
    response = requests.post(f"{API_URL}/files/{file_id}/deduplicate")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Deduplication completed successfully")
        print(f"   Result ID: {result['result_id']}")
        print(f"   Duplicate count: {result['duplicate_count']}")
        print(f"   Stats: {json.dumps(result['stats'], indent=2)}")
        return result
    else:
        print(f"❌ Failed to run deduplication: {response.text}")
        return None

def get_results(result_id):
    """Get deduplication results"""
    print(f"Getting results for {result_id}")
    response = requests.get(f"{API_URL}/results/{result_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Retrieved results successfully")
        print(f"   Timestamp: {result['timestamp']}")
        print(f"   Number of duplicates: {len(result['duplicates'])}")
        return result
    else:
        print(f"❌ Failed to get results: {response.text}")
        return None

def run_ai_analysis(result_id, rows):
    """Run AI analysis on selected rows"""
    print(f"Running AI analysis for {len(rows)} rows")
    response = requests.post(
        f"{API_URL}/results/{result_id}/ai-analysis",
        json={"rows": rows}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ AI analysis completed successfully")
        print(f"   Scores: {result['scores']}")
        print(f"   Summary: {result['summary']}")
        return result
    else:
        print(f"❌ Failed to run AI analysis: {response.text}")
        return None

def export_results(result_id, format_type):
    """Export results in the specified format"""
    print(f"Exporting results in {format_type} format")
    response = requests.get(f"{API_URL}/results/{result_id}/export/{format_type}")
    
    if response.status_code == 200:
        filename = f"export_results.{format_type}"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ Results exported successfully to {filename}")
        return filename
    else:
        print(f"❌ Failed to export results: {response.text}")
        return None

def main():
    """Main function to test the API"""
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = TEST_FILE
    
    print_step(0, "Checking if API is running")
    if not check_api_running():
        print("Please start the API with 'uvicorn app:app --reload' and try again")
        return
    
    print_step(1, "Uploading file")
    upload_result = upload_file(test_file)
    if not upload_result:
        return
    
    file_id = upload_result['file_id']
    
    print_step(2, "Setting column mapping")
    # Use detected columns or set manually
    column_map = upload_result['detected_columns']
    if not set_column_mapping(file_id, column_map):
        return
    
    print_step(3, "Running deduplication")
    dedup_result = run_deduplication(file_id)
    if not dedup_result:
        return
    
    result_id = dedup_result['result_id']
    
    print_step(4, "Getting results")
    results = get_results(result_id)
    if not results:
        return
    
    # Only run AI analysis if there are duplicates that need AI
    if results['duplicates'] and any(d.get('NeedsAI') for d in results['duplicates']):
        print_step(5, "Running AI analysis")
        # Prepare rows for AI analysis
        ai_rows = []
        for master in results['duplicates']:
            for dup in master.get('Duplicates', []):
                if dup.get('NeedsAI'):
                    ai_rows.append({
                        "Name1": master['MasterName'],
                        "Name2": dup['Name'],
                        "Addr1": master['MasterAddress'],
                        "Addr2": dup['Address'],
                        "Name%": dup['Name%'],
                        "Addr%": dup['Addr%'],
                        "Overall%": dup['Overall%'],
                        "uid": dup['uid']
                    })
                    # Limit to 5 rows for testing
                    if len(ai_rows) >= 5:
                        break
        
        if ai_rows:
            run_ai_analysis(result_id, ai_rows)
    
    print_step(6, "Exporting results")
    export_results(result_id, "csv")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()