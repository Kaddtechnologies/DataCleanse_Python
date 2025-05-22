# Enhanced Deduplication API

This enhanced version of the deduplication API provides flexible blocking strategies and AI-based confidence scoring to improve duplicate detection.

## Features

- **Multiple Blocking Strategies**:
  - **Prefix Blocking**: Uses the first 4 characters of name and first character of city
  - **Metaphone Blocking**: Uses phonetic encoding of names (better for company names)
  - **Soundex Blocking**: Alternative phonetic algorithm
  - **N-gram Blocking**: Uses character 3-grams from names

- **AI Confidence Scoring**:
  - Uses OpenAI's GPT-4.1-nano model to evaluate the confidence of each duplicate match
  - Adds an additional confidence score (0-1) to each duplicate

- **Configurable Thresholds**:
  - Adjust name similarity threshold (default: 70)
  - Adjust overall similarity threshold (default: 70)

## Getting Started

### Prerequisites

- Docker
- Python 3.8+
- Required Python packages: `fastapi`, `uvicorn`, `pandas`, `python-multipart`, `thefuzz`, `jellyfish`, `python-dotenv`, `requests`

### Running the API

#### Using Docker

1. Build and run the Docker container:
   ```
   ./build_enhanced_docker.ps1
   ```

2. The API will be available at `http://localhost:8001`

#### Running Locally

1. Install dependencies:
   ```
   pip install fastapi uvicorn pandas python-multipart thefuzz[speedup] openpyxl jellyfish python-dotenv requests
   ```

2. Run the API:
   ```
   uvicorn enhanced_app:app --reload
   ```

3. The API will be available at `http://localhost:8000`

### Testing the API

Use the provided test script to test different blocking strategies:

```
# Test with default settings (prefix blocking only)
python test_enhanced_api.py

# Test with metaphone blocking
python test_enhanced_api.py --metaphone

# Test with a combination of strategies
python test_enhanced_api.py --prefix --metaphone --soundex

# Test with all strategies and AI confidence scoring
python test_enhanced_api.py --all --ai

# Compare all blocking strategies
python test_enhanced_api.py --compare
```

## API Endpoint

### POST /deduplicate/

Processes an uploaded file with specified column mappings to find duplicates.

#### Parameters

- `file`: CSV or XLSX file to be deduplicated
- `column_map_json`: JSON string of column mappings
- `use_prefix`: Whether to use prefix blocking (default: true)
- `use_metaphone`: Whether to use metaphone blocking (default: false)
- `use_soundex`: Whether to use soundex blocking (default: false)
- `use_ngram`: Whether to use n-gram blocking (default: false)
- `use_ai`: Whether to use AI for confidence scoring (default: false)
- `name_threshold`: Threshold for name similarity (0-100, default: 70)
- `overall_threshold`: Threshold for overall similarity (0-100, default: 70)

#### Example Request

```bash
curl -X POST "http://localhost:8001/deduplicate/?use_prefix=true&use_metaphone=true&use_ai=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data.csv;type=text/csv" \
  -F "column_map_json={\"customer_name\":\"Name 1\",\"address\":\"Street\",\"city\":\"City\",\"country\":\"Country\",\"tpi\":\"TPI-Nummer\"}"
```

## Performance Considerations

- **Prefix Blocking**: Fastest, but may miss some duplicates
- **Metaphone Blocking**: Good balance of speed and accuracy for company names
- **Soundex Blocking**: Similar to metaphone, but with different phonetic encoding
- **N-gram Blocking**: Most comprehensive but slowest
- **AI Confidence Scoring**: Adds significant processing time but improves accuracy

## Integration with UI

The API is designed to work with a UI that allows users to select which blocking strategies to use and whether to enable AI confidence scoring. This gives users full control over the deduplication process.