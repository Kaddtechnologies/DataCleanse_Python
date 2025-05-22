# DataCleansing API

A modular FastAPI application for finding and managing duplicate records in datasets.

## Project Structure

The application has been organized into a modular structure for better maintainability:

```
DataCleansing/
├── app.py                  # Entry point that maintains backward compatibility
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── src/                    # Source code directory
    ├── __init__.py         # Package initialization
    ├── main.py             # FastAPI app and endpoints
    ├── models/             # Data models
    │   ├── __init__.py
    │   └── data_models.py  # Pydantic models for validation
    ├── utils/              # Utility functions
    │   ├── __init__.py
    │   ├── text_processing.py  # Text normalization utilities
    │   ├── fuzzy_matching.py   # String similarity functions
    │   └── ai_scoring.py       # AI confidence scoring
    └── core/               # Core business logic
        ├── __init__.py
        └── deduplication.py    # Main deduplication algorithm
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd DataCleansing
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the API

Start the API server:

```
python app.py
```

The API will be available at http://localhost:8000.

### API Endpoints

- `GET /`: Root endpoint with a health check message
- `GET /api/health`: Health check endpoint
- `POST /api/find-duplicates`: Main endpoint for processing files and finding duplicates

### Example API Request

```python
import requests

url = "http://localhost:8000/api/find-duplicates"
files = {"file": open("sample_data.csv", "rb")}
form_data = {
    "customer_name_column": "CompanyName",
    "address_column": "Address",
    "city_column": "City",
    "country_column": "Country",
    "use_prefix": "true",
    "name_threshold": "70",
    "overall_threshold": "70"
}

response = requests.post(url, files=files, data=form_data)
result = response.json()
```

## Development

### Adding New Features

1. Identify the appropriate module for your feature
2. Implement the feature with proper documentation
3. Update tests if applicable
4. Update the README if necessary

## License

[Specify your license here]