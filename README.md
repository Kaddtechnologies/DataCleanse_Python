# Duplicate Finder API

A FastAPI application for customer duplicate detection, designed for deployment in Azure Container Instances.

## Features

- **File Upload**: Support for CSV and Excel files
- **Dynamic Column Mapping**: No fixed headers required
- **Fast Fuzzy Matching**: Using `thefuzz` with `python-levenshtein`
- **AI Analysis**: For potential duplicates using OpenAI
- **Export Functionality**: Export results as CSV or Excel

## Local Development

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```
   uvicorn app:app --reload
   ```
6. Access the API documentation at http://localhost:8000/docs

## API Endpoints

- `POST /upload`: Upload a CSV or Excel file
- `GET /files/{file_id}/columns`: Get columns and suggested mappings
- `POST /files/{file_id}/column-mapping`: Set column mapping
- `POST /files/{file_id}/deduplicate`: Run deduplication
- `GET /results/{result_id}`: Get deduplication results
- `POST /results/{result_id}/ai-analysis`: Run AI analysis
- `GET /results/{result_id}/export/{format}`: Export results (csv or excel)

## Docker

Build the Docker image:
```
docker build -t duplicate-finder-api .
```

Run the container:
```
docker run -p 8000:8000 -d duplicate-finder-api
```

## Deploying to Azure Container Instances

### Prerequisites

- Azure CLI
- Azure Container Registry (ACR) access

### Steps

1. Log in to Azure:
   ```
   az login
   ```

2. Build and push the Docker image to ACR:
   ```
   az acr build --registry <your-acr-name> --image duplicate-finder-api:latest .
   ```

3. Create an Azure Container Instance:
   ```
   az container create \
     --resource-group <your-resource-group> \
     --name duplicate-finder-api \
     --image <your-acr-name>.azurecr.io/duplicate-finder-api:latest \
     --dns-name-label duplicate-finder-api \
     --ports 8000 \
     --environment-variables OPENAI_API_KEY=<your-openai-api-key>
   ```

4. Get the FQDN of your container:
   ```
   az container show \
     --resource-group <your-resource-group> \
     --name duplicate-finder-api \
     --query ipAddress.fqdn \
     --output tsv
   ```

5. Access your API at `http://<fqdn>:8000`

## Using the API

1. Upload a file:
   ```
   curl -X POST -F "file=@your_file.csv" http://localhost:8000/upload
   ```

2. Get the file columns:
   ```
   curl http://localhost:8000/files/{file_id}/columns
   ```

3. Set column mapping:
   ```
   curl -X POST -H "Content-Type: application/json" \
     -d '{"customer_name": "Customer", "address": "Address"}' \
     http://localhost:8000/files/{file_id}/column-mapping
   ```

4. Run deduplication:
   ```
   curl -X POST http://localhost:8000/files/{file_id}/deduplicate
   ```

5. Get results:
   ```
   curl http://localhost:8000/results/{result_id}
   ```

6. Export results:
   ```
   curl http://localhost:8000/results/{result_id}/export/csv > results.csv
   ```

## License

This project is licensed under the MIT License.