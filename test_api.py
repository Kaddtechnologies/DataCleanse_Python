import requests
import json

url = "http://localhost:8000/deduplicate/"

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

# Send the request
response = requests.post(url, files=files)

# Print the response
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")