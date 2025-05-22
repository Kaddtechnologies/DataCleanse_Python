"""
DataCleansing Application Entry Point
------------------------------------
This file serves as the entry point for the application, maintaining backward compatibility
with existing code while using the modular structure for better maintainability.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app from the main module
from src.main import app

# If this file is run directly, start the server
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default 8000
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting DataCleansing API on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)