"""
FastAPI Main Application
-----------------------
This is the main application file that defines the FastAPI app and endpoints.
It integrates all the modules to provide a RESTful API for data deduplication.
"""

import os
import io
import time
import pandas as pd
import jellyfish
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Optional, Any
import logging

from src.models.data_models import (
    DeduplicationColumnMap, 
    DuplicateRecordDetail,
    MasterRecord,
    DeduplicationStats,
    DeduplicationResponse
)
from src.core.deduplication import build_duplicate_df

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Cleansing API",
    description="API for finding and managing duplicate records in datasets",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    """Root endpoint that returns a simple health check message."""
    return {"message": "Data Cleansing API is running!"}

@app.post("/api/find-duplicates", response_model=DeduplicationResponse)
async def find_duplicates(
    file: UploadFile = File(...),
    customer_name_column: Optional[str] = Form(None),
    address_column: Optional[str] = Form(None),
    city_column: Optional[str] = Form(None),
    country_column: Optional[str] = Form(None),
    tpi_column: Optional[str] = Form(None),
    use_prefix: bool = Form(True),
    use_metaphone: bool = Form(False),
    use_soundex: bool = Form(False),
    use_ngram: bool = Form(False),
    use_ai: bool = Form(False),
    name_threshold: int = Form(70),
    overall_threshold: int = Form(70),
    max_records: Optional[int] = Form(None)
):
    """
    Process an uploaded file to find duplicate records.
    
    This endpoint accepts CSV or Excel files and identifies potential duplicate records
    based on the specified column mappings and similarity thresholds. It supports
    multiple blocking strategies and can use AI for confidence scoring.
    
    Args:
        file: The CSV or Excel file to process
        customer_name_column: Column name containing customer/company names
        address_column: Column name containing addresses
        city_column: Column name containing cities
        country_column: Column name containing countries
        tpi_column: Column name containing TPI (third-party identifier)
        use_prefix: Whether to use prefix blocking
        use_metaphone: Whether to use metaphone phonetic blocking
        use_soundex: Whether to use soundex phonetic blocking
        use_ngram: Whether to use n-gram blocking
        use_ai: Whether to use AI for confidence scoring
        name_threshold: Minimum threshold for name similarity (0-100)
        overall_threshold: Minimum threshold for overall similarity (0-100)
        max_records: Maximum number of master records to process (for testing)
        
    Returns:
        DeduplicationResponse: Object containing the deduplication results and statistics
        
    Raises:
        HTTPException: If there are errors processing the file or finding duplicates
    """
    try:
        start_time = time.time()
        
        # Log the request parameters
        logger.info(f"Received deduplication request with parameters: "
                   f"name_col={customer_name_column}, "
                   f"addr_col={address_column}, "
                   f"city_col={city_column}, "
                   f"country_col={country_column}, "
                   f"tpi_col={tpi_column}, "
                   f"use_prefix={use_prefix}, "
                   f"use_metaphone={use_metaphone}, "
                   f"use_soundex={use_soundex}, "
                   f"use_ngram={use_ngram}, "
                   f"use_ai={use_ai}, "
                   f"name_threshold={name_threshold}, "
                   f"overall_threshold={overall_threshold}")
        
        # Check if at least customer name column is provided
        if not customer_name_column:
            raise HTTPException(
                status_code=400,
                detail="At least customer_name_column must be provided"
            )
        
        # Create the column mapping object
        col_map = DeduplicationColumnMap(
            customer_name=customer_name_column,
            address=address_column,
            city=city_column,
            country=country_column,
            tpi=tpi_column
        )
        
        # Read the uploaded file into a pandas DataFrame
        file_content = await file.read()
        file_obj = io.BytesIO(file_content)
        
        # Try to determine the file type by extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_obj)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_obj)
            else:
                # If we can't determine from extension, try CSV first, then Excel
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj)
                except Exception:
                    file_obj.seek(0)
                    df = pd.read_excel(file_obj)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading file: {str(e)}"
            )
        
        # Log the number of records loaded
        logger.info(f"Loaded {len(df)} records from {file.filename}")
        
        # Convert all column names to strings if they aren't already
        df.columns = df.columns.astype(str)
        
        # Call the deduplication function
        results_df, block_stats = build_duplicate_df(
            df=df,
            col_map=col_map,
            use_prefix=use_prefix,
            use_metaphone=use_metaphone,
            use_soundex=use_soundex,
            use_ngram=use_ngram,
            use_ai=use_ai,
            name_threshold=name_threshold,
            overall_threshold=overall_threshold,
            max_records=max_records
        )
        
        # Log the number of master records and total duplicates found
        if not results_df.empty:
            total_dupes = results_df['DuplicateCount'].sum()
            logger.info(f"Found {len(results_df)} master records with {total_dupes} total duplicates")
        else:
            logger.info("No duplicates found")
        
        # Convert results DataFrame to MasterRecord Pydantic models
        master_records = []
        
        for _, row in results_df.iterrows():
            # Convert each duplicate record to DuplicateRecordDetail
            duplicates = [
                DuplicateRecordDetail(
                    Row=dup.get("Row"),
                    Name=dup.get("Name"),
                    Address=dup.get("Address"),
                    City=dup.get("City"),
                    Country=dup.get("Country"),
                    TPI=dup.get("TPI"),
                    Name_score=dup.get("Name_score"),
                    Addr_score=dup.get("Addr_score"),
                    City_score=dup.get("City_score"),
                    Country_score=dup.get("Country_score"),
                    TPI_score=dup.get("TPI_score"),
                    Overall_score=dup.get("Overall_score"),
                    IsLowConfidence=dup.get("IsLowConfidence", True),
                    BlockType=dup.get("BlockType"),
                    MatchMethod=dup.get("MatchMethod"),
                    BestNameMatchMethod=dup.get("BestNameMatchMethod"),
                    BestAddrMatchMethod=dup.get("BestAddrMatchMethod"),
                    LLM_conf=dup.get("LLM_conf"),
                    uid=dup.get("uid")
                )
                for dup in row['Duplicates']
            ]
            
            # Create MasterRecord model
            master_record = MasterRecord(
                MasterRow=row['MasterRow'],
                MasterName=row['MasterName'],
                MasterAddress=row['MasterAddress'],
                MasterCity=row.get('MasterCity'),
                MasterCountry=row.get('MasterCountry'),
                MasterTPI=row.get('MasterTPI'),
                DuplicateCount=row['DuplicateCount'],
                AvgSimilarity=row['AvgSimilarity'],
                IsLowConfidenceGroup=row['IsLowConfidenceGroup'],
                Duplicates=duplicates,
                master_uid=row['master_uid']
            )
            
            master_records.append(master_record)
        
        # Calculate statistics
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Create the statistics model
        stats = DeduplicationStats(
            total_records=len(df),
            master_records=len(master_records),
            total_duplicates=sum(mr.DuplicateCount for mr in master_records),
            duplicate_percentage=round(sum(mr.DuplicateCount for mr in master_records) / len(df) * 100, 1) if len(df) > 0 else 0,
            low_confidence_matches=sum(1 for mr in master_records if mr.IsLowConfidenceGroup),
            processing_time_seconds=round(elapsed_time, 2),
            used_prefix_blocking=use_prefix,
            used_metaphone_blocking=use_metaphone,
            used_soundex_blocking=use_soundex,
            used_ngram_blocking=use_ngram,
            used_ai_scoring=use_ai,
            block_count=block_stats.get("total_blocks", 0),
            max_block_size=block_stats.get("max_block_size", 0),
            avg_block_size=round(block_stats.get("avg_block_size", 0), 1),
            name_threshold=name_threshold,
            overall_threshold=overall_threshold
        )
        
        # Create the full response
        response = DeduplicationResponse(
            message="Deduplication completed successfully",
            stats=stats,
            results=master_records
        )
        
        return response
        
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Error in find_duplicates: {str(e)}\n{traceback.format_exc()}")
        
        # Return a more user-friendly error message
        raise HTTPException(
            status_code=500,
            detail=f"Error processing deduplication: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time()
    }

# If this file is run directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 