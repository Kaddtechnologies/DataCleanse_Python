"""
Data Models
-----------
This module contains all Pydantic models used for data validation and response structures.
These models define the expected data formats for inputs and outputs in the API.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, field_validator


class DeduplicationColumnMap(BaseModel):
    """
    Defines the mapping between logical column names and user-provided column names.
    This allows the application to work with different CSV/Excel formats by mapping
    the columns to standardized field names used internally.
    """
    customer_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    tpi: Optional[str] = None

    @field_validator('customer_name')
    def customer_name_must_be_provided(cls, v: Optional[str], values: Any) -> Optional[str]:
        """Validate that customer_name is provided (though it's technically optional)"""
        return v


class DuplicateRecordDetail(BaseModel):
    """
    Represents details about a single record that has been identified as a duplicate.
    Contains all fields from the original record along with similarity scores.
    """
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
    """
    Represents a master record with its detected duplicates.
    A master record is the primary record to which duplicates are compared.
    """
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
    """
    Statistics about the deduplication process, such as counts of 
    high, medium, and low confidence duplicate groups.
    """
    high_confidence_duplicates_groups: int
    medium_confidence_duplicates_groups: int
    low_confidence_duplicates_groups: int
    block_stats: Dict[str, Any]
    total_master_records_with_duplicates: int
    total_potential_duplicate_records: int


class DeduplicationResponse(BaseModel):
    """
    The response model for the deduplication API endpoint,
    containing the results and any error messages.
    """
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None 