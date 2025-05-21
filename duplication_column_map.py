from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any

class DeduplicationColumnMap(BaseModel):
    """
    Defines the mapping from logical field names (used by the deduplication algorithm)
    to the actual column header names in the uploaded file.
    """
    customer_name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    tpi: Optional[str] = None  # Tax Payer ID or similar unique identifier

    # Ensure at least one mapping is provided, typically customer_name
    @field_validator('*', pre=True, check_fields=False)
    def check_at_least_one_field(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not any(values.values()): # Checks if all mapped values are None or empty
            # This validation might be too strict if we allow processing with e.g. only TPI.
            # For now, we assume at least one field relevant to fuzzy matching should be mapped.
            # If customer_name is crucial, a specific validator for it might be better.
            pass # Relaxing this for now, endpoint logic will handle missing crucial fields.
        return values

class DeduplicationStats(BaseModel):
    """
    Statistics about the deduplication process.
    """
    high_confidence_duplicates: int
    medium_confidence_duplicates: int
    low_confidence_duplicates: int # Renamed from needs_ai
    block_stats: Dict[str, Any]

class DuplicateRecordDetail(BaseModel):
    """
    Detailed information for a single potential duplicate record.
    """
    Row: int
    Name: Optional[str] = None
    Address: Optional[str] = None
    Name_score: Optional[int] = None # Using underscore for consistency with potential direct df_to_dict
    Addr_score: Optional[int] = None
    Overall_score: int
    IsLowConfidence: bool # Was NeedsAI
    LLM_conf: Optional[float] = None # Keeping for structure, but will be None
    uid: str

class MasterRecord(BaseModel):
    """
    Represents a master record and its potential duplicates.
    """
    MasterRow: int
    MasterName: Optional[str] = None
    MasterAddress: Optional[str] = None
    DuplicateCount: int
    AvgSimilarity: float
    IsLowConfidenceGroup: bool # Was NeedsAI, indicates if any duplicate in the group is low confidence
    Duplicates: list[DuplicateRecordDetail]
    master_uid: str

class DeduplicationResponse(BaseModel):
    """
    The overall response structure for the deduplication endpoint.
    """
    duplicate_group_count: int
    total_potential_duplicates: int
    duplicates: list[MasterRecord]
    stats: DeduplicationStats