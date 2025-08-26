"""
Background Processing Module

Handles background processing of AI emails and JSON logging of results.
"""

import time
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import Optional

from .templates import TemplateType
from .email_generator import process_ai_dataframe


def log_ai_results_to_json(request_id: str, ai_results: dict, original_ai_rows: pd.DataFrame, processing_time: float, ai_uuid_mapping: Optional[dict] = None):
    """
    Log completed AI email results to structured JSON file
    
    Args:
        request_id: Unique identifier for the request
        ai_results: Dictionary of generated email results
        original_ai_rows: Original AI rows DataFrame
        processing_time: Time taken to process the AI emails
        ai_uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        # Prepare structured data
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "total_ai_emails": len(ai_results),
            "results": []
        }
        
        # Add each AI result
        for _, row in original_ai_rows.iterrows():
            # Use the same UUID that was used in the CSV placeholder
            row_uuid = ai_uuid_mapping.get(row.name) if ai_uuid_mapping else str(uuid.uuid4())
            
            result_data = {
                "uuid": row_uuid,  # Use the same UUID from CSV placeholder
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
                "template_type": str(row.get('template_type', '')),
                "generated_email": ai_results.get(row.name, "Error: No result generated"),
                "row_index": int(row.name) if pd.notna(row.name) else None
            }
            log_entry["results"].append(result_data)
        
        # Write to JSON file in root folder
        filename = f"ai_results_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"AI results logged to {filename}")
        
    except Exception as e:
        print(f"Error logging AI results: {str(e)}")


def log_ai_error_to_json(request_id: str, error: Exception, ai_rows: pd.DataFrame):
    """
    Log AI processing errors to JSON file
    
    Args:
        request_id: Unique identifier for the request
        error: The exception that occurred
        ai_rows: Original AI rows DataFrame
    """
    try:
        error_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "status": "failed",
            "total_ai_emails": len(ai_rows)
        }
        
        filename = f"ai_error_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(error_entry, f, indent=2, ensure_ascii=False)
        print(f"AI error logged to {filename}")
    except Exception as log_error:
        print(f"Failed to log AI error: {str(log_error)}")


async def process_ai_emails_background(request_id: str, ai_rows: pd.DataFrame, fallback_template_type: Optional[TemplateType] = None, ai_uuid_mapping: Optional[dict] = None):
    """
    Background processing of AI emails with JSON logging
    
    Args:
        request_id: Unique identifier for the request
        ai_rows: DataFrame containing AI rows to process
        fallback_template_type: Template type to use when row template_type is empty
        ai_uuid_mapping: Mapping of row indices to UUIDs from CSV placeholders
    """
    try:
        print(f"Starting background AI processing for request {request_id}")
        start_time = time.time()
        
        # Process AI emails
        ai_results = await process_ai_dataframe(ai_rows, fallback_template_type)
        
        processing_time = time.time() - start_time
        print(f"Background AI processing completed for request {request_id} in {processing_time:.2f}s")
        
        # Log results to JSON
        log_ai_results_to_json(request_id, ai_results, ai_rows, processing_time, ai_uuid_mapping)
        
    except Exception as e:
        print(f"Background AI processing failed for request {request_id}: {str(e)}")
        
        # Log error to JSON
        log_ai_error_to_json(request_id, e, ai_rows)


def create_ai_placeholders(ai_rows: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create UUID placeholders for AI emails and return mapping
    
    Args:
        ai_rows: DataFrame containing AI rows
        
    Returns:
        Tuple of (ai_placeholders dict, ai_uuid_mapping dict)
    """
    ai_placeholders = {}
    ai_uuid_mapping = {}
    
    if not ai_rows.empty:
        print(f"Adding placeholder UUIDs for {len(ai_rows)} AI emails...")
        for _, row in ai_rows.iterrows():
            placeholder_uuid = str(uuid.uuid4())
            ai_placeholders[row.name] = f"AI_PROCESSING:{placeholder_uuid}"
            ai_uuid_mapping[row.name] = placeholder_uuid  # Store for JSON logging
    
    return ai_placeholders, ai_uuid_mapping
