from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import Optional
import pandas as pd
import io
import asyncio
import os
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import template functionality
from .templates import (
    TemplateType, 
    DEFAULT_TEMPLATE_TYPE, 
    get_all_templates,
    get_template_info
)

# Import modular components
from .csv_processor import (
    validate_and_enhance_csv,
    get_csv_info,
    split_dataframe_by_intelligence,
    validate_csv_size,
    validate_csv_not_empty
)
from .email_generator import (
    generate_single_email,
    process_template_dataframe
)
from .background_processor import (
    process_ai_emails_background,
    create_ai_placeholders,
    create_all_placeholders,
    process_all_emails_background
)
# Database imports
from .database.services import EmailRequestService, GeneratedEmailService
from .database.connection import get_database_manager


def create_database_records_and_placeholders(
    request_id: str,
    original_filename: str,
    df,
    template_rows,
    ai_rows,
    file_size_bytes: int,
    template_type: Optional[TemplateType] = None
) -> tuple[dict, dict]:
    """
    Create database records for the request and email placeholders
    
    For large datasets (>1000 rows), uses hybrid approach:
    1. Create request record immediately 
    2. Generate UUID placeholders in memory
    3. Defer bulk email record creation to background
    
    Returns:
        Tuple of (placeholders dict, uuid_mapping dict)
    """
    try:
        # Always create the email request record immediately
        EmailRequestService.create_request(
            request_id=request_id,
            original_filename=original_filename,
            total_rows=len(df),
            template_llm_rows=len(template_rows),
            ai_research_rows=len(ai_rows),
            file_size_bytes=file_size_bytes,
            fallback_template_type=template_type
        )
        
        # For large datasets, use hybrid approach
        dataset_size = len(df)
        if dataset_size > 1000:
            print(f"Large dataset detected ({dataset_size} rows). Using hybrid approach:")
            print("- Request record created immediately")
            print("- Email placeholders generated in memory")
            print("- Database email records will be bulk-created in background")
            
            # Generate UUIDs in memory without database operations
            all_placeholders = {}
            all_uuid_mapping = {}
            
            for _, row in df.iterrows():
                # Determine processing type
                intelligence_used = bool(row.get('intelligence', False))
                processing_prefix = "AI_PROCESSING" if intelligence_used else "TEMPLATE_LLM_PROCESSING"
                
                # Generate UUID in memory
                import uuid
                placeholder_uuid = str(uuid.uuid4())
                
                all_placeholders[row.name] = f"{processing_prefix}:{placeholder_uuid}"
                all_uuid_mapping[row.name] = placeholder_uuid
            
            # Schedule background bulk database creation
            print(f"Scheduling background bulk creation of {dataset_size} email records...")
            asyncio.create_task(
                _bulk_create_email_records_background(request_id, df, all_uuid_mapping)
            )
            
            return all_placeholders, all_uuid_mapping
        
        else:
            # For smaller datasets, use original synchronous approach
            print(f"Small dataset ({dataset_size} rows). Using synchronous database creation.")
            return _create_email_records_synchronous(request_id, df)
        
    except Exception as e:
        print(f"Warning: Failed to create database records: {e}")
        print("Falling back to memory-only placeholders...")
        
        # Fall back to the original placeholder creation if database fails
        return create_all_placeholders(df)


def _create_email_records_synchronous(request_id: str, df) -> tuple[dict, dict]:
    """
    Original synchronous email record creation for small datasets
    """
    all_placeholders = {}
    all_uuid_mapping = {}
    
    for _, row in df.iterrows():
        # Determine processing type
        intelligence_used = bool(row.get('intelligence', False))
        processing_type = "ai_with_research" if intelligence_used else "template_llm"
        
        # Create email placeholder in database
        email_record = GeneratedEmailService.create_email_placeholder(
            request_id=request_id,
            csv_row_index=int(row.name),
            name=str(row['name']),
            company=str(row['company']),
            linkedin_url=str(row['linkedin_url']),
            intelligence_used=intelligence_used,
            template_type=str(row.get('template_type', '')),
            processing_type=processing_type
        )
        
        # Create placeholder strings for CSV response
        placeholder_uuid_str = str(email_record.placeholder_uuid)
        processing_prefix = "AI_PROCESSING" if intelligence_used else "TEMPLATE_LLM_PROCESSING"
        
        all_placeholders[row.name] = f"{processing_prefix}:{placeholder_uuid_str}"
        all_uuid_mapping[row.name] = placeholder_uuid_str
    
    print(f"Database records created: 1 request + {len(df)} email placeholders")
    return all_placeholders, all_uuid_mapping


async def _bulk_create_email_records_background(
    request_id: str, 
    df, 
    uuid_mapping: dict
):
    """
    Background task to bulk-create email records in database
    
    Uses batch operations for efficiency with large datasets
    """
    try:
        print(f"Starting background bulk creation of {len(df)} email records for request {request_id}")
        
        # Import here to avoid circular imports
        from .database.services import GeneratedEmailService
        
        # Prepare all records for bulk insertion
        email_records_data = []
        for _, row in df.iterrows():
            intelligence_used = bool(row.get('intelligence', False))
            processing_type = "ai_with_research" if intelligence_used else "template_llm"
            
            record_data = {
                'request_id': request_id,
                'csv_row_index': int(row.name),
                'name': str(row['name']),
                'company': str(row['company']),
                'linkedin_url': str(row['linkedin_url']),
                'intelligence_used': intelligence_used,
                'template_type': str(row.get('template_type', '')),
                'processing_type': processing_type,
                'placeholder_uuid': uuid_mapping[row.name],
                'status': 'processing'
            }
            email_records_data.append(record_data)
        
        # Use bulk operations to create all records efficiently
        success_count = await GeneratedEmailService.bulk_create_email_placeholders(email_records_data)
        
        if success_count == len(df):
            print(f"Successfully bulk-created {success_count} email records for request {request_id}")
        else:
            print(f"Partial success: {success_count}/{len(df)} email records created for request {request_id}")
            
    except Exception as e:
        print(f"Error in background bulk email record creation for request {request_id}: {e}")
        # Note: We don't raise here since this is a background task
        # The email processing can still continue with the in-memory UUIDs


from .utils import (
    merge_results_in_order,
    generate_request_id,
    calculate_file_size_mb
)

# Configuration settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Default: 100, configurable via environment
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))  # Maximum allowed batch size
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "50000"))  # Maximum CSV rows to process
INTELLIGENCE_BATCH_SIZE = int(os.getenv("INTELLIGENCE_BATCH_SIZE", "5"))  # Smaller batches for AI processing
AI_FALLBACK_TO_TEMPLATE = os.getenv("AI_FALLBACK_TO_TEMPLATE", "true").lower() == "true"

app = FastAPI(
    title="Batch Email Generator",
    description="Generate personalized outreach emails from CSV data using configurable templates, fake LinkedIn research, and AI-powered generation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Welcome to Batch Email Generator API - NOW WITH UNIFIED LLM PROCESSING",
        "version": "2.1.0",
        "major_update": "All emails now use LLM generation (both template and AI types)",
        "features": [
            "Unified LLM-based email generation",
            "Template-guided LLM generation",
            "AI-powered personalization with LinkedIn research", 
            "Industry-aware data generation",
            "Background processing for all email types",
            "UUID placeholder system",
            "Unified result logging"
        ],
        "processing_types": {
            "template_llm": "LLM generation using template as base prompt (intelligence=false)",
            "ai_research": "LLM generation with fake LinkedIn research (intelligence=true)"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "templates": "/templates",
            "generate_emails": "/generate-emails",
            "test_metadata": "/process-emails-metadata",
            "task_status": "/task-status/{task_id}"
        },
        "important_note": "NO immediate email results - all processing happens in background with LLM API calls"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    # Check AI configuration
    try:
        from .ai_generator import validate_ai_configuration
        ai_valid, ai_error = validate_ai_configuration()
        ai_status = "configured" if ai_valid else f"not configured: {ai_error}"
    except Exception as e:
        ai_status = f"error: {str(e)}"
    
    return {
        "status": "healthy", 
        "message": "Batch Email Generator is running",
        "version": "2.0.0",
        "config": {
            "batch_size": BATCH_SIZE,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_csv_rows": MAX_CSV_ROWS,
            "intelligence_batch_size": INTELLIGENCE_BATCH_SIZE,
            "ai_fallback_enabled": AI_FALLBACK_TO_TEMPLATE
        },
        "capabilities": {
            "templates": "available",
            "fake_linkedin_research": "available",
            "ai_generation": ai_status
        }
    }

@app.get("/templates")
async def get_available_templates():
    """Get list of available email templates"""
    return {
        "available_templates": get_all_templates(),
        "default_template": DEFAULT_TEMPLATE_TYPE.value,
        "total_templates": len(get_all_templates())
    }

@app.post("/process-emails-metadata")
async def process_emails_metadata(
    file: UploadFile = File(...),
    template_type: Optional[TemplateType] = Form(None, description="Fallback template type when CSV template_type column is empty. Uses sales_outreach if not provided.")
):
    """
    Process CSV and return metadata only (for testing large files via Swagger UI)
    
    This endpoint processes the CSV file and generates a few sample emails but returns
    only metadata instead of the full processed file. For testing large datasets
    through Swagger UI without download limitations.
    
    - **file**: CSV file with required columns: name, company, linkedin_url
              Optional columns: intelligence (true/false), template_type (template name)
    - **template_type**: Fallback email template type when template_type column is empty. Uses sales_outreach if not provided.
    
    **CSV Column Details:**
    - **name** (required): Contact's name
    - **company** (required): Contact's company name
    - **linkedin_url** (required): Contact's LinkedIn profile URL
    - **intelligence** (optional): true/false - Use AI research and generation (not implemented in Phase 1)
    - **template_type** (optional): Per-row template selection (overrides the fallback template_type parameter)
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    Returns: JSON with processing metadata and sample emails (no file download)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV content and structure
        validate_csv_not_empty(df)
        df = validate_and_enhance_csv(df)
        validate_csv_size(df, MAX_CSV_ROWS)
        
        # Calculate processing information for new parallel approach
        ai_rows, template_rows = split_dataframe_by_intelligence(df)
        
        # Generate sample emails from first few rows
        sample_size = min(3, len(df))
        sample_df = df.head(sample_size)
        
        sample_emails = []
        for _, row in sample_df.iterrows():
            email = await generate_single_email(row, template_type)
            sample_emails.append({
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
                "intelligence": row.get('intelligence', False),
                "template_type": row.get('template_type', ''),
                "generated_email": email[:200] + "..." if len(email) > 200 else email
            })
        
        return {
            "status": "success",
            "message": "CSV processed successfully - metadata only (NEW: All emails use LLM)",
            "file_info": {
                "filename": file.filename,
                "size_bytes": len(contents),
                "size_mb": round(len(contents) / (1024 * 1024), 2)
            },
            "processing_info": {
                "total_rows": len(df),
                "template_llm_rows": len(template_rows),  # These now use LLM too
                "ai_research_rows": len(ai_rows),
                "processing_method": "unified-llm-background",
                "immediate_emails": 0,  # No immediate processing anymore
                "background_emails": len(df),  # All emails go to background
                "llm_batch_size": INTELLIGENCE_BATCH_SIZE,
                "estimated_output_size_mb": round((len(df) * 1000) / (1024 * 1024), 1),
                "estimated_processing_time_minutes": round((len(df) * 4) / 60, 1)  # ~4 seconds per LLM call
            },
            "template_info": get_template_info(template_type),
            "sample_emails": sample_emails,
            "important_changes": [
                "All emails now use LLM generation (both template and AI types)",
                "No immediate results - all processing happens in background",
                "Response contains UUID placeholders for all emails",
                "Results logged to unified_results_*.json files"
            ],
            "note": "This is metadata only. Use /generate-emails endpoint to get CSV with UUID placeholders. All processing now happens in background with LLM."
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except Exception as e:
        # Log the full error for debugging while returning safe message to user
        print(f"Unexpected error in process_emails_metadata: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error occurred while processing your request. Please try again or contact support if the issue persists."
        )










@app.post("/generate-emails")
async def generate_emails(
    file: UploadFile = File(...),
    template_type: Optional[TemplateType] = Form(None, description="Fallback template type when CSV template_type column is empty. Uses sales_outreach if not provided.")
):
    """
    Generate personalized emails from CSV data using unified LLM background processing
    
    **MAJOR CHANGE**: All emails now use LLM generation (both template and AI types)
    
    - **file**: CSV file with required columns: name, company, linkedin_url
              Optional columns: intelligence (true/false), template_type (template name)
    - **template_type**: Fallback email template type when template_type column is empty. Uses sales_outreach if not provided.
    
    **CSV Column Details:**
    - **name** (required): Contact's name
    - **company** (required): Contact's company name  
    - **linkedin_url** (required): Contact's LinkedIn profile URL
    - **intelligence** (optional): true/false - Use AI with LinkedIn research vs LLM with template base
    - **template_type** (optional): Per-row template selection (overrides the fallback template_type parameter)
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    **NEW Processing Architecture:**
    - **Template LLM emails** (intelligence=false): LLM generation using template as base (~3-5s per email)
    - **AI Research emails** (intelligence=true): LLM generation with LinkedIn research (~5-8s per email)
    - **All Background Processing**: NO immediate results - everything processed in background
    - **Immediate Response**: CSV with UUID placeholders for ALL emails
    - **Unified Logging**: All results logged to unified_results_*.json files
    
    **Response Headers:**
    - X-Request-ID: Unique identifier for tracking background processing
    - X-Immediate-Emails: "0" (no immediate emails anymore)
    - X-Background-Template-LLM-Emails: Number of template LLM emails processing
    - X-Background-AI-Research-Emails: Number of AI research emails processing
    - X-LLM-Status: "processing" (all emails use LLM)
    - X-JSON-Logs: Pattern for finding unified result JSON files
    
    Returns: CSV file with UUID placeholders for ALL emails (no immediate results)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV content and structure
        validate_csv_not_empty(df)
        df = validate_and_enhance_csv(df)
        validate_csv_size(df, MAX_CSV_ROWS)
        
        # UNIFIED LLM PROCESSING: All emails now use LLM (both template and AI)
        ai_rows, template_rows = split_dataframe_by_intelligence(df)
        
        # Generate unique request ID for tracking
        request_id = generate_request_id()
        
        print(f"Request {request_id}: Processing {len(df)} rows (ALL with LLM)")
        print(f"   Template LLM emails: {len(template_rows)} (background)")
        print(f"   AI research emails: {len(ai_rows)} (background)")
        
        # 1. Create database records for request and email placeholders
        all_placeholders, all_uuid_mapping = create_database_records_and_placeholders(
            request_id=request_id,
            original_filename=file.filename,
            df=df,
            template_rows=template_rows,
            ai_rows=ai_rows,
            file_size_bytes=len(contents),
            template_type=template_type
        )
        generated_emails = merge_results_in_order(df, all_placeholders)
        
        # 2. Start unified background processing for ALL emails
        print(f"Starting unified background LLM processing for request {request_id}")
        asyncio.create_task(
            process_all_emails_background(request_id, df, template_type, all_uuid_mapping)
        )
        
        # Add generated emails to dataframe
        df['generated_email'] = generated_emails
        
        # Convert dataframe to CSV string
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        # Create streaming response
        def iter_csv():
            yield csv_content
        
        # All emails are now processed in background with LLM
        print(f"Request {request_id}: Returning CSV with {len(df)} placeholder UUIDs, all processing in background")
        
        # Calculate final file size for headers
        final_size_mb = calculate_file_size_mb(csv_content)
        
        return StreamingResponse(
            iter_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=generated_{file.filename}",
                "X-Request-ID": request_id,
                "X-Total-Rows": str(len(df)),
                "X-Background-Template-LLM-Emails": str(len(template_rows)),
                "X-Background-AI-Research-Emails": str(len(ai_rows)),
                "X-Immediate-Emails": "0",  # No immediate emails anymore
                "X-Processing-Method": "unified-llm-background",
                "X-Template-Type": template_type.value if template_type else DEFAULT_TEMPLATE_TYPE.value,
                "X-LLM-Status": "processing",  # All emails use LLM now
                "X-JSON-Logs": f"unified_results_{request_id}_*.json",
                "X-File-Size-MB": str(final_size_mb),
                "X-Original-Filename": file.filename,
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except Exception as e:
        # Log the full error for debugging while returning safe message to user
        print(f"Unexpected error in generate_emails: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error occurred while processing your request. Please try again or contact support if the issue persists."
        )


# Database Query Endpoints

@app.get("/requests/{request_id}")
async def get_request_details(request_id: str):
    """
    Get detailed information about a specific request
    """
    try:
        from .database.services import AnalyticsService
        
        summary = AnalyticsService.get_request_summary(request_id)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        
        return {"request_summary": summary, "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving request: {str(e)}")


@app.get("/requests/{request_id}/emails")
async def get_request_emails(request_id: str, limit: int = 100):
    """
    Get all emails for a specific request
    """
    try:
        from .database.services import GeneratedEmailService
        
        emails = GeneratedEmailService.get_emails_for_request(request_id)
        if not emails:
            raise HTTPException(status_code=404, detail=f"No emails found for request {request_id}")
        
        # Apply limit
        limited_emails = emails[:limit]
        
        # Format response
        formatted_emails = []
        for email in limited_emails:
            formatted_emails.append({
                "id": email.id,
                "uuid": str(email.placeholder_uuid),
                "name": email.name,
                "company": email.company,
                "generated_email": email.generated_email,
                "status": email.status,
                "processing_type": email.processing_type,
                "total_tokens": email.total_tokens,
                "cost_usd": float(email.cost_usd or 0),
                "created_at": email.created_at.isoformat() if email.created_at else None
            })
        
        return {
            "request_id": request_id,
            "total_emails": len(emails),
            "returned_count": len(formatted_emails),
            "emails": formatted_emails
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving emails: {str(e)}")


@app.get("/emails/{email_uuid}")
async def get_email_by_uuid(email_uuid: str):
    """
    Get a specific email by its UUID
    """
    try:
        from .database.services import GeneratedEmailService
        
        email = GeneratedEmailService.get_email_by_uuid(email_uuid)
        if not email:
            raise HTTPException(status_code=404, detail=f"Email with UUID {email_uuid} not found")
        
        return {
            "id": email.id,
            "request_id": email.request_id,
            "uuid": str(email.placeholder_uuid),
            "name": email.name,
            "company": email.company,
            "generated_email": email.generated_email,
            "status": email.status,
            "processing_type": email.processing_type,
            "total_tokens": email.total_tokens,
            "cost_usd": float(email.cost_usd or 0),
            "created_at": email.created_at.isoformat() if email.created_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving email: {str(e)}")


@app.get("/analytics/performance")
async def get_performance_analytics(days: int = 7):
    """
    Get system performance analytics
    """
    try:
        from .database.services import AnalyticsService
        
        performance_data = AnalyticsService.get_performance_report(days)
        
        return {
            "period_days": days,
            "performance_data": performance_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")


# Production runner
if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting Batch Email Generator on {host}:{port}")
    print(f"Configuration: Batch size={BATCH_SIZE}, Max rows={MAX_CSV_ROWS}")
    print(f"Workers: {workers}, Reload: {reload}")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Can't use multiple workers with reload
        reload=reload,
        access_log=True,
        log_level="info"
    )