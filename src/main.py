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
        
        # 1. Create placeholders for ALL emails (both types now use LLM)
        all_placeholders, all_uuid_mapping = create_all_placeholders(df)
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