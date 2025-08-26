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
    create_ai_placeholders
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
        "message": "Welcome to Batch Email Generator API",
        "version": "2.0.0",
        "features": [
            "Template-based email generation",
            "Fake LinkedIn research",
            "AI-powered personalization", 
            "Industry-aware data generation",
            "Batch processing with fallback"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "templates": "/templates",
            "generate_emails": "/generate-emails",
            "test_metadata": "/process-emails-metadata"
        }
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
            "message": "CSV processed successfully - metadata only",
            "file_info": {
                "filename": file.filename,
                "size_bytes": len(contents),
                "size_mb": round(len(contents) / (1024 * 1024), 2)
            },
            "processing_info": {
                "total_rows": len(df),
                "ai_rows": len(ai_rows),
                "template_rows": len(template_rows),
                "processing_method": "parallel",
                "ai_batch_size": INTELLIGENCE_BATCH_SIZE,
                "estimated_output_size_mb": round((len(df) * 1000) / (1024 * 1024), 1)  # Rough estimate
            },
            "template_info": get_template_info(template_type),
            "sample_emails": sample_emails,
            "note": "This is metadata only. Use /generate-emails endpoint to download the full processed CSV file."
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
    Generate personalized emails from CSV data using background processing for optimal UX
    
    - **file**: CSV file with required columns: name, company, linkedin_url
              Optional columns: intelligence (true/false), template_type (template name)
    - **template_type**: Fallback email template type when template_type column is empty. Uses sales_outreach if not provided.
    
    **CSV Column Details:**
    - **name** (required): Contact's name
    - **company** (required): Contact's company name  
    - **linkedin_url** (required): Contact's LinkedIn profile URL
    - **intelligence** (optional): true/false - Use AI research and generation vs templates
    - **template_type** (optional): Per-row template selection (overrides the fallback template_type parameter)
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    **Processing Architecture:**
    - **Template emails** (intelligence=false): Processed IMMEDIATELY (~0.01s per email)
    - **AI emails** (intelligence=true): Processed in BACKGROUND (~3-8s per email)
    - **Immediate Response**: CSV with template emails + UUID placeholders for AI emails
    - **Background Logging**: AI results logged to JSON files when complete
    
    **Response Headers:**
    - X-Request-ID: Unique identifier for tracking background processing
    - X-Immediate-Emails: Number of template emails returned immediately
    - X-Background-AI-Emails: Number of AI emails processing in background
    - X-AI-Status: "processing" or "none"
    - X-JSON-Logs: Pattern for finding AI result JSON files
    
    Returns: CSV file with template emails + AI placeholders (UUIDs)
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
        
        # BACKGROUND PROCESSING: Split DataFrame for immediate vs background processing
        ai_rows, template_rows = split_dataframe_by_intelligence(df)
        
        # Generate unique request ID for tracking
        request_id = generate_request_id()
        
        print(f"Request {request_id}: Processing {len(df)} rows")
        print(f"   Template emails: {len(template_rows)} (immediate)")
        print(f"   AI emails: {len(ai_rows)} (background)")
        
        # 1. Process template emails IMMEDIATELY
        template_results = {}
        if not template_rows.empty:
            print(f"Processing {len(template_rows)} template emails immediately...")
            import time
            start_time = time.time()
            template_results = await process_template_dataframe(template_rows, template_type)
            template_time = time.time() - start_time
            print(f"Template emails completed in {template_time:.3f}s")
        
        # 2. Add placeholder UUIDs for AI emails
        ai_placeholders, ai_uuid_mapping = create_ai_placeholders(ai_rows)
        
        # 3. Combine immediate results with AI placeholders
        all_results = {**template_results, **ai_placeholders}
        generated_emails = merge_results_in_order(df, all_results)
        
        # 4. Start background AI processing (fire and forget)
        if not ai_rows.empty:
            print(f"Starting background AI processing for request {request_id}")
            asyncio.create_task(
                process_ai_emails_background(request_id, ai_rows, template_type, ai_uuid_mapping)
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
        
        immediate_emails = len(template_rows)
        background_emails = len(ai_rows) 
        print(f"Request {request_id}: Returning {immediate_emails} immediate emails, {background_emails} processing in background")
        
        # Calculate final file size for headers
        final_size_mb = calculate_file_size_mb(csv_content)
        
        return StreamingResponse(
            iter_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=generated_{file.filename}",
                "X-Request-ID": request_id,
                "X-Total-Rows": str(len(df)),
                "X-Immediate-Emails": str(len(template_rows)),
                "X-Background-AI-Emails": str(len(ai_rows)),
                "X-Processing-Method": "background",
                "X-Template-Type": template_type.value if template_type else DEFAULT_TEMPLATE_TYPE.value,
                "X-AI-Status": "processing" if len(ai_rows) > 0 else "none",
                "X-JSON-Logs": "ai_results_*.json" if len(ai_rows) > 0 else "none",
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