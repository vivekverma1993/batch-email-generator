from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from jinja2 import Template
from typing import Optional
import pandas as pd
import io
import asyncio
import os
import time
import uuid
import json
from datetime import datetime
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import template functionality
from .templates import (
    TemplateType, 
    DEFAULT_TEMPLATE_TYPE, 
    get_template_content, 
    get_all_templates,
    get_template_info
)

# Import AI and research functionality
from .linkedin_research import research_linkedin_profile
from .ai_generator import generate_ai_email

# Configuration settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Default: 100, configurable via environment
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))  # Maximum allowed batch size
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "50000"))  # Maximum CSV rows to process

# AI-specific configuration
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
        
        # Check if CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate required columns and add optional columns with defaults
        df = validate_and_enhance_csv(df)
        
        # Validate CSV size limits
        if len(df) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV too large. Maximum allowed rows: {MAX_CSV_ROWS}, received: {len(df)}"
            )
        
        # Calculate processing information for new parallel approach
        ai_rows = df[df['intelligence'] == True]
        template_rows = df[df['intelligence'] == False]
        
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

def validate_and_enhance_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate CSV columns and add missing optional columns with defaults
    
    Required columns: name, company, linkedin_url
    Optional columns: intelligence (defaults to False), template_type (defaults to None)
    """
    # Validate required columns
    required_columns = ['name', 'company', 'linkedin_url']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required columns: {', '.join(missing_columns)}. Required: {', '.join(required_columns)}"
        )
    
    # Add optional columns with defaults if missing
    if 'intelligence' not in df.columns:
        df['intelligence'] = False
    else:
        # Convert intelligence column to boolean, handling various formats
        df['intelligence'] = df['intelligence'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
    
    if 'template_type' not in df.columns:
        df['template_type'] = ''  # Will use default template
    else:
        # Clean template_type values - handle NaN and empty strings
        df['template_type'] = df['template_type'].fillna('')  # Replace NaN with empty string
        df['template_type'] = df['template_type'].astype(str)  # Ensure string type
        df['template_type'] = df['template_type'].replace('nan', '')  # Handle string 'nan'
    
    return df


def render_email_template(template_str: str, name: str, company: str, linkedin_url: str) -> str:
    """Render an email template with provided data"""
    try:
        template = Template(template_str)
        return template.render(
            name=name,
            company=company,
            linkedin_url=linkedin_url
        ).strip()
    except Exception as e:
        return f"Error rendering template: {str(e)}"

async def generate_single_email(row: pd.Series, fallback_template_type: Optional[TemplateType] = None) -> str:
    """Generate a single personalized email using AI research or templates based on intelligence column"""
    try:
        use_intelligence = row.get('intelligence', False)
        
        if use_intelligence:
            return await generate_intelligent_email(row, fallback_template_type)
        else:
            return await generate_template_email(row, fallback_template_type)
    
    except Exception as e:
        return f"Error generating email: {str(e)}"


async def generate_intelligent_email(row: pd.Series, fallback_template_type: Optional[TemplateType] = None) -> str:
    """Generate AI-enhanced email using fake LinkedIn research and OpenAI"""
    try:
        user_info = {
            'name': str(row.get('name', '')),
            'company': str(row.get('company', '')),
            'linkedin_url': str(row.get('linkedin_url', ''))
        }
        
        # Determine template type for AI prompt
        row_template_type = row.get('template_type')
        if row_template_type and row_template_type.strip():
            try:
                template_type = TemplateType(row_template_type.strip())
            except (ValueError, AttributeError):
                template_type = fallback_template_type
        else:
            template_type = fallback_template_type
        
        template_type_str = template_type.value if template_type else 'sales_outreach'
        
        # Generate fake LinkedIn research
        research_result = await research_linkedin_profile(
            name=user_info['name'], 
            company=user_info['company'], 
            linkedin_url=user_info['linkedin_url']
        )
        
        # Generate AI email
        ai_result = await generate_ai_email(research_result, user_info, template_type_str)
        
        if ai_result.status.value == "success":
            return ai_result.email_content or "[AI generation succeeded but no content returned]"
        else:
            if AI_FALLBACK_TO_TEMPLATE:
                print(f"AI generation failed ({ai_result.error_message}), falling back to template for {user_info['name']}")
                return await generate_template_email(row, fallback_template_type)
            else:
                return f"AI generation failed: {ai_result.error_message}"
    
    except Exception as e:
        if AI_FALLBACK_TO_TEMPLATE:
            print(f"Unexpected error in AI generation ({str(e)}), falling back to template")
            return await generate_template_email(row, fallback_template_type)
        else:
            return f"Error in AI email generation: {str(e)}"


async def generate_template_email(row: pd.Series, fallback_template_type: Optional[TemplateType] = None) -> str:
    """Generate traditional template-based email"""
    try:
        # Determine template type for this row
        row_template_type = row.get('template_type')
        
        # Use row template if specified, otherwise use fallback
        if row_template_type and row_template_type.strip():
            try:
                # Try to convert string to TemplateType enum
                template_type = TemplateType(row_template_type.strip())
            except (ValueError, AttributeError):
                # Invalid template type, use fallback
                template_type = fallback_template_type
        else:
            template_type = fallback_template_type
        
        # Get template content for this specific row
        template_content = get_template_content(template_type)
        template = Template(template_content)
        
        # Convert row to dict for template rendering
        context = {
            'name': str(row.get('name', '')),
            'company': str(row.get('company', '')),
            'linkedin_url': str(row.get('linkedin_url', ''))
        }
        
        # Render template with context
        email_content = template.render(**context)
        return email_content.strip()
    
    except Exception as e:
        return f"Error generating template email: {str(e)}"

async def process_ai_dataframe(ai_df: pd.DataFrame, fallback_template_type: Optional[TemplateType] = None) -> dict:
    """Process AI DataFrame with smaller batches and rate limiting"""
    if ai_df.empty:
        return {}
    
    print(f"AI TASK STARTED: Processing {len(ai_df)} rows with OpenAI")
    start_time = time.time()
    results = {}
    
    # Use smaller batches for AI processing to manage API rate limits
    ai_batches = create_batches(ai_df, INTELLIGENCE_BATCH_SIZE)
    print(f"    AI processing: {len(ai_df)} rows in {len(ai_batches)} batches of {INTELLIGENCE_BATCH_SIZE}")
    
    for batch_idx, batch_df in enumerate(ai_batches, 1):
        batch_start = time.time()
        print(f"    AI batch {batch_idx}/{len(ai_batches)} ({len(batch_df)} rows) - starting...")
        
        # Process AI batch with concurrent tasks
        batch_tasks = []
        for _, row in batch_df.iterrows():
            task = generate_intelligent_email(row, fallback_template_type)
            batch_tasks.append((row.name, task))
        
        batch_results = await asyncio.gather(*[task for _, task in batch_tasks])
        
        # Store results with original index
        for (original_idx, _), result in zip(batch_tasks, batch_results):
            results[original_idx] = result
        
        batch_time = time.time() - batch_start
        print(f"    AI batch {batch_idx}/{len(ai_batches)} completed in {batch_time:.2f}s")
        
        # Add small delay between AI batches for rate limiting
        if batch_idx < len(ai_batches):
            await asyncio.sleep(0.5)
    
    total_ai_time = time.time() - start_time
    print(f"  AI TASK COMPLETED: {len(results)} emails in {total_ai_time:.2f}s")
    
    return results


async def process_template_dataframe(template_df: pd.DataFrame, fallback_template_type: Optional[TemplateType] = None) -> dict:
    """Process template DataFrame with large batches (fast processing)"""
    if template_df.empty:
        return {}
    
    print(f"  TEMPLATE TASK STARTED: Processing {len(template_df)} rows (concurrent)")
    start_time = time.time()
    
    # Template generation is fast, so we can use larger batches
    template_tasks = []
    for _, row in template_df.iterrows():
        task = generate_template_email(row, fallback_template_type)
        template_tasks.append((row.name, task))
    
    # Process all template emails concurrently (they're fast)
    template_results = await asyncio.gather(*[task for _, task in template_tasks])
    
    # Store results with original index
    results = {}
    for (original_idx, _), result in zip(template_tasks, template_results):
        results[original_idx] = result
    
    template_time = time.time() - start_time
    print(f"  TEMPLATE TASK COMPLETED: {len(results)} emails in {template_time:.3f}s")
    
    return results


def merge_results_in_order(original_df: pd.DataFrame, results: dict) -> list[str]:
    """Merge results back in original DataFrame order"""
    ordered_emails = []
    for _, row in original_df.iterrows():
        email = results.get(row.name, "Error: No result generated")
        ordered_emails.append(email)
    
    return ordered_emails


def log_ai_results_to_json(request_id: str, ai_results: dict, original_ai_rows: pd.DataFrame, processing_time: float, ai_uuid_mapping: Optional[dict] = None):
    """Log completed AI email results to structured JSON file"""
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


async def process_ai_emails_background(request_id: str, ai_rows: pd.DataFrame, fallback_template_type: Optional[TemplateType] = None, ai_uuid_mapping: Optional[dict] = None):
    """Background processing of AI emails with JSON logging"""
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
        error_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "failed",
            "total_ai_emails": len(ai_rows)
        }
        
        filename = f"ai_error_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(error_entry, f, indent=2, ensure_ascii=False)
            print(f"AI error logged to {filename}")
        except Exception as log_error:
            print(f"Failed to log AI error: {str(log_error)}")


# TODO: asyncio.gather has some disadvantages like missing error handling and memory usage.
# TODO: Consider using a different approach for batch processing. like using a asyncio.TaskGroup
# to manage the tasks and handle errors.
async def process_batch(batch_df: pd.DataFrame, fallback_template_type: Optional[TemplateType] = None) -> list[str]:
    """Process a batch of rows in parallel with per-row template selection"""
    # Generate emails for all rows in this batch concurrently
    # Each row can have its own template_type or use the fallback
    batch_emails = await asyncio.gather(*[
        generate_single_email(row, fallback_template_type)
        for _, row in batch_df.iterrows()
    ])
    return batch_emails

def create_batches(df: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Split DataFrame into batches of specified size"""
    batches = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        batches.append(batch)
    return batches

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
        
        # Check if CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate required columns and add optional columns with defaults
        df = validate_and_enhance_csv(df)
        
        # Validate CSV size limits
        if len(df) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV too large. Maximum allowed rows: {MAX_CSV_ROWS}, received: {len(df)}"
            )
        
        # BACKGROUND PROCESSING: Split DataFrame for immediate vs background processing
        ai_rows = df[df['intelligence'] == True].copy()
        template_rows = df[df['intelligence'] == False].copy()
        
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())[:8]  # Short UUID for easy reference
        
        print(f"Request {request_id}: Processing {len(df)} rows")
        print(f"   Template emails: {len(template_rows)} (immediate)")
        print(f"   AI emails: {len(ai_rows)} (background)")
        
        # 1. Process template emails IMMEDIATELY
        template_results = {}
        if not template_rows.empty:
            print(f"Processing {len(template_rows)} template emails immediately...")
            start_time = time.time()
            template_results = await process_template_dataframe(template_rows, template_type)
            template_time = time.time() - start_time
            print(f"Template emails completed in {template_time:.3f}s")
        
        # 2. Add placeholder UUIDs for AI emails
        ai_placeholders = {}
        ai_uuid_mapping = {}  # Store UUIDs for later use in JSON logging
        if not ai_rows.empty:
            print(f"Adding placeholder UUIDs for {len(ai_rows)} AI emails...")
            for _, row in ai_rows.iterrows():
                placeholder_uuid = str(uuid.uuid4())
                ai_placeholders[row.name] = f"AI_PROCESSING:{placeholder_uuid}"
                ai_uuid_mapping[row.name] = placeholder_uuid  # Store for JSON logging
        
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
        final_size_mb = round(len(csv_content) / (1024 * 1024), 2)
        
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