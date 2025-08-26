from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from jinja2 import Template
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
    get_template_content, 
    get_all_templates,
    get_template_info
)

# Configuration settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Default: 100, configurable via environment
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))  # Maximum allowed batch size
MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "50000"))  # Maximum CSV rows to process

app = FastAPI(
    title="Batch Email Generator",
    description="Generate personalized outreach emails from CSV data using configurable templates and async batch processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Welcome to Batch Email Generator API",
        "version": "1.0.0",
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
    return {
        "status": "healthy", 
        "message": "Batch Email Generator is running",
        "version": "1.0.0",
        "config": {
            "batch_size": BATCH_SIZE,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_csv_rows": MAX_CSV_ROWS
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
    template_type: Optional[TemplateType] = Form(None, description="Email template type. Uses sales_outreach if not provided.")
):
    """
    Process CSV and return metadata only (for testing large files via Swagger UI)
    
    This endpoint processes the CSV file and generates a few sample emails but returns
    only metadata instead of the full processed file. For testing large datasets
    through Swagger UI without download limitations.
    
    - **file**: CSV file with columns: name, company, linkedin_url
    - **template_type**: Email template type from available templates. Uses sales_outreach if not provided.
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    Returns: JSON with processing metadata and sample emails (no file download)
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Get template content from repository
    email_template = get_template_content(template_type)
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['name', 'company', 'linkedin_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}. Required: {', '.join(required_columns)}"
            )
        
        # Check if CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate CSV size limits
        if len(df) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV too large. Maximum allowed rows: {MAX_CSV_ROWS}, received: {len(df)}"
            )
        
        # Create Jinja2 template object
        jinja_template = Template(email_template)
        
        # Calculate batch information
        batches = create_batches(df, BATCH_SIZE)
        total_batches = len(batches)
        
        # Generate sample emails from first few rows
        sample_size = min(3, len(df))
        sample_df = df.head(sample_size)
        
        sample_emails = []
        for _, row in sample_df.iterrows():
            email = await generate_single_email(row, jinja_template)
            sample_emails.append({
                "name": str(row['name']),
                "company": str(row['company']),
                "linkedin_url": str(row['linkedin_url']),
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
                "batch_size": BATCH_SIZE,
                "total_batches": total_batches,
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

async def generate_single_email(row: pd.Series, template: Template) -> str:
    """Generate a single personalized email using the template"""
    try:
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
        return f"Error generating email: {str(e)}"

# TODO: asyncio.gather has some disadvantages like missing error handling and memory usage.
# TODO: Consider using a different approach for batch processing. like using a asyncio.TaskGroup
# to manage the tasks and handle errors.
async def process_batch(batch_df: pd.DataFrame, template: Template) -> list[str]:
    """Process a batch of rows in parallel"""
    # Generate emails for all rows in this batch concurrently
    batch_emails = await asyncio.gather(*[
        generate_single_email(row, template)
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
    template_type: Optional[TemplateType] = Form(None, description="Email template type. Uses sales_outreach if not provided.")
):
    """
    Generate personalized emails from CSV data using batch processing
    
    - **file**: CSV file with columns: name, company, linkedin_url
    - **template_type**: Email template type from available templates. Uses sales_outreach if not provided.
    
    Available template types: sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email
    
    Batch size is configured via BATCH_SIZE environment variable (default: 100).
    
    Returns: CSV file with original data plus generated_email column
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Get template content from repository
    email_template = get_template_content(template_type)
    
    # Read file contents
    contents = await file.read()
    
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['name', 'company', 'linkedin_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}. Required: {', '.join(required_columns)}"
            )
        
        # Check if CSV has data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate CSV size limits
        if len(df) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV too large. Maximum allowed rows: {MAX_CSV_ROWS}, received: {len(df)}"
            )
        
        # Create template object
        jinja_template = Template(email_template)
        
        # Split DataFrame into batches for efficient processing
        batches = create_batches(df, BATCH_SIZE)
        total_batches = len(batches)
        
        print(f"Processing {len(df)} rows in {total_batches} batches of {BATCH_SIZE}")
        
        # Process batches sequentially, but rows within each batch in parallel
        all_generated_emails = []
        for batch_idx, batch_df in enumerate(batches, 1):
            print(f"Processing batch {batch_idx}/{total_batches} ({len(batch_df)} rows)")
            
            # Process current batch in parallel
            batch_emails = await process_batch(batch_df, jinja_template)
            all_generated_emails.extend(batch_emails)
        
        # Combine all generated emails
        generated_emails = all_generated_emails
        
        # Add generated emails to dataframe
        df['generated_email'] = generated_emails
        
        # Convert dataframe to CSV string
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        # Create streaming response
        def iter_csv():
            yield csv_content
        
        print(f"Successfully generated {len(generated_emails)} emails in {total_batches} batches")
        
        # Calculate final file size for headers
        final_size_mb = round(len(csv_content) / (1024 * 1024), 2)
        
        return StreamingResponse(
            iter_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=generated_{file.filename}",
                "Content-Length": str(len(csv_content)),
                "X-Total-Rows": str(len(df)),
                "X-Batch-Size": str(BATCH_SIZE),
                "X-Total-Batches": str(total_batches),
                "X-Template-Type": template_type.value if template_type else DEFAULT_TEMPLATE_TYPE.value,
                "X-Processing-Time": "Available in server logs",
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