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
        
        # Calculate batch information
        batches = create_batches(df, BATCH_SIZE)
        total_batches = len(batches)
        
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
    Generate personalized emails from CSV data using batch processing
    
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
    
    **Processing Logic:**
    - Each row can specify its own template_type for personalized template selection
    - If template_type column is empty/missing, falls back to the template_type parameter
    - Batch size is configured via BATCH_SIZE environment variable (default: 100)
    - Intelligence column prepares for future AI-enhanced email generation
    
    Returns: CSV file with original data plus generated_email column
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
        
        # Split DataFrame into batches for efficient processing
        batches = create_batches(df, BATCH_SIZE)
        total_batches = len(batches)
        
        print(f"Processing {len(df)} rows in {total_batches} batches of {BATCH_SIZE}")
        
        # Process batches sequentially, but rows within each batch in parallel
        all_generated_emails = []
        for batch_idx, batch_df in enumerate(batches, 1):
            print(f"Processing batch {batch_idx}/{total_batches} ({len(batch_df)} rows)")
            
            # Process current batch in parallel (each row can have its own template)
            batch_emails = await process_batch(batch_df, template_type)
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