from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from jinja2 import Template
from typing import Optional
import pandas as pd
import io

app = FastAPI(
    title="Batch Email Generator",
    description="Generate personalized outreach emails from CSV data",
    version="1.0.0"
)

# Default sales outreach template
DEFAULT_TEMPLATE = """Subject: Quick question about {{company}}'s growth strategy

Hi {{name}},

I've been following {{company}}'s impressive growth and noticed your role there. Your background caught my attention, especially your experience in scaling operations.

I work with companies similar to {{company}} to help them streamline their processes and reduce operational costs by 20-30%. Given your position, I thought you might be interested in a brief conversation about how we've helped similar organizations achieve significant efficiency gains.

Would you be open to a 15-minute call this week? I'd love to share some specific examples relevant to {{company}}'s industry.

You can view my background here: {{linkedin_url}}

Best regards,
Vivek Verma

P.S. If this isn't a priority right now, I completely understand. Feel free to keep my contact for future reference."""

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {"message": "Welcome to Batch Email Generator API"}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    return {"status": "healthy", "message": "Batch Email Generator is running"}

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

@app.post("/generate-emails")
async def generate_emails(
    file: UploadFile = File(...),
    template: Optional[str] = Form(None, description="Custom email template. Uses default if not provided.")
):
    """
    Generate personalized emails from CSV data
    
    - **file**: CSV file with columns: name, company, linkedin_url, email_template
    - **template**: Uses default sales template if not provided.
    
    Returns: CSV file with original data plus generated_email column
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Use provided template or default
    email_template = template if template else DEFAULT_TEMPLATE
    
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
        
        # Use first row for sample email generation
        first_row = df.iloc[0]
        sample_email = render_email_template(
            email_template,
            str(first_row['name']),
            str(first_row['company']),
            str(first_row['linkedin_url'])
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoded CSV")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
    
    return {
        "message": "CSV file uploaded and template processed successfully", 
        "filename": file.filename,
        "size": len(contents),
        "rows_found": len(df),
        "sample_data_used": {
            "name": str(first_row['name']),
            "company": str(first_row['company']),
            "linkedin_url": str(first_row['linkedin_url'])
        },
        "template_preview": sample_email[:200] + "..." if len(sample_email) > 200 else sample_email
    }