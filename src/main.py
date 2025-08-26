from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(
    title="Batch Email Generator",
    description="Generate personalized outreach emails from CSV data",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {"message": "Welcome to Batch Email Generator API"}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    return {"status": "healthy", "message": "Batch Email Generator is running"}

@app.post("/generate-emails")
async def generate_emails(file: UploadFile = File(...)):
    """
    Generate personalized emails from CSV data
    
    - **file**: CSV file with columns: name, company, linkedin_url, email_template
    
    Returns: CSV file with original data plus generated_email column
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file contents
    contents = await file.read()
    
    return {
        "message": "CSV file uploaded successfully", 
        "filename": file.filename,
        "size": len(contents)
    }