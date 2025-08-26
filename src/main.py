from fastapi import FastAPI

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