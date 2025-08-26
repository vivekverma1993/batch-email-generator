# Batch Email Generator

A high-performance FastAPI application that generates personalized outreach emails from CSV data using customizable templates and parallel async processing.

## Overview

The Batch Email Generator is designed to streamline personalized email outreach campaigns by automatically generating customized emails from contact data. Upload a CSV file with contact information, apply a personalized template, and download the results with generated emails ready for your outreach campaigns.

**Perfect for**: Sales teams, recruiters, marketing professionals, and anyone needing to send personalized emails at scale.

## Features

### Core Functionality
- **CSV Upload & Processing**: Upload contact lists with `name`, `company`, and `linkedin_url` fields
- **Template-Based Generation**: Use Jinja2 templates with placeholders like `{{name}}`, `{{company}}`
- **Parallel Async Processing**: High-speed email generation using FastAPI's async capabilities
- **Configurable Templates**: Provide custom templates or use intelligent defaults
- **CSV Export**: Download results with an added `generated_email` column

### Technical Highlights
- **RESTful API**: Clean, documented endpoints for easy integration
- **Input Validation**: Robust error handling and data validation
- **Scalable Architecture**: Async processing handles large datasets efficiently
- **Template Flexibility**: Support for complex Jinja2 templating logic

## Tech Stack

- **Backend**: FastAPI (Python 3.8+)
- **Data Processing**: Pandas
- **Templating**: Jinja2
- **Server**: Uvicorn
- **Deployment**: Docker-ready, supports Render/Railway free tiers

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/batch-email-generator.git
   cd batch-email-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Usage

### API Endpoints

#### 1. Generate Emails
**POST** `/generate-emails`

Upload a CSV file and generate personalized emails using a template.

**Parameters:**
- `file` (required): CSV file with columns: `name`, `company`, `linkedin_url`
- `template` (optional): Custom Jinja2 template string

#### 2. Health Check
**GET** `/health`

Check if the service is running.

### Example Request

```bash
curl -X POST "http://localhost:8000/generate-emails" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@contacts.csv" \
  -F "template=Hi {{name}}, I noticed your work at {{company}}. Your LinkedIn profile ({{linkedin_url}}) shows impressive experience. Would love to connect!"
```

### Sample Input CSV

```csv
name,company,linkedin_url
John Smith,TechCorp Inc,https://linkedin.com/in/johnsmith
Sarah Johnson,DataFlow Solutions,https://linkedin.com/in/sarahjohnson
Mike Chen,CloudTech Enterprises,https://linkedin.com/in/mikechen
```

### Example Response

The API returns a CSV file with the original data plus a `generated_email` column:

```csv
name,company,linkedin_url,generated_email
John Smith,TechCorp Inc,https://linkedin.com/in/johnsmith,"Hi John Smith, I noticed your work at TechCorp Inc. Your LinkedIn profile (https://linkedin.com/in/johnsmith) shows impressive experience. Would love to connect!"
Sarah Johnson,DataFlow Solutions,https://linkedin.com/in/sarahjohnson,"Hi Sarah Johnson, I noticed your work at DataFlow Solutions. Your LinkedIn profile (https://linkedin.com/in/sarahjohnson) shows impressive experience. Would love to connect!"
Mike Chen,CloudTech Enterprises,https://linkedin.com/in/mikechen,"Hi Mike Chen, I noticed your work at CloudTech Enterprises. Your LinkedIn profile (https://linkedin.com/in/mikechen) shows impressive experience. Would love to connect!"
```

### Default Template

If no template is provided, the system uses this default sales outreach template:

```
Subject: Quick question about {{company}}'s growth strategy

Hi {{name}},

I've been following {{company}}'s impressive growth and noticed your role there. Your background caught my attention, especially your experience in scaling operations.

I work with companies similar to {{company}} to help them streamline their processes and reduce operational costs by 20-30%. Given your position, I thought you might be interested in a brief conversation about how we've helped similar organizations achieve significant efficiency gains.

Would you be open to a 15-minute call this week? I'd love to share some specific examples relevant to {{company}}'s industry.

You can view my background here: {{linkedin_url}}

Best regards,
[Your Name]

P.S. If this isn't a priority right now, I completely understand. Feel free to keep my contact for future reference.
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## Deployment

### Local Development
```bash
uvicorn src.main:app --reload --port 8000
```


## Future Improvements

### Planned Extensions
- **Multi-Agent Research System**: Automatic LinkedIn profile enrichment for more personalized templates
- **Template Management API**: CRUD operations for saving and managing multiple templates
- **Web Frontend**: Simple file upload interface with real-time progress tracking
- **Advanced Analytics**: Email performance tracking and template effectiveness metrics
- **Integration Hub**: Direct connections to email platforms (Gmail, Outlook, SendGrid)
- **AI-Powered Templates**: LLM-generated personalization based on LinkedIn data

### Technical Enhancements
- **Database Integration**: Persistent storage for templates and generation history
- **Rate Limiting**: API throttling for production usage
- **Authentication**: JWT-based user management
- **Caching**: Redis integration for improved performance
- **Monitoring**: Comprehensive logging and metrics collection




