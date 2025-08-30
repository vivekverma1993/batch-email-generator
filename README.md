# Batch Email Generator

A high-performance FastAPI application that generates personalized outreach emails from CSV data using **dual-mode processing**: fast template-based generation and intelligent AI-powered personalization with background processing.

## Overview

The Batch Email Generator revolutionizes personalized email outreach by offering two distinct processing modes in a single request. Choose **template-based generation** for instant results (~0.01s per email) or **AI-powered generation** for highly personalized emails (~3-8s per email) with fake LinkedIn research integration.

**Key Innovation**: Hybrid processing architecture that returns template emails immediately while processing AI emails in the background, ensuring optimal user experience without waiting for slow AI operations.

**Perfect for**: Sales teams, recruiters, marketing professionals, and anyone needing to send personalized emails at scale with varying levels of personalization.

## Features

### Dual-Mode Email Generation
- **Template Mode** (`intelligence=false`): Lightning-fast Jinja2 template rendering (~0.01s per email)
- **AI Mode** (`intelligence=true`): OpenAI-powered personalization with fake LinkedIn research (~3-8s per email)
- **Hybrid Processing**: Mix both modes in a single CSV for optimal flexibility

### 🧠 Intelligence Layer
- **Fake LinkedIn Research**: Industry-aware profile generation with realistic job titles, skills, and experience
- **AI-Powered Personalization**: OpenAI integration for highly customized email content
- **Background Processing**: AI emails processed asynchronously with UUID placeholders
- **JSON Result Logging**: Completed AI emails logged to timestamped JSON files for tracking

### Core Functionality  
- **Enhanced CSV Processing**: Support for `name`, `company`, `linkedin_url`, `intelligence`, `template_type` columns
- **7 Professional Templates**: Sales outreach, recruitment, networking, partnership, follow-up, introduction, cold email
- **Per-Row Template Selection**: Override global template with row-specific `template_type` column
- **Immediate Response Architecture**: Get template results instantly, AI results via background processing

### Technical Highlights
- **Hybrid Response System**: Immediate CSV download with template emails + AI placeholders
- **Request Tracking**: Unique request IDs for monitoring background AI processing
- **Fallback Logic**: Automatic fallback to templates if AI generation fails
- **Rate Limiting**: Built-in delays and batch processing for OpenAI API compliance
- **Scalable Architecture**: Handles up to 50,000 rows with configurable batch sizes

## Tech Stack

- **Backend**: FastAPI (Python 3.8+)
- **Data Processing**: Pandas
- **Templating**: Jinja2
- **AI Integration**: OpenAI GPT for intelligent email generation
- **Server**: Uvicorn with async background processing
- **Deployment**: Docker-ready, supports Render/Railway free tiers

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key (for AI-powered email generation)

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

3. **Set up environment variables** (Optional for AI features)
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "BATCH_SIZE=100" >> .env
   echo "INTELLIGENCE_BATCH_SIZE=5" >> .env
   echo "MAX_CSV_ROWS=50000" >> .env
   echo "AI_FALLBACK_TO_TEMPLATE=true" >> .env
   ```

4. **Run the application**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

### Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | None | Required for AI email generation |
| `BATCH_SIZE` | 100 | Template processing batch size |
| `INTELLIGENCE_BATCH_SIZE` | 5 | AI processing batch size (smaller for rate limiting) |
| `MAX_CSV_ROWS` | 50000 | Maximum rows per CSV upload |
| `AI_FALLBACK_TO_TEMPLATE` | true | Fall back to templates if AI fails |

## Usage

### API Endpoints

#### 1. Generate Emails
**POST** `/generate-emails`

Upload a CSV file and generate personalized emails using hybrid template/AI processing.

**Parameters:**
- `file` (required): CSV file with columns: `name`, `company`, `linkedin_url`, `intelligence` (optional), `template_type` (optional)
- `template_type` (optional): Fallback template type when CSV template_type column is empty

**Response:** CSV file with immediate template emails + UUID placeholders for background AI emails

#### 2. Process Emails Metadata  
**POST** `/process-emails-metadata`

Test endpoint that returns processing metadata and sample emails instead of full CSV download.

#### 3. Available Templates
**GET** `/templates`

Get list of all available email templates and their descriptions.

#### 4. Health Check
**GET** `/health`

Check service status and AI configuration.

### Example Request

```bash
curl -X POST "http://localhost:8000/generate-emails" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@contacts.csv" \
  -F "template_type=sales_outreach"
```

### Enhanced CSV Format

The system now supports enhanced CSV format with intelligence and template control:

```csv
name,company,linkedin_url,intelligence,template_type
John Smith,TechCorp Inc,https://linkedin.com/in/johnsmith,false,sales_outreach
Sarah Johnson,DataFlow Solutions,https://linkedin.com/in/sarahjohnson,true,recruitment
Mike Chen,CloudTech Enterprises,https://linkedin.com/in/mikechen,false,networking
```

**CSV Column Details:**
- **name** (required): Contact's full name
- **company** (required): Contact's company name  
- **linkedin_url** (required): Contact's LinkedIn profile URL
- **intelligence** (optional): `true` for AI generation, `false` for templates (default: false)
- **template_type** (optional): Per-row template override (sales_outreach, recruitment, networking, partnership, follow_up, introduction, cold_email)

### Example Response

The API returns a CSV file immediately with template emails and UUID placeholders for AI emails:

```csv
name,company,linkedin_url,intelligence,template_type,generated_email
John Smith,TechCorp Inc,https://linkedin.com/in/johnsmith,false,sales_outreach,"Subject: Quick question about TechCorp Inc's growth strategy

Hi John Smith,

I've been following TechCorp Inc's impressive growth and noticed your role there..."
Sarah Johnson,DataFlow Solutions,https://linkedin.com/in/sarahjohnson,true,recruitment,"AI_PROCESSING:a1b2c3d4-e5f6-7890-abcd-ef1234567890"
Mike Chen,CloudTech Enterprises,https://linkedin.com/in/mikechen,false,networking,"Subject: Connecting with fellow professionals

Hi Mike Chen,

I came across your profile and was impressed by your work at CloudTech Enterprises..."
```

**Response Headers Include:**
- `X-Request-ID`: Unique tracking ID for this request
- `X-Immediate-Emails`: Number of template emails processed immediately  
- `X-Background-AI-Emails`: Number of AI emails processing in background
- `X-AI-Status`: "processing" or "none"

### AI Results Tracking

AI emails are processed in the background and results are logged to JSON files:

```json
{
  "request_id": "a1b2c3d4",
  "timestamp": "2024-01-15T10:30:00",
  "processing_time_seconds": 45.2,
  "total_ai_emails": 1,
  "results": [
    {
      "uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "name": "Sarah Johnson", 
      "company": "DataFlow Solutions",
      "generated_email": "Subject: Exciting Data Science Opportunity\n\nHi Sarah,\n\nI noticed your impressive background in data analytics at DataFlow Solutions..."
    }
  ]
}
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

## Available Templates

The system includes 7 professional email templates optimized for different use cases:

| Template Type | Use Case | Key Features |
|---------------|----------|--------------|
| `sales_outreach` | B2B lead generation | Growth-focused, efficiency benefits, clear CTA |
| `recruitment` | Talent acquisition | Skills-focused, opportunity highlights, career growth |
| `networking` | Professional connections | Industry insights, mutual connections, knowledge sharing |
| `partnership` | Business partnerships | Strategic alignment, mutual benefits, collaboration focus |
| `follow_up` | Re-engagement | Previous interaction reference, value-add, gentle persistence |
| `introduction` | Warm introductions | Mutual connections, credibility building, clear purpose |
| `cold_email` | Initial outreach | Attention-grabbing, personalized research, strong value prop |

## Processing Modes

### 🔥 Template Mode (`intelligence=false`)
- **Speed**: ~0.01 seconds per email
- **Cost**: Free (no API calls)
- **Quality**: Professional, consistent templates with variable substitution
- **Best for**: High-volume campaigns, consistent messaging, budget-conscious users

### 🧠 AI Mode (`intelligence=true`)  
- **Speed**: ~3-8 seconds per email
- **Cost**: OpenAI API usage (~$0.001-0.01 per email depending on model)
- **Quality**: Highly personalized using fake LinkedIn research + AI generation
- **Best for**: High-value prospects, personalized outreach, maximum engagement

### Hybrid Mode (Mix in single CSV)
- **Speed**: Immediate templates + background AI
- **Cost**: Only pay for AI emails
- **Quality**: Best of both worlds
- **Best for**: Strategic segmentation, A/B testing, optimal resource allocation

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## Deployment

### Local Development
```bash
uvicorn src.main:app --reload --port 8000
```


## Advanced Features

### Implemented Intelligence Features
- **AI-Powered Email Generation**: OpenAI GPT integration with personalized prompts
- **Fake LinkedIn Research**: Industry-aware profile generation for AI context
- **Hybrid Processing Architecture**: Immediate templates + background AI processing  
- **JSON Result Logging**: Structured tracking of AI email generation results
- **Template Management**: 7 professional templates with per-row selection
- **Background Processing**: Non-blocking AI generation with UUID tracking

### Future Enhancements

#### Next Phase Extensions
- **Real LinkedIn Integration**: Actual profile scraping (with proper rate limiting and compliance)
- **Web Frontend**: React-based file upload interface with real-time progress tracking
- **Advanced Analytics**: Email performance tracking and template effectiveness metrics
- **Integration Hub**: Direct connections to email platforms (Gmail, Outlook, SendGrid)
- **Enhanced AI Models**: Support for multiple LLM providers (Anthropic, Azure OpenAI)

#### Technical Roadmap
- **Database Integration**: Persistent storage for templates, generation history, and user preferences
- **Advanced Rate Limiting**: Intelligent API throttling based on provider limits
- **Authentication & Authorization**: JWT-based user management with role-based access
- **Caching Layer**: Redis integration for improved performance and cost optimization
- **Monitoring & Observability**: Comprehensive logging, metrics, and alerting system
- **A/B Testing Framework**: Template and AI prompt effectiveness testing




