"""
Email Generation Module

Handles all email generation logic including template-based and AI-powered generation.
"""

import asyncio
import time
import os
import pandas as pd
from jinja2 import Template
from typing import Optional

from .templates import TemplateType, get_template_content
from .linkedin_research import research_linkedin_profile
from .ai_generator import generate_ai_email
from .utils import create_batches


# Configuration
INTELLIGENCE_BATCH_SIZE = int(os.getenv("INTELLIGENCE_BATCH_SIZE", "5"))
AI_FALLBACK_TO_TEMPLATE = os.getenv("AI_FALLBACK_TO_TEMPLATE", "true").lower() == "true"


def render_email_template(template_str: str, name: str, company: str, linkedin_url: str) -> str:
    """
    Render an email template with provided data
    
    Args:
        template_str: Jinja2 template string
        name: Contact's name
        company: Contact's company
        linkedin_url: Contact's LinkedIn URL
        
    Returns:
        Rendered email content
    """
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
    """
    Generate a single personalized email using AI research or templates based on intelligence column
    
    Args:
        row: DataFrame row containing contact information
        fallback_template_type: Template type to use when row template_type is empty
        
    Returns:
        Generated email content
    """
    try:
        use_intelligence = row.get('intelligence', False)
        
        if use_intelligence:
            return await generate_intelligent_email(row, fallback_template_type)
        else:
            return await generate_template_email(row, fallback_template_type)
    
    except Exception as e:
        return f"Error generating email: {str(e)}"


async def generate_intelligent_email(row: pd.Series, fallback_template_type: Optional[TemplateType] = None) -> str:
    """
    Generate AI-enhanced email using fake LinkedIn research and OpenAI
    
    Args:
        row: DataFrame row containing contact information
        fallback_template_type: Template type to use when row template_type is empty
        
    Returns:
        AI-generated email content
    """
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
    """
    Generate traditional template-based email
    
    Args:
        row: DataFrame row containing contact information
        fallback_template_type: Template type to use when row template_type is empty
        
    Returns:
        Template-generated email content
    """
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
    """
    Process AI DataFrame with smaller batches and rate limiting
    
    Args:
        ai_df: DataFrame containing rows that need AI processing
        fallback_template_type: Template type to use when row template_type is empty
        
    Returns:
        Dictionary mapping row indices to generated email content
    """
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
    """
    Process template DataFrame with large batches (fast processing)
    
    Args:
        template_df: DataFrame containing rows that need template processing
        fallback_template_type: Template type to use when row template_type is empty
        
    Returns:
        Dictionary mapping row indices to generated email content
    """
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
