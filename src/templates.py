"""
Email Template Repository

This module contains all email templates and related functionality for the 
Batch Email Generator.
"""

from enum import Enum
from typing import Optional
from fastapi import HTTPException


class TemplateType(str, Enum):
    """Available email template types"""
    SALES_OUTREACH = "sales_outreach"
    RECRUITMENT = "recruitment"
    NETWORKING = "networking"
    PARTNERSHIP = "partnership"
    FOLLOW_UP = "follow_up"
    INTRODUCTION = "introduction"
    COLD_EMAIL = "cold_email"


# Template Repository 
TEMPLATE_REPOSITORY = {
    TemplateType.SALES_OUTREACH: """Subject: Quick question about {{company}}'s growth strategy

Hi {{name}},

I've been following {{company}}'s impressive growth and noticed your role there. Your background caught my attention, especially your experience in scaling operations.

I work with companies similar to {{company}} to help them streamline their processes and reduce operational costs by 20-30%. Given your position, I thought you might be interested in a brief conversation about how we've helped similar organizations achieve significant efficiency gains.

Would you be open to a 15-minute call this week? I'd love to share some specific examples relevant to {{company}}'s industry.

You can view my background here: {{linkedin_url}}

Best regards,
[Your Name]

P.S. If this isn't a priority right now, I completely understand. Feel free to keep my contact for future reference.""",

    TemplateType.RECRUITMENT: """Subject: Exciting opportunity at [Company Name] - Your {{company}} experience caught our attention

Hi {{name}},

I came across your profile and was impressed by your work at {{company}}. Your background aligns perfectly with a role we're looking to fill.

We're seeking talented professionals with your expertise to join our growing team. Based on your experience at {{company}}, I believe you'd be a great fit for this position.

Would you be interested in a brief 15-minute call to discuss this opportunity? I'd love to share more details about the role and how your skills from {{company}} would contribute to our team's success.

You can learn more about my background here: {{linkedin_url}}

Best regards,
[Your Name]
[Company] Talent Acquisition

Looking forward to connecting with you!""",

    TemplateType.NETWORKING: """Subject: Fellow {{company}} industry professional - Would love to connect

Hi {{name}},

I noticed your impressive work at {{company}} and would love to connect with a fellow professional in our industry.

Your experience at {{company}} particularly caught my attention, and I believe we could have some interesting conversations about industry trends and best practices.

Would you be open to a brief virtual coffee chat? I'm always eager to learn from professionals with your background and share insights from my own experience.

Feel free to check out my profile: {{linkedin_url}}

Best regards,
[Your Name]

Looking forward to connecting!""",

    TemplateType.PARTNERSHIP: """Subject: Potential partnership opportunity between {{company}} and [Your Company]

Hi {{name}},

I've been following {{company}}'s work and am impressed by your team's achievements. I believe there could be some exciting partnership opportunities between {{company}} and our organization.

Given your role at {{company}}, I thought you'd be the right person to explore potential collaboration opportunities that could benefit both our organizations.

Would you be interested in a brief discussion about how we might work together? I'd love to share some ideas and hear your thoughts on potential synergies.

You can learn more about my background here: {{linkedin_url}}

Best regards,
[Your Name]
[Your Company]

Excited about the possibilities!""",

    TemplateType.FOLLOW_UP: """Subject: Following up on our connection - {{company}} insights

Hi {{name}},

I hope this message finds you well. I wanted to follow up on our previous conversation and see how things are progressing at {{company}}.

Your insights about {{company}}'s approach were really valuable, and I've been thinking about some of the points you raised. I'd love to continue our discussion when you have a moment.

Would you be available for a brief call this week? I have some updates to share and would appreciate your perspective on a few industry developments.

As always, you can find more about my work here: {{linkedin_url}}

Best regards,
[Your Name]

Looking forward to reconnecting!""",

    TemplateType.INTRODUCTION: """Subject: Introduction and admiration for {{company}}'s work

Hi {{name}},

I wanted to reach out and introduce myself. I've been following {{company}}'s journey and am truly impressed by the work your team is doing.

Your role at {{company}} and the company's recent achievements have caught my attention, and I'd love the opportunity to connect with someone driving such innovative work.

I'd appreciate the chance for a brief introduction call to learn more about your experience at {{company}} and share a bit about my own background in the industry.

You can learn more about my work here: {{linkedin_url}}

Best regards,
[Your Name]

Hope to connect soon!""",

    TemplateType.COLD_EMAIL: """Subject: {{name}}, impressed by your work at {{company}}

Hi {{name}},

I hope this email finds you well. I came across your profile and was impressed by your work at {{company}}.

I'm reaching out because I believe we might have some mutual interests in [relevant area], and your experience at {{company}} would bring valuable perspective to a conversation.

Would you be open to a brief 15-minute call? I'd love to learn more about your work at {{company}} and share some insights from my own experience.

You can find my background here: {{linkedin_url}}

Best regards,
[Your Name]

Thank you for your time!"""
}

# Default template type
DEFAULT_TEMPLATE_TYPE = TemplateType.SALES_OUTREACH


def get_template_description(template_type: TemplateType) -> str:
    """Get human-readable description for template types"""
    descriptions = {
        TemplateType.SALES_OUTREACH: "Professional sales outreach for business development",
        TemplateType.RECRUITMENT: "Recruiting talented professionals for job opportunities",
        TemplateType.NETWORKING: "Building professional relationships and industry connections",
        TemplateType.PARTNERSHIP: "Exploring business partnership and collaboration opportunities",
        TemplateType.FOLLOW_UP: "Following up on previous conversations and connections",
        TemplateType.INTRODUCTION: "Introducing yourself and expressing admiration for their work",
        TemplateType.COLD_EMAIL: "General cold outreach for various professional purposes"
    }
    return descriptions.get(template_type, "Professional email template")


def get_template_content(template_type: Optional[TemplateType] = None) -> str:
    """Get template content by type, with fallback to default"""
    if template_type is None:
        template_type = DEFAULT_TEMPLATE_TYPE
    
    if template_type not in TEMPLATE_REPOSITORY:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid template type: {template_type}. Available types: {[t.value for t in TemplateType]}"
        )
    
    return TEMPLATE_REPOSITORY[template_type]


def get_all_templates():
    """Get all available templates with metadata"""
    return [
        {
            "name": template_type.value,
            "description": get_template_description(template_type),
            "preview": TEMPLATE_REPOSITORY[template_type][:100] + "..."
        }
        for template_type in TemplateType
    ]


def get_template_info(template_type: Optional[TemplateType] = None):
    """Get template information for API responses"""
    if template_type is None:
        template_type = DEFAULT_TEMPLATE_TYPE
    
    return {
        "template_type": template_type.value,
        "template_description": get_template_description(template_type)
    }
