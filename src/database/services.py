"""
Database Services for Email Generator

Business logic layer for database operations.
Replaces JSON file operations with database queries.
"""

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, text
from datetime import datetime, timedelta
import logging
import uuid
import asyncio

from .models import (
    EmailRequest, GeneratedEmail, ProcessingBatch, 
    ProcessingError, SystemMetric
)
from .connection import get_database_manager
from ..templates import TemplateType

logger = logging.getLogger(__name__)


class EmailRequestService:
    """Service for managing email requests"""
    
    @staticmethod
    def create_request(
        request_id: str,
        original_filename: str,
        total_rows: int,
        template_llm_rows: int,
        ai_research_rows: int,
        file_size_bytes: Optional[int] = None,
        fallback_template_type: Optional[TemplateType] = None
    ) -> EmailRequest:
        """Create a new email request record"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            request = EmailRequest(
                request_id=request_id,
                original_filename=original_filename,
                total_rows=total_rows,
                template_llm_rows=template_llm_rows,
                ai_research_rows=ai_research_rows,
                file_size_bytes=file_size_bytes,
                file_size_mb=file_size_bytes / (1024 * 1024) if file_size_bytes else None,
                fallback_template_type=fallback_template_type.value if fallback_template_type else None,
                processing_started_at=datetime.utcnow()
            )
            
            session.add(request)
            session.flush()  # Get the ID
            return request
    
    @staticmethod
    def update_request_completion(
        request_id: str,
        status: str,
        successful_emails: int,
        failed_emails: int,
        total_tokens: int,
        estimated_cost: float,
        processing_time: float
    ) -> bool:
        """Update request with completion data"""
        db_manager = get_database_manager()
        
        try:
            with db_manager.session_scope() as session:
                request = session.query(EmailRequest).filter_by(request_id=request_id).first()
                if not request:
                    return False
                
                request.status = status
                request.successful_emails = successful_emails
                request.failed_emails = failed_emails
                request.total_llm_tokens_used = total_tokens
                request.estimated_cost_usd = estimated_cost
                request.total_processing_time_seconds = processing_time
                request.processing_completed_at = datetime.utcnow()
                
                return True
        except Exception as e:
            logger.error(f"Failed to update request completion: {e}")
            return False
    
    @staticmethod
    def get_request(request_id: str) -> Optional[EmailRequest]:
        """Get request by ID"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            return session.query(EmailRequest).filter_by(request_id=request_id).first()
    
    @staticmethod
    def get_recent_requests(limit: int = 10) -> List[EmailRequest]:
        """Get recent requests"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            return session.query(EmailRequest)\
                         .order_by(desc(EmailRequest.created_at))\
                         .limit(limit)\
                         .all()


class GeneratedEmailService:
    """Service for managing generated emails"""
    
    @staticmethod
    def create_email_placeholder(
        request_id: str,
        csv_row_index: int,
        name: str,
        company: str,
        linkedin_url: str,
        intelligence_used: bool,
        template_type: Optional[str],
        processing_type: str,
        placeholder_uuid: Optional[str] = None
    ) -> GeneratedEmail:
        """Create email record with placeholder (before processing)"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            email = GeneratedEmail(
                request_id=request_id,
                csv_row_index=csv_row_index,
                name=name,
                company=company,
                linkedin_url=linkedin_url,
                intelligence_used=intelligence_used,
                template_type=template_type,
                processing_type=processing_type,
                placeholder_uuid=uuid.UUID(placeholder_uuid) if placeholder_uuid else uuid.uuid4(),
                status='processing'
            )
            
            session.add(email)
            session.flush()
            return email
    
    @staticmethod
    def update_email_completion(
        email_id: int,
        generated_email: str,
        llm_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        processing_time: float,
        cost: float,
        linkedin_research_quality: Optional[float] = None,
        linkedin_research_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update email with generated content and metadata"""
        db_manager = get_database_manager()
        
        try:
            with db_manager.session_scope() as session:
                email = session.query(GeneratedEmail).filter_by(id=email_id).first()
                if not email:
                    return False
                
                email.generated_email = generated_email
                email.llm_model_used = llm_model
                email.prompt_tokens = prompt_tokens
                email.completion_tokens = completion_tokens
                email.total_tokens = prompt_tokens + completion_tokens
                email.processing_time_seconds = processing_time
                email.cost_usd = cost
                email.linkedin_research_quality = linkedin_research_quality
                email.linkedin_research_time_seconds = linkedin_research_time
                email.error_message = error_message
                email.status = 'completed' if generated_email and not error_message else 'failed'
                email.processing_completed_at = datetime.utcnow()
                
                return True
        except Exception as e:
            logger.error(f"Failed to update email completion: {e}")
            return False
    
    @staticmethod
    def get_emails_for_request(request_id: str) -> List[GeneratedEmail]:
        """Get all emails for a request"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            return session.query(GeneratedEmail)\
                         .filter_by(request_id=request_id)\
                         .order_by(GeneratedEmail.csv_row_index)\
                         .all()
    
    @staticmethod
    def get_email_by_uuid(placeholder_uuid: str) -> Optional[GeneratedEmail]:
        """Get email by placeholder UUID"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            return session.query(GeneratedEmail)\
                         .filter_by(placeholder_uuid=uuid.UUID(placeholder_uuid))\
                         .first()
    
    @staticmethod
    def search_emails_by_content(
        search_term: str, 
        limit: int = 50
    ) -> List[GeneratedEmail]:
        """Search emails by content using full-text search"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            # Use PostgreSQL full-text search
            return session.query(GeneratedEmail)\
                         .filter(func.to_tsvector('english', GeneratedEmail.generated_email)\
                                .match(search_term))\
                         .order_by(desc(GeneratedEmail.created_at))\
                         .limit(limit)\
                         .all()
    
    @staticmethod
    async def bulk_create_email_placeholders(email_records_data: List[Dict[str, Any]]) -> int:
        """
        Bulk create email placeholder records for improved performance
        
        Args:
            email_records_data: List of dictionaries containing email record data
            
        Returns:
            Number of records successfully created
        """
        if not email_records_data:
            return 0
        
        db_manager = get_database_manager()
        batch_size = 1000  # Process in batches to avoid memory issues
        total_created = 0
        
        try:
            # Process in batches
            for i in range(0, len(email_records_data), batch_size):
                batch = email_records_data[i:i + batch_size]
                
                with db_manager.session_scope() as session:
                    # Prepare batch records
                    batch_records = []
                    for record_data in batch:
                        email = GeneratedEmail(
                            request_id=record_data['request_id'],
                            csv_row_index=record_data['csv_row_index'],
                            name=record_data['name'],
                            company=record_data['company'],
                            linkedin_url=record_data['linkedin_url'],
                            intelligence_used=record_data['intelligence_used'],
                            template_type=record_data['template_type'],
                            processing_type=record_data['processing_type'],
                            placeholder_uuid=uuid.UUID(record_data['placeholder_uuid']),
                            status=record_data['status']
                        )
                        batch_records.append(email)
                    
                    # Bulk insert batch
                    session.bulk_save_objects(batch_records)
                    total_created += len(batch_records)
                    
                    logger.info(f"Bulk created batch {i//batch_size + 1}: {len(batch_records)} records")
                
                # Add small delay between batches to avoid overwhelming the database
                if i + batch_size < len(email_records_data):
                    await asyncio.sleep(0.01)  # 10ms delay
            
            logger.info(f"Successfully bulk created {total_created} email placeholder records")
            return total_created
            
        except Exception as e:
            logger.error(f"Error in bulk_create_email_placeholders: {e}")
            raise


class ProcessingBatchService:
    """Service for managing processing batches"""
    
    @staticmethod
    def create_batch(
        request_id: str,
        batch_type: str,
        batch_number: int,
        total_batches: int,
        emails_in_batch: int,
        batch_size_limit: int = 5
    ) -> ProcessingBatch:
        """Create a new processing batch"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            batch = ProcessingBatch(
                request_id=request_id,
                batch_type=batch_type,
                batch_number=batch_number,
                total_batches=total_batches,
                emails_in_batch=emails_in_batch,
                batch_size_limit=batch_size_limit,
                started_at=datetime.utcnow(),
                status='processing'
            )
            
            session.add(batch)
            session.flush()
            return batch
    
    @staticmethod
    def update_batch_completion(
        batch_id: int,
        successful_emails: int,
        failed_emails: int,
        batch_tokens: int,
        batch_cost: float,
        processing_time: float,
        error_message: Optional[str] = None
    ) -> bool:
        """Update batch with completion data"""
        db_manager = get_database_manager()
        
        try:
            with db_manager.session_scope() as session:
                batch = session.query(ProcessingBatch).filter_by(id=batch_id).first()
                if not batch:
                    return False
                
                batch.successful_emails = successful_emails
                batch.failed_emails = failed_emails
                batch.batch_tokens_used = batch_tokens
                batch.batch_cost_usd = batch_cost
                batch.processing_time_seconds = processing_time
                batch.error_message = error_message
                batch.status = 'completed' if not error_message else 'failed'
                batch.completed_at = datetime.utcnow()
                
                return True
        except Exception as e:
            logger.error(f"Failed to update batch completion: {e}")
            return False


class ProcessingErrorService:
    """Service for managing processing errors"""
    
    @staticmethod
    def log_error(
        error_type: str,
        error_message: str,
        request_id: Optional[str] = None,
        email_id: Optional[int] = None,
        batch_id: Optional[int] = None,
        processing_type: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> ProcessingError:
        """Log a processing error"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            error = ProcessingError(
                error_type=error_type,
                error_code=error_code,
                error_message=error_message,
                error_details=error_details,
                request_id=request_id,
                email_id=email_id,
                batch_id=batch_id,
                processing_type=processing_type
            )
            
            session.add(error)
            session.flush()
            return error
    
    @staticmethod
    def get_recent_errors(hours: int = 24, limit: int = 100) -> List[ProcessingError]:
        """Get recent errors"""
        db_manager = get_database_manager()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with db_manager.session_scope() as session:
            return session.query(ProcessingError)\
                         .filter(ProcessingError.occurred_at >= cutoff_time)\
                         .order_by(desc(ProcessingError.occurred_at))\
                         .limit(limit)\
                         .all()


class SystemMetricService:
    """Service for managing system metrics"""
    
    @staticmethod
    def record_metric(
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        request_id: Optional[str] = None,
        time_period: Optional[str] = None
    ) -> SystemMetric:
        """Record a system metric"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                request_id=request_id,
                time_period=time_period
            )
            
            session.add(metric)
            session.flush()
            return metric
    
    @staticmethod
    def get_metric_summary(
        metric_name: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get metric summary for the last N days"""
        db_manager = get_database_manager()
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with db_manager.session_scope() as session:
            metrics = session.query(SystemMetric)\
                           .filter(and_(
                               SystemMetric.metric_name == metric_name,
                               SystemMetric.recorded_at >= cutoff_time
                           ))\
                           .all()
            
            if not metrics:
                return {
                    "metric_name": metric_name,
                    "count": 0,
                    "total": 0,
                    "average": 0,
                    "min": 0,
                    "max": 0
                }
            
            values = [float(m.metric_value) for m in metrics]
            
            return {
                "metric_name": metric_name,
                "count": len(values),
                "total": sum(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "unit": metrics[0].metric_unit,
                "period_days": days
            }


class AnalyticsService:
    """Service for analytics and reporting"""
    
    @staticmethod
    def get_request_summary(request_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive request summary"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            # Use the request_summary view we created
            result = session.execute(text("""
                SELECT * FROM request_summary WHERE request_id = :request_id
            """), {"request_id": request_id}).fetchone()
            
            if not result:
                return None
            
            return {
                "request_id": result.request_id,
                "original_filename": result.original_filename,
                "total_rows": result.total_rows,
                "template_llm_rows": result.template_llm_rows,
                "ai_research_rows": result.ai_research_rows,
                "status": result.status,
                "created_at": result.created_at,
                "total_processing_time_seconds": float(result.total_processing_time_seconds or 0),
                "successful_emails": result.successful_emails,
                "failed_emails": result.failed_emails,
                "total_llm_tokens_used": result.total_llm_tokens_used,
                "estimated_cost_usd": float(result.estimated_cost_usd or 0),
                "total_generated_emails": result.total_generated_emails,
                "avg_email_processing_time": float(result.avg_email_processing_time or 0),
                "actual_tokens_used": result.actual_tokens_used or 0,
                "actual_cost_usd": float(result.actual_cost_usd or 0)
            }
    
    @staticmethod
    def get_performance_report(days: int = 7) -> List[Dict[str, Any]]:
        """Get recent performance report"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            results = session.execute(text("""
                SELECT * FROM recent_performance 
                WHERE date >= CURRENT_DATE - INTERVAL :days || ' days'
                ORDER BY date DESC
            """), {"days": days}).fetchall()
            
            return [
                {
                    "date": result.date.isoformat(),
                    "requests_processed": result.requests_processed,
                    "total_emails_requested": result.total_emails_requested,
                    "total_emails_completed": result.total_emails_completed,
                    "total_emails_failed": result.total_emails_failed,
                    "avg_processing_time": float(result.avg_processing_time or 0),
                    "total_tokens_consumed": result.total_tokens_consumed or 0,
                    "total_cost_usd": float(result.total_cost_usd or 0)
                }
                for result in results
            ]
    
    @staticmethod
    def get_template_effectiveness() -> List[Dict[str, Any]]:
        """Get template effectiveness metrics"""
        db_manager = get_database_manager()
        
        with db_manager.session_scope() as session:
            results = session.execute(text("""
                SELECT 
                    template_type,
                    COUNT(*) as total_emails,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_emails,
                    AVG(processing_time_seconds) as avg_processing_time,
                    AVG(cost_usd) as avg_cost_per_email,
                    SUM(total_tokens) as total_tokens
                FROM generated_emails 
                WHERE template_type IS NOT NULL
                GROUP BY template_type
                ORDER BY total_emails DESC
            """)).fetchall()
            
            return [
                {
                    "template_type": result.template_type,
                    "total_emails": result.total_emails,
                    "successful_emails": result.successful_emails,
                    "success_rate": result.successful_emails / result.total_emails if result.total_emails > 0 else 0,
                    "avg_processing_time": float(result.avg_processing_time or 0),
                    "avg_cost_per_email": float(result.avg_cost_per_email or 0),
                    "total_tokens": result.total_tokens or 0
                }
                for result in results
            ]
