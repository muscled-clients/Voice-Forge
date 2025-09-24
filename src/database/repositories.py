"""
VoiceForge Database Repositories
Clean repository pattern for database operations
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import bcrypt
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from src.database.models import (
    User, Plan, APIKey, Developer, UserSubscription,
    Transcription, WebSocketSession, BatchJob, BatchJobFile,
    UsageRecord, Invoice, PaymentMethod, UsageMetric,
    RateLimit, HealthCheck,
    UserRole, SubscriptionStatus, TranscriptionStatus, JobStatus
)

class UserRepository:
    """Repository for user operations"""
    
    def __init__(self):
        pass
    
    def get_by_email(self, session: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return session.query(User).filter(
            User.email == email,
            User.is_active == True
        ).first()
    
    def get_by_id(self, session: Session, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID"""
        return session.query(User).filter(
            User.id == user_id,
            User.is_active == True
        ).first()
    
    def get_by_api_key(self, session: Session, api_key: str) -> Optional[User]:
        """Get user by API key (through APIKey table)"""
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        api_key_record = session.query(APIKey).filter(
            APIKey.key_hash == api_key_hash,
            APIKey.is_active == True
        ).first()
        
        if api_key_record:
            # Update last used timestamp
            api_key_record.last_used = datetime.utcnow()
            api_key_record.usage_count += 1
            session.commit()
            
            return api_key_record.user
        return None
    
    def create(self, session: Session, user_data: Dict[str, Any]) -> User:
        """Create new user from dictionary data"""
        # Create user with provided data
        user = User(**user_data)
        
        session.add(user)
        session.flush()  # Get user ID without committing
        
        session.commit()
        session.refresh(user)
        
        return user
    
    def verify_password(self, email: str, password: str) -> Optional[User]:
        """Verify user password"""
        user = self.get_by_email(email)
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            # Update last login
            user.last_login = datetime.utcnow()
            self.db.commit()
            return user
        return None
    
    def update_email_verified(self, user_id: uuid.UUID, verified: bool = True) -> bool:
        """Update user email verification status"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if user:
            user.email_verified = verified
            user.email_verification_token = None
            self.db.commit()
            return True
        return False
    
    def update_subscription(self, user_id: uuid.UUID, plan_id: uuid.UUID,
                          stripe_customer_id: Optional[str] = None) -> bool:
        """Update user subscription"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if user:
            user.plan_id = plan_id
            if stripe_customer_id:
                user.stripe_customer_id = stripe_customer_id
            user.subscription_status = SubscriptionStatus.ACTIVE
            self.db.commit()
            return True
        return False

class PlanRepository:
    """Repository for subscription plans"""
    
    def __init__(self):
        pass
    
    def get_all(self, db: Session) -> List[Plan]:
        """Get all active plans"""
        return db.query(Plan).filter(
            Plan.is_active == True
        ).order_by(Plan.price_monthly).all()
    
    def get_by_name(self, db: Session, name: str) -> Optional[Plan]:
        """Get plan by name"""
        return db.query(Plan).filter(
            Plan.name == name,
            Plan.is_active == True
        ).first()
    
    def get_by_id(self, db: Session, plan_id: str) -> Optional[Plan]:
        """Get plan by ID"""
        return db.query(Plan).filter(
            Plan.id == plan_id,
            Plan.is_active == True
        ).first()

class TranscriptionRepository:
    """Repository for transcription operations"""
    
    def __init__(self):
        pass
    
    def create(self, db: Session, transcription_data: Dict) -> Transcription:
        """Create new transcription record"""
        transcription = Transcription(**transcription_data)
        db.add(transcription)
        db.commit()
        db.refresh(transcription)
        return transcription
    
    def update_result(self, db: Session, transcription_id: uuid.UUID,
                     transcription_text: str, confidence_score: float,
                     duration_seconds: float, processing_time_ms: int,
                     model_used: str, language: Optional[str] = None,
                     segments: Optional[Dict] = None,
                     speakers: Optional[Dict] = None) -> bool:
        """Update transcription with results"""
        transcription = db.query(Transcription).filter(
            Transcription.id == transcription_id
        ).first()
        
        if transcription:
            transcription.transcription_text = transcription_text
            transcription.confidence_score = confidence_score
            transcription.duration_seconds = duration_seconds
            transcription.processing_time_ms = processing_time_ms
            transcription.model_used = model_used
            transcription.language = language
            transcription.segments = segments
            transcription.speakers = speakers
            transcription.status = TranscriptionStatus.COMPLETED
            transcription.completed_at = datetime.utcnow()
            transcription.word_count = len(transcription_text.split()) if transcription_text else 0
            transcription.billing_duration_seconds = max(duration_seconds, 60.0)  # Min 1 minute
            
            if speakers and isinstance(speakers, dict):
                transcription.speaker_count = speakers.get('count', 0)
            
            db.commit()
            return True
        return False
    
    def get_user_transcriptions(self, db: Session, user_id: uuid.UUID,
                               limit: int = 50, offset: int = 0) -> List[Transcription]:
        """Get user's transcription history"""
        return db.query(Transcription).filter(
            Transcription.user_id == user_id
        ).order_by(desc(Transcription.created_at)).limit(limit).offset(offset).all()
    
    def get_user_stats(self, db: Session, user_id: str, days: int = 30) -> Dict:
        """Get user's transcription statistics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stats = db.query(
            func.count(Transcription.id).label('total_transcriptions'),
            func.count(func.nullif(Transcription.status, TranscriptionStatus.COMPLETED.value)).label('completed_transcriptions'),
            func.coalesce(func.sum(Transcription.duration_seconds), 0).label('total_seconds'),
            func.coalesce(func.avg(Transcription.confidence_score), 0).label('avg_confidence'),
            func.count(func.distinct(Transcription.language)).label('languages_used'),
            func.count(func.distinct(func.date(Transcription.created_at))).label('active_days')
        ).filter(
            Transcription.user_id == user_id,
            Transcription.created_at >= cutoff_date,
            Transcription.status == TranscriptionStatus.COMPLETED.value
        ).first()
        
        return {
            'total_requests': stats.total_transcriptions or 0,
            'requests_today': stats.completed_transcriptions or 0,  # Simplified for now
            'requests_this_month': stats.total_transcriptions or 0,
            'avg_processing_time': float(stats.avg_confidence or 0) * 1000,  # Simplified
            'total_words': 0,  # Would need word_count sum
            'total_bytes_processed': 0,  # Would need file_size sum
            'total_minutes': (stats.total_seconds or 0) / 60.0,
            'avg_confidence': float(stats.avg_confidence or 0),
            'languages_used': stats.languages_used or 0,
            'active_days': stats.active_days or 0
        }
    
    def get_language_distribution(self, db: Session, user_id: str, limit: int = 5) -> List[Dict]:
        """Get user's language usage distribution"""
        results = db.query(
            Transcription.language,
            func.count(Transcription.id).label('count')
        ).filter(
            Transcription.user_id == user_id,
            Transcription.language.isnot(None),
            Transcription.status == TranscriptionStatus.COMPLETED.value
        ).group_by(
            Transcription.language
        ).order_by(
            desc(func.count(Transcription.id))
        ).limit(limit).all()
        
        return [{"language": r.language, "count": r.count} for r in results]

class WebSocketSessionRepository:
    """Repository for WebSocket session operations"""
    
    def __init__(self):
        pass
    
    def create(self, db: Session, session_data: Dict) -> WebSocketSession:
        """Create WebSocket session record"""
        ws_session = WebSocketSession(**session_data)
        db.add(ws_session)
        db.commit()
        db.refresh(ws_session)
        return ws_session
    
    def update(self, db: Session, session_id: str, update_data: Dict) -> bool:
        """Update WebSocket session with final stats"""
        ws_session = db.query(WebSocketSession).filter(
            WebSocketSession.session_id == session_id
        ).first()
        
        if ws_session:
            for key, value in update_data.items():
                if hasattr(ws_session, key):
                    setattr(ws_session, key, value)
            ws_session.last_activity = datetime.utcnow()
            db.commit()
            return True
        return False

class APIKeyRepository:
    """Repository for API key operations"""
    
    def __init__(self):
        pass
    
    def create(self, session: Session, api_key_data: Dict) -> APIKey:
        """Create new API key"""
        api_key = APIKey(**api_key_data)
        session.add(api_key)
        session.commit()
        session.refresh(api_key)
        return api_key
    
    def get_by_hash(self, session: Session, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash"""
        return session.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    
    def get_by_user_id(self, session: Session, user_id: str) -> Optional[APIKey]:
        """Get first active API key for user"""
        return session.query(APIKey).filter(
            APIKey.user_id == user_id,
            APIKey.is_active == True
        ).first()
    
    def update(self, session: Session, api_key_id: str, data: Dict) -> bool:
        """Update API key"""
        api_key = session.query(APIKey).filter(APIKey.id == api_key_id).first()
        if api_key:
            for key, value in data.items():
                setattr(api_key, key, value)
            session.commit()
            return True
        return False
    
    def deactivate(self, session: Session, api_key_id: str) -> bool:
        """Deactivate API key"""
        return self.update(session, api_key_id, {"is_active": False})

class UsageRepository:
    """Repository for usage tracking and billing"""
    
    def __init__(self):
        pass
    
    def create(self, db: Session, usage_data: Dict) -> UsageRecord:
        """Create usage record for billing"""
        record = UsageRecord(**usage_data)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    
    def get_user_monthly_usage(self, db: Session, user_id: str,
                              month: Optional[datetime] = None) -> Dict:
        """Get user's monthly usage"""
        if not month:
            month = datetime.utcnow()
        
        start_date = month.replace(day=1).date()
        if month.month == 12:
            end_date = month.replace(year=month.year + 1, month=1, day=1).date()
        else:
            end_date = month.replace(month=month.month + 1, day=1).date()
        
        # Get transcription stats
        trans_stats = db.query(
            func.count(Transcription.id).label('api_requests'),
            func.coalesce(func.sum(Transcription.duration_seconds), 0).label('total_seconds'),
            func.coalesce(func.sum(Transcription.cost_cents), 0).label('cost_cents')
        ).filter(
            Transcription.user_id == user_id,
            func.date(Transcription.created_at) >= start_date,
            func.date(Transcription.created_at) < end_date,
            Transcription.status == TranscriptionStatus.COMPLETED
        ).first()
        
        return {
            'transcription_minutes': (trans_stats.total_seconds or 0) / 60.0,
            'api_requests': trans_stats.api_requests or 0,
            'cost_cents': trans_stats.cost_cents or 0
        }
    
    def _calculate_unit_cost(self, usage_type: str, unit: str) -> int:
        """Calculate unit cost in cents"""
        # Pricing in cents per unit
        pricing = {
            'transcription': {'minutes': 10},  # $0.10 per minute
            'streaming': {'minutes': 15},      # $0.15 per minute
            'api_call': {'requests': 1},       # $0.01 per request
        }
        
        return pricing.get(usage_type, {}).get(unit, 0)

class RateLimitRepository:
    """Repository for rate limiting"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_and_update(self, identifier: str, identifier_type: str,
                        limit_per_window: int = 1000,
                        window_seconds: int = 3600) -> bool:
        """Check if within rate limit and update counter"""
        window_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        # Get or create rate limit record
        rate_limit = self.db.query(RateLimit).filter(
            RateLimit.identifier == identifier,
            RateLimit.identifier_type == identifier_type,
            RateLimit.window_start == window_start
        ).first()
        
        if not rate_limit:
            rate_limit = RateLimit(
                identifier=identifier,
                identifier_type=identifier_type,
                window_start=window_start,
                window_duration_seconds=window_seconds,
                limit_per_window=limit_per_window,
                requests_count=0
            )
            self.db.add(rate_limit)
        
        # Check if within limit
        if rate_limit.requests_count >= limit_per_window:
            return False
        
        # Update counter
        rate_limit.requests_count += 1
        rate_limit.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def get_current_usage(self, identifier: str, identifier_type: str) -> int:
        """Get current usage count for identifier"""
        window_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        rate_limit = self.db.query(RateLimit).filter(
            RateLimit.identifier == identifier,
            RateLimit.identifier_type == identifier_type,
            RateLimit.window_start == window_start
        ).first()
        
        return rate_limit.requests_count if rate_limit else 0

class AnalyticsRepository:
    """Repository for analytics and monitoring"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_usage_metric(self, **kwargs) -> UsageMetric:
        """Log usage metric for analytics"""
        metric = UsageMetric(**kwargs)
        self.db.add(metric)
        self.db.commit()
        return metric
    
    def log_health_check(self, service_name: str, check_type: str,
                        status: str, response_time_ms: Optional[int] = None,
                        error_message: Optional[str] = None,
                        details: Optional[Dict] = None) -> HealthCheck:
        """Log health check result"""
        health_check = HealthCheck(
            service_name=service_name,
            check_type=check_type,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            details=details
        )
        
        self.db.add(health_check)
        self.db.commit()
        return health_check
    
    def get_system_stats(self) -> Dict:
        """Get system-wide statistics"""
        # User stats
        user_stats = self.db.query(
            func.count(User.id).label('total_users'),
            func.count(func.nullif(User.is_active, False)).label('active_users')
        ).first()
        
        # Transcription stats
        trans_stats = self.db.query(
            func.count(Transcription.id).label('total_transcriptions'),
            func.coalesce(func.sum(Transcription.duration_seconds), 0).label('total_seconds')
        ).filter(
            Transcription.status == TranscriptionStatus.COMPLETED
        ).first()
        
        # Recent activity
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_stats = self.db.query(
            func.count(Transcription.id).label('transcriptions_24h'),
            func.count(func.distinct(Transcription.user_id)).label('active_users_24h')
        ).filter(
            Transcription.created_at >= recent_cutoff
        ).first()
        
        return {
            'total_users': user_stats.total_users or 0,
            'active_users': user_stats.active_users or 0,
            'total_transcriptions': trans_stats.total_transcriptions or 0,
            'total_minutes_processed': (trans_stats.total_seconds or 0) / 60.0,
            'transcriptions_24h': recent_stats.transcriptions_24h or 0,
            'active_users_24h': recent_stats.active_users_24h or 0
        }