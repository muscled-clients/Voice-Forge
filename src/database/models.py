"""
VoiceForge SQLAlchemy Database Models
Production-ready database schema using SQLAlchemy ORM
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Date, 
    ForeignKey, Text, JSON, DECIMAL, BigInteger, Enum as SQLEnum,
    UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, INET, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

Base = declarative_base()

# Enums
class UserRole(enum.Enum):
    ADMIN = "ADMIN"
    USER = "USER"
    DEVELOPER = "DEVELOPER"
    ENTERPRISE = "ENTERPRISE"

class SubscriptionStatus(enum.Enum):
    ACTIVE = "ACTIVE"
    CANCELED = "CANCELED"
    PAST_DUE = "PAST_DUE"
    UNPAID = "UNPAID"
    TRIALING = "TRIALING"
    INACTIVE = "INACTIVE"

class TranscriptionStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class JobStatus(enum.Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class InvoiceStatus(enum.Enum):
    DRAFT = "DRAFT"
    OPEN = "OPEN"
    PAID = "PAID"
    VOID = "VOID"
    UNCOLLECTIBLE = "UNCOLLECTIBLE"

# ======================
# CORE MODELS
# ======================

class Plan(Base):
    """Subscription plans"""
    __tablename__ = 'plans'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    description = Column(Text)
    monthly_limit = Column(Integer, nullable=False)  # Minutes per month
    price_monthly = Column(DECIMAL(10, 2), nullable=False)
    price_yearly = Column(DECIMAL(10, 2))
    features = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    stripe_price_id = Column(String(255))
    stripe_product_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="plan")
    subscriptions = relationship("UserSubscription", back_populates="plan")

class User(Base):
    """User accounts"""
    __tablename__ = 'users'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    company = Column(String(255))
    api_key = Column(String(255), unique=True, index=True)  # Legacy - to be removed
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    plan_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.plans.id'))
    
    # Status fields
    is_active = Column(Boolean, default=True, index=True)
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255))
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime(timezone=True))
    
    # Stripe integration
    stripe_customer_id = Column(String(255), unique=True, index=True)
    trial_ends_at = Column(DateTime(timezone=True))
    subscription_status = Column(SQLEnum(SubscriptionStatus), default=SubscriptionStatus.INACTIVE)
    
    # API Usage tracking
    api_calls_used = Column(Integer, default=0)
    api_calls_limit = Column(Integer, default=1000)
    last_api_call = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Additional data
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    plan = relationship("Plan", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    transcriptions = relationship("Transcription", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("UserSubscription", back_populates="user", cascade="all, delete-orphan")
    websocket_sessions = relationship("WebSocketSession", back_populates="user", cascade="all, delete-orphan")
    batch_jobs = relationship("BatchJob", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="user", cascade="all, delete-orphan")
    payment_methods = relationship("PaymentMethod", back_populates="user", cascade="all, delete-orphan")
    developer_profile = relationship("Developer", back_populates="user", uselist=False, cascade="all, delete-orphan")

class Developer(Base):
    """Developer profiles for API management"""
    __tablename__ = 'developers'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), unique=True)
    email = Column(String(255), nullable=False)
    full_name = Column(String(255))
    company = Column(String(255))
    api_keys_created = Column(Integer, default=0)
    last_api_key_created = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="developer_profile")

class APIKey(Base):
    """API Keys with enhanced security"""
    __tablename__ = 'api_keys'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(20), nullable=False)  # For display (vf_xxxx)
    name = Column(String(255))
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    rate_limit = Column(Integer, default=100)
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(DateTime(timezone=True), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    scopes = Column(JSON, default=lambda: ["transcribe:read", "transcribe:write"])
    ip_whitelist = Column(ARRAY(INET))
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class UserSubscription(Base):
    """User subscription details"""
    __tablename__ = 'user_subscriptions'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False)
    plan_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.plans.id'))
    stripe_subscription_id = Column(String(255), unique=True, index=True)
    status = Column(SQLEnum(SubscriptionStatus), nullable=False, index=True)
    current_period_start = Column(DateTime(timezone=True))
    current_period_end = Column(DateTime(timezone=True), index=True)
    cancel_at_period_end = Column(Boolean, default=False)
    canceled_at = Column(DateTime(timezone=True))
    trial_start = Column(DateTime(timezone=True))
    trial_end = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    plan = relationship("Plan", back_populates="subscriptions")
    usage_records = relationship("UsageRecord", back_populates="subscription")
    invoices = relationship("Invoice", back_populates="subscription")

# ======================
# TRANSCRIPTION MODELS
# ======================

class Transcription(Base):
    """Transcription records with Phase 3 features"""
    __tablename__ = 'transcriptions'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False, index=True)
    session_id = Column(String(255), index=True)  # For WebSocket sessions
    
    # File information
    file_name = Column(String(255))
    file_path = Column(Text)  # S3 path or local path
    file_size = Column(BigInteger)
    file_hash = Column(String(64))
    mime_type = Column(String(100))
    
    # Transcription results
    duration_seconds = Column(Float)
    language = Column(String(10))
    detected_languages = Column(JSON)  # Multiple language detection
    transcription_text = Column(Text)
    segments = Column(JSON)  # Word-level timestamps
    confidence_score = Column(Float)
    word_count = Column(Integer)
    
    # Phase 3: Advanced features
    speakers = Column(JSON)  # Speaker diarization results
    speaker_count = Column(Integer)
    noise_reduction_applied = Column(Boolean, default=False)
    noise_level_db = Column(Float)
    
    # Processing information
    processing_time_ms = Column(Integer)
    model_used = Column(String(50))
    whisper_version = Column(String(20))
    status = Column(SQLEnum(TranscriptionStatus), default=TranscriptionStatus.PENDING, index=True)
    error_message = Column(Text)
    
    # Billing
    cost_cents = Column(Integer, default=0)
    billing_duration_seconds = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True))
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="transcriptions")
    batch_job_files = relationship("BatchJobFile", back_populates="transcription")
    
    # Indexes
    __table_args__ = (
        Index('idx_transcriptions_user_created', 'user_id', 'created_at'),
        {'schema': 'voiceforge'}
    )

class WebSocketSession(Base):
    """WebSocket streaming sessions"""
    __tablename__ = 'websocket_sessions'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False, index=True)
    connection_id = Column(String(255))
    status = Column(String(50), default='active', index=True)
    
    # Audio processing stats
    duration_seconds = Column(Float)
    total_audio_seconds = Column(Float)
    bytes_processed = Column(BigInteger, default=0)
    transcription_count = Column(Integer, default=0)
    
    # Phase 3 configuration
    vad_enabled = Column(Boolean, default=True)
    speaker_diarization = Column(Boolean, default=False)
    language_detection = Column(Boolean, default=False)
    noise_reduction = Column(Boolean, default=False)
    
    # Connection details
    ip_address = Column(INET)
    user_agent = Column(Text)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="websocket_sessions")

# ======================
# BATCH PROCESSING
# ======================

class BatchJob(Base):
    """Batch transcription jobs"""
    __tablename__ = 'batch_jobs'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False, index=True)
    job_name = Column(String(255))
    job_type = Column(String(50), default='transcription')
    
    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    priority = Column(Integer, default=5)
    
    # Progress
    total_files = Column(Integer)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    success_files = Column(Integer, default=0)
    
    # Configuration
    callback_url = Column(Text)
    webhook_events = Column(JSON, default=list)
    processing_options = Column(JSON, default=dict)
    
    # Results
    results = Column(JSON)
    error_log = Column(JSON)
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    processing_time_seconds = Column(Float)
    
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="batch_jobs")
    files = relationship("BatchJobFile", back_populates="batch_job", cascade="all, delete-orphan")

class BatchJobFile(Base):
    """Files in batch jobs"""
    __tablename__ = 'batch_job_files'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_job_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.batch_jobs.id', ondelete='CASCADE'), nullable=False)
    transcription_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.transcriptions.id', ondelete='CASCADE'))
    
    file_name = Column(String(255), nullable=False)
    file_size = Column(BigInteger)
    file_path = Column(Text)
    processing_order = Column(Integer)
    
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    processing_time_ms = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    batch_job = relationship("BatchJob", back_populates="files")
    transcription = relationship("Transcription", back_populates="batch_job_files")

# ======================
# BILLING & PAYMENTS
# ======================

class UsageRecord(Base):
    """Usage tracking for billing"""
    __tablename__ = 'usage_records'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False)
    transcription_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.transcriptions.id', ondelete='SET NULL'))
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.user_subscriptions.id', ondelete='SET NULL'))
    
    # Usage metrics
    usage_type = Column(String(50), nullable=False)  # 'transcription', 'streaming', 'api_call'
    quantity = Column(Float, nullable=False)  # minutes, requests, etc.
    unit = Column(String(20), nullable=False)  # 'minutes', 'requests', 'mb'
    unit_cost_cents = Column(Integer)
    total_cost_cents = Column(Integer)
    
    # Billing period
    billing_period_start = Column(Date)
    billing_period_end = Column(Date)
    billed = Column(Boolean, default=False, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="usage_records")
    subscription = relationship("UserSubscription", back_populates="usage_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_user_period', 'user_id', 'billing_period_start', 'billing_period_end'),
        {'schema': 'voiceforge'}
    )

class Invoice(Base):
    """Customer invoices"""
    __tablename__ = 'invoices'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.user_subscriptions.id'))
    stripe_invoice_id = Column(String(255), unique=True)
    
    # Invoice details
    invoice_number = Column(String(50), unique=True)
    subtotal_cents = Column(Integer, nullable=False)
    tax_cents = Column(Integer, default=0)
    total_cents = Column(Integer, nullable=False)
    currency = Column(String(3), default='USD')
    
    # Status
    status = Column(SQLEnum(InvoiceStatus), nullable=False)
    paid = Column(Boolean, default=False)
    
    # Dates
    billing_period_start = Column(Date)
    billing_period_end = Column(Date)
    issued_at = Column(DateTime(timezone=True), server_default=func.now())
    due_date = Column(Date)
    paid_at = Column(DateTime(timezone=True))
    
    # URLs
    invoice_pdf_url = Column(Text)
    hosted_invoice_url = Column(Text)
    
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="invoices")
    subscription = relationship("UserSubscription", back_populates="invoices")

class PaymentMethod(Base):
    """Customer payment methods"""
    __tablename__ = 'payment_methods'
    __table_args__ = {'schema': 'voiceforge'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('voiceforge.users.id', ondelete='CASCADE'), nullable=False)
    stripe_payment_method_id = Column(String(255), unique=True, nullable=False)
    
    # Card details (masked)
    type = Column(String(20))  # card, bank_account
    brand = Column(String(20))  # visa, mastercard, etc.
    last4 = Column(String(4))
    exp_month = Column(Integer)
    exp_year = Column(Integer)
    
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    extra_data = Column('metadata', JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="payment_methods")

# ======================
# ANALYTICS & MONITORING
# ======================

class UsageMetric(Base):
    """Usage analytics (TimescaleDB optimized)"""
    __tablename__ = 'usage_metrics'
    __table_args__ = {'schema': 'analytics'}
    
    time = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    user_id = Column(UUID(as_uuid=True), index=True)
    endpoint = Column(String(255), index=True)
    method = Column(String(10))
    status_code = Column(Integer, index=True)
    response_time_ms = Column(Integer)
    
    # Audio metrics
    audio_duration_seconds = Column(Float)
    file_size_bytes = Column(BigInteger)
    model_used = Column(String(50))
    language = Column(String(10))
    
    # Connection details
    ip_address = Column(INET)
    user_agent = Column(Text)
    api_key_id = Column(UUID(as_uuid=True))
    session_id = Column(String(255))
    
    # Error tracking
    error_type = Column(String(100))
    error_message = Column(Text)
    
    extra_data = Column('metadata', JSON, default=dict)
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_metrics_user_time', 'user_id', 'time'),
        Index('idx_usage_metrics_endpoint_time', 'endpoint', 'time'),
        {'schema': 'analytics'}
    )

class RateLimit(Base):
    """Rate limiting tracking"""
    __tablename__ = 'rate_limits'
    __table_args__ = {'schema': 'monitoring'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identifier = Column(String(255), nullable=False)  # IP, user_id, api_key
    identifier_type = Column(String(20), nullable=False)  # 'ip', 'user', 'api_key'
    endpoint = Column(String(255))
    
    requests_count = Column(Integer, default=0)
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_duration_seconds = Column(Integer, default=3600)
    limit_per_window = Column(Integer, default=1000)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_rate_limits_identifier', 'identifier', 'identifier_type', 'window_start'),
        UniqueConstraint('identifier', 'identifier_type', 'window_start', name='uq_rate_limit_window'),
        {'schema': 'monitoring'}
    )

class HealthCheck(Base):
    """System health checks"""
    __tablename__ = 'health_checks'
    __table_args__ = {'schema': 'monitoring'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(100), nullable=False)
    check_type = Column(String(50), nullable=False)  # 'database', 'whisper', 'storage', 'redis'
    status = Column(String(20), nullable=False)  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms = Column(Integer)
    error_message = Column(Text)
    details = Column(JSON)
    checked_at = Column(DateTime(timezone=True), server_default=func.now())