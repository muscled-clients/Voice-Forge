"""
VoiceForge SDK Models

Data models and types used throughout the SDK.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict


class TranscriptionStatus(str, Enum):
    """Status of a transcription job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"
    WEBM = "webm"


class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class LanguageCode(str, Enum):
    """Supported language codes"""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    RU = "ru"
    JA = "ja"
    KO = "ko"
    ZH = "zh"
    AR = "ar"
    HI = "hi"
    TR = "tr"
    PL = "pl"
    NL = "nl"
    SV = "sv"
    DA = "da"
    NO = "no"
    FI = "fi"
    CS = "cs"


class Word(BaseModel):
    """Individual word in transcription with timing information"""
    model_config = ConfigDict(extra="forbid")
    
    text: str = Field(..., description="The word text")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    speaker_id: Optional[int] = Field(None, description="Speaker ID (if diarization enabled)")


class DiarizationSegment(BaseModel):
    """Speaker diarization segment"""
    model_config = ConfigDict(extra="forbid")
    
    speaker_id: int = Field(..., description="Unique speaker identifier")
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Diarization confidence")
    text: Optional[str] = Field(None, description="Text spoken in this segment")


class LanguageAlternative(BaseModel):
    """Alternative language detection result"""
    model_config = ConfigDict(extra="forbid")
    
    code: str = Field(..., description="Language code")
    name: str = Field(..., description="Language name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class LanguageDetectionResult(BaseModel):
    """Language detection result"""
    model_config = ConfigDict(extra="forbid")
    
    code: str = Field(..., description="Detected language code")
    name: str = Field(..., description="Detected language name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    method: str = Field(..., description="Detection method used")
    alternatives: Optional[List[LanguageAlternative]] = Field(
        None, description="Alternative language detections"
    )


class TranscriptionOptions(BaseModel):
    """Options for transcription processing"""
    model_config = ConfigDict(extra="forbid")
    
    model: Optional[str] = Field(None, description="Model to use for transcription")
    language: Optional[str] = Field(None, description="Expected language code")
    enable_diarization: bool = Field(False, description="Enable speaker diarization")
    enable_punctuation: bool = Field(True, description="Enable automatic punctuation")
    enable_word_timestamps: bool = Field(True, description="Include word-level timestamps")
    enable_language_detection: bool = Field(False, description="Auto-detect language")
    max_speakers: Optional[int] = Field(None, ge=1, le=10, description="Maximum number of speakers")
    boost_keywords: Optional[List[str]] = Field(None, description="Keywords to boost recognition")
    custom_vocabulary: Optional[List[str]] = Field(None, description="Custom vocabulary words")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sampling temperature")
    compression_ratio_threshold: Optional[float] = Field(None, description="Compression ratio threshold")
    logprob_threshold: Optional[float] = Field(None, description="Log probability threshold")
    no_speech_threshold: Optional[float] = Field(None, description="No speech threshold")


class TranscriptionJob(BaseModel):
    """Transcription job information and results"""
    model_config = ConfigDict(extra="forbid")
    
    # Basic job info
    id: str = Field(..., description="Unique job identifier")
    user_id: str = Field(..., description="User who created the job")
    status: TranscriptionStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Input details
    filename: str = Field(..., description="Original filename")
    audio_format: Optional[AudioFormat] = Field(None, description="Audio file format")
    audio_duration: Optional[float] = Field(None, description="Audio duration in seconds")
    audio_size: Optional[int] = Field(None, description="Audio file size in bytes")
    
    # Processing details
    model_used: Optional[str] = Field(None, description="Model used for transcription")
    language_code: Optional[str] = Field(None, description="Language used/detected")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    # Results (available when completed)
    transcript: Optional[str] = Field(None, description="Full transcription text")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    words: Optional[List[Word]] = Field(None, description="Word-level results")
    diarization: Optional[List[DiarizationSegment]] = Field(None, description="Speaker diarization")
    detected_language: Optional[LanguageDetectionResult] = Field(None, description="Language detection result")
    
    # Metadata
    word_count: Optional[int] = Field(None, description="Number of words transcribed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    # Error info (if failed)
    error: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")


class StreamingResult(BaseModel):
    """Real-time streaming transcription result"""
    model_config = ConfigDict(extra="forbid")
    
    transcript: str = Field(..., description="Current transcript text")
    is_final: bool = Field(..., description="Whether this is a final result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    words: Optional[List[Word]] = Field(None, description="Word-level timing")
    alternatives: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative transcriptions")
    timestamp: float = Field(..., description="Result timestamp")
    duration: Optional[float] = Field(None, description="Audio duration processed")
    language: Optional[str] = Field(None, description="Detected/specified language")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelInfo(BaseModel):
    """Information about an available model"""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(..., description="Human-readable model name")
    model_id: str = Field(..., description="Model identifier")
    type: str = Field(..., description="Model type (e.g., 'whisper', 'canary')")
    description: str = Field(..., description="Model description")
    supported_languages: List[str] = Field(..., description="Supported language codes")
    max_duration: Optional[int] = Field(None, description="Maximum audio duration in seconds")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy benchmark score")
    latency_ms: Optional[int] = Field(None, description="Average latency in milliseconds")
    is_available: bool = Field(..., description="Whether model is currently available")
    requires_gpu: bool = Field(..., description="Whether model requires GPU")


class UserInfo(BaseModel):
    """User account information"""
    model_config = ConfigDict(extra="forbid")
    
    id: str = Field(..., description="User identifier")
    email: str = Field(..., description="User email address")
    full_name: str = Field(..., description="User's full name")
    tier: UserTier = Field(..., description="Subscription tier")
    credits_remaining: int = Field(..., description="Credits remaining")
    is_active: bool = Field(..., description="Whether account is active")
    created_at: datetime = Field(..., description="Account creation date")
    updated_at: Optional[datetime] = Field(None, description="Last update date")


class UserAnalytics(BaseModel):
    """User usage analytics"""
    model_config = ConfigDict(extra="forbid")
    
    user_id: str = Field(..., description="User identifier")
    total_transcriptions: int = Field(..., description="Total number of transcriptions")
    total_audio_minutes: float = Field(..., description="Total audio processed in minutes")
    total_credits_used: int = Field(..., description="Total credits consumed")
    average_confidence: Optional[float] = Field(None, description="Average transcription confidence")
    most_used_language: Optional[str] = Field(None, description="Most frequently used language")
    most_used_model: Optional[str] = Field(None, description="Most frequently used model")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (completed/total)")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")


class QuotaInfo(BaseModel):
    """User quota and usage information"""
    model_config = ConfigDict(extra="forbid")
    
    user_id: str = Field(..., description="User identifier")
    tier: UserTier = Field(..., description="Subscription tier")
    credits_total: int = Field(..., description="Total credits allocated")
    credits_used: int = Field(..., description="Credits used this period")
    credits_remaining: int = Field(..., description="Credits remaining")
    monthly_limit: int = Field(..., description="Monthly usage limit")
    monthly_used: int = Field(..., description="Monthly usage so far")
    reset_date: datetime = Field(..., description="Next quota reset date")


class APIKeyInfo(BaseModel):
    """API key information"""
    model_config = ConfigDict(extra="forbid")
    
    id: str = Field(..., description="API key identifier")
    name: str = Field(..., description="Human-readable key name")
    key_prefix: str = Field(..., description="First few characters of the key")
    created_at: datetime = Field(..., description="Key creation date")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether key is active")
    permissions: List[str] = Field(..., description="Key permissions")


class PaginatedResponse(BaseModel):
    """Paginated API response"""
    model_config = ConfigDict(extra="forbid")
    
    items: List[Any] = Field(..., description="List of items")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class UploadInfo(BaseModel):
    """File upload information"""
    model_config = ConfigDict(extra="forbid")
    
    upload_url: str = Field(..., description="URL for file upload")
    upload_fields: Dict[str, Any] = Field(..., description="Additional upload fields")
    expires_at: datetime = Field(..., description="Upload URL expiration")
    max_file_size: int = Field(..., description="Maximum file size in bytes")


class BatchJob(BaseModel):
    """Batch transcription job information"""
    model_config = ConfigDict(extra="forbid")
    
    batch_id: str = Field(..., description="Batch identifier")
    jobs: List[TranscriptionJob] = Field(..., description="Individual transcription jobs")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    total_jobs: int = Field(..., description="Total number of jobs in batch")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    status: str = Field(..., description="Overall batch status")


class WebhookEvent(BaseModel):
    """Webhook event data"""
    model_config = ConfigDict(extra="forbid")
    
    event_type: str = Field(..., description="Type of event")
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event payload")
    user_id: str = Field(..., description="Associated user ID")


# Type aliases for convenience
TranscriptionResult = Union[TranscriptionJob, Dict[str, Any]]
StreamingCallback = Any  # Callable[[StreamingResult], None]
ProgressCallback = Any   # Callable[[float], None]