"""
VoiceForge Python SDK

Official Python SDK for the VoiceForge Speech-to-Text API.
Provides easy-to-use interfaces for transcription, streaming, and batch processing.
"""

from .client import VoiceForgeClient
from .models import (
    TranscriptionJob,
    TranscriptionOptions,
    StreamingResult,
    Word,
    DiarizationSegment,
    LanguageDetectionResult,
    UserInfo,
    ModelInfo,
    TranscriptionStatus,
    AudioFormat,
    UserTier,
)
from .exceptions import (
    VoiceForgeError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    TranscriptionError,
    FileError,
)
from .__version__ import __version__

__all__ = [
    # Main client
    "VoiceForgeClient",
    
    # Models
    "TranscriptionJob",
    "TranscriptionOptions", 
    "StreamingResult",
    "Word",
    "DiarizationSegment",
    "LanguageDetectionResult",
    "UserInfo",
    "ModelInfo",
    
    # Enums
    "TranscriptionStatus",
    "AudioFormat",
    "UserTier",
    
    # Exceptions
    "VoiceForgeError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "TranscriptionError",
    "FileError",
    
    # Version
    "__version__",
]