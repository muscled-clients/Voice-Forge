"""
VoiceForge SDK Exceptions

Custom exception classes for the VoiceForge Python SDK.
"""

from typing import Optional, Dict, Any


class VoiceForgeError(Exception):
    """Base exception for all VoiceForge SDK errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"status_code={self.status_code})"
        )


class APIError(VoiceForgeError):
    """API request failed with an error response"""
    pass


class AuthenticationError(VoiceForgeError):
    """Authentication failed or token is invalid"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, **kwargs)


class AuthorizationError(VoiceForgeError):
    """User is not authorized to perform this action"""
    
    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(message, **kwargs)


class RateLimitError(VoiceForgeError):
    """Rate limit exceeded"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ValidationError(VoiceForgeError):
    """Request validation failed"""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_errors = field_errors or {}


class TranscriptionError(VoiceForgeError):
    """Transcription processing failed"""
    
    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.job_id = job_id


class FileError(VoiceForgeError):
    """File operation failed"""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.filename = filename


class AudioError(FileError):
    """Audio file processing failed"""
    pass


class NetworkError(VoiceForgeError):
    """Network communication failed"""
    
    def __init__(
        self,
        message: str = "Network request failed",
        original_error: Optional[Exception] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.original_error = original_error


class TimeoutError(VoiceForgeError):
    """Request or operation timed out"""
    
    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


class ConfigurationError(VoiceForgeError):
    """SDK configuration is invalid"""
    pass


class WebSocketError(VoiceForgeError):
    """WebSocket connection or operation failed"""
    
    def __init__(
        self,
        message: str,
        connection_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.connection_id = connection_id


class QuotaExceededError(VoiceForgeError):
    """User quota or credits exceeded"""
    
    def __init__(
        self,
        message: str = "Quota exceeded",
        credits_remaining: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.credits_remaining = credits_remaining


class ModelUnavailableError(VoiceForgeError):
    """Requested model is not available"""
    
    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.model_id = model_id


class StreamingError(VoiceForgeError):
    """Streaming operation failed"""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.session_id = session_id


# Utility functions for error handling
def handle_api_error(response_data: Dict[str, Any], status_code: int) -> VoiceForgeError:
    """Convert API error response to appropriate exception"""
    
    error_code = response_data.get("code")
    message = response_data.get("message", "API request failed")
    details = response_data.get("details", {})
    
    # Map specific error codes to exception types
    error_mapping = {
        "AUTHENTICATION_FAILED": AuthenticationError,
        "INVALID_TOKEN": AuthenticationError,
        "TOKEN_EXPIRED": AuthenticationError,
        "INSUFFICIENT_PERMISSIONS": AuthorizationError,
        "RATE_LIMIT_EXCEEDED": RateLimitError,
        "VALIDATION_ERROR": ValidationError,
        "TRANSCRIPTION_FAILED": TranscriptionError,
        "FILE_TOO_LARGE": FileError,
        "INVALID_AUDIO_FORMAT": AudioError,
        "QUOTA_EXCEEDED": QuotaExceededError,
        "INSUFFICIENT_CREDITS": QuotaExceededError,
        "MODEL_UNAVAILABLE": ModelUnavailableError,
        "STREAMING_ERROR": StreamingError,
        "WEBSOCKET_ERROR": WebSocketError,
    }
    
    # Handle rate limiting with retry-after
    if error_code == "RATE_LIMIT_EXCEEDED":
        retry_after = details.get("retry_after")
        return RateLimitError(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            retry_after=retry_after
        )
    
    # Handle validation errors with field details
    if error_code == "VALIDATION_ERROR":
        field_errors = details.get("field_errors", {})
        return ValidationError(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            field_errors=field_errors
        )
    
    # Handle transcription errors with job ID
    if error_code == "TRANSCRIPTION_FAILED":
        job_id = details.get("job_id")
        return TranscriptionError(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            job_id=job_id
        )
    
    # Handle file errors with filename
    if error_code in ["FILE_TOO_LARGE", "INVALID_AUDIO_FORMAT"]:
        filename = details.get("filename")
        error_class = AudioError if "AUDIO" in error_code else FileError
        return error_class(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            filename=filename
        )
    
    # Handle quota errors with credits info
    if error_code in ["QUOTA_EXCEEDED", "INSUFFICIENT_CREDITS"]:
        credits_remaining = details.get("credits_remaining")
        return QuotaExceededError(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            credits_remaining=credits_remaining
        )
    
    # Handle model unavailable errors
    if error_code == "MODEL_UNAVAILABLE":
        model_id = details.get("model_id")
        return ModelUnavailableError(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            model_id=model_id
        )
    
    # Handle streaming/websocket errors
    if error_code in ["STREAMING_ERROR", "WEBSOCKET_ERROR"]:
        session_id = details.get("session_id")
        connection_id = details.get("connection_id")
        error_class = StreamingError if error_code == "STREAMING_ERROR" else WebSocketError
        return error_class(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details,
            session_id=session_id,
            connection_id=connection_id
        )
    
    # Use specific exception type if available
    exception_class = error_mapping.get(error_code, APIError)
    
    return exception_class(
        message=message,
        error_code=error_code,
        status_code=status_code,
        details=details
    )


def handle_http_error(status_code: int, response_text: str = "") -> VoiceForgeError:
    """Convert HTTP status code to appropriate exception"""
    
    if status_code == 401:
        return AuthenticationError("Authentication required or token invalid")
    elif status_code == 403:
        return AuthorizationError("Insufficient permissions")
    elif status_code == 429:
        return RateLimitError("Rate limit exceeded")
    elif status_code == 422:
        return ValidationError("Request validation failed")
    elif 400 <= status_code < 500:
        return APIError(f"Client error: {status_code}")
    elif 500 <= status_code < 600:
        return APIError(f"Server error: {status_code}")
    else:
        return NetworkError(f"HTTP error: {status_code}")


# Exception context manager for better error handling
class ErrorContext:
    """Context manager for enhanced error handling"""
    
    def __init__(self, operation: str):
        self.operation = operation
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            # Enhance exception with operation context
            if hasattr(exc_val, "details"):
                exc_val.details["operation"] = self.operation
            else:
                # Wrap non-VoiceForge exceptions
                if not isinstance(exc_val, VoiceForgeError):
                    wrapped_exc = VoiceForgeError(
                        f"Error during {self.operation}: {str(exc_val)}",
                        details={"operation": self.operation, "original_error": str(exc_val)}
                    )
                    # Replace the exception
                    raise wrapped_exc from exc_val
        return False  # Don't suppress the exception