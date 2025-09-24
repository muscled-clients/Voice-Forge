"""
VoiceForge Python SDK Client

Main client class for interacting with the VoiceForge API.
"""

import asyncio
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, AsyncIterator, Callable
from urllib.parse import urljoin

import httpx
import websockets
from pydantic import ValidationError as PydanticValidationError

from .models import (
    TranscriptionJob,
    TranscriptionOptions,
    StreamingResult,
    ModelInfo,
    UserInfo,
    UserAnalytics,
    QuotaInfo,
    APIKeyInfo,
    BatchJob,
    PaginatedResponse,
    UploadInfo,
    TranscriptionStatus,
    AudioFormat,
)
from .exceptions import (
    VoiceForgeError,
    APIError,
    AuthenticationError,
    NetworkError,
    TimeoutError,
    ValidationError,
    handle_api_error,
    handle_http_error,
    ErrorContext,
)


class VoiceForgeClient:
    """
    VoiceForge API Client
    
    Main interface for interacting with the VoiceForge Speech-to-Text API.
    
    Example:
        ```python
        import voiceforge
        
        # Initialize client
        client = voiceforge.VoiceForgeClient(api_key="your_api_key")
        
        # Transcribe a file
        job = await client.transcribe_file("audio.wav")
        print(f"Transcript: {job.transcript}")
        
        # Stream real-time transcription
        async for result in client.stream_transcription():
            print(f"Partial: {result.transcript}")
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize VoiceForge client
        
        Args:
            api_key: API key for authentication (can also set VOICEFORGE_API_KEY env var)
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            user_agent: Custom user agent string
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("VOICEFORGE_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key is required. Set VOICEFORGE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up HTTP client
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": user_agent or f"voiceforge-python/1.0.0",
            "Accept": "application/json",
        }
        
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        
        # WebSocket connection for streaming
        self._ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self._streaming_session_id: Optional[str] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Clean up client resources"""
        if self.http_client:
            await self.http_client.aclose()
        
        if self._ws_connection:
            await self._ws_connection.close()
    
    # Authentication and User Management
    
    async def get_user_info(self) -> UserInfo:
        """Get current user information"""
        with ErrorContext("get_user_info"):
            response = await self._request("GET", "/api/v1/auth/me")
            return UserInfo(**response)
    
    async def get_user_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> UserAnalytics:
        """Get user usage analytics"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if language:
            params["language"] = language
        if model:
            params["model"] = model
        
        with ErrorContext("get_user_analytics"):
            response = await self._request("GET", "/api/v1/analytics/user", params=params)
            return UserAnalytics(**response)
    
    async def get_quota_info(self) -> QuotaInfo:
        """Get user quota and usage information"""
        with ErrorContext("get_quota_info"):
            response = await self._request("GET", "/api/v1/auth/quota")
            return QuotaInfo(**response)
    
    # Models and Capabilities
    
    async def list_models(self) -> List[ModelInfo]:
        """Get list of available transcription models"""
        with ErrorContext("list_models"):
            response = await self._request("GET", "/api/v1/models")
            return [ModelInfo(**model) for model in response["models"]]
    
    async def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model"""
        with ErrorContext("get_model_info"):
            response = await self._request("GET", f"/api/v1/models/{model_id}")
            return ModelInfo(**response)
    
    # Transcription - YouTube
    
    async def transcribe_youtube(
        self,
        url: str,
        language: str = "auto",
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio from YouTube video
        
        Args:
            url: YouTube video URL
            language: Language code or "auto" for detection
            wait_for_completion: Whether to wait for completion
            
        Returns:
            Transcription results
        """
        with ErrorContext("transcribe_youtube"):
            response = await self._request(
                "POST",
                "/api/v1/transcribe/youtube",
                json={"url": url, "language": language}
            )
            return response
    
    # Transcription - Single File
    
    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        options: Optional[TranscriptionOptions] = None,
        wait_for_completion: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TranscriptionJob:
        """
        Transcribe an audio file
        
        Args:
            file_path: Path to audio file
            options: Transcription options
            wait_for_completion: Whether to wait for transcription to complete
            progress_callback: Optional callback for progress updates
            
        Returns:
            TranscriptionJob with results (if wait_for_completion=True)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        with ErrorContext("transcribe_file"):
            # Create transcription job
            job = await self._upload_and_transcribe(file_path, options)
            
            if wait_for_completion:
                return await self._wait_for_completion(job.id, progress_callback)
            
            return job
    
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        options: Optional[TranscriptionOptions] = None,
        wait_for_completion: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TranscriptionJob:
        """
        Transcribe audio from bytes
        
        Args:
            audio_bytes: Audio data as bytes
            filename: Original filename (for format detection)
            options: Transcription options
            wait_for_completion: Whether to wait for completion
            progress_callback: Optional progress callback
            
        Returns:
            TranscriptionJob with results
        """
        with ErrorContext("transcribe_bytes"):
            job = await self._upload_bytes_and_transcribe(audio_bytes, filename, options)
            
            if wait_for_completion:
                return await self._wait_for_completion(job.id, progress_callback)
            
            return job
    
    async def transcribe_url(
        self,
        audio_url: str,
        options: Optional[TranscriptionOptions] = None,
        wait_for_completion: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> TranscriptionJob:
        """
        Transcribe audio from URL
        
        Args:
            audio_url: URL to audio file
            options: Transcription options
            wait_for_completion: Whether to wait for completion
            progress_callback: Optional progress callback
            
        Returns:
            TranscriptionJob with results
        """
        with ErrorContext("transcribe_url"):
            data = {"audio_url": audio_url}
            
            if options:
                data.update(options.model_dump(exclude_none=True))
            
            response = await self._request("POST", "/api/v1/transcribe/url", json=data)
            job = TranscriptionJob(**response)
            
            if wait_for_completion:
                return await self._wait_for_completion(job.id, progress_callback)
            
            return job
    
    # Transcription - Batch Processing
    
    async def transcribe_batch(
        self,
        file_paths: List[Union[str, Path]],
        options: Optional[TranscriptionOptions] = None,
        wait_for_completion: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> BatchJob:
        """
        Transcribe multiple files in batch
        
        Args:
            file_paths: List of audio file paths
            options: Transcription options (applied to all files)
            wait_for_completion: Whether to wait for all jobs to complete
            progress_callback: Optional callback for job progress (job_id, progress)
            
        Returns:
            BatchJob with individual TranscriptionJob results
        """
        with ErrorContext("transcribe_batch"):
            # Upload all files
            jobs = []
            for file_path in file_paths:
                job = await self.transcribe_file(
                    file_path, options, wait_for_completion=False
                )
                jobs.append(job)
            
            batch_job = BatchJob(
                batch_id=f"batch_{int(time.time())}",
                jobs=jobs,
                created_at=jobs[0].created_at if jobs else None,
                total_jobs=len(jobs),
                completed_jobs=0,
                failed_jobs=0,
                status="processing"
            )
            
            if wait_for_completion:
                return await self._wait_for_batch_completion(batch_job, progress_callback)
            
            return batch_job
    
    # Job Management
    
    async def get_job(self, job_id: str) -> TranscriptionJob:
        """Get transcription job by ID"""
        with ErrorContext("get_job"):
            response = await self._request("GET", f"/api/v1/transcribe/jobs/{job_id}")
            return TranscriptionJob(**response)
    
    async def list_jobs(
        self,
        status: Optional[TranscriptionStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> PaginatedResponse:
        """List user's transcription jobs"""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        
        with ErrorContext("list_jobs"):
            response = await self._request("GET", "/api/v1/transcribe/jobs", params=params)
            
            jobs = [TranscriptionJob(**job) for job in response["jobs"]]
            return PaginatedResponse(
                items=jobs,
                total_count=response["total_count"],
                page=offset // limit + 1,
                per_page=limit,
                total_pages=(response["total_count"] + limit - 1) // limit,
                has_next=offset + limit < response["total_count"],
                has_prev=offset > 0,
            )
    
    async def cancel_job(self, job_id: str) -> TranscriptionJob:
        """Cancel a transcription job"""
        with ErrorContext("cancel_job"):
            response = await self._request("POST", f"/api/v1/transcribe/jobs/{job_id}/cancel")
            return TranscriptionJob(**response)
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a transcription job"""
        with ErrorContext("delete_job"):
            await self._request("DELETE", f"/api/v1/transcribe/jobs/{job_id}")
            return True
    
    # Real-time Streaming
    
    async def stream_transcription(
        self,
        options: Optional[TranscriptionOptions] = None,
        audio_format: AudioFormat = AudioFormat.WAV,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
    ) -> AsyncIterator[StreamingResult]:
        """
        Start real-time streaming transcription
        
        Args:
            options: Transcription options
            audio_format: Audio format for streaming
            sample_rate: Audio sample rate
            chunk_duration: Duration of each audio chunk in seconds
            
        Yields:
            StreamingResult objects with partial and final transcripts
        """
        with ErrorContext("stream_transcription"):
            ws_url = self.base_url.replace("http", "ws") + "/api/v1/transcribe/stream"
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with websockets.connect(ws_url, extra_headers=headers) as websocket:
                # Send configuration
                config = {
                    "audio_format": audio_format.value,
                    "sample_rate": sample_rate,
                    "chunk_duration": chunk_duration,
                }
                
                if options:
                    config.update(options.model_dump(exclude_none=True))
                
                await websocket.send(json.dumps({"type": "config", "data": config}))
                
                # Listen for results
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data["type"] == "result":
                            yield StreamingResult(**data["data"])
                        elif data["type"] == "error":
                            raise APIError(data["data"]["message"])
                        elif data["type"] == "end":
                            break
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        raise APIError(f"Invalid WebSocket message: {e}")
    
    async def send_audio_chunk(self, audio_chunk: bytes):
        """Send audio chunk for streaming transcription"""
        if not self._ws_connection:
            raise VoiceForgeError("No active streaming session")
        
        await self._ws_connection.send(audio_chunk)
    
    async def end_streaming_session(self):
        """End the current streaming session"""
        if self._ws_connection:
            await self._ws_connection.send(json.dumps({"type": "end"}))
            await self._ws_connection.close()
            self._ws_connection = None
    
    # API Keys Management
    
    async def list_api_keys(self) -> List[APIKeyInfo]:
        """List user's API keys"""
        with ErrorContext("list_api_keys"):
            response = await self._request("GET", "/api/v1/auth/api-keys")
            return [APIKeyInfo(**key) for key in response["api_keys"]]
    
    async def create_api_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
    ) -> APIKeyInfo:
        """Create a new API key"""
        data = {"name": name}
        if permissions:
            data["permissions"] = permissions
        
        with ErrorContext("create_api_key"):
            response = await self._request("POST", "/api/v1/auth/api-keys", json=data)
            return APIKeyInfo(**response)
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        with ErrorContext("revoke_api_key"):
            await self._request("DELETE", f"/api/v1/auth/api-keys/{key_id}")
            return True
    
    # Private methods
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make HTTP request with error handling and retries"""
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.http_client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                )
                
                # Handle successful responses
                if 200 <= response.status_code < 300:
                    if response.headers.get("content-type", "").startswith("application/json"):
                        return response.json()
                    return response.text
                
                # Handle error responses
                await self._handle_error_response(response)
                
            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise TimeoutError(f"Request timed out after {self.timeout}s") from e
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except httpx.NetworkError as e:
                if attempt == self.max_retries:
                    raise NetworkError(f"Network error: {str(e)}") from e
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                if attempt == self.max_retries:
                    raise VoiceForgeError(f"Unexpected error: {str(e)}") from e
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def _handle_error_response(self, response: httpx.Response):
        """Handle error responses from the API"""
        try:
            error_data = response.json()
            raise handle_api_error(error_data, response.status_code)
        except (json.JSONDecodeError, KeyError):
            raise handle_http_error(response.status_code, response.text)
    
    async def _upload_and_transcribe(
        self,
        file_path: Path,
        options: Optional[TranscriptionOptions] = None,
    ) -> TranscriptionJob:
        """Upload file and create transcription job"""
        
        # Detect content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        
        # Prepare form data
        files = {
            "audio": (file_path.name, file_path.open("rb"), content_type)
        }
        
        data = {}
        if options:
            data.update(options.model_dump(exclude_none=True))
        
        response = await self._request(
            "POST", "/api/v1/transcribe", data=data, files=files
        )
        
        return TranscriptionJob(**response)
    
    async def _upload_bytes_and_transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        options: Optional[TranscriptionOptions] = None,
    ) -> TranscriptionJob:
        """Upload audio bytes and create transcription job"""
        
        content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            content_type = "application/octet-stream"
        
        files = {
            "audio": (filename, audio_bytes, content_type)
        }
        
        data = {}
        if options:
            data.update(options.model_dump(exclude_none=True))
        
        response = await self._request(
            "POST", "/api/v1/transcribe", data=data, files=files
        )
        
        return TranscriptionJob(**response)
    
    async def _wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        poll_interval: float = 2.0,
        max_wait_time: float = 3600.0,  # 1 hour
    ) -> TranscriptionJob:
        """Wait for transcription job to complete"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            job = await self.get_job(job_id)
            
            if job.status == TranscriptionStatus.COMPLETED:
                if progress_callback:
                    progress_callback(1.0)
                return job
            
            elif job.status == TranscriptionStatus.FAILED:
                raise TranscriptionError(
                    f"Transcription failed: {job.error}",
                    job_id=job_id
                )
            
            elif job.status == TranscriptionStatus.CANCELLED:
                raise TranscriptionError(
                    "Transcription was cancelled",
                    job_id=job_id
                )
            
            # Estimate progress (rough estimate based on processing time)
            if progress_callback and job.status == TranscriptionStatus.PROCESSING:
                elapsed = time.time() - job.created_at.timestamp()
                estimated_duration = job.audio_duration or 60  # Default estimate
                progress = min(0.9, elapsed / (estimated_duration * 0.1))  # Assume 10% of audio duration
                progress_callback(progress)
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Transcription did not complete within {max_wait_time}s")
    
    async def _wait_for_batch_completion(
        self,
        batch_job: BatchJob,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        poll_interval: float = 5.0,
        max_wait_time: float = 7200.0,  # 2 hours
    ) -> BatchJob:
        """Wait for batch transcription to complete"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check status of all jobs
            completed = 0
            failed = 0
            
            for i, job in enumerate(batch_job.jobs):
                if job.status in [TranscriptionStatus.COMPLETED, TranscriptionStatus.FAILED]:
                    continue
                
                # Update job status
                updated_job = await self.get_job(job.id)
                batch_job.jobs[i] = updated_job
                
                if progress_callback:
                    progress = 0.0
                    if updated_job.status == TranscriptionStatus.COMPLETED:
                        progress = 1.0
                    elif updated_job.status == TranscriptionStatus.PROCESSING:
                        progress = 0.5  # Rough estimate
                    
                    progress_callback(job.id, progress)
            
            # Count completed and failed jobs
            for job in batch_job.jobs:
                if job.status == TranscriptionStatus.COMPLETED:
                    completed += 1
                elif job.status == TranscriptionStatus.FAILED:
                    failed += 1
            
            batch_job.completed_jobs = completed
            batch_job.failed_jobs = failed
            
            # Check if batch is complete
            if completed + failed == batch_job.total_jobs:
                batch_job.status = "completed" if failed == 0 else "partially_failed"
                return batch_job
            
            await asyncio.sleep(poll_interval)
        
        batch_job.status = "timeout"
        raise TimeoutError(f"Batch transcription did not complete within {max_wait_time}s")