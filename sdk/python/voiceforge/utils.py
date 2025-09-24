"""
VoiceForge SDK Utilities

Helper functions and utilities for working with audio files and the API.
"""

import asyncio
import hashlib
import io
import mimetypes
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple, BinaryIO
from urllib.parse import urlparse

from .exceptions import AudioError, FileError, ValidationError
from .models import AudioFormat, TranscriptionOptions


# Audio format detection and validation
SUPPORTED_FORMATS = {
    ".wav": AudioFormat.WAV,
    ".mp3": AudioFormat.MP3, 
    ".m4a": AudioFormat.M4A,
    ".flac": AudioFormat.FLAC,
    ".ogg": AudioFormat.OGG,
    ".webm": AudioFormat.WEBM,
}

MIME_TYPE_MAPPING = {
    "audio/wav": AudioFormat.WAV,
    "audio/wave": AudioFormat.WAV,
    "audio/mpeg": AudioFormat.MP3,
    "audio/mp3": AudioFormat.MP3,
    "audio/mp4": AudioFormat.M4A,
    "audio/x-m4a": AudioFormat.M4A,
    "audio/flac": AudioFormat.FLAC,
    "audio/ogg": AudioFormat.OGG,
    "audio/webm": AudioFormat.WEBM,
}


def detect_audio_format(file_path: Union[str, Path]) -> Optional[AudioFormat]:
    """
    Detect audio format from file extension
    
    Args:
        file_path: Path to audio file
        
    Returns:
        AudioFormat enum value or None if not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    return SUPPORTED_FORMATS.get(extension)


def detect_audio_format_from_mime(mime_type: str) -> Optional[AudioFormat]:
    """
    Detect audio format from MIME type
    
    Args:
        mime_type: MIME type string
        
    Returns:
        AudioFormat enum value or None if not supported
    """
    return MIME_TYPE_MAPPING.get(mime_type.lower())


def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate audio file and return metadata
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with file metadata
        
    Raises:
        FileError: If file doesn't exist or isn't accessible
        AudioError: If file format is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileError(f"File not found: {file_path}", filename=str(file_path))
    
    if not file_path.is_file():
        raise FileError(f"Path is not a file: {file_path}", filename=str(file_path))
    
    # Check file size
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise AudioError(f"File is empty: {file_path}", filename=str(file_path))
    
    if file_size > 100 * 1024 * 1024:  # 100MB limit
        raise AudioError(
            f"File too large: {file_size} bytes (max: 100MB)",
            filename=str(file_path)
        )
    
    # Detect format
    audio_format = detect_audio_format(file_path)
    if not audio_format:
        raise AudioError(
            f"Unsupported audio format: {file_path.suffix}",
            filename=str(file_path)
        )
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    return {
        "path": str(file_path),
        "filename": file_path.name,
        "size": file_size,
        "format": audio_format,
        "mime_type": mime_type,
    }


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Calculate hash of file contents
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def estimate_transcription_time(
    file_size: int,
    duration_seconds: Optional[float] = None,
    model_type: str = "whisper-base"
) -> float:
    """
    Estimate transcription processing time
    
    Args:
        file_size: File size in bytes
        duration_seconds: Audio duration in seconds (if known)
        model_type: Model type being used
        
    Returns:
        Estimated processing time in seconds
    """
    # Base processing ratios (processing_time / audio_duration)
    model_ratios = {
        "whisper-tiny": 0.05,
        "whisper-base": 0.1,
        "whisper-small": 0.15,
        "whisper-medium": 0.2,
        "whisper-large": 0.3,
        "canary": 0.08,
    }
    
    ratio = model_ratios.get(model_type, 0.15)  # Default ratio
    
    if duration_seconds:
        return duration_seconds * ratio
    
    # Rough estimate based on file size (assuming ~1MB per minute for compressed audio)
    estimated_duration = file_size / (1024 * 1024) * 60
    return estimated_duration * ratio


def split_audio_into_chunks(
    audio_data: bytes,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> List[bytes]:
    """
    Split audio data into smaller chunks for streaming
    
    Args:
        audio_data: Audio data as bytes
        chunk_size: Size of each chunk in bytes
        
    Returns:
        List of audio chunks
    """
    chunks = []
    offset = 0
    
    while offset < len(audio_data):
        chunk = audio_data[offset:offset + chunk_size]
        chunks.append(chunk)
        offset += chunk_size
    
    return chunks


def merge_transcription_segments(segments: List[Dict[str, Any]]) -> str:
    """
    Merge multiple transcription segments into a single text
    
    Args:
        segments: List of transcription segments with 'text' field
        
    Returns:
        Combined transcript text
    """
    return " ".join(segment.get("text", "").strip() for segment in segments if segment.get("text"))


def format_timestamp(seconds: float, format: str = "srt") -> str:
    """
    Format timestamp for subtitle formats
    
    Args:
        seconds: Time in seconds
        format: Output format ('srt', 'vtt', 'ass')
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    if format.lower() == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    elif format.lower() == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    elif format.lower() == "ass":
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export_to_srt(words: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """
    Export word-level timestamps to SRT subtitle format
    
    Args:
        words: List of word dictionaries with 'text', 'start_time', 'end_time'
        output_path: Optional path to save SRT file
        
    Returns:
        SRT content as string
    """
    if not words:
        return ""
    
    srt_content = []
    subtitle_index = 1
    
    # Group words into subtitle chunks (max 10 words or 5 seconds)
    current_chunk = []
    chunk_start = None
    
    for word in words:
        if not current_chunk:
            chunk_start = word.get("start_time", 0)
        
        current_chunk.append(word["text"])
        
        # Create subtitle if chunk is full or time gap is large
        should_break = (
            len(current_chunk) >= 10 or
            (word.get("end_time", 0) - chunk_start) >= 5.0 or
            word == words[-1]  # Last word
        )
        
        if should_break:
            start_time = format_timestamp(chunk_start, "srt")
            end_time = format_timestamp(word.get("end_time", chunk_start + 1), "srt")
            text = " ".join(current_chunk)
            
            srt_content.append(f"{subtitle_index}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # Empty line
            
            subtitle_index += 1
            current_chunk = []
    
    srt_text = "\n".join(srt_content)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
    
    return srt_text


def export_to_vtt(words: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """
    Export word-level timestamps to WebVTT format
    
    Args:
        words: List of word dictionaries with timing info
        output_path: Optional path to save VTT file
        
    Returns:
        VTT content as string
    """
    if not words:
        return "WEBVTT\n\n"
    
    vtt_content = ["WEBVTT", ""]
    
    # Group words into cues
    current_chunk = []
    chunk_start = None
    
    for word in words:
        if not current_chunk:
            chunk_start = word.get("start_time", 0)
        
        current_chunk.append(word["text"])
        
        should_break = (
            len(current_chunk) >= 10 or
            (word.get("end_time", 0) - chunk_start) >= 5.0 or
            word == words[-1]
        )
        
        if should_break:
            start_time = format_timestamp(chunk_start, "vtt")
            end_time = format_timestamp(word.get("end_time", chunk_start + 1), "vtt")
            text = " ".join(current_chunk)
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
            
            current_chunk = []
    
    vtt_text = "\n".join(vtt_content)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(vtt_text)
    
    return vtt_text


def create_transcription_options(
    model: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs
) -> TranscriptionOptions:
    """
    Create TranscriptionOptions with validation
    
    Args:
        model: Model name
        language: Language code
        **kwargs: Additional options
        
    Returns:
        TranscriptionOptions instance
        
    Raises:
        ValidationError: If options are invalid
    """
    try:
        return TranscriptionOptions(
            model=model,
            language=language,
            **kwargs
        )
    except Exception as e:
        raise ValidationError(f"Invalid transcription options: {e}")


def download_file(url: str, output_path: Optional[str] = None) -> str:
    """
    Download file from URL
    
    Args:
        url: URL to download from
        output_path: Local path to save file (optional)
        
    Returns:
        Path to downloaded file
        
    Raises:
        FileError: If download fails
    """
    try:
        import httpx
        
        if not output_path:
            # Create temporary file
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "downloaded_audio"
            output_path = os.path.join(tempfile.gettempdir(), filename)
        
        with httpx.stream("GET", url) as response:
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        
        return output_path
        
    except Exception as e:
        raise FileError(f"Failed to download file: {e}")


async def async_download_file(url: str, output_path: Optional[str] = None) -> str:
    """
    Asynchronously download file from URL
    
    Args:
        url: URL to download from
        output_path: Local path to save file (optional)
        
    Returns:
        Path to downloaded file
    """
    import httpx
    
    if not output_path:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "downloaded_audio"
        output_path = os.path.join(tempfile.gettempdir(), filename)
    
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
    
    return output_path


def batch_process_files(
    file_paths: List[Union[str, Path]],
    batch_size: int = 10
) -> List[List[Path]]:
    """
    Split file list into batches for processing
    
    Args:
        file_paths: List of file paths
        batch_size: Maximum files per batch
        
    Returns:
        List of file path batches
    """
    file_paths = [Path(p) for p in file_paths]
    batches = []
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        batches.append(batch)
    
    return batches


def find_audio_files(
    directory: Union[str, Path],
    recursive: bool = True,
    include_formats: Optional[List[AudioFormat]] = None
) -> List[Path]:
    """
    Find all audio files in a directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        include_formats: List of formats to include (default: all supported)
        
    Returns:
        List of audio file paths
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        raise FileError(f"Directory not found: {directory}")
    
    if include_formats is None:
        include_formats = list(AudioFormat)
    
    extensions = [
        ext for ext, fmt in SUPPORTED_FORMATS.items()
        if fmt in include_formats
    ]
    
    audio_files = []
    
    if recursive:
        for ext in extensions:
            audio_files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
    
    return sorted(audio_files)


def create_progress_callback(description: str = "Processing") -> Tuple[Any, Any]:
    """
    Create a progress callback function with rich progress bar
    
    Args:
        description: Description for progress bar
        
    Returns:
        Tuple of (progress_callback, progress_bar)
    """
    try:
        from rich.progress import Progress, TaskID
        
        progress = Progress()
        task = progress.add_task(description, total=100)
        
        def callback(percent: float):
            progress.update(task, completed=int(percent * 100))
        
        return callback, progress
        
    except ImportError:
        # Fallback to simple print-based progress
        def simple_callback(percent: float):
            print(f"\r{description}: {percent:.1%}", end="", flush=True)
            if percent >= 1.0:
                print()  # New line when complete
        
        return simple_callback, None


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    async def async_wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries:
                    raise e
                
                delay = base_delay * (backoff_factor ** attempt)
                await asyncio.sleep(delay)
    
    def sync_wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries:
                    raise e
                
                delay = base_delay * (backoff_factor ** attempt)
                time.sleep(delay)
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper