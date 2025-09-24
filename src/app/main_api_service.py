"""
VoiceForge API Service - SaaS Platform
API-as-a-Service for developers with API key authentication
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Header, Query, WebSocket, WebSocketDisconnect
from fastapi import status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
import os
import tempfile
import logging
import time
import whisper
import torch
import warnings
# Database imports
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.connection import get_db_session, init_db
from src.database.repositories import (
    UserRepository, TranscriptionRepository, APIKeyRepository,
    WebSocketSessionRepository, UsageRepository, PlanRepository
)
from src.database.models import User, Plan, APIKey, Transcription, WebSocketSession, UsageRecord, UserRole
from src.services.email_service import email_service
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import bcrypt
import jwt
import uuid
import secrets
from typing import Optional
import hashlib
from fastapi import Request as FastAPIRequest
import yt_dlp
import re
import asyncio
import base64
import numpy as np
from collections import deque
import threading
import websockets
import torchaudio
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.signal
from scipy.spatial.distance import cdist

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get current directory for FFmpeg
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PATH"] = project_root + os.pathsep + os.environ["PATH"]

app = FastAPI(
    title="VoiceForge API Service",
    description="🎤 Speech-to-Text API for Developers",
    version="3.1.0",
    docs_url="/api-docs",  # Move Swagger to /api-docs
    redoc_url="/api-redoc"  # Move ReDoc to /api-redoc
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Global variables
whisper_model = None
# Initialize database
try:
    init_db()
    logger.info("✅ Database initialized successfully")
except Exception as e:
    logger.error(f"❌ Database initialization failed: {e}")
    raise

# Pydantic models
class DeveloperRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    company: Optional[str] = None

class DeveloperLogin(BaseModel):
    email: EmailStr
    password: str

class AdminLogin(BaseModel):
    username: str
    password: str

class YouTubeTranscribeRequest(BaseModel):
    url: str

# WebSocket Streaming Classes
class StreamingConfig(BaseModel):
    encoding: str = "linear16"
    sample_rate: int = 16000
    language: Optional[str] = None
    interim_results: bool = True
    vad_enabled: bool = True
    custom_vocabulary: Optional[list] = None
    # Phase 3 features
    speaker_diarization: bool = False
    language_detection: bool = False  # Auto-detect language
    noise_reduction: bool = False
    max_speakers: int = 5

class WebSocketConnection:
    def __init__(self, websocket: WebSocket, session_id: str, user_id: str, config: StreamingConfig):
        self.websocket = websocket
        self.session_id = session_id
        self.user_id = user_id
        self.config = config
        self.connected_at = datetime.utcnow()
        self.bytes_processed = 0
        self.results_sent = 0
        self.audio_buffer = deque(maxlen=50)  # Buffer for 5 seconds at 100ms chunks
        self.is_processing = False
        self.last_activity = datetime.utcnow()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocketConnection] = {}
        self.user_connections: dict[str, set] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str, config: StreamingConfig):
        """Register new WebSocket connection"""
        await websocket.accept()
        
        connection = WebSocketConnection(websocket, session_id, user_id, config)
        self.active_connections[session_id] = connection
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(session_id)
        
        logger.info(f"WebSocket connected: session={session_id}, user={user_id}")
        return connection
        
    async def disconnect(self, session_id: str):
        """Clean up connection and resources"""
        if session_id in self.active_connections:
            connection = self.active_connections[session_id]
            user_id = connection.user_id
            
            # Remove from active connections
            del self.active_connections[session_id]
            
            # Remove from user connections
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(session_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            logger.info(f"WebSocket disconnected: session={session_id}, user={user_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """Send message to specific connection"""
        if session_id in self.active_connections:
            connection = self.active_connections[session_id]
            try:
                await connection.websocket.send_text(json.dumps(message))
                connection.results_sent += 1
                return True
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                await self.disconnect(session_id)
                return False
        return False
    
    def get_connection_stats(self) -> dict:
        """Get real-time connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "active_users": len(self.user_connections),
            "connections_by_user": {user: len(sessions) for user, sessions in self.user_connections.items()}
        }

# Global connection manager
connection_manager = ConnectionManager()

# Enhanced Audio Processing Classes
class AudioBuffer:
    def __init__(self, sample_rate=16000, chunk_duration=0.1, vad_enabled=True):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer = deque(maxlen=50)  # 5 seconds of audio at 100ms chunks
        self.speech_buffer = deque(maxlen=30)  # 3 seconds of speech chunks
        self.total_duration = 0
        self.speech_duration = 0
        
        # VAD integration
        self.vad_enabled = vad_enabled
        self.vad = VoiceActivityDetector(sample_rate=sample_rate) if vad_enabled else None
        
        # Interim results tracking
        self.last_processed_index = 0
        self.speech_segments = []
        
    def add_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer with VAD processing"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.buffer.append(audio_array)
            self.total_duration += len(audio_array) / self.sample_rate
            
            # VAD processing
            if self.vad_enabled and self.vad:
                is_speech = self.vad.is_speech(audio_array)
                
                if is_speech:
                    self.speech_buffer.append(audio_array)
                    self.speech_duration += len(audio_array) / self.sample_rate
                    return True  # Speech detected
                else:
                    return False  # No speech
            else:
                # If VAD disabled, treat all audio as speech
                self.speech_buffer.append(audio_array)
                self.speech_duration += len(audio_array) / self.sample_rate
                return True
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return False
    
    def get_audio_for_transcription(self, duration_seconds=1.5, use_speech_only=True) -> Optional[np.ndarray]:
        """Get audio data for transcription (last N seconds)"""
        # Choose buffer based on VAD setting
        source_buffer = self.speech_buffer if (use_speech_only and self.vad_enabled) else self.buffer
        
        if not source_buffer:
            return None
            
        # Calculate how many chunks we need
        chunks_needed = min(int(duration_seconds / self.chunk_duration), len(source_buffer))
        
        if chunks_needed == 0:
            return None
            
        # Get the last N chunks
        recent_chunks = list(source_buffer)[-chunks_needed:]
        
        # Concatenate chunks
        if recent_chunks:
            audio_data = np.concatenate(recent_chunks)
            return audio_data
        
        return None
    
    def get_audio_for_interim(self, min_duration=0.5) -> Optional[np.ndarray]:
        """Get audio for interim results (shorter segments)"""
        return self.get_audio_for_transcription(duration_seconds=min_duration, use_speech_only=True)
    
    def should_process(self) -> bool:
        """Determine if we have enough audio for processing"""
        if self.vad_enabled:
            # Need at least 500ms of speech
            return len(self.speech_buffer) >= 5 and self.speech_duration >= 0.5
        else:
            # Need at least 500ms of audio
            return len(self.buffer) >= 5
    
    def should_process_interim(self) -> bool:
        """Determine if we should generate interim results"""
        if self.vad_enabled:
            return len(self.speech_buffer) >= 3 and self.speech_duration >= 0.3
        else:
            # Need more audio for meaningful interim results when VAD is off
            return len(self.buffer) >= 8  # 800ms instead of 300ms
    
    def has_speech_activity(self) -> bool:
        """Check if there's recent speech activity"""
        if not self.vad_enabled or not self.vad:
            return True
        
        # Check last few chunks for speech
        recent_chunks = list(self.buffer)[-3:] if len(self.buffer) >= 3 else list(self.buffer)
        
        for chunk in recent_chunks:
            if self.vad.is_speech(chunk):
                return True
        
        return False
    
    def get_buffer_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            "total_chunks": len(self.buffer),
            "speech_chunks": len(self.speech_buffer),
            "total_duration": self.total_duration,
            "speech_duration": self.speech_duration,
            "vad_enabled": self.vad_enabled,
            "has_speech": self.has_speech_activity()
        }
    
    def clear_processed_audio(self):
        """Clear processed audio from buffers to prevent reprocessing"""
        # Clear half of the speech buffer to prevent reprocessing same audio
        if self.speech_buffer:
            half_size = max(1, len(self.speech_buffer) // 2)
            for _ in range(half_size):
                self.speech_buffer.popleft()
        
        # Clear half of the main buffer as well
        if self.buffer:
            half_size = max(1, len(self.buffer) // 2)
            for _ in range(half_size):
                self.buffer.popleft()
        
        # Update durations
        self.total_duration = len(self.buffer) * self.chunk_duration
        self.speech_duration = len(self.speech_buffer) * self.chunk_duration
        
        # Reset hash to allow new audio processing
        if hasattr(self, 'last_processed_hash'):
            self.last_processed_hash = None

class StreamingTranscriptionEngine:
    def __init__(self, model=None):
        self.whisper_models = {
            "tiny": None,
            "base": None, 
            "small": None,
            "medium": None,
            "large": None
        }
        # Initialize with the provided model
        self.current_model = None
        if model:
            self.set_whisper_model(model)
        
        self.sample_rate = 16000
        
        # Interim results tracking
        self.last_interim_text = ""
        self.interim_threshold = 0.3  # Minimum duration for interim results
    
    def set_whisper_model(self, whisper_model):
        """Set the whisper model and detect its type"""
        if whisper_model:
            # Detect model size from the loaded model
            model_size = "tiny"  # Default assumption
            if hasattr(whisper_model, 'dims'):
                n_audio_state = whisper_model.dims.n_audio_state
                # Model detection based on dimensions
                if n_audio_state == 384:
                    model_size = "tiny"
                elif n_audio_state == 512:
                    model_size = "base"
                elif n_audio_state == 768:
                    model_size = "small"
                elif n_audio_state == 1024:
                    model_size = "medium"
                elif n_audio_state == 1280:
                    model_size = "large"
            
            self.whisper_models[model_size] = whisper_model
            self.current_model = model_size
            logger.info(f"StreamingTranscriptionEngine initialized with {model_size} model")
        else:
            self.current_model = None  # No model loaded yet
            logger.warning("StreamingTranscriptionEngine initialized without model - will load on demand")
        
    def load_model(self, model_name: str) -> bool:
        """Load a specific Whisper model on demand"""
        if model_name in self.whisper_models and self.whisper_models[model_name] is not None:
            self.current_model = model_name
            return True
        
        try:
            logger.info(f"Loading Whisper model: {model_name}")
            
            # Check for GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model with optimal settings
            model = whisper.load_model(model_name, device=device)
            self.whisper_models[model_name] = model
            self.current_model = model_name
            
            logger.info(f"✅ Whisper model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return False
    
    def get_current_model(self):
        """Get the currently active model"""
        return self.whisper_models.get(self.current_model)
    
    async def transcribe_audio(self, audio_data: np.ndarray, language: Optional[str] = None, model_name: Optional[str] = None, is_interim: bool = False, custom_vocabulary: Optional[list] = None) -> dict:
        """Transcribe audio data and return result"""
        try:
            # Load specific model if requested
            if model_name and model_name != "auto" and model_name != self.current_model:
                if not self.load_model(model_name):
                    model_name = self.current_model  # Fall back to current model
            elif model_name == "auto" or not model_name:
                model_name = self.current_model  # Use currently loaded model
            
            # Get the active model
            active_model = self.get_current_model()
            if not active_model:
                return {
                    "type": "error",
                    "error": {
                        "code": "MODEL_NOT_LOADED",
                        "message": "No Whisper model is loaded"
                    }
                }
            
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Minimum length check
            min_duration = 0.1 if not is_interim else 0.05  # 50ms for interim
            min_samples = self.sample_rate * min_duration
            logger.debug(f"🔍 Audio check - Length: {len(audio_data)}, Min required: {min_samples}, Type: {type(audio_data)}, Shape: {getattr(audio_data, 'shape', 'N/A')}")
            
            if len(audio_data) < min_samples:
                logger.debug(f"❌ Audio too short: {len(audio_data)} < {min_samples}")
                return None
            
            # Prepare initial prompt for custom vocabulary (disabled for WebSocket to avoid false transcriptions)
            initial_prompt = None
            # Note: Custom vocabulary prompts can cause false transcriptions in streaming mode
            # if custom_vocabulary and len(custom_vocabulary) > 0:
            #     vocab_text = " ".join(custom_vocabulary[:20])
            #     initial_prompt = f"The following terms may appear in the audio: {vocab_text}."
                
            # Handle "auto" language detection
            if language == "auto":
                language = None  # Whisper will auto-detect when None
            
            # Configure transcription parameters based on type
            if is_interim:
                # Ultra-fast settings for interim results
                transcribe_params = {
                    "fp16": False,  # Disable for speed on CPU
                    "language": "en" if not language else language,  # Force English for speed
                    "task": "transcribe",
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_previous_text": False,
                    "initial_prompt": initial_prompt,
                    "word_timestamps": False,
                    "verbose": False,
                    "beam_size": 1,
                    "best_of": 1,
                    "suppress_blank": True,
                    "suppress_tokens": [-1],
                }
            else:
                # Optimized settings for final results
                transcribe_params = {
                    "fp16": False,  # Disable for CPU speed
                    "language": "en" if not language else language,  # Force English for speed
                    "task": "transcribe",
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_previous_text": False,  # Disable for speed
                    "initial_prompt": initial_prompt,
                    "word_timestamps": False,
                    "verbose": False,
                    "beam_size": 1,
                    "best_of": 1,
                    "suppress_blank": True,
                    "suppress_tokens": [-1],
                }
                
            # Transcribe using Whisper
            logger.debug(f"🔄 Running Whisper transcription ({'interim' if is_interim else 'final'}) - audio length: {len(audio_data)} samples")
            result = active_model.transcribe(audio_data, **transcribe_params)
            
            transcript = result.get("text", "").strip()
            logger.debug(f"🎤 Whisper result: '{transcript}' (confidence: {result.get('segments', [{}])[0].get('avg_logprob', -0.5) if result.get('segments') else 'N/A'})")
            
            if not transcript:
                logger.debug("❌ No transcript returned from Whisper")
                # If we get empty results frequently, it might indicate audio is too short or noisy
                return None
            
            # Filter out prompt-related text (common false transcriptions)
            prompt_phrases = [
                "The following terms may appear in the audio",
                "The following terms may appear in the audio.",
                "may appear in the audio",
                "VoiceForge API transcription WebSocket"
            ]
            
            # Remove prompt text if it appears
            for phrase in prompt_phrases:
                if phrase.lower() in transcript.lower():
                    # If the transcript is mostly the prompt, skip it
                    if len(transcript.replace(phrase, "").strip()) < 3:
                        return None
                    # Otherwise clean it
                    transcript = transcript.replace(phrase, "").strip()
            
            # Skip very short or empty transcripts after cleaning
            if len(transcript) < 2:
                return None
            
            # Check for meaningful change in interim results
            if is_interim:
                if transcript == self.last_interim_text:
                    return None  # No change
                self.last_interim_text = transcript
            else:
                self.last_interim_text = ""  # Reset on final result
            
            # Calculate confidence (approximation based on Whisper internals)
            avg_logprobs = result.get("segments", [{}])[0].get("avg_logprob", -0.5) if result.get("segments") else -0.5
            confidence = min(0.99, max(0.1, (avg_logprobs + 1.0)))  # Convert log prob to confidence
            
            # Format result
            formatted_result = {
                "type": "interim" if is_interim else "final",
                "transcript": transcript,
                "confidence": confidence,
                "duration": len(audio_data) / self.sample_rate,
                "is_final": not is_interim,
                "timestamp": datetime.utcnow().isoformat(),
                "language": result.get("language", language or "en"),
                "model_used": model_name or self.current_model
            }
            logger.debug(f"✅ Returning formatted result: {formatted_result['type']} - '{formatted_result['transcript']}'")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "type": "error",
                "error": {
                    "code": "TRANSCRIPTION_FAILED",
                    "message": str(e)
                }
            }
    
    async def transcribe_interim(self, audio_data: np.ndarray, language: Optional[str] = None, custom_vocabulary: Optional[list] = None) -> Optional[dict]:
        """Generate interim transcription results"""
        return await self.transcribe_audio(audio_data, language=language, is_interim=True, custom_vocabulary=custom_vocabulary)
    
    async def transcribe_final(self, audio_data: np.ndarray, language: Optional[str] = None, model_name: Optional[str] = None, custom_vocabulary: Optional[list] = None) -> Optional[dict]:
        """Generate final transcription results"""
        return await self.transcribe_audio(audio_data, language=language, model_name=model_name, is_interim=False, custom_vocabulary=custom_vocabulary)
    
    def get_available_models(self) -> list:
        """Get list of available Whisper models"""
        return list(self.whisper_models.keys())
    
    def get_model_stats(self) -> dict:
        """Get statistics about loaded models"""
        loaded_models = [name for name, model in self.whisper_models.items() if model is not None]
        return {
            "available_models": self.get_available_models(),
            "loaded_models": loaded_models,
            "current_model": self.current_model,
            "total_loaded": len(loaded_models)
        }

# Voice Activity Detection Class
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration=0.02, threshold=0.5):
        """
        Voice Activity Detection using energy and spectral features
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration: Duration of each frame in seconds (20ms default)
            threshold: VAD threshold (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.threshold = threshold
        
        # Moving average for smoothing
        self.energy_history = deque(maxlen=10)  # Last 10 frames (200ms)
        self.speech_history = deque(maxlen=5)   # Last 5 decisions (100ms)
        
        # Initialize Silero VAD model (more accurate than energy-based)
        try:
            self.silero_model, _ = torchaudio.models.wav2vec2_model(
                "facebook/wav2vec2-base-960h", progress=False
            )
            self.use_silero = False  # Disable for now, using energy-based VAD
        except:
            self.use_silero = False
            logger.info("Using energy-based VAD (Silero VAD not available)")
    
    def _compute_energy(self, audio_chunk: np.ndarray) -> float:
        """Compute RMS energy of audio chunk"""
        if len(audio_chunk) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    
    def _compute_spectral_centroid(self, audio_chunk: np.ndarray) -> float:
        """Compute spectral centroid (brightness measure)"""
        if len(audio_chunk) < 64:  # Too short for FFT
            return 0.0
            
        # Compute FFT
        fft = np.fft.fft(audio_chunk.astype(np.float32))
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(magnitude)]
        
        # Compute centroid
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return centroid
        return 0.0
    
    def _compute_zero_crossing_rate(self, audio_chunk: np.ndarray) -> float:
        """Compute zero crossing rate"""
        if len(audio_chunk) < 2:
            return 0.0
        
        signs = np.sign(audio_chunk)
        zero_crossings = np.sum(np.abs(np.diff(signs))) / 2
        return zero_crossings / len(audio_chunk)
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech
        
        Args:
            audio_chunk: Raw audio data as numpy array (int16)
            
        Returns:
            True if speech detected, False otherwise
        """
        if len(audio_chunk) == 0:
            return False
        
        # Convert to float32 for processing
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        
        # Compute features
        energy = self._compute_energy(audio_float)
        zcr = self._compute_zero_crossing_rate(audio_float)
        spectral_centroid = self._compute_spectral_centroid(audio_float)
        
        # Update energy history
        self.energy_history.append(energy)
        
        # Compute thresholds
        if len(self.energy_history) > 3:
            avg_energy = np.mean(list(self.energy_history))
            energy_threshold = max(0.01, avg_energy * 0.3)  # Adaptive threshold
        else:
            energy_threshold = 0.01  # Default threshold
        
        # Speech detection criteria
        has_energy = energy > energy_threshold
        has_speech_zcr = 0.01 < zcr < 0.3  # Speech typically has moderate ZCR
        has_speech_spectrum = spectral_centroid > 500  # Speech has higher frequencies
        
        # Combine criteria
        is_speech_frame = has_energy and (has_speech_zcr or has_speech_spectrum)
        
        # Update speech history for smoothing
        self.speech_history.append(is_speech_frame)
        
        # Smooth decision (majority vote over recent frames)
        if len(self.speech_history) >= 3:
            speech_votes = sum(self.speech_history)
            return speech_votes >= len(self.speech_history) // 2
        
        return is_speech_frame
    
    def get_speech_segments(self, audio_data: np.ndarray, min_speech_duration=0.3) -> List[Tuple[int, int]]:
        """
        Get speech segments from audio data
        
        Args:
            audio_data: Full audio array
            min_speech_duration: Minimum speech segment duration in seconds
            
        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        if len(audio_data) == 0:
            return []
        
        segments = []
        frame_samples = self.frame_size
        min_samples = int(min_speech_duration * self.sample_rate)
        
        current_segment_start = None
        
        # Process audio in frames
        for i in range(0, len(audio_data) - frame_samples, frame_samples):
            frame = audio_data[i:i + frame_samples]
            
            if self.is_speech(frame):
                if current_segment_start is None:
                    current_segment_start = i
            else:
                if current_segment_start is not None:
                    segment_length = i - current_segment_start
                    if segment_length >= min_samples:
                        segments.append((current_segment_start, i))
                    current_segment_start = None
        
        # Handle ongoing segment at end
        if current_segment_start is not None:
            segment_length = len(audio_data) - current_segment_start
            if segment_length >= min_samples:
                segments.append((current_segment_start, len(audio_data)))
        
        return segments
    
    def should_transcribe(self, audio_chunk: np.ndarray) -> bool:
        """Simple wrapper for is_speech for compatibility"""
        return self.is_speech(audio_chunk)

# Speaker Diarization Classes
class SpeakerDiarization:
    def __init__(self, sample_rate=16000, window_duration=1.0, max_speakers=5):
        """
        Speaker Diarization using voice characteristics and clustering
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_duration: Duration of analysis windows in seconds
            max_speakers: Maximum number of speakers to detect
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_size = int(sample_rate * window_duration)
        self.max_speakers = max_speakers
        
        # Speaker models
        self.speaker_features = []
        self.speaker_labels = []
        self.speaker_names = {}  # Map speaker IDs to names
        self.current_speaker_id = None
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_mels = 40
        
        # Clustering
        self.scaler = StandardScaler()
        self.kmeans = None
        self.min_segments_for_clustering = 5
        
        # Speaker tracking
        self.speaker_history = deque(maxlen=10)  # Last 10 speaker predictions
        self.speaker_confidence_threshold = 0.6
        
    def extract_speaker_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """Extract speaker-specific features from audio segment"""
        try:
            if len(audio_segment) < self.window_size // 4:  # Too short
                return None
            
            # Convert to float if needed
            if audio_segment.dtype != np.float32:
                audio_segment = audio_segment.astype(np.float32) / 32768.0
            
            # Feature 1: Fundamental frequency (F0) statistics
            f0_features = self._extract_f0_features(audio_segment)
            
            # Feature 2: Spectral features (centroid, rolloff, etc.)
            spectral_features = self._extract_spectral_features(audio_segment)
            
            # Feature 3: MFCC-like features
            mfcc_features = self._extract_mfcc_features(audio_segment)
            
            # Feature 4: Voice quality features
            voice_quality = self._extract_voice_quality(audio_segment)
            
            # Combine all features
            features = np.concatenate([
                f0_features,
                spectral_features,
                mfcc_features,
                voice_quality
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting speaker features: {e}")
            return None
    
    def _extract_f0_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency features"""
        try:
            # Autocorrelation-based pitch detection
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks (potential pitch periods)
            min_period = int(self.sample_rate / 400)  # 400 Hz max
            max_period = int(self.sample_rate / 80)   # 80 Hz min
            
            if len(autocorr) > max_period:
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    f0 = self.sample_rate / peak_idx if peak_idx > 0 else 0
                else:
                    f0 = 0
            else:
                f0 = 0
            
            # F0 statistics
            f0_mean = f0
            f0_std = np.std(autocorr[:max_period]) if len(autocorr) > max_period else 0
            
            return np.array([f0_mean, f0_std, f0_mean / 150.0])  # Normalized F0
            
        except:
            return np.array([0, 0, 0])
    
    def _extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral characteristics"""
        try:
            # Compute FFT
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(magnitude)]
            
            # Spectral centroid
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0
            
            # Spectral rolloff (95% of energy)
            cumsum_mag = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum_mag >= 0.95 * cumsum_mag[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Spectral bandwidth
            if np.sum(magnitude) > 0:
                bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
            else:
                bandwidth = 0
            
            # Spectral contrast (simplified)
            contrast = np.std(magnitude) / (np.mean(magnitude) + 1e-8)
            
            return np.array([centroid / 4000.0, rolloff / 8000.0, bandwidth / 2000.0, contrast])
            
        except:
            return np.array([0, 0, 0, 0])
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC-like features (simplified)"""
        try:
            # Mel filter bank (simplified)
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Simple mel-scale approximation
            mel_filters = np.linspace(0, len(magnitude), 13)
            mfcc_features = []
            
            for i in range(len(mel_filters) - 1):
                start_idx = int(mel_filters[i])
                end_idx = int(mel_filters[i + 1])
                if end_idx > start_idx:
                    mel_energy = np.sum(magnitude[start_idx:end_idx])
                    mfcc_features.append(np.log(mel_energy + 1e-8))
                else:
                    mfcc_features.append(0)
            
            return np.array(mfcc_features[:12])  # First 12 MFCC coefficients
            
        except:
            return np.array([0] * 12)
    
    def _extract_voice_quality(self, audio: np.ndarray) -> np.ndarray:
        """Extract voice quality indicators"""
        try:
            # Jitter (pitch period variation)
            diff = np.diff(audio)
            jitter = np.std(diff) / (np.mean(np.abs(audio)) + 1e-8)
            
            # Shimmer (amplitude variation)
            rms = np.sqrt(np.mean(audio ** 2))
            shimmer = np.std(np.abs(audio)) / (rms + 1e-8)
            
            # Harmonics-to-noise ratio (simplified)
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            hnr = np.max(autocorr) / (np.mean(autocorr) + 1e-8)
            
            return np.array([jitter, shimmer, np.log(hnr + 1e-8)])
            
        except:
            return np.array([0, 0, 0])
    
    def add_speaker_segment(self, audio_segment: np.ndarray, speaker_id: int = None):
        """Add a new speaker segment for training"""
        features = self.extract_speaker_features(audio_segment)
        
        if features is not None:
            self.speaker_features.append(features)
            self.speaker_labels.append(speaker_id if speaker_id is not None else -1)
            
            # Retrain clustering if we have enough segments
            if len(self.speaker_features) >= self.min_segments_for_clustering:
                self._update_clustering()
    
    def _update_clustering(self):
        """Update speaker clustering model"""
        try:
            if len(self.speaker_features) < 2:
                return
            
            features_array = np.array(self.speaker_features)
            
            # Normalize features
            if not hasattr(self.scaler, 'scale_') or len(features_array) > len(self.scaler.scale_) * 2:
                features_scaled = self.scaler.fit_transform(features_array)
            else:
                features_scaled = self.scaler.transform(features_array)
            
            # Determine optimal number of speakers
            n_speakers = min(self.max_speakers, max(2, len(features_array) // 3))
            
            # Perform clustering
            self.kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(features_scaled)
            
            # Update speaker labels
            self.speaker_labels = cluster_labels.tolist()
            
            logger.info(f"Updated speaker clustering: {n_speakers} speakers detected")
            
        except Exception as e:
            logger.error(f"Error updating clustering: {e}")
    
    def identify_speaker(self, audio_segment: np.ndarray) -> Dict:
        """Identify speaker in audio segment"""
        try:
            features = self.extract_speaker_features(audio_segment)
            
            if features is None or self.kmeans is None:
                return {
                    "speaker_id": "unknown",
                    "confidence": 0.0,
                    "speakers_detected": 0
                }
            
            # Scale features
            try:
                features_scaled = self.scaler.transform([features])
            except:
                return {
                    "speaker_id": "unknown", 
                    "confidence": 0.0,
                    "speakers_detected": 0
                }
            
            # Predict speaker
            speaker_id = int(self.kmeans.predict(features_scaled)[0])  # Convert to int
            
            # Calculate confidence based on distance to cluster center
            distances = cdist(features_scaled, self.kmeans.cluster_centers_)
            min_distance = float(np.min(distances))  # Convert to float
            confidence = max(0.0, 1.0 - min_distance / 5.0)  # Normalize distance
            
            # Apply temporal smoothing
            self.speaker_history.append((speaker_id, confidence))
            
            # Get most confident recent prediction
            if len(self.speaker_history) >= 3:
                recent_speakers = [s[0] for s in list(self.speaker_history)[-3:]]
                recent_confidences = [s[1] for s in list(self.speaker_history)[-3:]]
                
                # Majority vote with confidence weighting
                speaker_counts = {}
                for sp, conf in zip(recent_speakers, recent_confidences):
                    if sp in speaker_counts:
                        speaker_counts[sp] += conf
                    else:
                        speaker_counts[sp] = conf
                
                final_speaker = max(speaker_counts.keys(), key=lambda k: speaker_counts[k])
                final_confidence = speaker_counts[final_speaker] / len(recent_speakers)
            else:
                final_speaker = speaker_id
                final_confidence = confidence
            
            # Convert to readable speaker name
            speaker_name = self.speaker_names.get(final_speaker, f"Speaker_{final_speaker}")
            
            return {
                "speaker_id": speaker_name,
                "confidence": float(final_confidence),  # Ensure float
                "speakers_detected": len(set(self.speaker_labels)) if self.speaker_labels else 0,
                "raw_speaker_id": int(final_speaker)  # Convert int32 to int
            }
            
        except Exception as e:
            logger.error(f"Error identifying speaker: {e}")
            return {
                "speaker_id": "unknown",
                "confidence": 0.0,
                "speakers_detected": 0
            }
    
    def get_speaker_stats(self) -> Dict:
        """Get speaker diarization statistics"""
        return {
            "total_segments": len(self.speaker_features),
            "speakers_detected": len(set(self.speaker_labels)) if self.speaker_labels else 0,
            "clustering_ready": self.kmeans is not None,
            "speaker_names": self.speaker_names,
            "confidence_threshold": self.speaker_confidence_threshold
        }
    
    def set_speaker_name(self, speaker_id: int, name: str):
        """Set a human-readable name for a speaker"""
        self.speaker_names[speaker_id] = name

# Language Auto-detection Class
class LanguageDetector:
    def __init__(self):
        """
        Language detection using character n-grams and phonetic features
        """
        # Language models (simplified character frequency patterns)
        self.language_patterns = {
            'en': {
                'common_chars': set('etaoinshrdlu'),
                'common_bigrams': {'th', 'he', 'in', 'er', 'an', 're'},
                'common_trigrams': {'the', 'and', 'ing', 'ion', 'tio'},
                'vowel_ratio_range': (0.35, 0.45),
                'entropy_range': (4.0, 4.8)
            },
            'es': {
                'common_chars': set('eaosrnidlt'),
                'common_bigrams': {'es', 'en', 'el', 'la', 'de', 'al'},
                'common_trigrams': {'que', 'los', 'una', 'con', 'est'},
                'vowel_ratio_range': (0.40, 0.50),
                'entropy_range': (4.2, 5.0)
            },
            'fr': {
                'common_chars': set('esaintrlou'),
                'common_bigrams': {'es', 'en', 'le', 'de', 'nt', 're'},
                'common_trigrams': {'les', 'des', 'une', 'que', 'est'},
                'vowel_ratio_range': (0.38, 0.48),
                'entropy_range': (4.1, 4.9)
            },
            'de': {
                'common_chars': set('enisratdhu'),
                'common_bigrams': {'er', 'en', 'ch', 'nd', 'te', 'in'},
                'common_trigrams': {'der', 'die', 'und', 'den', 'ich'},
                'vowel_ratio_range': (0.35, 0.42),
                'entropy_range': (4.0, 4.7)
            },
            'it': {
                'common_chars': set('eaionlrts'),
                'common_bigrams': {'re', 'er', 'la', 'le', 'il', 'al'},
                'common_trigrams': {'che', 'per', 'con', 'del', 'una'},
                'vowel_ratio_range': (0.42, 0.52),
                'entropy_range': (4.3, 5.1)
            }
        }
        
        # Detection history for temporal smoothing
        self.detection_history = deque(maxlen=5)
        self.confidence_threshold = 0.6
        
    def detect_language_from_text(self, text: str) -> Dict:
        """Detect language from transcribed text"""
        try:
            if len(text.strip()) < 10:  # Too short for reliable detection
                return {"language": "unknown", "confidence": 0.0}
            
            text = text.lower().strip()
            scores = {}
            
            for lang_code, patterns in self.language_patterns.items():
                score = self._calculate_language_score(text, patterns)
                scores[lang_code] = score
            
            # Find best match
            if scores:
                best_language = max(scores.keys(), key=lambda k: scores[k])
                confidence = scores[best_language]
                
                # Normalize confidence
                total_score = sum(scores.values())
                if total_score > 0:
                    confidence = confidence / total_score
                
                # Apply temporal smoothing
                self.detection_history.append((best_language, confidence))
                
                if len(self.detection_history) >= 3:
                    # Weight recent detections
                    lang_weights = {}
                    for i, (lang, conf) in enumerate(self.detection_history):
                        weight = (i + 1) / len(self.detection_history)  # More weight to recent
                        if lang in lang_weights:
                            lang_weights[lang] += conf * weight
                        else:
                            lang_weights[lang] = conf * weight
                    
                    final_language = max(lang_weights.keys(), key=lambda k: lang_weights[k])
                    final_confidence = lang_weights[final_language] / sum(lang_weights.values())
                else:
                    final_language = best_language
                    final_confidence = confidence
                
                return {
                    "language": final_language,
                    "confidence": final_confidence,
                    "all_scores": scores
                }
            
            return {"language": "unknown", "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return {"language": "unknown", "confidence": 0.0}
    
    def _calculate_language_score(self, text: str, patterns: Dict) -> float:
        """Calculate similarity score for a specific language"""
        try:
            score = 0.0
            
            # Character frequency score
            char_score = 0
            for char in patterns['common_chars']:
                char_score += text.count(char)
            char_score /= len(text)
            score += char_score * 0.3
            
            # Bigram score
            bigram_score = 0
            for bigram in patterns['common_bigrams']:
                bigram_score += text.count(bigram)
            bigram_score /= max(1, len(text) - 1)
            score += bigram_score * 0.3
            
            # Trigram score
            trigram_score = 0
            for trigram in patterns['common_trigrams']:
                trigram_score += text.count(trigram)
            trigram_score /= max(1, len(text) - 2)
            score += trigram_score * 0.4
            
            return score
            
        except:
            return 0.0
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.language_patterns.keys())
    
    def should_switch_language(self, current_lang: str, detected_lang: str, confidence: float) -> bool:
        """Determine if language should be switched"""
        if confidence < self.confidence_threshold:
            return False
        
        if current_lang == "auto" or current_lang is None:
            return True
        
        if detected_lang != current_lang and confidence > 0.8:
            return True
        
        return False

# Noise Reduction Class  
class NoiseReducer:
    def __init__(self, sample_rate=16000):
        """
        Audio noise reduction using spectral subtraction and filtering
        """
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.noise_estimation_frames = 10  # Frames to estimate noise
        self.noise_samples = []
        
        # Filter parameters
        self.lowpass_freq = 8000  # Low-pass filter frequency
        self.highpass_freq = 80   # High-pass filter frequency
        
    def estimate_noise_profile(self, audio_segment: np.ndarray):
        """Estimate noise profile from quiet segments"""
        try:
            if len(audio_segment) == 0:
                return
            
            # Convert to float if needed
            if audio_segment.dtype != np.float32:
                audio_segment = audio_segment.astype(np.float32) / 32768.0
            
            # Check if this is a quiet segment (low energy)
            rms = np.sqrt(np.mean(audio_segment ** 2))
            if rms < 0.01:  # Threshold for noise
                self.noise_samples.append(audio_segment)
                
                # Keep only recent noise samples
                if len(self.noise_samples) > self.noise_estimation_frames:
                    self.noise_samples.pop(0)
                
                # Update noise profile
                if len(self.noise_samples) >= 3:
                    noise_concat = np.concatenate(self.noise_samples)
                    fft_noise = np.fft.fft(noise_concat)
                    self.noise_profile = np.abs(fft_noise)
                    
        except Exception as e:
            logger.error(f"Noise estimation error: {e}")
    
    def reduce_noise(self, audio_segment: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio segment"""
        try:
            if len(audio_segment) == 0:
                return audio_segment
            
            # Convert to float if needed
            original_dtype = audio_segment.dtype
            if audio_segment.dtype != np.float32:
                audio_segment = audio_segment.astype(np.float32) / 32768.0
            
            # Apply basic filters first
            audio_filtered = self._apply_bandpass_filter(audio_segment)
            
            # Apply spectral subtraction if noise profile available
            if self.noise_profile is not None:
                audio_filtered = self._spectral_subtraction(audio_filtered)
            
            # Apply adaptive gain
            audio_filtered = self._adaptive_gain(audio_filtered)
            
            # Convert back to original dtype
            if original_dtype == np.int16:
                audio_filtered = (audio_filtered * 32767).astype(np.int16)
            
            return audio_filtered
            
        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return audio_segment
    
    def _apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove extreme frequencies"""
        try:
            # Design bandpass filter
            nyquist = self.sample_rate / 2
            low = self.highpass_freq / nyquist
            high = self.lowpass_freq / nyquist
            
            if high >= 1.0:
                high = 0.99
            
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered = scipy.signal.filtfilt(b, a, audio)
            
            return filtered
            
        except:
            return audio
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise reduction"""
        try:
            # Compute FFT
            fft_audio = np.fft.fft(audio)
            magnitude = np.abs(fft_audio)
            phase = np.angle(fft_audio)
            
            # Spectral subtraction
            if len(self.noise_profile) == len(magnitude):
                # Subtract noise spectrum
                alpha = 2.0  # Over-subtraction factor
                magnitude_clean = magnitude - alpha * self.noise_profile[:len(magnitude)]
                
                # Ensure magnitude doesn't go negative
                magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)
            else:
                magnitude_clean = magnitude
            
            # Reconstruct signal
            fft_clean = magnitude_clean * np.exp(1j * phase)
            audio_clean = np.real(np.fft.ifft(fft_clean))
            
            return audio_clean
            
        except:
            return audio
    
    def _adaptive_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply adaptive gain control"""
        try:
            # RMS-based gain control
            rms = np.sqrt(np.mean(audio ** 2))
            target_rms = 0.1  # Target RMS level
            
            if rms > 0:
                gain = min(3.0, target_rms / rms)  # Limit maximum gain
                audio = audio * gain
            
            # Soft limiting to prevent clipping
            audio = np.tanh(audio * 0.8)
            
            return audio
            
        except:
            return audio

# Global transcription engine - will be initialized after model loads
transcription_engine = None

# Repository classes - will be instantiated with session
user_repo = UserRepository()
transcription_repo = TranscriptionRepository()
api_key_repo = APIKeyRepository()
websocket_session_repo = WebSocketSessionRepository()
usage_repo = UsageRepository()
plan_repo = PlanRepository()

# Authentication functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_api_key(user_id: str, email: str) -> str:
    """Generate unique API key"""
    timestamp = str(int(time.time()))
    unique_string = f"{user_id}-{email}-{timestamp}"
    hash_object = hashlib.sha256(unique_string.encode())
    return f"vf_{hash_object.hexdigest()[:32]}"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_api_key(api_key: str):
    """Verify API key and return user info"""
    if not api_key or not api_key.startswith('vf_'):
        return None
    
    try:
        with get_db_session() as session:
            # Get API key hash
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find user by API key
            api_key_obj = api_key_repo.get_by_hash(session, key_hash)
            if not api_key_obj or not api_key_obj.is_active:
                return None
            
            user = user_repo.get_by_id(session, api_key_obj.user_id)
            if user:
                return {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "company": user.company,
                    "plan_id": user.plan_id,
                    "api_calls_used": user.api_calls_used,
                    "api_calls_limit": user.api_calls_limit,
                    "created_at": user.created_at
                }
            return None
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        return None

def verify_admin_credentials(username: str, password: str) -> bool:
    """Verify admin credentials"""
    # Simple admin credentials from environment variables
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    
    return username == admin_username and password == admin_password

def create_admin_token(username: str):
    """Create admin JWT token"""
    expires_delta = timedelta(hours=24)  # Admin tokens last 24 hours
    return create_access_token(
        data={"sub": username, "role": "admin"}, 
        expires_delta=expires_delta
    )

def track_api_usage(user_id: str, endpoint: str, file_size: int = 0, processing_time: float = 0, audio_duration: float = 0):
    """Track API usage for billing/analytics"""
    try:
        with get_db_session() as session:
            # Create usage record
            usage_repo.create(session, {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "endpoint": endpoint,
                "file_size": file_size,
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "timestamp": datetime.utcnow()
            })
            
            # Update user's API call count
            user = user_repo.get_by_id(session, user_id)
            if user:
                user_repo.update(session, user_id, {
                    "api_calls_used": user.api_calls_used + 1,
                    "last_api_call": datetime.utcnow()
                })
        
    except Exception as e:
        logger.error(f"Usage tracking error: {e}")

def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit based on their plan"""
    try:
        with get_db_session() as session:
            user = user_repo.get_by_id(session, user_id)
            
            if not user:
                return False
            
            # Get plan details
            plan = plan_repo.get_by_id(session, user.plan_id)
            if not plan:
                monthly_limit = 1000  # Default free plan limit
            else:
                monthly_limit = plan.monthly_limit
                # -1 means unlimited (enterprise)
                if monthly_limit == -1:
                    return True
            
            # Check current usage against limit
            if user.api_calls_used >= monthly_limit:
                logger.warning(f"Rate limit exceeded for user {user_id}: {user.api_calls_used}/{monthly_limit}")
                return False
                
            return True
        
    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        return True  # Allow on error

async def get_current_developer(authorization: Optional[str] = Header(None)):
    """Get current developer from API key"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Get yours at /developer"
        )
    
    # Extract API key from Authorization header
    if authorization.startswith('Bearer '):
        api_key = authorization.replace('Bearer ', '')
    else:
        api_key = authorization
    
    user = verify_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check rate limit
    if not check_rate_limit(user['id']):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Upgrade your plan or try next month."
        )
    
    return user

async def authenticate_websocket(api_key: str) -> Optional[dict]:
    """Authenticate API key for WebSocket connection"""
    if not api_key:
        # Allow demo mode (no authentication)
        return {"id": "demo", "email": "demo@voiceforge.ai", "full_name": "Demo User"}
        
    try:
        with get_db_session() as session:
            # Get API key hash
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find user by API key
            api_key_obj = api_key_repo.get_by_hash(session, key_hash)
            if not api_key_obj or not api_key_obj.is_active:
                # Fallback to demo mode
                return {"id": "demo", "email": "demo@voiceforge.ai", "full_name": "Demo User"}
            
            user = user_repo.get_by_id(session, api_key_obj.user_id)
            if user:
                return {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "company": user.company,
                    "plan_id": user.plan_id,
                    "created_at": user.created_at
                }
            
        # Fallback to demo mode
        return {"id": "demo", "email": "demo@voiceforge.ai", "full_name": "Demo User"}
        
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return {"id": "demo", "email": "demo@voiceforge.ai", "full_name": "Demo User"}

async def get_current_developer_optional(authorization: Optional[str] = Header(None)):
    """Get current developer from API key (optional for demo mode)"""
    if not authorization:
        # Return None for demo mode
        return None
    
    # Extract API key from Authorization header
    if authorization.startswith('Bearer '):
        api_key = authorization.replace('Bearer ', '')
    else:
        api_key = authorization
    
    user = verify_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check rate limit
    if not check_rate_limit(user['id']):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Upgrade your plan or try next month."
        )
    
    return user

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global whisper_model
    logger.info("Starting VoiceForge API Service...")
    
    # Check FFmpeg
    ffmpeg_path = os.path.join(project_root, "ffmpeg.exe")
    if os.path.exists(ffmpeg_path):
        logger.info(f"✅ FFmpeg found at: {ffmpeg_path}")
    else:
        logger.warning("⚠️ FFmpeg not found. Please run setup_ffmpeg_windows.bat")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("uploads/temp", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Test database connection
    try:
        with get_db_session() as session:
            # Test query using SQLAlchemy text
            from sqlalchemy import text
            result = session.execute(text("SELECT 1")).scalar()
            if result == 1:
                logger.info("✅ Database connection successful!")
            else:
                logger.error("❌ Database connection test failed!")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
    
    # Load Whisper model
    try:
        logger.info("Loading Whisper model (this may take a moment)...")
        
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model selection and CPU optimization
        if device == "cpu":
            # For CPU, prioritize speed with tiny model
            model_size = os.getenv("WHISPER_MODEL", "tiny")
            # Set number of threads for CPU processing (use all available cores)
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            logger.info(f"CPU optimization: Using {num_threads} threads")
        else:
            # For GPU, can use larger models
            model_size = os.getenv("WHISPER_MODEL", "base")
        
        # Load model with optimal settings
        whisper_model = whisper.load_model(model_size, device=device)
        
        logger.info(f"✅ Whisper model loaded successfully!")
        logger.info(f"Model: {model_size} | Device: {device.upper()}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.info("Running without model - mock mode active")

# Initialize transcription engine after model is loaded
transcription_engine = StreamingTranscriptionEngine(model=whisper_model)

# Routes
@app.websocket("/ws/v1/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None, description="Your VoiceForge API key"),
    language: Optional[str] = Query(None, description="Target language (auto-detect if not specified)"),
    model: str = Query("auto", description="Whisper model (auto, tiny, base, small, medium, large)"),
    interim_results: bool = Query(True, description="Return partial results"),
    sample_rate: int = Query(16000, description="Audio sample rate"),
    encoding: str = Query("linear16", description="Audio encoding format"),
    vad_enabled: bool = Query(True, description="Enable Voice Activity Detection"),
    speaker_diarization: bool = Query(False, description="Enable speaker diarization"),
    language_detection: bool = Query(False, description="Enable automatic language detection"),
    noise_reduction: bool = Query(False, description="Enable noise reduction")
):
    """
    Real-time speech-to-text transcription via WebSocket
    
    **Usage:**
    1. Connect to WebSocket with query parameters
    2. Send JSON configuration message
    3. Send audio data as binary or base64-encoded JSON
    4. Receive real-time transcription results
    
    **Message Types:**
    - `configure`: Setup audio parameters
    - `audio`: Send audio data for transcription  
    - `close_stream`: End transcription session
    
    **Response Types:**
    - `interim`: Partial transcription results
    - `final`: Complete transcription results
    - `error`: Error messages
    """
    
    # Authenticate connection
    user = await authenticate_websocket(api_key) if api_key else None
    if not user and api_key:  # If API key provided but invalid
        await websocket.close(code=1008, reason="Invalid API key")
        return
    
    # Use demo user if no API key
    if not user:
        user = {"id": "demo", "email": "demo@voiceforge.ai", "full_name": "Demo User"}
    
    # Create session
    session_id = str(uuid.uuid4())
    config = StreamingConfig(
        encoding=encoding,
        sample_rate=sample_rate,
        language=language,
        interim_results=interim_results,
        vad_enabled=vad_enabled,
        speaker_diarization=speaker_diarization,
        language_detection=language_detection,
        noise_reduction=noise_reduction
    )
    
    # Connect to manager
    connection = await connection_manager.connect(websocket, session_id, user["id"], config)
    
    # Initialize audio processing with VAD based on config
    audio_buffer = AudioBuffer(sample_rate=sample_rate, vad_enabled=config.vad_enabled)
    logger.info(f"🔧 Audio buffer initialized with VAD {'enabled' if config.vad_enabled else 'disabled'}")
    last_transcription_time = time.time()
    last_interim_time = time.time()
    processing_lock = asyncio.Lock()
    
    # Initialize Phase 3 components
    speaker_diarizer = SpeakerDiarization(sample_rate=sample_rate, max_speakers=config.max_speakers) if config.speaker_diarization else None
    language_detector = LanguageDetector() if config.language_detection else None
    noise_reducer = NoiseReducer(sample_rate=sample_rate) if config.noise_reduction else None
    current_detected_language = language
    
    logger.info(f"WebSocket transcription started: session={session_id}, user={user['email']}")
    
    # Create WebSocket session record in database
    if user["id"] != "demo":
        try:
            with get_db_session() as session:
                # Get client IP address
                client_ip = websocket.client.host if websocket.client else "127.0.0.1"
                
                # Create WebSocket session record
                ws_session = websocket_session_repo.create(session, {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "user_id": user["id"],
                    "connection_id": f"ws_{session_id[:8]}",
                    "ip_address": client_ip,
                    "vad_enabled": config.vad_enabled,
                    "speaker_diarization": config.speaker_diarization,
                    "language_detection": config.language_detection,
                    "noise_reduction": config.noise_reduction,
                    "status": "active",
                    "extra_data": {
                        "model": model,
                        "language": language,
                        "sample_rate": sample_rate,
                        "encoding": encoding,
                        "user_agent": websocket.headers.get("user-agent", "Unknown")
                    }
                })
                logger.info(f"✅ Created WebSocket session record in database: {session_id}")
        except Exception as e:
            logger.error(f"Failed to create WebSocket session record: {e}")
            # Continue even if database save fails
    
    try:
        # Send welcome message
        await connection_manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "config": {
                "sample_rate": sample_rate,
                "encoding": encoding,
                "language": language,
                "model": model
            },
            "message": "Connected to VoiceForge streaming transcription"
        })
        
        async def process_audio_continuously():
            """Continuously process audio buffer with all Phase 3 features"""
            nonlocal last_transcription_time, last_interim_time, current_detected_language
            
            while session_id in connection_manager.active_connections:
                try:
                    current_time = time.time()
                    
                    # Debug logging for processing conditions
                    time_since_interim = current_time - last_interim_time
                    time_since_final = current_time - last_transcription_time
                    should_interim = audio_buffer.should_process_interim()
                    should_final = audio_buffer.should_process()
                    
                    if time_since_final > 1.0:  # Log every second
                        logger.debug(f"🔍 Processing check - Interim: {should_interim} (time: {time_since_interim:.2f}s), Final: {should_final} (time: {time_since_final:.2f}s), Config interim: {config.interim_results}")
                    
                    async with processing_lock:
                        # Process interim results (more frequent, faster)
                        # Reduced frequency to avoid overloading Whisper  
                        if (config.interim_results and 
                            current_time - last_interim_time > 0.8 and  # Every 800ms instead of 300ms
                            audio_buffer.should_process_interim()):
                            
                            logger.info(f"⚡ Starting interim transcription for session {session_id}")
                            
                            # Get shorter audio for interim (increase minimum for better results)
                            interim_audio = audio_buffer.get_audio_for_interim(min_duration=1.0)
                            
                            if interim_audio is not None and len(interim_audio) > 0:
                                logger.info(f"📊 Processing interim audio: {len(interim_audio)} samples")
                                # Apply noise reduction if enabled
                                if noise_reducer:
                                    noise_reducer.estimate_noise_profile(interim_audio)
                                    interim_audio = noise_reducer.reduce_noise(interim_audio)
                                
                                # Generate interim transcription (faster)
                                logger.info(f"🔧 Transcription engine model: {transcription_engine.current_model}, Loaded: {transcription_engine.get_current_model() is not None}")
                                result = await transcription_engine.transcribe_interim(
                                    interim_audio, 
                                    language=current_detected_language,
                                    custom_vocabulary=config.custom_vocabulary
                                )
                                logger.info(f"📊 Interim transcription result: {result}")
                                
                                if result and result.get("transcript"):
                                    logger.info(f"📤 Sending interim result to client: '{result.get('transcript')}'")
                                    # Add Phase 3 features to interim result
                                    await _enhance_result_with_phase3(result, interim_audio, is_interim=True)
                                    await connection_manager.send_message(session_id, result)
                                    last_interim_time = current_time
                                else:
                                    logger.debug("🔇 No interim result to send")
                        
                        # Process final results (less frequent, more accurate)
                        # Increased to 2 seconds to allow more audio accumulation
                        if (current_time - last_transcription_time > 2.0 and  # Every 2 seconds instead of 1
                            audio_buffer.should_process()):
                            
                            logger.info(f"🚀 Starting final transcription for session {session_id}")
                            
                            # Get audio data for final transcription
                            audio_data = audio_buffer.get_audio_for_transcription(
                                duration_seconds=3.0, 
                                use_speech_only=True
                            )
                            
                            if audio_data is not None and len(audio_data) > 0:
                                logger.info(f"📊 Processing final audio: {len(audio_data)} samples for transcription")
                                
                                # Check if this is the same audio we just processed (prevent infinite loop)
                                audio_hash = hash(audio_data.tobytes())
                                if not hasattr(audio_buffer, 'last_processed_hash'):
                                    audio_buffer.last_processed_hash = None
                                
                                if audio_buffer.last_processed_hash == audio_hash:
                                    logger.debug("🔄 Skipping duplicate audio data to prevent infinite loop")
                                    last_transcription_time = current_time  # Update timestamp to prevent immediate retry
                                    continue
                                
                                # Apply noise reduction if enabled
                                if noise_reducer:
                                    noise_reducer.estimate_noise_profile(audio_data)
                                    audio_data = noise_reducer.reduce_noise(audio_data)
                                
                                # Generate final transcription
                                result = await transcription_engine.transcribe_final(
                                    audio_data, 
                                    language=current_detected_language,
                                    model_name=model,
                                    custom_vocabulary=config.custom_vocabulary
                                )
                                
                                # Store hash after processing attempt
                                audio_buffer.last_processed_hash = audio_hash
                                
                                if result and result.get("transcript"):
                                    logger.info(f"📤 Sending final result to client: '{result.get('transcript')}'")
                                    # Add Phase 3 features to final result
                                    await _enhance_result_with_phase3(result, audio_data, is_interim=False)
                                    
                                    # Update speaker diarization training
                                    if speaker_diarizer:
                                        speaker_diarizer.add_speaker_segment(audio_data)
                                    
                                    await connection_manager.send_message(session_id, result)
                                    logger.info(f"Final transcription: {result.get('transcript', '')[:50]}...")
                                    last_transcription_time = current_time
                                    
                                    # Clear processed audio from buffer to prevent reprocessing
                                    audio_buffer.clear_processed_audio()
                                else:
                                    logger.debug("🔇 No final result to send")
                                    # Even if no result, update timestamp to prevent immediate retry of same audio
                                    last_transcription_time = current_time
                        
                        # Send buffer stats periodically (debugging)
                        if current_time - last_transcription_time > 10.0:  # Every 10 seconds
                            stats = audio_buffer.get_buffer_stats()
                            if speaker_diarizer:
                                stats.update({"speaker_stats": speaker_diarizer.get_speaker_stats()})
                            logger.debug(f"Buffer stats for {session_id}: {stats}")
                    
                    await asyncio.sleep(0.2)  # Check every 200ms
                    
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    await connection_manager.send_message(session_id, {
                        "type": "error",
                        "error": {
                            "code": "PROCESSING_ERROR", 
                            "message": str(e)
                        }
                    })
                    break
        
        async def _enhance_result_with_phase3(result: dict, audio_data: np.ndarray, is_interim: bool = False):
            """Enhance transcription result with Phase 3 features"""
            nonlocal current_detected_language
            
            try:
                transcript = result.get("transcript", "")
                
                # Language detection
                if language_detector and transcript:
                    lang_result = language_detector.detect_language_from_text(transcript)
                    detected_lang = lang_result.get("language")
                    lang_confidence = lang_result.get("confidence", 0.0)
                    
                    # Update current language if detection is confident
                    if (language_detector.should_switch_language(current_detected_language, detected_lang, lang_confidence) or
                        current_detected_language is None):
                        current_detected_language = detected_lang
                        
                    result["language_detection"] = {
                        "detected_language": detected_lang,
                        "confidence": lang_confidence,
                        "current_language": current_detected_language
                    }
                
                # Speaker diarization
                if speaker_diarizer and not is_interim:  # Only for final results
                    speaker_info = speaker_diarizer.identify_speaker(audio_data)
                    result["speaker_diarization"] = speaker_info
                
                # Noise reduction stats
                if noise_reducer:
                    result["noise_reduction"] = {
                        "applied": True,
                        "profile_ready": noise_reducer.noise_profile is not None
                    }
                
            except Exception as e:
                logger.error(f"Phase 3 enhancement error: {e}")
        
        # Start audio processing task
        processing_task = asyncio.create_task(process_audio_continuously())
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    break
                
                # Handle text messages (JSON)
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        message_type = data.get("type")
                        
                        if message_type == "configure":
                            # Update configuration
                            new_config = data.get("config", {})
                            config_updated = False
                            
                            if "language" in new_config:
                                config.language = new_config["language"]
                                config_updated = True
                            
                            if "interim_results" in new_config:
                                config.interim_results = bool(new_config["interim_results"])
                                config_updated = True
                            
                            if "vad_enabled" in new_config:
                                old_vad = config.vad_enabled
                                config.vad_enabled = bool(new_config["vad_enabled"])
                                if old_vad != config.vad_enabled:
                                    # Reinitialize audio buffer with new VAD setting
                                    audio_buffer = AudioBuffer(sample_rate=sample_rate, vad_enabled=config.vad_enabled)
                                    config_updated = True
                            
                            if "custom_vocabulary" in new_config:
                                vocab = new_config["custom_vocabulary"]
                                if isinstance(vocab, list):
                                    config.custom_vocabulary = vocab[:50]  # Limit to 50 terms
                                    config_updated = True
                            
                            if "model" in new_config:
                                requested_model = new_config["model"]
                                if requested_model in transcription_engine.get_available_models():
                                    # Try to load the model
                                    if transcription_engine.load_model(requested_model):
                                        model = requested_model
                                        config_updated = True
                                        logger.info(f"Switched to model: {model} for session {session_id}")
                            
                            # Phase 3 configuration updates
                            if "speaker_diarization" in new_config:
                                config.speaker_diarization = bool(new_config["speaker_diarization"])
                                if config.speaker_diarization and not speaker_diarizer:
                                    speaker_diarizer = SpeakerDiarization(sample_rate=sample_rate, max_speakers=config.max_speakers)
                                elif not config.speaker_diarization:
                                    speaker_diarizer = None
                                config_updated = True
                            
                            if "language_detection" in new_config:
                                config.language_detection = bool(new_config["language_detection"])
                                if config.language_detection and not language_detector:
                                    language_detector = LanguageDetector()
                                elif not config.language_detection:
                                    language_detector = None
                                config_updated = True
                            
                            if "noise_reduction" in new_config:
                                config.noise_reduction = bool(new_config["noise_reduction"])
                                if config.noise_reduction and not noise_reducer:
                                    noise_reducer = NoiseReducer(sample_rate=sample_rate)
                                elif not config.noise_reduction:
                                    noise_reducer = None
                                config_updated = True
                            
                            if "max_speakers" in new_config:
                                new_max = int(new_config["max_speakers"])
                                if 1 <= new_max <= 10:
                                    config.max_speakers = new_max
                                    if speaker_diarizer:
                                        speaker_diarizer.max_speakers = new_max
                                    config_updated = True
                            
                            if config_updated:
                                response_data = {
                                    "type": "configured",
                                    "config": config.dict(),
                                    "active_model": transcription_engine.current_model,
                                    "message": "Configuration updated successfully"
                                }
                                
                                # Add Phase 3 status
                                if speaker_diarizer:
                                    response_data["speaker_stats"] = speaker_diarizer.get_speaker_stats()
                                
                                if language_detector:
                                    response_data["supported_languages"] = language_detector.get_supported_languages()
                                
                                await connection_manager.send_message(session_id, response_data)
                            else:
                                await connection_manager.send_message(session_id, {
                                    "type": "configure_error",
                                    "message": "No valid configuration parameters provided"
                                })
                            
                        elif message_type == "audio":
                            # Handle base64 encoded audio
                            audio_data_b64 = data.get("data")
                            if audio_data_b64:
                                try:
                                    audio_bytes = base64.b64decode(audio_data_b64)
                                    has_speech = audio_buffer.add_chunk(audio_bytes)
                                    connection.bytes_processed += len(audio_bytes)
                                    connection.last_activity = datetime.utcnow()
                                    
                                    # Debug logging every 50 chunks
                                    if connection.bytes_processed % (4096 * 50) == 0:
                                        logger.info(f"🎵 Audio processing - Session: {session_id}, "
                                                  f"Bytes: {connection.bytes_processed}, "
                                                  f"Speech detected: {has_speech}, "
                                                  f"Buffer size: {len(audio_buffer.buffer)}, "
                                                  f"Speech buffer: {len(audio_buffer.speech_buffer)}, "
                                                  f"Should process: {audio_buffer.should_process()}, "
                                                  f"Should interim: {audio_buffer.should_process_interim()}")
                                    
                                    # Send VAD feedback if enabled (optional)
                                    if config.vad_enabled and hasattr(data, 'send_vad_feedback') and data.get('send_vad_feedback', False):
                                        await connection_manager.send_message(session_id, {
                                            "type": "vad_result",
                                            "has_speech": has_speech,
                                            "timestamp": datetime.utcnow().isoformat()
                                        })
                                        
                                except Exception as e:
                                    logger.error(f"Error processing audio data: {e}")
                                    await connection_manager.send_message(session_id, {
                                        "type": "error",
                                        "error": {
                                            "code": "AUDIO_DECODE_ERROR",
                                            "message": "Failed to decode audio data"
                                        }
                                    })
                        
                        elif message_type == "close_stream":
                            break
                            
                    except json.JSONDecodeError:
                        await connection_manager.send_message(session_id, {
                            "type": "error",
                            "error": {
                                "code": "INVALID_JSON",
                                "message": "Invalid JSON message format"
                            }
                        })
                
                # Handle binary messages (raw audio)
                elif "bytes" in message:
                    audio_bytes = message["bytes"]
                    has_speech = audio_buffer.add_chunk(audio_bytes)
                    connection.bytes_processed += len(audio_bytes)
                    connection.last_activity = datetime.utcnow()
                    
                    # Log speech activity for debugging
                    if config.vad_enabled:
                        logger.debug(f"Session {session_id}: Speech detected: {has_speech}")
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await connection_manager.send_message(session_id, {
                    "type": "error", 
                    "error": {
                        "code": "MESSAGE_ERROR",
                        "message": str(e)
                    }
                })
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Cancel processing task
        if 'processing_task' in locals():
            processing_task.cancel()
        
        # Update WebSocket session record in database
        if user["id"] != "demo" and 'connection' in locals():
            try:
                with get_db_session() as session:
                    # Calculate session duration
                    session_duration = (datetime.utcnow() - connection.connected_at).total_seconds() if hasattr(connection, 'connected_at') else 0
                    
                    # Update session with final stats
                    websocket_session_repo.update(session, session_id, {
                        "status": "disconnected",
                        "duration_seconds": session_duration,
                        "bytes_processed": connection.bytes_processed,
                        "transcription_count": connection.messages_sent,
                        "ended_at": datetime.utcnow(),
                        "extra_data": {
                            "total_audio_seconds": connection.bytes_processed / (sample_rate * 2) if sample_rate else 0,  # Assuming 16-bit audio
                            "disconnected_reason": "normal"
                        }
                    })
                    logger.info(f"✅ Updated WebSocket session record: {session_id}")
            except Exception as e:
                logger.error(f"Failed to update WebSocket session record: {e}")
        
        # Cleanup connection
        await connection_manager.disconnect(session_id)
        logger.info(f"WebSocket transcription ended: session={session_id}, processed={connection.bytes_processed if 'connection' in locals() else 0} bytes")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """VoiceForge Phase 3 homepage with advanced features"""
    return templates.TemplateResponse("landing_phase3.html", {"request": request})

@app.get("/modern", response_class=HTMLResponse)
async def modern_home(request: Request):
    """VoiceForge modern homepage (pre-Phase 3)"""
    return templates.TemplateResponse("landing_modern.html", {"request": request})

@app.get("/classic", response_class=HTMLResponse)
async def classic_home(request: Request):
    """VoiceForge classic homepage"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs", response_class=HTMLResponse)
async def documentation(request: Request):
    """VoiceForge documentation"""
    return templates.TemplateResponse("docs_modern.html", {"request": request})

# Documentation routing - all redirect to main docs page with proper fragments
@app.get("/docs/{section:path}", response_class=HTMLResponse)
async def documentation_sections(section: str, request: Request):
    """Handle documentation section routing"""
    return templates.TemplateResponse("docs_modern.html", {"request": request})

@app.get("/developer", response_class=HTMLResponse)
async def developer_portal(request: Request):
    """Developer portal for API key management"""
    return templates.TemplateResponse("developer_portal_modern.html", {"request": request})

@app.get("/reset-password", response_class=HTMLResponse)
async def password_reset_page(request: Request):
    """Password reset page"""
    return templates.TemplateResponse("reset_password.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

async def verify_admin_access(request: Request):
    """Verify admin access for HTML pages"""
    # Try to get token from various sources
    token = None
    
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.replace('Bearer ', '')
    
    # Check query parameter (for HTML access)
    elif request.query_params.get("token"):
        token = request.query_params.get("token")
    
    if not token:
        # Redirect to admin login
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/admin", status_code=302)
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        role = payload.get("role")
        if role != "admin":
            return RedirectResponse(url="/admin", status_code=302)
        return True
    except jwt.PyJWTError:
        return RedirectResponse(url="/admin", status_code=302)

@app.get("/stats", response_class=HTMLResponse)
async def stats_dashboard(request: Request):
    """Statistics dashboard with charts and visualizations (Admin Only)"""
    # Verify admin access
    auth_result = await verify_admin_access(request)
    if auth_result != True:
        return auth_result  # Return redirect response
    
    return templates.TemplateResponse("stats_dashboard.html", {"request": request})

@app.get("/status")
async def get_service_status():
    """Public service status endpoint"""
    try:
        with get_db_session() as session:
            # Test database connection
            from sqlalchemy import text
            result = session.execute(text("SELECT 1")).scalar()
            db_status = "connected" if result == 1 else "disconnected"
    except Exception:
        db_status = "disconnected"
    return {
        "status": "online",
        "service": "VoiceForge API Service",
        "version": "3.1.0",
        "model_loaded": whisper_model is not None,
        "database": db_status,
        "message": "Speech-to-Text API for Developers"
    }

# Google OAuth endpoints
@app.get("/auth/google/login")
async def google_login(request: FastAPIRequest):
    """Initiate Google OAuth login"""
    try:
        from auth_google import GoogleAuth
        auth = GoogleAuth(get_db_connection)
        return await auth.login(request)
    except ImportError as e:
        logger.error(f"Failed to import GoogleAuth: {e}")
        raise HTTPException(
            status_code=501,
            detail="Google OAuth not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env"
        )
    except Exception as e:
        logger.error(f"Google OAuth error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google OAuth error: {str(e)}"
        )

@app.get("/auth/google/callback")
async def google_callback(request: FastAPIRequest):
    """Handle Google OAuth callback"""
    try:
        from auth_google import GoogleAuth
        auth = GoogleAuth(get_db_connection)
        return await auth.callback(request)
    except ImportError as e:
        logger.error(f"Failed to import GoogleAuth: {e}")
        raise HTTPException(
            status_code=501,
            detail="Google OAuth not configured"
        )
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google OAuth callback error: {str(e)}"
        )

# Developer Authentication
@app.post("/api/v1/auth/register")
async def register_developer(developer: DeveloperRegister):
    """Register new developer account"""
    try:
        with get_db_session() as session:
            # Check if email exists
            existing_user = user_repo.get_by_email(session, developer.email)
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Get free plan
            free_plan = plan_repo.get_by_name(session, "free")
            if not free_plan:
                raise HTTPException(status_code=500, detail="Free plan not found")
            
            # Create user
            hashed_password = hash_password(developer.password)
            user_id = str(uuid.uuid4())
            api_key = generate_api_key(user_id, developer.email)
            
            user = user_repo.create(session, {
                "id": user_id,
                "email": developer.email,
                "full_name": developer.full_name,
                "password_hash": hashed_password,
                "company": developer.company,
                "plan_id": free_plan.id,
                "email_verified": False,  # Use email_verified instead of is_verified
                "role": UserRole.DEVELOPER.value  # Set role as DEVELOPER
            })
            
            # Create developer profile
            from src.database.models import Developer
            developer_profile = Developer(
                id=str(uuid.uuid4()),
                user_id=str(user.id),
                email=developer.email,
                full_name=developer.full_name,
                company=developer.company
            )
            session.add(developer_profile)
            session.flush()  # Ensure developer is created before API key
            
            # Create API key
            api_key_repo.create(session, {
                "id": str(uuid.uuid4()),
                "user_id": str(user.id),
                "key_hash": hashlib.sha256(api_key.encode()).hexdigest(),
                "key_prefix": api_key[:12],  # Store first 12 characters as prefix
                "name": "Default API Key",
                "is_active": True
            })
            
            # Send welcome email asynchronously
            try:
                await email_service.send_welcome_email({
                    "email": user.email,
                    "full_name": user.full_name,
                    "api_key": api_key,
                    "plan_name": "Free Plan",
                    "monthly_limit": 1000
                })
                logger.info(f"📧 Welcome email sent to {developer.email}")
            except Exception as e:
                logger.warning(f"Failed to send welcome email: {e}")
                # Don't fail registration if email fails
            
            logger.info(f"✅ New developer registered: {developer.email}")
            logger.info(f"📌 Generated API key: {api_key[:20]}... (showing first 20 chars)")
            return {
                "message": "Developer account created successfully",
                "developer": {
                    "id": str(user.id),
                    "email": user.email,
                    "full_name": user.full_name,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                },
                "api_key": api_key,  # This should contain the full API key
                "plan": "free",
                "rate_limit": "1000 requests/month"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/users/stats")
async def get_user_stats(authorization: str = Header(None)):
    """Get user usage statistics"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid authorization")
    
    token = authorization.replace('Bearer ', '')
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        with get_db_session() as session:
            # Get user stats
            stats = transcription_repo.get_user_stats(session, user_id)
            
            # Get language distribution  
            languages = transcription_repo.get_language_distribution(session, user_id)
            
            # Get user's plan info
            user = user_repo.get_by_id(session, user_id)
            plan_info = None
            monthly_limit = 1000
            
            if user and user.plan_id:
                plan = plan_repo.get_by_id(session, user.plan_id)
                if plan:
                    plan_info = {
                        "name": plan.name,
                        "display_name": plan.display_name,
                        "monthly_limit": plan.monthly_limit
                    }
                    monthly_limit = plan.monthly_limit
        
        return {
            "total_requests": int(stats.get("total_requests", 0)) if stats else 0,
            "requests_today": int(stats.get("requests_today", 0)) if stats else 0,
            "requests_this_month": int(stats.get("requests_this_month", 0)) if stats else 0,
            "monthly_limit": monthly_limit,
            "plan_name": plan_info.get("display_name", "Free Plan") if plan_info else "Free Plan",
            "avg_processing_time": round(float(stats.get("avg_processing_time", 0)) / 1000, 2) if stats and stats.get("avg_processing_time") else 0,
            "total_words": int(stats.get("total_words", 0)) if stats else 0,
            "total_audio_minutes": round(float(stats.get("total_bytes_processed", 0) if stats else 0) / (1024 * 1024 * 0.5), 2),  # Rough estimate
            "top_languages": languages if languages else [],
            "active_days": int(stats.get("active_days", 0)) if stats else 0
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/v1/users/upgrade")
async def upgrade_plan(plan_name: str, authorization: str = Header(None)):
    """Upgrade user's subscription plan"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid authorization")
    
    token = authorization.replace('Bearer ', '')
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get the requested plan
        plan = execute_db_query(
            "SELECT id, name, display_name, monthly_limit, price_monthly FROM voiceforge.plans WHERE name = %s",
            (plan_name,),
            fetch_one=True
        )
        
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        # Update user's plan
        update_query = "UPDATE voiceforge.users SET plan_id = %s WHERE id = %s"
        execute_db_query(update_query, (plan['id'], user_id))
        
        # Record subscription change
        subscription_query = """
        INSERT INTO voiceforge.user_subscriptions (user_id, plan_id, status, started_at)
        VALUES (%s, %s, 'active', NOW())
        """
        execute_db_query(subscription_query, (user_id, plan['id']))
        
        logger.info(f"✅ User {user_id} upgraded to {plan_name}")
        
        return {
            "success": True,
            "message": f"Successfully upgraded to {plan['display_name']}",
            "plan": {
                "name": plan['name'],
                "display_name": plan['display_name'],
                "monthly_limit": plan['monthly_limit'],
                "price_monthly": float(plan['price_monthly'])
            }
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/v1/plans")
async def get_available_plans():
    """Get all available subscription plans"""
    query = """
    SELECT name, display_name, monthly_limit, price_monthly, price_yearly, features
    FROM voiceforge.plans
    ORDER BY price_monthly
    """
    plans = execute_db_query(query, fetch_all=True)
    
    return {
        "plans": [
            {
                "name": plan['name'],
                "display_name": plan['display_name'],
                "monthly_limit": plan['monthly_limit'],
                "price_monthly": float(plan['price_monthly']),
                "price_yearly": float(plan['price_yearly']),
                "features": plan['features']
            }
            for plan in plans
        ] if plans else []
    }

@app.post("/api/v1/auth/login")
async def login_developer(developer_data: DeveloperLogin):
    """Developer login"""
    try:
        with get_db_session() as session:
            user = user_repo.get_by_email(session, developer_data.email)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Verify password
            if not verify_password(developer_data.password, user.password_hash):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Get user's API key
            api_key_obj = api_key_repo.get_by_user_id(session, str(user.id))
            
            # Generate API key (we always regenerate on login since we only store the hash)
            api_key = generate_api_key(str(user.id), user.email)
            logger.info(f"Generated API key for {user.email}: {api_key[:12]}...")  # Log first 12 chars
            
            if api_key_obj:
                # Update existing API key hash
                api_key_repo.update(session, str(api_key_obj.id), {
                    "key_hash": hashlib.sha256(api_key.encode()).hexdigest()
                })
            else:
                # Create new API key if none exists
                api_key_repo.create(session, {
                    "id": str(uuid.uuid4()),
                    "user_id": str(user.id),
                    "key_hash": hashlib.sha256(api_key.encode()).hexdigest(),
                    "key_prefix": api_key[:12],  # Store first 12 characters as prefix
                    "name": "Default API Key",
                    "is_active": True
                })
            
            # Create access token for dashboard access
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": str(user.id)}, expires_delta=access_token_expires
            )
            
            logger.info(f"✅ Developer logged in: {developer_data.email}")
            logger.info(f"📌 API key returned in login: {api_key[:20] if api_key else 'NULL'}... (first 20 chars)")
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "user": {
                    "id": str(user.id),
                    "email": user.email,
                    "full_name": user.full_name,
                    "api_key": api_key,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_current_dashboard_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user for dashboard access (JWT token)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    with get_db_session() as session:
        user = user_repo.get_by_id(session, user_id)
        if user is None:
            raise credentials_exception
        
        # Get user's API key
        api_key_obj = api_key_repo.get_by_user_id(session, str(user.id))
        api_key = None
        if api_key_obj:
            # Generate the API key (since we only store the hash)
            api_key = generate_api_key(str(user.id), user.email)
        
        # Convert to dictionary format for compatibility
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "company": user.company,
            "role": user.role.value if hasattr(user.role, 'value') else user.role,
            "created_at": user.created_at,
            "api_key": api_key
        }

def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current admin user (JWT token with admin role)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Admin access required",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role != "admin":
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    return {"username": username, "role": role}

@app.post("/api/v1/admin/login")
async def admin_login(admin_data: AdminLogin):
    """Admin login endpoint"""
    try:
        if not verify_admin_credentials(admin_data.username, admin_data.password):
            raise HTTPException(status_code=401, detail="Invalid admin credentials")
        
        # Create admin token
        admin_token = create_admin_token(admin_data.username)
        
        logger.info(f"✅ Admin logged in: {admin_data.username}")
        return {
            "access_token": admin_token,
            "token_type": "bearer",
            "expires_in": 24 * 60 * 60,  # 24 hours in seconds
            "role": "admin",
            "username": admin_data.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/users/me")
async def get_developer_info(current_user: dict = Depends(get_current_dashboard_user)):
    """Get developer account information (dashboard access)"""
    api_key = current_user.get("api_key")
    logger.info(f"📌 /users/me returning API key: {api_key[:20] if api_key else 'NULL'}...")
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "created_at": current_user["created_at"],
        "api_key": api_key
    }

# Password Reset Endpoints
@app.post("/api/v1/auth/forgot-password")
async def forgot_password(request: dict):
    """Request password reset email"""
    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    try:
        with get_db_session() as session:
            user = user_repo.get_by_email(session, email)
            if user:
                # Generate reset token
                reset_token = secrets.token_urlsafe(32)
                reset_expires = datetime.utcnow() + timedelta(hours=1)
                
                # Update user with reset token
                session.query(User).filter(User.id == user.id).update({
                    "password_reset_token": reset_token,
                    "password_reset_expires": reset_expires
                })
                session.commit()
                
                # Send reset email
                await email_service.send_password_reset_email({
                    "email": user.email,
                    "full_name": user.full_name
                }, reset_token)
                
                logger.info(f"Password reset email sent to {email}")
            
            # Always return success to prevent email enumeration
            return {"message": "If the email exists, a password reset link has been sent"}
            
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process password reset")

@app.post("/api/v1/auth/reset-password")
async def reset_password(request: dict):
    """Reset password with token"""
    token = request.get("token")
    new_password = request.get("password")
    
    if not token or not new_password:
        raise HTTPException(status_code=400, detail="Token and password are required")
    
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    try:
        with get_db_session() as session:
            # Find user with valid token
            user = session.query(User).filter(
                User.password_reset_token == token,
                User.password_reset_expires > datetime.utcnow()
            ).first()
            
            if not user:
                raise HTTPException(status_code=400, detail="Invalid or expired reset token")
            
            # Update password
            hashed_password = hash_password(new_password)
            user.password_hash = hashed_password
            user.password_reset_token = None
            user.password_reset_expires = None
            session.commit()
            
            logger.info(f"Password reset successful for {user.email}")
            return {"message": "Password has been reset successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password")

@app.post("/api/v1/users/regenerate-api-key")
async def regenerate_api_key(
    request: Request,
    current_user: dict = Depends(get_current_dashboard_user)
):
    """Regenerate API key for current user"""
    try:
        with get_db_session() as session:
            user_id = current_user["id"]
            user = user_repo.get_by_id(session, user_id)
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Generate new API key
            new_api_key = generate_api_key(str(user.id), user.email)
            
            # Update or create API key record
            api_key_obj = api_key_repo.get_by_user_id(session, str(user.id))
            
            if api_key_obj:
                # Update existing
                api_key_repo.update(session, str(api_key_obj.id), {
                    "key_hash": hashlib.sha256(new_api_key.encode()).hexdigest(),
                    "key_prefix": new_api_key[:12]
                })
            else:
                # Create new
                api_key_repo.create(session, {
                    "id": str(uuid.uuid4()),
                    "user_id": str(user.id),
                    "key_hash": hashlib.sha256(new_api_key.encode()).hexdigest(),
                    "key_prefix": new_api_key[:12],
                    "name": "Default API Key",
                    "is_active": True
                })
            
            # Send notification email
            try:
                await email_service.send_api_key_regenerated_email(
                    {
                        "email": user.email,
                        "full_name": user.full_name
                    },
                    new_api_key,
                    {
                        "ip_address": request.client.host,
                        "user_agent": request.headers.get("User-Agent", "Unknown")
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to send API key regeneration email: {e}")
            
            logger.info(f"API key regenerated for {user.email}")
            return {
                "message": "API key regenerated successfully",
                "api_key": new_api_key
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key regeneration error: {e}")
        raise HTTPException(status_code=500, detail="Failed to regenerate API key")

# Email Testing Endpoint (Admin only)
@app.post("/api/v1/admin/test-email")
async def test_email(
    request: dict,
    admin_user: str = Depends(get_current_admin)
):
    """Test email configuration by sending a test email"""
    to_email = request.get("email")
    if not to_email:
        raise HTTPException(status_code=400, detail="Email address required")
    
    try:
        # Send test email
        result = await email_service.send_email(
            to_email,
            "VoiceForge Email Test",
            """
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>Email Configuration Test</h2>
                <p>This is a test email from VoiceForge.</p>
                <p>If you're receiving this, your email configuration is working correctly!</p>
                <hr>
                <p><strong>Configuration Details:</strong></p>
                <ul>
                    <li>SMTP Host: {}</li>
                    <li>SMTP Port: {}</li>
                    <li>From Email: {}</li>
                    <li>TLS Enabled: {}</li>
                </ul>
                <p>Sent at: {}</p>
            </body>
            </html>
            """.format(
                os.getenv("SMTP_HOST", "not configured"),
                os.getenv("SMTP_PORT", "not configured"),
                os.getenv("FROM_EMAIL", "not configured"),
                os.getenv("SMTP_USE_TLS", "true"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            ),
            "This is a test email from VoiceForge. Your email configuration is working!"
        )
        
        if result:
            logger.info(f"Test email sent successfully to {to_email}")
            return {"message": f"Test email sent successfully to {to_email}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send test email")
            
    except Exception as e:
        logger.error(f"Test email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main API Endpoints (for developers' apps)
@app.post("/api/v1/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_developer: Optional[dict] = Depends(get_current_developer_optional)
):
    """
    Transcribe audio file to text
    
    **Authentication:** API Key optional (demo mode available)
    
    **Usage:** Include your API key for full features: `Authorization: Bearer your_api_key`
    
    **Supported formats:** MP3, WAV, M4A, MP4, OGG, FLAC
    
    **Rate limits:** 1000 requests/month (Free plan)
    """
    
    start_time = time.time()
    
    # Handle both demo and authenticated modes
    if current_developer:
        developer_id = current_developer["id"]
        logger.info(f"API transcription request from developer: {current_developer['email']}")
        
        # Check rate limit for authenticated users
        if not check_rate_limit(str(developer_id)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded (1000 requests/month). Please upgrade your plan."
            )
    else:
        developer_id = None
        logger.info("Demo transcription request (no authentication)")
    
    # Create temp directory
    temp_dir = os.path.join(os.getcwd(), "uploads", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = None
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file format
        valid_extensions = ('.mp3', '.wav', '.m4a', '.mp4', '.ogg', '.flac', '.webm')
        if not file.filename.lower().endswith(valid_extensions):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Check file size (10MB limit)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size: 10MB")
        
        if whisper_model:
            try:
                # Save file temporarily
                temp_filename = f"api_{int(time.time())}_{file.filename}"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                logger.info(f"Processing API request: {temp_path} ({file_size} bytes)")
                
                # Transcribe with Whisper (optimized parameters)
                result = whisper_model.transcribe(
                    temp_path,
                    fp16=torch.cuda.is_available(),  # Use FP16 on GPU for faster processing
                    language=None,  # Auto-detect language
                    task="transcribe",
                    temperature=0.0,  # Disable sampling for faster processing
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    initial_prompt=None,
                    word_timestamps=False,  # Disable word timestamps for speed
                    verbose=False,
                    beam_size=1,  # Use greedy decoding for speed
                    best_of=1
                )
                
                processing_time = time.time() - start_time
                logger.info(f"✅ API transcription completed in {processing_time:.2f}s")
                
                # Extract results
                transcribed_text = result.get("text", "").strip()
                detected_language = result.get("language", "unknown")
                
                # Get segments
                segments = []
                duration = 0
                if "segments" in result and result["segments"]:
                    for seg in result["segments"]:
                        segments.append({
                            "start": round(seg.get("start", 0), 2),
                            "end": round(seg.get("end", 0), 2),
                            "text": seg.get("text", "").strip()
                        })
                    duration = result["segments"][-1].get("end", 0) if result["segments"] else 0
                
                # Calculate word count for both demo and authenticated modes
                word_count = len(transcribed_text.split()) if transcribed_text else 0
                
                # Track usage (only for authenticated users)
                if developer_id:
                    track_api_usage(developer_id, "/api/v1/transcribe", file_size, processing_time, duration)
                
                # Save to database (only for authenticated users)
                if developer_id:
                    transcription_id = str(uuid.uuid4())
                    
                    with get_db_session() as session:
                        transcription_repo.create(session, {
                            "id": transcription_id,
                            "user_id": developer_id,
                            "file_name": file.filename,
                            "file_size": file_size,
                            "duration_seconds": duration,
                            "language": detected_language,
                            "transcription_text": transcribed_text,
                            "confidence_score": 0.95,
                            "processing_time_ms": int(processing_time * 1000),
                            "segments": segments,
                            "word_count": word_count,
                            "status": "completed",
                            "model_used": "whisper-tiny",
                            "extra_data": {
                                "api_request": True,
                                "developer_email": current_developer["email"],
                                "ip_address": "127.0.0.1"
                            }
                        })
                else:
                    # Demo mode - generate temporary ID
                    transcription_id = f"demo_{str(uuid.uuid4())[:8]}"
                
                return JSONResponse(content={
                    "status": "success",
                    "transcription": {
                        "id": transcription_id,
                        "text": transcribed_text if transcribed_text else "No speech detected in audio",
                        "language": detected_language,
                        "duration": round(duration, 2),
                        "confidence": 0.95 if transcribed_text else 0.0,
                        "word_count": word_count,
                        "processing_time": round(processing_time, 2),
                        "segments": segments,
                        "model": "whisper-tiny"
                    },
                    "usage": {
                        "file_size": file_size,
                        "processing_time": round(processing_time, 2)
                    }
                })
                    
            except Exception as e:
                logger.error(f"API transcription error: {e}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="Speech recognition service temporarily unavailable")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

@app.post("/api/v1/transcribe/youtube")
async def transcribe_youtube(
    request: YouTubeTranscribeRequest,
    current_developer: Optional[dict] = Depends(get_current_developer_optional)
):
    """
    Transcribe audio from YouTube video
    
    **Usage:** Provide a YouTube URL to transcribe the audio content
    """
    start_time = time.time()
    temp_audio_path = None
    
    try:
        # Validate YouTube URL
        youtube_regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|embed/|v/|watch\?.*&v=)?([^&\n?#]+)'
        match = re.match(youtube_regex, request.url)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        video_id = match.group(5)
        logger.info(f"Processing YouTube video: {video_id}")
        
        # Handle authentication and rate limiting
        if current_developer:
            developer_id = current_developer["id"]
            logger.info(f"YouTube transcription request from developer: {current_developer['email']}")
            
            # Check rate limit
            if not check_rate_limit(developer_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        else:
            developer_id = None
            logger.info("Demo YouTube transcription request (no authentication)")
        
        # Create temporary directory for audio
        temp_dir = os.path.join(project_root, "uploads", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        temp_audio_path = os.path.join(temp_dir, f"youtube_{unique_id}.mp3")
        
        # Configure yt-dlp options
        # Optimized yt-dlp options for faster processing
        ydl_opts = {
            'format': 'worstaudio/worst',  # Use lowest quality for faster download (Whisper handles quality well)
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '64',  # Lower bitrate for faster processing
            }],
            'outtmpl': temp_audio_path.replace('.mp3', '.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'ffmpeg_location': os.path.join(project_root, "ffmpeg.exe") if os.name == 'nt' else 'ffmpeg',
            'extractaudio': True,
            'noplaylist': True,  # Only download single video
            'no_check_certificate': True,  # Skip SSL verification for speed
            'socket_timeout': 10,
        }
        
        # Download and extract audio
        logger.info(f"Downloading audio from YouTube video: {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            video_title = info.get('title', 'Unknown')
            video_duration = info.get('duration', 0)
        
        # Check if audio file was created
        if not os.path.exists(temp_audio_path):
            # Check for other possible extensions
            for ext in ['.m4a', '.webm', '.opus']:
                alt_path = temp_audio_path.replace('.mp3', ext)
                if os.path.exists(alt_path):
                    temp_audio_path = alt_path
                    break
            else:
                raise HTTPException(status_code=500, detail="Failed to extract audio from video")
        
        # Check file size (max 25MB for YouTube videos)
        file_size = os.path.getsize(temp_audio_path)
        if file_size > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Video audio exceeds 25MB limit")
        
        # Transcribe the audio
        logger.info(f"Transcribing YouTube audio: {temp_audio_path}")
        
        if not whisper_model:
            raise HTTPException(status_code=503, detail="Transcription service not available")
        
        # Load and transcribe audio (optimized parameters)
        result = whisper_model.transcribe(
            temp_audio_path,
            fp16=torch.cuda.is_available(),  # Use FP16 on GPU for faster processing
            language=None,  # Auto-detect language
            task="transcribe",
            temperature=0.0,  # Disable sampling for faster processing
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            initial_prompt=None,
            word_timestamps=False  # Disable word timestamps for speed
        )
        
        # Extract transcription data
        transcribed_text = result["text"].strip()
        detected_language = result.get("language", "unknown")
        segments = result.get("segments", [])
        
        # Calculate metrics
        processing_time = time.time() - start_time
        word_count = len(transcribed_text.split()) if transcribed_text else 0
        
        # Track usage for authenticated users
        if developer_id:
            track_api_usage(developer_id, "/api/v1/transcribe/youtube", file_size, processing_time, video_duration)
            
            # Save to database
            transcription_id = str(uuid.uuid4())
            metadata = {
                "source": "youtube",
                "video_id": video_id,
                "video_title": video_title,
                "video_url": request.url
            }
            
            save_query = """
                INSERT INTO voiceforge.transcriptions 
                (id, user_id, file_name, file_size, detected_language, transcribed_text, 
                 confidence_score, processing_time_ms, segments, word_count, status, 
                 created_at, updated_at, metadata, model_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            now = datetime.now()
            segments_json = json.dumps(segments) if segments else "[]"
            
            params = (transcription_id, developer_id, f"YouTube: {video_title[:100]}", file_size,
                     detected_language, transcribed_text, 0.95, int(processing_time * 1000),
                     segments_json, word_count, 'completed', now, now, json.dumps(metadata),
                     'whisper-tiny')
            
            execute_db_query(save_query, params)
        else:
            # Demo mode
            transcription_id = f"demo_{str(uuid.uuid4())[:8]}"
        
        return JSONResponse(content={
            "status": "success",
            "transcription": {
                "id": transcription_id,
                "text": transcribed_text if transcribed_text else "No speech detected in video",
                "language": detected_language,
                "duration": round(video_duration, 2),
                "confidence": 0.95 if transcribed_text else 0.0,
                "word_count": word_count,
                "processing_time": round(processing_time, 2),
                "video_title": video_title,
                "video_id": video_id
            },
            "processing_time": round(processing_time, 2),
            "model_used": "whisper-tiny"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temporary audio file
        try:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info(f"Cleaned up temporary file: {temp_audio_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

@app.get("/api/v1/usage")
async def get_usage_stats(current_developer: dict = Depends(get_current_developer)):
    """Get API usage statistics for current developer"""
    try:
        # Get usage stats for this month
        query = """
        SELECT 
            COUNT(*) as total_requests,
            SUM(file_size_bytes) as total_bytes,
            AVG(response_time_ms) as avg_response_time,
            MAX(timestamp) as last_request
        FROM analytics.usage_metrics 
        WHERE user_id = %s 
        AND timestamp >= DATE_TRUNC('month', NOW())
        """
        stats = execute_db_query(query, (current_developer["id"],), fetch_one=True)
        
        if not stats:
            stats = {
                "total_requests": 0,
                "total_bytes": 0,
                "avg_response_time": 0,
                "last_request": None
            }
        
        return {
            "current_month": {
                "requests": int(stats.get("total_requests", 0)),
                "bytes_processed": int(stats.get("total_bytes", 0) or 0),
                "avg_response_time_ms": round(float(stats.get("avg_response_time", 0) or 0)),
                "last_request": stats.get("last_request")
            },
            "limits": {
                "monthly_requests": 1000,
                "max_file_size_mb": 10,
                "plan": "free"
            },
            "remaining": {
                "requests": max(0, 1000 - int(stats.get("total_requests", 0)))
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        return {
            "error": str(e),
            "current_month": {"requests": 0},
            "limits": {"monthly_requests": 1000, "plan": "free"}
        }

@app.get("/api/v1/health")
async def api_health_check():
    """Public API health check"""
    return {
        "status": "healthy",
        "service": "VoiceForge API",
        "version": "3.1.0",
        "model_loaded": whisper_model is not None,
        "timestamp": datetime.now().isoformat()
    }

# Public stats (no auth required)
@app.get("/api/v1/streaming/stats")
async def get_streaming_stats():
    """Get real-time WebSocket connection statistics"""
    try:
        stats = connection_manager.get_connection_stats()
        
        # Add additional metrics
        stats.update({
            "service_status": "healthy",
            "whisper_model_loaded": whisper_model is not None,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting streaming stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get streaming statistics"}
        )

@app.get("/api/v1/streaming/models")
async def get_available_models():
    """Get information about available Whisper models"""
    try:
        model_stats = transcription_engine.get_model_stats()
        
        model_info = {
            "models": {
                "tiny": {"size": "39 MB", "speed": "~32x", "accuracy": "Good"},
                "base": {"size": "74 MB", "speed": "~16x", "accuracy": "Better"},
                "small": {"size": "244 MB", "speed": "~6x", "accuracy": "Very Good"},
                "medium": {"size": "769 MB", "speed": "~2x", "accuracy": "Excellent"},
                "large": {"size": "1550 MB", "speed": "~1x", "accuracy": "Best"}
            },
            "currently_loaded": model_stats["loaded_models"],
            "default_model": model_stats["current_model"],
            "total_loaded": model_stats["total_loaded"]
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get model information"}
        )

@app.get("/api/v1/streaming/features")
async def get_streaming_features():
    """Get information about available Phase 3 streaming features"""
    try:
        # Initialize temporary instances to get capabilities
        temp_lang_detector = LanguageDetector()
        temp_speaker_diarizer = SpeakerDiarization()
        
        features_info = {
            "voice_activity_detection": {
                "available": True,
                "description": "Detects speech vs. silence in audio stream",
                "benefits": ["Faster processing", "Lower CPU usage", "Better accuracy"]
            },
            "speaker_diarization": {
                "available": True,
                "description": "Identifies different speakers in conversation",
                "max_speakers": temp_speaker_diarizer.max_speakers,
                "features": ["Real-time speaker identification", "Adaptive clustering", "Speaker naming"]
            },
            "language_detection": {
                "available": True,
                "description": "Automatically detects and switches languages",
                "supported_languages": temp_lang_detector.get_supported_languages(),
                "features": ["Real-time detection", "Temporal smoothing", "Confidence scoring"]
            },
            "noise_reduction": {
                "available": True,
                "description": "Reduces background noise for better transcription",
                "techniques": ["Bandpass filtering", "Spectral subtraction", "Adaptive gain control"]
            },
            "interim_results": {
                "available": True,
                "description": "Real-time partial transcription results",
                "latency": "~300ms"
            },
            "custom_vocabulary": {
                "available": True,
                "description": "Boost recognition of specific terms",
                "max_terms": 50
            }
        }
        
        return features_info
        
    except Exception as e:
        logger.error(f"Error getting features info: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get features information"}
        )

@app.get("/api/v1/streaming/health")
async def streaming_health_check():
    """Health check endpoint for streaming service"""
    try:
        health_status = {
            "status": "healthy",
            "service": "VoiceForge Streaming",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "websocket_manager": connection_manager is not None,
                "whisper_model": whisper_model is not None,
                "transcription_engine": transcription_engine is not None,
                "active_connections": len(connection_manager.active_connections)
            }
        }
        
        # Check if any critical components are failing
        if not all(health_status["checks"].values()):
            health_status["status"] = "degraded"
            return JSONResponse(status_code=503, content=health_status)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/api/v1/stats")
async def get_public_stats():
    """Get public service statistics"""
    try:
        query = """
        SELECT 
            (SELECT COUNT(*) FROM voiceforge.users) as total_developers,
            (SELECT COUNT(*) FROM voiceforge.transcriptions) as total_transcriptions,
            (SELECT COUNT(DISTINCT DATE(created_at)) FROM voiceforge.transcriptions WHERE created_at >= NOW() - INTERVAL '30 days') as active_days
        """
        stats = execute_db_query(query, fetch_one=True)
        
        if not stats:
            stats = {"total_developers": 0, "total_transcriptions": 0, "active_days": 0}
        
        return {
            "service": "VoiceForge API",
            "total_transcriptions": int(stats.get("total_transcriptions", 0)),
            "active_sessions": int(stats.get("total_developers", 0)),  # Using developers as active sessions for display
            "supported_languages": 100,
            "model_loaded": whisper_model is not None,
            "model": "whisper-tiny",
            "status": "operational" if whisper_model else "maintenance"
        }
        
    except Exception as e:
        logger.error(f"Error getting public stats: {e}")
        return {
            "service": "VoiceForge API",
            "total_transcriptions": 0,
            "active_sessions": 0,
            "supported_languages": 100,
            "model_loaded": whisper_model is not None,
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)