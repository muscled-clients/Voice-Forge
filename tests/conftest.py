import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
import tempfile
from pathlib import Path
import uuid

from app.main import app
from app.db.session import Base, get_session
from app.core.config import settings
from app.core.auth import AuthService
from app.db.models import User


# Test database configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
    echo=False
)


@pytest_asyncio.fixture
async def db_session():
    """Create a test database session"""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(test_engine) as session:
        yield session


@pytest.fixture
def override_get_session(db_session):
    """Override the database session dependency"""
    def _override_get_session():
        return db_session
    
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(override_get_session):
    """Create a test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sync_client():
    """Create a synchronous test client"""
    return TestClient(app)


@pytest_asyncio.fixture
async def test_user(db_session):
    """Create a test user"""
    user = await AuthService.create_user(
        db=db_session,
        email="test@example.com",
        password="testpassword123",
        full_name="Test User"
    )
    return user


@pytest_asyncio.fixture
async def authenticated_client(client, test_user):
    """Create an authenticated test client"""
    # Create access token
    token = AuthService.create_access_token(
        data={"sub": test_user.id, "email": test_user.email}
    )
    
    # Set authorization header
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture
def test_audio_file():
    """Create a test audio file"""
    # Create a temporary WAV file
    temp_dir = Path(tempfile.mkdtemp())
    audio_file = temp_dir / "test_audio.wav"
    
    # Generate simple test audio (silence)
    import wave
    import numpy as np
    
    duration = 2.0  # seconds
    sample_rate = 16000
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    with wave.open(str(audio_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    yield audio_file
    
    # Cleanup
    audio_file.unlink(missing_ok=True)
    temp_dir.rmdir()


@pytest.fixture
def test_audio_bytes():
    """Generate test audio as bytes"""
    import wave
    import numpy as np
    import io
    
    duration = 1.0  # seconds
    sample_rate = 16000
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    return buffer.getvalue()


@pytest.fixture
def mock_model_manager():
    """Mock model manager for testing"""
    from unittest.mock import Mock, AsyncMock
    
    manager = Mock()
    manager.is_ready = Mock(return_value=True)
    manager.get_available_models = Mock(return_value=[
        {
            "name": "whisper-base",
            "model_id": "openai/whisper-base",
            "type": "whisper",
            "languages": ["en"],
            "loaded": True
        }
    ])
    manager.transcribe = AsyncMock(return_value={
        "text": "Hello, this is a test transcription.",
        "language": "en",
        "confidence": 0.95,
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.98},
            {"word": "this", "start": 0.6, "end": 0.8, "confidence": 0.95},
            {"word": "is", "start": 0.9, "end": 1.0, "confidence": 0.97},
            {"word": "a", "start": 1.1, "end": 1.2, "confidence": 0.93},
            {"word": "test", "start": 1.3, "end": 1.6, "confidence": 0.96},
            {"word": "transcription", "start": 1.7, "end": 2.5, "confidence": 0.94}
        ]
    })
    
    return manager


@pytest.fixture(autouse=True)
def override_model_manager(mock_model_manager):
    """Override model manager for all tests"""
    app.state.model_manager = mock_model_manager
    yield
    # Cleanup is handled by test teardown


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_job_data():
    """Test data for transcription jobs"""
    return {
        "job_id": str(uuid.uuid4()),
        "model": "whisper-base",
        "language": "en",
        "options": {
            "punctuation": True,
            "diarization": False
        }
    }


@pytest.fixture
def test_user_data():
    """Test data for user creation"""
    return {
        "email": f"test-{uuid.uuid4()}@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


class TestSettings:
    """Test-specific settings"""
    database_url = TEST_DATABASE_URL
    redis_host = "localhost"
    redis_port = 6379
    redis_db = 15  # Use different DB for tests
    secret_key = "test-secret-key"
    jwt_expiration_hours = 1
    rate_limit_enabled = False  # Disable for tests
    log_level = "ERROR"  # Reduce noise in tests


@pytest.fixture(autouse=True)
def use_test_settings():
    """Use test settings for all tests"""
    original_settings = {}
    
    # Store original values
    for key, value in TestSettings.__dict__.items():
        if not key.startswith('_'):
            original_settings[key] = getattr(settings, key, None)
            setattr(settings, key, value)
    
    yield
    
    # Restore original values
    for key, value in original_settings.items():
        if value is not None:
            setattr(settings, key, value)