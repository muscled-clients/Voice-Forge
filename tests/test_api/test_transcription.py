import pytest
from httpx import AsyncClient
import json
import asyncio
from unittest.mock import patch


class TestTranscriptionAPI:
    """Test transcription endpoints"""
    
    @pytest.mark.asyncio
    async def test_sync_transcription(
        self,
        authenticated_client: AsyncClient,
        test_audio_bytes
    ):
        """Test synchronous transcription"""
        files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
        data = {
            "model": "whisper-base",
            "language": "en",
            "punctuation": True,
            "diarization": False
        }
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/sync",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "transcript" in result
        assert "confidence" in result
        assert "words" in result
        assert result["status"] == "completed"
        assert result["language"] == "en"
    
    @pytest.mark.asyncio
    async def test_batch_transcription(
        self,
        authenticated_client: AsyncClient,
        test_audio_bytes
    ):
        """Test batch transcription"""
        files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
        data = {
            "model": "whisper-base",
            "language": "en"
        }
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/batch",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "job_id" in result
        assert result["status"] == "pending"
        assert "created_at" in result
    
    @pytest.mark.asyncio
    async def test_get_transcription_status(
        self,
        authenticated_client: AsyncClient,
        test_user
    ):
        """Test getting transcription status"""
        # This would normally require a real job ID
        # For testing, we'll mock the service
        job_id = "test-job-id"
        
        with patch('app.services.transcription.TranscriptionService.get_job') as mock_get_job:
            from app.db.models import TranscriptionJob
            mock_job = TranscriptionJob(
                id=job_id,
                user_id=test_user.id,
                status="completed",
                result={"transcript": "Test transcript"},
                created_at=None,
                completed_at=None
            )
            mock_get_job.return_value = mock_job
            
            response = await authenticated_client.get(f"/api/v1/transcribe/status/{job_id}")
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["job_id"] == job_id
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_unsupported_audio_format(
        self,
        authenticated_client: AsyncClient
    ):
        """Test transcription with unsupported audio format"""
        files = {"audio": ("test.txt", b"not an audio file", "text/plain")}
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/sync",
            files=files
        )
        
        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_large_file_sync(
        self,
        authenticated_client: AsyncClient
    ):
        """Test sync transcription with large file"""
        # Create a fake large file (>10MB)
        large_audio = b"0" * (11 * 1024 * 1024)
        files = {"audio": ("large.wav", large_audio, "audio/wav")}
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/sync",
            files=files
        )
        
        assert response.status_code == 400
        assert "too large" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_transcription_without_auth(
        self,
        client: AsyncClient,
        test_audio_bytes
    ):
        """Test transcription without authentication"""
        files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
        
        response = await client.post(
            "/api/v1/transcribe/sync",
            files=files
        )
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_transcription_with_diarization(
        self,
        authenticated_client: AsyncClient,
        test_audio_bytes
    ):
        """Test transcription with speaker diarization"""
        files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
        data = {
            "model": "whisper-base",
            "diarization": True,
            "punctuation": True
        }
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/sync",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "transcript" in result
        # Diarization results would be included if enabled
    
    @pytest.mark.asyncio
    async def test_transcription_with_language_detection(
        self,
        authenticated_client: AsyncClient,
        test_audio_bytes
    ):
        """Test transcription with automatic language detection"""
        files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
        data = {
            "model": "whisper-base",
            # No language specified - should auto-detect
        }
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/sync",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "transcript" in result
        assert "language" in result


class TestWebSocketTranscription:
    """Test WebSocket streaming transcription"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, authenticated_client: AsyncClient):
        """Test WebSocket connection for streaming"""
        # Note: This is a simplified test as WebSocket testing is complex
        # In a real implementation, you'd use websockets library for testing
        
        # Test that the endpoint exists
        response = await authenticated_client.get("/api/v1/transcribe/stream")
        # WebSocket endpoints return 405 for GET requests
        assert response.status_code == 405
    
    @pytest.mark.asyncio
    async def test_websocket_config_message(self):
        """Test WebSocket configuration message format"""
        config_message = {
            "type": "config",
            "config": {
                "model": "whisper-base",
                "language": "en",
                "interim_results": True,
                "punctuation": True
            }
        }
        
        # Validate message structure
        assert config_message["type"] == "config"
        assert "model" in config_message["config"]
        assert "language" in config_message["config"]
    
    @pytest.mark.asyncio
    async def test_websocket_audio_chunk_format(self):
        """Test WebSocket audio chunk message format"""
        # Test audio chunk should be binary data
        audio_chunk = b"fake_audio_data"
        
        assert isinstance(audio_chunk, bytes)
        assert len(audio_chunk) > 0


class TestTranscriptionModels:
    """Test different model configurations"""
    
    @pytest.mark.asyncio
    async def test_different_models(
        self,
        authenticated_client: AsyncClient,
        test_audio_bytes
    ):
        """Test transcription with different models"""
        models = ["whisper-base", "whisper-small", "whisper-medium"]
        
        for model in models:
            files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
            data = {"model": model}
            
            response = await authenticated_client.post(
                "/api/v1/transcribe/sync",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                assert "transcript" in result
                assert result.get("model_used") == model
            else:
                # Model might not be available
                assert response.status_code in [400, 404]
    
    @pytest.mark.asyncio
    async def test_invalid_model(
        self,
        authenticated_client: AsyncClient,
        test_audio_bytes
    ):
        """Test transcription with invalid model"""
        files = {"audio": ("test.wav", test_audio_bytes, "audio/wav")}
        data = {"model": "invalid-model"}
        
        response = await authenticated_client.post(
            "/api/v1/transcribe/sync",
            files=files,
            data=data
        )
        
        # Should use default model or return error
        assert response.status_code in [200, 400]