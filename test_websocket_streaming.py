#!/usr/bin/env python3
"""
VoiceForge WebSocket Streaming Test Client

This script tests the WebSocket streaming transcription endpoint.
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import wave
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingTestClient:
    def __init__(self, api_key=None, language="en", sample_rate=16000):
        self.api_key = api_key
        self.language = language
        self.sample_rate = sample_rate
        self.websocket = None
        
    async def connect(self):
        """Connect to VoiceForge streaming WebSocket"""
        # Build WebSocket URL
        url = f"ws://localhost:8000/ws/v1/transcribe?sample_rate={self.sample_rate}&language={self.language}"
        if self.api_key:
            url += f"&api_key={self.api_key}"
            
        logger.info(f"Connecting to: {url}")
        
        try:
            self.websocket = await websockets.connect(url)
            logger.info("✅ Connected to VoiceForge streaming service")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def send_configuration(self, custom_config=None):
        """Send initial configuration"""
        default_config = {
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "language": self.language,
            "interim_results": True,
            "vad_enabled": True,
            "custom_vocabulary": ["VoiceForge", "API", "transcription", "WebSocket"]
        }
        
        if custom_config:
            default_config.update(custom_config)
        
        config_message = {
            "type": "configure",
            "config": default_config
        }
        
        await self.websocket.send(json.dumps(config_message))
        logger.info(f"📋 Configuration sent: {default_config}")
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk as base64 encoded JSON"""
        audio_b64 = base64.b64encode(audio_data).decode()
        
        message = {
            "type": "audio",
            "data": audio_b64
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def send_binary_audio(self, audio_data: bytes):
        """Send raw audio data as binary message"""
        await self.websocket.send(audio_data)
    
    async def listen_for_results(self):
        """Listen for transcription results"""
        try:
            async for message in self.websocket:
                try:
                    result = json.loads(message)
                    message_type = result.get("type")
                    
                    if message_type == "connected":
                        logger.info(f"🔗 {result.get('message', 'Connected')}")
                        logger.info(f"📊 Session ID: {result.get('session_id')}")
                        
                    elif message_type in ["interim", "final"]:
                        transcript = result.get("transcript", "")
                        confidence = result.get("confidence", 0.0)
                        is_final = result.get("is_final", False)
                        
                        status = "FINAL" if is_final else "INTERIM"
                        logger.info(f"🎤 {status}: {transcript} (confidence: {confidence:.2f})")
                        
                    elif message_type == "error":
                        error = result.get("error", {})
                        logger.error(f"❌ Error: {error.get('code', 'UNKNOWN')} - {error.get('message', 'Unknown error')}")
                        
                    elif message_type == "configured":
                        logger.info("⚙️ Configuration updated")
                        
                    else:
                        logger.info(f"📨 Received: {result}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"📨 Non-JSON message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed")
        except Exception as e:
            logger.error(f"❌ Listen error: {e}")
    
    async def generate_test_audio(self, duration_seconds=10, frequency=440):
        """Generate test audio (sine wave)"""
        logger.info(f"🎵 Generating {duration_seconds}s test audio at {frequency}Hz")
        
        # Generate sine wave
        t = np.linspace(0, duration_seconds, int(self.sample_rate * duration_seconds), False)
        wave_data = np.sin(frequency * 2 * np.pi * t)
        
        # Convert to 16-bit PCM
        audio_data = (wave_data * 32767).astype(np.int16)
        
        return audio_data.tobytes()
    
    async def send_test_audio(self, duration_seconds=10, chunk_duration=0.1, frequency=440):
        """Send test audio in chunks"""
        logger.info(f"📡 Sending test audio: {duration_seconds}s in {chunk_duration}s chunks")
        
        # Generate test audio
        audio_bytes = await self.generate_test_audio(duration_seconds, frequency)
        
        # Calculate chunk size
        chunk_size = int(self.sample_rate * chunk_duration * 2)  # 2 bytes per sample (16-bit)
        
        # Send in chunks
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await self.send_audio_chunk(chunk)
            
            # Wait between chunks to simulate real-time
            await asyncio.sleep(chunk_duration)
            
        logger.info("📡 Test audio sent completely")
    
    async def close(self):
        """Close the connection"""
        if self.websocket:
            # Send close message
            close_message = {"type": "close_stream"}
            await self.websocket.send(json.dumps(close_message))
            
            # Close WebSocket
            await self.websocket.close()
            logger.info("🔌 Connection closed")

async def test_basic_connection():
    """Test basic WebSocket connection"""
    logger.info("🧪 Testing basic WebSocket connection...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        # Start listening for results
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Send configuration
        await client.send_configuration()
        
        # Wait a bit for connection to stabilize
        await asyncio.sleep(1)
        
        # Send some test audio
        await client.send_test_audio(duration_seconds=5, chunk_duration=0.1)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Close connection
        await client.close()
        
        # Cancel listening task
        listen_task.cancel()
    
    logger.info("✅ Basic connection test completed")

async def test_with_api_key():
    """Test connection with API key"""
    logger.info("🧪 Testing connection with API key...")
    
    # Use demo API key (will fall back to demo mode)
    client = StreamingTestClient(api_key="demo-key-123", language="en")
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        await client.send_configuration()
        await asyncio.sleep(1)
        await client.send_test_audio(duration_seconds=3, chunk_duration=0.1)
        await asyncio.sleep(2)
        await client.close()
        
        listen_task.cancel()
    
    logger.info("✅ API key test completed")

async def test_error_handling():
    """Test error handling"""
    logger.info("🧪 Testing error handling...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Send invalid message
        try:
            await client.websocket.send("invalid json message")
            await asyncio.sleep(1)
        except Exception as e:
            logger.info(f"Expected error caught: {e}")
        
        # Send invalid audio data
        try:
            invalid_message = {
                "type": "audio",
                "data": "invalid-base64-data!!!"
            }
            await client.websocket.send(json.dumps(invalid_message))
            await asyncio.sleep(1)
        except Exception as e:
            logger.info(f"Expected error caught: {e}")
        
        await asyncio.sleep(2)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Error handling test completed")

async def test_vad_features():
    """Test Voice Activity Detection features"""
    logger.info("🧪 Testing VAD features...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Test with VAD enabled
        await client.send_configuration({"vad_enabled": True})
        await asyncio.sleep(1)
        
        # Send mixed audio (silence + speech simulation)
        logger.info("🔇 Sending silence...")
        silence_audio = np.zeros(int(client.sample_rate * 1.0), dtype=np.int16)  # 1 second silence
        await client.send_audio_chunk(silence_audio.tobytes())
        await asyncio.sleep(0.5)
        
        logger.info("🎤 Sending simulated speech...")
        await client.send_test_audio(duration_seconds=3, chunk_duration=0.1)
        
        await asyncio.sleep(3)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ VAD features test completed")

async def test_interim_results():
    """Test interim results functionality"""
    logger.info("🧪 Testing interim results...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Enable interim results
        await client.send_configuration({
            "interim_results": True,
            "vad_enabled": True
        })
        await asyncio.sleep(1)
        
        # Send longer audio to see interim updates
        await client.send_test_audio(duration_seconds=5, chunk_duration=0.1)
        
        await asyncio.sleep(5)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Interim results test completed")

async def test_custom_vocabulary():
    """Test custom vocabulary support"""
    logger.info("🧪 Testing custom vocabulary...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Configure with custom vocabulary
        await client.send_configuration({
            "custom_vocabulary": ["VoiceForge", "API", "WebSocket", "transcription", "streaming"],
            "vad_enabled": True
        })
        await asyncio.sleep(1)
        
        await client.send_test_audio(duration_seconds=3, chunk_duration=0.1)
        
        await asyncio.sleep(3)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Custom vocabulary test completed")

async def test_model_switching():
    """Test model switching functionality"""
    logger.info("🧪 Testing model switching...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Test switching to base model
        await client.send_configuration({"model": "base"})
        await asyncio.sleep(2)
        
        await client.send_test_audio(duration_seconds=2, chunk_duration=0.1)
        await asyncio.sleep(2)
        
        # Switch back to tiny model
        await client.send_configuration({"model": "tiny"})
        await asyncio.sleep(1)
        
        await client.send_test_audio(duration_seconds=2, chunk_duration=0.1)
        await asyncio.sleep(2)
        
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Model switching test completed")

async def test_api_endpoints():
    """Test new API endpoints"""
    logger.info("🧪 Testing API endpoints...")
    
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            # Test streaming health
            response = await client.get("http://localhost:8000/api/v1/streaming/health")
            logger.info(f"🏥 Health check: {response.status_code} - {response.json().get('status', 'unknown')}")
            
            # Test streaming stats
            response = await client.get("http://localhost:8000/api/v1/streaming/stats")
            logger.info(f"📊 Streaming stats: {response.status_code}")
            
            # Test models info
            response = await client.get("http://localhost:8000/api/v1/streaming/models")
            if response.status_code == 200:
                models = response.json()
                logger.info(f"🤖 Available models: {list(models.get('models', {}).keys())}")
                logger.info(f"🔄 Loaded models: {models.get('currently_loaded', [])}")
            
            # Test Phase 3 features info
            response = await client.get("http://localhost:8000/api/v1/streaming/features")
            if response.status_code == 200:
                features = response.json()
                logger.info(f"🎛️ Available features: {list(features.keys())}")
                if "language_detection" in features:
                    supported_langs = features["language_detection"].get("supported_languages", [])
                    logger.info(f"🌍 Supported languages: {supported_langs}")
            
    except Exception as e:
        logger.error(f"❌ API endpoint test error: {e}")
    
    logger.info("✅ API endpoints test completed")

async def test_speaker_diarization():
    """Test speaker diarization features"""
    logger.info("🧪 Testing speaker diarization...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Enable speaker diarization
        await client.send_configuration({
            "speaker_diarization": True,
            "max_speakers": 3,
            "vad_enabled": True,
            "interim_results": False  # Only final results for speaker info
        })
        await asyncio.sleep(1)
        
        # Simulate different speakers by sending audio with pauses
        logger.info("🎤 Simulating Speaker 1...")
        await client.send_test_audio(duration_seconds=2, chunk_duration=0.1, frequency=220)
        await asyncio.sleep(1)
        
        logger.info("🎤 Simulating Speaker 2...")
        await client.send_test_audio(duration_seconds=2, chunk_duration=0.1, frequency=440)
        await asyncio.sleep(1)
        
        logger.info("🎤 Simulating Speaker 3...")
        await client.send_test_audio(duration_seconds=2, chunk_duration=0.1, frequency=880)
        
        await asyncio.sleep(5)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Speaker diarization test completed")

async def test_language_detection():
    """Test language auto-detection"""
    logger.info("🧪 Testing language detection...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Enable language detection
        await client.send_configuration({
            "language_detection": True,
            "language": "auto",  # Auto-detect mode
            "vad_enabled": True
        })
        await asyncio.sleep(1)
        
        # Send audio that could trigger language detection
        await client.send_test_audio(duration_seconds=4, chunk_duration=0.1)
        
        await asyncio.sleep(4)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Language detection test completed")

async def test_noise_reduction():
    """Test noise reduction features"""
    logger.info("🧪 Testing noise reduction...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Enable noise reduction
        await client.send_configuration({
            "noise_reduction": True,
            "vad_enabled": True
        })
        await asyncio.sleep(1)
        
        # Send quiet audio first (for noise profiling)
        logger.info("🔇 Sending quiet audio for noise profiling...")
        quiet_audio = np.random.normal(0, 0.005, int(client.sample_rate * 1.0))  # Low-level noise
        quiet_audio = (quiet_audio * 32767).astype(np.int16)
        await client.send_audio_chunk(quiet_audio.tobytes())
        await asyncio.sleep(1)
        
        # Send louder audio (simulated speech with noise)
        logger.info("🎤 Sending speech with noise...")
        await client.send_test_audio(duration_seconds=3, chunk_duration=0.1)
        
        await asyncio.sleep(3)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ Noise reduction test completed")

async def test_all_phase3_features():
    """Test all Phase 3 features together"""
    logger.info("🧪 Testing all Phase 3 features combined...")
    
    client = StreamingTestClient()
    
    if await client.connect():
        listen_task = asyncio.create_task(client.listen_for_results())
        
        # Enable all Phase 3 features
        await client.send_configuration({
            "speaker_diarization": True,
            "language_detection": True,
            "noise_reduction": True,
            "max_speakers": 3,
            "vad_enabled": True,
            "interim_results": True,
            "custom_vocabulary": ["VoiceForge", "streaming", "transcription"]
        })
        await asyncio.sleep(1)
        
        # Send comprehensive test audio
        await client.send_test_audio(duration_seconds=6, chunk_duration=0.1)
        
        await asyncio.sleep(6)
        await client.close()
        listen_task.cancel()
    
    logger.info("✅ All Phase 3 features test completed")

async def main():
    """Run all tests"""
    logger.info("🚀 Starting VoiceForge WebSocket Streaming Tests (Phase 3)")
    logger.info("=" * 70)
    
    try:
        # Phase 1 tests
        logger.info("🔥 Phase 1 Tests")
        await test_basic_connection()
        await asyncio.sleep(2)
        
        # Phase 2 tests
        logger.info("🔥 Phase 2 Tests")
        await test_vad_features()
        await asyncio.sleep(2)
        
        await test_interim_results()
        await asyncio.sleep(2)
        
        await test_custom_vocabulary()
        await asyncio.sleep(2)
        
        await test_model_switching()
        await asyncio.sleep(2)
        
        # Phase 3 tests
        logger.info("🔥 Phase 3 Tests")
        await test_speaker_diarization()
        await asyncio.sleep(2)
        
        await test_language_detection()
        await asyncio.sleep(2)
        
        await test_noise_reduction()
        await asyncio.sleep(2)
        
        await test_all_phase3_features()
        await asyncio.sleep(2)
        
        # API and error handling
        logger.info("🔥 System Tests")
        await test_api_endpoints()
        await asyncio.sleep(1)
        
        await test_error_handling()
        
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
    
    logger.info("=" * 70)
    logger.info("🏁 All Phase 3 tests completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted")
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")