#!/usr/bin/env python
"""
Test WebSocket Session Creation
Verify that WebSocket sessions are properly saved to the database
"""

import asyncio
import websockets
import json
import base64
import os
from dotenv import load_dotenv

load_dotenv()

async def test_websocket_session():
    """Test WebSocket connection and verify session creation"""
    
    # Configuration
    ws_url = "ws://localhost:8000/ws/v1/transcribe"
    
    # Connect without API key for demo mode
    params = {
        "language": "en",
        "model": "tiny",
        "interim_results": "true",
        "vad_enabled": "true",
        "speaker_diarization": "false",
        "language_detection": "false",
        "noise_reduction": "false"
    }
    
    # Build URL with parameters
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{ws_url}?{query_string}"
    
    print(f"🔗 Connecting to: {full_url}")
    
    try:
        async with websockets.connect(full_url) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            welcome_data = json.loads(welcome)
            print(f"📨 Welcome message received:")
            print(f"   Session ID: {welcome_data.get('session_id')}")
            print(f"   Type: {welcome_data.get('type')}")
            
            session_id = welcome_data.get('session_id')
            
            # Send configuration
            config_message = {
                "type": "configure",
                "config": {
                    "sample_rate": 16000,
                    "encoding": "linear16",
                    "language": "en"
                }
            }
            await websocket.send(json.dumps(config_message))
            print("📤 Configuration sent")
            
            # Send a small test audio chunk
            test_audio = bytes([0] * 1600)  # 100ms of silence at 16kHz
            audio_message = {
                "type": "audio",
                "data": base64.b64encode(test_audio).decode('utf-8')
            }
            await websocket.send(json.dumps(audio_message))
            print("🎵 Test audio sent")
            
            # Wait a moment for processing
            await asyncio.sleep(1)
            
            # Close the stream properly
            close_message = {"type": "close_stream"}
            await websocket.send(json.dumps(close_message))
            print("🛑 Close stream message sent")
            
            # Wait for any final messages
            await asyncio.sleep(0.5)
            
            print(f"\n✨ Test completed!")
            print(f"📊 Session ID: {session_id}")
            print(f"💾 Check database for WebSocket session record")
            print(f"   Table: voiceforge.websocket_sessions")
            print(f"   Query: SELECT * FROM voiceforge.websocket_sessions WHERE session_id = '{session_id}';")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing WebSocket Session Creation")
    print("=" * 50)
    result = asyncio.run(test_websocket_session())
    
    if result:
        print("\n✅ WebSocket session test passed!")
        print("Check your PostgreSQL database to verify the session was saved.")
    else:
        print("\n❌ WebSocket session test failed!")
        print("Make sure the API server is running on http://localhost:8000")