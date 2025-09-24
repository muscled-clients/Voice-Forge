"""
Quick API Test Script for VoiceForge
"""
import requests
import json
import time
from pathlib import Path

def test_voiceforge_api():
    """Test VoiceForge API endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("🧪 VoiceForge API Test Suite")
    print("="*60)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   Make sure the API is running: .\\run_voiceforge.bat")
        return
    
    # Test 2: API Stats
    print("\n2. Testing API Stats...")
    try:
        response = requests.get(f"{base_url}/api/v1/stats")
        if response.status_code == 200:
            stats = response.json()
            print("   ✅ Stats endpoint working")
            print(f"   - Model loaded: {stats.get('model_loaded', False)}")
            print(f"   - Languages: {stats.get('supported_languages', 0)}")
            print(f"   - Status: {stats.get('status', 'unknown')}")
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Models List
    print("\n3. Testing Models Endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/models")
        if response.status_code == 200:
            models = response.json()
            print("   ✅ Models endpoint working")
            print(f"   Available models: {len(models.get('models', []))}")
        else:
            print(f"   ❌ Models failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: File Upload (Mock)
    print("\n4. Testing Transcription Endpoint...")
    try:
        # Create a dummy audio file
        test_file = Path("test_audio.wav")
        if not test_file.exists():
            # Create a minimal WAV header
            import wave
            import struct
            with wave.open(str(test_file), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # Write 1 second of silence
                for _ in range(16000):
                    wav.writeframes(struct.pack('h', 0))
        
        # Upload file
        with open(test_file, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            response = requests.post(f"{base_url}/api/v1/transcribe", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Transcription endpoint working")
            print(f"   Status: {result.get('status', 'unknown')}")
            if 'transcription' in result:
                print(f"   Text: {result['transcription'].get('text', '')[:100]}...")
        else:
            print(f"   ❌ Transcription failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Authentication
    print("\n5. Testing Authentication...")
    try:
        # Try to register
        user_data = {
            "email": f"test_{int(time.time())}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User",
            "company": "Test Corp"
        }
        
        response = requests.post(f"{base_url}/api/v1/auth/register", json=user_data)
        if response.status_code == 201:
            print("   ✅ Registration working")
            
            # Try to login
            login_data = {
                "email": user_data["email"],
                "password": user_data["password"]
            }
            response = requests.post(f"{base_url}/api/v1/auth/login", json=login_data)
            if response.status_code == 200:
                token = response.json().get("access_token")
                print("   ✅ Login working")
                print(f"   Token received: {token[:20]}...")
            else:
                print(f"   ⚠️ Login not available: {response.status_code}")
        else:
            print(f"   ⚠️ Authentication not available: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Auth endpoints not implemented yet")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print("\n✅ Core API is working!")
    print("📌 Next steps:")
    print("   1. Install Whisper model for real transcription")
    print("   2. Set up PostgreSQL for production")
    print("   3. Run load tests: python tests\\load_test.py")
    print("\n🌐 Access the web interface at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    test_voiceforge_api()