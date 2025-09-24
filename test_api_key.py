"""Test API key authentication"""
import requests
import os

# Replace with your actual API key from the developer portal
API_KEY = "vf_YOUR_API_KEY_HERE"

# Test file path
test_file = "audio-sample-1.mp3"

if not os.path.exists(test_file):
    print("❌ Test file not found. Please ensure audio-sample-1.mp3 exists")
    exit(1)

# Test transcription with API key
print("Testing API with authentication...")
with open(test_file, 'rb') as f:
    files = {'file': f}
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    response = requests.post(
        'http://localhost:8000/api/v1/transcribe',
        files=files,
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API key authentication works!")
        print(f"Transcription: {result['transcription']['text'][:100]}...")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())