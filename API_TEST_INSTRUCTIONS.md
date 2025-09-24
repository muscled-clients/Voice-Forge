# VoiceForge API Testing Instructions

## How to Test Your API Key

### 1. Get Your API Key
1. Go to http://localhost:8000/developer
2. Sign up or login with your account
3. Copy your API key from the dashboard (starts with `vf_`)

### 2. Test with cURL
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -F "file=@audio-sample-1.mp3"
```

### 3. Test with Python
```python
import requests

API_KEY = "vf_YOUR_API_KEY_HERE"  # Replace with your actual API key

with open("audio-sample-1.mp3", "rb") as f:
    files = {"file": f}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(
        "http://localhost:8000/api/v1/transcribe",
        files=files,
        headers=headers
    )
    
    print(response.json())
```

### 4. Test with JavaScript
```javascript
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:8000/api/v1/transcribe', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer vf_YOUR_API_KEY_HERE'
    },
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

## What's Working

✅ **API Key Generation**: Each registered developer gets a unique API key
✅ **Authentication**: API keys are verified for each request
✅ **Rate Limiting**: 1000 requests/month for free tier (tracked in database)
✅ **Usage Tracking**: All API calls are logged to the database
✅ **Real Stats**: Developer portal shows actual usage statistics from database
✅ **Demo Mode**: Homepage demo works without authentication

## Database Storage

When using an API key, the following is stored in the database:
- Transcription records in `voiceforge.transcriptions` table
- Usage metrics in `analytics.usage_metrics` table
- User information in `voiceforge.users` table

Demo mode transcriptions (from homepage) are NOT saved to the database.