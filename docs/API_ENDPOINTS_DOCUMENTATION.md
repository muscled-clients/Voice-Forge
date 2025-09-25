# VoiceForge API Endpoints Documentation

## 🎯 Complete API Endpoints List

### 🔐 Authentication & User Management

#### 1. **Register New Developer**
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe",
  "company": "Acme Corp"
}
```

#### 2. **Login**
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### 3. **Get Current User**
```http
GET /api/v1/users/me
Authorization: Bearer YOUR_ACCESS_TOKEN
```

#### 4. **Get User Statistics**
```http
GET /api/v1/users/stats
Authorization: Bearer YOUR_ACCESS_TOKEN
```

#### 5. **Forgot Password**
```http
POST /api/v1/auth/forgot-password
Content-Type: application/json

{
  "email": "user@example.com"
}
```

#### 6. **Reset Password**
```http
POST /api/v1/auth/reset-password
Content-Type: application/json

{
  "token": "reset_token_from_email",
  "password": "new_password123"
}
```

#### 7. **Regenerate API Key**
```http
POST /api/v1/users/regenerate-api-key
Authorization: Bearer YOUR_ACCESS_TOKEN
```

### 🎤 Transcription Endpoints

#### 1. **Transcribe Audio File**
```http
POST /api/v1/transcribe
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

file: (binary audio file)
language: (optional) e.g., "en"
model: (optional) e.g., "whisper-base"
```

**Response:**
```json
{
  "status": "success",
  "transcription": {
    "id": "transcription_id",
    "text": "Transcribed text here",
    "language": "en",
    "duration": 10.5,
    "confidence": 0.95,
    "word_count": 50,
    "segments": [...],
    "model": "whisper-base"
  }
}
```

#### 2. **Transcribe YouTube Video**
```http
POST /api/v1/transcribe/youtube
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "language": "auto"
}
```

#### 3. **Real-time WebSocket Transcription**
```javascript
// WebSocket endpoint
ws://localhost:8000/ws/v1/transcribe?api_key=YOUR_API_KEY

// Send audio chunks as binary data
// Receive transcription results in real-time
```

### 📊 Usage & Analytics

#### 1. **Get API Usage**
```http
GET /api/v1/usage
Authorization: Bearer YOUR_API_KEY
```

#### 2. **Get Detailed Stats**
```http
GET /api/v1/stats
Authorization: Bearer YOUR_API_KEY
```

### 💰 Billing & Plans

#### 1. **Get Available Plans**
```http
GET /api/v1/plans
```

#### 2. **Upgrade Plan**
```http
POST /api/v1/users/upgrade
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "plan_id": "pro_plan_id",
  "payment_method_id": "stripe_payment_method"
}
```

### 🏥 Health & Status

#### 1. **Health Check**
```http
GET /api/v1/health
```

#### 2. **Streaming Service Health**
```http
GET /api/v1/streaming/health
```

#### 3. **Streaming Stats**
```http
GET /api/v1/streaming/stats
```

#### 4. **Available Models**
```http
GET /api/v1/streaming/models
```

#### 5. **Features**
```http
GET /api/v1/streaming/features
```

### 👨‍💼 Admin Endpoints

#### 1. **Admin Login**
```http
POST /api/v1/admin/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin_password"
}
```

#### 2. **Test Email (Admin Only)**
```http
POST /api/v1/admin/test-email
Authorization: Bearer ADMIN_TOKEN
Content-Type: application/json

{
  "email": "test@example.com"
}
```

## 🐍 Python SDK Support Status

### ✅ Currently Supported:
- Basic transcription (`/api/v1/transcribe`)
- User info (`/api/v1/users/me`)
- User stats (`/api/v1/users/stats`)

### ⏳ Needs Implementation:
- YouTube transcription
- WebSocket real-time streaming
- Usage analytics
- Plan management
- Password reset flow
- Admin functions

## 🚀 SDK Usage Examples

### Basic Transcription
```python
from voiceforge import VoiceForgeClient

client = VoiceForgeClient(api_key="YOUR_API_KEY")

# Transcribe file
result = await client.transcribe_file("audio.mp3")
print(result.text)
```

### YouTube Transcription (To Be Implemented)
```python
# Future implementation
result = await client.transcribe_youtube("https://youtube.com/watch?v=...")
print(result.text)
```

### Real-time Streaming (To Be Implemented)
```python
# Future implementation
async with client.stream_transcription() as stream:
    # Send audio chunks
    await stream.send_audio(audio_chunk)
    
    # Receive transcriptions
    async for result in stream:
        print(f"Partial: {result.text}")
```

### WebSocket Direct Usage (Current)
```python
import asyncio
import websockets

async def stream_audio():
    uri = "ws://localhost:8000/ws/v1/transcribe?api_key=YOUR_API_KEY"
    
    async with websockets.connect(uri) as websocket:
        # Send audio data
        with open("audio.wav", "rb") as f:
            audio_data = f.read()
            await websocket.send(audio_data)
        
        # Receive transcription
        result = await websocket.recv()
        print(f"Transcription: {result}")
```

## 📝 Missing Documentation

The following features exist in the API but are not fully documented:

1. **Batch Processing** - Multiple file transcription
2. **Language Detection** - Auto-detect audio language
3. **Speaker Diarization** - Identify different speakers
4. **Custom Vocabulary** - Add domain-specific terms
5. **Export Formats** - SRT, VTT, JSON outputs
6. **Webhook Callbacks** - Async job notifications

## 🔧 Developer Portal Updates Needed

1. Add YouTube transcription example
2. Add WebSocket streaming documentation
3. Add usage/analytics endpoint docs
4. Add billing/upgrade documentation
5. Add rate limiting information
6. Add error code reference

## 📦 SDK Improvements Needed

1. **Implement missing endpoints:**
   - YouTube transcription
   - WebSocket streaming wrapper
   - Usage analytics
   - Plan management

2. **Add convenience methods:**
   - Batch transcription
   - Progress callbacks
   - Auto-retry on failure
   - File format validation

3. **CLI improvements:**
   - Add YouTube download command
   - Add streaming mode
   - Add batch processing
   - Add output format options

## 🎯 Action Items

### High Priority:
- [ ] Update SDK to support all endpoints
- [ ] Create comprehensive API documentation page
- [ ] Add interactive API explorer
- [ ] Publish SDK to PyPI

### Medium Priority:
- [ ] Add code examples for all endpoints
- [ ] Create video tutorials
- [ ] Add Postman/Insomnia collections
- [ ] Create OpenAPI/Swagger spec

### Low Priority:
- [ ] Add SDK for other languages (JavaScript, Go, Java)
- [ ] Create webhook testing tools
- [ ] Add API playground
- [ ] Create migration guides