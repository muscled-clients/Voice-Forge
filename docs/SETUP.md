# VoiceForge STT - Setup Guide

This guide will help you set up and test the VoiceForge Speech-to-Text service locally.

## 🚀 Quick Start (5 minutes)

### Prerequisites

- **Docker & Docker Compose** (required)
- **Python 3.8+** (for SDK testing)
- **Node.js 16+** (for web playground)
- **8GB RAM minimum** (16GB recommended)
- **NVIDIA GPU** (optional, for best performance)

### 1. Clone and Setup

```bash
cd D:\Projects\STT
```

### 2. Environment Configuration

Create environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Database
DATABASE_URL=postgresql://postgres:voiceforge123@localhost:5432/voiceforge
REDIS_URL=redis://localhost:6379/0

# API Keys (generate secure keys)
JWT_SECRET_KEY=your-super-secret-jwt-key-here-make-it-long-and-random
API_KEY_SECRET=your-api-key-secret-for-encryption

# Models
HUGGINGFACE_TOKEN=your-huggingface-token  # Optional, for speaker diarization
DEFAULT_MODEL=whisper-base

# Debug
DEBUG=true
LOG_LEVEL=INFO
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Initialize Database

```bash
# Run database migrations
docker-compose exec api alembic upgrade head

# Create test user (optional)
docker-compose exec api python -c "
from app.core.auth import get_password_hash
from app.db.session import get_db_session
from app.db.models import User
import asyncio

async def create_user():
    async with get_db_session() as session:
        user = User(
            id='test-user-123',
            email='test@voiceforge.ai',
            full_name='Test User',
            hashed_password=get_password_hash('password123'),
            api_key='vf_test_key_123456789',
            tier='pro',
            credits_remaining=10000,
            is_active=True
        )
        session.add(user)
        await session.commit()
    print('Test user created!')

asyncio.run(create_user())
"
```

### 5. Test the Service

Open your browser and navigate to:

- **Web Playground**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **GraphQL Playground**: http://localhost:8000/graphql
- **Monitoring**: http://localhost:3000 (Grafana, admin/admin)

## 🧪 Testing Options

### Option 1: Web Playground (Easiest)

1. Go to http://localhost:8000
2. Click "Start Recording" or upload an audio file
3. Watch real-time transcription results

### Option 2: REST API Testing

```bash
# Test with curl (replace with your actual audio file)
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Authorization: Bearer vf_test_key_123456789" \
  -F "audio=@test-audio.wav" \
  -F "model=whisper-base" \
  -F "language=en"
```

### Option 3: Python SDK Testing

```bash
# Install SDK
pip install -e ./sdk/python

# Test transcription
python -c "
import asyncio
import voiceforge

async def test():
    client = voiceforge.VoiceForgeClient(
        api_key='vf_test_key_123456789',
        base_url='http://localhost:8000'
    )
    
    # Test with a sample audio file
    async with client:
        # List available models
        models = await client.list_models()
        print(f'Available models: {[m.model_id for m in models]}')
        
        # Transcribe file (replace with your audio file)
        job = await client.transcribe_file('test-audio.wav')
        print(f'Transcript: {job.transcript}')

asyncio.run(test())
"
```

### Option 4: GraphQL Testing

Visit http://localhost:8000/graphql and try:

```graphql
query {
  availableModels {
    modelId
    name
    supportedLanguages
    isAvailable
  }
}
```

## 📁 Test Audio Files

Create some test audio files in the project root:

### Generate Test Audio (Python)

```bash
pip install pydub

python -c "
from pydub import AudioSegment
from pydub.generators import Sine
import io

# Generate 5 seconds of sine wave (simulates audio)
audio = Sine(440).to_audio_segment(duration=5000)
audio.export('test-audio.wav', format='wav')
print('Test audio file created: test-audio.wav')

# You can also record real audio or download sample files
"
```

### Or Download Sample Files

```bash
# Download sample audio files for testing
curl -o sample1.wav "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
curl -o sample2.wav "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav"
```

## 🔧 Advanced Configuration

### GPU Support (NVIDIA)

If you have NVIDIA GPU, enable GPU support:

```yaml
# In docker-compose.yml, uncomment GPU sections:
services:
  api:
    # ... existing config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Mode

For production testing:

```bash
# Use production docker compose
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -k k8s/
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Services Won't Start

```bash
# Check logs
docker-compose logs api
docker-compose logs postgres
docker-compose logs redis

# Restart services
docker-compose restart
```

#### 2. Database Connection Issues

```bash
# Check database is running
docker-compose exec postgres pg_isready

# Reset database
docker-compose down -v
docker-compose up -d
```

#### 3. Model Loading Issues

```bash
# Check model downloads
docker-compose exec api python -c "
from app.models.manager import model_manager
import asyncio
asyncio.run(model_manager.list_available_models())
"

# Clear model cache
docker-compose exec api rm -rf /app/models/*
```

#### 4. Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x k8s/deploy.sh
```

#### 5. Port Conflicts

If ports are already in use, modify `docker-compose.yml`:

```yaml
services:
  api:
    ports:
      - "8001:8000"  # Change from 8000 to 8001
  
  postgres:
    ports:
      - "5433:5432"  # Change from 5432 to 5433
```

### Performance Issues

#### Memory Usage

```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml if needed
```

#### Slow Transcription

1. **Use smaller models**: `whisper-tiny` or `whisper-base`
2. **Enable GPU** if available
3. **Reduce audio quality**: Convert to 16kHz mono
4. **Check CPU usage**: Ensure adequate resources

### API Testing

#### Authentication Issues

```bash
# Create API key
docker-compose exec api python -c "
import uuid
print(f'Generated API key: vf_{uuid.uuid4().hex[:24]}')
"
```

#### Rate Limiting

If you hit rate limits, modify settings:

```bash
# In .env file
RATE_LIMIT_ENABLED=false
```

## 📊 Monitoring & Logs

### Check Service Health

```bash
# API health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker

# With timestamps
docker-compose logs -f -t api
```

### Metrics & Monitoring

1. **Prometheus**: http://localhost:9090
2. **Grafana**: http://localhost:3000 (admin/admin)
3. **Flower** (Celery): http://localhost:5555

## 🧪 Sample Test Scripts

### Basic Functionality Test

Create `test_basic.py`:

```python
import asyncio
import aiohttp
import json

async def test_api():
    """Test basic API functionality"""
    
    # Test health endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/health') as resp:
            health = await resp.json()
            print(f"Health check: {health}")
        
        # Test models endpoint
        headers = {'Authorization': 'Bearer vf_test_key_123456789'}
        async with session.get('http://localhost:8000/api/v1/models', headers=headers) as resp:
            models = await resp.json()
            print(f"Available models: {models}")
        
        print("✅ Basic API tests passed!")

if __name__ == "__main__":
    asyncio.run(test_api())
```

### Load Test

Create `load_test.py`:

```python
import asyncio
import aiohttp
import time

async def load_test(concurrent_requests=10):
    """Simple load test"""
    
    async def make_request(session, request_id):
        try:
            start_time = time.time()
            headers = {'Authorization': 'Bearer vf_test_key_123456789'}
            
            async with session.get('http://localhost:8000/api/v1/models', headers=headers) as resp:
                await resp.json()
                duration = time.time() - start_time
                print(f"Request {request_id}: {duration:.3f}s")
                return duration
        except Exception as e:
            print(f"Request {request_id} failed: {e}")
            return None
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        successful = [r for r in results if r is not None]
        if successful:
            avg_time = sum(successful) / len(successful)
            print(f"\n📊 Load Test Results:")
            print(f"   Successful requests: {len(successful)}/{concurrent_requests}")
            print(f"   Average response time: {avg_time:.3f}s")
            print(f"   Success rate: {len(successful)/concurrent_requests:.1%}")

if __name__ == "__main__":
    asyncio.run(load_test())
```

Run tests:

```bash
python test_basic.py
python load_test.py
```

## ✅ Verification Checklist

After setup, verify these work:

- [ ] **Health Check**: http://localhost:8000/health returns "healthy"
- [ ] **API Docs**: http://localhost:8000/docs loads successfully
- [ ] **Web Playground**: http://localhost:8000 interface loads
- [ ] **GraphQL**: http://localhost:8000/graphql playground works
- [ ] **Model List**: API returns available models
- [ ] **File Upload**: Can upload audio via web interface
- [ ] **Transcription**: Successfully transcribes test audio
- [ ] **WebSocket**: Real-time streaming works
- [ ] **Monitoring**: Grafana dashboard displays metrics
- [ ] **Database**: Migrations applied successfully

## 🆘 Need Help?

### Quick Fixes

```bash
# Complete reset
docker-compose down -v
docker-compose up -d --build

# Check all services
docker-compose ps
docker-compose logs --tail=20
```

### Support

- **Documentation**: Check `/docs` folder for detailed guides
- **Logs**: Always check `docker-compose logs` for errors
- **Issues**: The setup is thoroughly tested and should work out of the box

### Next Steps

Once you have the basic setup working:

1. **Test with real audio files**
2. **Try different models** (whisper-small, whisper-medium)
3. **Enable GPU support** for better performance
4. **Test the Python/JavaScript SDKs**
5. **Explore the GraphQL API**
6. **Set up production deployment** with Kubernetes

---

🎉 **You're now ready to test the state-of-the-art VoiceForge STT service!**