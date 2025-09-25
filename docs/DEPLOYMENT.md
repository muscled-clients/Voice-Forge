# VoiceForge Production Deployment Guide

## 📋 Table of Contents
- [Infrastructure Requirements](#infrastructure-requirements)
- [Deployment Options](#deployment-options)
- [Technical Stack](#technical-stack)
- [Cost Analysis](#cost-analysis)
- [Deployment Steps](#deployment-steps)
- [Performance Optimization](#performance-optimization)
- [Monitoring & Scaling](#monitoring--scaling)

## 🔧 Infrastructure Requirements

### Minimum Production Requirements
- **GPU**: NVIDIA GPU with CUDA 11.7+ support
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 100GB SSD minimum
- **CPU**: 4+ vCPUs
- **Network**: 1Gbps connection
- **OS**: Ubuntu 22.04 LTS

### GPU Recommendations
| GPU Model | Performance | Cost/Hour | Use Case |
|-----------|------------|-----------|----------|
| NVIDIA T4 | Good | ~$0.526 | Best value for production |
| NVIDIA A10G | Better | ~$1.05 | High-volume processing |
| NVIDIA V100 | Excellent | ~$3.06 | Enterprise scale |
| NVIDIA A100 | Best | ~$5.12 | Maximum performance |

## 🚀 Deployment Options

### Option A: AWS (Recommended) ⭐
```yaml
Infrastructure:
  Compute: EC2 g4dn.xlarge
  Database: RDS PostgreSQL (db.t3.medium)
  Storage: S3 for audio files
  CDN: CloudFront
  Region: us-east-1 (lowest cost)

Specifications:
  - NVIDIA T4 GPU (16GB VRAM)
  - 4 vCPUs
  - 16GB RAM
  - Up to 10 Gbps network
  - Auto-scaling support

Monthly Cost: ~$400-600
```

### Option B: Google Cloud Platform
```yaml
Infrastructure:
  Compute: n1-standard-4 + NVIDIA T4
  Database: Cloud SQL PostgreSQL
  Storage: Cloud Storage
  CDN: Cloud CDN
  Region: us-central1

Specifications:
  - NVIDIA T4 GPU
  - 4 vCPUs
  - 15GB RAM
  - Integration with Google OAuth

Monthly Cost: ~$350-550
```

### Option C: Azure
```yaml
Infrastructure:
  Compute: NC4as T4 v3
  Database: Azure Database for PostgreSQL
  Storage: Azure Blob Storage
  CDN: Azure CDN
  Region: East US

Specifications:
  - NVIDIA T4 GPU
  - 4 vCPUs
  - 28GB RAM
  - Enterprise support

Monthly Cost: ~$450-650
```

### Option D: Railway/Render (Managed)
```yaml
Railway:
  - GPU instances (Beta)
  - Built-in PostgreSQL
  - Automatic SSL
  - GitHub integration
  - Monthly Cost: ~$500-700

Render:
  - GPU instances available
  - Managed PostgreSQL
  - Auto-deploy from Git
  - Monthly Cost: ~$600-800
```

### Option E: Self-Hosted VPS
```yaml
Providers:
  Hetzner Cloud:
    - Dedicated GPU servers
    - Location: Germany
    - Cost: €150-300/month
  
  OVHcloud:
    - GPU instances
    - Location: Europe/Canada
    - Cost: €200-400/month
```

## 📦 Technical Stack

### Core Dependencies
```bash
# Python & Framework
python==3.11+
fastapi==0.115.0
uvicorn[standard]==0.30.1
gunicorn==21.2.0

# AI/ML with GPU Support
torch==2.1.0+cu121  # CUDA 12.1 version
openai-whisper==20231117
faster-whisper==1.0.0  # Alternative: 2x faster

# Database
psycopg2-binary==2.9.9
redis==5.0.1

# YouTube Support
yt-dlp==2024.1.0

# Authentication
authlib==1.3.0
python-jose[cryptography]==3.3.0

# Async Processing
celery==5.3.4
flower==2.0.1  # Monitoring
```

### Docker Configuration
```dockerfile
# Dockerfile.production
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3.11 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3.11 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=small
ENV CUDA_VISIBLE_DEVICES=0

# Run with Gunicorn
CMD ["gunicorn", "src.app.main_api_service:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

### Docker Compose
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/voiceforge_db
      - REDIS_URL=redis://redis:6379
      - WHISPER_MODEL=small
      - GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}
      - GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}
    depends_on:
      - db
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./uploads:/app/uploads

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=voiceforge_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
```

## 💰 Cost Analysis

### Monthly Cost Breakdown

#### Small Scale (1,000 hours/month)
```
AWS EC2 g4dn.xlarge: $380
RDS PostgreSQL:       $30
S3 Storage (100GB):   $3
CloudFront CDN:       $10
Total:                ~$425/month
```

#### Medium Scale (10,000 hours/month)
```
AWS EC2 g4dn.xlarge: $380
RDS PostgreSQL:       $60
S3 Storage (1TB):     $23
CloudFront CDN:       $50
Elastic IP:           $4
Total:                ~$520/month
```

#### Large Scale (100,000+ hours/month)
```
AWS EC2 g4dn.2xlarge x2: $1,520
RDS PostgreSQL (large):   $240
S3 Storage (10TB):        $230
CloudFront CDN:           $200
Load Balancer:            $25
Total:                    ~$2,215/month
```

## 📊 Deployment Steps

### 1. AWS Deployment Example

```bash
# Step 1: Launch EC2 Instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-groups voiceforge-sg

# Step 2: Install NVIDIA Drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-525

# Step 3: Install Docker with GPU Support
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install -y nvidia-container-toolkit

# Step 4: Clone Repository
git clone https://github.com/voiceforge/stt-service.git
cd stt-service

# Step 5: Set Environment Variables
cp .env.example .env.production
nano .env.production

# Step 6: Build and Run
docker-compose -f docker-compose.production.yml up -d

# Step 7: Setup SSL with Certbot
sudo certbot --nginx -d api.voiceforge.ai
```

### 2. Database Setup

```sql
-- Create production database
CREATE DATABASE voiceforge_prod;
CREATE USER voiceforge_user WITH ENCRYPTED PASSWORD 'strong_password';
GRANT ALL PRIVILEGES ON DATABASE voiceforge_prod TO voiceforge_user;

-- Enable extensions
\c voiceforge_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for performance
CREATE INDEX idx_transcriptions_user_id ON transcriptions(user_id);
CREATE INDEX idx_transcriptions_created_at ON transcriptions(created_at);
CREATE INDEX idx_api_keys_key ON api_keys(key);
CREATE INDEX idx_users_email ON users(email);
```

## ⚡ Performance Optimization

### 1. Whisper Model Optimization
```python
# Use faster-whisper for 2x speed improvement
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda", compute_type="float16")
```

### 2. Redis Caching
```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_transcription(audio_hash):
    cached = redis_client.get(f"trans:{audio_hash}")
    if cached:
        return json.loads(cached)
    return None

def cache_transcription(audio_hash, result, ttl=3600):
    redis_client.setex(
        f"trans:{audio_hash}",
        ttl,
        json.dumps(result)
    )
```

### 3. Nginx Configuration
```nginx
# nginx.conf
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    upstream voiceforge {
        least_conn;
        server app:8000 fail_timeout=0;
    }

    server {
        listen 80;
        server_name api.voiceforge.ai;
        
        client_max_body_size 100M;
        keepalive_timeout 65;

        location / {
            proxy_pass http://voiceforge;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Host $http_host;
            proxy_redirect off;
            proxy_buffering off;
        }

        location /static/ {
            alias /app/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### 4. Gunicorn Configuration
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

## 📈 Monitoring & Scaling

### Monitoring Setup
```yaml
# Prometheus metrics
prometheus:
  scrape_interval: 15s
  targets:
    - voiceforge:8000/metrics

# Grafana dashboards
dashboards:
  - GPU utilization
  - Request latency
  - Transcription duration
  - Error rates
  - Database connections
```

### Auto-scaling Configuration
```yaml
# AWS Auto Scaling
scaling_policy:
  target_tracking:
    metric: GPU_Utilization
    target_value: 70
  scale_up:
    threshold: 80%
    duration: 5 minutes
  scale_down:
    threshold: 30%
    duration: 15 minutes
  min_instances: 1
  max_instances: 5
```

### Health Checks
```python
@app.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "database": check_database(),
        "redis": check_redis(),
        "gpu": torch.cuda.is_available(),
        "model_loaded": whisper_model is not None,
        "disk_space": check_disk_space(),
        "memory": check_memory()
    }
    
    if not all(checks.values()):
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "checks": checks}
        )
    
    return checks
```

## 🔒 Security Considerations

### Environment Variables
```bash
# .env.production
SECRET_KEY=<generate-strong-key>
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
GOOGLE_CLIENT_ID=<oauth-client-id>
GOOGLE_CLIENT_SECRET=<oauth-secret>
SENTRY_DSN=<error-tracking>
AWS_ACCESS_KEY_ID=<aws-key>
AWS_SECRET_ACCESS_KEY=<aws-secret>
```

### SSL/TLS Setup
```bash
# Certbot for Let's Encrypt
sudo certbot certonly --standalone -d api.voiceforge.ai
```

### Firewall Rules
```bash
# UFW configuration
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 5432/tcp  # PostgreSQL (restrict to app server)
sudo ufw enable
```

## 🚨 Backup Strategy

### Database Backup
```bash
# Daily backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump voiceforge_prod > backup_$TIMESTAMP.sql
aws s3 cp backup_$TIMESTAMP.sql s3://voiceforge-backups/
```

### Redis Persistence
```conf
# redis.conf
save 900 1
save 300 10
save 60 10000
appendonly yes
```

## 📝 Deployment Checklist

- [ ] GPU instance provisioned
- [ ] NVIDIA drivers installed
- [ ] Docker with GPU support configured
- [ ] PostgreSQL database created
- [ ] Redis cache configured
- [ ] Environment variables set
- [ ] SSL certificates installed
- [ ] Nginx reverse proxy configured
- [ ] Monitoring tools setup
- [ ] Backup strategy implemented
- [ ] Health checks configured
- [ ] Auto-scaling policies defined
- [ ] Security groups configured
- [ ] DNS records updated
- [ ] Load testing completed

## 🆘 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Slow Transcription
```python
# Use smaller model or batch processing
model = whisper.load_model("tiny")  # For faster processing
```

## 📞 Support

For deployment assistance:
- Documentation: https://docs.voiceforge.ai
- Email: devops@voiceforge.ai
- Discord: https://discord.gg/voiceforge

---

**Last Updated**: January 2025
**Version**: 1.0.0