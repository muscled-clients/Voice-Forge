#!/usr/bin/env python3
"""
Local Setup Script for VoiceForge STT
Run this to set up VoiceForge without Docker
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, shell=True):
    """Run command and return success"""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Create virtual environment
    if not run_command("python -m venv venv"):
        return False
    
    # Activate and install
    if os.name == 'nt':  # Windows
        activate_cmd = r"venv\Scripts\activate && pip install -r requirements.txt"
    else:  # Unix
        activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
    
    return run_command(activate_cmd)

def setup_database():
    """Setup local SQLite database"""
    print("\n🗄️ Setting up database...")
    
    # Create .env for local development
    env_content = """# Local Development Environment
DEBUG=true
LOG_LEVEL=INFO

# SQLite Database (local)
DATABASE_URL=sqlite:///./voiceforge.db

# Redis (optional - will use memory if not available)
REDIS_URL=redis://localhost:6379/0

# JWT Secret
JWT_SECRET_KEY=local-development-secret-key-change-in-production

# API Settings
DEFAULT_MODEL=whisper-base
HUGGINGFACE_TOKEN=

# Local server
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Features
ENABLE_DIARIZATION=false
ENABLE_LANGUAGE_DETECTION=true
ENABLE_STREAMING=true
ENABLE_BATCH_PROCESSING=true
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    return True

def create_startup_script():
    """Create startup script"""
    print("\n🚀 Creating startup script...")
    
    if os.name == 'nt':  # Windows
        startup_content = """@echo off
echo Starting VoiceForge STT Service...
call venv\\Scripts\\activate
python -c "
import asyncio
from app.db.session import init_db
asyncio.run(init_db())
print('Database initialized')
"
echo Starting server...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
        script_name = "start_voiceforge.bat"
    else:  # Unix
        startup_content = """#!/bin/bash
echo "Starting VoiceForge STT Service..."
source venv/bin/activate
python -c "
import asyncio
from app.db.session import init_db
asyncio.run(init_db())
print('Database initialized')
"
echo "Starting server..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
        script_name = "start_voiceforge.sh"
    
    with open(script_name, "w") as f:
        f.write(startup_content)
    
    if os.name != 'nt':
        os.chmod(script_name, 0o755)
    
    print(f"✅ Created {script_name}")
    return True

def create_test_script():
    """Create test script"""
    print("\n🧪 Creating test script...")
    
    test_content = '''#!/usr/bin/env python3
"""Test VoiceForge local setup"""

import asyncio
import aiohttp
import json

async def test_api():
    """Test basic API functionality"""
    base_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            print("Testing health endpoint...")
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print(f"✅ Health: {health}")
                else:
                    print(f"❌ Health check failed: {resp.status}")
                    return
            
            # Test models endpoint  
            print("Testing models endpoint...")
            async with session.get(f"{base_url}/api/v1/models") as resp:
                if resp.status == 200:
                    models = await resp.json()
                    print(f"✅ Available models: {len(models.get('models', []))}")
                    for model in models.get('models', [])[:3]:
                        print(f"   - {model.get('name', model.get('model_id'))}")
                else:
                    print(f"❌ Models endpoint failed: {resp.status}")
            
            print("\\n🎉 Local setup is working!")
            print(f"🌐 Web UI: {base_url}")
            print(f"📚 API Docs: {base_url}/docs")
            print(f"🔧 GraphQL: {base_url}/graphql")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure the server is running with: start_voiceforge.bat")

if __name__ == "__main__":
    asyncio.run(test_api())
'''
    
    with open("test_local.py", "w") as f:
        f.write(test_content)
    
    print("✅ Created test_local.py")
    return True

def main():
    """Main setup function"""
    print("🔧 VoiceForge Local Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        print("Try manually:")
        print("  python -m venv venv")
        if os.name == 'nt':
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Create scripts
    if not create_startup_script():
        return False
    
    if not create_test_script():
        return False
    
    print("\n" + "=" * 40)
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Start the service:")
    if os.name == 'nt':
        print("   start_voiceforge.bat")
    else:
        print("   ./start_voiceforge.sh")
    
    print("2. In another terminal, test it:")
    print("   python test_local.py")
    
    print("\n3. Open your browser:")
    print("   http://localhost:8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)