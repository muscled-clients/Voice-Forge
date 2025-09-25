# VoiceForge Python SDK Setup Guide

## 🚀 Quick Start

### Option 1: Install from Local Source (Current)

Since the SDK is not yet published to PyPI, you can install it locally:

```bash
# Clone or download the SDK
cd sdk/python

# Install in development mode
pip install -e .

# Or install with all features
pip install -e ".[all]"
```

### Option 2: Direct Usage (Without Installation)

```python
import sys
sys.path.append('path/to/sdk/python')

from voiceforge import VoiceForgeClient

client = VoiceForgeClient(api_key="your_api_key")
```

## 📦 Publishing to PyPI (For Maintainers)

### First Time Setup

1. **Create PyPI Account**
   - Register at https://pypi.org/account/register/
   - Verify email

2. **Install Build Tools**
   ```bash
   pip install build twine
   ```

3. **Configure PyPI Credentials**
   Create `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   repository = https://upload.pypi.org/legacy/
   username = __token__
   password = pypi-YOUR_API_TOKEN_HERE

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-YOUR_TEST_API_TOKEN_HERE
   ```

### Publishing Process

1. **Update Version**
   ```python
   # sdk/python/voiceforge/__version__.py
   __version__ = "0.1.0"  # Increment as needed
   ```

2. **Build Package**
   ```bash
   cd sdk/python
   python -m build
   ```

3. **Test on TestPyPI First**
   ```bash
   python -m twine upload --repository testpypi dist/*
   
   # Test installation
   pip install -i https://test.pypi.org/simple/ voiceforge-python
   ```

4. **Publish to PyPI**
   ```bash
   python -m twine upload dist/*
   ```

5. **Verify Installation**
   ```bash
   pip install voiceforge-python
   ```

## 🔧 Current API Compatibility

The SDK is designed for the VoiceForge API v1. Current endpoints:

- `POST /api/v1/transcribe` - Main transcription endpoint
- `GET /api/v1/users/me` - Get user information  
- `GET /api/v1/users/stats` - Get usage statistics

### Basic Usage Example

```python
import asyncio
from voiceforge import VoiceForgeClient

async def transcribe():
    client = VoiceForgeClient(
        api_key="your_api_key",
        base_url="http://localhost:8000"  # or https://api.voiceforge.ai
    )
    
    # For now, use direct HTTP call
    import httpx
    
    async with httpx.AsyncClient() as http:
        with open("audio.mp3", "rb") as f:
            files = {"file": ("audio.mp3", f, "audio/mpeg")}
            headers = {"Authorization": f"Bearer {client.api_key}"}
            
            response = await http.post(
                f"{client.base_url}/api/v1/transcribe",
                files=files,
                headers=headers
            )
            
            result = response.json()
            print(result["transcription"]["text"])

# Run
asyncio.run(transcribe())
```

## 📝 SDK Development TODO

- [ ] Update response models to match actual API
- [ ] Implement WebSocket streaming support
- [ ] Add batch transcription support
- [ ] Create comprehensive test suite
- [ ] Add retry logic for failed requests
- [ ] Implement progress callbacks
- [ ] Add CLI commands
- [ ] Create more examples
- [ ] Publish to PyPI

## 🤝 Contributing

To contribute to the SDK:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Documentation

Full documentation available at:
- API Docs: http://localhost:8000/docs
- SDK Reference: sdk/python/README.md

## 🐛 Known Issues

1. **Async job tracking not implemented** - The current API returns results immediately
2. **WebSocket streaming needs update** - Endpoint URLs need to be updated
3. **Model fields mismatch** - Some response fields don't match the SDK models

## 📧 Support

For issues or questions:
- Email: support@voiceforge.ai
- GitHub Issues: [Create an issue](https://github.com/voiceforge/sdk/issues)