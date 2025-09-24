#!/usr/bin/env python
"""
Simple VoiceForge SDK Example
Demonstrates basic transcription using the VoiceForge Python SDK
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path to import local SDK
sys.path.insert(0, str(Path(__file__).parent.parent))

from voiceforge import VoiceForgeClient


async def main():
    """Simple transcription example"""
    
    # Initialize client with API key
    # You can set VOICEFORGE_API_KEY environment variable or pass directly
    api_key = "your_api_key_here"  # Replace with your actual API key
    
    # For local development/testing
    client = VoiceForgeClient(
        api_key=api_key,
        base_url="http://localhost:8000"  # Local server
    )
    
    # Example 1: Transcribe a file
    audio_file = "path/to/your/audio.mp3"  # Replace with your audio file
    
    try:
        print(f"Transcribing {audio_file}...")
        
        # Since our current API doesn't support async job tracking,
        # we'll use the direct endpoint
        import httpx
        
        # Direct API call for now
        async with httpx.AsyncClient() as http_client:
            with open(audio_file, 'rb') as f:
                files = {'file': (Path(audio_file).name, f, 'audio/mpeg')}
                headers = {'Authorization': f'Bearer {api_key}'}
                
                response = await http_client.post(
                    f"{client.base_url}/api/v1/transcribe",
                    files=files,
                    headers=headers,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("\n✅ Transcription successful!")
                    print(f"Text: {result['transcription']['text']}")
                    print(f"Language: {result['transcription']['language']}")
                    print(f"Duration: {result['transcription']['duration']}s")
                    print(f"Word Count: {result['transcription']['word_count']}")
                    print(f"Confidence: {result['transcription']['confidence']}")
                    print(f"Processing Time: {result['transcription']['processing_time']}s")
                else:
                    print(f"❌ Error: {response.status_code} - {response.text}")
                    
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


def sync_main():
    """Wrapper for sync execution"""
    asyncio.run(main())


if __name__ == "__main__":
    print("🎤 VoiceForge SDK Example - Simple Transcription")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("VOICEFORGE_API_KEY")
    if not api_key:
        print("⚠️ Please set your API key:")
        print("  export VOICEFORGE_API_KEY='your_key_here'")
        print("  or edit this script and set api_key directly")
        sys.exit(1)
    
    sync_main()