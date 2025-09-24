# VoiceForge Python SDK

Official Python SDK for the VoiceForge Speech-to-Text API. Transform audio into text with state-of-the-art accuracy and ultra-low latency.

[![PyPI version](https://badge.fury.io/py/voiceforge-python.svg)](https://badge.fury.io/py/voiceforge-python)
[![Python versions](https://img.shields.io/pypi/pyversions/voiceforge-python.svg)](https://pypi.org/project/voiceforge-python/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Multiple Model Support**: Whisper, Canary, and custom models
- **Real-time Streaming**: WebSocket-based streaming transcription
- **Batch Processing**: Efficient processing of multiple files
- **Speaker Diarization**: Identify different speakers
- **Language Detection**: Automatic language identification
- **Word-level Timestamps**: Precise timing for each word
- **Export Formats**: SRT, VTT, JSON, and plain text
- **Async/Await Support**: Modern Python async programming
- **CLI Tool**: Command-line interface for quick tasks
- **Rich Error Handling**: Detailed error messages and retry logic

## Installation

### Basic Installation

```bash
pip install voiceforge-python
```

### With Audio Processing

```bash
pip install voiceforge-python[audio]
```

### With CLI Tools

```bash
pip install voiceforge-python[cli]
```

### Full Installation

```bash
pip install voiceforge-python[all]
```

## Quick Start

### API Key Setup

Get your API key from the [VoiceForge Dashboard](https://dashboard.voiceforge.ai) and set it as an environment variable:

```bash
export VOICEFORGE_API_KEY="your_api_key_here"
```

### Basic Transcription

```python
import asyncio
import voiceforge

async def main():
    # Initialize client
    client = voiceforge.VoiceForgeClient()
    
    # Transcribe a file
    async with client:
        job = await client.transcribe_file("meeting.wav")
        print(f"Transcript: {job.transcript}")

asyncio.run(main())
```

### Streaming Transcription

```python
import asyncio
import voiceforge

async def stream_example():
    client = voiceforge.VoiceForgeClient()
    
    async with client:
        async for result in client.stream_transcription():
            if result.is_final:
                print(f"Final: {result.transcript}")
            else:
                print(f"Partial: {result.transcript}")

asyncio.run(stream_example())
```

## API Reference

### VoiceForgeClient

The main client class for interacting with the API.

```python
client = voiceforge.VoiceForgeClient(
    api_key="your_api_key",          # Optional if set in environment
    base_url="https://api.voiceforge.ai",  # API base URL
    timeout=30.0,                    # Request timeout in seconds
    max_retries=3,                   # Maximum retry attempts
)
```

### Transcription Methods

#### transcribe_file()

Transcribe an audio file:

```python
job = await client.transcribe_file(
    file_path="audio.wav",
    options=voiceforge.TranscriptionOptions(
        model="whisper-medium",
        language="en",
        enable_diarization=True,
        enable_word_timestamps=True,
    ),
    wait_for_completion=True,
    progress_callback=lambda p: print(f"Progress: {p:.1%}")
)
```

#### transcribe_bytes()

Transcribe audio from bytes:

```python
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

job = await client.transcribe_bytes(
    audio_bytes=audio_bytes,
    filename="audio.wav",
    options=options
)
```

#### transcribe_url()

Transcribe audio from URL:

```python
job = await client.transcribe_url(
    audio_url="https://example.com/audio.mp3",
    options=options
)
```

#### transcribe_batch()

Batch transcribe multiple files:

```python
files = ["audio1.wav", "audio2.mp3", "audio3.flac"]
batch_job = await client.transcribe_batch(
    file_paths=files,
    options=options,
    progress_callback=lambda job_id, progress: print(f"{job_id}: {progress:.1%}")
)
```

### Streaming Transcription

Real-time streaming transcription:

```python
async for result in client.stream_transcription(
    options=voiceforge.TranscriptionOptions(
        model="whisper-base",
        language="en"
    ),
    audio_format=voiceforge.AudioFormat.WAV,
    sample_rate=16000
):
    print(f"{'[FINAL]' if result.is_final else '[PARTIAL]'} {result.transcript}")
```

### Job Management

#### Get Job Status

```python
job = await client.get_job("job_id_here")
print(f"Status: {job.status}")
print(f"Transcript: {job.transcript}")
```

#### List Jobs

```python
jobs = await client.list_jobs(
    status=voiceforge.TranscriptionStatus.COMPLETED,
    limit=20
)

for job in jobs.items:
    print(f"{job.id}: {job.filename}")
```

#### Cancel Job

```python
cancelled_job = await client.cancel_job("job_id_here")
```

### Models and Analytics

#### List Available Models

```python
models = await client.list_models()
for model in models:
    print(f"{model.model_id}: {model.name}")
```

#### Get User Analytics

```python
analytics = await client.get_user_analytics(
    start_date="2024-01-01",
    end_date="2024-01-31"
)
print(f"Total transcriptions: {analytics.total_transcriptions}")
print(f"Success rate: {analytics.success_rate:.1%}")
```

## Transcription Options

Customize transcription behavior with `TranscriptionOptions`:

```python
options = voiceforge.TranscriptionOptions(
    # Model selection
    model="whisper-medium",           # Model to use
    language="en",                    # Expected language (optional)
    
    # Features
    enable_diarization=True,          # Speaker identification
    enable_punctuation=True,          # Automatic punctuation
    enable_word_timestamps=True,      # Word-level timing
    enable_language_detection=False,  # Auto-detect language
    
    # Diarization settings
    max_speakers=5,                   # Maximum number of speakers
    
    # Performance tuning
    temperature=0.0,                  # Sampling temperature
    compression_ratio_threshold=2.4,  # Compression ratio threshold
    logprob_threshold=-1.0,           # Log probability threshold
    no_speech_threshold=0.6,          # No speech threshold
    
    # Custom vocabulary
    boost_keywords=["VoiceForge", "API"],  # Keywords to boost
    custom_vocabulary=["VoiceForge"],       # Custom vocabulary
)
```

## Audio Format Support

VoiceForge supports multiple audio formats:

- **WAV** (`.wav`) - Uncompressed, best quality
- **MP3** (`.mp3`) - Compressed, widely supported
- **M4A** (`.m4a`) - Apple's compressed format
- **FLAC** (`.flac`) - Lossless compression
- **OGG** (`.ogg`) - Open source compressed format
- **WebM** (`.webm`) - Web-optimized format

```python
# Format detection is automatic
job = await client.transcribe_file("audio.mp3")  # Detects MP3 format
```

## Export Formats

Export transcripts in various formats:

### Plain Text

```python
print(job.transcript)
```

### JSON with Full Details

```python
import json
print(json.dumps(job.model_dump(), indent=2, default=str))
```

### SRT Subtitles

```python
from voiceforge.utils import export_to_srt

if job.words:
    word_dicts = [word.model_dump() for word in job.words]
    srt_content = export_to_srt(word_dicts, output_path="subtitles.srt")
```

### WebVTT Subtitles

```python
from voiceforge.utils import export_to_vtt

if job.words:
    word_dicts = [word.model_dump() for word in job.words]
    vtt_content = export_to_vtt(word_dicts, output_path="subtitles.vtt")
```

## Error Handling

The SDK provides detailed error handling:

```python
import voiceforge

try:
    async with voiceforge.VoiceForgeClient() as client:
        job = await client.transcribe_file("nonexistent.wav")
        
except voiceforge.FileError as e:
    print(f"File error: {e}")
    
except voiceforge.AuthenticationError as e:
    print(f"Auth error: {e}")
    
except voiceforge.RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
    
except voiceforge.TranscriptionError as e:
    print(f"Transcription failed: {e}")
    print(f"Job ID: {e.job_id}")
    
except voiceforge.VoiceForgeError as e:
    print(f"API error: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Status code: {e.status_code}")
```

## CLI Usage

The SDK includes a powerful command-line interface:

### Transcribe a File

```bash
voiceforge transcribe audio.wav --model whisper-medium --language en
```

### Batch Processing

```bash
voiceforge batch ./audio_files/ --output-dir ./transcripts/ --format srt
```

### List Jobs

```bash
voiceforge jobs --status completed --limit 10
```

### Get Job Details

```bash
voiceforge job job_12345 --save transcript.txt
```

### Configure API Key

```bash
voiceforge configure --api-key your_api_key
```

For full CLI documentation, run:

```bash
voiceforge --help
```

## Advanced Usage

### Custom Progress Tracking

```python
from voiceforge.utils import create_progress_callback

# Create a rich progress bar
progress_callback, progress_bar = create_progress_callback("Transcribing")

with progress_bar:
    job = await client.transcribe_file(
        "long_audio.wav",
        progress_callback=progress_callback
    )
```

### Retry with Backoff

```python
from voiceforge.utils import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=1.0)
async def transcribe_with_retry():
    return await client.transcribe_file("audio.wav")

job = await transcribe_with_retry()
```

### Audio File Validation

```python
from voiceforge.utils import validate_audio_file

try:
    file_info = validate_audio_file("audio.wav")
    print(f"Format: {file_info['format']}")
    print(f"Size: {file_info['size']} bytes")
except voiceforge.AudioError as e:
    print(f"Invalid audio file: {e}")
```

### Find Audio Files

```python
from voiceforge.utils import find_audio_files

audio_files = find_audio_files(
    directory="./audio",
    recursive=True,
    include_formats=[voiceforge.AudioFormat.WAV, voiceforge.AudioFormat.MP3]
)

for file_path in audio_files:
    print(f"Found: {file_path}")
```

## Model Information

### Available Models

| Model | Type | Languages | GPU Required | Accuracy | Speed |
|-------|------|-----------|--------------|----------|-------|
| whisper-tiny | Whisper | 99+ | No | Good | Very Fast |
| whisper-base | Whisper | 99+ | No | Better | Fast |
| whisper-small | Whisper | 99+ | Yes | Good | Medium |
| whisper-medium | Whisper | 99+ | Yes | Very Good | Medium |
| whisper-large | Whisper | 99+ | Yes | Excellent | Slower |
| canary-1b | NVIDIA | 100+ | Yes | Excellent | Fast |

### Language Support

VoiceForge supports 100+ languages including:

- **European**: English, Spanish, French, German, Italian, Portuguese, Russian, Polish, Dutch, Swedish, Danish, Norwegian, Finnish, Czech
- **Asian**: Japanese, Korean, Chinese (Mandarin), Hindi, Arabic, Turkish
- **And many more...**

## Performance Tips

### 1. Choose the Right Model

- **whisper-base**: Best for real-time applications
- **whisper-medium**: Good balance of speed and accuracy
- **whisper-large**: Best accuracy for critical applications
- **canary-1b**: Best for multilingual content

### 2. Optimize Audio Files

- Use WAV format for best quality
- 16kHz sample rate is optimal
- Mono audio is sufficient for speech

### 3. Batch Processing

Use batch processing for multiple files:

```python
# Process files in batches of 10
from voiceforge.utils import batch_process_files

file_batches = batch_process_files(audio_files, batch_size=10)
for batch in file_batches:
    batch_job = await client.transcribe_batch(batch)
```

### 4. Streaming for Real-time

Use streaming for real-time applications:

```python
# Lower latency with smaller chunks
async for result in client.stream_transcription(
    chunk_duration=0.1,  # 100ms chunks
    options=options
):
    if result.is_final:
        process_final_result(result)
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
import asyncio
import voiceforge

app = Flask(__name__)
client = voiceforge.VoiceForgeClient()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    # Run async transcription
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        job = loop.run_until_complete(
            client.transcribe_bytes(
                audio_bytes=audio_bytes,
                filename=audio_file.filename
            )
        )
        return jsonify({
            'transcript': job.transcript,
            'confidence': job.confidence
        })
    finally:
        loop.close()
```

### FastAPI Application

```python
from fastapi import FastAPI, UploadFile, File
import voiceforge

app = FastAPI()
client = voiceforge.VoiceForgeClient()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    
    async with client:
        job = await client.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=audio.filename
        )
    
    return {
        "transcript": job.transcript,
        "confidence": job.confidence,
        "language": job.language_code
    }
```

### Jupyter Notebook

```python
# Install in notebook
!pip install voiceforge-python[all]

import voiceforge
import asyncio

# Set up client
client = voiceforge.VoiceForgeClient(api_key="your_key")

# Transcribe with progress bar
async def transcribe_with_progress(file_path):
    from IPython.display import display, clear_output
    
    def progress_callback(percent):
        clear_output(wait=True)
        print(f"Progress: {'█' * int(percent * 20):<20} {percent:.1%}")
    
    async with client:
        job = await client.transcribe_file(
            file_path,
            progress_callback=progress_callback
        )
    
    clear_output(wait=True)
    print("✅ Transcription completed!")
    return job

# Run transcription
job = await transcribe_with_progress("audio.wav")
print(f"Transcript: {job.transcript}")
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/voiceforge/voiceforge-python
cd voiceforge-python
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://docs.voiceforge.ai](https://docs.voiceforge.ai)
- **API Reference**: [https://docs.voiceforge.ai/api](https://docs.voiceforge.ai/api)
- **Issues**: [GitHub Issues](https://github.com/voiceforge/voiceforge-python/issues)
- **Discord**: [Join our community](https://discord.gg/voiceforge)
- **Email**: support@voiceforge.ai

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.