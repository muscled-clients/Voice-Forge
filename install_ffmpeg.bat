@echo off
echo ========================================
echo Installing FFmpeg for Audio Processing
echo ========================================
echo.

echo FFmpeg is required for Whisper to process audio files.
echo.
echo Please choose an option:
echo 1. Download FFmpeg manually
echo 2. Install via pip (ffmpeg-python)
echo.

echo Installing ffmpeg-python wrapper...
pip install ffmpeg-python

echo.
echo ========================================
echo IMPORTANT: FFmpeg Installation
echo ========================================
echo.
echo If transcription fails, you need to install FFmpeg:
echo.
echo Option 1 (Recommended):
echo   1. Download from: https://www.gyan.dev/ffmpeg/builds/
echo   2. Extract to C:\ffmpeg
echo   3. Add C:\ffmpeg\bin to your PATH
echo.
echo Option 2 (Via Chocolatey if installed):
echo   choco install ffmpeg
echo.
echo Option 3 (Via Scoop if installed):
echo   scoop install ffmpeg
echo.
echo After installing FFmpeg, restart the VoiceForge server.
echo ========================================
echo.
pause