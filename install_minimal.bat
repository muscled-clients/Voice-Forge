@echo off
echo ========================================
echo Installing Minimal VoiceForge Requirements
echo ========================================
echo.

echo [1/5] Installing core web framework...
pip install fastapi uvicorn python-multipart jinja2

echo [2/5] Installing authentication...
pip install python-jose[cryptography] passlib[bcrypt]

echo [3/5] Installing database...
pip install psycopg2-binary sqlalchemy

echo [4/5] Installing AI models (this may take time)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper

echo [5/5] Installing testing tools...
pip install aiohttp pytest httpx

echo.
echo ========================================
echo Minimal installation complete!
echo ========================================
echo.
echo To run VoiceForge:
echo   .\run_voiceforge.bat
echo.
pause