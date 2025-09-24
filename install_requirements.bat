@echo off
echo ========================================
echo Installing VoiceForge Requirements
echo ========================================
echo.

echo [1/3] Installing database dependencies...
pip install psycopg2-binary sqlalchemy alembic

echo [2/3] Installing diarization dependencies...
pip install pyannote.audio speechbrain torchaudio

echo [3/3] Installing testing dependencies...
pip install aiohttp pandas matplotlib pytest-asyncio locust

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: setup_postgres.bat (to set up database)
echo 2. Run: python src\app\main_full.py (to start API)
echo 3. Run: python tests\load_test.py (to run performance tests)
echo.
pause