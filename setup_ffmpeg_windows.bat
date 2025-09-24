@echo off
echo ========================================
echo FFmpeg Setup for VoiceForge STT
echo ========================================
echo.

REM Check if ffmpeg already exists
if exist ffmpeg.exe (
    echo FFmpeg already exists in project directory!
    ffmpeg -version
    echo.
    pause
    exit /b 0
)

echo [1/4] Downloading FFmpeg for Windows...
echo This may take a few minutes...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'ffmpeg-temp.zip' -UseBasicParsing}"

if not exist ffmpeg-temp.zip (
    echo ERROR: Failed to download FFmpeg!
    echo Please check your internet connection.
    pause
    exit /b 1
)

echo [2/4] Extracting FFmpeg...
powershell -Command "& {Expand-Archive -Path 'ffmpeg-temp.zip' -DestinationPath 'ffmpeg-extract' -Force}"

echo [3/4] Moving FFmpeg executables to project directory...
for /d %%i in (ffmpeg-extract\ffmpeg-*) do (
    copy "%%i\bin\ffmpeg.exe" . >nul 2>&1
    copy "%%i\bin\ffprobe.exe" . >nul 2>&1
    copy "%%i\bin\ffplay.exe" . >nul 2>&1
)

echo [4/4] Cleaning up temporary files...
rmdir /s /q ffmpeg-extract 2>nul
del ffmpeg-temp.zip 2>nul

REM Verify installation
if exist ffmpeg.exe (
    echo.
    echo ========================================
    echo SUCCESS! FFmpeg installed successfully!
    echo ========================================
    echo.
    echo Testing FFmpeg...
    ffmpeg -version | findstr version
    echo.
    echo FFmpeg is now ready to use with VoiceForge.
    echo Please restart the VoiceForge server now.
) else (
    echo.
    echo ERROR: FFmpeg installation failed!
    echo Please try downloading manually from:
    echo https://ffmpeg.org/download.html
)

echo.
pause