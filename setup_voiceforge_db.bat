@echo off
echo ========================================
echo VoiceForge Database Setup (PostgreSQL)
echo ========================================
echo.

echo [1/3] Creating VoiceForge database...
set PGPASSWORD=Gateway123
psql -U postgres -c "CREATE DATABASE voiceforge_db;" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Database 'voiceforge_db' created successfully
) else (
    echo ⚠️  Database 'voiceforge_db' may already exist - continuing...
)

echo.
echo [2/3] Setting up database schema...
psql -U postgres -d voiceforge_db -f database\init_simple.sql
if %errorlevel% equ 0 (
    echo ✅ Database schema created successfully
) else (
    echo ❌ Error creating schema
    pause
    exit /b 1
)

echo.
echo [3/3] Testing database connection...
psql -U postgres -d voiceforge_db -c "SELECT 'VoiceForge DB Ready!' as status;"
if %errorlevel% equ 0 (
    echo ✅ Database connection test successful
) else (
    echo ❌ Database connection test failed
)

echo.
echo ========================================
echo VoiceForge Database Setup Complete!
echo ========================================
echo.
echo Connection Details:
echo   Host: localhost
echo   Port: 5432
echo   Database: voiceforge_db
echo   Username: postgres
echo   Password: Gateway123
echo.
echo pgAdmin Connection:
echo   Add new connection with above details
echo.
pause