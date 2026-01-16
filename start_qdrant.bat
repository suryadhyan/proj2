@echo off
echo Starting Qdrant vector database...
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Docker is running. Starting Qdrant...
echo.

REM Stop existing Qdrant container if any
docker stop qdrant 2>nul
docker rm qdrant 2>nul

REM Start Qdrant
docker run -d --name qdrant ^
    -p 6333:6333 ^
    -p 6334:6334 ^
    -v D:\qdrant_storage:/qdrant/storage:z ^
    qdrant/qdrant

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS: Qdrant is now running!
    echo ========================================
    echo.
    echo Qdrant HTTP API: http://localhost:6333
    echo Qdrant Web UI: http://localhost:6333/dashboard
    echo.
    echo You can now run the tests:
    echo   python test_complete_flow.py
    echo.
) else (
    echo.
    echo ERROR: Failed to start Qdrant
    pause
)

pause
