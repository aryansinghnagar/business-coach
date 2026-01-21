@echo off
echo Stopping all Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Clearing port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    taskkill /F /PID %%a 2>nul
)
timeout /t 1 /nobreak >nul

echo Starting server...
cd /d "%~dp0"
start "Business Meeting Copilot" python app.py

timeout /t 3 /nobreak >nul
echo.
echo Server should be starting. Check http://localhost:5000
echo.
pause
