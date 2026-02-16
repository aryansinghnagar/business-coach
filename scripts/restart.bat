@echo off
REM Restart server: clear port 5000 then start app (safe to run from GUI launcher)
echo Stopping server on port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
)
timeout /t 1 /nobreak >nul

echo Starting server...
cd /d "%~dp0.."
start "Business Meeting Copilot" python app.py

timeout /t 2 /nobreak >nul
echo Server restarted. Check http://localhost:5000
echo.
pause
