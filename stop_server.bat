@echo off
REM Stop server by clearing port 5000 (safe to run from GUI launcher - does not kill launcher)
echo Stopping server on port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
)
echo Port 5000 cleared.
timeout /t 1 /nobreak >nul
