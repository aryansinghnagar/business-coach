@echo off
echo Stopping all Python processes...
taskkill /F /IM python.exe 2>nul

echo Clearing port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    taskkill /F /PID %%a 2>nul
)

echo All instances stopped.
timeout /t 2 /nobreak >nul
