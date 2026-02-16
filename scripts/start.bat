@echo off
REM Business Meeting Copilot - Desktop launcher
REM Double-click from anywhere, or run from project root: scripts\start.bat

cd /d "%~dp0"
echo Starting Business Meeting Copilot...
python launch.py

if errorlevel 1 (
    echo.
    echo Error: Could not start the launcher.
    echo Make sure Python is installed and in your PATH.
    echo.
    pause
)
