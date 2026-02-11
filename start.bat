@echo off
REM Business Meeting Copilot - Desktop launcher
REM Double-click to open the launcher (server + tests)

echo Starting Business Meeting Copilot...
python launcher.py

if errorlevel 1 (
    echo.
    echo Error: Could not start the control panel.
    echo Make sure Python is installed and in your PATH.
    echo.
    pause
)
