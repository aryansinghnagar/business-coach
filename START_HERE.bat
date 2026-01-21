@echo off
REM Simple launcher script for Windows
REM Double-click this file to start the GUI launcher

echo Starting Business Meeting Copilot Launcher...
python gui_launcher.py

if errorlevel 1 (
    echo.
    echo Error: Could not start the launcher.
    echo Make sure Python is installed and in your PATH.
    echo.
    pause
)
