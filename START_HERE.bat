@echo off
REM Business Meeting Copilot - Unified Control Panel
REM Double-click to open the control panel (server + tests)

echo Starting Business Meeting Copilot Control Panel...
python control_panel.py

if errorlevel 1 (
    echo.
    echo Error: Could not start the control panel.
    echo Make sure Python is installed and in your PATH.
    echo.
    pause
)
