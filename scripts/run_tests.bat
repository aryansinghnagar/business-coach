@echo off
REM Run Business Meeting Copilot test suite (CLI)
REM For GUI with failure details: run scripts\start.bat then use Tests tab
REM Usage: run_tests.bat [--verbose] [pattern]
cd /d "%~dp0.."
python run_tests.py %*
