@echo off
REM Run Business Meeting Copilot test suite (CLI)
REM For GUI with failure details: python control_panel.py (Tests tab)
REM Usage: run_tests.bat [--verbose] [pattern]
python run_tests.py %*
