@echo off
REM Run Business Meeting Copilot test suite (CLI)
REM For GUI with failure details: python launcher.py (Tests tab)
REM Usage: run_tests.bat [--verbose] [pattern]
python test_runner.py %*
