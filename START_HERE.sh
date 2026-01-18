#!/bin/bash
# Simple launcher script for Linux/Mac
# Double-click this file (or run from terminal) to start the GUI launcher

echo "Starting Business Meeting Copilot Launcher..."
python3 gui_launcher.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Could not start the launcher."
    echo "Make sure Python 3 is installed."
    echo ""
    read -p "Press Enter to exit..."
fi
