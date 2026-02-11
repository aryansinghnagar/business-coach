#!/bin/bash
# Business Meeting Copilot - Desktop launcher
# Run from terminal to open the launcher (server + tests)

echo "Starting Business Meeting Copilot..."
python3 launcher.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Could not start the control panel."
    echo "Make sure Python 3 is installed."
    echo ""
    read -p "Press Enter to exit..."
fi
