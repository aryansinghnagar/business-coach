#!/bin/bash
# Business Meeting Copilot - Desktop launcher
# Run from project root: ./scripts/start.sh  (or run from scripts/: ./start.sh)

cd "$(dirname "$0")"
echo "Starting Business Meeting Copilot..."
python3 launch.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Could not start the launcher."
    echo "Make sure Python 3 is installed."
    echo ""
    read -p "Press Enter to exit..."
fi
