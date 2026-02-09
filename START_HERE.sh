#!/bin/bash
# Business Meeting Copilot - Unified Control Panel
# Run from terminal to open the control panel (server + tests)

echo "Starting Business Meeting Copilot Control Panel..."
python3 control_panel.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Could not start the control panel."
    echo "Make sure Python 3 is installed."
    echo ""
    read -p "Press Enter to exit..."
fi
