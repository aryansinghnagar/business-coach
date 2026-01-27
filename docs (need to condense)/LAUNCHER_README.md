# GUI Launcher for Business Meeting Copilot

## Overview

The GUI Launcher provides a simple, user-friendly interface for starting and stopping the Business Meeting Copilot application. It's designed for non-technical users who want to run the application without using the command line.

## Features

- **Simple Interface**: Easy-to-use buttons for starting and stopping the server
- **Visual Status Indicator**: Color-coded status (green = running, red = stopped, yellow = starting/stopping)
- **Real-time Logs**: See server output in real-time
- **Browser Integration**: One-click button to open the application in your browser
- **Automatic Status Checking**: Automatically detects if the server is running

## How to Use

### Starting the Launcher

1. **Double-click** `gui_launcher.py` to start the launcher
   
   OR
   
2. **From command line**:
   ```bash
   python gui_launcher.py
   ```

### Using the Launcher

1. **Start the Server**:
   - Click the "▶ Start Server" button
   - Wait for the status indicator to turn green
   - The log area will show server startup messages

2. **Open in Browser**:
   - Once the server is running (green status), click "🌐 Open in Browser"
   - This will open `http://localhost:5000` in your default browser

3. **Stop the Server**:
   - When you're done, click the "■ Stop Server" button
   - The server will shut down gracefully

4. **View Logs**:
   - The "Server Logs" area shows real-time output from the server
   - Scroll to see all messages

### Status Indicators

- **🟢 Green**: Server is running and ready
- **🔴 Red**: Server is stopped
- **🟡 Yellow**: Server is starting or stopping

## Requirements

The launcher uses Python's built-in `tkinter` library, which comes with most Python installations. If you get an error about tkinter not being available:

**Windows**: Usually included with Python
**macOS**: Usually included with Python
**Linux**: May need to install:
```bash
sudo apt-get install python3-tk  # Ubuntu/Debian
sudo yum install python3-tkinter  # CentOS/RHEL
```

## Troubleshooting

### Launcher won't start
- Make sure Python is installed and in your PATH
- Check that tkinter is available: `python -c "import tkinter"`

### Server won't start
- Check the log area for error messages
- Make sure port 5000 is not already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Can't open browser
- Make sure the server is running (green status)
- Try manually opening `http://localhost:5000` in your browser

### Server won't stop
- Wait a few seconds for graceful shutdown
- If it doesn't stop, close the launcher window (it will force-stop the server)

## Technical Details

- The launcher runs the Flask server as a subprocess
- Status is checked every 2 seconds automatically
- Server output is captured and displayed in real-time
- The launcher will attempt to stop the server when closed

## Stopping and Restarting

- **GUI**: Click "■ Stop Server", then "▶ Start Server".
- **Batch scripts**: Use `stop_server.bat` to stop, `restart_server.bat` to restart.
- **Command line**: Stop with `taskkill /F /IM python.exe` (or find PID on port 5000 and kill it). Start with `python app.py`.
- **Task Manager**: End `python.exe` processes if needed, then start again via launcher or `python app.py`.

**Port 5000 still in use:** Wait a few seconds after stopping, or check with `netstat -ano | findstr :5000` and free the port.

## Alternative: Command Line

If you prefer using the command line, you can still run:
```bash
python app.py
```

The GUI launcher is just a convenience wrapper around this command.
