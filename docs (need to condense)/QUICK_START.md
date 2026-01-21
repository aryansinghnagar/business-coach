# Quick Start Guide - GUI Launcher

## For Non-Technical Users

### Windows Users

1. **Double-click** `START_HERE.bat` file
   - OR double-click `gui_launcher.py`
   - OR run: `python gui_launcher.py`

2. **Click "▶ Start Server"** button

3. **Wait for green status** (takes 5-10 seconds)

4. **Click "🌐 Open in Browser"** button

5. **When done, click "■ Stop Server"**

### Mac/Linux Users

1. **Double-click** `START_HERE.sh` file
   - OR run: `python3 gui_launcher.py`

2. **Click "▶ Start Server"** button

3. **Wait for green status** (takes 5-10 seconds)

4. **Click "🌐 Open in Browser"** button

5. **When done, click "■ Stop Server"**

## What You'll See

- **Green Circle** = Server is running ✓
- **Red Circle** = Server is stopped
- **Yellow Circle** = Server is starting/stopping

## Troubleshooting

**"Cannot find app.py" error:**
- Make sure you're running the launcher from the project folder
- The `gui_launcher.py` file should be in the same folder as `app.py`

**"Python not found" error:**
- Install Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

**Server won't start:**
- Check the "Server Logs" area for error messages
- Make sure port 5000 is not already in use
- Try closing other applications that might be using port 5000

**Browser won't open:**
- Make sure the server is running (green status)
- Try manually opening: http://localhost:5000

## That's It!

The GUI launcher makes it easy to start and stop the application without using the command line.
