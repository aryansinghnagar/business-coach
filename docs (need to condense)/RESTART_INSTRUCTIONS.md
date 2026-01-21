# How to Stop and Restart the Project

## Option 1: Using the GUI Launcher (Recommended)

1. **Open the GUI Launcher**: Run `gui_launcher.py` or double-click `START_HERE.bat`
2. **Click "Stop Server"** button to stop the server
3. **Click "Start Server"** button to restart it

## Option 2: Using Batch Scripts

### To Stop:
- Double-click `stop_server.bat`
- OR run: `stop_server.bat` from command prompt

### To Restart:
- Double-click `restart_server.bat`
- OR run: `restart_server.bat` from command prompt

## Option 3: Manual Command Line

### Stop:
```bash
# Stop all Python processes
taskkill /F /IM python.exe

# Or stop specific port
netstat -ano | findstr :5000
taskkill /F /PID <PID_NUMBER>
```

### Start:
```bash
cd c:\Users\Aryan\business-meeting-copilot
python app.py
```

## Option 4: Using Task Manager

1. Open Task Manager (Ctrl+Shift+Esc)
2. Find `python.exe` processes
3. Right-click and select "End Task"
4. Restart using one of the methods above

## Troubleshooting

**Port 5000 still in use:**
- Wait a few seconds after stopping
- Check if another application is using port 5000
- Restart your computer if needed

**Server won't start:**
- Check if port 5000 is free: `netstat -ano | findstr :5000`
- Make sure all Python processes are stopped
- Check for error messages in the console
