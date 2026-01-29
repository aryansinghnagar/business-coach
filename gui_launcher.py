"""
Lightweight GUI Launcher for Business Meeting Copilot
Simple, fast launcher. Stop and Restart buttons run stop_server.bat and
restart_server.bat (Windows); on other platforms uses built-in logic.

Usage:
    python gui_launcher.py
"""

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Error: Tkinter not available. Please install python3-tk")
    sys.exit(1)

import subprocess
import threading
import sys
import os
import time
import socket
from pathlib import Path

class LightweightLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Business Meeting Copilot")
        self.root.geometry("400x200")
        self.root.resizable(False, False)
        
        # Server state
        self.server_process = None
        self.is_running = False
        self._checking = False
        self.script_dir = Path(__file__).resolve().parent
        
        # Setup UI
        self.setup_ui()
        
        # Start status monitoring
        self.check_status()
    
    def setup_ui(self):
        """Create minimal, efficient UI."""
        # Main container
        main = ttk.Frame(self.root, padding="15")
        main.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main, text="Business Meeting Copilot", font=("Arial", 14, "bold"))
        title.pack(pady=(0, 15))
        
        # Status row
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_indicator = tk.Canvas(status_frame, width=16, height=16, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 8))
        
        self.status_label = ttk.Label(status_frame, text="Checking...", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)
        
        # Buttons row
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_server, width=10)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_server, width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.restart_btn = ttk.Button(btn_frame, text="Restart", command=self.restart_server, width=10, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=2)
        
        self.browser_btn = ttk.Button(btn_frame, text="Open Browser", command=self.open_browser, width=12, state=tk.DISABLED)
        self.browser_btn.pack(side=tk.LEFT, padx=2)
        
        # Log line
        self.log_label = ttk.Label(main, text="", font=("Consolas", 8), foreground="gray", wraplength=370)
        self.log_label.pack(fill=tk.X, pady=(5, 0))
    
    def update_status(self, text, color="gray"):
        """Update status indicator and label."""
        self.status_label.config(text=text)
        self.status_indicator.delete("all")
        bg = self.status_indicator.cget("bg")
        self.status_indicator.create_oval(2, 2, 14, 14, fill=color, outline="black", width=1)
        self.status_indicator.config(bg=bg)
    
    def log(self, message):
        """Update log line (single line only for lightweight)."""
        self.log_label.config(text=message[:80] + "..." if len(message) > 80 else message)
    
    def check_port(self, port=5000, timeout=0.3):
        """Fast socket check if port is open (something is listening)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex(("localhost", port))
            try:
                sock.close()
            except Exception:
                pass
            return result == 0
        except Exception:
            return False
    
    def _port_closed(self, port=5000, retries=5, delay=0.5):
        """Return True if port has no listener. Retries to allow OS to release port."""
        for _ in range(retries):
            if not self.check_port(port, timeout=0.5):
                return True
            time.sleep(delay)
        return False
    
    def check_status(self):
        """Check server status (non-blocking, efficient)."""
        if self._checking:
            self.root.after(3000, self.check_status)
            return
        
        self._checking = True
        
        def check():
            port_open = self.check_port()
            
            # Check managed process (local ref to avoid NoneType if cleared by another thread)
            proc = self.server_process
            process_running = (proc is not None and proc.poll() is None)
            
            if port_open or process_running:
                if not self.is_running:
                    self.is_running = True
                    self.root.after(0, self._update_ui_running)
            else:
                if self.is_running:
                    self.is_running = False
                    self.root.after(0, self._update_ui_stopped)
            
            self._checking = False
            self.root.after(3000, self.check_status)
        
        threading.Thread(target=check, daemon=True).start()
    
    def _update_ui_running(self):
        """Update UI when server is running."""
        self.update_status("Running", "green")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.restart_btn.config(state=tk.NORMAL)
        self.browser_btn.config(state=tk.NORMAL)
    
    def _update_ui_stopped(self):
        """Update UI when server is stopped."""
        self.update_status("Stopped", "red")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)
        self.browser_btn.config(state=tk.DISABLED)
    
    def start_server(self):
        """Start the server."""
        if self.server_process is not None:
            messagebox.showinfo("Info", "Server is already running.")
            return
        
        app_file = self.script_dir / "app.py"
        if not app_file.exists():
            messagebox.showerror("Error", f"Cannot find app.py in:\n{self.script_dir}")
            return
        
        self.log("Starting server...")
        self.update_status("Starting...", "yellow")
        self.start_btn.config(state=tk.DISABLED)
        
        def run():
            try:
                os.chdir(self.script_dir)
                proc = subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                self.server_process = proc
                self.root.after(0, lambda: self.log("Server starting..."))
                
                # Read output (minimal, just to keep pipe from blocking); use proc so stop() can set server_process=None
                for line in iter(proc.stdout.readline, ''):
                    if proc.poll() is not None:
                        break
                
                if proc.poll() is not None:
                    self.root.after(0, lambda: self.log(f"Server stopped (code {proc.returncode})"))
                self.server_process = None
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Error: {str(e)}"))
                self.server_process = None
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _clear_port_5000(self):
        """Kill all processes using port 5000 (any state: LISTENING, ESTABLISHED, etc.)."""
        flags = subprocess.CREATE_NO_WINDOW if (sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW")) else 0
        my_pid = os.getpid()
        pids_to_kill = set()
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=flags,
                )
                # Match any line with :5000 (LISTENING, ESTABLISHED, TIME_WAIT, etc.)
                for line in (result.stdout or "").split("\n"):
                    if ":5000" not in line:
                        continue
                    parts = line.strip().split()
                    # PID is last column; ensure it's numeric
                    if len(parts) >= 5 and parts[-1].isdigit():
                        pid = int(parts[-1])
                        if pid != my_pid:
                            pids_to_kill.add(pid)
                for pid in pids_to_kill:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True,
                            timeout=3,
                            creationflags=flags,
                        )
                    except Exception:
                        pass
            else:
                result = subprocess.run(
                    ["lsof", "-ti:5000"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0 and result.stdout.strip():
                    for pid in result.stdout.strip().split("\n"):
                        pid = pid.strip()
                        if pid.isdigit() and int(pid) != my_pid:
                            try:
                                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
                            except Exception:
                                pass
        except Exception:
            pass
    
    def _run_stop_script(self):
        """Run stop_server.bat on Windows (then port is always cleared via _clear_port_5000)."""
        if sys.platform == "win32":
            stop_bat = self.script_dir / "stop_server.bat"
            if stop_bat.exists():
                try:
                    subprocess.run(
                        [str(stop_bat)],
                        cwd=str(self.script_dir),
                        shell=True,
                        timeout=15,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                    )
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"Stop script: {str(e)}"))
        self.server_process = None
    
    def stop_server(self):
        """Stop the server and clear port 5000 (script + explicit clear)."""
        self.log("Stopping server...")
        self.update_status("Stopping...", "yellow")
        self.stop_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)
        
        def stop():
            # 1) Terminate our managed process if any (local ref to avoid race with run() thread)
            proc = self.server_process
            self.server_process = None
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        if proc.poll() is None:
                            proc.kill()
                    except Exception:
                        pass
            # 2) Run stop script (Windows)
            self._run_stop_script()
            # 3) Clear port 5000 (run twice to catch slow-to-exit processes)
            self._clear_port_5000()
            time.sleep(1.0)
            self._clear_port_5000()
            time.sleep(1.2)
            # 4) Verify with retries (OS may need a moment to release the port)
            if self._port_closed(5000, retries=5, delay=0.5):
                self.root.after(0, lambda: self.log("Server stopped"))
                self.root.after(0, self._update_ui_stopped)
            else:
                self.root.after(0, lambda: self.log("Port 5000 may still be in use"))
                self.root.after(0, self._update_ui_stopped)
        
        threading.Thread(target=stop, daemon=True).start()
    
    def restart_server(self):
        """Restart the server using restart_server script (or stop + start)."""
        if sys.platform == "win32":
            restart_bat = self.script_dir / "restart_server.bat"
            if restart_bat.exists():
                self.log("Running restart script...")
                self.update_status("Restarting...", "yellow")
                self.stop_btn.config(state=tk.DISABLED)
                self.restart_btn.config(state=tk.DISABLED)
                self.server_process = None
                # Run script in background (batch has 'pause' so we don't wait); server starts in new window
                try:
                    subprocess.Popen(
                        [str(restart_bat)],
                        cwd=str(self.script_dir),
                        shell=True,
                    )
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"Restart script error: {str(e)}"))
                    self.root.after(0, self._update_ui_stopped)
                    return
                self.root.after(0, lambda: self.log("Restart script started (check console)"))
                return
        
        # Non-Windows or script missing: stop (clear port) then start
        if not self.is_running:
            messagebox.showinfo("Info", "Server is not running.")
            return
        
        def restart():
            proc = self.server_process
            self.server_process = None
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            self._clear_port_5000()
            time.sleep(1.2)
            self.root.after(0, self.start_server)
        
        threading.Thread(target=restart, daemon=True).start()
    
    def open_browser(self):
        """Open browser."""
        if not self.is_running:
            messagebox.showwarning("Warning", "Server is not running.")
            return
        
        import webbrowser
        try:
            webbrowser.open("http://localhost:5000")
            self.log("Opening browser...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open browser: {str(e)}")
    
    def on_closing(self):
        """Handle window close."""
        proc = self.server_process
        if proc is not None:
            self.stop_server()
            time.sleep(0.5)
        self.root.destroy()


def main():
    """Main entry point."""
    try:
        root = tk.Tk()
        app = LightweightLauncher(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
