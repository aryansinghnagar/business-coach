"""
Simple GUI Launcher for Business Meeting Copilot
Allows non-technical users to easily start and stop the application.

Usage:
    python gui_launcher.py
    
Or simply double-click the file to run it.
"""

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError as e:
    print(f"Error: Tkinter is not available. {e}")
    print("\nPlease install tkinter:")
    print("  Windows: Usually included with Python")
    print("  Linux: sudo apt-get install python3-tk")
    print("  Mac: Usually included with Python")
    sys.exit(1)

import subprocess
import threading
import sys
import os
import time
try:
    import requests
except ImportError:
    print("Warning: requests module not found. Some features may not work.")
    requests = None
from pathlib import Path

class ProjectLauncher:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title("Business Meeting Copilot - Launcher")
            self.root.geometry("750x600")
            self.root.resizable(True, True)
            
            # Set minimum window size
            self.root.minsize(600, 500)
            
            # Server process
            self.server_process = None
            self.is_running = False
            
            # Setup UI
            self.setup_ui()
            
            # Check initial status
            self.check_server_status()
            
            # Auto-check status every 2 seconds
            self.auto_check_status()
        except Exception as e:
            # If UI setup fails, show error
            error_msg = f"Error initializing GUI: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            try:
                messagebox.showerror("Initialization Error", error_msg)
            except:
                pass
            raise
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Log area should expand
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Business Meeting Copilot",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 15), sticky=tk.W+tk.E)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        status_frame.columnconfigure(1, weight=1)
        
        # Status label and indicator in a horizontal frame
        status_inner = ttk.Frame(status_frame)
        status_inner.grid(row=0, column=0, sticky=tk.W)
        
        self.status_label = ttk.Label(
            status_inner,
            text="Checking...",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status indicator
        try:
            bg_color = status_frame.cget("background")
        except:
            bg_color = "SystemButtonFace"  # Default Windows background
        
        self.status_indicator = tk.Canvas(
            status_inner,
            width=20,
            height=20,
            highlightthickness=0,
            bg=bg_color
        )
        self.status_indicator.pack(side=tk.LEFT)
        
        # Control buttons - use grid for better layout
        button_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        # Start button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Server",
            command=self.start_server,
            width=25
        )
        self.start_button.grid(row=0, column=0, padx=5, sticky=tk.W+tk.E)
        
        # Stop button
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop Server",
            command=self.stop_server,
            width=25,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=5, sticky=tk.W+tk.E)
        
        # Open Browser button
        self.browser_button = ttk.Button(
            button_frame,
            text="Open in Browser",
            command=self.open_browser,
            width=25
        )
        self.browser_button.grid(row=0, column=2, padx=5, sticky=tk.W+tk.E)
        
        # Log output area
        log_frame = ttk.LabelFrame(main_frame, text="Server Logs", padding="5")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#f5f5f5",
            fg="#000000"
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.config(state=tk.DISABLED)
        
        # Info section
        info_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        info_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        
        info_text = (
            "1. Click 'Start Server' to launch the application\n"
            "2. Wait for the server to start (status will turn green)\n"
            "3. Click 'Open in Browser' to access the application\n"
            "4. Click 'Stop Server' when you're done"
        )
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            font=("Arial", 9),
            foreground="gray",
            justify=tk.LEFT
        )
        info_label.pack(anchor=tk.W)
    
    def log(self, message):
        """Add a message to the log area."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def update_status(self, status, color="gray"):
        """Update the status indicator and label."""
        self.status_label.config(text=status)
        self.status_indicator.delete("all")
        # Get background color to match frame
        bg_color = self.status_indicator.cget("bg")
        self.status_indicator.create_oval(
            2, 2, 18, 18,
            fill=color,
            outline="black",
            width=2
        )
    
    def check_server_status(self):
        """Check if the server is running."""
        if requests is None:
            # If requests is not available, just check if process exists
            if self.server_process is not None and self.server_process.poll() is None:
                self.is_running = True
                self.update_status("Server is Running", "green")
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.browser_button.config(state=tk.NORMAL)
                return True
            else:
                self.is_running = False
                if self.server_process is None:
                    self.update_status("Server is Stopped", "red")
                else:
                    self.update_status("Server is Starting...", "yellow")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                if not self.is_running:
                    self.browser_button.config(state=tk.DISABLED)
                return False
        
        try:
            response = requests.get("http://localhost:5000", timeout=1)
            if response.status_code == 200:
                self.is_running = True
                self.update_status("Server is Running", "green")
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.browser_button.config(state=tk.NORMAL)
                return True
        except Exception:
            self.is_running = False
            if self.server_process is None:
                self.update_status("Server is Stopped", "red")
            else:
                self.update_status("Server is Starting...", "yellow")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if not self.is_running:
                self.browser_button.config(state=tk.DISABLED)
            return False
    
    def auto_check_status(self):
        """Automatically check server status periodically."""
        self.check_server_status()
        self.root.after(2000, self.auto_check_status)  # Check every 2 seconds
    
    def start_server(self):
        """Start the Flask server."""
        if self.server_process is not None:
            messagebox.showinfo("Info", "Server is already starting or running.")
            return
        
        # Check if app.py exists
        script_dir = Path(__file__).parent
        app_file = script_dir / "app.py"
        if not app_file.exists():
            messagebox.showerror("Error", f"Cannot find app.py in:\n{script_dir}\n\nPlease make sure you're running this from the project directory.")
            return
        
        self.log("Starting server...")
        self.update_status("Starting Server...", "yellow")
        self.start_button.config(state=tk.DISABLED)
        
        # Start server in a separate thread
        def run_server():
            try:
                # Change to project directory
                os.chdir(script_dir)
                
                # Start Flask server
                self.server_process = subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
                
                self.root.after(0, self.log, "Server process started. Waiting for initialization...")
                
                # Read output in real-time
                for line in iter(self.server_process.stdout.readline, ''):
                    if line:
                        self.root.after(0, self.log, line.strip())
                    
                    # Check if process ended
                    if self.server_process.poll() is not None:
                        break
                
                # Process ended
                if self.server_process.poll() is not None:
                    return_code = self.server_process.returncode
                    self.root.after(0, self.log, f"Server process ended with code {return_code}")
                    self.root.after(0, lambda: self.update_status("Server Stopped", "red"))
                    self.server_process = None
                    self.is_running = False
                    self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                    if return_code != 0:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Server stopped unexpectedly. Check the logs for details."))
                    
            except Exception as e:
                error_msg = f"Error starting server: {str(e)}"
                self.root.after(0, self.log, error_msg)
                self.root.after(0, lambda: self.update_status("Error", "red"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.server_process = None
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def stop_server(self):
        """Stop the Flask server."""
        self.log("Stopping server...")
        self.update_status("Stopping Server...", "yellow")
        self.stop_button.config(state=tk.DISABLED)
        
        # Stop our managed process
        if self.server_process is not None:
            try:
                # Terminate the process
                self.server_process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    self.server_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self.server_process.kill()
                    self.server_process.wait()
                
                self.log("Server process stopped.")
            except Exception as e:
                self.log(f"Error stopping server process: {str(e)}")
                # Try to kill it anyway
                try:
                    if self.server_process:
                        self.server_process.kill()
                except:
                    pass
            finally:
                self.server_process = None
        
        # Also try to kill any other Python processes on port 5000
        try:
            if sys.platform == "win32":
                # Windows: Find and kill processes using port 5000
                import subprocess as sp
                try:
                    # Find PID using port 5000
                    result = sp.run(
                        ['netstat', '-ano'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    for line in result.stdout.split('\n'):
                        if ':5000' in line and 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) > 4:
                                pid = parts[-1]
                                try:
                                    sp.run(['taskkill', '/F', '/PID', pid], 
                                          capture_output=True, timeout=1)
                                    self.log(f"Killed process on port 5000 (PID: {pid})")
                                except:
                                    pass
                except:
                    pass
        except Exception as e:
            self.log(f"Error clearing port 5000: {str(e)}")
        
        self.log("Server stopped successfully.")
        self.is_running = False
        self.update_status("Server is Stopped", "red")
        self.start_button.config(state=tk.NORMAL)
        self.browser_button.config(state=tk.DISABLED)
    
    def open_browser(self):
        """Open the application in the default web browser."""
        if not self.is_running:
            messagebox.showwarning("Warning", "Server is not running. Please start the server first.")
            return
        
        import webbrowser
        url = "http://localhost:5000"
        self.log(f"Opening browser: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open browser: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event."""
        if self.server_process is not None:
            self.log("Stopping server before exit...")
            self.stop_server()
            time.sleep(1)  # Give it a moment to stop
        self.root.destroy()


def main():
    """Main entry point."""
    try:
        # Test if tkinter is available
        root = tk.Tk()
        
        # Set up error handling
        def handle_exception(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions."""
            import traceback
            error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            print(f"Uncaught exception:\n{error_msg}")
            try:
                messagebox.showerror(
                    "Error",
                    f"An error occurred:\n\n{str(exc_value)}\n\nCheck console for details."
                )
            except:
                pass
        
        sys.excepthook = handle_exception
        
        # Create and run the application
        app = ProjectLauncher(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nTkinter may not be installed. Please install it:")
        print("  Windows: Usually included with Python")
        print("  Linux: sudo apt-get install python3-tk")
        print("  Mac: Usually included with Python")
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"Error starting GUI launcher: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
