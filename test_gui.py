"""
Quick test script to verify GUI components work correctly.
Run this to test if tkinter and all GUI components are working.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

def test_gui():
    """Test basic GUI functionality."""
    root = tk.Tk()
    root.title("GUI Test")
    root.geometry("600x400")
    
    # Test frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title = ttk.Label(main_frame, text="GUI Component Test", font=("Arial", 14, "bold"))
    title.pack(pady=10)
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=10)
    
    btn1 = ttk.Button(button_frame, text="Button 1", width=20)
    btn1.pack(side=tk.LEFT, padx=5)
    
    btn2 = ttk.Button(button_frame, text="Button 2", width=20)
    btn2.pack(side=tk.LEFT, padx=5)
    
    btn3 = ttk.Button(button_frame, text="Button 3", width=20)
    btn3.pack(side=tk.LEFT, padx=5)
    
    # Text area
    log_frame = ttk.LabelFrame(main_frame, text="Test Log", padding="10")
    log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    text_area = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    text_area.pack(fill=tk.BOTH, expand=True)
    text_area.insert("1.0", "If you can see this text and all three buttons above, the GUI is working correctly!\n\n")
    text_area.config(state=tk.DISABLED)
    
    # Status
    status_label = ttk.Label(main_frame, text="Status: GUI Test Running", foreground="green")
    status_label.pack(pady=5)
    
    def show_message():
        messagebox.showinfo("Test", "Message box is working!")
    
    test_btn = ttk.Button(main_frame, text="Test Message Box", command=show_message)
    test_btn.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    print("Testing GUI components...")
    print("If the window opens and shows buttons, the GUI is working!")
    test_gui()
