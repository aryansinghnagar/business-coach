"""
Test script to diagnose GUI launcher issues.
Run this to check if all dependencies are available.
"""

import sys

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    # Test tkinter
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext, messagebox
        print("✓ tkinter: OK")
    except ImportError as e:
        print(f"✗ tkinter: FAILED - {e}")
        print("  Install: sudo apt-get install python3-tk (Linux)")
        return False
    
    # Test other modules
    modules = [
        ("subprocess", None),
        ("threading", None),
        ("os", None),
        ("time", None),
        ("pathlib", "Path"),
        ("requests", None),
    ]
    
    all_ok = True
    for module_name, attr in modules:
        try:
            mod = __import__(module_name)
            if attr:
                getattr(mod, attr)
            print(f"✓ {module_name}: OK")
        except ImportError as e:
            print(f"✗ {module_name}: FAILED - {e}")
            all_ok = False
        except AttributeError as e:
            print(f"✗ {module_name}.{attr}: FAILED - {e}")
            all_ok = False
    
    return all_ok

def test_tkinter_basic():
    """Test basic tkinter functionality."""
    print("\nTesting tkinter basic functionality...")
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide window
        root.destroy()
        print("✓ tkinter basic: OK")
        return True
    except Exception as e:
        print(f"✗ tkinter basic: FAILED - {e}")
        return False

def test_gui_launcher_import():
    """Test if gui_launcher can be imported."""
    print("\nTesting gui_launcher import...")
    try:
        import gui_launcher
        print("✓ gui_launcher import: OK")
        return True
    except SyntaxError as e:
        print(f"✗ gui_launcher syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ gui_launcher import: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("GUI Launcher Diagnostic Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Run tests
    imports_ok = test_imports()
    tkinter_ok = test_tkinter_basic()
    launcher_ok = test_gui_launcher_import()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Tkinter: {'PASS' if tkinter_ok else 'FAIL'}")
    print(f"Launcher: {'PASS' if launcher_ok else 'FAIL'}")
    
    if imports_ok and tkinter_ok and launcher_ok:
        print("\n✓ All tests passed! GUI launcher should work.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
