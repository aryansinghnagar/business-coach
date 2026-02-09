#!/usr/bin/env python3
"""
Unified Control Panel for Business Meeting Copilot

Single entry point for:
  • Start / Stop / Restart server
  • Open app in browser
  • Run tests with detailed results and troubleshooting hints

Usage:
    python control_panel.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
except ImportError:
    print("Error: Tkinter not available. Please install python3-tk")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths and theme
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

THEME = {
    "bg": "#f5f5f7",
    "card_bg": "#ffffff",
    "text": "#1d1d1f",
    "text_muted": "#6e6e73",
    "accent": "#0071e3",
    "accent_hover": "#0077ed",
    "success": "#34c759",
    "warning": "#ff9500",
    "danger": "#ff3b30",
    "border": "#d2d2d7",
}


# ---------------------------------------------------------------------------
# Test runner (structured results for GUI)
# ---------------------------------------------------------------------------


def _suggest_reasons(test_id: str, error_text: str, is_error: bool) -> list[str]:
    """Analyze error and return suggested possible reasons."""
    suggestions: list[str] = []
    err_lower = (error_text or "").lower()

    if "attributeerror" in err_lower:
        suggestions.append("A method or property may have been renamed or removed.")
        suggestions.append("Check if the class/object API changed recently.")
    if "importerror" in err_lower or "modulenotfounderror" in err_lower:
        suggestions.append("A required package may not be installed. Run: pip install -r requirements.txt")
        suggestions.append("Check for circular imports or wrong import paths.")
    if "assertionerror" in err_lower:
        suggestions.append("The test expected different behavior. The code may have changed.")
        suggestions.append("Verify the API response format matches what the test expects.")
    if "keyerror" in err_lower:
        suggestions.append("A dictionary key may be missing. Check the API response structure.")
    if "typeerror" in err_lower:
        suggestions.append("Wrong type passed or returned (e.g. None instead of a value).")
    if "connection" in err_lower or "timeout" in err_lower:
        suggestions.append("External service (Azure, Foundry) may be unreachable.")
        suggestions.append("Tests that call real APIs should be mocked.")
    if "openai" in err_lower or "azure" in err_lower:
        suggestions.append("Azure/Foundry config may be missing or invalid.")
        suggestions.append("Ensure API keys are set in config or environment.")
    if "face" in err_lower or "mediapipe" in err_lower or "cv2" in err_lower:
        suggestions.append("Face detection dependencies (MediaPipe, OpenCV) may not be installed.")
        suggestions.append("Run: pip install mediapipe opencv-python-headless")
    if "config" in err_lower:
        suggestions.append("A config value may be missing or invalid in config.py.")
    if "routes" in err_lower or "blueprint" in err_lower:
        suggestions.append("Flask route or request handling may have changed.")
    if "json" in err_lower:
        suggestions.append("Request/response JSON structure may have changed.")
    if "404" in err_lower or "500" in err_lower:
        suggestions.append("Endpoint may have moved or the route is not registered correctly.")

    if not suggestions:
        suggestions.append("Review the full error message and traceback for details.")
        if is_error:
            suggestions.append("Errors often indicate import issues or missing dependencies.")

    return suggestions


def run_tests_with_results(pattern: str | None = None) -> tuple[list[str], list, list]:
    """
    Run tests and return (passed_ids, failed_items, error_items).
    Each failed/error item: (test_id, short_desc, full_traceback, suggestions)
    """
    import unittest

    loader = unittest.TestLoader()
    start_dir = str(PROJECT_ROOT / "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")

    if pattern:
        def gather(s, acc):
            for t in s:
                if isinstance(t, unittest.TestSuite):
                    gather(t, acc)
                else:
                    acc.append(t)
        all_tests: list = []
        gather(suite, all_tests)
        filtered = unittest.TestSuite()
        pat = pattern.lower()
        for t in all_tests:
            if pat in str(t).lower():
                filtered.addTest(t)
        if list(filtered):
            suite = filtered

    result = unittest.TestResult()
    suite.run(result)

    def collect_ids(s, acc):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                collect_ids(t, acc)
            else:
                acc.append(str(t))

    all_ids: list[str] = []
    collect_ids(suite, all_ids)
    failed_ids = {str(t[0]) for t in result.failures + result.errors}

    passed = [tid for tid in all_ids if tid not in failed_ids]
    failed = []
    errors = []

    for test, tb in result.failures:
        tid = str(test)
        short = tb.split("\n")[-1].strip() if tb else "Assertion failed"
        failed.append((tid, short, tb or "", _suggest_reasons(tid, tb, False)))

    for test, tb in result.errors:
        tid = str(test)
        short = tb.split("\n")[-1].strip() if tb else "Error"
        errors.append((tid, short, tb or "", _suggest_reasons(tid, tb, True)))

    return passed, failed, errors


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def check_port(port: int = 5000, timeout: float = 0.3) -> bool:
    """Return True if something is listening on the port."""
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


def _port_closed(port: int = 5000, retries: int = 5, delay: float = 0.5) -> bool:
    for _ in range(retries):
        if not check_port(port, timeout=0.5):
            return True
        time.sleep(delay)
    return False


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------------------
# Control Panel UI
# ---------------------------------------------------------------------------


class ControlPanel:
    """Unified control panel with Server and Tests tabs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Business Meeting Copilot — Control Panel")
        self.root.geometry("720x560")
        self.root.minsize(600, 480)
        self.root.resizable(True, True)
        self.root.configure(bg=THEME["bg"])

        self.script_dir = PROJECT_ROOT
        self.server_process: subprocess.Popen | None = None
        self.is_running = False
        self._checking = False

        self._build_ui()
        self.check_status()

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg=THEME["bg"], padx=20, pady=16)
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(main, bg=THEME["bg"])
        header.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            header,
            text="Business Meeting Copilot",
            font=("Segoe UI", 18, "bold"),
            fg=THEME["text"],
            bg=THEME["bg"],
        ).pack(anchor=tk.W)

        tk.Label(
            header,
            text="Control server and run tests from one place",
            font=("Segoe UI", 10),
            fg=THEME["text_muted"],
            bg=THEME["bg"],
        ).pack(anchor=tk.W)

        # Notebook (tabs)
        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Server tab ---
        server_frame = tk.Frame(notebook, bg=THEME["bg"], padx=12, pady=12)
        notebook.add(server_frame, text="  Server  ")

        card = tk.Frame(server_frame, bg=THEME["card_bg"], padx=20, pady=20)
        card.pack(fill=tk.X, pady=(0, 12))
        card.configure(highlightbackground=THEME["border"], highlightthickness=1)

        status_row = tk.Frame(card, bg=THEME["card_bg"])
        status_row.pack(fill=tk.X, pady=(0, 14))

        self.status_indicator = tk.Canvas(
            status_row, width=12, height=12, highlightthickness=0, bg=THEME["card_bg"]
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = tk.Label(
            status_row,
            text="Checking...",
            font=("Segoe UI", 12),
            fg=THEME["text"],
            bg=THEME["card_bg"],
        )
        self.status_label.pack(side=tk.LEFT)

        btn_row = tk.Frame(card, bg=THEME["card_bg"])
        btn_row.pack(fill=tk.X)

        self.start_btn = self._btn(
            btn_row, "Start server", self._start_server,
            primary=True,
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = self._btn(btn_row, "Stop", self._stop_server, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.restart_btn = self._btn(btn_row, "Restart", self._restart_server, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.browser_btn = self._btn(
            btn_row, "Open in browser", self._open_browser,
            link_style=True, state=tk.DISABLED,
        )
        self.browser_btn.pack(side=tk.LEFT)

        log_frame = tk.Frame(server_frame, bg=THEME["card_bg"], padx=12, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.configure(highlightbackground=THEME["border"], highlightthickness=1)

        self.log_label = tk.Label(
            log_frame,
            text="",
            font=("Segoe UI", 9),
            fg=THEME["text_muted"],
            bg=THEME["card_bg"],
            wraplength=600,
            justify=tk.LEFT,
        )
        self.log_label.pack(anchor=tk.W)

        # --- Tests tab ---
        tests_frame = tk.Frame(notebook, bg=THEME["bg"], padx=12, pady=12)
        notebook.add(tests_frame, text="  Tests  ")

        toolbar = tk.Frame(tests_frame, bg=THEME["bg"])
        toolbar.pack(fill=tk.X, pady=(0, 10))

        run_btn = self._btn(toolbar, "Run tests", lambda: None, primary=True)
        run_btn.pack(side=tk.LEFT, padx=(0, 12))

        filter_var = tk.StringVar()
        filter_entry = tk.Entry(
            toolbar,
            textvariable=filter_var,
            font=("Segoe UI", 10),
            width=22,
            relief=tk.FLAT,
            bg=THEME["card_bg"],
            fg=THEME["text"],
            highlightthickness=1,
            highlightbackground=THEME["border"],
        )
        filter_entry.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            toolbar,
            text="Filter by name (e.g. api)",
            font=("Segoe UI", 9),
            fg=THEME["text_muted"],
            bg=THEME["bg"],
        ).pack(side=tk.LEFT)

        summary_var = tk.StringVar(value="Click 'Run tests' to start.")
        tk.Label(
            toolbar,
            textvariable=summary_var,
            font=("Segoe UI", 10),
            fg=THEME["text"],
            bg=THEME["bg"],
        ).pack(side=tk.RIGHT)

        content = tk.Frame(tests_frame, bg=THEME["bg"])
        content.pack(fill=tk.BOTH, expand=True)

        left_card = tk.Frame(content, bg=THEME["card_bg"], padx=12, pady=12)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        left_card.configure(highlightbackground=THEME["border"], highlightthickness=1)

        tk.Label(
            left_card,
            text="Tests",
            font=("Segoe UI", 10, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["card_bg"],
        ).pack(anchor=tk.W, pady=(0, 8))

        list_frame = tk.Frame(left_card, bg=THEME["card_bg"])
        list_frame.pack(fill=tk.BOTH, expand=True)

        scroll_list = tk.Scrollbar(list_frame)
        scroll_list.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(
            list_frame,
            font=("Segoe UI", 10),
            bg=THEME["card_bg"],
            fg=THEME["text"],
            selectbackground="#e8f4fd",
            selectforeground=THEME["text"],
            activestyle="none",
            yscrollcommand=scroll_list.set,
            highlightthickness=0,
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_list.config(command=listbox.yview)

        right_card = tk.Frame(content, bg=THEME["card_bg"], padx=12, pady=12)
        right_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        right_card.configure(highlightbackground=THEME["border"], highlightthickness=1)

        tk.Label(
            right_card,
            text="Details",
            font=("Segoe UI", 10, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["card_bg"],
        ).pack(anchor=tk.W, pady=(0, 8))

        detail_text = scrolledtext.ScrolledText(
            right_card,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            bg=THEME["card_bg"],
            fg=THEME["text"],
            relief=tk.FLAT,
        )
        detail_text.pack(fill=tk.BOTH, expand=True)
        detail_text.insert(tk.END, "Click 'Run tests' to run the suite.", "hint")
        detail_text.tag_config("hint", foreground=THEME["text_muted"], font=("Segoe UI", 10))
        detail_text.config(state=tk.DISABLED)

        results_ref: dict = {"passed": [], "failed": [], "errors": []}

        def update_results(passed: list, failed: list, errors: list) -> None:
            results_ref["passed"] = passed
            results_ref["failed"] = failed
            results_ref["errors"] = errors
            listbox.delete(0, tk.END)
            for tid in passed:
                listbox.insert(tk.END, "  ✓  " + tid)
            for tid, *_ in failed:
                listbox.insert(tk.END, "  ✗  " + tid)
            for tid, *_ in errors:
                listbox.insert(tk.END, "  ✗  " + tid + " (error)")

            total = len(passed) + len(failed) + len(errors)
            n_fail = len(failed) + len(errors)

            if total == 0:
                summary_var.set("No tests matched. Try a different filter.")
            elif n_fail == 0:
                summary_var.set(f"All {total} tests passed.")
            else:
                summary_var.set(f"{len(passed)} passed, {n_fail} failed of {total}. Click a failed test for details.")

            detail_text.config(state=tk.NORMAL)
            detail_text.delete("1.0", tk.END)
            if n_fail == 0:
                detail_text.insert(tk.END, "All tests passed. The app is working correctly.", "success")
                detail_text.tag_config("success", foreground=THEME["success"], font=("Segoe UI", 11))
            else:
                detail_text.insert(
                    tk.END,
                    "Select a failed test in the list to see the error and possible reasons.",
                    "hint",
                )
                detail_text.tag_config("hint", foreground=THEME["text_muted"], font=("Segoe UI", 10))
            detail_text.config(state=tk.DISABLED)

        def run_tests() -> None:
            pattern = filter_var.get().strip() or None
            run_btn.config(state=tk.DISABLED)
            listbox.delete(0, tk.END)
            detail_text.config(state=tk.NORMAL)
            detail_text.delete("1.0", tk.END)
            summary_var.set("Running tests...")
            detail_text.insert(tk.END, "Running tests...", "hint")
            detail_text.tag_config("hint", foreground=THEME["text_muted"], font=("Segoe UI", 10))
            detail_text.config(state=tk.DISABLED)
            self.root.update_idletasks()

            def do() -> None:
                passed, failed, errors = run_tests_with_results(pattern)
                self.root.after(0, lambda: (update_results(passed, failed, errors), run_btn.config(state=tk.NORMAL)))

            threading.Thread(target=do, daemon=True).start()

        def on_select(evt: tk.Event) -> None:
            sel = evt.widget.curselection()
            if not sel:
                return
            idx = int(sel[0])
            passed = results_ref["passed"]
            failed = results_ref["failed"]
            errors = results_ref["errors"]
            n_pass = len(passed)
            if idx < n_pass:
                detail_text.config(state=tk.NORMAL)
                detail_text.delete("1.0", tk.END)
                detail_text.insert(tk.END, "This test passed.", "success")
                detail_text.tag_config("success", foreground=THEME["success"], font=("Segoe UI", 11))
                detail_text.config(state=tk.DISABLED)
                return
            if idx < n_pass + len(failed):
                item = failed[idx - n_pass]
            else:
                item = errors[idx - n_pass - len(failed)]

            tid, short, full_tb, suggestions = item
            detail_text.config(state=tk.NORMAL)
            detail_text.delete("1.0", tk.END)
            detail_text.insert(tk.END, "Test\n", "head")
            detail_text.insert(tk.END, tid + "\n\n", "normal")
            detail_text.insert(tk.END, "Error\n", "head")
            detail_text.insert(tk.END, short + "\n\n", "normal")
            detail_text.insert(tk.END, "Possible reasons\n", "head")
            for s in suggestions:
                detail_text.insert(tk.END, "  • " + s + "\n", "bullet")
            detail_text.insert(tk.END, "\nFull traceback\n", "head")
            detail_text.insert(tk.END, full_tb or "(none)", "trace")
            detail_text.tag_config("head", foreground=THEME["accent"], font=("Segoe UI", 10, "bold"))
            detail_text.tag_config("bullet", foreground=THEME["success"])
            detail_text.tag_config("trace", foreground=THEME["text_muted"], font=("Consolas", 9))
            detail_text.config(state=tk.DISABLED)

        run_btn.config(command=run_tests)
        listbox.bind("<<ListboxSelect>>", on_select)

        tests_run_once = [False]

        def on_tab_change(evt: tk.Event) -> None:
            if notebook.index(notebook.select()) == 1 and not tests_run_once[0]:
                tests_run_once[0] = True
                run_tests()

        notebook.bind("<<NotebookTabChanged>>", on_tab_change)

    def _btn(
        self,
        parent: tk.Widget,
        text: str,
        command,
        *,
        primary: bool = False,
        link_style: bool = False,
        state: str = tk.NORMAL,
    ) -> tk.Button:
        if primary:
            return tk.Button(
                parent,
                text=text,
                command=command,
                font=("Segoe UI", 10, "bold"),
                fg="white",
                bg=THEME["accent"],
                activeforeground="white",
                activebackground=THEME["accent_hover"],
                relief=tk.FLAT,
                padx=20,
                pady=10,
                cursor="hand2",
                bd=0,
                state=state,
            )
        if link_style:
            return tk.Button(
                parent,
                text=text,
                command=command,
                font=("Segoe UI", 10),
                fg=THEME["accent"],
                bg=THEME["card_bg"],
                activeforeground=THEME["accent_hover"],
                activebackground=THEME["bg"],
                relief=tk.FLAT,
                padx=16,
                pady=10,
                cursor="hand2",
                bd=0,
                state=state,
            )
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10),
            fg=THEME["text"],
            bg=THEME["border"],
            activeforeground=THEME["text"],
            activebackground="#c7c7cc",
            relief=tk.FLAT,
            padx=16,
            pady=10,
            cursor="hand2",
            bd=0,
            state=state,
        )

    def _update_status(self, text: str, color: str | None = None) -> None:
        self.status_label.config(text=text)
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color or THEME["text_muted"], outline="")
        self.status_indicator.config(bg=THEME["card_bg"])

    def _log(self, msg: str) -> None:
        short = msg[:120] + "..." if len(msg) > 120 else msg
        self.log_label.config(text=short)

    def check_status(self) -> None:
        if self._checking:
            self.root.after(3000, self.check_status)
            return
        self._checking = True

        def check() -> None:
            port_open = check_port()
            proc = self.server_process
            running = proc is not None and proc.poll() is None
            if port_open or running:
                if not self.is_running:
                    self.is_running = True
                    self.root.after(0, self._ui_running)
            else:
                if self.is_running:
                    self.is_running = False
                    self.root.after(0, self._ui_stopped)
            self._checking = False
            self.root.after(3000, self.check_status)

        threading.Thread(target=check, daemon=True).start()

    def _ui_running(self) -> None:
        self._update_status("Server is running", THEME["success"])
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.restart_btn.config(state=tk.NORMAL)
        self.browser_btn.config(state=tk.NORMAL)

    def _ui_stopped(self) -> None:
        self._update_status("Server is stopped", THEME["danger"])
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)
        self.browser_btn.config(state=tk.DISABLED)

    def _start_server(self) -> None:
        if self.server_process is not None:
            messagebox.showinfo("Info", "The server is already running.")
            return
        app_file = self.script_dir / "app.py"
        if not app_file.exists():
            messagebox.showerror("Error", f"Cannot find app.py in:\n{self.script_dir}")
            return

        self._log("Starting server...")
        self._update_status("Starting...", THEME["warning"])
        self.start_btn.config(state=tk.DISABLED)

        def run() -> None:
            try:
                os.chdir(self.script_dir)
                proc = subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=CREATE_NO_WINDOW,
                )
                self.server_process = proc
                self.root.after(0, lambda: self._log("Server is starting..."))
                for _ in iter(proc.stdout.readline, ""):
                    if proc.poll() is not None:
                        break
                if proc.poll() is not None:
                    self.root.after(0, lambda: self._log(f"Server stopped (code {proc.returncode})"))
                self.server_process = None
            except Exception as e:
                self.root.after(0, lambda: self._log(f"Error: {str(e)}"))
                self.server_process = None
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

        threading.Thread(target=run, daemon=True).start()

    def _clear_port(self) -> None:
        my_pid = os.getpid()
        pids = set()
        try:
            if sys.platform == "win32":
                r = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=CREATE_NO_WINDOW,
                )
                for line in (r.stdout or "").split("\n"):
                    if ":5000" not in line:
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[-1].isdigit():
                        pid = int(parts[-1])
                        if pid != my_pid:
                            pids.add(pid)
                for pid in pids:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True,
                            timeout=3,
                            creationflags=CREATE_NO_WINDOW,
                        )
                    except Exception:
                        pass
            else:
                r = subprocess.run(["lsof", "-ti:5000"], capture_output=True, text=True, timeout=3)
                if r.returncode == 0 and r.stdout.strip():
                    for pid in r.stdout.strip().split("\n"):
                        pid = pid.strip()
                        if pid.isdigit() and int(pid) != my_pid:
                            try:
                                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
                            except Exception:
                                pass
        except Exception:
            pass

    def _run_stop_script(self) -> None:
        if sys.platform == "win32":
            bat = self.script_dir / "stop_server.bat"
            if bat.exists():
                try:
                    subprocess.run(
                        [str(bat)],
                        cwd=str(self.script_dir),
                        shell=True,
                        timeout=15,
                        creationflags=CREATE_NO_WINDOW,
                    )
                except Exception as e:
                    self.root.after(0, lambda: self._log(f"Stop script: {str(e)}"))
        self.server_process = None

    def _stop_server(self) -> None:
        self._log("Stopping server...")
        self._update_status("Stopping...", THEME["warning"])
        self.stop_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)

        def stop() -> None:
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
            self._run_stop_script()
            self._clear_port()
            time.sleep(1.0)
            self._clear_port()
            time.sleep(1.2)
            if _port_closed(5000, retries=5, delay=0.5):
                self.root.after(0, lambda: self._log("Server stopped."))
            else:
                self.root.after(0, lambda: self._log("Port 5000 may still be in use."))
            self.root.after(0, self._ui_stopped)

        threading.Thread(target=stop, daemon=True).start()

    def _restart_server(self) -> None:
        if sys.platform == "win32":
            bat = self.script_dir / "restart_server.bat"
            if bat.exists():
                self._log("Running restart script...")
                self._update_status("Restarting...", THEME["warning"])
                self.stop_btn.config(state=tk.DISABLED)
                self.restart_btn.config(state=tk.DISABLED)
                self.server_process = None
                try:
                    subprocess.Popen([str(bat)], cwd=str(self.script_dir), shell=True)
                except Exception as e:
                    self.root.after(0, lambda: self._log(f"Restart error: {str(e)}"))
                    self.root.after(0, self._ui_stopped)
                    return
                self.root.after(0, lambda: self._log("Restart script started. Check the console window."))
                return

        if not self.is_running:
            messagebox.showinfo("Info", "The server is not running.")
            return

        def restart() -> None:
            proc = self.server_process
            self.server_process = None
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            self._clear_port()
            time.sleep(1.2)
            self.root.after(0, self._start_server)

        threading.Thread(target=restart, daemon=True).start()

    def _open_browser(self) -> None:
        if not self.is_running:
            messagebox.showwarning("Warning", "The server is not running. Start it first.")
            return
        import webbrowser
        try:
            webbrowser.open("http://localhost:5000")
            self._log("Opening browser...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open browser: {e}")

    def on_closing(self) -> None:
        if self.server_process is not None:
            self._stop_server()
            time.sleep(0.5)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = ControlPanel(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
