"""
=============================================================================
BUSINESS MEETING COPILOT — APPLICATION ENTRY POINT (app.py)
=============================================================================

WHAT THIS FILE DOES (in plain language):
----------------------------------------
This is the "front door" of the server. When you run "python app.py", the computer
starts a web server that other parts of the project use. The server:

  1. Sends the main webpage (index.html) to your browser when you open the app.
  2. Listens for requests from the browser (e.g. "start engagement detection",
     "send this text to the AI", "give me the current engagement score").
  3. Talks to Azure (Microsoft's cloud) for AI chat and speech, and runs our
     own code for face/engagement analysis.

Think of it like a receptionist: the browser is the visitor, and this server
is the building that routes each request to the right room (chat, config,
engagement detection, etc.). The actual "rooms" are defined in routes.py.

HOW TO RUN:
-----------
  - From project root:  python app.py
  - By default the app is at:  http://localhost:5000
  - You can also use the launcher:  scripts\\start.bat  (Windows) or  ./scripts/start.sh  (Mac/Linux)

CONFIGURATION:
--------------
  - Settings (API keys, ports, etc.) come from the .env file and config.py.
  - Never put real API keys in the code; use environment variables.
  - See README.md (project root) for full setup.
=============================================================================
"""

# ---------------------------------------------------------------------------
# Step 1: Load environment variables from .env (before anything else)
# ---------------------------------------------------------------------------
# The .env file holds secrets and settings (e.g. API keys). We load it from
# the same folder as this file so that config.py can read those values.
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Step 2: Import the web framework and our own modules
# ---------------------------------------------------------------------------
from flask import Flask
from flask_cors import CORS
from flask_compress import Compress

from routes import register_routes
import config

# ---------------------------------------------------------------------------
# Step 3: Warn the user if important settings are missing
# ---------------------------------------------------------------------------
# This prints a friendly warning (e.g. "SPEECH_KEY is not set") so you know
# what to add to .env. It does NOT put any secrets in the code.
config.warn_missing_config()


def _check_speech_token_once():
    """
    Try to get a Speech token once at startup; if it fails with 401 (bad key),
    print clear instructions so the user knows how to fix it.

    Why: A 401 error from Azure is confusing without context. This gives the
    user a short, actionable message (e.g. "Use a key from an Azure Speech
    resource" and where to set SPEECH_REGION).
    """
    if not (config.SPEECH_KEY and config.SPEECH_KEY.strip()):
        return
    try:
        from services import get_speech_service
        get_speech_service().get_speech_token()
    except ValueError as e:
        if "401" in str(e) or "Permission Denied" in str(e):
            import sys
            print(
                "\n  Speech token (401): SPEECH_KEY or SPEECH_REGION is invalid.\n"
                "  Use a key from an Azure *Speech* resource (Portal → Create 'Speech' or 'Cognitive Services').\n"
                "  Copy Key 1 from Keys and Endpoint; set SPEECH_REGION to that resource's region (e.g. eastus, centralindia).\n",
                file=sys.stderr,
            )
    except Exception:
        pass


_check_speech_token_once()


def create_app() -> Flask:
    """
    Create and configure the Flask application (the web server).

    What it does:
      - Creates a new Flask "app" object (the core of our server).
      - Enables CORS so the browser can call our API from different origins
        (e.g. from a different port or from the browser extension).
      - Enables compression so big responses (like engagement state) are
        sent in a smaller form when the client supports it.
      - Registers all our URL routes (pages, chat, config, engagement) by
        calling register_routes(app). After this, URLs like /chat, /engagement/state
        are wired up to the right Python functions.

    Returns:
        The configured Flask application. We use this single app object
        everywhere (e.g. to run the server at the bottom of this file).
    """
    app = Flask(__name__)

    # Allow the browser to call our API from another origin (e.g. extension or different port).
    # In production you would restrict this to specific domains.
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Compress responses (gzip) when the client supports it. Saves bandwidth
    # on endpoints that return a lot of data (e.g. /engagement/state).
    Compress(app)

    # Attach all our URL rules (/, /chat, /engagement/start, etc.) to this app.
    # The actual handlers live in routes.py.
    register_routes(app)

    return app


# ---------------------------------------------------------------------------
# Create the one global Flask application
# ---------------------------------------------------------------------------
# Every part of the server uses this same "app" object. It is created once
# when this file is loaded.
app = create_app()


# ---------------------------------------------------------------------------
# Run the server when this file is executed directly (e.g. "python app.py")
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    When you run "python app.py" from the command line, the block below runs.

    - If FLASK_DEBUG is true: use Flask's built-in development server (with
      auto-reload and debugger, but it prints a warning that it's not for production).
    - Otherwise: use Waitress, a production-style server that can handle multiple
      requests at once (we use 6 threads). No warning, suitable for local or
      light production use.

    Host and port come from config (default: 0.0.0.0:5000 so the server is
    reachable from other devices on your network, not just localhost).
    """
    if config.FLASK_DEBUG:
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=True
        )
    else:
        import waitress
        waitress.serve(app, host=config.FLASK_HOST, port=config.FLASK_PORT, threads=6)
