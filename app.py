"""
Business Meeting Copilot — Flask application entry point.

AI meeting coach with Azure AI Foundry chat, Speech (STT for transcript and speech cue analysis),
and real-time engagement detection (video + optional partner audio). Insights are text-only (no avatar/TTS). Run with
`python app.py`; default http://localhost:5000. Config: .env or config.py / environment.
See docs/DOCUMENTATION.md for full project documentation.
"""

# Load .env from the same directory as this file so env vars are available to config.py
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask
from flask_cors import CORS
from flask_compress import Compress
from routes import register_routes
import config

# Warn at startup when required env vars (e.g. API keys) are missing; no secrets in code.
config.warn_missing_config()


def _check_speech_token_once():
    """If SPEECH_KEY is set, try to get a token and warn on 401 so user sees clear instructions."""
    if not (config.SPEECH_KEY and config.SPEECH_KEY.strip()):
        return
    try:
        from services.azure_speech import get_speech_service
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
    Create and configure the Flask application.

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)

    # Enable CORS for all routes and origins
    # In production, restrict this to specific origins
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Compress JSON and other responses when client supports gzip (reduces bandwidth for /engagement/state etc.)
    Compress(app)

    # Register route blueprints (pages, chat, config, engagement)
    register_routes(app)

    return app


# Create the Flask application
app = create_app()


if __name__ == "__main__":
    """
    Run the Flask application.
    Uses Waitress (production WSGI) by default to avoid the dev-server warning;
    set FLASK_DEBUG=true for development (auto-reload, debugger, but shows warning).
    Configuration is loaded from config.py, with environment variables
    taking precedence over default values.
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
