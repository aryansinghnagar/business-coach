"""
Business Meeting Copilot — Flask application entry point.

AI meeting coach with Azure OpenAI chat, Speech (STT/TTS), talking avatar,
and real-time engagement detection (video + optional partner audio). Run with
`python app.py`; default http://localhost:5000. Config: config.py or env.
See docs/DOCUMENTATION.md for full documentation.
"""

from flask import Flask
from flask_cors import CORS
from routes import api
import config


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
    
    # Register all API routes
    app.register_blueprint(api)
    
    return app


# Create the Flask application
app = create_app()


if __name__ == "__main__":
    """
    Run the Flask development server.
    
    Configuration is loaded from config.py, with environment variables
    taking precedence over default values.
    """
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
