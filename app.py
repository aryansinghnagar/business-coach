"""
Business Meeting Copilot - Main Application

A Flask-based application that provides an AI-powered business coaching assistant
with speech recognition, text-to-speech, talking avatar, and engagement detection.

Features:
- Real-time chat with Azure OpenAI (GPT-4)
- Speech-to-text and text-to-speech via Azure Speech Service
- Talking avatar with WebRTC support and lip-sync
- Real-time engagement state detection from video feeds
- Optional Azure Cognitive Search integration (On Your Data)
- Streaming responses for real-time interaction

Usage:
    python app.py

The application will start on http://localhost:5000 by default.
Configuration can be modified in config.py or via environment variables.
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
