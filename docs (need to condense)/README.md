# Business Meeting Copilot

An AI-powered business coaching assistant with real-time speech recognition, text-to-speech, and talking avatar capabilities. Built with Flask and Azure services.

## Features

- **AI-Powered Chat**: Real-time conversations with Azure OpenAI (GPT-4)
- **Meeting Coach AI**: Specialized business meeting coach that provides real-time insights based on engagement state
- **Engagement State Detection**: Real-time video analysis using MediaPipe or Azure Face API to detect engagement levels (0-100 score) with visual feedback
- **Engagement-Aware Insights**: AI analyzes meeting partner engagement levels and provides tailored, actionable recommendations
- **Speech Recognition (STT)**: Multi-language speech-to-text via Azure Speech Service
- **Talking Avatar**: Full avatar video with lip-synced audio using Azure Avatar service
- **Avatar Audio (TTS)**: Natural voice synthesis with lip-sync video playback
- **Subtitles Support**: Optional subtitles display synchronized with avatar speech
- **Streaming Responses**: Real-time streaming of AI responses
- **On Your Data**: Optional Azure Cognitive Search integration for custom knowledge bases

## Project Structure

```
business-meeting-copilot/
├── app.py                              # Main Flask application entry point
├── gui_launcher.py                     # GUI launcher (start/stop server)
├── config.py                           # Centralized configuration
├── routes.py                           # Flask route handlers
├── engagement_state_detector.py       # Engagement detection orchestration
├── requirements.txt                   # Python dependencies
├── index.html                         # Frontend interface
├── static/js/                         # Frontend modules (session, engagement, video source, etc.)
├── services/                          # Azure OpenAI, Speech, Face API
├── utils/                             # Face detection, signifiers, video, scoring, context
├── weights/                           # Optional signifier_weights.json
└── docs (need to condense)/           # All project documentation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for full structure and doc index.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with API key and endpoint
- Azure Speech Service account with API key
- (Optional) Azure Cognitive Search for On Your Data feature
- Webcam or video source for engagement detection (optional but recommended)

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application**:
   
   Edit `config.py` or set environment variables:
   
   ```bash
   # Azure OpenAI
   export AZURE_OPENAI_KEY="your-openai-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export DEPLOYMENT_NAME="gpt-4o"
   
   # Azure Speech Service
   export SPEECH_KEY="your-speech-key"
   export SPEECH_REGION="eastus2"
   
   # Optional: Azure Cognitive Search
   export AZURE_COG_SEARCH_ENDPOINT="https://your-search.search.windows.net"
   export AZURE_COG_SEARCH_API_KEY="your-search-key"
   export AZURE_COG_SEARCH_INDEX_NAME="your-index-name"
   ```

   **Note**: For production, always use environment variables instead of hardcoding credentials in `config.py`.

## Usage

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Start using the application**:
   - Click "Initialize Avatar Session" to set up the talking avatar with video and audio
   - Once initialized, you'll see the avatar video in the main area
   - Click "Start Microphone" to enable voice input (speech-to-text)
   - Type messages in the input field or speak into your microphone
   - The AI assistant will respond with text, avatar video (with lip sync), and audio
   - Use "Stop Speaking" to interrupt AI responses if needed
   - Click "Close Avatar Session" when done

## API Endpoints

Full reference: [API-REFERENCE.md](API-REFERENCE.md).

- **Chat**: `POST /chat`, `POST /chat/stream` (streaming with optional engagement context)
- **Config**: `GET /config/all`, `GET /config/face-detection`, `PUT /config/face-detection`, and other config endpoints
- **Speech / Avatar**: `GET /speech/token`, `GET /avatar/relay-token`
- **Engagement**: `POST /engagement/start`, `POST /engagement/stop`, `POST /engagement/upload-video`, `GET /engagement/state`, `GET /engagement/context`, `GET /engagement/debug`
- **Weights**: `GET /weights/signifiers`, `PUT /weights/signifiers`

## Engagement Context Integration

The AI coach is designed to provide insights based on meeting partner engagement state. See [ENGAGEMENT_CONTEXT.md](ENGAGEMENT_CONTEXT.md) for detailed instructions on how to integrate engagement state information into conversations.

**Quick Example:**
```
User: "Meeting partner engagement: HIGH - actively asking questions"
User: "How should I present my proposal now?"
```

The AI will provide tailored recommendations based on the engagement level.

## Configuration

All configuration is centralized in `config.py`. The module supports:

- **Environment Variables**: All settings can be overridden via environment variables
- **Default Values**: Sensible defaults are provided for development
- **Type Safety**: Proper type hints and validation

### Key Configuration Sections

1. **Azure OpenAI**: API key, endpoint, deployment name, API version
2. **Azure Speech**: API key, region, private endpoint settings
3. **Azure Face API**: API key, endpoint, region (optional, for engagement detection)
4. **Cognitive Search**: Endpoint, API key, index name (optional)
5. **Face Detection**: Method selection (MediaPipe or Azure Face API)
6. **STT/TTS**: Locales, voice selection, custom voice endpoints
7. **Avatar**: Character, style, features (subtitles, auto-reconnect, etc.)
8. **System Prompt**: Customizable AI assistant personality and behavior

## Architecture

### Modular Design

The application follows a clean, modular architecture:

- **`app.py`**: Application factory pattern for Flask app creation
- **`routes.py`**: All HTTP endpoints organized by functionality
- **`services/`**: Business logic for Azure service integrations
- **`utils/`**: Reusable utility functions
- **`config.py`**: Centralized configuration management

### Service Layer

- **`AzureOpenAIService`**: Handles chat completions and streaming
- **`AzureSpeechService`**: Manages speech tokens and relay tokens
- **`AzureFaceAPIService`**: Handles face detection and emotion analysis (optional)

### Engagement Detection System

- **`EngagementStateDetector`**: Main orchestrator for engagement detection
- **`VideoSourceHandler`**: Unified interface for video sources (webcam, files, streams)
- **`EngagementScorer`**: Computes engagement metrics and scores
- **`ContextGenerator`**: Generates actionable insights for AI coaching
- **Face Detection Backends**: MediaPipe (default) or Azure Face API (optional)

### Benefits

- **Maintainability**: Clear separation of concerns
- **Testability**: Services can be tested independently
- **Scalability**: Easy to add new features or services
- **Documentation**: Comprehensive docstrings and comments

## Development

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for better code clarity
- Include docstrings for all functions and classes
- Keep functions focused and single-purpose

### Adding New Features

1. Add configuration to `config.py` if needed
2. Create service classes in `services/` for external integrations
3. Add route handlers in `routes.py`
4. Update this README with new endpoints or features

## Security Notes

⚠️ **Important**: 

- Never commit API keys or secrets to version control
- Use environment variables for all sensitive credentials in production
- The current `config.py` contains example keys - replace them with your own
- Consider using Azure Key Vault or similar secret management for production

## Troubleshooting

### Common Issues

1. **"Failed to get speech token"**
   - Verify your `SPEECH_KEY` and `SPEECH_REGION` are correct
   - Check that your Azure Speech Service resource is active

2. **"Failed to get AI response"**
   - Verify your `AZURE_OPENAI_KEY` and endpoint are correct
   - Ensure your deployment name matches an existing deployment
   - Check API version compatibility

3. **Speech recognition not working**
   - Verify microphone permissions are granted in your browser
   - Check that `SPEECH_KEY` and `SPEECH_REGION` are correct
   - Ensure the browser supports Web Speech API
   - Check browser console for any errors

4. **Avatar video/audio not working**
   - Ensure you've clicked "Initialize Avatar Session" before using voice features
   - Verify WebRTC relay token endpoint is accessible
   - Check browser console for WebRTC connection errors
   - Ensure browser supports WebRTC (Chrome, Edge, Firefox recommended)
   - Check that video element appears in the remoteVideo container
   - Verify microphone permissions are granted for STT

## License

This project is provided as-is for educational and development purposes.

## Contributing

When contributing:
1. Follow the existing code structure and style
2. Add documentation for new features
3. Update this README with relevant changes
4. Test thoroughly before submitting

## Documentation

See **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** for the full index. Key docs:

- **[README.md](README.md)** - This file (overview, setup, usage)
- **[QUICK_START.md](QUICK_START.md)** - Minimal steps to run the app
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Project structure and design
- **[CONFIGURATION.md](CONFIGURATION.md)** - All config options
- **[API-REFERENCE.md](API-REFERENCE.md)** - All API endpoints
- **[ENGAGEMENT_DETECTION_DOCUMENTATION.md](ENGAGEMENT_DETECTION_DOCUMENTATION.md)** - User guide for engagement
- **[EXPRESSION_SIGNIFIERS_DOCUMENTATION.md](EXPRESSION_SIGNIFIERS_DOCUMENTATION.md)** - 30 signifiers reference
- **[AZURE_FACE_API_INTEGRATION.md](AZURE_FACE_API_INTEGRATION.md)** - Azure Face API setup
- **[LAUNCHER_README.md](LAUNCHER_README.md)** - GUI launcher and restart options

## Support

For issues related to:
- **Azure Services**: Check Azure documentation and service status
- **Application Code**: Review error logs and check configuration
- **Frontend**: Check browser console for JavaScript errors
- **Engagement Detection**: See [ENGAGEMENT_DETECTION_DOCUMENTATION.md](ENGAGEMENT_DETECTION_DOCUMENTATION.md)
