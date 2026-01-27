# Architecture Overview

## Project Structure

```
business-meeting-copilot/
├── app.py                              # Main Flask application (entry point)
├── config.py                           # Centralized configuration management
├── routes.py                           # All Flask route handlers
├── gui_launcher.py                     # GUI launcher (start/stop server)
├── engagement_state_detector.py        # Engagement detection orchestration
├── requirements.txt                    # Python dependencies
├── index.html                          # Frontend interface
├── static/js/                          # Frontend modules
│   ├── session-manager.js              # Session lifecycle, avatar + engagement
│   ├── avatar-chat-manager.js          # Chat, streaming, speech
│   ├── engagement-detector.js          # Engagement API polling
│   ├── engagement-bar.js              # Engagement bar UI
│   ├── signifier-panel.js             # 30 signifiers panel
│   └── video-source-selector.js       # Video source + face detection method
│
├── services/                           # Service layer (business logic)
│   ├── __init__.py
│   ├── azure_openai.py                 # Azure OpenAI integration
│   ├── azure_speech.py                 # Azure Speech Service integration
│   └── azure_face_api.py               # Azure Face API integration
│
├── utils/                              # Utilities and detection/scoring
│   ├── __init__.py
│   ├── helpers.py                      # Helpers, build_config_response
│   ├── video_source_handler.py         # Video source (webcam/file/stream)
│   ├── engagement_scorer.py           # EngagementMetrics, composite score
│   ├── context_generator.py           # EngagementContext for AI
│   ├── face_detection_interface.py    # FaceDetectorInterface
│   ├── mediapipe_detector.py          # MediaPipe face detector (468 pts)
│   ├── azure_face_detector.py         # Azure Face API detector (27 pts)
│   ├── azure_landmark_mapper.py       # Expand Azure 27 → MediaPipe-like 468
│   ├── face_detection_preference.py   # Runtime MediaPipe vs Azure preference
│   ├── business_meeting_feature_extractor.py  # 100 blendshape-style features
│   ├── expression_signifiers.py       # 30 expression signifiers (0–100)
│   ├── signifier_weights.py           # Weights for signifiers and groups
│   └── engagement_levels.py           # EngagementLevel enum helpers
│
└── weights/                            # Optional ML/signifier weights
    └── signifier_weights.json         # signifier [30], group [4]
```

## Design Principles

### 1. Separation of Concerns
- **Routes** (`routes.py`): Handle HTTP requests/responses only
- **Services** (`services/`): Contain business logic and external API interactions
- **Utils** (`utils/`): Reusable helper functions
- **Config** (`config.py`): All configuration in one place

### 2. Modularity
- Each service is self-contained and can be tested independently
- Services use dependency injection pattern (global instances for simplicity)
- Clear interfaces between modules

### 3. Configuration Management
- All settings centralized in `config.py`
- Environment variable support for production
- Type hints for better IDE support and documentation

### 4. Documentation
- Comprehensive docstrings for all functions and classes
- Inline comments for complex logic
- README with usage instructions

## Data Flow

### Chat Request Flow
```
Frontend (index.html)
    ↓ HTTP POST /chat/stream
Routes (routes.py)
    ↓ Extract messages, enable_oyd flag
AzureOpenAIService (services/azure_openai.py)
    ↓ Build request, handle streaming
Azure OpenAI API
    ↓ Stream response chunks
Routes (routes.py)
    ↓ Format as Server-Sent Events
Frontend (index.html)
    ↓ Display in real-time
```

### Configuration Flow
```
Frontend (index.html)
    ↓ HTTP GET /config/all
Routes (routes.py)
    ↓ Call helper function
Helpers (utils/helpers.py)
    ↓ Aggregate config from config.py
Routes (routes.py)
    ↓ Return JSON
Frontend (index.html)
    ↓ Initialize application
```

## Key Components

### AzureOpenAIService
- **Purpose**: Handle all Azure OpenAI interactions
- **Methods**:
  - `chat_completion()`: Non-streaming chat
  - `stream_chat_completion()`: Streaming chat with On Your Data support
- **Features**: Automatic system prompt injection, On Your Data integration

### AzureSpeechService
- **Purpose**: Manage Azure Speech Service tokens
- **Methods**:
  - `get_speech_token()`: Get access token for STT/TTS
  - `get_avatar_relay_token()`: Get WebRTC relay token for avatar
- **Features**: Error handling, timeout management

### AzureFaceAPIService
- **Purpose**: Handle Azure Face API interactions for face detection
- **Methods**:
  - `detect_faces()`: Detect faces and extract landmarks/attributes
  - `extract_landmarks_from_face()`: Extract facial landmarks
  - `extract_emotion_from_face()`: Extract emotion scores
  - `extract_head_pose_from_face()`: Extract head pose angles
- **Features**: Emotion analysis, head pose estimation, attribute detection

### EngagementStateDetector
- **Purpose**: Real-time engagement state detection from video feeds
- **Methods**:
  - `start_detection()`: Start detection from video source
  - `stop_detection()`: Stop detection and release resources
  - `get_current_state()`: Get current engagement state (thread-safe)
- **Features**: 
  - Supports MediaPipe and Azure Face API backends
  - Multi-threaded video processing
  - Real-time scoring (0-100)
  - Context generation for AI coaching

### Configuration Module
- **Purpose**: Centralized configuration with environment variable support
- **Features**:
  - Type hints for all settings
  - Helper functions for conditional logic (e.g., `is_cognitive_search_enabled()`)
  - Sensible defaults for development
  - Face detection method selection (MediaPipe/Azure Face API)

## Extension Points

### Adding a New Service
1. Create a new file in `services/` (e.g., `services/new_service.py`)
2. Create a service class with clear methods
3. Add configuration to `config.py`
4. Create route handlers in `routes.py`
5. Update `utils/helpers.py` if config needs to be exposed to frontend
6. Update README with new endpoints

### Adding a New Route
1. Add route handler function in `routes.py`
2. Use appropriate service from `services/`
3. Follow existing error handling patterns
4. Add docstring describing the endpoint
5. Update README API documentation

### Modifying Configuration
1. Add new settings to `config.py`
2. Use environment variables for sensitive data
3. Add helper functions if conditional logic is needed
4. Update `build_config_response()` in `utils/helpers.py` if needed for frontend

## Testing Strategy

### Unit Tests (Recommended)
- Test each service class independently
- Mock external API calls
- Test configuration loading
- Test helper functions

### Integration Tests (Recommended)
- Test route handlers with test client
- Verify response formats
- Test error handling

### Manual Testing
- Start server: `python app.py`
- Test each endpoint via browser or Postman
- Verify frontend integration

## Security Considerations

1. **API Keys**: Never commit to version control
2. **CORS**: Restrict origins in production
3. **Input Validation**: Add validation for user inputs
4. **Rate Limiting**: Consider adding rate limiting for production
5. **Error Messages**: Don't expose sensitive details in error responses

## Performance Considerations

1. **Streaming**: Already implemented for real-time responses
2. **Connection Pooling**: Requests library handles this automatically
3. **Caching**: Consider caching configuration responses
4. **Async**: Could be migrated to async/await for better concurrency

## Future Enhancements

1. **Logging**: Add structured logging (e.g., using `logging` module)
2. **Error Tracking**: Integrate error tracking service (e.g., Sentry)
3. **Metrics**: Add application metrics and monitoring
4. **Database**: Add database for conversation history
5. **Authentication**: Add user authentication and authorization
6. **API Versioning**: Add API versioning for backward compatibility
