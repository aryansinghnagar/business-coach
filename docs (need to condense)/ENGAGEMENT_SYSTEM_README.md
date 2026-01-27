# Engagement State Detection System - Technical Overview

## System Architecture

The Engagement State Detection System is a modular, real-time video analysis system that integrates seamlessly with the Business Meeting Copilot. It consists of four main components:

### 1. Core Detector (`engagement_state_detector.py`)
- Main orchestrator for engagement detection
- Manages video processing pipeline
- Thread-safe state management
- Provides public API for starting/stopping detection

### 2. Video Source Handler (`utils/video_source_handler.py`)
- Unified interface for multiple video sources
- Supports webcam, video files, and streams
- Abstracts OpenCV video capture complexity

### 3. Engagement Scorer (`utils/engagement_scorer.py`)
- Computes detailed engagement metrics
- Combines multiple factors into overall score
- Configurable weights for different metrics

### 4. Context Generator (`utils/context_generator.py`)
- Translates technical metrics into actionable insights
- Generates natural language context for AI coaching
- Provides suggestions based on engagement levels

### 5. Face Detection Backends
- **MediaPipe Detector** (`utils/mediapipe_detector.py`): Local processing, 468 landmarks (**default**).
- **Azure Face API Detector** (`utils/azure_face_detector.py`): Cloud processing, 27 landmarks + emotions (optional).
- **Interface** (`utils/face_detection_interface.py`): Pluggable abstraction layer.
- **Azure Landmark Mapper** (`utils/azure_landmark_mapper.py`): Expands Azure 27 → MediaPipe-like 468 for signifiers.
- **Face Detection Preference** (`utils/face_detection_preference.py`): Runtime choice (MediaPipe vs Azure).

### 6. Expression Signifiers & Scoring
- **Expression Signifier Engine** (`utils/expression_signifiers.py`): 30 expression signifiers (0–100 each), composite score.
- **Signifier Weights** (`utils/signifier_weights.py`): Loads signifier/group weights from URL or `weights/signifier_weights.json`.
- **Engagement Scorer** (`utils/engagement_scorer.py`): EngagementMetrics (attention, eye_contact, etc.) used alongside signifiers.

## File Structure

```
business-meeting-copilot/
├── engagement_state_detector.py       # Main detector orchestration
├── utils/
│   ├── video_source_handler.py        # Video source abstraction
│   ├── engagement_scorer.py          # Engagement metrics
│   ├── context_generator.py           # Context for AI
│   ├── expression_signifiers.py      # 30 signifiers + composite score
│   ├── signifier_weights.py          # Weights for signifiers/groups
│   ├── face_detection_interface.py   # Face detector abstraction
│   ├── mediapipe_detector.py         # MediaPipe (default)
│   ├── azure_face_detector.py        # Azure Face API (optional)
│   ├── azure_landmark_mapper.py      # Azure 27 → 468 expansion
│   └── face_detection_preference.py   # Runtime method preference
├── weights/signifier_weights.json    # Optional weights
├── routes.py                          # API endpoints
└── index.html / static/js/            # Frontend (session, engagement, video source)
```

## Key Features

### Real-time Processing
- Processes video at ~30 FPS
- Low latency (<100ms)
- Thread-safe implementation
- Smoothing for stable readings

### Multi-source Support
- Webcam (default camera)
- Local video files
- Video streams (WebRTC, RTSP, HTTP)

### Comprehensive Metrics
- Attention level
- Eye contact quality
- Facial expressiveness
- Head movement stability
- Facial symmetry
- Mouth activity

### AI Integration
- Automatic context generation
- Seamless integration with AI coach
- Real-time suggestions

## API Endpoints

### Start Detection
```
POST /engagement/start
Body: { "sourceType": "webcam|file|stream", "sourcePath": "optional" }
```

### Stop Detection
```
POST /engagement/stop
```

### Get Current State
```
GET /engagement/state
Returns: { score, level, metrics, context, ... }
```

### Get Formatted Context
```
GET /engagement/context
Returns: { context: "formatted string", score, level }
```

## Usage Example

```python
from engagement_state_detector import EngagementStateDetector, VideoSourceType

# Create detector
detector = EngagementStateDetector()

# Start detection from webcam
detector.start_detection(VideoSourceType.WEBCAM)

# Get current state
state = detector.get_current_state()
if state:
    print(f"Engagement: {state.score:.1f} ({state.level.name})")
    print(f"Context: {state.context.summary}")

# Stop detection
detector.stop_detection()
```

## Dependencies

- `mediapipe==0.10.9` - Face detection and landmark extraction (default)
- `opencv-python==4.9.0.80` - Video capture and processing
- `numpy==1.26.4` - Numerical computations
- `azure-cognitiveservices-vision-face==0.7.0` - Azure Face API (optional)

## Performance Considerations

- **CPU Usage**: Moderate (~20-30% on modern CPUs)
- **Memory**: ~100-200MB for video buffers
- **Network**: Minimal (only API calls, no video streaming)
- **GPU**: Optional (MediaPipe can use GPU acceleration)

## Testing

To test the system:

1. Start the Flask server: `python app.py`
2. Open `http://localhost:5000` in browser
3. Initialize avatar session (triggers automatic detection)
4. Observe engagement bar updating in real-time
5. Check browser console for any errors

## Future Enhancements

Potential improvements:
- Multi-face detection and tracking
- Historical engagement trends
- Custom metric weights configuration
- Export engagement reports
- Integration with meeting recording

## Face Detection Backends

The system supports two face detection backends:

1. **MediaPipe** (default): Local processing, 468 landmarks, no API costs
2. **Azure Face API** (optional): Cloud processing, 27 landmarks + emotions, requires API key

See [AZURE_FACE_API_INTEGRATION.md](AZURE_FACE_API_INTEGRATION.md) for detailed comparison and configuration.

## Notes

- MediaPipe provides 468 facial landmarks; Azure Face API provides 27 landmarks + emotion data
- 100 key features are extracted by combining related landmarks (adapts to backend)
- Smoothing window defaults to 10 frames for stability
- All processing is done server-side for privacy
- Automatic fallback to MediaPipe if Azure Face API is unavailable
