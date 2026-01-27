# Azure Face API Integration Guide

## Overview

The Engagement State Detection System now supports Azure Face API as an optional alternative to MediaPipe for face detection. This allows you to choose between:

- **MediaPipe** (default): Local processing, no API calls, 468 facial landmarks
- **Azure Face API**: Cloud-based processing, requires API key, 27 facial landmarks + emotion analysis

## Configuration

### Environment Variables

Add the following environment variables to enable Azure Face API:

```bash
# Azure Face API Configuration
AZURE_FACE_API_KEY=your_api_key_here
AZURE_FACE_API_ENDPOINT=https://your-region.api.cognitive.microsoft.com
AZURE_FACE_API_REGION=your-region  # Optional but recommended

# Face Detection Method Selection
FACE_DETECTION_METHOD=azure_face_api  # or "mediapipe" (default)
```

### Configuration in config.py

The system automatically detects if Azure Face API is configured:

```python
# Check if Azure Face API is enabled
if config.is_azure_face_api_enabled():
    # Azure Face API is available
    pass

# Get configuration
face_api_config = config.get_azure_face_api_config()
```

## Usage

### Automatic Selection

The system automatically selects the detection method based on configuration:

1. If `FACE_DETECTION_METHOD=azure_face_api` and Azure Face API is configured → Uses Azure Face API
2. Otherwise → Falls back to MediaPipe

### Manual Selection via API

You can specify the detection method when starting engagement detection:

```javascript
// Start with Azure Face API
fetch('http://localhost:5000/engagement/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        sourceType: 'webcam',
        detectionMethod: 'azure_face_api'  // or 'mediapipe'
    })
})
```

### Programmatic Usage

```python
from engagement_state_detector import EngagementStateDetector, VideoSourceType

# Use Azure Face API
detector = EngagementStateDetector(detection_method='azure_face_api')
detector.start_detection(VideoSourceType.WEBCAM)

# Or use MediaPipe (default)
detector = EngagementStateDetector(detection_method='mediapipe')
detector.start_detection(VideoSourceType.WEBCAM)
```

## API Endpoints

### Get Face Detection Configuration

```http
GET /config/face-detection
```

Response:
```json
{
    "method": "mediapipe",
    "mediapipeAvailable": true,
    "azureFaceApiAvailable": true
}
```

### Get Azure Face API Configuration

```http
GET /config/azure-face-api
```

Response (if configured):
```json
{
    "endpoint": "https://your-region.api.cognitive.microsoft.com",
    "apiKey": "***",
    "region": "your-region",
    "enabled": true
}
```

Response (if not configured):
```json
{
    "enabled": false
}
```

## Differences Between Methods

### MediaPipe

**Advantages:**
- ✅ No API calls (local processing)
- ✅ No additional costs
- ✅ Works offline
- ✅ 468 facial landmarks (more detailed)
- ✅ Lower latency (no network calls)
- ✅ Privacy-friendly (all processing local)

**Disadvantages:**
- ❌ Requires MediaPipe installation
- ❌ Higher CPU usage
- ❌ No built-in emotion detection

### Azure Face API

**Advantages:**
- ✅ Built-in emotion analysis
- ✅ Head pose estimation (pitch, yaw, roll)
- ✅ Additional attributes (age, gender, glasses, etc.)
- ✅ Lower CPU usage (cloud processing)
- ✅ Professional-grade accuracy

**Disadvantages:**
- ❌ Requires internet connection
- ❌ API costs (pay per request)
- ❌ Network latency
- ❌ Fewer landmarks (27 vs 468)
- ❌ Requires API key configuration

## Feature Comparison

| Feature | MediaPipe | Azure Face API |
|---------|-----------|----------------|
| Facial Landmarks | 468 points | 27 points |
| Emotion Detection | ❌ | ✅ (8 emotions) |
| Head Pose | Calculated | Provided |
| Age Estimation | ❌ | ✅ |
| Gender Detection | ❌ | ✅ |
| Glasses Detection | ❌ | ✅ |
| Processing Location | Local | Cloud |
| Internet Required | ❌ | ✅ |
| Cost | Free | Pay per request |

## Implementation Details

### Architecture

The system uses a pluggable interface (`FaceDetectorInterface`) that allows switching between detection methods:

```
engagement_state_detector.py
    ├── FaceDetectorInterface (abstract)
    ├── MediaPipeFaceDetector (implements interface)
    └── AzureFaceAPIDetector (implements interface)
```

### Feature Extraction

Both methods extract 100 key features, but use different approaches:

- **MediaPipe**: Uses all 468 landmarks to compute detailed features
- **Azure Face API**: Uses 27 landmarks + emotion data to compute features

The engagement scoring system normalizes these differences to provide consistent scores (0-100) regardless of the detection method.

### Fallback Behavior

If Azure Face API is selected but not configured or unavailable:

1. System automatically falls back to MediaPipe
2. Warning message is logged
3. Detection continues normally

## Performance Considerations

### MediaPipe
- **Latency**: ~10-30ms per frame (local processing)
- **Throughput**: ~30-60 FPS depending on hardware
- **CPU Usage**: Moderate to high (20-40% on modern CPUs)

### Azure Face API
- **Latency**: ~100-300ms per frame (network + processing)
- **Throughput**: Limited by API rate limits (typically 20-30 requests/second)
- **CPU Usage**: Low (minimal local processing)

## Best Practices

1. **For Real-time Applications**: Use MediaPipe for lower latency
2. **For Batch Processing**: Use Azure Face API for better accuracy and emotion analysis
3. **For Privacy-Critical Applications**: Use MediaPipe (all processing local)
4. **For Cost Optimization**: Use MediaPipe (no API costs)
5. **For Maximum Accuracy**: Use Azure Face API (professional-grade detection)

## Troubleshooting

### Azure Face API Not Working

1. **Check API Key**: Verify `AZURE_FACE_API_KEY` is set correctly
2. **Check Endpoint**: Ensure `AZURE_FACE_API_ENDPOINT` is correct format
3. **Check Region**: Verify region matches your Azure resource
4. **Check Network**: Ensure internet connection is available
5. **Check Quota**: Verify you haven't exceeded API rate limits

### Fallback to MediaPipe

If Azure Face API fails, the system automatically falls back to MediaPipe. Check logs for:
- "Warning: Azure Face API not available, falling back to MediaPipe"
- "Failed to initialize Azure Face API service"

### Performance Issues

- **High Latency (Azure)**: Check network connection and API endpoint location
- **High CPU (MediaPipe)**: Reduce video resolution or frame rate
- **Rate Limiting (Azure)**: Implement request throttling or use MediaPipe

## Migration Guide

### Switching from MediaPipe to Azure Face API

1. Set environment variables:
   ```bash
   export AZURE_FACE_API_KEY=your_key
   export AZURE_FACE_API_ENDPOINT=https://your-region.api.cognitive.microsoft.com
   export FACE_DETECTION_METHOD=azure_face_api
   ```

2. Restart the application

3. Verify configuration:
   ```bash
   curl http://localhost:5000/config/face-detection
   ```

### Switching from Azure Face API to MediaPipe

1. Set environment variable:
   ```bash
   export FACE_DETECTION_METHOD=mediapipe
   ```

2. Or simply remove Azure Face API environment variables

3. Restart the application

## Cost Estimation

Azure Face API pricing (as of 2024):
- **Free Tier**: 30,000 transactions/month
- **Standard Tier**: ~$1 per 1,000 transactions

For a typical meeting (1 hour, 30 FPS):
- Frames: 1 hour × 30 FPS × 3600 seconds = 108,000 frames
- Cost: ~$108 per hour (at standard pricing)

**Recommendation**: Use MediaPipe for real-time applications, Azure Face API for batch analysis or when emotion detection is critical.

## Security Considerations

### Azure Face API
- API keys should be stored securely (environment variables, not in code)
- Use private endpoints if available
- Monitor API usage for unusual activity
- Consider data residency requirements

### MediaPipe
- All processing is local
- No data leaves your system
- Better for privacy-sensitive applications

## Conclusion

Both detection methods provide excellent engagement detection capabilities. Choose based on your specific requirements:

- **Choose MediaPipe** if you need: Low latency, offline operation, privacy, cost efficiency
- **Choose Azure Face API** if you need: Emotion analysis, professional accuracy, cloud processing, additional attributes

The system seamlessly handles switching between methods, so you can experiment with both to find the best fit for your use case.
