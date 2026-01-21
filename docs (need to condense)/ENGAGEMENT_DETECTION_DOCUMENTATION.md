# Engagement State Detection System - User Documentation

## Overview

The Engagement State Detection System is an advanced feature of the Business Meeting Copilot that analyzes video feeds in real-time to determine how engaged meeting participants are. Using advanced facial recognition technology (MediaPipe), the system extracts 100 key facial features and computes an engagement score from 0 to 100, providing actionable insights to help optimize your business meetings.

## What is Engagement Detection?

Engagement detection analyzes facial expressions, eye contact, head movements, and other visual cues to assess how attentive and interested a meeting participant is. The system provides:

- **Real-time Engagement Score**: A score from 0-100 indicating overall engagement level
- **Visual Feedback**: A vertical engagement bar displayed next to the video feed
- **Actionable Insights**: Contextual suggestions for improving meeting dynamics
- **AI Coaching Integration**: Automatic context sharing with the AI coach for personalized recommendations

## How It Works

### Technical Process

1. **Video Capture**: The system captures video from your selected source (webcam, video file, or stream)
2. **Face Detection**: MediaPipe Face Mesh detects and tracks faces in the video feed
3. **Feature Extraction**: 100 key facial features are extracted, including:
   - Eye openness and movement
   - Eyebrow position
   - Mouth activity and expressions
   - Head pose and orientation
   - Facial symmetry
   - Micro-expressions
4. **Engagement Scoring**: Multiple metrics are computed and combined:
   - **Attention** (25% weight): Eye openness and focus
   - **Eye Contact** (20% weight): Gaze direction and eye alignment
   - **Facial Expressiveness** (15% weight): Expression activity and variation
   - **Head Movement** (15% weight): Stability and orientation
   - **Symmetry** (10% weight): Facial symmetry indicating focus
   - **Mouth Activity** (15% weight): Speaking, smiling, and mouth movements
5. **Context Generation**: The system generates contextual information and suggestions
6. **Visualization**: Results are displayed in real-time on the engagement bar

### Engagement Levels

The system categorizes engagement into five levels:

- **VERY_HIGH (85-100)**: Highly engaged, showing strong interest
- **HIGH (70-85)**: Actively engaged and receptive
- **MEDIUM (45-70)**: Moderately engaged, listening but not actively participating
- **LOW (25-45)**: Low engagement, showing signs of distraction
- **VERY_LOW (0-25)**: Very low engagement, possibly disengaged or distracted

## Using the System

### Starting Engagement Detection

Engagement detection can be started in several ways:

#### Automatic Start (Recommended)
When you initialize an avatar session and video becomes available, engagement detection automatically starts if a video stream is detected.

#### Manual Start via API
You can manually start detection by calling the API endpoint:

```javascript
// Start detection from webcam
fetch('http://localhost:5000/engagement/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        sourceType: 'webcam'  // or 'file' or 'stream'
    })
})
```

#### Supported Video Sources

1. **Webcam** (`sourceType: 'webcam'`)
   - Uses your default camera (usually index 0)
   - Best for live meetings
   - No additional configuration needed

2. **Video File** (`sourceType: 'file'`)
   - Analyze recorded meetings
   - Requires `sourcePath` parameter with file path
   - Example: `{ sourceType: 'file', sourcePath: '/path/to/video.mp4' }`

3. **Video Stream** (`sourceType: 'stream'`)
   - Analyze streaming video (WebRTC, RTSP, HTTP streams)
   - Requires `sourcePath` parameter with stream URL
   - Example: `{ sourceType: 'stream', sourcePath: 'rtsp://example.com/stream' }`

### Viewing Engagement Data

#### Engagement Bar

The engagement bar appears to the left of the video feed and displays:

- **Vertical Bar**: Visual representation of engagement score (0-100%)
  - Color gradient: Red (low) → Orange → Yellow → Green → Blue (high)
- **Score Value**: Current engagement score displayed in the center
- **Level Indicator**: Text label showing engagement level (e.g., "HIGH", "MEDIUM")

The bar updates in real-time (approximately every 500ms) as engagement changes.

#### Reading the Engagement Bar

- **High Bar (70-100%)**: Green/Blue colors indicate high engagement
- **Medium Bar (45-70%)**: Yellow/Orange colors indicate moderate engagement
- **Low Bar (0-45%)**: Red/Orange colors indicate low engagement
- **No Face Detected**: Bar shows "--" and "No Face" indicator

### Getting Engagement Context

The system provides detailed context that can be used for AI coaching:

#### Via API

```javascript
// Get current engagement state
const response = await fetch('http://localhost:5000/engagement/state')
const data = await response.json()

console.log('Score:', data.score)
console.log('Level:', data.level)
console.log('Metrics:', data.metrics)
console.log('Context:', data.context)
```

#### Context Structure

The context includes:

- **Summary**: Brief description of engagement state
- **Level Description**: Detailed explanation of the engagement level
- **Key Indicators**: List of behavioral indicators observed
- **Suggested Actions**: Recommended actions to improve engagement
- **Risk Factors**: Potential concerns or issues
- **Opportunities**: Opportunities to capitalize on current engagement

### Stopping Detection

To stop engagement detection:

```javascript
fetch('http://localhost:5000/engagement/stop', {
    method: 'POST'
})
```

Detection automatically stops when you close the avatar session.

## Understanding the Metrics

### Attention Score
Measures how focused the participant is:
- **High (70-100)**: Eyes open, focused forward, minimal blinking
- **Low (0-40)**: Eyes closing, unfocused, excessive blinking

### Eye Contact Score
Measures gaze direction and eye alignment:
- **High (70-100)**: Looking directly at camera/speaker
- **Low (0-40)**: Looking away, avoiding eye contact

### Facial Expressiveness Score
Measures facial expression activity:
- **High (70-100)**: Active expressions, showing interest
- **Low (0-40)**: Minimal expressions, passive appearance

### Head Movement Score
Measures head stability:
- **High (70-100)**: Stable head position, focused
- **Low (0-40)**: Excessive movement, restlessness

### Symmetry Score
Measures facial symmetry:
- **High (70-100)**: Symmetric face, indicating focus
- **Low (0-50)**: Asymmetric features, possible distraction

### Mouth Activity Score
Measures mouth movement and activity:
- **High (70-100)**: Active speaking or showing interest
- **Low (0-40)**: Minimal mouth activity, not responding

## Interpreting Results

### High Engagement (70-100)

**What it means**: The participant is actively engaged and showing strong interest.

**What to do**:
- Present key proposals or main points
- Seek commitments or decisions
- Advance to next steps
- Capitalize on momentum
- Ask for feedback or input

**Example Context**: "High engagement detected (score: 82/100). The meeting partner is actively engaged and receptive. This is an optimal time to present key points or seek commitments."

### Medium Engagement (45-70)

**What it means**: The participant is listening but not actively participating.

**What to do**:
- Invite participation with questions
- Provide more compelling details or examples
- Address potential concerns proactively
- Build buy-in by highlighting benefits
- Use storytelling to maintain interest

**Example Context**: "Moderate engagement detected (score: 58/100). The meeting partner is listening but may need encouragement to participate more actively."

### Low Engagement (0-45)

**What it means**: The participant shows signs of distraction or disinterest.

**What to do**:
- Ask direct questions to re-engage
- Change topic or approach
- Check in: "Is this the right time?" or "Would you like me to clarify?"
- Use their name to regain attention
- Simplify the message
- Pause and allow them to refocus

**Example Context**: "Low engagement detected (score: 32/100). The meeting partner shows signs of distraction or disinterest. Consider changing approach or checking in."

## Integration with AI Coach

The engagement detection system automatically integrates with the AI coaching feature:

1. **Automatic Context Sharing**: Engagement context is available to the AI coach
2. **Personalized Recommendations**: The AI uses engagement data to provide tailored suggestions
3. **Real-time Insights**: Get immediate feedback on meeting dynamics

### Using Engagement Context in Chat

You can ask the AI coach questions about engagement:

- "What should I do based on the current engagement level?"
- "How can I improve engagement?"
- "Is this a good time to present my proposal?"

The AI will use the current engagement state to provide contextual recommendations.

## Troubleshooting

### No Face Detected

**Problem**: Engagement bar shows "--" and "No Face"

**Solutions**:
- Ensure the video source is active and showing a face
- Check camera permissions
- Verify video quality and lighting
- Ensure face is clearly visible and not obscured

### Engagement Score Not Updating

**Problem**: Score remains static or doesn't change

**Solutions**:
- Verify detection is running (check browser console)
- Ensure video feed is active
- Check network connection to backend
- Restart detection if needed

### Low Accuracy

**Problem**: Engagement scores don't seem accurate

**Solutions**:
- Ensure good lighting conditions
- Position camera at eye level
- Minimize background distractions
- Ensure stable video connection
- Allow a few seconds for smoothing to stabilize

### Performance Issues

**Problem**: System is slow or laggy

**Solutions**:
- Close other applications using the camera
- Reduce video resolution if possible
- Check system resources (CPU, memory)
- Ensure stable internet connection for streams

## Best Practices

1. **Lighting**: Ensure good, even lighting on faces
2. **Camera Position**: Position camera at eye level for best results
3. **Stable Connection**: Use wired connections when possible for video streams
4. **Privacy**: Be aware that video is being analyzed - inform participants
5. **Calibration**: Allow a few seconds for the system to stabilize after starting
6. **Context**: Use engagement data as one input among many - consider other factors

## Privacy and Ethics

- **Video Processing**: Video is processed locally/on the server - not stored unless configured
- **Consent**: Inform meeting participants that engagement detection is active
- **Data Usage**: Engagement data is used only for meeting optimization
- **Transparency**: Participants can see their engagement score in real-time

## Technical Details

### System Requirements

- Python 3.8+
- MediaPipe 0.10.9+
- OpenCV 4.9.0+
- NumPy 1.26.4+
- Modern web browser with WebRTC support

### Performance

- **Processing Speed**: ~30 FPS on modern hardware
- **Latency**: <100ms processing delay
- **Accuracy**: Optimized for business meeting scenarios
- **Resource Usage**: Moderate CPU usage, minimal memory footprint

### Limitations

- Requires clear face visibility
- Works best with frontal face views
- May be affected by:
  - Poor lighting
  - Extreme angles
  - Face coverings
  - Multiple faces (tracks primary face only)

## Support and Feedback

For issues, questions, or feedback about the engagement detection system:

1. Check this documentation first
2. Review browser console for error messages
3. Check backend logs for processing errors
4. Verify all dependencies are installed correctly

## Conclusion

The Engagement State Detection System provides valuable real-time insights into meeting dynamics, helping you optimize your business meetings for better outcomes. By understanding engagement levels and acting on the provided suggestions, you can improve participation, decision-making, and overall meeting effectiveness.

Remember: Engagement detection is a tool to assist you - always consider the full context of your meeting and use your judgment alongside the system's recommendations.
