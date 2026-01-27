# Engagement Bar Display System - Documentation

## Overview

The Engagement Bar Display System provides a visual representation of real-time engagement scores with smooth animations, dynamic color gradients, and intelligent frame averaging to prevent sudden fluctuations.

## Features

### 1. **Visual Display**
- **Vertical Bar**: Fills from bottom (0%) to top (100%) based on engagement score
- **Dynamic Color Gradient**: Smoothly transitions from red (0) → yellow (50) → green (100)
- **Smooth Animations**: CSS transitions for height and color changes
- **Real-time Updates**: Updates every 500ms with averaged values

### 2. **Frame Averaging**
- **10-Frame Buffer**: Maintains history of last 10 engagement scores
- **Smooth Transitions**: Prevents sudden jumps and fluctuations
- **Level Averaging**: Uses most common level from recent frames
- **Face Detection Averaging**: Majority vote for face detection status

### 3. **Modular Architecture**
- **EngagementBarDisplay**: Handles visual rendering and averaging
- **EngagementDetector**: Manages API communication and polling
- **Separation of Concerns**: Clear boundaries between display and data

## Architecture

```
┌─────────────────────────────────────────┐
│         EngagementDetector             │
│  (API Communication & Polling)          │
│  - Polls backend every 500ms            │
│  - Handles start/stop lifecycle         │
│  - Error handling and recovery          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       EngagementBarDisplay               │
│  (Visual Rendering & Averaging)          │
│  - 10-frame averaging buffer             │
│  - Color gradient calculation            │
│  - DOM updates and animations            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│            DOM Elements                  │
│  - engagementBarFill (bar fill)          │
│  - engagementBarValue (score text)        │
│  - engagementLevel (level indicator)      │
└─────────────────────────────────────────┘
```

## Module Reference

### EngagementBarDisplay Class

#### Constructor
```javascript
new EngagementBarDisplay(options)
```

**Options:**
- `averagingWindow` (number, default: 10): Number of frames to average
- `updateInterval` (number, default: 500): Update interval in milliseconds
- `barFillId` (string, default: 'engagementBarFill'): ID of bar fill element
- `barValueId` (string, default: 'engagementBarValue'): ID of value element
- `levelIndicatorId` (string, default: 'engagementLevel'): ID of level indicator

#### Methods

##### `update(score, level, faceDetected)`
Update the engagement bar with new values.

**Parameters:**
- `score` (number): Engagement score (0-100)
- `level` (string): Engagement level (VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
- `faceDetected` (boolean): Whether a face was detected

**Example:**
```javascript
engagementBarDisplay.update(75.5, 'HIGH', true);
```

##### `reset()`
Reset the engagement bar to initial state (clears history and resets display).

##### `getCurrentScore()`
Get the current averaged score.

**Returns:** `number` - Current averaged score (0-100)

##### `getCurrentLevel()`
Get the current averaged level.

**Returns:** `string` - Current averaged level

### EngagementDetector Class

#### Constructor
```javascript
new EngagementDetector(options)
```

**Options:**
- `barDisplay` (EngagementBarDisplay): Engagement bar display instance
- `apiBaseUrl` (string, default: 'http://localhost:5000'): Base URL for API
- `pollInterval` (number, default: 500): Polling interval in milliseconds

#### Methods

##### `start(sourceType, sourcePath, detectionMethod)`
Start engagement detection.

**Parameters:**
- `sourceType` (string): 'webcam', 'file', or 'stream'
- `sourcePath` (string|null): Path to video file or stream URL
- `detectionMethod` (string|null): 'mediapipe' or 'azure_face_api'

**Returns:** `Promise<boolean>` - True if started successfully

**Example:**
```javascript
await engagementDetector.start('webcam', null, 'mediapipe');
```

##### `stop()`
Stop engagement detection.

**Returns:** `Promise<boolean>` - True if stopped successfully

##### `getStatus()`
Get current detection status.

**Returns:** `Object` with:
- `isActive` (boolean): Whether detection is active
- `sourceType` (string|null): Current source type
- `sourcePath` (string|null): Current source path
- `detectionMethod` (string|null): Current detection method
- `consecutiveErrors` (number): Number of consecutive errors

## Color Gradient System

The engagement bar uses a dynamic color gradient that transitions smoothly based on the engagement score:

### Color Mapping

| Score Range | Color | RGB Values |
|------------|-------|------------|
| 0-50 | Red → Yellow | (255, 0, 0) → (255, 255, 0) |
| 50-100 | Yellow → Green | (255, 255, 0) → (0, 255, 0) |

### Implementation

The color is calculated using linear interpolation:

```javascript
if (score <= 50) {
    // Red to Yellow
    ratio = score / 50;
    r = 255;
    g = 255 * ratio;
    b = 0;
} else {
    // Yellow to Green
    ratio = (score - 50) / 50;
    r = 255 * (1 - ratio);
    g = 255;
    b = 0;
}
```

The gradient is applied as a CSS linear gradient from top (lighter) to bottom (more intense).

## Frame Averaging Algorithm

### Score Averaging
Scores are averaged using a simple moving average:

```javascript
averageScore = sum(scoreHistory) / scoreHistory.length
```

### Level Averaging
Levels use a majority vote system:

```javascript
// Count occurrences of each level
levelCounts = countOccurrences(levelHistory)

// Find most common level
mostCommonLevel = level with highest count
```

### Face Detection Averaging
Face detection uses majority vote:

```javascript
faceDetected = (trueCount > faceDetectedHistory.length / 2)
```

## Usage Examples

### Basic Usage

```javascript
// Initialize system
const barDisplay = new EngagementBarDisplay({ averagingWindow: 10 });
const detector = new EngagementDetector({ barDisplay });

// Start detection
await detector.start('webcam', null, 'mediapipe');

// System automatically updates bar display
// No manual updates needed

// Stop detection
await detector.stop();
```

### Custom Configuration

```javascript
// Custom averaging window and update interval
const barDisplay = new EngagementBarDisplay({
    averagingWindow: 15,  // Average over 15 frames
    updateInterval: 300   // Update every 300ms
});

// Custom API endpoint
const detector = new EngagementDetector({
    barDisplay: barDisplay,
    apiBaseUrl: 'https://api.example.com',
    pollInterval: 300
});
```

### Manual Updates (Advanced)

```javascript
// Manually update bar display (if not using detector)
barDisplay.update(75.5, 'HIGH', true);

// Get current values
const currentScore = barDisplay.getCurrentScore();
const currentLevel = barDisplay.getCurrentLevel();

// Reset display
barDisplay.reset();
```

## CSS Styling

The engagement bar uses the following CSS classes:

### `.engagement-bar-container`
Container for the entire engagement bar component.

### `.engagement-bar-wrapper`
Wrapper for the bar itself (background container).

### `.engagement-bar-fill`
The actual fill bar that changes height and color.

**Key Properties:**
- `position: absolute; bottom: 0;` - Anchored to bottom
- `height: 0-100%` - Set dynamically via JavaScript
- `background` - Set dynamically via JavaScript (color gradient)
- `transition: height 0.5s, background 0.3s` - Smooth animations

### `.engagement-bar-value`
Text overlay showing the numeric score.

### `.engagement-level-indicator`
Indicator showing the engagement level text.

## Integration with Main Application

The engagement bar system is automatically initialized when the page loads:

```javascript
// Automatic initialization
window.addEventListener('DOMContentLoaded', initializeEngagementSystem);

function initializeEngagementSystem() {
    engagementBarDisplay = new EngagementBarDisplay({ averagingWindow: 10 });
    engagementDetector = new EngagementDetector({ barDisplay: engagementBarDisplay });
}
```

The system integrates with the avatar session:

```javascript
// Start detection when avatar session starts
async function initializeAvatarSession() {
    // ... avatar initialization ...
    
    // Start engagement detection
    await startEngagementDetection('stream', null);
}

// Stop detection when session ends
function stopSession() {
    // ... session cleanup ...
    
    // Stop engagement detection
    await stopEngagementDetection();
}
```

## Performance Considerations

### Frame Averaging
- **Memory**: Stores only last 10 scores (minimal memory footprint)
- **CPU**: Simple arithmetic operations (negligible impact)
- **Smoothness**: Prevents jarring visual updates

### Polling
- **Frequency**: 500ms intervals (2 updates per second)
- **Network**: Minimal bandwidth (small JSON responses)
- **Error Handling**: Automatic recovery after errors

### DOM Updates
- **Cached Elements**: DOM elements cached for performance
- **CSS Transitions**: Hardware-accelerated animations
- **Minimal Reflows**: Only updates necessary properties

## Troubleshooting

### Bar Not Updating
1. **Check initialization**: Ensure `initializeEngagementSystem()` is called
2. **Check DOM elements**: Verify elements exist with correct IDs
3. **Check API**: Verify backend is running and accessible
4. **Check console**: Look for JavaScript errors

### Color Not Changing
1. **Check score range**: Ensure scores are 0-100
2. **Check CSS**: Verify `.engagement-bar-fill` styles are not overridden
3. **Check JavaScript**: Verify `getColorForScore()` is being called

### Smoothness Issues
1. **Increase averaging window**: Use 15-20 frames for smoother updates
2. **Check update frequency**: Reduce polling interval if needed
3. **Check browser performance**: Ensure sufficient CPU/GPU resources

## Best Practices

### 1. **Initialization Timing**
Initialize after DOM is fully loaded:
```javascript
document.addEventListener('DOMContentLoaded', initializeEngagementSystem);
```

### 2. **Error Handling**
Always handle errors gracefully:
```javascript
try {
    await detector.start('webcam');
} catch (error) {
    console.error('Failed to start detection:', error);
    // Fallback behavior
}
```

### 3. **Cleanup**
Always stop detection when done:
```javascript
window.addEventListener('beforeunload', () => {
    detector.stop();
});
```

### 4. **Configuration**
Adjust averaging window based on needs:
- **High precision**: Lower window (5-7 frames)
- **Smooth display**: Higher window (15-20 frames)
- **Default**: 10 frames (balanced)

## API Integration

The engagement bar system integrates with the backend API:

### Endpoints Used

#### `POST /engagement/start`
Start engagement detection.

**Request:**
```json
{
  "sourceType": "webcam",
  "sourcePath": null,
  "detectionMethod": "mediapipe"
}
```

#### `GET /engagement/state`
Get current engagement state.

**Response:**
```json
{
  "score": 75.5,
  "level": "HIGH",
  "faceDetected": true,
  "metrics": { ... },
  "context": { ... }
}
```

#### `POST /engagement/stop`
Stop engagement detection.

## File Structure

```
business-meeting-copilot/
├── static/
│   └── js/
│       ├── engagement-bar.js      # EngagementBarDisplay class
│       └── engagement-detector.js  # EngagementDetector class
├── index.html                      # Main HTML (includes modules)
└── routes.py                       # Backend API (serves static files)
```

## License

See main project LICENSE file.

## Support

For issues, questions, or contributions, please refer to the main project repository.
