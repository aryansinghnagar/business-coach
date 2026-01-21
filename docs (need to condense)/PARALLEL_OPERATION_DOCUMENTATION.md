# Parallel Operation System - Documentation

## Overview

The Business Meeting Copilot now supports **parallel operation** of avatar chat and engagement detection systems. Both systems run simultaneously, providing real-time engagement insights while the avatar provides AI coaching advice. The system includes intelligent interruption handling, allowing users to interrupt avatar responses seamlessly.

## Architecture

### Parallel Operation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Session Manager                          │
│  (Coordinates parallel operation of all systems)           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Engagement   │   │ Avatar Chat  │   │ Streaming    │
│ Detector     │   │ Manager      │   │ Handler      │
│              │   │              │   │              │
│ - Polls API  │   │ - Manages    │   │ - Handles    │
│ - Updates    │   │   messages   │   │   streaming  │
│   bar        │   │ - Handles    │   │ - Supports   │
│              │   │   speech     │   │   interrupt   │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Engagement   │   │ Azure        │   │ Backend API  │
│ Bar Display  │   │ Avatar       │   │ (Flask)      │
│              │   │ Synthesizer  │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

## Key Features

### 1. **Parallel Operation**
- **Engagement Detection**: Continuously monitors meeting partner engagement
- **Avatar Chat**: Provides real-time AI coaching advice
- **Independent Operation**: Both systems run simultaneously without interference
- **Shared Context**: Engagement context automatically included in chat responses

### 2. **Streaming Interruption**
- **User Interruption**: Users can interrupt avatar responses by:
  - Sending a new message (text input)
  - Speaking (voice input)
  - Clicking "Stop Speaking" button
- **Clean Cancellation**: Streaming is cleanly canceled, no partial responses
- **Immediate Response**: New query is processed immediately after interruption

### 3. **Modular Architecture**
- **SessionManager**: Coordinates parallel operation
- **AvatarChatManager**: Manages chat and speech synthesis
- **EngagementDetector**: Manages engagement detection
- **EngagementBarDisplay**: Handles visual display

## Module Reference

### SessionManager Class

Manages parallel operation of avatar chat and engagement detection.

#### Methods

##### `startSession(engagementSourceType, engagementSourcePath, detectionMethod)`
Start parallel session with both systems.

**Parameters:**
- `engagementSourceType` (string): 'webcam', 'file', or 'stream'
- `engagementSourcePath` (string|null): Path for engagement detection
- `detectionMethod` (string|null): 'mediapipe' or 'azure_face_api'

**Example:**
```javascript
await sessionManager.startSession('stream', null, 'mediapipe');
```

##### `stopSession()`
Stop all systems and clean up resources.

##### `interruptStreaming()`
Interrupt current streaming response (called automatically on user input).

##### `streamChatResponse(messages, options)`
Stream chat response with interruption support.

**Options:**
- `enableOyd` (boolean): Enable On Your Data
- `systemPrompt` (string): System prompt override
- `includeEngagement` (boolean): Include engagement context (default: true)
- `onChunk` (Function): Callback for each token
- `onComplete` (Function): Callback on completion
- `onError` (Function): Callback for errors

### AvatarChatManager Class

Manages avatar chat functionality including streaming and speech synthesis.

#### Methods

##### `handleUserQuery(userQuery, userQueryHTML, imgUrlPath)`
Handle user query and stream response.

**Features:**
- Automatically interrupts current response
- Streams response with engagement context
- Queues speech synthesis
- Updates UI in real-time

##### `interruptCurrentResponse()`
Interrupt both streaming and speaking.

##### `speak(text, endingSilenceMs)`
Speak text using avatar synthesizer.

##### `stopSpeaking()`
Stop current speech synthesis.

## Usage Flow

### 1. Initialization

```javascript
// Systems are initialized automatically on page load
// Session manager coordinates both systems
sessionManager = new SessionManager({
    engagementDetector: engagementDetector,
    apiBaseUrl: 'http://localhost:5000'
});

// Avatar chat manager initialized when avatar session starts
avatarChatManager = new AvatarChatManager({
    sessionManager: sessionManager,
    avatarSynthesizer: avatarSynthesizer,
    appConfig: appConfig
});
```

### 2. Starting Parallel Session

```javascript
// When avatar session initializes, start parallel operation
async function initializeAvatarSession() {
    // ... avatar initialization ...
    
    // Start parallel session (engagement + chat)
    await sessionManager.startSession('stream', null, 'mediapipe');
}
```

### 3. User Interaction

```javascript
// User sends message - automatically interrupts if needed
async function sendMessage() {
    // AvatarChatManager automatically interrupts current response
    await avatarChatManager.handleUserQuery(message, message, "");
}

// User speaks - automatically interrupts if needed
speechRecognizer.recognized = async (s, e) => {
    // AvatarChatManager automatically interrupts current response
    await avatarChatManager.handleUserQuery(userQuery, "", "");
};
```

### 4. Parallel Operation

Both systems operate independently:

- **Engagement Detection**: Polls `/engagement/state` every 500ms, updates bar display
- **Avatar Chat**: Streams responses, synthesizes speech, updates chat UI
- **No Interference**: Systems don't block each other

## Interruption Handling

### How It Works

1. **User Action**: User sends message or speaks
2. **Interruption Signal**: `interruptCurrentResponse()` is called
3. **Stream Cancellation**: Current stream reader is canceled
4. **Speech Stop**: Current speech synthesis is stopped
5. **Queue Clear**: Pending speech queue is cleared
6. **New Query**: New query is processed immediately

### Implementation Details

```javascript
// In AvatarChatManager
interruptCurrentResponse() {
    // Stop streaming
    if (this.sessionManager) {
        this.sessionManager.interruptStreaming();
    }
    
    // Stop speaking
    this.stopSpeaking();
}

// In SessionManager
interruptStreaming() {
    this.shouldStopStreaming = true;
    
    if (this.currentStreamReader) {
        this.currentStreamReader.cancel();
    }
    
    this.isStreaming = false;
}
```

## Code Organization

### File Structure

```
business-meeting-copilot/
├── static/
│   └── js/
│       ├── session-manager.js        # Parallel operation coordinator
│       ├── avatar-chat-manager.js    # Chat and speech management
│       ├── engagement-bar.js         # Engagement bar display
│       └── engagement-detector.js    # Engagement detection
├── index.html                        # Main HTML (streamlined)
└── routes.py                         # Backend API
```

### Code Simplification

**Before**: Duplicate code for interruption handling in multiple places
**After**: Centralized interruption handling in `AvatarChatManager`

**Before**: Manual stream management in `handleUserQuery`
**After**: Streamlined streaming via `SessionManager.streamChatResponse`

**Before**: Separate engagement detection lifecycle
**After**: Unified session lifecycle via `SessionManager`

## Benefits

### 1. **Better User Experience**
- Users can interrupt avatar responses naturally
- No waiting for avatar to finish before asking new questions
- Smooth, responsive interaction

### 2. **Parallel Processing**
- Engagement detection runs continuously
- Chat responses include real-time engagement context
- No performance degradation

### 3. **Cleaner Code**
- Modular architecture
- Separation of concerns
- Easier to maintain and extend

### 4. **Robust Error Handling**
- Automatic recovery from errors
- Graceful degradation
- Clear error messages

## Best Practices

### 1. **Always Use SessionManager**
Don't start systems independently - use `SessionManager` for coordination:

```javascript
// ✅ Good
await sessionManager.startSession('stream', null, 'mediapipe');

// ❌ Bad
await engagementDetector.start('stream');
// Then separately handle chat...
```

### 2. **Let AvatarChatManager Handle Interruptions**
Don't manually interrupt - let the manager handle it:

```javascript
// ✅ Good
await avatarChatManager.handleUserQuery(message, message, "");

// ❌ Bad
stopSpeaking();
interruptStreaming();
await handleUserQuery(message, message, "");
```

### 3. **Check Session State**
Before operations, check if session is active:

```javascript
if (!sessionManager || !sessionManager.isSessionActive) {
    alert('Please initialize session first');
    return;
}
```

## Troubleshooting

### Avatar Not Responding
1. **Check session state**: Verify `sessionManager.isSessionActive`
2. **Check avatar initialization**: Verify `avatarChatManager` exists
3. **Check console**: Look for JavaScript errors

### Engagement Bar Not Updating
1. **Check parallel operation**: Verify `sessionManager.startSession()` was called
2. **Check engagement detector**: Verify `engagementDetector.isActive`
3. **Check API**: Verify backend is running

### Interruption Not Working
1. **Check interruption handler**: Verify `interruptCurrentResponse()` is called
2. **Check stream state**: Verify `sessionManager.shouldStopStreaming` is set
3. **Check speech state**: Verify `avatarChatManager.isSpeaking` is checked

## API Integration

### Engagement Context Auto-Inclusion

Engagement context is automatically included in chat requests:

```javascript
// In routes.py - automatically prepends engagement context
if include_engagement and engagement_detector:
    state = engagement_detector.get_current_state()
    if state and state.face_detected:
        engagement_context = context_generator.format_for_ai(state.context)
        messages[-1]["content"] = engagement_msg + messages[-1]["content"]
```

### Streaming Endpoint

The `/chat/stream` endpoint supports interruption:

- Stream can be canceled mid-response
- Clean cancellation (no partial data)
- New requests processed immediately

## Performance Considerations

### Parallel Operation Impact
- **CPU**: Minimal - systems use different resources
- **Memory**: Minimal - modular design prevents leaks
- **Network**: Engagement polling (500ms) + Chat streaming (as needed)

### Optimization Tips
1. **Adjust polling frequency**: Reduce if needed (default: 500ms)
2. **Use engagement averaging**: Prevents UI jitter (default: 10 frames)
3. **Monitor session state**: Clean up properly on stop

## Migration Guide

### From Old Code

**Old:**
```javascript
// Separate engagement detection
await startEngagementDetection('webcam');

// Manual chat handling
await handleUserQuery(message, message, "");
```

**New:**
```javascript
// Unified session management
await sessionManager.startSession('webcam', null, 'mediapipe');

// Streamlined chat handling
await avatarChatManager.handleUserQuery(message, message, "");
```

### Backward Compatibility

Old functions are maintained for compatibility:
- `speak()` - Delegates to `AvatarChatManager`
- `stopSpeaking()` - Delegates to `AvatarChatManager`
- `handleUserQuery()` - Delegates to `AvatarChatManager`

## License

See main project LICENSE file.

## Support

For issues, questions, or contributions, please refer to the main project repository.
