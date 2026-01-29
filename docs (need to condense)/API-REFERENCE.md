# API Reference

Base URL: `http://localhost:5000` (default). All routes are served under the main app; there is no `/api` prefix.

---

## Static & Pages

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve `index.html` |
| GET | `/static/<path>` | Serve static assets (JS, CSS) |
| GET | `/chat.html` | Serve chat page (if exists) |
| GET | `/favicon.ico` | Favicon (204) |

---

## Chat

### POST `/chat`

Non-streaming chat.

**Request:** `{"message": "User message"}`  
**Response:** `{"response": "AI reply"}`

---

### POST `/chat/stream`

Streaming chat with conversation history and optional On Your Data / engagement context.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "enableOyd": false,
  "systemPrompt": "optional custom prompt",
  "includeEngagement": true
}
```

**Response:** Server-Sent Events (SSE) stream.

---

## Speech & Avatar

| Method | Path | Description |
|--------|------|-------------|
| GET | `/speech/token` | Speech service token for STT/TTS |
| GET | `/config/speech` | Speech service config |
| GET | `/avatar/relay-token` | WebRTC relay token for avatar |

---

## Configuration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/config/all` | Full app config (used by frontend) |
| GET | `/config/openai` | OpenAI config |
| GET | `/config/cognitive-search` | Cognitive Search config |
| GET | `/config/stt-tts` | STT/TTS config |
| GET | `/config/avatar` | Avatar config |
| GET | `/config/system-prompt` | System prompt |

### Face Detection

| Method | Path | Description |
|--------|------|-------------|
| GET | `/config/face-detection` | Current face detection config and active `method` |
| PUT | `/config/face-detection` | Set method: `{"method": "mediapipe" \| "azure_face_api"}` |
| GET | `/config/azure-face-api` | Azure Face API config / availability |

### Weights

| Method | Path | Description |
|--------|------|-------------|
| GET | `/weights/signifiers` | Current signifier [30] and group [4] weights |
| PUT | `/weights/signifiers` | Update weights: `{"signifier": [...], "group": [...]}` |

---

## Engagement Detection

### POST `/engagement/start`

Start detection from a video source.

**Request:**
```json
{
  "sourceType": "webcam" | "file" | "stream",
  "sourcePath": "path for file/stream (optional)",
  "detectionMethod": "mediapipe" | "azure_face_api" (optional)
}
```

**Response:** `{"success": true, "message": "...", "detectionMethod": "mediapipe"}`

---

### POST `/engagement/stop`

Stop engagement detection.

**Response:** `{"success": true, "message": "Engagement detection stopped"}`

---

### POST `/engagement/upload-video`

Upload a video file for detection.  
**Request:** `multipart/form-data` with key `video`.  
**Response:** `{"success": true, "filePath": "...", "message": "..."}`

---

### GET `/engagement/state`

Current engagement state (polling).

**Response:**  
`{"score": 0–100, "level": "HIGH"|…, "metrics": {...}, "context": {...}, "faceDetected": true|false, "signifierScores": {...}, ...}`

---

### GET `/engagement/context`

Formatted engagement context for AI.

**Response:** `{"context": "Meeting partner engagement: ..."}`

---

### GET `/engagement/debug`

Debug info (e.g. metrics history, detector state). Used for troubleshooting.

---

## Error Responses

- **400** – Invalid or missing body/parameters; `{"error": "..."}`  
- **404** – Resource not found  
- **500** – Server error; `{"error": "...", "details": "..."}`  
- **502** – Upstream (e.g. Azure) failure
