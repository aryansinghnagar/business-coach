# Configuration Reference

This document describes all configuration options for the Business Meeting Copilot. Configuration is centralized in `config.py` and can be overridden via environment variables.

---

## Azure OpenAI

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_KEY` | *(from config)* | API key for Azure OpenAI |
| `AZURE_OPENAI_ENDPOINT` | *(from config)* | Azure OpenAI endpoint URL |
| `DEPLOYMENT_NAME` | `gpt-4o` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-11-20` | API version |

---

## Azure Speech Service

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECH_KEY` | *(from config)* | Azure Speech API key |
| `SPEECH_REGION` | `centralindia` | Speech service region |
| `SPEECH_PRIVATE_ENDPOINT_ENABLED` | `false` | Use private endpoint |
| `SPEECH_PRIVATE_ENDPOINT` | *(none)* | Private endpoint URL if enabled |

---

## Azure Cognitive Search (On Your Data)

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_COG_SEARCH_ENDPOINT` | `""` | Search service endpoint |
| `AZURE_COG_SEARCH_API_KEY` | `""` | Search API key |
| `AZURE_COG_SEARCH_INDEX_NAME` | `""` | Index name |

Use `config.is_cognitive_search_enabled()` to check if On Your Data is configured.

---

## Azure Face API (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_FACE_API_KEY` | *(from config)* | Face API key |
| `AZURE_FACE_API_ENDPOINT` | *(from config)* | Face API endpoint URL |
| `AZURE_FACE_API_REGION` | `centralindia` | Face API region |

Use `config.is_azure_face_api_enabled()` to check if Azure Face API is available.

---

## Face Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `FACE_DETECTION_METHOD` | `mediapipe` | `"mediapipe"` (default, recommended) or `"azure_face_api"` |
| `LIGHTWEIGHT_MODE` | `false` | If true: MediaPipe only, smaller buffer, process every 2nd frame |

**MediaPipe** (default): Local processing, 468 landmarks, fast.  
**Azure Face API**: Cloud-based, 27 landmarks + emotions. Requires Azure Face API configuration.

---

## Signifier Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNIFIER_WEIGHTS_URL` | *(none)* | URL to fetch weights JSON (overrides file) |
| `SIGNIFIER_WEIGHTS_PATH` | `weights/signifier_weights.json` | Local path to weights file |

Weights format: `{"signifier": [30 floats], "group": [4 floats]}`. Used by `ExpressionSignifierEngine` for composite scoring.

---

## Speech / TTS

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_LOCALES` | `en-US,de-DE,...` | Comma-separated STT locales |
| `TTS_VOICE` | `en-US-AndrewMultilingualNeural` | TTS voice name |
| `CUSTOM_VOICE_ENDPOINT_ID` | `""` | Custom neural voice endpoint |
| `CONTINUOUS_CONVERSATION` | `false` | Continuous conversation mode |

---

## Avatar

| Variable | Default | Description |
|----------|---------|-------------|
| `AVATAR_CHARACTER` | `jeff` | Avatar character |
| `AVATAR_STYLE` | `business` | Avatar style |
| `PHOTO_AVATAR` | `false` | Use photo-based avatar |
| `CUSTOMIZED_AVATAR` | `false` | Use customized avatar |
| `USE_BUILT_IN_VOICE` | `false` | Use built-in voice |
| `AUTO_RECONNECT_AVATAR` | `false` | Auto-reconnect avatar on disconnect |
| `USE_LOCAL_VIDEO_FOR_IDLE` | `false` | Use local video when idle |
| `SHOW_SUBTITLES` | `false` | Show subtitles |

---

## Flask Server

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Bind host |
| `FLASK_PORT` | `5000` | Bind port |
| `FLASK_DEBUG` | `false` | Debug mode |

---

## Runtime Face Detection Preference

The active face detection method can be set at runtime via:

- **API**: `PUT /config/face-detection` with `{"method": "mediapipe"}` or `{"method": "azure_face_api"}`
- **Video source flow**: When starting engagement detection, the client can send `detectionMethod` in the request body.

If not set, the app uses `FACE_DETECTION_METHOD` from config (default: `mediapipe`).
