# Troubleshooting

Quick reference for common issues and where to find more detail.

---

## Starting & Running

| Issue | What to try | More info |
|-------|-------------|-----------|
| Launcher won't start | Ensure Python is on PATH; run `python -c "import tkinter"` | [LAUNCHER_README.md](LAUNCHER_README.md) |
| Server won't start | Check port 5000: `netstat -ano \| findstr :5000`; stop other Python processes | [LAUNCHER_README.md](LAUNCHER_README.md) |
| Port 5000 in use | Stop server, wait a few seconds, or free the port | [LAUNCHER_README.md](LAUNCHER_README.md) |

---

## Azure & AI

| Issue | What to try | More info |
|-------|-------------|-----------|
| Failed to get speech token | Check `SPEECH_KEY` and `SPEECH_REGION`; verify Speech resource is active | [README.md](README.md#troubleshooting) |
| Failed to get AI response | Check `AZURE_OPENAI_KEY` and endpoint; verify deployment name and API version | [README.md](README.md#troubleshooting) |
| Speech recognition not working | Grant microphone permission; confirm `SPEECH_KEY` / region; check browser console | [README.md](README.md#troubleshooting) |

---

## Avatar & Video

| Issue | What to try | More info |
|-------|-------------|-----------|
| Avatar video/audio not working | Click "Initialize Avatar Session" first; check relay token and WebRTC in console | [README.md](README.md#troubleshooting) |
| Video source modal not showing | Ensure avatar/session init has run; check `video-source-selector.js` and console | [VIDEO_SOURCE_SELECTION_DOCUMENTATION.md](VIDEO_SOURCE_SELECTION_DOCUMENTATION.md) |

---

## Engagement Detection

| Issue | What to try | More info |
|-------|-------------|-----------|
| No face detected | Check lighting, camera, and that face is in frame; try MediaPipe if using Azure | [ENGAGEMENT_DEBUG_GUIDE.md](ENGAGEMENT_DEBUG_GUIDE.md) |
| Score stuck or low | Use `GET /engagement/debug`; check signifier scores and metrics; verify video source | [ENGAGEMENT_DEBUG_GUIDE.md](ENGAGEMENT_DEBUG_GUIDE.md) |
| Azure Face API fails | Ensure key/endpoint are set; check network and API limits; app falls back to MediaPipe after repeated empty results | [AZURE_FACE_API_INTEGRATION.md](AZURE_FACE_API_INTEGRATION.md) |
| Wrong face detection method | Set method in video source modal, or via `PUT /config/face-detection`; default is MediaPipe | [CONFIGURATION.md](CONFIGURATION.md) |

---

## Configuration

| Issue | What to try | More info |
|-------|-------------|-----------|
| Wrong config in use | Prefer environment variables over `config.py`; restart server after changes | [CONFIGURATION.md](CONFIGURATION.md) |
| Face detection method not sticking | Set via `PUT /config/face-detection` or in video source modal when starting engagement | [CONFIGURATION.md](CONFIGURATION.md#runtime-face-detection-preference) |

---

## Debug Endpoints

- `GET /engagement/state` – Current engagement state (score, metrics, signifiers).
- `GET /engagement/debug` – Debug info (e.g. metrics history, detector state).
- `GET /engagement/context` – Formatted engagement context string for the AI.

See [ENGAGEMENT_DEBUG_GUIDE.md](ENGAGEMENT_DEBUG_GUIDE.md) for engagement-specific debugging steps.
