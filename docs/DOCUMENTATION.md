# Business Meeting Copilot — Documentation

Single reference for setup, usage, architecture, APIs, engagement detection, and troubleshooting.

---

## 1. Overview

**Business Meeting Copilot** is an AI-powered meeting coach with real-time speech, avatar, and engagement detection. It uses Azure OpenAI, Azure Speech, and optional Azure Face API; engagement is computed from video (MediaPipe or Azure) and optional partner audio.

### Features

- **AI chat**: Azure OpenAI (GPT-4), streaming, optional On Your Data
- **Meeting coach**: Real-time insights from engagement state and B2B cues
- **Engagement detection**: Video analysis → 30 signifiers → 4 groups (G1–G4) → score (0–100), combination-based insights, B2B opportunity detection
- **Audiovisual insights**: Visual spikes + phrase-triggered (partner STT) + 24 B2B opportunity features → popup + TTS. Combining facial and vocal cues (multimodal fusion) improves insight quality and is supported by research on engagement detection.
- **Speech**: STT/TTS via Azure Speech; talking avatar with lip-sync (WebRTC)
- **Video sources**: Webcam, Meeting Partner Video (share + audio), or file; optional meeting tab/window audio for speech cues when using webcam or file

### Project structure

```
business-meeting-copilot/
├── app.py                    # Flask entry point
├── config.py                 # Configuration (env overrides)
├── routes.py                 # All HTTP routes
├── engagement_state_detector.py  # Engagement orchestration, spike/opportunity alerts
├── gui_launcher.py           # GUI start/stop server
├── index.html                # Frontend
├── static/js/                # Session, engagement, avatar, video-source
├── services/                 # azure_openai, azure_speech, insight_generator
├── utils/                    # Face detection, signifiers, scoring, B2B opportunity detector
├── weights/                  # Optional signifier_weights.json
├── docs/                     # DOCUMENTATION.md (this file)
└── docs (need to condense)/   # README points to docs/DOCUMENTATION.md
```

---

## 2. Quick Start

### Run with GUI (recommended)

1. **Windows**: Double-click `START_HERE.bat` or run `python gui_launcher.py`
2. **Mac/Linux**: Run `./START_HERE.sh` or `python3 gui_launcher.py`
3. Click **▶ Start Server**, wait for green status
4. Click **🌐 Open in Browser** → `http://localhost:5000`
5. Initialize Avatar Session; when the video source modal appears, choose Webcam / Meeting Partner Video / File, pick **MediaPipe** or **Azure Face API**, then **Start Detection**
6. When done: **■ Stop Server**

### Run from command line

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Status indicators (launcher)

- **Green** = Server running  
- **Red** = Server stopped  
- **Yellow** = Starting/stopping  

---

## 3. Installation & Configuration

### Prerequisites

- Python 3.8+
- Azure OpenAI (API key, endpoint, deployment)
- Azure Speech (key, region)
- Optional: Azure Face API, Azure Cognitive Search (On Your Data)
- Webcam or video source for engagement

### Install

```bash
pip install -r requirements.txt
```

### Configuration

All settings are in `config.py`; override with environment variables (recommended for production). **API keys** (`AZURE_OPENAI_KEY`, `SPEECH_KEY`, `AZURE_FACE_API_KEY`) have no defaults and must be set via environment; do not commit real keys.

| Area | Key variables |
|------|----------------|
| **Azure OpenAI** | `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `DEPLOYMENT_NAME` |
| **Azure Speech** | `SPEECH_KEY`, `SPEECH_REGION` |
| **Azure Face API** | `AZURE_FACE_API_KEY`, `AZURE_FACE_API_ENDPOINT` (optional) |
| **Face detection** | `FACE_DETECTION_METHOD` = `mediapipe` (default) or `azure_face_api` |
| **Lightweight** | `LIGHTWEIGHT_MODE` = `true`: MediaPipe only, 640×360 webcam, process every 2nd frame, smaller buffer |
| **Cognitive Search** | `AZURE_COG_SEARCH_*` (optional, On Your Data) |
| **Server** | `FLASK_HOST`, `FLASK_PORT` (default `0.0.0.0`, `5000`) |

**Runtime face detection**: `PUT /config/face-detection` with `{"method": "mediapipe"}` or `"azure_face_api"`; or set when starting engagement via the video source modal.

---

## 4. Usage

### Launcher

- **Start**: ▶ Start Server → wait for green → 🌐 Open in Browser  
- **Stop**: ■ Stop Server  
- **Restart**: Stop then Start; if port 5000 is in use, wait a few seconds or free the port (`netstat -ano | findstr :5000` on Windows).  
- **Logs**: Shown in the launcher window.  
- **Tkinter**: If the launcher fails, ensure tkinter is available: `python -c "import tkinter"`. Linux may need `python3-tk`.

### Browser flow

1. Initialize Avatar Session (token + WebRTC).
2. Video source modal: choose Webcam, Meeting Partner Video, or File; select face detection method; Start Detection.
3. Engagement runs; score and metrics update on polling. Alerts (spike, phrase, opportunity) show as popup + TTS.
4. Use chat for coaching; engagement context can be included in requests.
5. Close session or Stop Detection when done.

### Video sources

- **Webcam**: Default camera (index 0).  
- **Meeting Partner Video**: Screen/tab share with `getDisplayMedia({ video: true, audio: true })`; partner audio is used for STT and phrase-triggered insights.  
- **File**: Upload via modal; server stores under `uploads/`.

### Engagement bar / UI

- Score 0–100 and level (VERY_LOW … VERY_HIGH).  
- No face: “No Face” / “--”.  
- Insights transcript in sidebar; popup + TTS for each alert.

---

## 5. Architecture

### Design

- **Routes** (`routes.py`): HTTP only.  
- **Services**: Azure OpenAI, Speech, insight generation.  
- **Utils**: Face detection, signifiers, scoring, video source, B2B opportunity detector.  
- **Config**: Single place (`config.py`) with env overrides.

### Engagement pipeline

```
Video frame → Face detection (MediaPipe or Azure)
  → Landmarks (468 or 27→468)
  → 30 signifiers (0–100)
  → 4 group means G1–G4
  → Score: (G1+G4)/2 (MediaPipe) or composite (Azure/Unified)
  → Combination-based detection + B2B opportunity detection
  → Pending alert → GET /engagement/state → OpenAI insight → popup + TTS
```

### Main components

- **EngagementStateDetector**: Video loop, face detection, signifiers, G1–G4, spike checks, opportunity checks, pending alert.  
- **ExpressionSignifierEngine**: 30 signifiers, group means, composite score.  
- **B2B opportunity detector**: 24 opportunity types from G1–G4 (and optional signifiers); priority order; cooldown per type.  
- **Insight generator**: Spike, aural (phrase), and opportunity insights via Azure OpenAI; stock fallbacks.

---

## 6. API Reference

Base URL: `http://localhost:5000` (default).

### Static & pages

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve `index.html` |
| GET | `/static/<path>` | Static assets |

### Chat

- **POST /chat** — Non-streaming. Body: `{"message": "..."}`. Response: `{"response": "..."}`.  
- **POST /chat/stream** — Streaming SSE. Body: `{"messages": [...], "enableOyd": false, "includeEngagement": true}`.

### Speech & avatar

| Method | Path | Description |
|--------|------|-------------|
| GET | `/speech/token` | Speech token (STT/TTS) |
| GET | `/config/speech` | Speech config |
| GET | `/avatar/relay-token` | WebRTC relay token |

### Configuration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/config/all` | Full config for frontend |
| GET | `/config/face-detection` | Face detection method and availability |
| PUT | `/config/face-detection` | Set method: `{"method": "mediapipe" \| "azure_face_api"}` |
| GET | `/weights/signifiers` | Signifier [30] and group [4] weights |
| PUT | `/weights/signifiers` | Update weights |

### Engagement

| Method | Path | Description |
|--------|------|-------------|
| POST | `/engagement/start` | Start detection. Body: `{"sourceType": "webcam" \| "file" \| "stream", "sourcePath": "...", "detectionMethod": "mediapipe" \| "azure_face_api"}` |
| POST | `/engagement/stop` | Stop detection |
| POST | `/engagement/upload-video` | Upload video file (multipart, key `video`) |
| POST | `/engagement/transcript` | Append partner transcript. Body: `{"text": "..."}` or plain text |
| GET | `/engagement/state` | Current state + consumed alert (spike/opportunity/aural) → insight in `response.alert.message` |
| GET | `/engagement/context` | Formatted engagement context for AI |
| GET | `/engagement/debug` | Debug info (metrics history, detector state) |
| GET | `/engagement/score-breakdown` | Score calculation breakdown |

### Errors

- 400: Invalid/missing body or parameters  
- 404: Resource not found  
- 500: Server error (`error`, `details`)  
- 502: Upstream (e.g. Azure) failure  

---

## 7. Engagement Detection

### Metric groups (G1–G4)

- **G1** — Interest & engagement (Duchenne, eye contact, eyebrow flash, head tilt, etc.).  
- **G2** — Cognitive load (look away, thinking brow, eye squint, stillness).  
- **G3** — Resistance (contempt, gaze aversion, lip compression; stored as 100 − raw so high = less resistance).  
- **G4** — Decision-ready (relaxed exhale, fixed gaze, smile transition).

### Score and levels

- **Score**: 0–100. MediaPipe path: `(G1 + G4) / 2`; Azure/unified use composite weights.  
- **Levels**: VERY_LOW (0–25), LOW (25–45), MEDIUM (45–70), HIGH (70–85), VERY_HIGH (85–100).

### Six display metrics

Attention, eye contact, facial expressiveness, head movement, symmetry, mouth activity — derived from signifiers and (when used) Azure emotions.

### Combination-based insight detection (not single spikes)

**Single-metric spikes are DISABLED** (`_enable_single_spike_alerts=False`). Instead, insights are triggered by **psychologically meaningful COMBINATIONS** of metrics:

- **Decision-ready**: G4 high + G1 high + G3 low resistance (Cialdini: commitment signals)
- **Cognitive overload**: G2 high + G1 low (Kahneman: System 2 maxed, decision fatigue)
- **Aha moment**: G2 was high (processing) → G1 now high (insight landed)
- **Skepticism**: G3 raw rising over time (nonverbal leakage, Mehrabian)
- **Genuine interest**: G1 high + G3 low (Duchenne smile + forward lean + eye contact = approach motivation)

**Composite-at-100** (legacy): When G1 (Interest), G4 (Decision-Ready), or overall composite score ≥ 90, a **composite_100 alert** is set. When any alert fires, next `GET /engagement/state` triggers insight generation (OpenAI) and returns the message in `alert.message`.

### B2B opportunity detection (facial + multimodal)

- **Source**: `utils/b2b_opportunity_detector.py`.  
- **Input**: Group means G1–G4, group history, optional 30 signifier scores, **and** (for multimodal) recent speech tags from `services/insight_generator.get_recent_speech_tags(12)`.  
- **Output**: First opportunity that fires (priority order) and is off cooldown → `(opportunity_id, context)`.  
- **29 types**: Five **multimodal** composites (facial + speech) are checked first: `decision_readiness_multimodal`, `cognitive_overload_multimodal`, `skepticism_objection_multimodal`, `aha_insight_multimodal`, `disengagement_multimodal`. Then 24 facial-only composites (e.g. `closing_window`, `decision_ready`, `buying_signal`, `cognitive_overload_risk`, `aha_moment`, `rapport_moment`).  
- **Multimodal corroboration**: Speech tags are appended when partner transcript matches trigger phrases (POST `/engagement/transcript`). Multimodal composites require both metric conditions **and** a matching recent speech tag (e.g. commitment/interest for decision-readiness, confusion/concern for cognitive overload). This reduces false positives and aligns with research on combined facial + speech cues.  
- **Cooldown**: 32 s per opportunity type.  
- **Flow**: Engagement detector calls `detect_opportunity(..., recent_speech_tags=get_recent_speech_tags(12))`; if an opportunity fires, sets `_pending_alert` type `"opportunity"`; API calls `generate_insight_for_opportunity(...)` (context includes `recent_speech_tags` when applicable) and returns message; same popup + TTS as spike/aural.

### Audiovisual insights (speech + visual)

- **Visual**: Spike or opportunity from engagement detector (including multimodal composites when facial + speech conditions align).  
- **Aural**: When source is Meeting Partner Video with audio, frontend runs STT on partner stream and POSTs to `/engagement/transcript`. Backend checks for **trigger phrases** (objection, interest, confusion, commitment, concern, timeline, budget, realization); on match **appends speech tag** (for composite detection) and sets **aural alert** (25 s cooldown).  
- **Multimodal flow**: Speech tags (category + phrase + timestamp) are stored in a ring buffer; `get_recent_speech_tags(12)` returns tags from the last 12 s. The engagement detector passes these into `detect_opportunity()` so multimodal composites (e.g. decision-readiness when commitment/interest was just said + face matches) can fire.  
- **Insight generation**: For spike, aural, or opportunity, backend uses Azure OpenAI with context (group/opportunity id, metrics, recent transcript, and for opportunities `recent_speech_tags` when present) and returns one short sentence; fallback stock messages in `services/insight_generator.py`.  
- **Frontend**: Same handler for all alert types: show popup and speak `alert.message` via TTS.

### Psychology and metrics

- Eye contact and head orientation proxy attention; mouth corner lift prioritized over eye squinch for smile authenticity; resistance cues (contempt, gaze aversion, lip compression) weighted; G2 (cognitive load) treated as meaningful in B2B (e.g. evaluating proposal).  
- **Metric math**: [docs/ENGAGEMENT_METRICS.md](ENGAGEMENT_METRICS.md) documents the mathematical logic for each of the 30 signifier metrics.  
- Implementation: signifier formulas in `utils/expression_signifiers.py`; weights in `utils/signifier_weights.py`; opportunity logic in `utils/b2b_opportunity_detector.py`.

---

## 8. Face Detection (MediaPipe vs Azure)

- **MediaPipe** (default): Local, 468 landmarks, no key required; lower latency, higher CPU.  
- **Azure Face API**: Cloud, 27 landmarks + emotions; key and endpoint required; fallback to MediaPipe after repeated empty results.  

**Config**: `FACE_DETECTION_METHOD=mediapipe` or `azure_face_api`; or set at runtime via `PUT /config/face-detection` or in the video source modal.  

**Unified mode**: When both are used, scores can be fused (config: `FUSION_AZURE_WEIGHT`, `FUSION_MEDIAPIPE_WEIGHT`).  

See `utils/mediapipe_detector.py`, `utils/azure_face_detector.py`, `utils/azure_landmark_mapper.py`.

---

## 9. Video Source Selection

- Modal appears when avatar session initializes and video is available.  
- **Options**: Webcam, Meeting Partner Video (stream + optional audio), Local Video File (upload).  
- **Face detection**: Choose MediaPipe or Azure Face API before Start Detection.  
- **File**: Upload via modal → `POST /engagement/upload-video` → path used in `POST /engagement/start`.  
- Implementation: `static/js/video-source-selector.js`; handlers in `index.html`.

---

## 10. Troubleshooting & Debug

### Starting and running

| Issue | Action |
|-------|--------|
| Launcher won’t start | Check Python on PATH; `python -c "import tkinter"` |
| Server won’t start | Check port 5000; stop other Python processes; see launcher logs |
| Port 5000 in use | Stop server, wait, or free port |

### Azure and AI

| Issue | Action |
|-------|--------|
| Failed to get speech token | Check `SPEECH_KEY`, `SPEECH_REGION`; verify Speech resource |
| Failed to get AI response | Check `AZURE_OPENAI_KEY`, endpoint, deployment name, API version |
| Speech recognition not working | Microphone permission; check browser console and Speech config |

### Avatar and video

| Issue | Action |
|-------|--------|
| Avatar video/audio not working | Initialize Avatar Session first; check relay token and WebRTC in console |
| Video source modal not showing | Ensure session init and video track; check console |

### Engagement

| Issue | Action |
|-------|--------|
| No face detected | Lighting, camera, face in frame; try MediaPipe if using Azure |
| Score stuck or low | `GET /engagement/debug`; check signifiers and metrics; verify video source |
| Azure Face API fails | Check key/endpoint/network; app falls back to MediaPipe after repeated empty |
| Wrong face detection method | Set in video source modal or `PUT /config/face-detection` |

### Debug endpoints

- **GET /engagement/state** — Score, metrics, signifiers, face detected.  
- **GET /engagement/debug** — Metrics history, detector state.  
- **GET /engagement/context** — Formatted context string for AI.  
- **GET /engagement/score-breakdown** — How the current score was computed.

---

## 11. Security & Best Practices

- **Secrets**: Do not commit API keys; use environment variables (or Key Vault) in production.  
- **CORS**: Restrict origins in production.  
- **Privacy**: Inform participants when engagement detection and/or recording is used.  
- **Code**: PEP 8; type hints; docstrings for public functions and classes; keep single-purpose functions.

---

## 12. Code References (key files)

| Area | Files |
|------|--------|
| App entry | `app.py`, `config.py` |
| Routes | `routes.py` |
| Engagement | `engagement_state_detector.py` |
| Signifiers & groups | `utils/expression_signifiers.py`, `utils/signifier_weights.py` |
| Scoring | `utils/engagement_scorer.py`, `utils/azure_engagement_metrics.py` |
| B2B opportunities | `utils/b2b_opportunity_detector.py` |
| Insights | `services/insight_generator.py` |
| Face detection | `utils/mediapipe_detector.py`, `utils/azure_face_detector.py` |
| Video source | `utils/video_source_handler.py`, `static/js/video-source-selector.js` |
| Frontend | `index.html`, `static/js/session-manager.js`, `static/js/engagement-detector.js` |
