# Business Meeting Copilot — Complete Project Documentation

This is the **single comprehensive guide** to the entire Business Meeting Copilot project. It explains what the project does, how it is organized, and how every major part works—in plain language so that even someone without a technical background can follow along.

---

## Table of Contents

Use these links to jump to any section.

1. [What Is This Project?](#1-what-is-this-project)
2. [How to Run the Project](#2-how-to-run-the-project)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Configuration (config.py)](#4-configuration-configpy)
5. [Application Entry Point (app.py)](#5-application-entry-point-apppy)
6. [Routes and API Endpoints (routes.py)](#6-routes-and-api-endpoints-routespy)
7. [Engagement State Detector](#7-engagement-state-detector)
8. [Services](#8-services)
9. [Utilities (utils)](#9-utilities-utils)
10. [Frontend (index.html and Static Files)](#10-frontend-indexhtml-and-static-files)
11. [Browser Extensions](#11-browser-extensions)
12. [Tests](#12-tests)
13. [Weights and Data Files](#13-weights-and-data-files)
14. [Scripts and Batch Files](#14-scripts-and-batch-files)

---

## 1. What Is This Project?

**Business Meeting Copilot** is an **AI meeting coach** that runs in your browser and on a small server. It helps someone running a meeting (for example, a sales call or a team discussion) by:

- **Watching** the meeting—either through your webcam or through a shared screen (e.g., the other person’s video in a call).
- **Listening** to what is said (speech-to-text) and how it is said (tone, pace).
- **Reading** faces and expressions to guess how engaged, confused, interested, or resistant people seem.
- **Suggesting** what to do next—for example: “They look confused; try clarifying,” or “They’re leaning in; good moment to ask for a commitment.”

In short: the app turns **video and audio** into **real-time engagement signals** and **coaching tips**, so the host can run better meetings.

### Main ideas

- **Engagement score (0–100):** A single number summarizing how “engaged” the person on camera appears (based on face, eyes, head movement, etc.).
- **Signifiers:** Thirty different facial and behavioral cues (e.g., eye contact, smile, frown, gaze away) that are combined to produce the score and to trigger specific insights.
- **Insights / popups:** Short, actionable messages (sometimes spoken aloud) that suggest what to do or what to watch for.
- **B2B focus:** The logic is tuned for business-to-business meetings (sales, negotiations, alignment) rather than casual chat.

---

## 2. How to Run the Project

1. **Install Python** (3.8 or newer) and install the project’s dependencies:
   - Open a terminal in the project folder and run:  
     `pip install -r requirements.txt`
2. **Start the server:**
   - **Windows:** Double-click `START_HERE.bat` or run `python app.py`.
   - **Mac/Linux:** Run `./START_HERE.sh` or `python app.py`.
3. **Open the app in your browser:**  
   Go to **http://localhost:5000** (or the port shown in the terminal).

The main page loads from `index.html`. From there you can start engagement detection (e.g., webcam or “partner” video), chat with the AI coach, and see insights and metrics.

---

## 3. Project Directory Structure

Here is what each folder and important file is for.

| Path | Purpose |
|------|--------|
| **app.py** | Entry point: starts the web server and loads the app. |
| **config.py** | All settings (API keys, URLs, feature flags, etc.). |
| **routes.py** | Defines every URL the server responds to (chat, engagement, config, etc.). |
| **engagement_state_detector.py** | Core logic: reads video, detects faces, computes engagement and alerts. |
| **control_panel.py** | Optional desktop control panel for the app. |
| **run_tests.py** | Runs the test suite. |
| **control_panel.py** | Optional desktop control panel (start/stop server, open browser, run tests). |
| **requirements.txt** | Python package list for `pip install`. |
| **docs/** | Documentation (this file). |
| **services/** | Backend “services”: Azure AI, speech, face API, insights, etc. |
| **utils/** | Reusable helpers: face detection, video handling, scoring, signifiers, etc. |
| **static/** | Web assets: dashboard HTML, JavaScript for engagement, chat, audio, etc. |
| **index.html** | Main single-page app (chat, engagement UI, video source selection). |
| **weights/** | JSON file with weights used to combine the 30 signifiers into scores. |
| **tests/** | Automated tests and test data. |
| **extension/** | Browser extension (generic). |
| **extension-chrome/** | Chrome extension. |
| **extension-firefox/** | Firefox extension. |
| **backup/** | Old copies of some modules (not used at runtime). |

---

## 4. Configuration (config.py)

`config.py` holds **all configurable options** for the project. Values can be overridden by **environment variables** (useful for production or different machines). Below, each block is explained in simple terms.

### 4.1 Azure AI Foundry (the “brain”)

- **What it is:** Microsoft’s Azure AI Foundry (successor to Azure OpenAI) provides the large language model (e.g., GPT) that powers the **chat and coaching responses**.
- **Main settings:**
  - **AZURE_FOUNDRY_KEY** – Secret key to call the API.
  - **AZURE_FOUNDRY_ENDPOINT** – Base URL of the service (e.g. `https://….openai.azure.com/`).
  - **FOUNDRY_DEPLOYMENT_NAME** – Name of the model deployment (e.g. `gpt-4o`).
  - **AZURE_FOUNDRY_API_VERSION** – API version string.
- A small helper **`_sanitize_azure_foundry_config()`** runs at load time to trim spaces and fix the endpoint URL. Older names like `AZURE_OPENAI_*` are still supported for compatibility.

### 4.2 Azure Speech Service

- **What it is:** Used for **speech-to-text (STT)** and **text-to-speech (TTS)** (e.g., reading insights aloud).
- **Main settings:**
  - **SPEECH_KEY**, **SPEECH_REGION** – Credentials and region.
  - **SPEECH_PRIVATE_ENDPOINT_ENABLED**, **SPEECH_PRIVATE_ENDPOINT** – For using a private endpoint instead of the public one.

### 4.3 Azure Cognitive Search

- **What it is:** Optional. Used for “On Your Data”–style features (searching your own documents to enrich answers).
- **Main settings:** **AZURE_COG_SEARCH_ENDPOINT**, **AZURE_COG_SEARCH_API_KEY**, **AZURE_COG_SEARCH_INDEX_NAME**. If these are not set, that feature is disabled.

### 4.4 Azure Face API

- **What it is:** Optional cloud service for **face and emotion detection**. Can be used instead of or together with MediaPipe (local).
- **Main settings:** **AZURE_FACE_API_KEY**, **AZURE_FACE_API_ENDPOINT**, **AZURE_FACE_API_REGION**.

### 4.5 Face detection (general)

- **FACE_DETECTION_METHOD** – How we detect faces: `"mediapipe"` (local), `"azure_face_api"` (cloud), `"auto"` (app chooses), or `"unified"` (both, combined).
- **MIN_FACE_CONFIDENCE** – Minimum confidence (0–1) to count a detection as a real face; lower = more tolerant in bad lighting.
- **LIGHTWEIGHT_MODE** – If true, use a lighter pipeline (e.g., fewer frames processed) to save CPU; still 720p.
- **TARGET_FPS_MIN** / **TARGET_FPS_MAX** – Target frame rate (e.g. 30–60 fps) for video processing.
- **DETECTION_WORKER_PROCESS** – If true, run detection in a separate process (advanced).
- **AUTO_DETECTION_SWITCHING** – If true and method is `"auto"`, the app can switch between Azure and MediaPipe based on device and latency.
- **AZURE_LATENCY_THRESHOLD_MS** – If Azure is slower than this (ms), prefer MediaPipe when in auto mode.
- **FUSION_AZURE_WEIGHT** / **FUSION_MEDIAPIPE_WEIGHT** – When using both Azure and MediaPipe, how much to weight each in the final score (e.g. 0.5 each).

### 4.6 Signifier and insight weights

- **SIGNIFIER_WEIGHTS_URL** – Optional URL to fetch signifier weights (30 numbers for signifiers, 4 for groups).
- **SIGNIFIER_WEIGHTS_PATH** – Local path (e.g. `weights/signifier_weights.json`) if no URL.
- **INSIGHT_WEIGHTS_URL** – Optional URL for insight-generation parameters (prompts, length, etc.).
- **SPEECH_CUE_LLM_FALLBACK_ENABLED** – Whether to use the LLM to classify phrases when regex doesn’t match (adds latency/cost).
- **SPEECH_CUE_LLM_FALLBACK_RATE_PER_MIN** – Max such LLM calls per minute.

### 4.7 Metric selection (which metrics run)

- **METRIC_SELECTOR_ENABLED** – If true, the app may reduce the number of metrics on weaker devices (high/medium/low tier).
- **METRIC_SELECTOR_OVERRIDE** – Force a tier: `"high"`, `"medium"`, or `"low"`.

### 4.8 Acoustic analysis

- **ACOUSTIC_ANALYSIS_ENABLED** – Whether to use voice/acoustic features (tone, etc.) from partner or mic.
- **ACOUSTIC_CONTEXT_MAX_AGE_SEC** – How long (seconds) acoustic context is considered “recent” for display and logic.

### 4.9 B2B opportunity and insight buffers

- **NEGATIVE_OPPORTUNITY_COOLDOWN_SEC** – Minimum time between “negative” opportunity insights (e.g. confusion, resistance).
- **POSITIVE_OPPORTUNITY_COOLDOWN_SEC** – Same for “positive” opportunities (e.g. closing, decision-ready).
- **MIN_CONCURRENT_FEATURES_NEGATIVE** / **MIN_CONCURRENT_FEATURES_POSITIVE** – How many cues (face, speech, acoustic) must agree before showing an insight (negative vs positive).
- **INSIGHT_BUFFER_SEC_NEGATIVE** – Minimum seconds between any **negative** insight popup (e.g. 8).
- **INSIGHT_BUFFER_SEC_POSITIVE** – Minimum seconds between any **positive** insight popup (e.g. 45).

### 4.10 Speech-to-Text / Text-to-Speech

- **STT_LOCALES** – Comma-separated locales for STT (e.g. en-US, de-DE).
- **TTS_VOICE** – Voice used for TTS (e.g. `en-US-AndrewMultilingualNeural`).
- **CUSTOM_VOICE_ENDPOINT_ID** – Optional custom voice endpoint.
- **CONTINUOUS_CONVERSATION** – Whether to keep conversation state across turns.

### 4.11 Avatar

- **AVATAR_CHARACTER**, **AVATAR_STYLE** – Which avatar and style to use.
- **PHOTO_AVATAR**, **CUSTOMIZED_AVATAR** – Use photo or custom avatar.
- **USE_BUILT_IN_VOICE**, **AUTO_RECONNECT_AVATAR**, **USE_LOCAL_VIDEO_FOR_IDLE**, **SHOW_SUBTITLES** – Avatar and UI options.

### 4.12 System prompt

- **SYSTEM_PROMPT** – A long text that defines the AI coach’s personality, expertise, and behavior. It tells the model to act as a meeting coach, when to speak vs. stay silent, how to use engagement levels, and how to format advice.

### 4.13 Application (Flask)

- **FLASK_PORT** – Port the server listens on (default 5000).
- **FLASK_DEBUG** – If true, use Flask’s built-in debug server with auto-reload; otherwise use Waitress.
- **FLASK_HOST** – Bind address (e.g. `0.0.0.0` for all interfaces).

### 4.14 Helper functions in config.py

- **is_cognitive_search_enabled()** – Returns True if Cognitive Search endpoint, key, and index are set.
- **get_cognitive_search_config()** – Returns a dict with Cognitive Search settings or `enabled: False`.
- **is_azure_face_api_enabled()** – Returns True if Face API key and endpoint are set.
- **get_azure_face_api_config()** – Returns Face API config dict or `enabled: False`.
- **get_face_detection_config()** – Returns face detection method and whether MediaPipe and Azure are available.
- **get_face_detection_config()** – Used by the API to expose detection options to the frontend.

---

## 5. Application Entry Point (app.py)

**app.py** is what you run to start the server (`python app.py`).

### What it does

1. **Imports** the Flask app framework, CORS, compression, and the **routes** (all URL handlers) from `routes.py`, plus `config`.
2. **`create_app()`**  
   - Creates a Flask application.  
   - Enables **CORS** so the browser can call the API from different origins.  
   - Enables **Compress** so responses (e.g. `/engagement/state`) can be gzip-compressed.  
   - Registers the **blueprint** that contains all API and page routes.  
   - Returns the configured app.
3. **Creates the app** once: `app = create_app()`.
4. **When run as main** (`if __name__ == "__main__"`):  
   - If **FLASK_DEBUG** is true: runs Flask’s built-in server (host, port, debug from config).  
   - Otherwise: uses **Waitress** (production-style server) with 6 threads.

So: **app.py** only creates and runs the web app; it does not define individual URLs. All URLs are defined in **routes.py**.

### 5.1 control_panel.py (optional)

**control_panel.py** is an **optional desktop helper** you can run with `python control_panel.py`. It opens a **small window** (using Tkinter) that lets you:

- **Start** the server (runs `python app.py` in the background).
- **Stop** the server (finds and terminates the server process).
- **Restart** the server (stop then start).
- **Open app in browser** (opens http://localhost:5000 in the default browser).
- **Run tests** and see results in the window, with simple **troubleshooting suggestions** if a test fails (e.g. “pip install -r requirements.txt”, “check API keys”).

It uses theme colors and a simple layout so non-technical users can start the app, run tests, or restart the server without using the command line. The project works fully without it; the control panel is only a convenience.

---

## 6. Routes and API Endpoints (routes.py)

**routes.py** defines every **URL** the server responds to and what each one does. Below, each route is listed with a short, plain-English description.

### 6.1 Pages and static files

- **GET /**  
  Serves the main page (**index.html**). This is the single-page app (chat, engagement, video source, etc.).

- **GET /dashboard**  
  Serves the **dashboard** page (e.g. **static/dashboard.html**), which can show engagement metrics and charts.

- **GET /static/<path:filename>**  
  Serves files from the **static** folder (e.g. JS, CSS). Used for dashboard and other assets.

- **GET /favicon.ico**  
  Sends the site’s favicon (or a 204 No Content if none).

### 6.2 Chat

- **POST /chat**  
  Sends a single user message to the AI (Azure Foundry) and returns one full response. Used for non-streaming chat.

- **POST /chat/stream**  
  Same as chat but the response is **streamed** (chunk by chunk), so the user sees text appearing in real time.

### 6.3 Speech and config (read-only)

- **GET /speech/token**  
  Returns a **token** and **region** for the Azure Speech Service so the browser can do STT/TTS without exposing the server’s key.

- **GET /config/speech**  
  Returns speech-related config (region, private endpoint, etc.).

- **GET /config/openai**  
- **GET /config/foundry**  
  Return Foundry/OpenAI config (endpoint, deployment, API version). Used so the frontend knows how to talk to the AI.

- **GET /config/cognitive-search**  
  Returns Cognitive Search config (enabled, endpoint, index, etc.).

- **GET /config/stt-tts**  
  Returns STT/TTS settings (locales, voice, etc.).

- **GET /config/avatar**  
  Returns avatar-related settings.

- **GET /config/system-prompt**  
  Returns the system prompt text (the “personality” of the coach).

- **GET /config/all**  
  Returns **all** configuration in one blob (used by the main UI to know what’s available).

### 6.4 Face detection and weights (config + overrides)

- **GET /config/face-detection**  
  Returns current face detection method and what’s available (MediaPipe, Azure).

- **PUT /config/face-detection**  
  Updates the face detection method (e.g. switch to MediaPipe or Azure).

- **GET /config/azure-face-api**  
  Returns Azure Face API config (enabled, endpoint, region).

- **GET /weights/signifiers**  
  Returns the current **signifier weights** (and group weights) used to compute engagement from the 30 signifiers.

- **PUT /weights/signifiers**  
  Updates signifier weights (e.g. from an external dashboard).

- **GET /weights/insight**  
  Returns insight-generation weights/parameters (prompt suffix, length, thresholds).

- **PUT /weights/insight**  
  Updates those parameters (e.g. for A/B testing or tuning).

### 6.5 Avatar

- **GET /avatar/relay-token**  
  Returns a token for the avatar/relay service so the client can drive the talking avatar.

### 6.6 Engagement detection (start, stop, video, transcript)

- **POST /engagement/start**  
  **Starts** engagement detection. Body can specify video source type (webcam, file, stream, partner) and optional path. The server creates or reuses an **EngagementStateDetector**, initializes the video source, and starts the detection loop in a background thread. Returns success/failure and the chosen detection method.

- **POST /engagement/upload-video**  
  Uploads a **video file** for use as the engagement source (file mode). Saves the file and can then be used with source type “file” and that path.

- **POST /engagement/partner-frame**  
  Receives **one frame** (image) from the browser (e.g. from a shared screen or partner video). The frame is stored and used by the “partner” video source so the detector can analyze the other person’s face.

- **POST /engagement/transcript**  
  Sends **transcript** text (e.g. from live STT of the meeting). Stored and used for speech-based cues and for insight generation (e.g. “they said ‘I’m not sure’ → confusion”).

- **POST /engagement/acoustic-context**  
  Sends **acoustic** features (pitch, energy, etc.) from the partner or mic. Used for “voice tone” cues (e.g. uncertainty, tension) in addition to words.

- **GET /engagement/video-feed**  
  **Streams** the current engagement video (the last processed frame, as JPEGs in a multipart stream). The browser can show this as a live feed (e.g. 30–60 fps).

- **POST /engagement/stop**  
  **Stops** engagement detection: stops the loop, releases the video source, clears transcript and related state.

### 6.7 Engagement state and insights (main consumer API)

- **GET /engagement/debug**  
  Returns **debug** info: whether the detector is running, method, FPS, whether there is state, consecutive “no face” count, etc.

- **GET /engagement/state**  
  **Most important** endpoint for the main UI. It:
  - Returns current **engagement state**: score, level, face_detected, metrics (attention, eye contact, etc.), context (summary, indicators, actions), signifier scores, Azure metrics if used, composite metrics, FPS, detection method.
  - If there is a **pending alert** (spike, opportunity, or aural) and the insight buffer allows it, the response also includes that alert and a suggested **insight text** (sometimes generated by Azure Foundry). The client can show this as a popup and/or play it with TTS. After the client uses it, a later call can record that the insight was shown (so buffers are respected).
  - Builds **context** from the current state for the AI coach (so chat can say things like “engagement is high right now”).

- **GET /api/engagement/context-and-response**  
  Returns **context and response** data used by the UI to show “what the coach sees” and a suggested response (often from the same context used for chat).

- **POST /api/engagement/record-response**  
  Records that the user **acknowledged or used** a suggested response (for analytics or tuning).

- **POST /api/engagement/set-additional-context**  
  Sets **additional context** text that is included when building prompts for the AI (e.g. “focus on pricing objections”).

- **POST /api/context-push**  
- **GET /api/context-push**  
  Used to **push** or **retrieve** context (e.g. for extensions or external tools).

- **GET /engagement/score-breakdown**  
  Returns a **step-by-step breakdown** of how the current engagement score was computed (for debugging and transparency).

- **GET /engagement/context**  
  Returns the **engagement context** (summary, level description, indicators, suggested actions, risks, opportunities) in a format suitable for display or for the AI.

### 6.8 Internal helpers in routes.py

- **`_debug_log(data)`**  
  Writes debug log entries (e.g. NDJSON) for troubleshooting; used in a few places in the engagement/insight flow.

- **`_get_context_generator()`**  
  Returns the shared **ContextGenerator** instance used to build human-readable context from engagement state.

- **`_build_engagement_context_bundle(additional_context)`**  
  Builds a **text bundle** of current engagement state and optional extra context, used as part of the prompt for chat and insights.

- **`_build_fresh_insight_context(...)`**  
  Builds the **context** passed to the insight generator when creating popup/insight text (state, metrics, speech tags, acoustic tags, etc.).

The rest of the logic in **routes.py** is the actual implementation of each route above (reading body, calling services, formatting JSON, handling errors).

---

## 7. Engagement State Detector

The **EngagementStateDetector** (in **engagement_state_detector.py**) is the **core component** that turns **video (and optionally transcript/acoustic)** into a single **engagement state** and **alerts**. Everything else (API, insights, chat) depends on this.

### 7.1 Main concepts

- **Video source:** Frames come from a **VideoSourceHandler** (webcam, file, stream, or “partner” — i.e. frames pushed by the browser).
- **Face detection:** Each frame is passed to a **face detector** (MediaPipe and/or Azure). The detector returns faces with **landmarks** (points on the face) and optionally **emotions** (Azure).
- **Signifiers:** From landmarks (and optional emotions), an **ExpressionSignifierEngine** computes **30 signifier scores** (0–100), e.g. eye contact, smile, gaze aversion, brow furrow.
- **Groups:** Those 30 are grouped into **4 groups (G1–G4)** and averaged to get group means. Groups represent things like “interest” (G1), “thinking” (G2), “resistance” (G3), “readiness” (G4).
- **Score:** An **EngagementScorer** combines metrics (attention, eye contact, expressiveness, etc.) into one **0–100** score and an **EngagementLevel** (e.g. VERY_LOW … VERY_HIGH).
- **Context:** A **ContextGenerator** turns the current state into **text** (summary, indicators, suggested actions) for the AI and for display.
- **Alerts:** The detector looks for **spikes** (sudden rises in a group), **composite_100** (very high composite), and **B2B opportunities** (e.g. “closing window,” “confusion”). When one fires and the **insight buffer** allows it, it becomes a **pending alert** that **GET /engagement/state** can return and turn into a popup + TTS.

### 7.2 Classes and data structures

- **EngagementLevel (Enum)**  
  One of: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH.  
  **`from_score(score)`** maps a 0–100 score to a level (e.g. 85+ → VERY_HIGH).

- **EngagementState (dataclass)**  
  Holds: **score**, **level**, **metrics** (attention, eye_contact, etc.), **context** (EngagementContext), **timestamp**, **face_detected**, **confidence**, **signifier_scores**, **detection_method**, **azure_metrics**, **composite_metrics**. This is what the API returns as “current state.”

- **EngagementStateDetector (class)**  
  - **`__init__(...)`**  
    Sets up: face detector (from config or preference), video handler, scorer, context generator, feature extractor, signifier engine, buffers and cooldowns for spikes/insights, FPS targets, and locks for thread safety.  
  - **`start_detection(source_type, source_path)`**  
    Resets state, initializes the video source (webcam/file/stream/partner), optionally warms up the camera, then starts **`_detection_loop`** in a **background thread**. Returns True on success.  
  - **`stop_detection()`**  
    Stops the loop, joins the thread, releases the video source and face detector.  
  - **`get_current_state()`**  
    Returns the latest **EngagementState** (thread-safe).  
  - **`get_fps()`**  
    Returns the current processing frame rate (averaged over recent frames).  
  - **`get_last_frame_jpeg()`**  
    Returns the latest frame as JPEG bytes for the video feed (thread-safe, copy-on-read).  
  - **`_detection_loop()`**  
    **Main loop** (runs in thread):  
    - Reads a frame from the video source.  
    - Stores it for the video feed.  
    - Updates the signifier engine’s FPS (for time-based thresholds).  
    - Optionally skips frames (lightweight mode, idle throttle, or adaptive throttle when score is stable).  
    - Calls **`_process_frame(frame)`** to get a new **EngagementState**.  
    - Updates **current_state**, tracks FPS, and optionally sleeps to cap at target max FPS.  
  - **`_process_frame(frame)`**  
    - Runs face detection.  
    - If Azure and repeatedly no faces, can fall back to MediaPipe.  
    - Extracts landmarks; if “unified,” may merge Azure emotions with MediaPipe landmarks.  
    - Runs **ExpressionSignifierEngine** to get 30 signifier scores.  
    - Optionally runs Azure secondary detector and **engagement_composites** for composite metrics.  
    - Runs **EngagementScorer** to get metrics and score.  
    - Runs **ContextGenerator** to get context.  
    - Runs **B2B opportunity detector** and spike/composite_100 logic to set a **pending alert** if conditions and buffers allow.  
    - Returns an **EngagementState** or a “no face” state.  
  - **`can_show_insight(alert)`**  
    True if enough time has passed since the last insight (using negative vs positive buffer).  
  - **`record_insight_shown(alert)`**  
    Called when an insight popup was shown; updates last-shown time.  
  - **`get_pending_alert()`** / **`get_and_clear_pending_alert()`** / **`clear_pending_alert()`**  
    Thread-safe access to the pending spike/opportunity/aural alert.  
  - **`get_score_breakdown()`**  
    Returns a step-by-step score breakdown for debugging.
  - **`_process_frame(frame)`** (internal)  
    Runs face detection, signifier engine, optional Azure secondary and composites, scorer, context generator, then spike/opportunity/composite_100 checks; returns **EngagementState** or no-face state.
  - **`_metrics_from_signifiers(s)`** (internal)  
    Builds **EngagementMetrics** from the 30 signifier scores when not using Azure.
  - **`_metrics_from_azure(azure_result)`** (internal)  
    Builds **EngagementMetrics** from Azure base/composite metrics when using Azure.
  - **`_make_no_face_state()`** (internal)  
    Returns an **EngagementState** for “no face detected” (low score, no signifiers).
  - **`_count_concurrent_features(alert, ...)`** (internal)  
    Counts how many facial, speech, and acoustic features support the alert (used to enforce MIN_CONCURRENT_FEATURES).
  - **`_check_composite_at_100(...)`** (internal)  
    If composite metrics hit 100 and cooldown allows, sets a “composite_100” pending alert.
  - **`_check_spike_alerts(group_means, now)`** (internal)  
    Detects sudden rises in group means (G1–G4) and sets a “spike” pending alert if thresholds and cooldowns allow.
  - **`_check_opportunity_alerts(...)`** (internal)  
    Calls the B2B opportunity detector; if one fires and buffers allow, sets an “opportunity” pending alert.
  - **`_is_positive_alert(alert)`** (internal)  
    Returns True if the alert is positive (spike, composite_100, or a positive opportunity ID).
  - **`_average_metrics()`** (internal)  
    Averages recent metrics history for smoothing.
  - **`_calculate_confidence()`** (internal)  
    Computes a 0–1 confidence for the current state (e.g. from face detection confidence and stability).

So: **engagement_state_detector.py** ties together **video → faces → signifiers → groups → score + context + alerts**, and exposes that as **current state** and **pending alert** for the rest of the app.

---

## 8. Services

**Services** are backend modules that talk to **external APIs** (Azure) or manage **shared state** (insights, acoustic context, request tracking). Each file and its main functions are described below.

### 8.1 services/azure_foundry.py

- **What it does:** Calls **Azure AI Foundry** (OpenAI-compatible) for **chat completions** (single and streaming).
- **AzureFoundryService (class)**  
  - **`chat_completion(messages, ...)`**  
    Sends a list of messages to the model and returns one completion.  
  - **`stream_chat_completion(messages, ...)`**  
    Same but yields chunks for streaming.  
- **`get_foundry_service()`**  
  Returns the singleton **AzureFoundryService** instance.  
- **`get_openai_service()`**  
  Alias for **get_foundry_service()** (backward compatibility).

### 8.2 services/azure_speech.py

- **What it does:** Wraps **Azure Speech Service** for **token** retrieval (so the browser can do STT/TTS).
- **AzureSpeechService (class)**  
  - **`get_speech_token()`**  
    Returns **token** and **region** for the client.  
- **`get_speech_service()`**  
  Returns the singleton **AzureSpeechService** instance.

### 8.3 services/azure_face_api.py

- **What it does:** Calls **Azure Face API** to detect faces (and optionally landmarks and attributes) in an image.
- **AzureFaceAPIService (class)**  
  - **`detect_faces(image_bgr, ...)`**  
    Sends the image to the Face API and returns a list of face results (e.g. rectangle, landmarks, emotions).  
  - **`test_connection()`**  
    Sends a small test image to check connectivity.  
- **`get_azure_face_api_service()`**  
  Returns the singleton if the API is configured, else None.

### 8.4 services/insight_generator.py

- **What it does:** Generates **insight text** for popups (spike, aural trigger, or B2B opportunity) and manages **speech tags**, **transcript**, and **pending aural alert**.
- **Main functions:**  
  - **`get_insight_weights()`** / **`set_insight_weights(data)`**  
    Get/set parameters for insight generation (prompt suffix, length, thresholds).  
  - **`check_speech_cues(text)`**  
    Checks transcript text for trigger phrases and discourse markers; returns list of (category, phrase).  
  - **`append_speech_tag(category, phrase)`**  
    Stores a speech tag with timestamp (used for “they said something that suggests X”).  
  - **`get_recent_speech_tags(within_sec)`**  
    Returns recent tags (for context and composites).  
  - **`clear_recent_speech_tags()`**  
    Clears the list.  
  - **`get_pending_aural_alert()`** / **`get_and_clear_pending_aural_alert()`** / **`clear_pending_aural_alert()`**  
    Manage the “phrase-triggered” alert (e.g. “they said ‘I’m not sure’ → confusion”).  
  - **`append_transcript(text)`** / **`get_recent_transcript()`** / **`clear_transcript()`**  
    Manage the recent transcript buffer.  
  - **`generate_insight_for_spike(alert, context_bundle, ...)`**  
    Generates short insight text for a **spike** or **composite_100** alert (optionally via Azure Foundry).  
  - **`generate_insight_for_aural_trigger(alert, context_bundle, ...)`**  
    Same for **aural** (phrase-triggered) alerts.  
  - **`generate_insight_for_opportunity(alert, context_bundle, ...)`**  
    Same for **B2B opportunity** alerts.  
- **Internal helpers** (e.g. **`_check_for_trigger_phrases`**, **`_build_rich_context_parts`**) support phrase matching, discourse alignment, and building the context string for the LLM.

### 8.5 services/acoustic_context_store.py

- **What it does:** Stores **acoustic windows** (e.g. pitch, energy) from the frontend and exposes recent **tags** and **negative strength** for insights and composites.
- **Functions:**  
  - **`append_acoustic_windows(windows)`**  
    Appends new windows to the store.  
  - **`get_recent_acoustic_context()`**  
    Returns a string summary of recent acoustic context (for prompts).  
  - **`get_recent_acoustic_tags()`**  
    Returns recent tags (e.g. “uncertainty,” “tension”).  
  - **`get_acoustic_negative_strength()`**  
    Returns a strength score for “negative” acoustic cues.  
  - **`clear_acoustic_context()`**  
    Clears the store.

### 8.6 services/engagement_request_tracker.py

- **What it does:** Tracks **when the last request** to **GET /engagement/state** (or similar) happened, so the detector can slow down when nobody is polling (idle).
- **Functions:**  
  - **`update_last_request()`**  
    Call this when /engagement/state is requested.  
  - **`get_last_request_time()`**  
    Returns timestamp of last request.  
  - **`is_idle(threshold_sec)`**  
    Returns True if no request has occurred for at least **threshold_sec** seconds.

### 8.7 services/detection_worker.py

- **What it does:** Optional **separate process** for running engagement detection (so heavy CPU doesn’t block Flask). Not required for normal operation.
- **`should_use_worker_process()`**  
  Returns True if config says to use the worker process.  
- **DetectionWorkerProcess (class)**  
  Manages the worker process and queues for sending frames and receiving state. Used only when **DETECTION_WORKER_PROCESS** is True.

---

## 9. Utilities (utils)

**utils** contains reusable logic: video handling, face detection, signifiers, scoring, B2B opportunities, context, composites, weights, and system capability. Below is a concise overview of each module and its main subroutines.

### 9.1 utils/video_source_handler.py

- **What it does:** **Single interface** for reading video frames from: webcam, file, stream URL, or “partner” (browser-pushed frames).
- **set_partner_frame(frame_bgr)**  
  Sets the latest frame received from the browser (partner source).  
- **get_partner_frame()**  
  Returns a copy of the latest partner frame, or None.  
- **has_partner_frame()**  
  True if a partner frame is available.  
- **set_partner_frame_from_bytes(image_bytes)**  
  Decodes image bytes (e.g. JPEG) to a frame and sets it as the partner frame; may resize if too wide.  
- **VideoSourceType (Enum)**  
  WEBCAM, FILE, STREAM, PARTNER.  
- **VideoSourceHandler (class)**  
  - **`initialize_source(source_type, source_path, lightweight)`**  
    Opens the source (e.g. OpenCV for webcam/file/stream; partner uses no OpenCV). Webcam is set to 720p and 30 or 60 fps.  
  - **`read_frame()`**  
    Returns (True, frame) or (False, None). For partner, reads from the shared partner frame.  
  - **`get_properties()`**  
    Returns dict with **width**, **height**, **fps**, **frame_count** (or 0/0/0/-1 when no source).  
  - **`set_frame_position(frame_number)`**  
    For file source only: seeks to frame number.  
  - **`release()`**  
    Closes the source and clears partner frame.

### 9.2 utils/expression_signifiers.py

- **What it does:** Computes **30 expression signifier scores** (0–100) from face **landmarks** (and optional Azure emotions), using a **temporal buffer** so that “sustained” behaviors (e.g. long eye closure, gaze aversion) are scored correctly. Used at 30–60 fps; thresholds scale with FPS.
- **Helpers:**  
  - **`_safe(landmarks, indices, dim)`** – Safe slice of landmark points.  
  - **`_ear(pts)`** – Eye Aspect Ratio from eye points.  
  - **`_median_recent(buf, key, n, default)`** – Median of last n values in buffer.  
  - **`_normalize_lm(landmarks, w, h)`** – Normalize landmark coordinates to image size.  
- **ExpressionSignifierEngine (class)**  
  - **`__init__(buffer_frames, weights_provider)`**  
    Sets up buffer, weights provider, and FPS reference (30).  
  - **`set_fps(fps)`**  
    Sets current FPS so time-based frame thresholds scale (e.g. 30–60 fps).  
  - **`reset()`**  
    Clears buffer and baselines.  
  - **`update(landmarks, face_result, frame_shape)`**  
    Appends one frame’s features (EAR, MAR, pitch, yaw, etc.) to the buffer and updates blink count.  
  - **`get_all_scores()`**  
    Returns a dict of all 30 signifier names → scores.  
  - Many **internal methods** (e.g. **_g1_eye_contact**, **_g3_eye_block**, **_g3_gaze_aversion**) implement each signifier from the buffer; they use **FPS-scaled** frame counts so durations (e.g. 400 ms) are consistent across frame rates.

### 9.3 utils/engagement_scorer.py

- **What it does:** Combines **metrics** (attention, eye contact, expressiveness, head movement, symmetry, mouth activity) into one **0–100 engagement score**.
- **EngagementMetrics (dataclass)**  
  Holds: attention, eye_contact, facial_expressiveness, head_movement, symmetry, mouth_activity.  
- **EngagementScorer (class)**  
  - **`score(signifier_scores, metrics_dict)`**  
    Computes overall score and level from signifier scores and optional metrics dict.  
  - **`compute_metrics(signifier_scores)`**  
    Derives the six metrics from the 30 signifiers (e.g. attention from EAR, activity from MAR).

### 9.4 utils/context_generator.py

- **What it does:** Turns **current engagement state** into **human-readable context** (summary, level description, key indicators, suggested actions, risks, opportunities) for the AI and for the UI.
- **EngagementContext (dataclass)**  
  Holds: summary, level_description, key_indicators, suggested_actions, risk_factors, opportunities.  
- **ContextGenerator (class)**  
  - **`generate_context(score, level, metrics, signifier_scores, ...)`**  
    Returns an **EngagementContext** filled with text tailored to the current state.

### 9.5 utils/face_detection_interface.py

- **What it does:** **Abstract interface** for face detectors so the rest of the app doesn’t care whether we use MediaPipe or Azure.
- **FaceDetectionResult**  
  Dataclass: **landmarks** (array), **bounding_box** (optional), **emotions** (optional dict), etc.  
- **FaceDetectorInterface (ABC)**  
  - **`detect_faces(frame)`** → list of **FaceDetectionResult**.  
  - **`get_name()`** → string (e.g. "mediapipe", "azure_face_api").  
  - **`close()`** – cleanup.

### 9.6 utils/mediapipe_detector.py

- **What it does:** **Face detection** using **MediaPipe** (local, no cloud). Returns landmarks and bounding box.
- **MediaPipeFaceDetector(FaceDetectorInterface)**  
  Implements **detect_faces**, **get_name**, **close**. Uses MediaPipe Face Mesh.

### 9.7 utils/azure_face_detector.py

- **What it does:** **Face detection** using **Azure Face API** (cloud). Uses **AzureFaceAPIService** to get faces and maps results to **FaceDetectionResult** (landmarks, emotions, etc.).
- **AzureFaceAPIDetector(FaceDetectorInterface)**  
  Implements **detect_faces**, **get_name**, **close**.

### 9.8 utils/azure_landmark_mapper.py

- **What it does:** **Maps Azure Face API landmarks** to a format compatible with **MediaPipe** (same point indices), so the same signifier engine can run on both.
- **expand_azure_landmarks_to_mediapipe(landmarks, bbox, frame_shape)**  
  Returns a landmark array in MediaPipe order; used when we have Azure landmarks but want to run the MediaPipe-based signifier pipeline.

### 9.9 utils/azure_engagement_metrics.py

- **What it does:** **Engagement metrics from Azure Face API** (emotions, head pose). Used when we use Azure for face; can be combined with MediaPipe signifiers in “unified” mode.
- **compute_base_metrics(face_result)**  
  Returns a dict of base metrics from Azure (e.g. emotion scores).  
- **compute_composite_metrics(face_result)**  
  Returns composite emotion-based metrics.  
- **compute_azure_engagement_score(face_result)**  
  Returns a 0–100 score from Azure data.  
- **get_all_azure_metrics(face_result)**  
  Returns full set of Azure metrics (base, composite, score) for the API.

### 9.10 utils/b2b_opportunity_detector.py

- **What it does:** **Detects “B2B opportunities”** from the four group means (G1–G4), history, speech tags, and acoustic tags—e.g. “closing window,” “confusion,” “decision-ready.” Each opportunity has a cooldown and optional conditions (e.g. “must have recent speech tag in category X”).
- **NEGATIVE_OPPORTUNITY_IDS** / **POSITIVE_OPPORTUNITY_IDS**  
  Sets of opportunity IDs used for cooldowns and buffers (negative vs positive).  
- **detect_opportunity(group_means, speech_tags, acoustic_tags, ...)**  
  Evaluates opportunities in priority order; returns the first that fires and has passed cooldown, or None.  
- **update_history_from_detector(group_means)**  
  Updates internal history used for trend-based opportunities.  
- **clear_opportunity_state()**  
  Clears cooldowns and history (e.g. when engagement stops).  
- Many **`_eval_*`** functions (e.g. **_eval_closing_window**, **_eval_confusion_moment**) implement one opportunity each (conditions on G1–G4, history, speech, acoustic).

### 9.11 utils/engagement_composites.py

- **What it does:** **Composite metrics** that combine **facial signifiers**, **speech tags**, and **acoustic tags** (e.g. “confusion” = face + “I don’t understand” + uncertain voice). Used for richer insights and for “composite_100” alerts.
- **compute_composite_metrics(state, speech_tags, acoustic_tags, ...)**  
  Returns a dict of composite names → 0–100 scores.  
- Helpers **`_has_category`**, **`_category_strength`** check and weight recent speech/acoustic tags.

### 9.12 utils/business_meeting_feature_extractor.py

- **What it does:** **Extracts** from landmarks and face result the **per-frame features** (EAR, MAR, pitch, yaw, roll, gaze, etc.) that the signifier engine and scorer need.
- **BusinessMeetingFeatureExtractor (class)**  
  - **`extract(frame, face_result, landmarks)`**  
    Returns a dict of feature name → value used by the signifier engine’s **update** and by scoring.

### 9.13 utils/signifier_weights.py

- **What it does:** **Loads and caches** the **signifier weights** (30 for signifiers, 4 for groups) from **config** (URL or local path). Provides a **weights_provider** callable for the signifier engine.
- **get_weights()**  
  Returns the current weights dict (signifier list, group list, optional fusion).  
- **load_weights()**  
  Loads from URL or file; applies and caches.  
- **set_weights(...)**  
  Updates in-memory weights (and optionally persists).  
- **get_fusion_weights()** / **set_fusion_weights(azure, mediapipe)**  
  Get/set Azure vs MediaPipe fusion weights when using both.  
- **build_weights_provider()**  
  Returns a callable that returns the current weights (for the engine).

### 9.14 utils/metric_selector.py

- **What it does:** **Chooses which metrics (signifiers)** run based on **device tier** (high/medium/low) from CPU and RAM, so weaker devices run fewer metrics.
- **MetricConfig**  
  Holds: list of active signifier keys, buffer size, etc.  
- **get_active_metrics()**  
  Returns the list of signifier keys to run for the current tier.  
- **get_active_metrics_with_config()**  
  Returns the full **MetricConfig** (used by the detector).  
- ** _get_system_resources()**, **_determine_tier()**, **_build_config(tier)**  
  Internal: get CPU/RAM, map to tier, build config.

### 9.15 utils/detection_capability.py

- **What it does:** **Evaluates device capability** (CPU cores, RAM) and **Azure Face API latency** to recommend **face detection method** (e.g. MediaPipe vs Azure) when in “auto” mode.
- **get_cpu_count()** / **get_memory_gb()**  
  Return CPU count and RAM in GB (or None if unavailable).  
- **get_device_tier()**  
  Returns "high", "medium", or "low".  
- **get_azure_latency_ms(...)**  
  Measures round-trip latency to Azure Face API.  
- **recommend_detection_method(...)**  
  Returns recommended method (e.g. "mediapipe", "unified") based on tier and latency.  
- **evaluate_capability()**  
  Returns a summary dict of capability and recommendation.

### 9.16 utils/face_detection_preference.py

- **What it does:** **Stores and retrieves** the user’s (or app’s) **preferred face detection method** (e.g. from config or API).
- **get_face_detection_method()**  
  Returns current method string.  
- **set_face_detection_method(method)**  
  Sets and returns the method.

### 9.17 utils/helpers.py

- **What it does:** **Builds the “config/all” response** so the frontend gets one big config object.
- **build_config_response()**  
  Returns a dict with speech, foundry, openai, cognitiveSearch, sttTts, avatar, systemPrompt, faceDetection, signifierWeights, azureFaceApi, acoustic. Used by **GET /config/all**.

### 9.18 utils/acoustic_interpreter.py

- **What it does:** **Interprets** raw **acoustic windows** (from the frontend) into **tags** (e.g. uncertainty, tension, disengagement) and passes them to the acoustic context store.
- **interpret_acoustic_windows(windows)**  
  Processes a list of window dicts and returns or stores interpreted tags (exact signature depends on implementation).

### 9.19 utils/detection_capability.py

- Already described above under **detection_capability**.

---

## 10. Frontend (index.html and Static Files)

### 10.1 index.html

**index.html** is the **main single-page application**. It typically includes:

- **Chat area:** Text box and send button; messages and AI responses (streaming or not).
- **Engagement section:** Button to start/stop engagement; choice of video source (webcam, partner/screen share, file); place to show the **engagement video feed** and **score/level**.
- **Insights / popups:** Area or overlay where **insight messages** appear (and optionally play as TTS).
- **Context/response:** Sometimes a “suggested response” or “what the coach sees” based on **GET /api/engagement/context-and-response**.
- **Scripts:** References to **static JS** modules (session manager, engagement detector, chat, avatar, video source selector, acoustic analyzer, etc.).

So **index.html** is the “shell”; the **behavior** lives in the **static JavaScript** files and in the **routes** that serve data.

### 10.2 static/dashboard.html

**dashboard.html** is a **separate dashboard page** (opened via **/dashboard**). It usually:

- Polls **GET /engagement/state** (or a similar endpoint) on an interval.
- Displays **engagement score**, **level**, **FPS**, **face detected**, and sometimes **signifier** or **composite** metrics in panels or charts.
- May show the **engagement video feed** (from **GET /engagement/video-feed**).

So it’s a **metrics-focused** view for monitoring engagement in real time.

### 10.3 static/js/*.js

- **session-manager.js**  
  Manages **session** and **API base URL**; may handle **chat** and **streaming** requests to **/chat** and **/chat/stream**, and **context-and-response** to **/api/engagement/context-and-response**.

- **engagement-detector.js**  
  - **Start/stop** engagement via **POST /engagement/start** and **POST /engagement/stop**.  
  - **Poll** **GET /engagement/state** on an interval.  
  - When **partner** source is selected, **capture** frames (e.g. from a shared screen or video element) and **POST** them to **/engagement/partner-frame** at ~30 fps.  
  - Show **alerts/insights** from the state response (popup, optional TTS).  
  - May update **signifier** or **composite** panels from the state payload.

- **avatar-chat-manager.js**  
  Handles **avatar**-related UI and possibly **relay token** from **GET /avatar/relay-token** for the talking avatar.

- **video-source-selector.js**  
  UI for choosing **video source** (webcam, file, stream, partner) and sending that choice when starting engagement.

- **acoustic-analyzer.js**  
  Gets **audio** (e.g. from partner or mic), computes **acoustic features** (or uses an external library), and **POST**s them to **/engagement/acoustic-context** at an interval.

- **signifier-panel.js** / **azure-metrics-panel.js** / **composite-metrics-panel.js**  
  Display **signifier scores**, **Azure metrics**, and **composite metrics** from the engagement state in panels or lists.

Together, these scripts implement **chat**, **engagement control**, **video source**, **partner frame upload**, **acoustic upload**, **state polling**, and **insight/alert display** on the main page and dashboard.

---

## 11. Browser Extensions

The project includes **browser extensions** so the app can interact with the user’s **browser** (e.g. get the current tab URL or share a tab’s video as the “partner” source).

- **extension/**  
  Generic or shared extension code (manifest, popup, sidepanel, background script).
- **extension-chrome/**  
  Chrome-specific extension (manifest, icons, popup, sidepanel, background).
- **extension-firefox/**  
  Firefox-specific extension (similar structure).

Typical behavior:

- **Popup:** Small UI (e.g. “Start copilot,” “Open app”).
- **Sidepanel:** Can show a panel that talks to the app (e.g. send **tab URL** or **stream** to the app).
- **Background script:** Listens for events, may pass messages between the page and the app (e.g. **http://localhost:5000**).

The **exact** flow (how the app gets the “partner” video) depends on the app’s design: often the **main page** (index.html) uses **getDisplayMedia** to share a tab/window and then **engagement-detector.js** sends those frames to **POST /engagement/partner-frame**. The extensions may add **convenience** (e.g. open the app with the current tab URL or inject a script). Documentation in the extension folders (if any) would describe their specific role.

---

## 12. Tests

- **run_tests.py**  
  **Orchestrates** the test suite (discovers and runs tests, prints results).
- **tests/**  
  - **test_api_endpoints.py**  
    Tests for **API routes** (e.g. /config, /engagement/start, /engagement/state) to ensure they return expected status and shape.  
  - **test_metrics_validation.py**  
    Tests for **metrics and signifiers** (ranges, formulas, edge cases).  
  - **test_services.py**  
    Tests for **services** (e.g. Azure Foundry, speech, insight generator) where possible without live keys.  
  - **test_utils.py**  
    Tests for **utils** (e.g. video handler, signifier engine, scorer, context generator).  
- **tests/fixtures/**  
  - **synthetic_landmarks.py**  
    **Fake landmark data** (e.g. neutral, smile, gaze aversion) used so tests don’t need real video.  
  - **README** (if present) described how to use fixtures.

Running **python run_tests.py** runs all of these and reports pass/fail. This helps ensure that changes to config, routes, detector, services, or utils don’t break the pipeline.

---

## 13. Weights and Data Files

- **weights/signifier_weights.json**  
  **JSON file** that contains:  
  - **signifier:** List of 30 numbers (weights for each signifier).  
  - **group:** List of 4 numbers (weights for the four groups G1–G4).  
  - Optionally **fusion** (e.g. azure/mediapipe weights).  

  The **ExpressionSignifierEngine** and **EngagementScorer** use these to combine the 30 scores into group means and then into the final engagement score. Editing this file (or overriding via **PUT /weights/signifiers** or config URL) changes how much each cue affects the result.

---

## 14. Scripts and Batch Files

- **START_HERE.bat** (Windows)  
  Typically runs **python app.py** (or sets up the environment and then runs it) so the user can start the server with one double-click.  
- **START_HERE.sh** (Mac/Linux)  
  Same idea: run the app (e.g. **python app.py**).  
- **restart_server.bat**  
  Stops and restarts the server (implementation may kill the process and run **app.py** again).  
- **stop_server.bat**  
  Stops the server (e.g. kill the Python process listening on the configured port).  
- **run_tests.bat**  
  Runs **python run_tests.py** so you can run the test suite with one click.

These are **convenience** scripts; the real behavior is in **app.py**, **routes.py**, and **run_tests.py**.

---

## Summary

- **Business Meeting Copilot** is an **AI meeting coach**: it uses **video** (webcam or partner) and **audio/transcript** to estimate **engagement** and **B2B opportunities**, and to produce **coaching insights** (popups, TTS, and chat).
- **config.py** centralizes all settings; **app.py** starts the server; **routes.py** defines every URL and wires them to the **engagement detector**, **services** (Foundry, speech, face API, insights, acoustic store), and **config**.
- The **EngagementStateDetector** is the core: it pulls **frames** from a **VideoSourceHandler**, runs **face detection** (MediaPipe and/or Azure), computes **30 signifiers** and **4 group means**, then **score**, **context**, and **alerts** (spikes, opportunities, aural). The **GET /engagement/state** endpoint exposes this as the main API for the UI.
- **Services** handle Azure and shared state; **utils** handle video, faces, signifiers, scoring, context, B2B opportunities, composites, weights, and device capability. The **frontend** (index.html and static JS) drives chat, engagement start/stop, video source, partner frames, acoustic upload, polling, and display of metrics and insights.
- **Tests** cover API, metrics, services, and utils; **weights/signifier_weights.json** configures how signifiers combine into the final score. **Scripts** (.bat, .sh) are for starting/stopping the server and running tests.

This document is the **single comprehensive reference** for the project. Use the **Table of Contents** at the top to jump to any section. For the very first time, read **§1 What Is This Project?** and **§2 How to Run the Project**, then **§3 Project Directory Structure**; after that, use the TOC to dive into any area you care about (config, routes, detector, services, utils, frontend, extensions, tests, or weights).
