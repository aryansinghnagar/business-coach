# Business Meeting Copilot — Complete Project Documentation

This is the **single comprehensive guide** to the entire Business Meeting Copilot project. It lives at the **project root** (`README.md`). It explains what the project does, how it is organized, and how every major part works—in plain language so that even someone without a technical background can follow along.

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
9. [Helpers (helpers.py)](#9-helpers-helperspy)
10. [Frontend (index.html and Static Files)](#10-frontend-indexhtml-and-static-files)
11. [Browser Extensions](#11-browser-extensions)
12. [Tests](#12-tests)
13. [Scripts and Batch Files](#13-scripts-and-batch-files)
14. [Composite Metrics (Detailed)](#14-composite-metrics-detailed)
15. [Metrics Reference (Formulas and Indices)](#15-metrics-reference-formulas-and-indices)
16. [Performance Optimizations](#16-performance-optimizations)
17. [Signifier Parameter Sweep](#17-signifier-parameter-sweep)
18. [Sensitivity and insight policy](#18-sensitivity-and-insight-policy)
19. [Where logic lives](#19-where-logic-lives)
20. [Development and cleanup](#20-development-and-cleanup)

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
- **Signifiers:** Forty-two facial and behavioral cues (e.g., eye contact, smile, frown, gaze away) that are combined to produce the score and to trigger specific insights.
- **Insights / popups:** Short, actionable text messages that suggest what to do or what to watch for (text only).
- **B2B focus:** The logic is tuned for business-to-business meetings (sales, negotiations, alignment) rather than casual chat.

---

## 2. How to Run the Project

1. **Install Python** (3.8 or newer) and install the project’s dependencies:
   - Open a terminal in the project folder and run:  
     `pip install -r requirements.txt`
2. **Start the server:**
   - **Windows:** Double-click `scripts\start.bat` (launcher) or run `python app.py` from the project root.
   - **Mac/Linux:** Run `./scripts/start.sh` or `python app.py` from the project root.
3. **Open the app in your browser:**  
   Go to **http://localhost:5000** (or the port shown in the terminal).

The main page loads from `index.html`. From there you can start engagement detection (e.g., webcam or “partner” video), start an insights session (STT passes spoken context to the AI for coaching), and see insights and metrics.

---

## 3. Project Directory Structure

Here is what each folder and important file is for.

| Path | Purpose |
|------|--------|
| **app.py** | Entry point: starts the web server and loads the app. |
| **config.py** | All settings (API keys, URLs, feature flags, etc.). |
| **routes.py** | Blueprints defining every URL (pages, chat/stream for insights, config, engagement). |
| **detector.py** | Core logic: reads video, detects faces, computes engagement and alerts. |
| **scripts/** | Launcher and run scripts: **launch.py**, **start.bat**, **start.sh**, **stop.bat**, **restart.bat**, **run_tests.bat**. See §13. |
| **run_tests.py** | Runs the test suite (from project root). |
| **bench.py** | Benchmark for get_all_scores(); run from project root: `python bench.py [N]`. |
| **requirements.txt** | Python package list for `pip install`. |
| **README.md** | This file (single consolidated documentation at project root). |
| **services.py**| Backend “services”: Azure AI, speech, face API, insights, etc. |
| **helpers.py** | Reusable helpers: face detection, video handling, scoring, signifiers, etc. (single module). |
| **static/** | Web assets: JavaScript for engagement, insights session (STT/streaming), video source, dashboard, etc. |
| **index.html** | Main single-page app (insights session, engagement UI, video source selection). |
| **extension/** | Browser extensions: **extension/chrome/** (Chrome/Edge), **extension/firefox/** (Firefox). |

---

## 4. Configuration (config.py)

`config.py` holds **all configurable options** for the project. Values can be overridden by **environment variables** (useful for production or different machines). Below, each block is explained in simple terms.

### 4.1 Azure AI Foundry (the “brain”)

- **What it is:** Microsoft’s Azure AI Foundry (successor to Azure OpenAI) provides the large language model (e.g., GPT) that powers the **chat and coaching responses**.
- **Main settings:** Set in environment (no default secrets in code). Edit `.env` with your values.
  - **AZURE_FOUNDRY_KEY** – Secret key to call the API.
  - **AZURE_FOUNDRY_ENDPOINT** – Base URL of the service (e.g. `https://….openai.azure.com/`).
  - **FOUNDRY_DEPLOYMENT_NAME** – Name of the model deployment (e.g. `gpt-4o`).
  - **AZURE_FOUNDRY_API_VERSION** – API version string.
- A small helper **`_sanitize_azure_foundry_config()`** runs at load time to trim spaces and fix the endpoint URL. Older names like `AZURE_OPENAI_*` are still supported for compatibility.

### 4.2 Azure Speech Service

- **What it is:** Used for **speech-to-text (STT)** so the client’s spoken words are converted to text and sent to the backend for transcript storage, speech cue analysis, and passing spoken context to the Foundry service for generating intelligent insights.
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

- **FACE_DETECTION_METHOD** – How we detect faces: `"mediapipe"` (local only), `"azure_face_api"` (cloud only), `"auto"` (app chooses), or `"unified"` (force both). Default **`"auto"`**: use **both MediaPipe and Azure** when the device has enough computing power and network speed; use **MediaPipe only** on low-end devices (old computers, lightweight cams) or when latency is high.
- **MIN_FACE_CONFIDENCE** – Minimum confidence (0–1) to count a detection as a real face; lower = more tolerant in bad lighting.
- **LIGHTWEIGHT_MODE** – If true, **force MediaPipe only** (no Azure) and use a lighter pipeline (fewer frames, smaller buffer). Set for old computers or lightweight webcams.
- **TARGET_FPS_MIN** / **TARGET_FPS_MAX** – Target frame rate (e.g. 30–60 fps) for video processing.
- **DETECTION_WORKER_PROCESS** – If true, run detection in a separate process (advanced).
- **AUTO_DETECTION_SWITCHING** – If true and method is `"auto"`, the app picks detection method from device tier and latency (unified vs MediaPipe only).
- **USE_UNIFIED_ONLY_FOR_HIGH_TIER** – If true (default), only **high-tier** devices (e.g. 4+ CPU cores, 8+ GB RAM) use **unified** (MediaPipe + Azure); **medium** and **low** tier use MediaPipe only. Set false to allow medium-tier devices to use unified when latency is good.
- **AZURE_LATENCY_THRESHOLD_MS** – If backend/Azure round-trip exceeds this (ms), prefer MediaPipe only when in auto mode.
- **FUSION_AZURE_WEIGHT** / **FUSION_MEDIAPIPE_WEIGHT** – When using both Azure and MediaPipe (unified), how much to weight each in the final score (e.g. 0.5 each).

### 4.6 Speech cue and insight tuning

- Signifier and fusion weights are fixed in code (**helpers.py**); no config file or backend.
- Insight weights (prompt_suffix, max_length, opportunity_thresholds) are pre-decided in **services.py**.
- **SPEECH_CUE_LLM_FALLBACK_ENABLED** – Whether to use the LLM to classify phrases when regex doesn’t match (adds latency/cost).
- **SPEECH_CUE_LLM_FALLBACK_RATE_PER_MIN** – Max such LLM calls per minute.

### 4.7 Metric selection and sensitivity

- **METRIC_SELECTOR_ENABLED** – If true, the app may reduce the number of metrics on weaker devices (high/medium/low tier).
- **METRIC_SELECTOR_OVERRIDE** – Force a tier: `"high"`, `"medium"`, or `"low"`.
- **HIGH_SENSITIVITY_METRICS** – When true (default), signifiers use shorter temporal windows and relaxed sustained thresholds so scores react to micro-changes in real time.
- **FULL_RANGE_METRICS** – When true (default), signifier scores are stretched so they span the full 0–100 range instead of hovering near the middle; "absent" signals move toward 0 and "strong" toward 100. Use with **HIGH_SENSITIVITY_METRICS** for maximum responsiveness.

### 4.8 Acoustic analysis

- **ACOUSTIC_ANALYSIS_ENABLED** – Whether to use voice/acoustic features (tone, etc.) from partner or mic.
- **ACOUSTIC_CONTEXT_MAX_AGE_SEC** – How long (seconds) acoustic context is considered “recent” for display and logic.

### 4.9 B2B opportunity and insight buffers

**Sensitivity policy:** We aim to **not miss** negative markers (confusion, resistance, disengagement)—so negative opportunities use **shorter cooldowns** and **lower thresholds**. We only recommend "take the next step" when we are confident—so positive opportunities use **longer cooldowns** and **higher thresholds** to reduce false positives. See **docs/SIMPLIFICATION_AND_VISION_PLAN.md** for full policy and developer checkpoints.

- **NEGATIVE_OPPORTUNITY_COOLDOWN_SEC** – Minimum time between “negative” opportunity insights (e.g. confusion, resistance).
- **POSITIVE_OPPORTUNITY_COOLDOWN_SEC** – Same for “positive” opportunities (e.g. closing, decision-ready).
- **MIN_CONCURRENT_FEATURES_NEGATIVE** / **MIN_CONCURRENT_FEATURES_POSITIVE** – How many cues (face, speech, acoustic) must agree before showing an insight (negative vs positive).
- **INSIGHT_BUFFER_SEC_NEGATIVE** – Minimum seconds between any **negative** insight popup (e.g. 8).
- **INSIGHT_BUFFER_SEC_POSITIVE** – Minimum seconds between any **positive** insight popup (e.g. 45).

### 4.10 Speech-to-Text (STT)

- **STT_LOCALES** – Comma-separated locales for STT (e.g. en-US, de-DE).
- **CONTINUOUS_CONVERSATION** – Whether to keep conversation state across turns. Spoken text is passed to the Foundry service for intelligent insights.

### 4.11 System prompt

- **SYSTEM_PROMPT** (in **config.py**) – The main system prompt is structured in **three tiers** so the model prioritizes correctly and token count is controlled:
  - **Tier 1 (Core):** Identity (elite meeting coach, real-time observation, trusted colleague); a single canonical “when to intervene vs. stay silent”; and how to respond (1–3 sentences, lead with insight, end with next step, no metric listing).
  - **Tier 2 (Research-Grounded Principles):** Short paragraphs with inline citations: nonverbal/paralinguistic (Mehrabian; Richmond, McCroskey, Johnson); psychological safety and idea attribution (Edmondson; inattentional blindness ~30%); virtual/hybrid (backchanneling, Fauville/VFO fatigue); vocal/acoustic (Scherer, Bachorowski, Ladd); engagement bands (HIGH/MEDIUM/LOW) and intervention rules; verbal–nonverbal mismatch and authenticity (Duchenne vs. performative); cognitive load and body signals (Sweller).
  - **Tier 3 (Reference):** Condensed capability domains (engagement, strategic insights, communication, relationship, decision facilitation, negotiation, conflict, stakeholder, cultural, virtual/crisis/time); meeting types (sales, strategy, negotiation, relationship); and advanced considerations (power, culture, timing, risk). A short closing block reinforces: ground in observation, re-engage when low and capitalize when high, trusted advisor goal.
- **Insight popups** (spike, aural, opportunity) do **not** use the full system prompt. They use local system strings in **services.py** (insight generator section) plus:
  - **\_POPUP_BREVITY** and **\_TONE_INSTRUCTIONS** (length and coach voice).
  - **\_INSIGHT_RESEARCH_SNIPPET** – A ~150–200 word constant that summarizes research-backed interpretation: elevated G3 + acoustic_tension/skepticism_objection → validate then address; confusion_multimodal/cognitive_load → clarify, simplify, check understanding; decision_readiness/closing_window → suggest next step or commitment; insights must be actionable in the next 10 seconds and cite the moment (face/voice/words), not raw metrics. This snippet is **appended to the system string** in `generate_insight_for_spike`, `generate_insight_for_aural_trigger`, and `generate_insight_for_opportunity` so all popups share the same framing.
  - **\_INSIGHT_PRINCIPLE_SUMMARY** – Optional 2–3 sentence summary (“Intervene only when actionable. Lead with insight; end with next step. No metric listing.”) is **prepended to the context bundle** when real-time context is sent, so the model sees the same core rules. Engagement level (LOW/MEDIUM/HIGH) is used to tailor the next step (re-engage vs. build vs. capitalize) when present in context.

### 4.12 Application (Flask)

- **FLASK_PORT** – Port the server listens on (default 5000).
- **FLASK_DEBUG** – If true, use Flask’s built-in debug server with auto-reload; otherwise use Waitress.
- **FLASK_HOST** – Bind address (e.g. `0.0.0.0` for all interfaces).

### 4.13 Helper functions in config.py

- **is_cognitive_search_enabled()** – Returns True if Cognitive Search endpoint, key, and index are set.
- **get_cognitive_search_config()** – Returns a dict with Cognitive Search settings or `enabled: False`.
- **is_azure_face_api_enabled()** – Returns True if Face API key and endpoint are set.
- **get_azure_face_api_config()** – Returns Face API config dict or `enabled: False`.
- **get_face_detection_config()** – Returns face detection method and whether MediaPipe and Azure are available; used by the API to expose detection options to the frontend.

---

## 5. Application Entry Point (app.py)

**app.py** is what you run to start the server (`python app.py`).

### What it does

1. **Imports** the Flask app framework, CORS, compression, and **register_routes** from **routes.py** (all URL handlers), plus `config`.
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

So: **app.py** only creates and runs the web app; it does not define individual URLs. All URLs are defined in **routes.py** (pages, chat, config, engagement blueprints).

### 5.1 Launcher (scripts/launch.py, optional)

The **launcher** lives in **scripts/**. Run it with `python scripts/launch.py` from the project root, or double-click **scripts\\start.bat** (Windows) or run **scripts/start.sh** (Mac/Linux). It opens a **small window** (using Tkinter) that lets you:

- **Start** the server (runs `python app.py` in the background).
- **Stop** the server (finds and terminates the server process).
- **Restart** the server (stop then start).
- **Open app in browser** (opens http://localhost:5000 in the default browser).
- **Run tests** and see results in the window, with simple **troubleshooting suggestions** if a test fails (e.g. “pip install -r requirements.txt”, “check API keys”).

It uses theme colors and a simple layout so non-technical users can start the app, run tests, or restart the server without using the command line. The project works fully without it; the launcher is only a convenience.

---

## 6. Routes and API Endpoints (routes.py)

**routes.py** defines every **URL** the server responds to via Flask blueprints: **pages** (/, /dashboard, static), **chat** (/chat, /chat/stream), **config** (/config/*, /speech/token), and **engagement** (/engagement/*, /api/engagement/*, /api/context-push). Engagement logic is centralized in **services.py** (engagement API section). Below, each route is listed with a short, plain-English description.

### 6.1 Pages and static files

- **GET /**  
  Serves the main page (**index.html**). This is the single-page app (chat, engagement, video source, etc.).

- **GET /dashboard**  
  Redirects to the main app (**/**). The engagement dashboard is inline in the main page (index.html).

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
  Returns a **token** and **region** for the Azure Speech Service so the browser can do STT without exposing the server’s key.

- **GET /config/speech**  
  Returns speech-related config (region, private endpoint, etc.).

- **GET /config/openai**  
- **GET /config/foundry**  
  Return Foundry/OpenAI config (endpoint, deployment, API version). Used so the frontend knows how to talk to the AI.

- **GET /config/cognitive-search**  
  Returns Cognitive Search config (enabled, endpoint, index, etc.).

- **GET /config/stt-tts**  
  Returns STT settings (locales, continuous conversation). Used so the frontend can run STT and pass spoken context to Foundry for insights.

- **GET /config/system-prompt**  
  Returns the system prompt text (the “personality” of the coach).

- **GET /config/all**  
  Returns **all** configuration in one blob (used by the main UI to know what’s available).

### 6.4 Face detection (config + overrides)

- **GET /config/face-detection**  
  Returns current face detection method and what’s available (MediaPipe, Azure).

- **PUT /config/face-detection**  
  Updates the face detection method (e.g. switch to MediaPipe or Azure).

- **GET /config/azure-face-api**  
  Returns Azure Face API config (enabled, endpoint, region).

### 6.5 Engagement detection (start, stop, video, transcript)

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

### 6.6 Engagement state and insights (main consumer API)

- **GET /engagement/debug**  
  Returns **debug** info: whether the detector is running, method, FPS, whether there is state, consecutive “no face” count, etc.

- **GET /engagement/state**  
  **Most important** endpoint for the main UI. It:
  - Returns current **engagement state**: score, level, face_detected, metrics (attention, eye contact, etc.), context (summary, indicators, actions), signifier scores, Azure metrics if used, composite metrics, FPS, detection method.
  - If there is a **pending alert** (spike, opportunity, or aural) and the insight buffer allows it, the response also includes that alert and a suggested **insight text** (sometimes generated by Azure Foundry). The client shows this as a text-only popup only when the Insights Session is live; metrics update in real time regardless. After the client uses it, a later call can record that the insight was shown (so buffers are respected).
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

### 6.7 Engagement context and insight helpers (services.py)

Route handlers in **routes.py** call **services.py** for all engagement logic. Key helpers in **services.py** include:

- **`get_context_store()`** — Returns the shared context store (last context sent, last response, etc.).
- **`build_engagement_context_bundle(additional_context)`** — Builds the text bundle of current engagement state and optional extra context, used as the prompt prefix for chat and insights.
- **`build_fresh_insight_context(...)`** — Builds the context passed to the insight generator when creating popup/insight text (state, metrics, speech tags, acoustic tags, etc.).
- **`get_context_generator()`** — Returns the shared **ContextGenerator** instance (from helpers) used to build human-readable context from engagement state.

The rest of the flow is reading request body, calling services, formatting JSON, and handling errors.

---

## 7. Engagement State Detector

The **EngagementStateDetector** (in **detector.py**) is the **core component** that turns **video (and optionally transcript/acoustic)** into a single **engagement state** and **alerts**. Everything else (API, insights, chat) depends on this.

### 7.1 Main concepts

- **Video source:** Frames come from a **VideoSourceHandler** (webcam, file, stream, or “partner” — i.e. frames pushed by the browser).
- **Face detection:** Each frame is passed to a **face detector** (MediaPipe and/or Azure). The detector returns faces with **landmarks** (points on the face) and optionally **emotions** (Azure).
- **Signifiers:** From landmarks (and optional emotions), an **ExpressionSignifierEngine** computes **42 signifier scores** (0–100), e.g. eye contact, smile, gaze aversion, brow furrow.
- **Groups:** Those 42 are grouped into **4 groups (G1–G4)** and averaged to get group means. Groups represent things like “interest” (G1), “thinking” (G2), “resistance” (G3), “readiness” (G4).
- **Score:** An **EngagementScorer** combines metrics (attention, eye contact, expressiveness, etc.) into one **0–100** score and an **EngagementLevel** (e.g. VERY_LOW … VERY_HIGH).
- **Context:** A **ContextGenerator** turns the current state into **text** (summary, indicators, suggested actions) for the AI and for display.
- **Alerts:** The detector looks for **spikes** (sudden rises in a group), **composite_100** (very high composite), and **B2B opportunities** (e.g. “closing window,” “confusion”). When one fires and the **insight buffer** allows it, it becomes a **pending alert** that **GET /engagement/state** can return and turn into a text-only popup.

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
    - Runs **ExpressionSignifierEngine** to get 42 signifier scores.  
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
    Builds **EngagementMetrics** from the 42 signifier scores when not using Azure.
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
  - **`_calculate_confidence()`** (internal)  
    Computes a 0–1 confidence for the current state (e.g. from face detection confidence and stability).

So: **detector.py** ties together **video → faces → signifiers → groups → score + context + alerts**, and exposes that as **current state** and **pending alert** for the rest of the app.

---

## 8. Services

**Services** are backend modules that talk to **external APIs** (Azure) or manage **shared state** (insights, acoustic context, request tracking). Each file and its main functions are described below.

### 8.1 services.py — Azure Foundry

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

### 8.2 services.py — Azure Speech

- **What it does:** Wraps **Azure Speech Service** for **token** retrieval (STT only). Spoken context is passed to the Foundry service for generating intelligent insights.
- **AzureSpeechService (class)**  
  - **`get_speech_token()`**  
    Returns **token** and **region** for the client so the browser can run STT and send transcript to the backend for speech cue analysis.  
- **`get_speech_service()`**  
  Returns the singleton **AzureSpeechService** instance.

### 8.3 services.py — Azure Face API

- **What it does:** Calls **Azure Face API** to detect faces (and optionally landmarks and attributes) in an image.
- **AzureFaceAPIService (class)**  
  - **`detect_faces(image_bgr, ...)`**  
    Sends the image to the Face API and returns a list of face results (e.g. rectangle, landmarks, emotions).  
  - **`test_connection()`**  
    Sends a small test image to check connectivity.  
- **`get_azure_face_api_service()`**  
  Returns the singleton if the API is configured, else None.

### 8.4 services.py — Insight generator

- **What it does:** Generates **insight text** for popups (spike, aural trigger, or B2B opportunity) and manages **speech tags**, **transcript**, and **pending aural alert**.
- **Main functions:**  
  - **`get_insight_weights()`**  
    Returns parameters for insight generation (prompt suffix, length, thresholds). Pre-decided in code.  
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

### 8.5 services.py — Acoustic context store

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

## 9. Helpers (helpers.py)

**helpers.py** is a single module containing reusable logic: video handling, face detection, signifiers, scoring, B2B opportunities, context, composites, weights, and system capability. The sections below describe the main areas within this module (historically from separate utils submodules).

### 9.1 Video source handler (helpers.py)

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
    Returns a dict of all 42 signifier names → scores.  
  - Many **internal methods** (e.g. **_g1_eye_contact**, **_g3_eye_block**, **_g3_gaze_aversion**) implement each signifier from the buffer; they use **FPS-scaled** frame counts so durations (e.g. 400 ms) are consistent across frame rates.

### 9.3 Engagement scorer (helpers.py)

- **What it does:** Combines **metrics** (attention, eye contact, expressiveness, head movement, symmetry, mouth activity) into one **0–100 engagement score**.
- **EngagementMetrics (dataclass)**  
  Holds: attention, eye_contact, facial_expressiveness, head_movement, symmetry, mouth_activity.  
- **EngagementScorer (class)**  
  - **`score(signifier_scores, metrics_dict)`**  
    Computes overall score and level from signifier scores and optional metrics dict.  
  - **`compute_metrics(signifier_scores)`**  
    Derives the six metrics from the 42 signifiers (e.g. attention from EAR, activity from MAR).

### 9.4 Context generator (helpers.py)

- **What it does:** Turns **current engagement state** into **human-readable context** (summary, level description, key indicators, suggested actions, risks, opportunities) for the AI and for the UI.
- **EngagementContext (dataclass)**  
  Holds: summary, level_description, key_indicators, suggested_actions, risk_factors, opportunities.  
- **ContextGenerator (class)**  
  - **`generate_context(score, level, metrics, signifier_scores, ...)`**  
    Returns an **EngagementContext** filled with text tailored to the current state.

### 9.5 Face detection interface (helpers.py)

- **What it does:** **Abstract interface** for face detectors so the rest of the app doesn’t care whether we use MediaPipe or Azure.
- **FaceDetectionResult**  
  Dataclass: **landmarks** (array), **bounding_box** (optional), **emotions** (optional dict), etc.  
- **FaceDetectorInterface (ABC)**  
  - **`detect_faces(frame)`** → list of **FaceDetectionResult**.  
  - **`get_name()`** → string (e.g. "mediapipe", "azure_face_api").  
  - **`close()`** – cleanup.

### 9.6 MediaPipe detector (helpers.py)

- **What it does:** **Face detection** using **MediaPipe** (local, no cloud). Returns landmarks and bounding box.
- **MediaPipeFaceDetector(FaceDetectorInterface)**  
  Implements **detect_faces**, **get_name**, **close**. Uses MediaPipe Face Mesh.

### 9.7 Azure Face detector (helpers.py)

- **What it does:** **Face detection** using **Azure Face API** (cloud). Uses **AzureFaceAPIService** to get faces and maps results to **FaceDetectionResult** (landmarks, emotions, etc.).
- **AzureFaceAPIDetector(FaceDetectorInterface)**  
  Implements **detect_faces**, **get_name**, **close**.

### 9.8 Azure landmark mapper (helpers.py)

- **What it does:** **Maps Azure Face API landmarks** to a format compatible with **MediaPipe** (same point indices), so the same signifier engine can run on both.
- **expand_azure_landmarks_to_mediapipe(landmarks, bbox, frame_shape)**  
  Returns a landmark array in MediaPipe order; used when we have Azure landmarks but want to run the MediaPipe-based signifier pipeline.

### 9.9 Azure engagement metrics (in detector.py)

- **What it does:** **Engagement metrics from Azure Face API** (emotions, B2B composites, 0–100 score). Inlined in **detector**; used when detection method is Azure or unified.
- **get_all_azure_metrics(face_result)**  
  Returns full set of Azure metrics (base, composite, score) for the API.  
- **get_azure_score_breakdown(face_result, ...)**  
  Returns step-by-step breakdown for “How is the score calculated?” (frontend).

### 9.10 B2B opportunity detector (helpers.py)

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

### 9.11 Engagement composites (helpers.py)

- **What it does:** **Composite metrics** that combine **facial signifiers**, **speech tags**, and **acoustic tags** (e.g. “confusion” = face + “I don’t understand” + uncertain voice). Used for richer insights and for “composite_100” alerts.
- **compute_composite_metrics(state, speech_tags, acoustic_tags, ...)**  
  Returns a dict of composite names → 0–100 scores.  
- Helpers **`_has_category`**, **`_category_strength`** check and weight recent speech/acoustic tags.

### 9.12 Signifier weights (helpers.py)

- **What it does:** Provides **signifier and group weights** (42 signifiers, 4 groups) as **in-code defaults only**; no config file or backend fetch. Provides a **weights_provider** callable for the signifier engine.
- **get_weights()**  
  Returns the current weights dict (signifier list, group list).  
- **load_weights()**  
  Applies in-code defaults (no file I/O).  
- **get_fusion_weights()**  
  Returns Azure vs MediaPipe fusion weights when using both.  
- **build_weights_provider()**  
  Returns a callable that returns the current weights (for the engine).

### 9.13 Metric selector (helpers.py)

- **What it does:** **Chooses which metrics (signifiers)** run based on **device tier** (high/medium/low) from CPU and RAM, so weaker devices run fewer metrics.
- **MetricConfig**  
  Holds: list of active signifier keys, buffer size, etc.  
- **get_active_metrics()**  
  Returns the list of signifier keys to run for the current tier.  
- **get_active_metrics_with_config()**  
  Returns the full **MetricConfig** (used by the detector).  
- ** _get_system_resources()**, **_determine_tier()**, **_build_config(tier)**  
  Internal: get CPU/RAM, map to tier, build config.

### 9.14 Detection capability (helpers.py)

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

### 9.15 Config helpers (config.py)

- **What it does:** **Config response** for GET /config/all and **face detection method preference** (runtime choice: mediapipe, azure_face_api, auto, unified).
- **build_config_response()**  
  Returns a dict with speech, foundry, cognitiveSearch, sttTts, systemPrompt, faceDetection, azureFaceApi, acoustic.
- **get_face_detection_method()** / **set_face_detection_method(method)**  
  Get/set the current face detection method (used by GET/PUT /config/face-detection).

### 9.16 services.py (acoustic interpretation + store)

- **What it does:** **Interprets** raw **acoustic windows** into **tags** and **NL summary**; in-memory store for recent windows.
- **interpret_acoustic_windows(windows)**  
  Processes a list of window dicts; returns (summary_string, list of tags).
- **append_acoustic_windows**, **get_recent_acoustic_context**, **get_recent_acoustic_tags**, **clear_acoustic_context**, etc.

---

## 10. Frontend (index.html and Static Files)

### 10.1 index.html

**index.html** is the **main single-page application**. It typically includes:

- **Engagement section:** Button to start/stop engagement; choice of video source (webcam, partner/screen share, file); place to show the **engagement video feed** and **score/level**.
- **Insights session:** Start/close session; microphone for STT; spoken context is sent to Foundry and streaming insights appear in the **Insight Transcript**.
- **Insights / popups:** Area or overlay where **insight messages** appear (text only).
- **API:** **GET /api/engagement/context-and-response** returns last context and AI response for API consumers.
- **Scripts:** References to **static JS** modules (session manager, engagement detector, video source selector, acoustic analyzer, app-main, etc.).

So **index.html** is the “shell”; the **behavior** lives in the **static JavaScript** files and in the **routes** that serve data.

### 10.2 static/js/*.js

- **session.js**  
  Manages **session** and **API base URL**; handles **streaming** requests to **/chat/stream** (when the user speaks or sends a message for insights) and **context-and-response** to **/api/engagement/context-and-response**.

- **engagement.js**  
  - **Start/stop** engagement via **POST /engagement/start** and **POST /engagement/stop**.  
  - **Poll** **GET /engagement/state** on an interval.  
  - When **partner** source is selected, **capture** frames (e.g. from a shared screen or video element) and **POST** them to **/engagement/partner-frame** at ~30 fps.  
  - Show **alerts/insights** from the state response (text-only popup).  
  - May update **signifier** or **composite** panels from the state payload.

- **app-main.js**  
  Ties together the UI: insights session (STT, microphone), streaming Foundry responses into the Insight Transcript when the user speaks or sends a message, engagement-triggered popups, and video source selection.

- **video.js**  
  UI for choosing **video source** (webcam, file, stream, partner) and sending that choice when starting engagement.

- **acoustic.js**  
  Gets **audio** (e.g. from partner or mic), computes **acoustic features** (or uses an external library), and **POST**s them to **/engagement/acoustic-context** at an interval.

- **signifiers.js** / **azure.js** / **composites.js**  
  Display **signifier scores**, **Azure metrics**, and **composite metrics** from the engagement state in panels or lists.

- **dashboard.js**  
  Drives the **inline engagement dashboard** in the main content area (metrics, context summary, append/send context).

- **app-main.js**  
  Main app initialization and wiring (e.g. config fetch, video source, session).

Together, these scripts implement **chat**, **engagement control**, **video source**, **partner frame upload**, **acoustic upload**, **state polling**, and **insight/alert display** on the main page and dashboard.

---

## 11. Browser Extensions

The project includes **browser extensions** so the app can run in a side panel on any webpage (e.g. Google Meet, Zoom, Teams). **The extension is standalone: it does not require the web app to be running on your machine.** Use a hosted instance of the app instead.

### 11.1 How it works

1. **Deploy the web app** to a server (e.g. Azure, AWS, or your own host) so it is available at a URL like `https://your-copilot.example.com`.
2. **Install the extension** (Chrome or Firefox; see below).
3. **Set the app URL** in the extension popup (or in the side panel on first open). Save the URL of your hosted Copilot app.
4. **Open the side panel** on any tab. The panel loads your app in an iframe and sends the current tab URL to it (for meeting-site detection). Everything runs in the browser; no local server needed.

### 11.2 Folder structure and install

- **extension/** — Single folder with browser-specific subfolders:
  - **extension/chrome/** — Chrome/Edge extension (manifest v3, icons, popup, sidepanel, background). Install: Open `chrome://extensions`, enable Developer mode, **Load unpacked** → select the `chrome` folder.
  - **extension/firefox/** — Firefox 109+ (manifest v2, similar structure). Install: Open `about:debugging` → This Firefox → **Load Temporary Add-on** → select `firefox/manifest.json`.

### 11.3 First-time setup

- Click the extension icon to open the **popup**. Enter your **App URL** (the full URL of your deployed Meeting Copilot app, e.g. `https://your-app.azurewebsites.net`), then click **Save URL**.
- Click **Open in side panel**. The first time, if no URL is saved, the panel shows a setup form; enter the URL there and click **Save and open**.
- To change the URL later, open the popup and edit the App URL, or in the panel click **Change URL** when the app cannot be reached.

### 11.4 Optional: local development

If you run the app locally (`python app.py`), you can set the App URL to `http://localhost:5000` so the extension loads your local instance. The extension works the same whether the app is local or hosted.

### 11.5 Partner video flow

The **exact** flow (how the app gets the “partner” video) depends on the app’s design: often the **main page** (index.html) uses **getDisplayMedia** to share a tab/window and then **engagement.js** sends those frames to **POST /engagement/partner-frame**. The extensions add convenience (e.g. open the app with the current tab URL).

---

## 12. Tests

- **run_tests.py** (project root)  
  Orchestrates the test suite: discovers and runs all **test_*.py** files from the project root (primarily **test_core.py**), prints results. Usage: `python run_tests.py [--verbose] [pattern]`.
- **test_core.py** (project root)  
  Single consolidated test file containing: API routes (/config, /engagement/start, /engagement/state, chat, context-and-response), config, engagement, services (Foundry, speech, insight generator), helpers (video handler, signifier engine, scorer, context generator), metrics and signifier validation, and signifier parameter sweep. Synthetic landmark data for signifier tests is inlined in this file.

Running **python run_tests.py** from the project root (or **scripts\\run_tests.bat** on Windows) runs the suite and reports pass/fail.

---

## 13. Scripts and Batch Files

All launcher and run scripts are in the **scripts/** folder. (Signifier and fusion weights are in-code in **helpers.py**; see §9.12.)

- **scripts/launch.py** — Desktop launcher (start/stop/restart server, open browser, run tests). Run from project root: `python scripts/launcher.py`, or use **scripts/start.bat** / **scripts/start.sh**.
- **scripts/start.bat** (Windows) — Double-click to open the launcher.  
- **scripts/start.sh** (Mac/Linux) — Run to open the launcher.  
- **scripts/restart.bat** — Stops and restarts the server (used by the launcher).  
- **scripts/stop.bat** — Stops the server (clears port 5000; used by the launcher).  
- **scripts/run_tests.bat** — Runs the test suite from the command line (calls **run_tests.py** from project root).

**Quick start:**  
- **Windows:** Double-click `scripts\start.bat`.  
- **Mac/Linux:** From project root, run `./scripts/start.sh` or `python3 scripts/launch.py`.  
- **Run app only (no launcher):** From project root, run `python app.py`.  
- **Run tests only:** From project root, run `python run_tests.py`, or on Windows `scripts\run_tests.bat`.

The real app logic is in **app.py**, **routes.py**, and **run_tests.py** at the project root.

---

## 14. Composite Metrics (Detailed)

Composite metrics combine **42 facial signifiers** (face-only; from **helpers.py**), **speech categories** (phrase cue detector), and **acoustic tags** (voice analyzer) into 0–100 scores. Two types: **multimodal composites** (weighted blends of G1–G4, speech, acoustic) and **condition-based composites** (specific signifier combinations, e.g. forward lean 60–80 AND eye contact ≥ 70).

**Group means (G1–G4):** G1 = Interest & Engagement; G2 = Cognitive Load; G3 = Low resistance (composites use 100−G3 for resistance); G4 = Decision-Ready.

**Condition-based examples:** topic_interest_facial (forward_lean 55–90, eye_contact ≥ 70); active_listening_facial (nodding, parted_lips, eye_contact); agreement_signals_facial (nodding, Duchenne, eye_contact); closing_window_facial (smile_transition, fixed_gaze, mouth_relax); resistance_cluster_facial (contempt + gaze_aversion or lip_compression + gaze_aversion). **Research:** Mehrabian, Wells & Petty, Ekman, Cialdini, Kahneman/Sweller, Edmondson, Scherer/Bachorowski/Ladd.

**Multimodal formulas (high level):** decision_readiness_multimodal = 0.55×G4 + 0.45×commit_interest; confusion_multimodal = 0.45×G2 + 0.40×confusion_concern + 25×has_uncertainty; disengagement_risk_multimodal combines (100−G1), no_commit, (100−G3), acoustic_disengagement. Condition-based scores refine some composites (e.g. +10 to decision_readiness when closing_window_facial ≥ 70). Implemented in **helpers.py** (engagement composites); tier-based visibility in **helpers.py** (metric selector).

**Usage:** Capitalize when decision_readiness and closing/agreement signals are high; re-engage when confusion or resistance high; build rapport when active_listening and receptivity high.

---

## 15. Metrics Reference (Formulas and Indices)

**Per-frame buffer keys** (ExpressionSignifierEngine): ear, eye_area, eyebrow_l/r, pitch, yaw, roll, mar, mar_inner, mouth_w/h, face_z, gaze_x/y, face_var, nose_std/nose_height, is_blink, face_scale, mouth_corner_asymmetry_ratio, face_movement.

**Landmark indices (MediaPipe 468):** LEFT_EYE, RIGHT_EYE (EAR, eye area); MOUTH, MOUTH_LEFT 61, MOUTH_RIGHT 17; LEFT_EYEBROW, RIGHT_EYEBROW; NOSE; NOSE_TIP 4, CHIN 175; FACE_LEFT 234, FACE_RIGHT 454.

**Basic metrics from signifiers** (engagement_detector._metrics_from_signifiers): attention = mean(duchenne, eye_contact, eyebrow_flash, pupil_dilation); eye_contact = g1_eye_contact; facial_expressiveness = 0.5×mean(expr) + 0.5×max(expr) over duchenne, parted_lips, smile_transition, softened_forehead, eyebrow_flash, micro_smile, eye_widening, brow_raise_sustained (so one strong expression can raise the score); head_movement = 0.5×mean(head_components) + 0.5×max(head_components), where head_components = head_tilt, rhythmic_nodding, forward_lean, and (100 − g2_stillness) so rapid/noticeable movement can reach high levels; symmetry = g1_facial_symmetry; mouth_activity = g1_parted_lips.

**Composite inputs:** group_means (g1–g4), speech_tags (category, phrase, time), acoustic_tags, acoustic_negative_strength. Speech strength = _category_strength(tags, categories, 12s). Composites implemented in **helpers.py** (compute_composite_metrics).

**Key ranges (calibration):** EAR 0.22–0.28 = open eyes; &lt; 0.16 = blink; 0.78–0.96×baseline = AU6 band for Duchenne. MAR 0.12–0.32 = parted_lips high; &lt; baseline×0.75 sustained = lip_compression. Yaw 0° = best eye contact; ~25° yaw or ~20° pitch ≈ 50. Gaze aversion: total_dev &gt; 10° for 8+ frames. Contempt: mouth_corner_asymmetry_ratio &gt; 0.14, above baseline + 0.06, 3/4 frames.

**Azure vs MediaPipe:** When Azure Face API is used, head_pose (pitch, yaw, roll) and emotions (e.g. contempt) can be used; g3_contempt can use Azure emotion value; other signifiers use same formulas with improved pose when Azure provides head_pose.

---

## 16. Performance Optimizations

Optimizations applied for speed, memory, and reliability while preserving behavior.

**routes.py:** Import get_recent_acoustic_tags; use hashlib at top level. Build metrics_summary, recent_speech_tags, acoustic_tags_list, composite_metrics once per request.

**engagement_detector.py:** JPEG quality 85 for video feed; use deque directly for adaptive score history (no list copy); pass has_recent_speech into _check_composite_at_100; use get_recent_acoustic_tags_and_negative_strength() once per frame.

**services.py:** get_recent_acoustic_tags_and_negative_strength() under one lock, one interpret_acoustic_windows() call.

**config.py:** Warn once for missing Azure Face API (avoid repeated log noise).

**Frontend:** Default poll interval 250 ms (was 200 ms) for GET /engagement/state.

**Summary:** Fewer locks and duplicate work per frame; correct imports; smaller MJPEG payloads; less allocation; fewer polls. Behavior unchanged.

---

## 17. Signifier Parameter Sweep

Synthetic landmark configurations (neutral, smile, gaze aversion, contempt, parted lips, lip compression, head tilt, brow raise/furrow) are used in **test_core.py** to validate that signifiers move in the expected direction (e.g. g1_duchenne higher for smile than neutral; g3_gaze_aversion higher for gaze averted; g1_eye_contact lower when gaze averted).

**Expected direction (research):** smile &gt; neutral: g1_duchenne, g4_smile_transition. gaze_aversion_80 &gt; neutral: g3_gaze_aversion; gaze_aversion &lt; neutral: g1_eye_contact. contempt ≥ neutral: g3_contempt. parted_lips &gt; lip_compression: g1_parted_lips; lip_compression &gt; neutral: g3_lip_compression. brow_raise &gt; neutral: g1_brow_raise_sustained.

**EAR / MAR / Yaw ranges:** EAR 0.22–0.28 = high attention; &lt; 0.16 = blink; 0.78–0.96×baseline = AU6 for g1_duchenne; &lt; 0.85×baseline sustained = g3_narrowed_pupils, g2_eye_squint. MAR 0.12–0.32 = g1_parted_lips high; &lt; baseline×0.75 sustained = g3_lip_compression, g2_mouth_tight_eval. Yaw 0° = best eye_contact; &gt; 10° for 8+ frames = g3_gaze_aversion.

Optional: run **python test_runner.py** with environment variable **WRITE_SIGNIFIER_SWEEP_DOC=1** to generate sweep results with current score ranges from the test suite.

---

## 18. Sensitivity and insight policy

We aim to **not miss** negative markers (confusion, resistance, disengagement)—so negative opportunities use **shorter cooldowns** and **lower thresholds**. We only recommend "take the next step" when we are confident—so positive opportunities use **longer cooldowns** and **higher thresholds** to reduce false positives. Do not change these without developer/product sign-off. Relevant settings: `config.py` (e.g. `NEGATIVE_OPPORTUNITY_COOLDOWN_SEC`, `MIN_CONCURRENT_FEATURES_*`), `engagement_detector.py` (e.g. `_spike_*`, `_min_concurrent_features_*`), and `helpers.py` (signifier and opportunity thresholds).

**Developer checkpoints** — Confirm with the developer before: changing sensitivity or thresholds, splitting or deleting helpers, deleting files/folders, removing features, renaming/moving files, changing engagement logic, or adding new opportunity types.

---

## 19. Where logic lives

| Concern | Location |
|--------|----------|
| Engagement score, level, bands | engagement_detector.py, helpers.py |
| 42 signifiers | helpers.py — ExpressionSignifierEngine, _g1_* … _g4_* |
| Composites | helpers.py — compute_composite_metrics |
| B2B opportunities | helpers.py — detect_opportunity, _check_cooldown; config.py for cooldowns |
| Spike detection | engagement_detector.py — _spike_*, _composite_100_* |
| Face detection | helpers.py — MediaPipe + Azure wrappers |
| Insights text | services — insight generator |
| Routes and API | routes.py |

---

## 20. Development and cleanup

- **Test:** `python test_runner.py`. **Benchmark:** `python bench.py [N]`.
- **Before deleting files or folders** (e.g. backup/, duplicate scripts): confirm with the developer. Extension layout: **extension/chrome/**, **extension/firefox/** (or **extension-chrome/**, **extension-firefox/** per your tree).
- **Before changing sensitivity or engagement logic:** confirm with the developer (see §18).

---

## Summary

- **Business Meeting Copilot** is an **AI meeting coach**: it uses **video** (webcam or partner) and **audio/transcript** (STT → backend for speech cue analysis) to estimate **engagement** and **B2B opportunities**, and to produce **coaching insights** (text-only popups and chat).
- **config.py** centralizes all settings and config response helpers; **app.py** starts the server; **routes.py** defines every URL and wires them to **services.py** (detector, context, Foundry, speech, face API, insights, acoustic store) and **config**.
- The **EngagementStateDetector** is the core: it pulls **frames** from a **VideoSourceHandler**, runs **face detection** (MediaPipe and/or Azure), computes **42 signifiers** and **4 group means**, then **score**, **context**, and **alerts** (spikes, opportunities, aural). The **GET /engagement/state** endpoint exposes this as the main API for the UI.
- **Services** handle Azure and shared state; **helpers.py** handles video, faces, signifiers, scoring, context, B2B opportunities, composites, weights, and device capability. The **frontend** (index.html and static JS) drives chat, engagement start/stop, video source, partner frames, acoustic upload, polling, and display of metrics and insights.
- **Tests** cover API, metrics, services, and helpers. Signifier and fusion weights are fixed in **helpers.py** (no config file). **Scripts** (.bat, .sh) are for starting/stopping the server and running tests.

This document is the **single comprehensive reference** for the project and lives at the project root (`README.md`). Use the **Table of Contents** at the top to jump to any section. For the very first time, read **§1 What Is This Project?** and **§2 How to Run the Project**, then **§3 Project Directory Structure**; after that, use the TOC to dive into any area you care about (config, routes, detector, services, helpers, frontend, extensions, tests).
