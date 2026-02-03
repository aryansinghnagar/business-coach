"""
Flask routes for Business Meeting Copilot.

Handles static files, chat (streaming + optional engagement/On Your Data), config,
speech/avatar tokens, engagement start/stop/state/transcript/debug, and weights.
See docs/DOCUMENTATION.md for API reference.
"""

from flask import Blueprint, request, jsonify, send_from_directory, Response
from services.azure_openai import openai_service
from services.insight_generator import (
    append_speech_tag,
    append_transcript,
    clear_pending_aural_alert,
    clear_recent_speech_tags,
    clear_transcript,
    generate_insight_for_aural_trigger,
    generate_insight_for_opportunity,
    generate_insight_for_spike,
    get_and_clear_pending_aural_alert,
    get_pending_aural_alert,
    get_recent_transcript,
    _check_for_trigger_phrases,
    _set_pending_aural_alert,
)
from utils.azure_engagement_metrics import binarize_metrics
from utils.b2b_opportunity_detector import clear_opportunity_state
from services.azure_speech import speech_service
from utils.helpers import build_config_response
from utils.face_detection_preference import get_face_detection_method, set_face_detection_method
from engagement_state_detector import EngagementStateDetector, EngagementLevel, VideoSourceType
from utils.video_source_handler import set_partner_frame_from_bytes
from utils.context_generator import ContextGenerator
from utils import signifier_weights
from typing import Optional
import math
import time
import config


# Create a blueprint for better organization
api = Blueprint('api', __name__)

# Global engagement detector instance (singleton pattern)
# Initialized on first use to avoid import-time initialization
engagement_detector: Optional[EngagementStateDetector] = None
context_generator = ContextGenerator()


# ============================================================================
# Static File Routes
# ============================================================================

@api.route("/")
def index():
    """
    Serve the main index.html page.
    
    Returns:
        Response: HTML file or error response
    """
    try:
        return send_from_directory(".", "index.html")
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 404


@api.route("/static/<path:filename>")
def static_files(filename):
    """
    Serve static files (JS, CSS, etc.).
    
    Args:
        filename: Path to static file
    
    Returns:
        Response: Static file or 404
    """
    try:
        return send_from_directory("static", filename)
    except FileNotFoundError:
        return jsonify({"error": f"Static file not found: {filename}"}), 404


@api.route("/favicon.ico")
def favicon():
    """
    Handle favicon requests.
    
    Returns:
        Response: Empty 204 response
    """
    return "", 204


# ============================================================================
# Chat Routes
# ============================================================================

@api.route("/chat", methods=["POST"])
def chat():
    """
    Non-streaming chat endpoint.
    
    Accepts a user message and returns a single AI response.
    
    Request Body:
        {
            "message": "User's message text"
        }
    
    Returns:
        JSON: {
            "response": "AI assistant's response"
        }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"error": "Missing message"}), 400

    try:
        messages = [
            {"role": "user", "content": user_input}
        ]
        response_text = openai_service.chat_completion(
            messages=messages,
            system_prompt=config.SYSTEM_PROMPT
        )
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({
            "error": "Failed to get AI response",
            "details": str(e)
        }), 500


@api.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint with support for conversation history and On Your Data.
    
    This endpoint streams responses using Server-Sent Events (SSE), allowing
    for real-time display of AI responses as they are generated.
    
    Request Body:
        {
            "messages": [
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Previous response"}
            ],
            "enableOyd": false,
            "systemPrompt": "Optional custom system prompt"
        }
    
    Returns:
        Response: Server-Sent Events stream
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    enable_oyd = data.get("enableOyd", False)
    system_prompt = data.get("systemPrompt", config.SYSTEM_PROMPT)
    include_engagement = data.get("includeEngagement", True)  # Default to True
    
    if not messages:
        return jsonify({"error": "Missing messages"}), 400
    
    # Real-time engagement context (no smoothing) for actionable avatar insights
    if include_engagement and engagement_detector:
        try:
            state = engagement_detector.get_current_state()
            if state and state.face_detected:
                engagement_context = context_generator.format_for_ai(state.context)
                if messages and messages[-1].get("role") == "user":
                    content = messages[-1].get("content")
                    # Only prepend to string content so we don't corrupt multimodal (array) messages
                    if isinstance(content, str):
                        engagement_msg = f"[MEETING CONTEXT]\n{engagement_context}\n[/MEETING CONTEXT]\n\n"
                        messages[-1]["content"] = engagement_msg + content
        except Exception as e:
            # If engagement context fails, continue without it
            print(f"Warning: Could not include engagement context: {e}")
    
    try:
        def generate():
            """Generator function for streaming responses."""
            try:
                for chunk in openai_service.stream_chat_completion(
                    messages=messages,
                    enable_oyd=enable_oyd,
                    system_prompt=system_prompt
                ):
                    yield chunk
            except Exception as e:
                yield f'data: {{"error": "Streaming error: {str(e)}"}}\n\n'
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        return jsonify({
            "error": "Failed to get AI response",
            "details": str(e)
        }), 500


# ============================================================================
# Speech Service Routes
# ============================================================================

@api.route("/speech/token", methods=["GET"])
def get_speech_token():
    """
    Get Azure Speech Service access token.
    
    Returns:
        JSON: {
            "token": "Access token string",
            "region": "Azure region"
        }
    """
    try:
        token_data = speech_service.get_speech_token()
        return jsonify(token_data)
    except Exception as e:
        error_details = str(e)
        status_code = 504 if "timeout" in error_details.lower() else 502
        return jsonify({
            "error": "Failed to get speech token",
            "details": error_details
        }), status_code


# ============================================================================
# Configuration Routes
# ============================================================================

@api.route("/config/speech", methods=["GET"])
def get_speech_config():
    """
    Get speech service configuration (region only, key stays on backend).
    
    Returns:
        JSON: {
            "region": "Azure region",
            "privateEndpointEnabled": false
        }
    """
    return jsonify({
        "region": config.SPEECH_REGION,
        "privateEndpointEnabled": config.SPEECH_PRIVATE_ENDPOINT_ENABLED
    })


@api.route("/config/openai", methods=["GET"])
def get_openai_config():
    """
    Get OpenAI configuration (endpoint and deployment only, key stays on backend).
    
    Returns:
        JSON: {
            "endpoint": "Azure OpenAI endpoint",
            "deploymentName": "Deployment name",
            "apiVersion": "API version"
        }
    """
    return jsonify({
        "endpoint": config.AZURE_OPENAI_ENDPOINT,
        "deploymentName": config.DEPLOYMENT_NAME,
        "apiVersion": "2023-06-01-preview"
    })


@api.route("/config/cognitive-search", methods=["GET"])
def get_cognitive_search_config():
    """
    Get Cognitive Search configuration if available.
    
    Returns:
        JSON: Configuration dictionary with enabled status and credentials,
              or {"enabled": false} if not configured
    """
    return jsonify(config.get_cognitive_search_config())


@api.route("/config/stt-tts", methods=["GET"])
def get_stt_tts_config():
    """
    Get Speech-to-Text and Text-to-Speech configuration.
    
    Returns:
        JSON: {
            "sttLocales": "Comma-separated locales",
            "ttsVoice": "TTS voice name",
            "customVoiceEndpointId": "Custom voice endpoint ID",
            "continuousConversation": false
        }
    """
    return jsonify({
        "sttLocales": config.STT_LOCALES,
        "ttsVoice": config.TTS_VOICE,
        "customVoiceEndpointId": config.CUSTOM_VOICE_ENDPOINT_ID,
        "continuousConversation": config.CONTINUOUS_CONVERSATION
    })


@api.route("/config/avatar", methods=["GET"])
def get_avatar_config():
    """
    Get Avatar configuration for audio synthesis.
    
    Returns:
        JSON: Dictionary containing all avatar-related settings
    """
    return jsonify({
        "character": config.AVATAR_CHARACTER,
        "style": config.AVATAR_STYLE,
        "photoAvatar": config.PHOTO_AVATAR,
        "customized": config.CUSTOMIZED_AVATAR,
        "useBuiltInVoice": config.USE_BUILT_IN_VOICE,
        "autoReconnect": config.AUTO_RECONNECT_AVATAR,
        "useLocalVideoForIdle": config.USE_LOCAL_VIDEO_FOR_IDLE,
        "showSubtitles": config.SHOW_SUBTITLES
    })


@api.route("/config/system-prompt", methods=["GET"])
def get_system_prompt():
    """
    Get the system prompt used for AI conversations.
    
    Returns:
        JSON: {
            "prompt": "System prompt text"
        }
    """
    return jsonify({
        "prompt": config.SYSTEM_PROMPT
    })


@api.route("/config/all", methods=["GET"])
def get_all_config():
    """
    Get all configuration in one endpoint.
    
    This is the primary configuration endpoint used by the frontend
    to initialize the application.
    
    Returns:
        JSON: Complete configuration dictionary with all service settings
    """
    return jsonify(build_config_response())


@api.route("/config/face-detection", methods=["GET", "PUT"])
def face_detection_config():
    """
    GET: Face detection configuration. "method" is the active backend (from toggle or config).
    PUT: Set method. Body: {"method": "mediapipe" | "azure_face_api" | "auto" | "unified"}.
    """
    if request.method == "GET":
        cfg = dict(config.get_face_detection_config())
        cfg["method"] = get_face_detection_method()
        return jsonify(cfg)
    if request.method == "PUT":
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json(silent=True) or {}
        method = data.get("method")
        if not method:
            return jsonify({"error": "Missing 'method'"}), 400
        try:
            set_face_detection_method(method)
            return jsonify({"method": get_face_detection_method()})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "Method not allowed"}), 405


@api.route("/config/azure-face-api", methods=["GET"])
def get_azure_face_api_config():
    """
    Get Azure Face API configuration if available.
    
    Returns:
        JSON: Configuration dictionary with enabled status and credentials,
              or {"enabled": false} if not configured
    """
    return jsonify(config.get_azure_face_api_config())


@api.route("/weights/signifiers", methods=["GET", "PUT"])
def signifier_weights_route():
    """
    GET: Return current signifier [30], group [4], and fusion weights. Loads from URL/file if not yet loaded.
    PUT: Update from ML backend. Body: {"signifier": [30], "group": [4], "fusion": {"azure": float, "mediapipe": float}}. Partial update.
    """
    if request.method == "GET":
        try:
            signifier_weights.load_weights()
            out = dict(signifier_weights.get_weights())
            azure_w, mp_w = signifier_weights.get_fusion_weights()
            out["fusion"] = {"azure": azure_w, "mediapipe": mp_w}
            return jsonify(out)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    if request.method == "PUT":
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json(silent=True) or {}
        try:
            out = signifier_weights.set_weights(
                signifier=data.get("signifier"),
                group=data.get("group"),
            )
            fusion = data.get("fusion")
            if isinstance(fusion, dict):
                a, m = fusion.get("azure"), fusion.get("mediapipe")
                if isinstance(a, (int, float)) and isinstance(m, (int, float)):
                    signifier_weights.set_fusion_weights(float(a), float(m))
            azure_w, mp_w = signifier_weights.get_fusion_weights()
            out["fusion"] = {"azure": azure_w, "mediapipe": mp_w}
            return jsonify(out)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Method not allowed"}), 405


# ============================================================================
# Avatar Routes (for audio output)
# ============================================================================

@api.route("/avatar/relay-token", methods=["GET"])
def get_avatar_relay_token():
    """
    Get WebRTC relay token for avatar audio connection.
    
    This token is used to establish WebRTC connections for avatar audio output.
    
    Returns:
        JSON: Dictionary containing relay token information (Urls, Username, Password)
    """
    try:
        relay_data = speech_service.get_avatar_relay_token()
        return jsonify(relay_data)
    except Exception as e:
        return jsonify({
            "error": "Failed to get relay token",
            "details": str(e)
        }), 502


# ============================================================================
# Engagement State Detection Routes
# ============================================================================

@api.route("/engagement/start", methods=["POST"])
def start_engagement_detection():
    """
    Start engagement state detection from a video source.
    
    Request Body:
        {
            "sourceType": "webcam" | "file" | "stream" | "partner",
            "sourcePath": "optional path for file/stream sources (not used for partner)"
        }
    Detection method is chosen automatically (device + network); no user option.
    
    Returns:
        JSON: {
            "success": true,
            "message": "Detection started",
            "detectionMethod": resolved method (mediapipe | azure_face_api | unified)
        }
    """
    global engagement_detector
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(silent=True) or {}
    source_type_str = data.get("sourceType", "webcam").lower()
    source_path = data.get("sourcePath")
    # App always chooses optimal method (auto); no user override
    detection_method = getattr(config, "FACE_DETECTION_METHOD", "auto").lower() or "auto"
    
    # Map string to enum
    source_type_map = {
        "webcam": VideoSourceType.WEBCAM,
        "file": VideoSourceType.FILE,
        "stream": VideoSourceType.STREAM,
        "partner": VideoSourceType.PARTNER
    }
    
    source_type = source_type_map.get(source_type_str)
    if not source_type:
        return jsonify({
            "error": f"Invalid sourceType: {source_type_str}. Must be 'webcam', 'file', 'stream', or 'partner'"
        }), 400
    
    # Partner source does not use sourcePath; frames are pushed via POST /engagement/partner-frame
    if source_type == VideoSourceType.PARTNER:
        source_path = None
    
    try:
        if engagement_detector:
            engagement_detector.stop_detection()

        lightweight = getattr(config, "LIGHTWEIGHT_MODE", False)
        if lightweight:
            detection_method = "mediapipe"
        signifier_weights.load_weights()

        engagement_detector = EngagementStateDetector(
            detection_method=detection_method,
            lightweight_mode=lightweight,
        )
        
        if not engagement_detector.start_detection(source_type, source_path):
            return jsonify({
                "error": "Failed to start detection. Check video source."
            }), 500
        
        return jsonify({
            "success": True,
            "message": f"Engagement detection started from {source_type_str}",
            "detectionMethod": engagement_detector.detection_method,
            "lightweightMode": getattr(engagement_detector, "lightweight_mode", False),
        })
    
    except Exception as e:
        return jsonify({
            "error": "Failed to start engagement detection",
            "details": str(e)
        }), 500


@api.route("/engagement/upload-video", methods=["POST"])
def upload_video_file():
    """
    Upload a video file for engagement detection.
    
    Returns:
        JSON: {
            "success": true,
            "filePath": "path/to/uploaded/file",
            "message": "File uploaded successfully"
        }
    """
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Create uploads directory if it doesn't exist
        import os
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        filename = file.filename
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        return jsonify({
            "success": True,
            "filePath": filepath,
            "message": "File uploaded successfully"
        })
    
    except Exception as e:
        return jsonify({
            "error": "Failed to upload video file",
            "details": str(e)
        }), 500


@api.route("/engagement/partner-frame", methods=["POST"])
def partner_frame():
    """
    Receive a single frame from the browser (Meeting Partner Video source).
    Expects raw JPEG body or multipart/form-data with an image file.
    Used when sourceType is 'partner': frontend captures from getDisplayMedia
    and POSTs frames here; the engagement detector reads them via get_partner_frame().
    """
    try:
        data = request.get_data()
        if not data:
            # Try form file (e.g. multipart with "frame" or "image")
            if request.files:
                f = request.files.get("frame") or request.files.get("image") or next(iter(request.files.values()), None)
                if f:
                    data = f.read()
        if not data:
            return jsonify({"error": "No image data"}), 400
        if not set_partner_frame_from_bytes(data):
            return jsonify({"error": "Invalid or unsupported image"}), 400
        return "", 204
    except Exception as e:
        return jsonify({"error": "Failed to process frame", "details": str(e)}), 500


@api.route("/engagement/transcript", methods=["POST"])
def engagement_transcript():
    """
    Receive recent speech transcript from the meeting partner's audio (Meeting Partner Video).
    Used when sourceType is 'partner': frontend runs STT on getDisplayMedia audio
    and POSTs transcript here. Also checks for B2B-relevant trigger phrases; if found,
    sets a pending aural alert (returned on next GET /engagement/state).
    Body: JSON { "text": "..." } or plain text.
    """
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "") or ""
        else:
            text = (request.get_data(as_text=True) or "").strip()
        if text:
            append_transcript(text)
            # Check for phrase-triggered insights (objection, interest, confusion, etc.)
            match = _check_for_trigger_phrases(text)
            if match:
                category, phrase = match
                append_speech_tag(category, phrase)  # For multimodal composite detection
                _set_pending_aural_alert(category, phrase)
        return "", 204
    except Exception as e:
        return jsonify({"error": "Failed to process transcript", "details": str(e)}), 500


@api.route("/engagement/video-feed", methods=["GET"])
def engagement_video_feed():
    """
    Stream the engagement detection video source as MJPEG.

    Returns:
        Response: multipart/x-mixed-replace stream of JPEG frames when
        engagement is running; 404 when not.
    """
    global engagement_detector

    if not engagement_detector or not engagement_detector.is_running:
        return jsonify({"error": "Engagement detection not started"}), 404

    boundary = b"frame"

    def generate():
        while True:
            if not engagement_detector or not engagement_detector.is_running:
                break
            jpeg = engagement_detector.get_last_frame_jpeg()
            if jpeg:
                part = (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                yield part
            time.sleep(0.033)  # ~30 FPS

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.route("/engagement/stop", methods=["POST"])
def stop_engagement_detection():
    """
    Stop engagement state detection.
    
    Returns:
        JSON: {
            "success": true,
            "message": "Detection stopped"
        }
    """
    global engagement_detector
    
    try:
        if engagement_detector:
            engagement_detector.stop_detection()
            engagement_detector = None
        clear_transcript()
        clear_pending_aural_alert()
        clear_recent_speech_tags()
        clear_opportunity_state()

        return jsonify({
            "success": True,
            "message": "Engagement detection stopped"
        })
    
    except Exception as e:
        return jsonify({
            "error": "Failed to stop engagement detection",
            "details": str(e)
        }), 500


@api.route("/engagement/debug", methods=["GET"])
def get_engagement_debug():
    """
    Get debug information about engagement detection.
    
    Returns:
        JSON: Debug information including FPS, frame count, etc.
    """
    global engagement_detector
    
    if not engagement_detector:
        return jsonify({
            "error": "Engagement detection not started",
            "detector_running": False
        }), 404
    
    try:
        state = engagement_detector.get_current_state()
        fps = engagement_detector.get_fps()
        
        debug_info = {
            "detector_running": engagement_detector.is_running,
            "detection_method": getattr(engagement_detector, "detection_method", "unknown"),
            "fps": fps,
            "has_state": state is not None,
            "consecutive_no_face": getattr(engagement_detector, "consecutive_no_face_frames", None),
        }
        
        if state:
            debug_info.update({
                "score": state.score,
                "level": state.level.name if state.level else "UNKNOWN",
                "face_detected": state.face_detected,
                "confidence": state.confidence,
                "metrics": {
                    "attention": state.metrics.attention,
                    "eye_contact": state.metrics.eye_contact,
                    "facial_expressiveness": state.metrics.facial_expressiveness,
                    "head_movement": state.metrics.head_movement,
                    "symmetry": state.metrics.symmetry,
                    "mouth_activity": state.metrics.mouth_activity
                }
            })
        
        return jsonify(debug_info)
    
    except Exception as e:
        return jsonify({
            "error": "Failed to get debug info",
            "details": str(e)
        }), 500


@api.route("/engagement/state", methods=["GET"])
def get_engagement_state():
    """
    Get current engagement state.
    
    Returns:
        JSON: {
            "score": 75.5,
            "level": "HIGH",
            "faceDetected": true,
            "confidence": 0.85,
            "metrics": {
                "attention": 80.0,
                "eyeContact": 75.0,
                "facialExpressiveness": 70.0,
                "headMovement": 85.0,
                "symmetry": 90.0,
                "mouthActivity": 65.0
            },
            "context": {
                "summary": "...",
                "levelDescription": "...",
                "keyIndicators": [...],
                "suggestedActions": [...],
                "riskFactors": [...],
                "opportunities": [...]
            }
        }
    """
    global engagement_detector
    
    if not engagement_detector:
        return jsonify({
            "error": "Engagement detection not started",
            "score": 0,
            "level": "UNKNOWN",
            "faceDetected": False
        }), 404
    
    try:
        state = engagement_detector.get_current_state()
        
        if not state:
            return jsonify({
                "score": 0,
                "level": "UNKNOWN",
                "faceDetected": False,
                "confidence": 0.0,
                "message": "No engagement data available yet",
                "detectionMethod": getattr(engagement_detector, "detection_method", None) or "mediapipe",
                "signifierScores": None,
                "azureMetrics": None,
            })
        
        # Format context for JSON response
        context_dict = {
            "summary": state.context.summary,
            "levelDescription": state.context.level_description,
            "keyIndicators": state.context.key_indicators,
            "suggestedActions": state.context.suggested_actions,
            "riskFactors": state.context.risk_factors,
            "opportunities": state.context.opportunities
        }
        
        # Format metrics for JSON response
        metrics_dict = {
            "attention": state.metrics.attention,
            "eyeContact": state.metrics.eye_contact,
            "facialExpressiveness": state.metrics.facial_expressiveness,
            "headMovement": state.metrics.head_movement,
            "symmetry": state.metrics.symmetry,
            "mouthActivity": state.metrics.mouth_activity
        }
        
        # Sanitize signifier scores (0 or 100 only; no NaN/Inf)
        sig_scores = state.signifier_scores
        if sig_scores and isinstance(sig_scores, dict):
            clean = {}
            for k, v in sig_scores.items():
                try:
                    f = float(v)
                    if not math.isfinite(f):
                        clean[k] = 0.0
                    elif f >= 50.0:
                        clean[k] = 100.0
                    else:
                        clean[k] = 0.0
                except (TypeError, ValueError):
                    clean[k] = 0.0
            sig_scores = clean
        elif not sig_scores:
            sig_scores = None

        # Binarize Azure metrics for display (base/composite -> 0 or 100)
        azure_metrics_out = None
        if state.azure_metrics and isinstance(state.azure_metrics, dict):
            base = state.azure_metrics.get("base") or {}
            composite = state.azure_metrics.get("composite") or {}
            azure_metrics_out = {
                "base": binarize_metrics(base, up=55.0),
                "composite": binarize_metrics(composite, up=55.0),
                "score": state.azure_metrics.get("score", 0.0),
            }

        # Return engagement state with all details
        response_data = {
            "score": float(state.score),
            "level": state.level.name if state.level else "UNKNOWN",
            "faceDetected": bool(state.face_detected),
            "confidence": float(state.confidence),
            "timestamp": float(state.timestamp),
            "metrics": metrics_dict,
            "context": context_dict,
            "fps": float(engagement_detector.get_fps()),
            "signifierScores": sig_scores,
            "detectionMethod": state.detection_method if state.detection_method else "mediapipe",
            "azureMetrics": azure_metrics_out,
        }
        
        # Include alert if present (spike or phrase-triggered); consumed on first return.
        # Global buffer between popups to avoid overwhelming user.
        m = response_data.get("metrics") or {}
        metrics_summary = {
            "attention": m.get("attention"),
            "eyeContact": m.get("eyeContact"),
            "facialExpressiveness": m.get("facialExpressiveness"),
        }
        engagement_alert = engagement_detector.get_pending_alert()
        if engagement_alert:
            alert = engagement_alert
            from_engagement = True
        else:
            aural = get_pending_aural_alert()
            alert = aural
            from_engagement = False
        if alert and not engagement_detector.can_show_insight():
            alert = None
            from_engagement = False
        # Aural (phrase-triggered) insights: show only when at least one visual feature is present
        if alert and not from_engagement:
            score = float(state.score) if state else 0.0
            sig = sig_scores or {}
            detected_signifiers = sum(1 for v in sig.values() if float(v) == 100.0)
            has_visual = score >= 60.0 or detected_signifiers >= 2
            if not has_visual:
                alert = None
        if alert:
            if from_engagement:
                engagement_detector.clear_pending_alert()
                if alert.get("type") in ("spike", "composite_100"):
                    try:
                        grp = alert.get("group", "g1")
                        if grp == "composite":
                            grp = "g1"
                        generated = generate_insight_for_spike(
                            group=grp,
                            metrics_summary=metrics_summary,
                            recent_transcript=get_recent_transcript(),
                        )
                        alert = dict(alert)
                        alert["message"] = (generated or "").strip() or alert.get("message", "Notable change in engagement.")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning("Insight generation failed: %s", e)
                elif alert.get("type") == "opportunity":
                    try:
                        generated = generate_insight_for_opportunity(
                            opportunity_id=alert.get("opportunity_id", ""),
                            context=alert.get("context"),
                            metrics_summary=metrics_summary,
                            recent_transcript=get_recent_transcript(),
                        )
                        alert = dict(alert)
                        alert["message"] = (generated or "").strip() or alert.get("message", "Opportunity detected—consider acting on it.")
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning("Opportunity insight generation failed: %s", e)
            else:
                get_and_clear_pending_aural_alert()
                try:
                    generated = generate_insight_for_aural_trigger(
                        category=alert.get("category", "interest"),
                        phrase=alert.get("phrase", ""),
                        metrics_summary=metrics_summary,
                    )
                    alert = {"type": "aural", "message": (generated or "").strip() or "Notable comment—consider responding."}
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Aural insight generation failed: %s", e)
                    alert = {"type": "aural", "message": "They said something noteworthy—consider acknowledging their point."}
            engagement_detector.record_insight_shown()
            response_data["alert"] = alert
        
        # Add cache control headers to ensure fresh data
        response = jsonify(response_data)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
    
    except Exception as e:
        return jsonify({
            "error": "Failed to get engagement state",
            "details": str(e)
        }), 500


@api.route("/engagement/score-breakdown", methods=["GET"])
def get_engagement_score_breakdown():
    """
    Get a step-by-step breakdown of how the current engagement score is calculated.
    Used by the frontend "How is the score calculated?" view.
    Returns detection method, formula steps, group/composite values, adjustments, and final score.
    """
    global engagement_detector
    if not engagement_detector:
        return jsonify({
            "error": "Engagement detection not started",
            "breakdown": None
        }), 404
    try:
        breakdown = engagement_detector.get_score_breakdown()
        if breakdown is None:
            return jsonify({
                "error": "No engagement data available yet",
                "breakdown": None
            }), 404
        return jsonify(breakdown)
    except Exception as e:
        return jsonify({
            "error": "Failed to get score breakdown",
            "details": str(e),
            "breakdown": None
        }), 500


@api.route("/engagement/context", methods=["GET"])
def get_engagement_context():
    """
    Get formatted engagement context for AI coaching.
    
    This endpoint returns the engagement context formatted as a string
    that can be directly included in AI prompts.
    
    Returns:
        JSON: {
            "context": "Formatted context string for AI",
            "score": 75.5,
            "level": "HIGH"
        }
    """
    global engagement_detector
    
    if not engagement_detector:
        return jsonify({
            "error": "Engagement detection not started"
        }), 404
    
    try:
        state = engagement_detector.get_current_state()
        
        if not state:
            return jsonify({
                "error": "No engagement data available"
            }), 404
        
        formatted_context = context_generator.format_for_ai(state.context)
        
        return jsonify({
            "context": formatted_context,
            "score": state.score,
            "level": state.level.name
        })
    
    except Exception as e:
        return jsonify({
            "error": "Failed to get engagement context",
            "details": str(e)
        }), 500
