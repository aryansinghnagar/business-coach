"""
Flask route blueprints for Business Meeting Copilot.

All routes in one module: pages, chat, config, engagement.
Register via register_routes(app) from app.py.
"""

import logging
import math
import os
import time

from flask import Blueprint, Flask, request, jsonify, Response, redirect, send_from_directory, url_for

import config
import services as engagement_api
from services import (
    get_speech_service,
    get_foundry_service,
    append_speech_tag,
    append_transcript,
    check_speech_cues,
    clear_pending_aural_alert,
    clear_recent_speech_tags,
    generate_insight_for_aural_trigger,
    generate_insight_for_opportunity,
    generate_insight_for_spike,
    get_and_clear_pending_aural_alert,
    get_discourse_boost,
    get_pending_aural_alert,
    get_recent_speech_tags,
    get_recent_transcript,
    _set_pending_aural_alert,
    append_acoustic_windows,
    get_recent_acoustic_tags,
)
from config import build_config_response, get_face_detection_method, set_face_detection_method
from helpers import set_partner_frame_from_bytes, VideoSourceType

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pages & static
# -----------------------------------------------------------------------------
pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Serve the main index.html page."""
    try:
        return send_from_directory(".", "index.html")
    except FileNotFoundError:
        return jsonify({"error": "index.html not found"}), 404


@pages_bp.route("/dashboard")
def dashboard():
    """Redirect to main app; engagement dashboard is now inline in the left panel."""
    return redirect(url_for("pages.index"))


@pages_bp.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files (JS, CSS, etc.)."""
    try:
        return send_from_directory("static", filename)
    except FileNotFoundError:
        return jsonify({"error": f"Static file not found: {filename}"}), 404


@pages_bp.route("/favicon.ico")
def favicon():
    """Handle favicon requests."""
    return "", 204


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------
chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """Non-streaming chat endpoint."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "Missing message"}), 400
    try:
        messages = [{"role": "user", "content": user_input}]
        response_text = get_foundry_service().chat_completion(
            messages=messages,
            system_prompt=config.SYSTEM_PROMPT,
        )
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": "Failed to get AI response", "details": str(e)}), 500


@chat_bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat with optional engagement context and On Your Data."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    enable_oyd = data.get("enableOyd", False)
    system_prompt = data.get("systemPrompt", config.SYSTEM_PROMPT)
    include_engagement = data.get("includeEngagement", True)
    if not messages:
        return jsonify({"error": "Missing messages"}), 400

    store = engagement_api.get_context_store()
    if include_engagement:
        try:
            full_context = engagement_api.build_engagement_context_bundle(None)
            if messages and messages[-1].get("role") == "user":
                content = messages[-1].get("content")
                if isinstance(content, str):
                    store["last_context_sent"] = full_context
                    messages[-1] = dict(messages[-1], content=full_context + "\n\n" + content)
        except Exception as e:
            print(f"Warning: Could not include engagement context: {e}")

    try:
        def generate():
            try:
                for chunk in get_foundry_service().stream_chat_completion(
                    messages=messages,
                    enable_oyd=enable_oyd,
                    system_prompt=system_prompt,
                ):
                    yield chunk
            except Exception as e:
                yield f'data: {{"error": "Streaming error: {str(e)}"}}\n\n'

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"error": "Failed to get AI response", "details": str(e)}), 500


# -----------------------------------------------------------------------------
# Config & weights
# -----------------------------------------------------------------------------
config_bp = Blueprint("config_routes", __name__)


@config_bp.route("/speech/token", methods=["GET"])
def get_speech_token():
    """Get Azure Speech Service access token."""
    try:
        token_data = get_speech_service().get_speech_token()
        return jsonify(token_data)
    except ValueError as e:
        return jsonify({"error": "Speech service not configured", "details": str(e)}), 503
    except Exception as e:
        status_code = 504 if "timeout" in str(e).lower() else 502
        return jsonify({"error": "Failed to get speech token", "details": str(e)}), status_code


@config_bp.route("/config/speech", methods=["GET"])
def get_speech_config():
    """Get speech service configuration (region only)."""
    return jsonify({
        "region": config.SPEECH_REGION,
        "privateEndpointEnabled": config.SPEECH_PRIVATE_ENDPOINT_ENABLED,
    })


@config_bp.route("/config/openai", methods=["GET"])
@config_bp.route("/config/foundry", methods=["GET"])
def get_foundry_config():
    """Get Azure AI Foundry configuration."""
    return jsonify({
        "endpoint": config.AZURE_FOUNDRY_ENDPOINT,
        "deploymentName": config.FOUNDRY_DEPLOYMENT_NAME,
        "apiVersion": "2023-06-01-preview",
    })


@config_bp.route("/config/cognitive-search", methods=["GET"])
def get_cognitive_search_config_route():
    """Get Cognitive Search configuration if available."""
    return jsonify(config.get_cognitive_search_config())


@config_bp.route("/config/stt-tts", methods=["GET"])
def get_stt_tts_config():
    """Get STT/TTS configuration."""
    return jsonify({
        "sttLocales": config.STT_LOCALES,
        "ttsVoice": config.TTS_VOICE,
        "customVoiceEndpointId": config.CUSTOM_VOICE_ENDPOINT_ID,
        "continuousConversation": config.CONTINUOUS_CONVERSATION,
    })


@config_bp.route("/config/avatar", methods=["GET"])
def get_avatar_config():
    """Get Avatar configuration."""
    return jsonify({
        "character": config.AVATAR_CHARACTER,
        "style": config.AVATAR_STYLE,
        "photoAvatar": config.PHOTO_AVATAR,
        "customized": config.CUSTOMIZED_AVATAR,
        "useBuiltInVoice": config.USE_BUILT_IN_VOICE,
        "autoReconnect": config.AUTO_RECONNECT_AVATAR,
        "useLocalVideoForIdle": config.USE_LOCAL_VIDEO_FOR_IDLE,
        "showSubtitles": config.SHOW_SUBTITLES,
    })


@config_bp.route("/config/system-prompt", methods=["GET"])
def get_system_prompt():
    """Get system prompt for AI conversations."""
    return jsonify({"prompt": config.SYSTEM_PROMPT})


@config_bp.route("/config/all", methods=["GET"])
def get_all_config():
    """Get all configuration in one endpoint."""
    return jsonify(build_config_response())


@config_bp.route("/config/face-detection", methods=["GET", "PUT"])
def face_detection_config():
    """GET: Face detection config. PUT: Set method (body: {"method": "mediapipe"|"azure_face_api"|"auto"|"unified"})."""
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


@config_bp.route("/config/azure-face-api", methods=["GET"])
def get_azure_face_api_config():
    """Get Azure Face API configuration if available."""
    return jsonify(config.get_azure_face_api_config())


# -----------------------------------------------------------------------------
# Engagement detection
# -----------------------------------------------------------------------------
engagement_bp = Blueprint("engagement", __name__)


@engagement_bp.route("/engagement/start", methods=["POST"])
def start_engagement_detection():
    """Start engagement state detection from a video source."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True) or {}
    source_type_str = (data.get("sourceType") or "webcam").lower()
    source_path = data.get("sourcePath")
    source_type_map = {
        "webcam": VideoSourceType.WEBCAM,
        "file": VideoSourceType.FILE,
        "stream": VideoSourceType.STREAM,
        "partner": VideoSourceType.PARTNER,
    }
    source_type = source_type_map.get(source_type_str)
    if not source_type:
        return jsonify({
            "error": f"Invalid sourceType: {source_type_str}. Must be 'webcam', 'file', 'stream', or 'partner'"
        }), 400
    if source_type == VideoSourceType.PARTNER:
        source_path = None
    try:
        ok, result = engagement_api.start_detection(source_type, source_path)
        if not ok:
            return jsonify({"error": result}), 500
        return jsonify({
            "success": True,
            "message": f"Engagement detection started from {source_type_str}",
            "detectionMethod": result["detectionMethod"],
            "lightweightMode": result["lightweightMode"],
        })
    except Exception as e:
        return jsonify({"error": "Failed to start engagement detection", "details": str(e)}), 500


@engagement_bp.route("/engagement/upload-video", methods=["POST"])
def upload_video_file():
    """Upload a video file for engagement detection."""
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        file = request.files["video"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)
        return jsonify({"success": True, "filePath": filepath, "message": "File uploaded successfully"})
    except Exception as e:
        return jsonify({"error": "Failed to upload video file", "details": str(e)}), 500


@engagement_bp.route("/engagement/partner-frame", methods=["POST"])
def partner_frame():
    """Receive a single frame from the browser (Meeting Partner Video source)."""
    try:
        data = request.get_data()
        if not data and request.files:
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


@engagement_bp.route("/engagement/transcript", methods=["POST"])
def engagement_transcript():
    """Receive recent speech transcript; check B2B cues and set pending aural alert if needed."""
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()
        else:
            text = (request.get_data(as_text=True) or "").strip()
        if text:
            append_transcript(text)
            matches = check_speech_cues(text)
            primary_alert_set = False
            for category, phrase in matches:
                discourse_boost = get_discourse_boost(text, category)
                append_speech_tag(category, phrase, discourse_boost=discourse_boost)
                if not primary_alert_set and _set_pending_aural_alert(category, phrase):
                    primary_alert_set = True
        return "", 204
    except Exception as e:
        return jsonify({"error": "Failed to process transcript", "details": str(e)}), 500


@engagement_bp.route("/engagement/acoustic-context", methods=["POST"])
def engagement_acoustic_context():
    """Receive windowed acoustic features from partner or mic audio."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json(silent=True) or {}
        windows = data.get("windows")
        if windows is None and isinstance(data, dict) and "loudness_norm" in data:
            windows = [data]
        if not windows or not isinstance(windows, list):
            return "", 204
        append_acoustic_windows(windows)
        return "", 204
    except Exception as e:
        return jsonify({"error": "Failed to process acoustic context", "details": str(e)}), 500


@engagement_bp.route("/engagement/video-feed", methods=["GET"])
def engagement_video_feed():
    """Stream engagement detection video source as MJPEG."""
    det = engagement_api.get_detector()
    if not det or not det.is_running:
        return jsonify({"error": "Engagement detection not started"}), 404
    boundary = b"frame"

    def generate():
        while True:
            d = engagement_api.get_detector()
            if not d or not d.is_running:
                break
            jpeg = d.get_last_frame_jpeg()
            if jpeg:
                part = (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                yield part
            time.sleep(1.0 / max(30, min(60, config.TARGET_FPS_MAX)))

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@engagement_bp.route("/engagement/stop", methods=["POST"])
def stop_engagement_detection():
    """Stop engagement state detection."""
    try:
        engagement_api.stop_detection()
        return jsonify({"success": True, "message": "Engagement detection stopped"})
    except Exception as e:
        return jsonify({"error": "Failed to stop engagement detection", "details": str(e)}), 500


@engagement_bp.route("/engagement/debug", methods=["GET"])
def get_engagement_debug():
    """Get debug information about engagement detection."""
    det = engagement_api.get_detector()
    if not det:
        return jsonify({"error": "Engagement detection not started", "detector_running": False}), 404
    try:
        state = det.get_current_state()
        fps = det.get_fps()
        debug_info = {
            "detector_running": det.is_running,
            "detection_method": getattr(det, "detection_method", None) or "unknown",
            "fps": fps,
            "has_state": state is not None,
            "consecutive_no_face": det.consecutive_no_face_frames,
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
                    "mouth_activity": state.metrics.mouth_activity,
                },
            })
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": "Failed to get debug info", "details": str(e)}), 500


@engagement_bp.route("/engagement/state", methods=["GET"])
def get_engagement_state():
    """Get current engagement state and optional alert with insight text."""
    engagement_api.update_last_request()
    det = engagement_api.get_detector()
    if not det:
        # Return 200 so polling does not log 404; frontend uses detectionStarted to show "not running"
        return jsonify({
            "detectionStarted": False,
            "score": 0,
            "level": "UNKNOWN",
            "faceDetected": False,
        })

    try:
        state = det.get_current_state()
        if not state:
            return jsonify({
                "detectionStarted": True,
                "score": 0,
                "level": "UNKNOWN",
                "faceDetected": False,
                "confidence": 0.0,
                "message": "No engagement data available yet",
                "detectionMethod": (det.detection_method or "mediapipe"),
                "signifierScores": None,
                "azureMetrics": None,
            })

        ctx = getattr(state, "context", None)
        m = getattr(state, "metrics", None)
        context_dict = {
            "summary": getattr(ctx, "summary", "") or "",
            "levelDescription": getattr(ctx, "level_description", "") or "",
            "keyIndicators": getattr(ctx, "key_indicators", []) or [],
            "suggestedActions": getattr(ctx, "suggested_actions", []) or [],
            "riskFactors": getattr(ctx, "risk_factors", []) or [],
            "opportunities": getattr(ctx, "opportunities", []) or [],
        }
        metrics_dict = {
            "attention": float(getattr(m, "attention", 0) or 0),
            "eyeContact": float(getattr(m, "eye_contact", 0) or 0),
            "facialExpressiveness": float(getattr(m, "facial_expressiveness", 0) or 0),
            "headMovement": float(getattr(m, "head_movement", 0) or 0),
            "symmetry": float(getattr(m, "symmetry", 0) or 0),
            "mouthActivity": float(getattr(m, "mouth_activity", 0) or 0),
        }
        sig_scores = state.signifier_scores
        if sig_scores and isinstance(sig_scores, dict):
            clean = {}
            for k, v in sig_scores.items():
                try:
                    f = float(v)
                    clean[k] = 0.0 if not math.isfinite(f) else max(0.0, min(100.0, f))
                except (TypeError, ValueError):
                    clean[k] = 0.0
            sig_scores = clean
        else:
            sig_scores = None
        azure_metrics_out = None
        if state.azure_metrics and isinstance(state.azure_metrics, dict):
            base = state.azure_metrics.get("base") or {}
            composite = state.azure_metrics.get("composite") or {}
            azure_metrics_out = {
                "base": base,
                "composite": composite,
                "score": state.azure_metrics.get("score", 0.0),
            }
        composite_metrics = getattr(state, "composite_metrics", None) or {}
        metrics_summary = {
            "attention": metrics_dict.get("attention"),
            "eyeContact": metrics_dict.get("eyeContact"),
            "facialExpressiveness": metrics_dict.get("facialExpressiveness"),
            "headMovement": metrics_dict.get("headMovement"),
            "symmetry": metrics_dict.get("symmetry"),
            "mouthActivity": metrics_dict.get("mouthActivity"),
        }
        recent_speech_tags = get_recent_speech_tags(12)
        acoustic_tags_list = get_recent_acoustic_tags()
        try:
            score_val = float(state.score) if state.score is not None else 0.0
        except (TypeError, ValueError):
            score_val = 0.0
        try:
            ts_val = float(state.timestamp) if getattr(state, "timestamp", None) is not None else time.time()
        except (TypeError, ValueError):
            ts_val = time.time()
        try:
            conf_val = float(state.confidence) if getattr(state, "confidence", None) is not None else 0.0
        except (TypeError, ValueError):
            conf_val = 0.0
        try:
            fps_val = float(det.get_fps())
        except (TypeError, ValueError, AttributeError):
            fps_val = 0.0
        level_name = "UNKNOWN"
        if getattr(state, "level", None) is not None and hasattr(state.level, "name"):
            level_name = state.level.name
        response_data = {
            "detectionStarted": True,
            "score": score_val,
            "level": level_name,
            "faceDetected": bool(getattr(state, "face_detected", False)),
            "confidence": conf_val,
            "timestamp": ts_val,
            "metrics": metrics_dict,
            "context": context_dict,
            "fps": fps_val,
            "signifierScores": sig_scores,
            "detectionMethod": getattr(state, "detection_method", None) or "mediapipe",
            "azureMetrics": azure_metrics_out,
            "compositeMetrics": composite_metrics,
        }
        signifier_scores = sig_scores or {}
        engagement_alert = det.get_pending_alert()
        if engagement_alert:
            alert, from_engagement = engagement_alert, True
        else:
            alert, from_engagement = get_pending_aural_alert(), False
        if alert and not det.can_show_insight(alert):
            logger.debug("can_show_insight blocked alert: %s", alert.get("type"))
            alert, from_engagement = None, False
        if alert and not from_engagement:
            score = score_val
            sig = sig_scores or {}
            try:
                elevated_signifiers = sum(1 for v in sig.values() if float(v) >= 65.0)
            except (TypeError, ValueError):
                elevated_signifiers = 0
            category = (alert or {}).get("category", "")
            if category in ("objection", "concern", "confusion"):
                has_visual = score >= 50.0 or elevated_signifiers >= 1
            else:
                has_visual = score >= 60.0 or elevated_signifiers >= 2
            if not has_visual:
                alert = None
        if alert:
            try:
                logger.debug("Alert before insight gen: type=%s opportunity=%s group=%s", alert.get("type"), alert.get("opportunity_id"), alert.get("group"))
                fresh_context = engagement_api.build_fresh_insight_context(
                state=state,
                signifier_scores=signifier_scores,
                composite_metrics=composite_metrics,
                metrics_summary=metrics_summary,
                recent_speech_tags=recent_speech_tags,
                acoustic_tags_list=acoustic_tags_list,
                )
                logger.debug("Fresh context built: score=%s len=%d", getattr(state, "score", None), len(fresh_context))
                if from_engagement:
                    det.clear_pending_alert()
                    if alert.get("type") in ("spike", "composite_100"):
                        try:
                            grp = alert.get("group", "g1")
                            if grp == "composite":
                                grp = "g1"
                            generated = generate_insight_for_spike(
                                group=grp,
                                metrics_summary=metrics_summary,
                                recent_transcript=get_recent_transcript(),
                                recent_context_bundle=fresh_context,
                                signifier_scores=signifier_scores,
                                composite_metrics=composite_metrics,
                                recent_speech_tags=recent_speech_tags,
                                acoustic_tags=acoustic_tags_list,
                            )
                            alert = dict(alert)
                            alert["message"] = (generated or "").strip() or alert.get("message", "Notable change in engagement.")
                            logger.debug("Spike insight: %s", (generated or "")[:90])
                        except Exception as e:
                            logger.warning("Insight generation failed: %s", e)
                    elif alert.get("type") == "opportunity":
                        try:
                            generated = generate_insight_for_opportunity(
                                opportunity_id=alert.get("opportunity_id", ""),
                                context=alert.get("context"),
                                metrics_summary=metrics_summary,
                                recent_transcript=get_recent_transcript(),
                                recent_context_bundle=fresh_context,
                                signifier_scores=signifier_scores,
                                composite_metrics=composite_metrics,
                                recent_speech_tags=recent_speech_tags,
                                acoustic_tags=acoustic_tags_list,
                            )
                            alert = dict(alert)
                            alert["message"] = (generated or "").strip() or alert.get("message", "Opportunity detectedâ€”consider acting on it.")
                            logger.debug("Opportunity insight: %s", (generated or "")[:90])
                        except Exception as e:
                            logger.warning("Opportunity insight generation failed: %s", e)
                else:
                    get_and_clear_pending_aural_alert()
                    try:
                        generated = generate_insight_for_aural_trigger(
                            category=alert.get("category", "interest"),
                            phrase=alert.get("phrase", ""),
                            metrics_summary=metrics_summary,
                            recent_context_bundle=fresh_context,
                            signifier_scores=signifier_scores,
                            composite_metrics=composite_metrics,
                            recent_speech_tags=recent_speech_tags,
                            acoustic_tags=acoustic_tags_list,
                        )
                        alert = {"type": "aural", "message": (generated or "").strip() or "Notable commentâ€”consider responding."}
                        logger.debug("Aural insight: %s", (generated or "")[:90])
                    except Exception as e:
                        logger.warning("Aural insight generation failed: %s", e)
                        alert = {"type": "aural", "message": "They said something noteworthyâ€”consider acknowledging their point."}
                det.record_insight_shown(alert)
                response_data["alert"] = alert
            except Exception as e:
                logger.exception("Alert/insight build failed for engagement state; returning state without alert: %s", e)

        response = jsonify(response_data)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        logger.exception("Failed to get engagement state: %s", e)
        return jsonify({"error": "Failed to get engagement state", "details": str(e)}), 500


@engagement_bp.route("/api/engagement/context-and-response", methods=["GET"])
def get_context_and_response():
    """Get last context sent to OpenAI and last AI response (sidebar/dashboard)."""
    store = engagement_api.get_context_store()
    return jsonify({
        "contextSent": store["last_context_sent"],
        "response": store["last_response"],
        "timestamp": store["last_response_timestamp"],
    })


@engagement_bp.route("/api/engagement/record-response", methods=["POST"])
def record_response():
    """Record the full assistant reply after stream completion."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True) or {}
    response_text = (data.get("response") or "").strip()
    store = engagement_api.get_context_store()
    store["last_response"] = response_text if response_text else None
    store["last_response_timestamp"] = time.time() if response_text else None
    return jsonify({"ok": True})


@engagement_bp.route("/api/engagement/set-additional-context", methods=["POST"])
def set_additional_context():
    """Store user-defined context for next batch sent to OpenAI."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True) or {}
    value = (data.get("additionalContext") or "").strip() or None
    store = engagement_api.get_context_store()
    store["pending_user_context"] = value
    return jsonify({"ok": True, "pending": value is not None})


@engagement_bp.route("/api/context-push", methods=["POST", "GET"])
def context_push():
    """Build context bundle, call OpenAI, store context and response, return both."""
    try:
        full_context = engagement_api.build_engagement_context_bundle(None)
        store = engagement_api.get_context_store()
        store["last_context_sent"] = full_context
        prompt = "Given the following real-time meeting context, what should the host know or do right now? Reply in 1â€“3 short sentences."
        messages = [{"role": "user", "content": full_context + "\n\n" + prompt}]
        response_text = get_foundry_service().chat_completion(
            messages=messages,
            system_prompt=config.SYSTEM_PROMPT,
            max_tokens=150,
        )
        response_text = (response_text or "").strip()
        store["last_response"] = response_text
        store["last_response_timestamp"] = time.time()
        return jsonify({"context": full_context, "response": response_text})
    except Exception as e:
        logger.exception("Context push failed: %s", e)
        return jsonify({"error": "Context push failed", "details": str(e)}), 500


@engagement_bp.route("/engagement/score-breakdown", methods=["GET"])
def get_engagement_score_breakdown():
    """Get step-by-step breakdown of current engagement score."""
    det = engagement_api.get_detector()
    if not det:
        return jsonify({"error": "Engagement detection not started", "breakdown": None}), 404
    try:
        breakdown = det.get_score_breakdown()
        if breakdown is None:
            return jsonify({"error": "No engagement data available yet", "breakdown": None}), 404
        return jsonify(breakdown)
    except Exception as e:
        return jsonify({"error": "Failed to get score breakdown", "details": str(e), "breakdown": None}), 500


@engagement_bp.route("/engagement/context", methods=["GET"])
def get_engagement_context():
    """Get formatted engagement context for AI coaching."""
    det = engagement_api.get_detector()
    if not det:
        return jsonify({"error": "Engagement detection not started"}), 404
    try:
        state = det.get_current_state()
        if not state:
            return jsonify({"error": "No engagement data available"}), 404
        formatted_context = engagement_api.get_context_generator().format_for_ai(state.context)
        return jsonify({
            "context": formatted_context,
            "score": state.score,
            "level": state.level.name,
        })
    except Exception as e:
        return jsonify({"error": "Failed to get engagement context", "details": str(e)}), 500


def register_routes(app: Flask) -> None:
    """Register all route blueprints on the app (no URL prefix)."""
    app.register_blueprint(pages_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(engagement_bp)

