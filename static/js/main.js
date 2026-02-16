"use strict";
// ========================================================================
// BUSINESS MEETING COPILOT — MAIN APPLICATION SCRIPT (app-main.js)
// ========================================================================
//
// WHAT THIS FILE DOES (in plain language):
// ----------------------------------------
// This script runs in the browser and ties together everything the user sees
// and does on the main page. It:
//
//   1. INSIGHTS SESSION — When the user clicks "Start Insights Session", we
//      set up speech-to-text (STT) for the user's microphone and (optionally)
//      for the "feed" audio (e.g. meeting partner). We send transcript to the
//      backend for speech cue analysis. Insights are text-only.
//
//   2. ENGAGEMENT — We start the engagement detector with a video source
//      (webcam, partner share, or file). The detector sends frames and gets
//      back scores and alerts. When there is an alert (e.g. spike or opportunity),
//      the backend generates short coaching text and we show it in a popup and
//      in the Insight Transcript box.
//
//   3. VIDEO SOURCE — If no video source is active when the user clicks "Start
//      Insights Session", we show a modal so they can choose: webcam, meeting
//      partner (screen/tab share), or file. After they choose, we start
//      engagement and then start the insights session.
//
//   4. SPOKEN CONTEXT — When the user speaks (microphone), STT text is sent to
//      the backend and to Foundry for intelligent insights; streaming responses
//      appear in the Insight Transcript (text only).
//
// FLOW (simplified):
//   User opens page -> We load config and init modules (session, engagement,
//   video selector, etc.). User selects video source (or we prompt) -> We start
//   engagement detection. User clicks "Start Insights Session" -> We init STT
//   and chat; we poll GET /engagement/state and show alerts as popups + in
//   transcript box.
//
// ========================================================================
// GLOBAL VARIABLES & CONFIGURATION
// ========================================================================
var speechRecognizer;
var feedSpeechRecognizer = null;  // STT on meeting partner (feed) audio for aural cues; transcript sent to backend
var _partnerAudioStream = null   // Partner audio stream stored when source has audio; STT started only on "Start Microphone"
var _partnerAudioStreamFromMic = false  // True when stream is from getUserMedia (mic); stop tracks on cleanup
var peerConnection;
var peerConnectionDataChannel;
var appConfig = null;

// Session Management (modular)
var sessionManager = null;
var engagementDetector = null;
var signifierPanel = null;
var azureMetricsPanel = null;
var compositeMetricsPanel = null;
var videoSourceSelector = null;

// Messages for Foundry streaming (system + user + assistant); used when user speaks or sends a message for insights
var insightMessages = [];

// UI State
var sessionActive = false;
var dataSources = [];

/** Update the Insights Session status pill (Idle / Live). */
function updateSessionStatusIndicator(isLive) {
    var el = document.getElementById('sessionStatusIndicator')
    if (!el) return
    if (isLive) {
        el.textContent = '● Live'
        el.className = 'session-status session-status-live'
        el.title = 'Insights session is running'
    } else {
        el.textContent = '● Idle'
        el.className = 'session-status session-status-idle'
        el.title = 'Insights session status'
    }
}
var videoSourcePromptShown = false;  // Track if video source prompt has been shown
/** When set, run after user selects a video source (e.g. to continue Start Insight Session). */
var _afterVideoSourceSelectedCallback = null;

// ========================================================================
// Metric spike alerts (popup when a metric group spikes)
// ========================================================================
var spikeToastTimeout = null;

/** Delay between each character when streaming insight text (ms). */
var INSIGHT_STREAM_CHAR_DELAY_MS = 24;

/**
 * Stream a string into one or more elements letter-by-letter, then run onComplete.
 * @param {string} text - Full insight text
 * @param {HTMLElement[]} elements - Elements to update (same text appended to each)
 * @param {{ onComplete?: function }} options - onComplete called when streaming finishes
 */
function streamInsightText(text, elements, options) {
    if (!text || !elements || elements.length === 0) {
        if (options && options.onComplete) options.onComplete();
        return;
    }
    var full = String(text);
    var index = 0;
    var delay = typeof INSIGHT_STREAM_CHAR_DELAY_MS === 'number' && INSIGHT_STREAM_CHAR_DELAY_MS > 0 ? INSIGHT_STREAM_CHAR_DELAY_MS : 24;

    function appendNext() {
        if (index >= full.length) {
            if (options && options.onComplete) options.onComplete();
            return;
        }
        index++;
        var segment = full.substring(0, index);
        for (var i = 0; i < elements.length; i++) {
            if (elements[i]) elements[i].textContent = segment;
        }
        setTimeout(appendNext, delay);
    }
    appendNext();
}

/**
 * Create a new transcript entry with timestamp and an empty span for streaming; return that span.
 * Caller will stream insight text into the returned span.
 * @returns {{ time: string, contentSpan: HTMLElement | null }}
 */
function createInsightTranscriptEntryForStreaming() {
    var box = document.getElementById('insightTranscript');
    if (!box) return { time: '', contentSpan: null };
    var time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    if (box.innerHTML === 'No insights yet.' || !box.innerHTML.trim()) {
        box.innerHTML = '';
    }
    var entry = document.createElement('div');
    entry.className = 'insight-entry';
    var timeSpan = document.createElement('span');
    timeSpan.className = 'insight-time';
    timeSpan.textContent = time;
    entry.appendChild(timeSpan);
    entry.appendChild(document.createElement('br'));
    var contentSpan = document.createElement('span');
    contentSpan.className = 'insight-streaming-text';
    entry.appendChild(contentSpan);
    box.appendChild(entry);
    box.scrollTop = box.scrollHeight;
    return { time: time, contentSpan: contentSpan };
}

/**
 * Append an insight response to the sidebar transcript box (for verification).
 * Used when not streaming (e.g. fallback). For streaming, use createInsightTranscriptEntryForStreaming + streamInsightText.
 * @param {string} message - Insight text from Azure AI Foundry
 */
function appendToInsightTranscript(message) {
    var box = document.getElementById('insightTranscript');
    if (!box) return;
    var time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    var entry = '<div class="insight-entry"><span class="insight-time">' + time + '</span><br/>' + String(message).replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>';
    if (box.innerHTML === 'No insights yet.' || !box.innerHTML.trim()) {
        box.innerHTML = entry;
    } else {
        box.innerHTML += entry;
    }
    box.scrollTop = box.scrollHeight;
    updateInsightSpotlight(message, time);
}

/**
 * Update the latest insight spotlight (above Controls) with full text and time.
 */
function updateInsightSpotlight(message, time) {
    var placeholder = document.getElementById('insightSpotlightPlaceholder');
    var textEl = document.getElementById('insightSpotlightText');
    var timeEl = document.getElementById('insightSpotlightTime');
    var linkEl = document.getElementById('insightSpotlightLink');
    if (placeholder) placeholder.style.display = 'none';
    if (textEl) {
        textEl.textContent = String(message).trim();
        textEl.style.display = 'block';
    }
    if (timeEl) {
        timeEl.textContent = time || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        timeEl.style.display = 'block';
    }
    if (linkEl) linkEl.style.display = 'inline';
}

/**
 * Clear the insight transcript (when session stops).
 */
function clearInsightTranscript() {
    var box = document.getElementById('insightTranscript');
    if (box) box.innerHTML = 'No insights yet.';
    // Reset latest insight spotlight
    var placeholder = document.getElementById('insightSpotlightPlaceholder');
    var textEl = document.getElementById('insightSpotlightText');
    var timeEl = document.getElementById('insightSpotlightTime');
    var linkEl = document.getElementById('insightSpotlightLink');
    if (placeholder) placeholder.style.display = 'block';
    if (textEl) { textEl.textContent = ''; textEl.style.display = 'none'; }
    if (timeEl) { timeEl.textContent = ''; timeEl.style.display = 'none'; }
    if (linkEl) linkEl.style.display = 'none';
}

/**
 * Handle metric spike alert: show popup and transcript with insight text streamed letter-by-letter.
 * Popups are shown only when Insights Session is live; metrics still update on dashboard when session is idle.
 */
function handleMetricSpikeAlert(alert) {
    if (!alert || !alert.message) return;
    if (!sessionActive) return;
    var msg = String(alert.message).trim();
    if (!msg) return;

    var toastEl = document.getElementById('metricSpikeToast');
    var layer = document.getElementById('insightPopupLayer');
    var spotlightTextEl = document.getElementById('insightSpotlightText');
    var spotlightTimeEl = document.getElementById('insightSpotlightTime');
    var spotlightPlaceholder = document.getElementById('insightSpotlightPlaceholder');
    var spotlightLink = document.getElementById('insightSpotlightLink');

    // 1. Create transcript entry with empty content span for streaming
    var transcript = createInsightTranscriptEntryForStreaming();
    var contentSpan = transcript.contentSpan;
    var timeStr = transcript.time;

    // 2. Show insight popup (toast empty; text will stream in) and prepare spotlight
    if (toastEl) {
        toastEl.textContent = '';
        toastEl.style.display = 'block';
        toastEl.classList.remove('hiding');
    }
    if (layer) {
        layer.classList.add('has-popup');
        layer.setAttribute('aria-hidden', 'false');
    }
    if (spotlightPlaceholder) spotlightPlaceholder.style.display = 'none';
    if (spotlightTextEl) {
        spotlightTextEl.textContent = '';
        spotlightTextEl.style.display = 'block';
    }
    if (spotlightTimeEl) {
        spotlightTimeEl.textContent = timeStr;
        spotlightTimeEl.style.display = 'block';
    }
    if (spotlightLink) spotlightLink.style.display = 'inline';

    // 3. Stream insight text letter-by-letter into popup, transcript entry, and spotlight
    var targets = [toastEl].filter(Boolean);
    if (contentSpan) targets.push(contentSpan);
    if (spotlightTextEl) targets.push(spotlightTextEl);

    streamInsightText(msg, targets, {
        onComplete: function () {
            if (contentSpan) {
                var box = document.getElementById('insightTranscript');
                if (box) box.scrollTop = box.scrollHeight;
            }
            // After streaming finishes, start 6s timer to hide popup
            if (spikeToastTimeout) clearTimeout(spikeToastTimeout);
            spikeToastTimeout = setTimeout(function () {
                if (toastEl) toastEl.classList.add('hiding');
                spikeToastTimeout = setTimeout(function () {
                    if (toastEl) {
                        toastEl.style.display = 'none';
                        toastEl.classList.remove('hiding');
                    }
                    if (layer) {
                        layer.classList.remove('has-popup');
                        layer.setAttribute('aria-hidden', 'true');
                    }
                    spikeToastTimeout = null;
                }, 250);
            }, 6000);
        }
    });

    function dismissInsightPopup() {
        if (spikeToastTimeout) clearTimeout(spikeToastTimeout);
        spikeToastTimeout = null;
        var t = document.getElementById('metricSpikeToast');
        var p = document.getElementById('insightPopupLayer');
        if (t && p) {
            t.classList.add('hiding');
            setTimeout(function () {
                t.style.display = 'none';
                t.classList.remove('hiding');
                p.classList.remove('has-popup');
                p.setAttribute('aria-hidden', 'true');
            }, 250);
        }
    }
    if (layer && !layer._dismissBound) {
        layer._dismissBound = true;
        layer.addEventListener('click', function (e) {
            if (e.target.id !== 'insightPopupLayer') return;
            dismissInsightPopup();
        });
        document.addEventListener('keydown', function escHandler(e) {
            if (e.key === 'Escape' && document.getElementById('insightPopupLayer').classList.contains('has-popup')) {
                dismissInsightPopup();
            }
        });
    }
}

// ========================================================================
// System Initialization
// ========================================================================

/**
 * Initialize all systems (engagement detection, session manager).
 * Insights/chat manager is initialized when insights session starts.
 */
function initializeSystems() {
    try {
        var apiBase = getApiBase();
        // Engagement metrics are displayed in the Dashboard (open via Open Dashboard button)
        signifierPanel = null;
        azureMetricsPanel = null;
        compositeMetricsPanel = null;
        // Initialize engagement detector (panels null; metrics available via GET /engagement/state and Dashboard)
        engagementDetector = new EngagementDetector({
            signifierPanel: null,
            azureMetricsPanel: null,
            compositeMetricsPanel: null,
            apiBaseUrl: apiBase,
            pollInterval: 200,
            onAlert: handleMetricSpikeAlert,
            onPartnerStreamStateChange: updateStopPartnerSharingButton
        });
        
        // Initialize session manager
        sessionManager = new SessionManager({
            engagementDetector: engagementDetector,
            apiBaseUrl: apiBase
        });
        
        // Initialize video source selector (same API base so Azure Face API availability check works)
        videoSourceSelector = new VideoSourceSelector({
            onSelect: handleVideoSourceSelected,
            onCancel: handleVideoSourceCanceled,
            apiBaseUrl: apiBase
        });
        
        // Show "Change Video Source" button after initialization
        const changeSourceBtn = document.getElementById('selectVideoSource');
        if (changeSourceBtn) {
            changeSourceBtn.style.display = 'inline-block';
        }
        
        // Clear insight transcript button
        var clearTranscriptBtn = document.getElementById('insightTranscriptClear');
        if (clearTranscriptBtn) {
            clearTranscriptBtn.addEventListener('click', function () {
                clearInsightTranscript();
            });
        }
        var spotlightLink = document.getElementById('insightSpotlightLink');
        if (spotlightLink) {
            spotlightLink.addEventListener('click', function (e) {
                e.preventDefault();
                var section = document.getElementById('insightTranscriptSection');
                if (section) section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        }
        // Context & AI: poll last context and response every 7s
        if (contextAndResponsePollIntervalId) clearInterval(contextAndResponsePollIntervalId);
        refreshContextAndResponse();
        contextAndResponsePollIntervalId = setInterval(refreshContextAndResponse, 7000);
        
        // Periodic context push: when session is live, push context to Foundry every 40s
        if (contextPushIntervalId) clearInterval(contextPushIntervalId);
        contextPushIntervalId = setInterval(function () {
            if (sessionManager && sessionManager.isSessionActive && typeof fetch !== 'undefined') {
                fetch(getApiBase() + '/api/context-push', { method: 'POST' })
                    .then(function (r) { return r.ok ? r.json() : null; })
                    .then(function (data) {
                        if (data && typeof refreshContextAndResponse === 'function') refreshContextAndResponse();
                    })
                    .catch(function () {});
            }
        }, 40000);
        
        console.log('Core systems initialized (engagement + session manager + video source selector)');
    } catch (error) {
        console.error('Error initializing systems:', error);
    }
}

// ========================================================================
// Video Source Selection & Engagement Feed Display
// ========================================================================

/** API base URL: same origin when served with backend, fallback for static open. */
function getApiBase() {
    if (typeof window !== 'undefined' && window.location && window.location.origin) {
        var o = window.location.origin;
        if (o && (o.indexOf('http://') === 0 || o.indexOf('https://') === 0)) {
            return o;
        }
    }
    return 'http://localhost:5000';
}
var apiBaseUrl = getApiBase();

/** Poll interval for Context & AI sidebar (context-and-response). */
var contextAndResponsePollIntervalId = null;
/** Interval for periodic context push when session is live. */
var contextPushIntervalId = null;
/**
 * Fetch and display last context sent to Foundry and last AI response.
 * Called on an interval and after chat stream completion.
 */
function refreshContextAndResponse() {
    fetch(getApiBase() + '/api/engagement/context-and-response?t=' + Date.now(), { method: 'GET' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
            if (!data) return;
            var ctxEl = document.getElementById('contextPassedToFoundry');
            var respEl = document.getElementById('aiResponseText');
            var tsEl = document.getElementById('contextAndResponseTimestamp');
            if (ctxEl) ctxEl.textContent = (data.contextSent && data.contextSent.length) ? data.contextSent : '—';
            if (respEl) respEl.textContent = (data.response && data.response.length) ? data.response : '—';
            if (tsEl) tsEl.textContent = data.timestamp ? 'Updated ' + new Date(data.timestamp * 1000).toLocaleTimeString() : '';
        })
        .catch(function () {});
}
if (typeof window !== 'undefined') window.refreshContextAndResponse = refreshContextAndResponse;

/**
 * Show the engagement detection video feed in the main display area.
 * Uses multipart/x-mixed-replace streaming for efficient frame updates (30–60 FPS).
 * The backend streams JPEG frames continuously until the feed is hidden.
 */
function showEngagementFeed() {
    var el = document.getElementById('engagementFeed');
    if (el) {
        var base = (sessionManager && sessionManager.apiBaseUrl) ? sessionManager.apiBaseUrl : apiBaseUrl;
        // Set src to streaming endpoint - browser handles multipart/x-mixed-replace automatically
        el.src = base + '/engagement/video-feed';
        el.style.display = 'block';
    }
    var wrap = document.getElementById('engagementFeedWrap');
    if (wrap) wrap.style.display = 'block';
}

/** Set the engagement metrics hint message (shown when no metrics available). */
function setEngagementHint(message) {
    var el = document.getElementById('engagementMetricsHint');
    if (el) {
        el.textContent = message || '';
        el.style.display = message ? 'block' : 'none';
    }
}

/**
 * Hide the engagement feed and stop requesting frames.
 * Setting src to empty string stops the browser from requesting more frames.
 */
function hideEngagementFeed() {
    var el = document.getElementById('engagementFeed');
    if (el) {
        el.src = ''; // Stop streaming
        el.style.display = 'none';
    }
    var wrap = document.getElementById('engagementFeedWrap');
    if (wrap) wrap.style.display = 'none';
}

/**
 * Prompt user to select video source for engagement detection.
 * 
 * @param {boolean} force - Force show even if already shown (default: false)
 */
function promptVideoSourceSelection(force = false) {
    // Don't show if already shown (unless forced)
    if (videoSourcePromptShown && !force) {
        return;
    }
    
    if (!videoSourceSelector) {
        console.warn('Video source selector not initialized');
        // Fallback to default (stream) with auto detection if systems are ready
        if (sessionManager) {
            sessionManager.startSession('stream', null, null)
                .then(function () { showEngagementFeed(); })
                .catch(err => {
                    console.warn('Could not start parallel session:', err);
                });
        }
        return;
    }
    
    videoSourceSelector.show();
    videoSourcePromptShown = true;
}

/**
 * Handle video source selection.
 * 
 * @param {Object} selection - Selected video source options
 */
function stopPartnerAudioStreamIfNeeded() {
    if (_partnerAudioStreamFromMic && _partnerAudioStream) {
        _partnerAudioStream.getTracks().forEach(function (t) { t.stop(); });
    }
    _partnerAudioStreamFromMic = false;
    _partnerAudioStream = null;
}

function runAfterVideoSourceSelectedCallback() {
    if (_afterVideoSourceSelectedCallback) {
        var fn = _afterVideoSourceSelectedCallback;
        _afterVideoSourceSelectedCallback = null;
        fn();
    }
}

async function handleVideoSourceSelected(selection) {
    const { sourceType, sourcePath, file, usePartnerAudio, useMicrophoneAudio } = selection;
    
    try {
        stopPartnerAudioStreamIfNeeded();
        // If file is selected, upload it first
        let finalSourcePath = sourcePath;
        
        if (sourceType === 'file' && file) {
            // Show loading indicator
            const confirmBtn = document.querySelector('#confirmVideoSource');
            const originalText = confirmBtn.textContent;
            confirmBtn.textContent = 'Uploading...';
            confirmBtn.disabled = true;
            
            try {
                // Upload file to server
                const formData = new FormData();
                formData.append('video', file);
                
                const uploadResponse = await fetch(getApiBase() + '/engagement/upload-video', {
                    method: 'POST',
                    body: formData
                });
                
                if (!uploadResponse.ok) {
                    const errorData = await uploadResponse.json().catch(() => ({ error: 'Upload failed' }));
                    throw new Error(errorData.error || 'Failed to upload video file');
                }
                
                const uploadResult = await uploadResponse.json();
                finalSourcePath = uploadResult.filePath;
            } finally {
                confirmBtn.textContent = originalText;
                confirmBtn.disabled = false;
            }
        }
        
        // App chooses optimal detection method (auto); no selection passed
        // Partner: getDisplayMedia must run under the same user gesture as the click.
        // Call it first (before any await), then start session and frame stream on success.
        if (sourceType === 'partner' && sessionManager && engagementDetector) {
            let stream;
            try {
                // Browser picker (tab/window/screen); request audio for speech cues when sharing a tab
                var displayOpts = { video: true, audio: true };
                if (/Chrome|Chromium|Edg\//.test(navigator.userAgent) && !/Firefox/.test(navigator.userAgent)) {
                    displayOpts.selfBrowserSurface = 'exclude';
                }
                stream = await navigator.mediaDevices.getDisplayMedia(displayOpts);
            } catch (shareErr) {
                if (shareErr.name === 'NotAllowedError') {
                    console.log('Share canceled by user');
                    return;
                }
                console.error('Partner capture failed:', shareErr);
                if (shareErr.name === 'NotSupportedError' || (shareErr.message && shareErr.message.indexOf('secure') !== -1)) {
                    alert('Screen share requires a secure context (localhost or HTTPS). Open the app in a secure page and try again.');
                } else {
                    alert('Screen share failed: ' + (shareErr.message || shareErr.name || 'Unknown error'));
                }
                return;
            }
            await sessionManager.startSession('partner', null, null);
            showEngagementFeed();
            setEngagementHint('');
            engagementDetector.startPartnerFrameStream(stream);
            if (stream.getAudioTracks().length > 0) {
                _partnerAudioStream = stream;
                console.log('Partner audio stored; start microphone when ready for aural cues.');
            } else {
                _partnerAudioStream = null;
            }
            console.log('Engagement detection started with Meeting Partner Video (video → engagement, audio → aural cues when mic started). Start Insights Session for popups.');
            videoSourcePromptShown = true;
            runAfterVideoSourceSelectedCallback();
            return;
        }
        
        if (sessionManager) {
            await sessionManager.startSession(sourceType, finalSourcePath, null);
            showEngagementFeed();
            setEngagementHint('');
            console.log('Engagement detection started with source: ' + sourceType);
        }
        if ((sourceType === 'webcam' || sourceType === 'file') && useMicrophoneAudio) {
            try {
                var audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                if (audioStream.getAudioTracks().length > 0) {
                    _partnerAudioStream = audioStream;
                    _partnerAudioStreamFromMic = true;
                    console.log('Microphone audio stored for speech and acoustic analysis; start microphone when ready for aural cues.');
                } else {
                    _partnerAudioStream = null;
                    _partnerAudioStreamFromMic = false;
                }
            } catch (audioErr) {
                if (audioErr.name === 'NotAllowedError') {
                    console.log('User denied microphone access; continuing without partner audio.');
                } else {
                    console.warn('Could not start microphone for partner audio:', audioErr);
                }
            }
        } else if ((sourceType === 'webcam' || sourceType === 'file') && usePartnerAudio) {
            try {
                var audioStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
                if (audioStream.getAudioTracks().length > 0) {
                    _partnerAudioStream = audioStream;
                    console.log('Partner audio stored for webcam/file source; start microphone when ready for aural cues.');
                } else {
                    _partnerAudioStream = null;
                    console.log('Display media has no audio track; continuing without partner audio.');
                }
            } catch (audioErr) {
                if (audioErr.name === 'NotAllowedError') {
                    console.log('User denied meeting tab audio; continuing with video only.');
                } else {
                    console.warn('Could not start meeting partner audio:', audioErr);
                }
            }
        }
        videoSourcePromptShown = true;
        runAfterVideoSourceSelectedCallback();
    } catch (error) {
        console.error('Error starting engagement detection with selected source:', error);
        alert(`Failed to start engagement detection: ${error.message}\n\nPlease try again or select a different source.`);
    }
}

/**
 * Show or hide the "Stop sharing" button based on partner stream state.
 * When partner stream ends, also stop feed-audio transcript (aural cues).
 * @param {boolean} isActive - Whether Meeting Partner Video capture is active
 */
function updateStopPartnerSharingButton(isActive) {
    var btn = document.getElementById('stopPartnerSharing');
    if (btn) btn.style.display = isActive ? 'inline-block' : 'none';
    if (!isActive) {
        stopFeedAudioTranscript();
        stopPartnerAudioStreamIfNeeded();
    }
}

/**
 * Stop sharing screen/tab for Meeting Partner Video (stops getDisplayMedia and frame sending).
 */
function stopPartnerSharing() {
    if (engagementDetector && engagementDetector.isPartnerStreamActive()) {
        engagementDetector.stopPartnerFrameStream();
    }
    stopFeedAudioTranscript();
    stopPartnerAudioStreamIfNeeded();
}

/**
 * Start speech recognition on the meeting partner (feed) audio stream.
 * Recognized text is sent to POST /engagement/transcript for B2B insight generation (aural cues).
 * Called when user presses "Start Microphone" and partner audio stream is stored (_partnerAudioStream).
 * @param {MediaStream} stream - getDisplayMedia stream with audio track (meeting partner)
 * @param {function} [onStarted] - optional callback when recognition has started (e.g. to update mic button)
 */
async function startFeedAudioTranscript(stream, onStarted) {
    if (!stream || stream.getAudioTracks().length === 0) return;
    stopFeedAudioTranscript();
    if (!appConfig) await loadConfig();
    if (!appConfig) {
        console.warn('Feed audio transcript: config not loaded, skipping.');
        return;
    }
    try {
        const tokenRes = await fetch(getApiBase() + '/speech/token');
        if (!tokenRes.ok) {
            console.warn('Feed audio transcript: failed to get speech token.');
            return;
        }
        const { token } = await tokenRes.json();
        const cogSvcRegion = appConfig.speech.region;
        const privateEndpointEnabled = appConfig.speech.privateEndpointEnabled || false;
        let speechRecognitionConfig;
        if (privateEndpointEnabled) {
            const privateEndpoint = appConfig.speech.privateEndpoint || '';
            speechRecognitionConfig = SpeechSDK.SpeechConfig.fromEndpoint(
                new URL('wss://' + privateEndpoint + '/stt/speech/universal/v2'),
                token
            );
        } else {
            speechRecognitionConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, cogSvcRegion);
        }
        speechRecognitionConfig.setProperty(SpeechSDK.PropertyId.SpeechServiceConnection_LanguageIdMode, 'Continuous');
        var sttLocales = (appConfig.sttTts && appConfig.sttTts.sttLocales) ? appConfig.sttTts.sttLocales.split(',') : ['en-US'];
        var autoDetectSourceLanguageConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages(sttLocales);
        var audioConfig = SpeechSDK.AudioConfig.fromStreamInput(stream);
        feedSpeechRecognizer = SpeechSDK.SpeechRecognizer.FromConfig(
            speechRecognitionConfig,
            autoDetectSourceLanguageConfig,
            audioConfig
        );
        feedSpeechRecognizer.recognized = function (s, e) {
            if (e.result.reason !== SpeechSDK.ResultReason.RecognizedSpeech) return;
            var text = (e.result.text || '').trim();
            if (!text) return;
            fetch(getApiBase() + '/engagement/transcript', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            }).catch(function (err) { console.warn('Feed transcript POST failed:', err); });
        };
        feedSpeechRecognizer.startContinuousRecognitionAsync(
            function () {
                console.log('Feed audio (aural cues) transcript started.');
                var acousticEnabled = appConfig.acoustic && appConfig.acoustic.acousticAnalysisEnabled !== false;
                if (window.AcousticAnalyzer) window.AcousticAnalyzer.start(stream, getApiBase(), acousticEnabled);
                if (typeof onStarted === 'function') onStarted();
            },
            function (err) { console.warn('Feed audio transcript start failed:', err); feedSpeechRecognizer = null; }
        );
    } catch (err) {
        console.warn('Start feed audio transcript failed:', err);
        feedSpeechRecognizer = null;
    }
}

/**
 * Stop speech recognition on feed audio and release recognizer.
 */
function stopFeedAudioTranscript() {
    if (window.AcousticAnalyzer) window.AcousticAnalyzer.stop();
    if (!feedSpeechRecognizer) return;
    try {
        feedSpeechRecognizer.stopContinuousRecognitionAsync(function () {}, function () {});
        feedSpeechRecognizer.close();
    } catch (e) {}
    feedSpeechRecognizer = null;
}

/**
 * Handle video source selection cancellation.
 */
function handleVideoSourceCanceled() {
    console.log('Video source selection canceled');
    _afterVideoSourceSelectedCallback = null;
    videoSourcePromptShown = true;
    // Fallback: start with webcam so engagement runs
    if (sessionManager && engagementDetector && !engagementDetector.isActive) {
        sessionManager.startSession('webcam', null, null)
            .then(function (ok) {
                if (ok) {
                    showEngagementFeed();
                    setEngagementHint('');
                } else {
                    setEngagementHint('Could not start. Click Change Video Source to retry.');
                }
            })
            .catch(function (err) {
                console.warn('Could not start engagement with webcam:', err);
                setEngagementHint('Could not connect. Click Change Video Source to retry.');
            });
    }
}

// Configuration from backend
async function loadConfig() {
    try {
        const response = await fetch(getApiBase() + '/config/all');
        if (!response.ok) throw new Error('Failed to fetch config: ' + response.status);
        appConfig = await response.json();
        console.log('Configuration loaded from backend', appConfig);
    } catch (error) {
        console.error('Failed to load configuration:', error);
        alert('Failed to load configuration from backend. Please ensure the server is running.');
    }
}

/**
 * Start Insights Session: STT for transcript and speech cue analysis; insight popups are text only.
 * If no video source is set yet, prompts user to choose one first.
 */
async function initializeInsightsSession() {
    // If engagement has no video source yet, show video source selector first; continue session start after they select.
    if (engagementDetector && !engagementDetector.isActive && videoSourceSelector) {
        _afterVideoSourceSelectedCallback = function () {
            doInitializeInsightsSession();
        };
        promptVideoSourceSelection(true);
        return;
    }
    doInitializeInsightsSession();
}

async function doInitializeInsightsSession() {
    if (!appConfig) {
        await loadConfig();
    }
    if (!appConfig) {
        alert('Failed to load configuration. Please refresh the page.');
        return;
    }

    const cogSvcRegion = appConfig.speech.region;
    const privateEndpointEnabled = appConfig.speech.privateEndpointEnabled || false

    const tokenRes = await fetch(getApiBase() + '/speech/token')
    if (!tokenRes.ok) {
        let msg = 'Failed to get speech token from backend.'
        try {
            const errBody = await tokenRes.json()
            if (errBody && (errBody.details || errBody.error)) {
                msg = errBody.details || errBody.error
            }
        } catch (_) {}
        alert(msg)
        return
    }
    const { token } = await tokenRes.json()

    // Speech recognizer for STT (user mic + feed transcript sent to backend for speech cue analysis)
    let speechRecognitionConfig
    if (privateEndpointEnabled) {
        const privateEndpoint = appConfig.speech.privateEndpoint || ''
        speechRecognitionConfig = SpeechSDK.SpeechConfig.fromEndpoint(
            new URL(`wss://${privateEndpoint}/stt/speech/universal/v2`),
            token
        )
    } else {
        speechRecognitionConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, cogSvcRegion)
    }
    speechRecognitionConfig.setProperty(SpeechSDK.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
    var sttLocales = (appConfig.sttTts && appConfig.sttTts.sttLocales) ? appConfig.sttTts.sttLocales.split(',') : ['en-US']
    var autoDetectSourceLanguageConfig = SpeechSDK.AutoDetectSourceLanguageConfig.fromLanguages(sttLocales)
    speechRecognizer = SpeechSDK.SpeechRecognizer.FromConfig(
        speechRecognitionConfig,
        autoDetectSourceLanguageConfig,
        SpeechSDK.AudioConfig.fromDefaultMicrophoneInput()
    )

    initMessages();

    sessionActive = true;
    updateSessionStatusIndicator(true);
    var initBtn = document.getElementById('initInsightsSession');
    var micBtn = document.getElementById('microphone');
    var stopBtn = document.getElementById('stopSession');
    if (initBtn) initBtn.disabled = true;
    if (micBtn) micBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = false;
    var msgInput = document.getElementById('userMessageInput');
    var sendBtn = document.getElementById('sendMessageBtn');
    if (msgInput) { msgInput.disabled = false; msgInput.focus(); }
    if (sendBtn) sendBtn.disabled = false;
}

// ========================================================================
// Session Management
// ========================================================================

/**
 * Stop insights session and clean up all resources.
 */
async function stopSession() {
    if (spikeToastTimeout) {
        clearTimeout(spikeToastTimeout);
        spikeToastTimeout = null;
    }
    if (sessionManager) {
        await sessionManager.stopSession();
    }
    stopFeedAudioTranscript();
    stopPartnerAudioStreamIfNeeded();

    try {
        if (speechRecognizer) {
            speechRecognizer.stopContinuousRecognitionAsync(function () {}, function () {});
            speechRecognizer.close();
        }
    } catch (e) { console.debug('Speech recognizer close:', e); }
    speechRecognizer = null;

    sessionActive = false;
    updateSessionStatusIndicator(false);
    clearInsightTranscript();
    hideEngagementFeed();

    var initBtn = document.getElementById('initInsightsSession');
    var micBtn = document.getElementById('microphone');
    var stopBtn = document.getElementById('stopSession');
    if (initBtn) initBtn.disabled = false;
    if (micBtn) micBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    var msgInput = document.getElementById('userMessageInput');
    var sendBtn = document.getElementById('sendMessageBtn');
    if (msgInput) { msgInput.disabled = true; msgInput.value = ''; }
    if (sendBtn) sendBtn.disabled = true;
}

// HTML encoding
function htmlEncode(text) {
    const entityMap = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
        '/': '&#x2F;'
    }
    return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match]);
}

// ========================================================================
// Message Management
// ========================================================================

/**
 * Initialize messages array for Foundry (system prompt) and data sources for cognitive search.
 */
function initMessages() {
    insightMessages = [];
    if (appConfig && appConfig.systemPrompt) {
        insightMessages.push({ role: 'system', content: appConfig.systemPrompt });
    }

    // Initialize data sources for cognitive search
    if (!appConfig) {
        console.error('Configuration not loaded');
        return;
    }
    
    if (appConfig.cognitiveSearch && appConfig.cognitiveSearch.enabled) {
        dataSources = [{
            type: 'AzureCognitiveSearch',
            parameters: {
                endpoint: appConfig.cognitiveSearch.endpoint,
                key: appConfig.cognitiveSearch.apiKey,
                indexName: appConfig.cognitiveSearch.indexName,
                semanticConfiguration: '',
                queryType: 'simple',
                fieldsMapping: {
                    contentFieldsSeparator: '\n',
                    contentFields: ['content'],
                    filepathField: null,
                    titleField: 'title',
                    urlField: null
                },
                inScope: true,
                roleInformation: appConfig.systemPrompt
            }
        }];
    } else {
        dataSources = [];
    }
}

/**
 * Clear insight message history and re-init (system message + data sources).
 */
function clearChatHistory() {
    insightMessages = [];
    initMessages();
}

// ========================================================================
// Foundry streaming (STT or typed message -> insight transcript)
// ========================================================================

/**
 * Send user message to Foundry and stream the response into the Insight Transcript.
 * Used when the user speaks (STT) or types a message; spoken context is passed for intelligent insights.
 *
 * @param {string} userQuery - User's message (e.g. from STT or input)
 * @param {string} userQueryHTML - HTML formatted user query (unused; for compatibility)
 * @param {string} imgUrlPath - Optional image URL path (unused in current flow)
 */
async function handleUserQuery(userQuery, userQueryHTML, imgUrlPath) {
    if (!sessionManager || !appConfig) {
        console.error('Session or config not ready');
        return;
    }
    var query = (userQuery || '').trim();
    if (!query) return;

    sessionManager.interruptStreaming();

    var contentMessage = query;
    if (imgUrlPath && imgUrlPath.trim()) {
        contentMessage = [
            { type: 'text', text: query },
            { type: 'image_url', image_url: { url: imgUrlPath } }
        ];
    }
    insightMessages.push({ role: 'user', content: contentMessage });

    var entry = createInsightTranscriptEntryForStreaming();
    var contentSpan = entry.contentSpan;
    var time = entry.time;

    if (contentSpan) {
        contentSpan.textContent = 'User: ' + query + '\nCoach: ';
    }

    try {
        await sessionManager.streamChatResponse(insightMessages, {
            enableOyd: appConfig.cognitiveSearch && appConfig.cognitiveSearch.enabled,
            systemPrompt: appConfig.systemPrompt,
            includeEngagement: true,
            cognitiveSearchEnabled: appConfig.cognitiveSearch && appConfig.cognitiveSearch.enabled,
            onChunk: function(token, fullReply) {
                if (contentSpan) {
                    var prefix = 'User: ' + query + '\nCoach: ';
                    contentSpan.textContent = prefix + (fullReply || '');
                }
                var box = document.getElementById('insightTranscript');
                if (box) box.scrollTop = box.scrollHeight;
            },
            onComplete: function(result) {
                if (result && result.assistantReply) {
                    insightMessages.push({ role: 'assistant', content: result.assistantReply });
                    updateInsightSpotlight(result.assistantReply, time);
                }
            },
            onError: function(error) {
                if (contentSpan) {
                    contentSpan.textContent = (contentSpan.textContent || '') + ' [Error: ' + (error && error.message ? error.message : 'Unknown') + ']';
                }
            }
        });
    } catch (err) {
        if (contentSpan) {
            contentSpan.textContent = (contentSpan.textContent || '') + ' [Error: ' + (err.message || 'Unknown') + ']';
        }
    }
}


function microphone() {
    if (!sessionActive) {
        alert('Please start insights session first, then start microphone.');
        return;
    }

    var micBtn = document.getElementById('microphone');
    if (micBtn && micBtn.innerHTML === 'Stop Microphone') {
        micBtn.disabled = true;
        if (feedSpeechRecognizer) {
            stopFeedAudioTranscript();
            micBtn.innerHTML = 'Start Microphone';
            micBtn.disabled = false;
        } else if (speechRecognizer) {
            speechRecognizer.stopContinuousRecognitionAsync(function () {
                micBtn.innerHTML = 'Start Microphone';
                micBtn.disabled = false;
            }, function (err) {
                console.log('Failed to stop continuous recognition:', err);
                micBtn.disabled = false;
            });
        }
        return;
    }

    micBtn = document.getElementById('microphone');
    if (micBtn) micBtn.disabled = true;

    if (_partnerAudioStream) {
        startFeedAudioTranscript(_partnerAudioStream, function () {
            var el = document.getElementById('microphone');
            if (el) { el.innerHTML = 'Stop Microphone'; el.disabled = false; }
        });
        return;
    }

    if (!speechRecognizer) {
        if (micBtn) micBtn.disabled = false;
        alert('Speech recognizer not ready. Start insights session first.');
        return;
    }

    speechRecognizer.recognizing = function(s, e) {
        if (e.result.reason !== SpeechSDK.ResultReason.RecognizingSpeech) return;
        if (sessionManager && sessionManager.isStreaming) {
            sessionManager.interruptStreaming();
        }
    };

    speechRecognizer.recognized = async function(s, e) {
        if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
            var userQuery = (e.result.text || '').trim();
            if (userQuery === '') return;

            if (sessionManager && sessionManager.isStreaming) {
                sessionManager.interruptStreaming();
                await new Promise(function(resolve) { setTimeout(resolve, 100); });
            }

            if (!appConfig || !appConfig.sttTts || !appConfig.sttTts.continuousConversation) {
                var mEl = document.getElementById('microphone');
                if (mEl) mEl.disabled = true;
                speechRecognizer.stopContinuousRecognitionAsync(function() {
                    var el = document.getElementById('microphone');
                    if (el) { el.innerHTML = 'Start Microphone'; el.disabled = false; }
                }, function(err) {
                    console.log('Failed to stop continuous recognition:', err);
                    var el = document.getElementById('microphone');
                    if (el) el.disabled = false;
                });
            }

            handleUserQuery(userQuery, '', '');
        }
    };

    speechRecognizer.startContinuousRecognitionAsync(function() {
        var el = document.getElementById('microphone');
        if (el) { el.innerHTML = 'Stop Microphone'; el.disabled = false; }
    }, function(err) {
        console.log('Failed to start continuous recognition:', err);
        var el = document.getElementById('microphone');
        if (el) el.disabled = false;
    });
}

// Message sending functions
function handleMessageKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        sendMessage()
    }
}

async function sendMessage() {
    var input = document.getElementById('userMessageInput');
    var sendBtn = document.getElementById('sendMessageBtn');
    if (!input) return;
    var message = (input.value || '').trim();
    if (!message) return;
    if (!appConfig) {
        alert('Configuration not loaded. Please refresh the page.');
        return;
    }
    if (!sessionActive) {
        alert('Please start insights session first.');
        return;
    }
    if (sessionManager && sessionManager.isStreaming) {
        sessionManager.interruptStreaming();
        await new Promise(function(r) { setTimeout(r, 100); });
    }
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    input.value = '';
    handleUserQuery(message, message, '');
    setTimeout(function() {
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.focus();
    }, 300);
}

// ========================================================================
// Initialization
// ========================================================================

// Initialize systems and auto-start engagement with webcam so metrics work immediately.
function onSystemsReady() {
    initializeSystems();
    // Show video source modal on first open so user selects webcam, partner, or file
    if (videoSourceSelector && !videoSourcePromptShown) {
        setTimeout(function () {
            promptVideoSourceSelection();
            setEngagementHint('Select a video source to see engagement metrics.');
        }, 300);
    }
    // Panel mode: auto-start webcam after 2.5s if user has not selected a source (metrics work without clicking)
    if (window.self !== window.top && sessionManager && engagementDetector) {
        setTimeout(function () {
            if (!engagementDetector.isActive) {
                videoSourcePromptShown = true;
                if (videoSourceSelector && typeof videoSourceSelector.hide === 'function') {
                    videoSourceSelector.hide();
                }
                sessionManager.startSession('webcam', null, null).then(function (ok) {
                    if (ok) {
                        showEngagementFeed();
                        setEngagementHint('Using default webcam; change via Change Video Source.');
                    }
                }).catch(function () {
                    setEngagementHint('Select a video source to see engagement metrics.');
                });
            }
        }, 2500);
    }
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', onSystemsReady);
} else {
    onSystemsReady();
}

window.onload = function() {
    loadConfig().then(function() {
        initMessages();
    });
    if (!sessionManager || !engagementDetector) {
        onSystemsReady();
    }
};