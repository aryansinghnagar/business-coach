// ========================================================================
        // Business Meeting Copilot - Main Application Script
        // ========================================================================
        // Flow: 1) Insights Session: STT (user mic + feed audio) -> transcript to backend for speech cue analysis
        //       2) Engagement detector: video from feed -> visual cues (G1-G4 spikes)
        //       3) On spike: backend fuses visual + aural context -> Azure AI Foundry -> insight text
        //       4) Frontend: popup (text only) + Insight Transcript box; no avatar/TTS
        // ========================================================================
        // Global Variables & Configuration
        // ========================================================================
        var speechRecognizer
        var feedSpeechRecognizer = null  // STT on meeting partner (feed) audio for aural cues; transcript sent to backend
        var _partnerAudioStream = null   // Partner audio stream stored when source has audio; STT started only on "Start Microphone"
        var _partnerAudioStreamFromMic = false  // True when stream is from getUserMedia (mic); stop tracks on cleanup
        var avatarSynthesizer
        var peerConnection
        var peerConnectionDataChannel
        var appConfig = null
        
        // Session Management (modular)
        var sessionManager = null
        var avatarChatManager = null
        var engagementDetector = null
        var signifierPanel = null
        var azureMetricsPanel = null
        var compositeMetricsPanel = null
        var videoSourceSelector = null
        
        // UI State
        var avatarAudioInitialized = false
        var sessionActive = false
        var dataSources = []
        
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
        var videoSourcePromptShown = false  // Track if video source prompt has been shown

        // ========================================================================
        // Metric spike alerts (popup when a metric group spikes)
        // ========================================================================
        var spikeToastTimeout = null;
        
        /**
         * Append an insight response to the sidebar transcript box (for verification).
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
            // Update latest insight spotlight (above Controls)
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
                timeEl.textContent = time;
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
         * Handle metric spike alert: show popup (text only) and append to transcript.
         * Popups are shown only when Insights Session is live; metrics still update on dashboard when session is idle.
         */
        function handleMetricSpikeAlert(alert) {
            if (!alert || !alert.message) return;
            if (!sessionActive) return;
            var msg = String(alert.message).trim();
            if (!msg) return;
            // 1. Append to sidebar transcript
            appendToInsightTranscript(msg);
            // 2. Show insight popup in separate overlay layer (text only; no avatar speech)
            var el = document.getElementById('metricSpikeToast');
            var layer = document.getElementById('insightPopupLayer');
            if (el && layer) {
                if (spikeToastTimeout) clearTimeout(spikeToastTimeout);
                el.textContent = msg;
                el.style.display = 'block';
                el.classList.remove('hiding');
                layer.classList.add('has-popup');
                layer.setAttribute('aria-hidden', 'false');
                spikeToastTimeout = setTimeout(function () {
                    el.classList.add('hiding');
                    spikeToastTimeout = setTimeout(function () {
                        el.style.display = 'none';
                        el.classList.remove('hiding');
                        layer.classList.remove('has-popup');
                        layer.setAttribute('aria-hidden', 'true');
                        spikeToastTimeout = null;
                    }, 250);
                }, 6000);
            }
            function dismissInsightPopup() {
                if (spikeToastTimeout) clearTimeout(spikeToastTimeout);
                spikeToastTimeout = null;
                var toastEl = document.getElementById('metricSpikeToast');
                var popupLayer = document.getElementById('insightPopupLayer');
                if (toastEl && popupLayer) {
                    toastEl.classList.add('hiding');
                    setTimeout(function () {
                        toastEl.style.display = 'none';
                        toastEl.classList.remove('hiding');
                        popupLayer.classList.remove('has-popup');
                        popupLayer.setAttribute('aria-hidden', 'true');
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
                    pollInterval: 250,
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
         * Start Insights Session: STT only (no avatar/TTS). Transcript is sent to backend
         * for speech cue analysis and Azure Foundry. Insight popups are text only.
         */
        async function initializeInsightsSession() {
            if (!appConfig) {
                await loadConfig()
            }
            if (!appConfig) {
                alert('Failed to load configuration. Please refresh the page.')
                return
            }

            const cogSvcRegion = appConfig.speech.region
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

            // Chat manager (text-only; no avatar synthesizer)
            if (sessionManager && !avatarChatManager) {
                avatarChatManager = new AvatarChatManager({
                    sessionManager: sessionManager,
                    avatarSynthesizer: null,
                    appConfig: appConfig
                })
                console.log('Chat manager initialized (text-only)')
            }
            if (sessionManager) {
                sessionManager.isAvatarInitialized = true
            }
            if (!avatarChatManager && appConfig) {
                initMessages()
            }

            sessionActive = true
            avatarAudioInitialized = true
            updateSessionStatusIndicator(true)
            var initBtn = document.getElementById('initAvatar')
            var micBtn = document.getElementById('microphone')
            var stopBtn = document.getElementById('stopSession')
            if (initBtn) initBtn.disabled = true
            if (micBtn) micBtn.disabled = false
            if (stopBtn) stopBtn.disabled = false
            var msgInput = document.getElementById('userMessageInput')
            var sendBtn = document.getElementById('sendMessageBtn')
            if (msgInput) { msgInput.disabled = false; msgInput.focus() }
            if (sendBtn) sendBtn.disabled = false
            if (!videoSourcePromptShown && videoSourceSelector) {
                promptVideoSourceSelection()
            }
        }
        
        // ========================================================================
        // Session Management
        // ========================================================================
        
        /**
         * Stop insights session and clean up all resources.
         */
        async function stopSession() {
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

            sessionActive = false;
            avatarAudioInitialized = false;
            updateSessionStatusIndicator(false);
            clearInsightTranscript();
            hideEngagementFeed();
            let avatarPipEl = document.getElementById('avatarPip');
            if (avatarPipEl) avatarPipEl.innerHTML = '';
            
            // Update UI
            var initBtn = document.getElementById('initAvatar');
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
            return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match])
        }

        // ========================================================================
        // Message Management
        // ========================================================================
        
        /**
         * Initialize messages array.
         */
        function initMessages() {
            if (avatarChatManager) {
                avatarChatManager.initMessages();
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
         * Clear chat history.
         */
        function clearChatHistory() {
            if (avatarChatManager) {
                avatarChatManager.clearChatHistory();
            }
            initMessages();
        }

        // ========================================================================
        // Speech Functions (Delegated to AvatarChatManager)
        // ========================================================================
        
        // ========================================================================
        // Speech Functions (Delegated to AvatarChatManager)
        // ========================================================================
        
        // ========================================================================
        // Speech Functions (Delegated to AvatarChatManager)
        // ========================================================================
        
        /**
         * Speak text - delegates to AvatarChatManager.
         * Kept for backward compatibility with existing code.
         */
        function speak(text, endingSilenceMs = 0) {
            if (avatarChatManager) {
                avatarChatManager.speak(text, endingSilenceMs);
            }
        }
        
        // Expose isSpeaking for UI updates (delegates to AvatarChatManager)
        // This allows existing code that checks isSpeaking to continue working
        Object.defineProperty(window, 'isSpeaking', {
            get: function() {
                return avatarChatManager ? avatarChatManager.isSpeaking : false;
            },
            configurable: true
        });

        // ========================================================================
        // Chat Handling (Streamlined)
        // ========================================================================
        
        /**
         * Handle user query - streamlined version using AvatarChatManager.
         * 
         * @param {string} userQuery - User's message
         * @param {string} userQueryHTML - HTML formatted user query
         * @param {string} imgUrlPath - Optional image URL path
         */
        async function handleUserQuery(userQuery, userQueryHTML, imgUrlPath) {
            if (!avatarChatManager) {
                console.error('Avatar chat manager not initialized');
                return;
            }
            
            await avatarChatManager.handleUserQuery(userQuery, userQueryHTML, imgUrlPath);
        }


        function microphone() {
            if (!avatarAudioInitialized || !sessionActive) {
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

            // Pause avatar as soon as user starts speaking (interim results)
            speechRecognizer.recognizing = (s, e) => {
                if (e.result.reason !== SpeechSDK.ResultReason.RecognizingSpeech) return;
                const partial = (e.result.text || '').trim();
                if (partial.length < 2) return;
                if (avatarChatManager && (avatarChatManager.isSpeaking || (avatarChatManager.sessionManager && avatarChatManager.sessionManager.isStreaming))) {
                    avatarChatManager.interruptCurrentResponse();
                }
            };

            speechRecognizer.recognized = async (s, e) => {
                if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
                    let userQuery = e.result.text.trim()
                    if (userQuery === '') {
                        return
                    }

                    // Interrupt avatar chat if still speaking/streaming (in case recognizing didn't fire)
                    if (avatarChatManager) {
                        avatarChatManager.interruptCurrentResponse();
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }

                    if (!appConfig || !appConfig.sttTts || !appConfig.sttTts.continuousConversation) {
                        document.getElementById('microphone').disabled = true
                        speechRecognizer.stopContinuousRecognitionAsync(() => {
                            document.getElementById('microphone').innerHTML = 'Start Microphone'
                            document.getElementById('microphone').disabled = false
                        }, (err) => {
                            console.log("Failed to stop continuous recognition:", err)
                            document.getElementById('microphone').disabled = false
                        })
                    }

                    handleUserQuery(userQuery, "", "")
                }
            }

            speechRecognizer.startContinuousRecognitionAsync(() => {
                document.getElementById('microphone').innerHTML = 'Stop Microphone'
                document.getElementById('microphone').disabled = false
            }, (err) => {
                console.log("Failed to start continuous recognition:", err)
                document.getElementById('microphone').disabled = false
            })
        }

        // Message sending functions
        function handleMessageKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                sendMessage()
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('userMessageInput');
            const sendBtn = document.getElementById('sendMessageBtn');
            if (!input) return;
            const message = (input.value || '').trim();
            
            if (!message) {
                return
            }
            
            if (!appConfig) {
                alert('Configuration not loaded. Please refresh the page.')
                return
            }
            
            if (!avatarAudioInitialized || !sessionActive) {
                alert('Please start insights session first.')
                return
            }
            
            // Interrupt avatar chat if speaking
            if (avatarChatManager) {
                avatarChatManager.interruptCurrentResponse();
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            // Disable input and button while processing
            input.disabled = true
            sendBtn.disabled = true
            
            // Clear input immediately for better UX
            input.value = ''
            
            // Send the message
            handleUserQuery(message, message, "")
            
            // Re-enable input and button after a brief moment
            setTimeout(() => {
                input.disabled = false
                sendBtn.disabled = false
                input.focus()
            }, 300)
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

        window.onload = () => {
            loadConfig().then(() => {
                if (avatarChatManager) {
                    avatarChatManager.initMessages();
                } else {
                    initMessages();
                }
            });
            if (!sessionManager || !engagementDetector) {
                onSystemsReady();
            }
        }