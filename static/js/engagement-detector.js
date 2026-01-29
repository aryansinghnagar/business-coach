/**
 * Engagement Detection Manager Module
 * 
 * This module manages engagement detection lifecycle, API communication,
 * and coordinates with the engagement bar display.
 * 
 * Features:
 * - Start/stop engagement detection
 * - Poll engagement state from backend
 * - Coordinate with engagement bar display
 * - Error handling and recovery
 */

class EngagementDetector {
    /**
     * Initialize the engagement detector.
     * 
     * @param {Object} options - Configuration options
     * @param {EngagementBarDisplay} options.barDisplay - Engagement bar display instance
     * @param {string} options.apiBaseUrl - Base URL for API (default: 'http://localhost:5000')
     * @param {number} options.pollInterval - Polling interval in milliseconds (default: 200, low latency)
     */
    constructor(options = {}) {
        this.barDisplay = options.barDisplay || null;
        this.signifierPanel = options.signifierPanel || null;
        this.azureMetricsPanel = options.azureMetricsPanel || null;
        this.apiBaseUrl = options.apiBaseUrl || 'http://localhost:5000';
        this.pollInterval = options.pollInterval || 200;
        this.onAlert = options.onAlert || null;
        this.onPartnerStreamStateChange = options.onPartnerStreamStateChange || null;
        
        // State management
        this.isActive = false;
        this.pollIntervalId = null;
        this.currentSourceType = null;
        this.currentSourcePath = null;
        this.currentDetectionMethod = null;
        
        // Error handling
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 5;
        
        // Request throttling
        this._pendingRequest = false;
        
        // Partner (Meeting Partner Video) capture: stream + frame-send loop
        this._partnerStream = null;
        this._partnerFrameIntervalId = null;
        this._partnerVideo = null;
        this._partnerCanvas = null;
    }
    
    /**
     * Start engagement detection.
     * 
     * @param {string} sourceType - Video source type ('webcam', 'file', 'stream', 'partner')
     * @param {string|null} sourcePath - Path to video file or stream URL (required for 'file'/'stream')
     * @param {string} detectionMethod - Detection method ('mediapipe' or 'azure_face_api')
     * @returns {Promise<boolean>} True if started successfully
     */
    async start(sourceType = 'webcam', sourcePath = null, detectionMethod = null) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/engagement/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sourceType: sourceType,
                    sourcePath: sourcePath,
                    detectionMethod: detectionMethod
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                console.error('Failed to start engagement detection:', error);
                return false;
            }
            
            const result = await response.json();
            console.log('Engagement detection started:', result.message);
            
            // Store current configuration
            this.currentSourceType = sourceType;
            this.currentSourcePath = sourcePath;
            this.currentDetectionMethod = result.detectionMethod || detectionMethod;
            this._setMetricsPanelVisibility(this.currentDetectionMethod || 'mediapipe');

            // Start polling
            this.isActive = true;
            this.consecutiveErrors = 0;
            this.startPolling();
            
            return true;
        } catch (error) {
            console.error('Error starting engagement detection:', error);
            return false;
        }
    }
    
    /**
     * Stop engagement detection.
     * 
     * @returns {Promise<boolean>} True if stopped successfully
     */
    async stop() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/engagement/stop`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                console.error('Failed to stop engagement detection');
                return false;
            }
            
            // Stop polling
            this.isActive = false;
            this.stopPolling();
            this.stopPartnerFrameStream();
            
            // Reset bar display and metrics panels
            if (this.barDisplay) {
                this.barDisplay.reset();
            }
            if (this.signifierPanel) {
                this.signifierPanel.reset();
            }
            if (this.azureMetricsPanel) {
                this.azureMetricsPanel.reset();
            }
            this._setMetricsPanelVisibility('mediapipe');
            
            // Clear state
            this.currentSourceType = null;
            this.currentSourcePath = null;
            this.currentDetectionMethod = null;
            this.consecutiveErrors = 0;
            
            return true;
        } catch (error) {
            console.error('Error stopping engagement detection:', error);
            return false;
        }
    }
    
    /**
     * Start polling for engagement state updates.
     */
    startPolling() {
        if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
        }
        
        // Poll immediately, then at intervals
        this.pollEngagementState();
        this.pollIntervalId = setInterval(() => {
            this.pollEngagementState();
        }, this.pollInterval);
    }
    
    /**
     * Stop polling for engagement state updates.
     */
    stopPolling() {
        if (this.pollIntervalId) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = null;
        }
    }
    
    /**
     * Poll the backend for current engagement state.
     * Uses request throttling to avoid excessive requests during rapid updates.
     */
    async pollEngagementState() {
        if (!this.isActive) {
            return;
        }
        
        // Throttle: skip if previous request is still pending
        if (this._pendingRequest) {
            return;
        }
        
        try {
            this._pendingRequest = true;
            // Add timestamp to prevent caching
            const response = await fetch(`${this.apiBaseUrl}/engagement/state?t=${Date.now()}`, {
                method: 'GET',
                cache: 'no-cache',
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            
            if (!response.ok) {
                if (response.status === 404) {
                    // Detection not started yet - not an error
                    return;
                }
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            // Reset error counter on success
            this.consecutiveErrors = 0;
            
            // Update bar display if available
            if (this.barDisplay && data.score !== undefined) {
                // Only update if we have valid data
                const score = parseFloat(data.score);
                if (!isNaN(score) && isFinite(score)) {
                    this.barDisplay.update(
                        score,
                        data.level || 'UNKNOWN',
                        data.faceDetected !== undefined ? data.faceDetected : false
                    );
                } else {
                    console.warn('Invalid score received:', data.score);
                }
            } else if (this.barDisplay) {
                // If no score but bar display exists, update with 0
                this.barDisplay.update(0, 'UNKNOWN', false);
            }
            
            // Show MediaPipe or Azure metrics panel based on detection method
            var method = (data.detectionMethod && data.detectionMethod.toLowerCase()) || 'mediapipe';
            this._setMetricsPanelVisibility(method);

            if (method === 'azure_face_api') {
                if (this.azureMetricsPanel) {
                    if (data.faceDetected === false) {
                        this.azureMetricsPanel.reset();
                    } else if (data.azureMetrics != null) {
                        this.azureMetricsPanel.update(data.azureMetrics);
                    }
                }
            } else {
                if (this.signifierPanel) {
                    if (data.faceDetected === false) {
                        this.signifierPanel.reset();
                    } else if (data.signifierScores != null) {
                        this.signifierPanel.update(data.signifierScores);
                    }
                }
            }

            // Engagement alerts (significant drop or plateau): inform via avatar
            if (data.alert && data.alert.message && typeof this.onAlert === 'function') {
                this.onAlert(data.alert);
            }
            
        } catch (error) {
            this.consecutiveErrors++;
            console.debug('Engagement state fetch error:', error);
            
            // Stop polling after too many consecutive errors
            if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
                console.warn('Too many consecutive errors, stopping engagement detection polling');
                this.stopPolling();
                this.isActive = false;
            }
        } finally {
            this._pendingRequest = false;
        }
    }
    
    /**
     * Show signifier panel (MediaPipe) or Azure metrics panel based on detection method.
     * @param {string} detectionMethod - 'mediapipe' | 'azure_face_api'
     */
    _setMetricsPanelVisibility(detectionMethod) {
        var sigEl = document.getElementById('signifierPanelContainer');
        var azureEl = document.getElementById('azureMetricsPanelContainer');
        if (!sigEl || !azureEl) return;
        var isAzure = (detectionMethod && detectionMethod.toLowerCase() === 'azure_face_api');
        sigEl.style.display = isAzure ? 'none' : '';
        azureEl.style.display = isAzure ? '' : 'none';
    }

    /**
     * Start capturing from a display MediaStream and sending frames to the backend.
     * Call this after start('partner') when the user has chosen a tab/window via getDisplayMedia.
     * 
     * @param {MediaStream} stream - Stream from navigator.mediaDevices.getDisplayMedia()
     */
    startPartnerFrameStream(stream) {
        this.stopPartnerFrameStream();
        this._partnerStream = stream;
        
        const video = document.createElement('video');
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;
        video.srcObject = stream;
        this._partnerVideo = video;
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        this._partnerCanvas = canvas;
        
        const maxWidth = 1280;
        const sendFrame = () => {
            if (!this._partnerVideo || !this._partnerStream || !this.isActive) return;
            const v = this._partnerVideo;
            if (v.readyState < 2 || v.videoWidth === 0) return;
            let w = v.videoWidth, h = v.videoHeight;
            if (w > maxWidth) {
                h = Math.round((h * maxWidth) / w);
                w = maxWidth;
            }
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
            ctx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, w, h);
            canvas.toBlob((blob) => {
                if (!blob || !this.isActive) return;
                fetch(`${this.apiBaseUrl}/engagement/partner-frame`, {
                    method: 'POST',
                    body: blob,
                    headers: { 'Content-Type': 'image/jpeg' }
                }).catch(() => {});
            }, 'image/jpeg', 0.75);
        };
        
        video.play().then(() => {
            this._partnerFrameIntervalId = setInterval(sendFrame, 66); // ~15 FPS: lower latency, less memory
            if (typeof this.onPartnerStreamStateChange === 'function') {
                this.onPartnerStreamStateChange(true);
            }
        }).catch((err) => {
            console.warn('Partner video play failed:', err);
            this.stopPartnerFrameStream();
        });
        
        // Stop when user stops sharing (track ended)
        stream.getTracks().forEach((track) => {
            track.onended = () => {
                this.stopPartnerFrameStream();
            };
        });
    }
    
    /**
     * Stop partner frame capture and release display media tracks.
     */
    stopPartnerFrameStream() {
        const hadStream = !!this._partnerStream;
        if (this._partnerFrameIntervalId) {
            clearInterval(this._partnerFrameIntervalId);
            this._partnerFrameIntervalId = null;
        }
        if (this._partnerStream) {
            this._partnerStream.getTracks().forEach((t) => t.stop());
            this._partnerStream = null;
        }
        this._partnerVideo = null;
        this._partnerCanvas = null;
        if (hadStream && typeof this.onPartnerStreamStateChange === 'function') {
            this.onPartnerStreamStateChange(false);
        }
    }
    
    /**
     * Whether the Meeting Partner Video capture (getDisplayMedia) is currently active.
     * @returns {boolean}
     */
    isPartnerStreamActive() {
        return !!this._partnerStream;
    }
    
    /**
     * Get current detection status.
     * 
     * @returns {Object} Status information
     */
    getStatus() {
        return {
            isActive: this.isActive,
            sourceType: this.currentSourceType,
            sourcePath: this.currentSourcePath,
            detectionMethod: this.currentDetectionMethod,
            consecutiveErrors: this.consecutiveErrors
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EngagementDetector;
}
