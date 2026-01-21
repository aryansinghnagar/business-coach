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
     * @param {number} options.pollInterval - Polling interval in milliseconds (default: 500)
     */
    constructor(options = {}) {
        this.barDisplay = options.barDisplay || null;
        this.signifierPanel = options.signifierPanel || null;
        this.apiBaseUrl = options.apiBaseUrl || 'http://localhost:5000';
        this.pollInterval = options.pollInterval || 500;
        this.onAlert = options.onAlert || null;
        
        // State management
        this.isActive = false;
        this.pollIntervalId = null;
        this.currentSourceType = null;
        this.currentSourcePath = null;
        this.currentDetectionMethod = null;
        
        // Error handling
        this.consecutiveErrors = 0;
        this.maxConsecutiveErrors = 5;
    }
    
    /**
     * Start engagement detection.
     * 
     * @param {string} sourceType - Video source type ('webcam', 'file', 'stream')
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
            
            // Reset bar display and signifier panel
            if (this.barDisplay) {
                this.barDisplay.reset();
            }
            if (this.signifierPanel) {
                this.signifierPanel.reset();
            }
            
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
     */
    async pollEngagementState() {
        if (!this.isActive) {
            return;
        }
        
        try {
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
            
            // Update 30 signifier metrics panel
            if (this.signifierPanel) {
                if (data.faceDetected === false) {
                    this.signifierPanel.reset();
                } else if (data.signifierScores != null) {
                    this.signifierPanel.update(data.signifierScores);
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
        }
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
