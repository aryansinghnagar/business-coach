/**
 * Engagement Bar Display Module
 * 
 * This module handles the visual display of engagement scores using a vertical
 * engagement bar with dynamic color gradients (red to yellow to green).
 * 
 * Features:
 * - Dynamic color gradient based on engagement score (0-100)
 * - Real-time updates (no averaging) for quick-actionable insights
 * - Modular and reusable design
 */

class EngagementBarDisplay {
    /**
     * Initialize the engagement bar display system.
     * 
     * @param {Object} options - Configuration options
     * @param {number} options.updateInterval - Update interval in milliseconds (default: 500)
     * @param {string} options.barFillId - ID of the bar fill element (default: 'engagementBarFill')
     * @param {string} options.barValueId - ID of the bar value element (default: 'engagementBarValue')
     * @param {string} options.levelIndicatorId - ID of the level indicator element (default: 'engagementLevel')
     */
    constructor(options = {}) {
        this.updateInterval = options.updateInterval || 500;
        this.barFillId = options.barFillId || 'engagementBarFill';
        this.barValueId = options.barValueId || 'engagementBarValue';
        this.levelIndicatorId = options.levelIndicatorId || 'engagementLevel';
        
        // DOM elements (cached for performance)
        this.barFill = null;
        this.barValue = null;
        this.levelIndicator = null;
        
        // Current values (real-time)
        this.currentScore = 0;
        this.currentLevel = 'UNKNOWN';
        this.currentFaceDetected = false;
        /** Smoothed score (EMA) for fluid bar animation. */
        this.smoothedScore = null;
        /** EMA alpha: higher = more responsive, lower = smoother. */
        this.smoothAlpha = options.smoothAlpha != null ? options.smoothAlpha : 0.35;

        // Initialize DOM elements
        this.initializeElements();
    }
    
    /**
     * Initialize and cache DOM elements.
     */
    initializeElements() {
        this.barFill = document.getElementById(this.barFillId);
        this.barValue = document.getElementById(this.barValueId);
        this.levelIndicator = document.getElementById(this.levelIndicatorId);
        
        if (!this.barFill || !this.barValue || !this.levelIndicator) {
            console.warn('Engagement bar elements not found. Ensure HTML is loaded.');
        }
    }
    
    /**
     * Update the engagement bar with a new score (real-time, no averaging).
     * 
     * @param {number} score - Engagement score (0-100)
     * @param {string} level - Engagement level (VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
     * @param {boolean} faceDetected - Whether a face was detected
     */
    update(score, level, faceDetected) {
        const validScore = Math.max(0, Math.min(100, parseFloat(score) || 0));
        const validLevel = level || 'UNKNOWN';
        const validFaceDetected = Boolean(faceDetected);
        
        this.currentScore = validScore;
        this.currentLevel = validLevel;
        this.currentFaceDetected = validFaceDetected;
        // Smooth score for fluid bar movement (EMA)
        if (validFaceDetected) {
            this.smoothedScore = this.smoothedScore != null
                ? this.smoothAlpha * validScore + (1 - this.smoothAlpha) * this.smoothedScore
                : validScore;
        } else {
            this.smoothedScore = null;
        }
        
        this.render();
    }
    
    /**
     * Render the engagement bar with current values.
     */
    render() {
        if (!this.barFill || !this.barValue || !this.levelIndicator) {
            return;
        }
        
        // Use smoothed score for display when face detected, else 0
        const displayScore = this.currentFaceDetected && this.smoothedScore != null
            ? this.smoothedScore
            : this.currentScore;
        const heightPercent = Math.max(0, Math.min(100, displayScore));
        this.barFill.style.height = heightPercent + '%';
        
        // Update bar color based on displayed score
        const color = this.getColorForScore(displayScore);
        this.barFill.style.background = color;
        
        // Update value display (show smoothed score when face detected)
        if (this.currentFaceDetected) {
            const val = (this.smoothedScore != null ? this.smoothedScore : this.currentScore);
            this.barValue.textContent = Math.round(val);
        } else {
            this.barValue.textContent = '--';
        }
        
        // Update level indicator
        this.updateLevelIndicator();
    }
    
    /**
     * Get color gradient for a given score.
     * 
     * Interpolates from red (0) -> yellow (50) -> green (100)
     * 
     * @param {number} score - Engagement score (0-100)
     * @returns {string} CSS gradient string
     */
    getColorForScore(score) {
        // Clamp score to 0-100
        score = Math.max(0, Math.min(100, score));
        
        let r, g, b;
        
        if (score <= 50) {
            // Red (255, 0, 0) to Yellow (255, 255, 0)
            const ratio = score / 50;
            r = 255;
            g = Math.round(255 * ratio);
            b = 0;
        } else {
            // Yellow (255, 255, 0) to Green (0, 255, 0)
            const ratio = (score - 50) / 50;
            r = Math.round(255 * (1 - ratio));
            g = 255;
            b = 0;
        }
        
        // Create gradient from bottom to top
        // Bottom color is more intense, top is lighter
        const bottomColor = `rgb(${r}, ${g}, ${b})`;
        const topColor = `rgba(${r}, ${g}, ${b}, 0.7)`;
        
        return `linear-gradient(180deg, ${topColor} 0%, ${bottomColor} 100%)`;
    }
    
    /**
     * Update the level indicator display.
     */
    updateLevelIndicator() {
        if (!this.levelIndicator) {
            return;
        }
        
        if (!this.currentFaceDetected) {
            this.levelIndicator.textContent = 'No Face';
            this.levelIndicator.style.background = 'rgba(107, 114, 128, 0.3)';
            return;
        }
        
        // Format level text
        const levelText = this.currentLevel.replace('_', ' ');
        this.levelIndicator.textContent = levelText;
        
        // Set background color based on level
        const levelColors = {
            'VERY_HIGH': 'rgba(34, 197, 94, 0.3)',   // Green
            'HIGH': 'rgba(34, 197, 94, 0.3)',        // Green
            'MEDIUM': 'rgba(251, 191, 36, 0.3)',     // Yellow
            'LOW': 'rgba(239, 68, 68, 0.3)',         // Red
            'VERY_LOW': 'rgba(239, 68, 68, 0.3)',    // Red
            'UNKNOWN': 'rgba(107, 114, 128, 0.3)'     // Gray
        };
        
        this.levelIndicator.style.background = levelColors[this.currentLevel] || levelColors['UNKNOWN'];
    }
    
    /**
     * Reset the engagement bar to initial state.
     */
    reset() {
        this.currentScore = 0;
        this.currentLevel = 'UNKNOWN';
        this.currentFaceDetected = false;
        this.smoothedScore = null;
        this.render();
    }
    
    /**
     * Get current averaged score.
     * 
     * @returns {number} Current averaged score
     */
    getCurrentScore() {
        return this.currentScore;
    }
    
    /**
     * Get current averaged level.
     * 
     * @returns {string} Current averaged level
     */
    getCurrentLevel() {
        return this.currentLevel;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EngagementBarDisplay;
}
