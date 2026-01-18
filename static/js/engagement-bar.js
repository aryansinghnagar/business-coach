/**
 * Engagement Bar Display Module
 * 
 * This module handles the visual display of engagement scores using a vertical
 * engagement bar with dynamic color gradients (red to yellow to green) and
 * smooth updates through frame averaging.
 * 
 * Features:
 * - Dynamic color gradient based on engagement score (0-100)
 * - 10-frame averaging for smooth display updates
 * - Smooth animations and transitions
 * - Modular and reusable design
 */

class EngagementBarDisplay {
    /**
     * Initialize the engagement bar display system.
     * 
     * @param {Object} options - Configuration options
     * @param {number} options.averagingWindow - Number of frames to average (default: 10)
     * @param {number} options.updateInterval - Update interval in milliseconds (default: 500)
     * @param {string} options.barFillId - ID of the bar fill element (default: 'engagementBarFill')
     * @param {string} options.barValueId - ID of the bar value element (default: 'engagementBarValue')
     * @param {string} options.levelIndicatorId - ID of the level indicator element (default: 'engagementLevel')
     */
    constructor(options = {}) {
        this.averagingWindow = options.averagingWindow || 10;
        this.updateInterval = options.updateInterval || 500;
        this.barFillId = options.barFillId || 'engagementBarFill';
        this.barValueId = options.barValueId || 'engagementBarValue';
        this.levelIndicatorId = options.levelIndicatorId || 'engagementLevel';
        
        // Score history for averaging
        this.scoreHistory = [];
        this.levelHistory = [];
        this.faceDetectedHistory = [];
        
        // DOM elements (cached for performance)
        this.barFill = null;
        this.barValue = null;
        this.levelIndicator = null;
        
        // Current smoothed values
        this.currentScore = 0;
        this.currentLevel = 'UNKNOWN';
        this.currentFaceDetected = false;
        
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
     * Update the engagement bar with a new score.
     * 
     * @param {number} score - Engagement score (0-100)
     * @param {string} level - Engagement level (VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH)
     * @param {boolean} faceDetected - Whether a face was detected
     */
    update(score, level, faceDetected) {
        // Validate input
        const validScore = Math.max(0, Math.min(100, parseFloat(score) || 0));
        const validLevel = level || 'UNKNOWN';
        const validFaceDetected = Boolean(faceDetected);
        
        // Add to history
        this.scoreHistory.push(validScore);
        this.levelHistory.push(validLevel);
        this.faceDetectedHistory.push(validFaceDetected);
        
        // Maintain history window (keep last 10 frames)
        if (this.scoreHistory.length > this.averagingWindow) {
            this.scoreHistory.shift();
            this.levelHistory.shift();
            this.faceDetectedHistory.shift();
        }
        
        // Calculate averaged values over last 10 frames
        const averagedScore = this.calculateAverageScore();
        const averagedLevel = this.calculateAverageLevel();
        const averagedFaceDetected = this.calculateAverageFaceDetected();
        
        // Update display only if values changed significantly (avoid unnecessary renders)
        const scoreChanged = Math.abs(this.currentScore - averagedScore) > 0.5;
        const levelChanged = this.currentLevel !== averagedLevel;
        const faceChanged = this.currentFaceDetected !== averagedFaceDetected;
        
        if (scoreChanged || levelChanged || faceChanged) {
            this.currentScore = averagedScore;
            this.currentLevel = averagedLevel;
            this.currentFaceDetected = averagedFaceDetected;
            
            this.render();
        }
    }
    
    /**
     * Calculate average score from history.
     * 
     * @returns {number} Averaged score (0-100)
     */
    calculateAverageScore() {
        if (this.scoreHistory.length === 0) {
            return 0;
        }
        
        const sum = this.scoreHistory.reduce((acc, score) => acc + score, 0);
        return sum / this.scoreHistory.length;
    }
    
    /**
     * Calculate most common level from history.
     * 
     * @returns {string} Most frequent engagement level
     */
    calculateAverageLevel() {
        if (this.levelHistory.length === 0) {
            return 'UNKNOWN';
        }
        
        // Count occurrences of each level
        const levelCounts = {};
        this.levelHistory.forEach(level => {
            levelCounts[level] = (levelCounts[level] || 0) + 1;
        });
        
        // Find most common level
        let maxCount = 0;
        let mostCommonLevel = 'UNKNOWN';
        for (const [level, count] of Object.entries(levelCounts)) {
            if (count > maxCount) {
                maxCount = count;
                mostCommonLevel = level;
            }
        }
        
        return mostCommonLevel;
    }
    
    /**
     * Calculate if face is detected (true if majority of recent frames detected face).
     * 
     * @returns {boolean} True if face detected in majority of frames
     */
    calculateAverageFaceDetected() {
        if (this.faceDetectedHistory.length === 0) {
            return false;
        }
        
        const trueCount = this.faceDetectedHistory.filter(detected => detected).length;
        return trueCount > this.faceDetectedHistory.length / 2;
    }
    
    /**
     * Render the engagement bar with current values.
     */
    render() {
        if (!this.barFill || !this.barValue || !this.levelIndicator) {
            return;
        }
        
        // Update bar height (0-100%)
        const heightPercent = Math.max(0, Math.min(100, this.currentScore));
        this.barFill.style.height = heightPercent + '%';
        
        // Update bar color based on score
        const color = this.getColorForScore(this.currentScore);
        this.barFill.style.background = color;
        
        // Update value display
        if (this.currentFaceDetected) {
            this.barValue.textContent = Math.round(this.currentScore);
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
        this.scoreHistory = [];
        this.levelHistory = [];
        this.faceDetectedHistory = [];
        
        this.currentScore = 0;
        this.currentLevel = 'UNKNOWN';
        this.currentFaceDetected = false;
        
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
