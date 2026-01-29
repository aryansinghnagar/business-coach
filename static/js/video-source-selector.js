/**
 * Video Source Selector Module
 * 
 * Provides a modal UI for selecting video source for engagement detection.
 * Supports webcam, meeting partner video, and local video file options.
 */

class VideoSourceSelector {
    /**
     * Initialize the video source selector.
     * 
     * @param {Object} options - Configuration options
     * @param {Function} options.onSelect - Callback when source is selected
     * @param {Function} options.onCancel - Callback when selection is canceled
     * @param {string} [options.apiBaseUrl='http://localhost:5000'] - Backend API base URL. Used to fetch face-detection config; if wrong or unreachable, Azure Face API option will be greyed out.
     */
    constructor(options = {}) {
        this.onSelect = options.onSelect || null;
        this.onCancel = options.onCancel || null;
        this.apiBaseUrl = options.apiBaseUrl || 'http://localhost:5000';
        this.modal = null;
        this.fileInput = null;
        this.azureAvailable = false;
        
        this.createModal();
        // Check Azure availability asynchronously and update UI
        this.checkAzureAvailability().then(() => {
            this.updateAzureAvailability();
        });
    }
    
    /**
     * Check if Azure Face API is available.
     */
    async checkAzureAvailability() {
        try {
            const url = `${this.apiBaseUrl.replace(/\/$/, '')}/config/face-detection`;
            const response = await fetch(url);
            if (response.ok) {
                const config = await response.json();
                this.azureAvailable = config.azureFaceApiAvailable || false;
            } else {
                this.azureAvailable = false;
            }
        } catch (error) {
            console.warn('Could not check Azure Face API availability:', error);
            this.azureAvailable = false;
        }
    }
    
    /**
     * Create the modal UI.
     */
    createModal() {
        // Create modal container
        this.modal = document.createElement('div');
        this.modal.id = 'videoSourceModal';
        this.modal.className = 'video-source-modal';
        this.modal.innerHTML = `
            <div class="video-source-modal-overlay"></div>
            <div class="video-source-modal-content">
                <div class="video-source-modal-header">
                    <h2>Select Video Source for Engagement Detection</h2>
                    <p>Choose the video source to analyze for meeting partner engagement</p>
                </div>
                
                <div class="video-source-options">
                    <div class="video-source-option" data-source="webcam">
                        <div class="video-source-icon">📹</div>
                        <div class="video-source-info">
                            <h3>Webcam</h3>
                            <p>Use your default webcam for engagement detection</p>
                        </div>
                        <div class="video-source-radio">
                            <input type="radio" name="videoSource" value="webcam" id="sourceWebcam" checked>
                            <label for="sourceWebcam"></label>
                        </div>
                    </div>
                    
                    <div class="video-source-option" data-source="partner">
                        <div class="video-source-icon">🖥️</div>
                        <div class="video-source-info">
                            <h3>Meeting Partner Video</h3>
                            <p>Share screen and select the meeting tab or window (e.g. pin one participant in Meet, Teams, or Zoom) so we analyze their video.</p>
                        </div>
                        <div class="video-source-radio">
                            <input type="radio" name="videoSource" value="partner" id="sourcePartner">
                            <label for="sourcePartner"></label>
                        </div>
                    </div>
                    
                    <div class="video-source-option" data-source="file">
                        <div class="video-source-icon">📁</div>
                        <div class="video-source-info">
                            <h3>Local Video File</h3>
                            <p>Upload and analyze a recorded video file</p>
                        </div>
                        <div class="video-source-radio">
                            <input type="radio" name="videoSource" value="file" id="sourceFile">
                            <label for="sourceFile"></label>
                        </div>
                    </div>
                </div>
                
                <div class="video-source-file-input" id="fileInputContainer" style="display: none;">
                    <label for="videoFileInput" class="file-input-label">
                        <span class="file-input-button">Choose Video File</span>
                        <span class="file-input-text" id="fileInputText">No file selected</span>
                    </label>
                    <input type="file" id="videoFileInput" accept="video/*" style="display: none;">
                    <div class="file-input-hint">Supported formats: MP4, AVI, MOV, WebM</div>
                </div>
                
                <div class="face-detection-method-selector">
                    <label>Face Detection Method</label>
                    <div class="face-detection-method-option">
                        <input type="radio" name="faceDetectionMethod" value="mediapipe" id="detectionMediaPipe" checked>
                        <div class="face-detection-method-info">
                            <div class="face-detection-method-title">MediaPipe (Default)</div>
                            <div class="face-detection-method-description">
                                Local processing, fast, 468 facial landmarks. Recommended for most users.
                            </div>
                        </div>
                    </div>
                    <div class="face-detection-method-option" id="azureDetectionOption">
                        <input type="radio" name="faceDetectionMethod" value="azure_face_api" id="detectionAzure">
                        <div class="face-detection-method-info">
                            <div class="face-detection-method-title">Azure Face API</div>
                            <div class="face-detection-method-description" id="azureDescription">
                                Cloud-based, 27 landmarks + emotion detection. Requires Azure configuration.
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="video-source-modal-footer">
                    <button class="btn-cancel" id="cancelVideoSource">Cancel</button>
                    <button class="btn-confirm" id="confirmVideoSource">Start Detection</button>
                </div>
            </div>
        `;
        
        // Add to document
        document.body.appendChild(this.modal);
        
        // Update Azure option availability
        this.updateAzureAvailability();
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    /**
     * Setup event listeners for the modal.
     */
    setupEventListeners() {
        // Radio button changes
        const radioButtons = this.modal.querySelectorAll('input[name="videoSource"]');
        radioButtons.forEach(radio => {
            radio.addEventListener('change', (e) => {
                const sourceType = e.target.value;
                const fileInputContainer = this.modal.querySelector('#fileInputContainer');
                
                if (sourceType === 'file') {
                    fileInputContainer.style.display = 'block';
                } else {
                    fileInputContainer.style.display = 'none';
                }
            });
        });
        
        // File input
        this.fileInput = this.modal.querySelector('#videoFileInput');
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            const fileText = this.modal.querySelector('#fileInputText');
            
            if (file) {
                fileText.textContent = file.name;
                fileText.style.color = '#4ade80';
            } else {
                fileText.textContent = 'No file selected';
                fileText.style.color = '#9ca3af';
            }
        });
        
        // Confirm button
        const confirmBtn = this.modal.querySelector('#confirmVideoSource');
        confirmBtn.addEventListener('click', () => {
            this.handleConfirm();
        });
        
        // Cancel button
        const cancelBtn = this.modal.querySelector('#cancelVideoSource');
        cancelBtn.addEventListener('click', () => {
            this.hide();
            if (this.onCancel) {
                this.onCancel();
            }
        });
        
        // Close on overlay click
        const overlay = this.modal.querySelector('.video-source-modal-overlay');
        overlay.addEventListener('click', () => {
            this.hide();
            if (this.onCancel) {
                this.onCancel();
            }
        });
        
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal.classList.contains('active')) {
                this.hide();
                if (this.onCancel) {
                    this.onCancel();
                }
            }
        });
    }
    
    /**
     * Update Azure Face API option availability.
     */
    updateAzureAvailability() {
        const azureOption = this.modal.querySelector('#azureDetectionOption');
        const azureRadio = this.modal.querySelector('#detectionAzure');
        const azureDescription = this.modal.querySelector('#azureDescription');
        
        if (!this.azureAvailable && azureOption && azureRadio) {
            azureRadio.disabled = true;
            if (azureDescription) {
                azureDescription.textContent = 'Azure Face API is not configured. Please set AZURE_FACE_API_KEY and AZURE_FACE_API_ENDPOINT.';
                azureDescription.style.color = 'rgba(255, 255, 255, 0.5)';
            }
        } else if (this.azureAvailable && azureOption && azureRadio) {
            azureRadio.disabled = false;
            if (azureDescription) {
                azureDescription.textContent = 'Cloud-based, 27 landmarks + emotion detection. Requires Azure configuration.';
                azureDescription.style.color = 'rgba(255, 255, 255, 0.7)';
            }
        }
    }
    
    /**
     * Handle confirm button click.
     */
    handleConfirm() {
        const selectedRadio = this.modal.querySelector('input[name="videoSource"]:checked');
        if (!selectedRadio) {
            alert('Please select a video source');
            return;
        }
        
        const sourceType = selectedRadio.value;
        let sourcePath = null;
        
        if (sourceType === 'file') {
            const file = this.fileInput.files[0];
            if (!file) {
                alert('Please select a video file');
                return;
            }
            
            // Store file reference for upload
            this.selectedFile = file;
            sourcePath = file.name; // Will be replaced with server path after upload
        }
        
        // Get selected face detection method (default to mediapipe)
        const detectionMethodRadio = this.modal.querySelector('input[name="faceDetectionMethod"]:checked');
        const detectionMethod = detectionMethodRadio ? detectionMethodRadio.value : 'mediapipe';
        
        // Hide modal first for better UX
        this.hide();
        
        if (this.onSelect) {
            this.onSelect({
                sourceType: sourceType,
                sourcePath: sourcePath,
                file: sourceType === 'file' ? this.selectedFile : null,
                detectionMethod: detectionMethod
            });
        }
    }
    
    /**
     * Show the modal.
     */
    show() {
        if (this.modal) {
            this.modal.classList.add('active');
            document.body.style.overflow = 'hidden';
            // Refresh Azure availability when showing modal
            this.checkAzureAvailability().then(() => {
                this.updateAzureAvailability();
            });
        }
    }
    
    /**
     * Hide the modal.
     */
    hide() {
        if (this.modal) {
            this.modal.classList.remove('active');
            document.body.style.overflow = '';
        }
    }
    
    /**
     * Clean up and remove modal.
     */
    destroy() {
        if (this.modal) {
            this.modal.remove();
            this.modal = null;
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VideoSourceSelector;
}
