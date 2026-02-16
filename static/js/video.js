"use strict";
// ========================================================================
// VIDEO SOURCE SELECTOR (video-source-selector.js)
// ========================================================================
//
// WHAT THIS MODULE DOES (in plain language):
// ------------------------------------------
// Shows a modal so the user can choose where the engagement detector gets its
// video: webcam, meeting partner (screen/tab share), or a local video file.
// When they pick one, we call the callback so the app can start detection.
//
// WHO USES THIS: app-main.js. When the user starts an insights session and no
// video source is set yet, we open this modal; on select we start engagement.
//

class VideoSourceSelector {
    /**
     * Initialize the video source selector.
     * App always uses optimal detection method (auto); no user choice.
     *
     * @param {Object} options - Configuration options
     * @param {Function} options.onSelect - Callback when source is selected
     * @param {Function} options.onCancel - Callback when selection is canceled
     * @param {string} [options.apiBaseUrl='http://localhost:5000'] - Backend API base URL
     */
    constructor(options = {}) {
        this.onSelect = options.onSelect || null;
        this.onCancel = options.onCancel || null;
        this.apiBaseUrl = options.apiBaseUrl || 'http://localhost:5000';
        this.modal = null;
        this.fileInput = null;
        
        this.createModal();
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
                            <p>After clicking Start Detection, the browser will ask you to choose what to share (tab, window, or screen) and whether to share audio. Select the meeting tab or window (e.g. pin one participant in Meet, Teams, or Zoom) and enable audio for facial and speech cues.</p>
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
                
                <div class="video-source-audio-option" id="partnerAudioOptionContainer">
                    <label class="video-source-checkbox-label">
                        <input type="checkbox" id="usePartnerAudioCheckbox" name="usePartnerAudio">
                        <span>Use meeting tab/window audio for speech and voice-tone cues (recommended for richer insights)</span>
                    </label>
                    <label class="video-source-checkbox-label">
                        <input type="checkbox" id="useMicrophoneAudioCheckbox" name="useMicrophoneAudio">
                        <span>Use microphone for speech cues and acoustic analysis (e.g. for testing)</span>
                    </label>
                    <p class="video-source-audio-note" id="partnerAudioIncludedNote" style="display: none;">Partner audio is included automatically.</p>
                </div>
                
                <div class="video-source-modal-footer">
                    <button class="btn-cancel" id="cancelVideoSource">Cancel</button>
                    <button class="btn-confirm" id="confirmVideoSource">Start Detection</button>
                </div>
            </div>
        `;
        
        // Add to document
        document.body.appendChild(this.modal);
        
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
                const partnerAudioCheckbox = this.modal.querySelector('#usePartnerAudioCheckbox');
                const partnerAudioNote = this.modal.querySelector('#partnerAudioIncludedNote');
                
                if (sourceType === 'file') {
                    fileInputContainer.style.display = 'block';
                } else {
                    fileInputContainer.style.display = 'none';
                }
                if (sourceType === 'partner') {
                    if (partnerAudioCheckbox) {
                        partnerAudioCheckbox.checked = false;
                        partnerAudioCheckbox.disabled = true;
                    }
                    const micCheckbox = this.modal.querySelector('#useMicrophoneAudioCheckbox');
                    if (micCheckbox) {
                        micCheckbox.checked = false;
                        micCheckbox.disabled = true;
                    }
                    if (partnerAudioNote) partnerAudioNote.style.display = 'block';
                } else {
                    if (partnerAudioCheckbox) partnerAudioCheckbox.disabled = false;
                    const micCheckbox = this.modal.querySelector('#useMicrophoneAudioCheckbox');
                    if (micCheckbox) micCheckbox.disabled = false;
                    if (partnerAudioNote) partnerAudioNote.style.display = 'none';
                }
            });
        });
        // Mutual exclusivity: microphone vs tab/window audio (for webcam/file only)
        const partnerAudioCheckbox = this.modal.querySelector('#usePartnerAudioCheckbox');
        const micCheckbox = this.modal.querySelector('#useMicrophoneAudioCheckbox');
        if (partnerAudioCheckbox && micCheckbox) {
            partnerAudioCheckbox.addEventListener('change', () => {
                if (partnerAudioCheckbox.checked) micCheckbox.checked = false;
            });
            micCheckbox.addEventListener('change', () => {
                if (micCheckbox.checked) partnerAudioCheckbox.checked = false;
            });
        }
        
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
        
        // Close on Escape key (store ref so we can remove on hide/destroy)
        this._escapeHandler = (e) => {
            if (e.key === 'Escape' && this.modal && this.modal.classList.contains('active')) {
                this.hide();
                if (this.onCancel) this.onCancel();
            }
        };
        document.addEventListener('keydown', this._escapeHandler);
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
        const partnerAudioCheckbox = this.modal.querySelector('#usePartnerAudioCheckbox');
        const micAudioCheckbox = this.modal.querySelector('#useMicrophoneAudioCheckbox');
        const usePartnerAudio = sourceType === 'partner' ? true : (partnerAudioCheckbox ? partnerAudioCheckbox.checked : false);
        const useMicrophoneAudio = (sourceType === 'webcam' || sourceType === 'file') && micAudioCheckbox && micAudioCheckbox.checked;
        
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
        
        // App chooses optimal detection method (auto); no selection passed
        this.hide();
        
        if (this.onSelect) {
            this.onSelect({
                sourceType: sourceType,
                sourcePath: sourcePath,
                file: sourceType === 'file' ? this.selectedFile : null,
                usePartnerAudio: usePartnerAudio,
                useMicrophoneAudio: useMicrophoneAudio
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
        if (this._escapeHandler) {
            document.removeEventListener('keydown', this._escapeHandler);
            this._escapeHandler = null;
        }
    }
    
    /**
     * Clean up and remove modal.
     */
    destroy() {
        if (this._escapeHandler) {
            document.removeEventListener('keydown', this._escapeHandler);
            this._escapeHandler = null;
        }
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
