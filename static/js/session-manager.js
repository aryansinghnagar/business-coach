/**
 * Session Manager Module
 * 
 * Manages parallel operation of avatar chat and engagement detection systems.
 * Handles session lifecycle, interruption handling, and coordination between systems.
 * 
 * Features:
 * - Parallel operation of avatar chat and engagement detection
 * - Streaming interruption handling
 * - Unified session state management
 * - Clean separation of concerns
 */

class SessionManager {
    /**
     * Initialize the session manager.
     * 
     * @param {Object} options - Configuration options
     * @param {EngagementDetector} options.engagementDetector - Engagement detector instance
     * @param {string} options.apiBaseUrl - Base URL for API (default: 'http://localhost:5000')
     */
    constructor(options = {}) {
        this.engagementDetector = options.engagementDetector || null;
        this.apiBaseUrl = options.apiBaseUrl || 'http://localhost:5000';
        
        // Session state
        this.isSessionActive = false;
        this.isAvatarInitialized = false;
        
        // Streaming state
        this.currentStreamReader = null;
        this.isStreaming = false;
        this.shouldStopStreaming = false;
        
        // Callbacks
        this.onStreamChunk = null;
        this.onStreamComplete = null;
        this.onStreamError = null;
    }
    
    /**
     * Start a session with both avatar chat and engagement detection.
     * 
     * @param {string} engagementSourceType - Video source type for engagement detection
     * @param {string|null} engagementSourcePath - Path for engagement detection
     * @param {string} detectionMethod - Detection method ('mediapipe' or 'azure_face_api')
     * @returns {Promise<boolean>} True if session started successfully
     */
    async startSession(engagementSourceType = 'stream', engagementSourcePath = null, detectionMethod = null) {
        try {
            // Start engagement detection in parallel
            if (this.engagementDetector) {
                await this.engagementDetector.start(engagementSourceType, engagementSourcePath, detectionMethod);
            }
            
            this.isSessionActive = true;
            return true;
        } catch (error) {
            console.error('Error starting session:', error);
            return false;
        }
    }
    
    /**
     * Stop the session and clean up all resources.
     * 
     * @returns {Promise<boolean>} True if session stopped successfully
     */
    async stopSession() {
        try {
            // Stop engagement detection
            if (this.engagementDetector) {
                await this.engagementDetector.stop();
            }
            
            // Stop any active streaming
            this.interruptStreaming();
            
            this.isSessionActive = false;
            this.isAvatarInitialized = false;
            
            return true;
        } catch (error) {
            console.error('Error stopping session:', error);
            return false;
        }
    }
    
    /**
     * Interrupt current streaming response.
     * Can be called when user sends a new message or speaks.
     */
    interruptStreaming() {
        this.shouldStopStreaming = true;
        
        // Cancel the stream reader if active
        if (this.currentStreamReader) {
            try {
                this.currentStreamReader.cancel();
            } catch (error) {
                console.debug('Error canceling stream:', error);
            }
            this.currentStreamReader = null;
        }
        
        this.isStreaming = false;
    }
    
    /**
     * Check if streaming should continue.
     * Streaming continues when avatar is ready and/or engagement session is active,
     * so chat works as soon as the avatar is initialized (even before video source is picked).
     *
     * @returns {boolean} True if streaming should continue
     */
    shouldContinueStreaming() {
        return !this.shouldStopStreaming && (this.isSessionActive || this.isAvatarInitialized);
    }
    
    /**
     * Start streaming chat response.
     * 
     * @param {Array} messages - Chat messages array
     * @param {Object} options - Streaming options
     * @param {Function} options.onChunk - Callback for each chunk
     * @param {Function} options.onComplete - Callback when complete
     * @param {Function} options.onError - Callback for errors
     * @returns {Promise<void>}
     */
    async streamChatResponse(messages, options = {}) {
        // Reset interruption flag
        this.shouldStopStreaming = false;
        this.isStreaming = true;
        
        // Store callbacks
        this.onStreamChunk = options.onChunk || null;
        this.onStreamComplete = options.onComplete || null;
        this.onStreamError = options.onError || null;
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: messages,
                    enableOyd: options.enableOyd || false,
                    systemPrompt: options.systemPrompt || null,
                    includeEngagement: options.includeEngagement !== false // Default to true
                })
            });
            
            if (!response.ok) {
                throw new Error(`Chat API response status: ${response.status} ${response.statusText}`);
            }
            
            const reader = response.body.getReader();
            this.currentStreamReader = reader;
            const decoder = new TextDecoder();
            
            let assistantReply = '';
            let toolContent = '';
            let previousChunkString = '';
            
            while (true) {
                // Check if streaming should be interrupted
                if (!this.shouldContinueStreaming()) {
                    reader.cancel();
                    break;
                }
                
                const { value, done } = await reader.read();
                
                if (done) {
                    break;
                }
                
                let chunkString = decoder.decode(value, { stream: true });
                if (previousChunkString !== '') {
                    chunkString = previousChunkString + chunkString;
                }
                
                // Check if chunk is complete
                if (!chunkString.endsWith('}\n\n') && !chunkString.endsWith('[DONE]\n\n')) {
                    previousChunkString = chunkString;
                    continue;
                }
                
                previousChunkString = '';
                
                // Process complete chunks
                chunkString.split('\n\n').forEach((line) => {
                    if (!this.shouldContinueStreaming()) {
                        return;
                    }
                    
                    try {
                        if (line.startsWith('data:') && !line.endsWith('[DONE]')) {
                            const responseJson = JSON.parse(line.substring(5).trim());
                            let responseToken = undefined;
                            
                            // Handle both standard and On Your Data (OYD) response formats
                            if (!options.enableOyd || !options.cognitiveSearchEnabled) {
                                // Standard format
                                if (responseJson.choices && responseJson.choices[0] && responseJson.choices[0].delta) {
                                    responseToken = responseJson.choices[0].delta.content;
                                }
                            } else {
                                // On Your Data format
                                if (responseJson.choices && responseJson.choices[0] && responseJson.choices[0].messages) {
                                    const message = responseJson.choices[0].messages[0];
                                    if (message.delta) {
                                        const role = message.delta.role;
                                        if (role === 'tool') {
                                            toolContent = (toolContent || '') + (message.delta.content || '');
                                        } else {
                                            responseToken = message.delta.content;
                                        }
                                    }
                                }
                            }
                            
                            if (responseToken !== undefined && responseToken !== null) {
                                assistantReply += responseToken;
                                
                                // Call chunk callback
                                if (this.onStreamChunk) {
                                    this.onStreamChunk(responseToken, assistantReply);
                                }
                            }
                        }
                    } catch (error) {
                        console.debug(`Error parsing chunk: ${error}`);
                    }
                });
            }
            
            // Streaming complete
            this.isStreaming = false;
            this.currentStreamReader = null;
            
            if (this.shouldStopStreaming) {
                // Stream was interrupted
                return;
            }
            
            // Call completion callback
            if (this.onStreamComplete) {
                this.onStreamComplete({
                    assistantReply: assistantReply,
                    toolContent: toolContent
                });
            }
            
        } catch (error) {
            this.isStreaming = false;
            this.currentStreamReader = null;
            
            if (this.shouldStopStreaming) {
                // Interrupted, not an error
                return;
            }
            
            // Call error callback
            if (this.onStreamError) {
                this.onStreamError(error);
            } else {
                console.error('Error streaming chat response:', error);
            }
        }
    }
    
    /**
     * Get current session status.
     * 
     * @returns {Object} Status information
     */
    getStatus() {
        return {
            isSessionActive: this.isSessionActive,
            isAvatarInitialized: this.isAvatarInitialized,
            isStreaming: this.isStreaming,
            engagementActive: this.engagementDetector ? this.engagementDetector.isActive : false
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SessionManager;
}
