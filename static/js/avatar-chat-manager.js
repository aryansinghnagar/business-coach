/**
 * Avatar Chat Manager Module
 * 
 * Manages avatar chat functionality including speech synthesis, streaming,
 * and interruption handling. Works in parallel with engagement detection.
 * 
 * Features:
 * - Streaming chat responses with interruption support
 * - Speech synthesis coordination
 * - Parallel operation with engagement detection
 * - Clean state management
 */

class AvatarChatManager {
    /**
     * Initialize the avatar chat manager.
     * 
     * @param {Object} options - Configuration options
     * @param {SessionManager} options.sessionManager - Session manager instance
     * @param {Object} options.avatarSynthesizer - Azure Avatar synthesizer instance
     * @param {Object} options.appConfig - Application configuration
     */
    constructor(options = {}) {
        this.sessionManager = options.sessionManager || null;
        this.avatarSynthesizer = options.avatarSynthesizer || null;
        this.appConfig = options.appConfig || null;
        
        // Speech state
        this.isSpeaking = false;
        this.isInterrupted = false;
        this.spokenTextQueue = [];
        this.speakingText = ''; // Exposed for subtitle updates
        
        // Chat state
        this.messages = [];
        this.sentenceLevelPunctuations = ['.', '?', '!', ':', ';', '。', '？', '！', '：', '；'];
        this.enableDisplayTextAlignmentWithSpeech = true;
        
        // UI update callbacks
        this.onMessageAdded = null;
        this.onResponseChunk = null;
        this.onResponseComplete = null;
        
        // Streaming state
        this.assistantReply = '';
        this.displaySentence = '';
    }
    
    /**
     * Handle user query and stream response.
     * 
     * @param {string} userQuery - User's message
     * @param {string} userQueryHTML - HTML formatted user query
     * @param {string} imgUrlPath - Optional image URL path
     */
    async handleUserQuery(userQuery, userQueryHTML, imgUrlPath) {
        // Interrupt any current streaming or speaking
        this.interruptCurrentResponse();
        
        // Reset streaming state
        this.assistantReply = '';
        this.displaySentence = '';
        
        // Prepare message content
        let contentMessage = userQuery;
        if (imgUrlPath && imgUrlPath.trim()) {
            contentMessage = [
                { "type": "text", "text": userQuery },
                { "type": "image_url", "image_url": { "url": imgUrlPath } }
            ];
        }
        
        // Add user message
        const chatMessage = {
            role: 'user',
            content: contentMessage
        };
        this.messages.push(chatMessage);
        
        // Update UI - show user message
        const chatHistoryTextArea = document.getElementById('chatHistory');
        if (chatHistoryTextArea) {
            if (chatHistoryTextArea.innerHTML !== '' && !chatHistoryTextArea.innerHTML.endsWith('\n\n')) {
                chatHistoryTextArea.innerHTML += '\n\n';
            }
            chatHistoryTextArea.innerHTML += imgUrlPath.trim() 
                ? "<br/><br/>User: " + (userQueryHTML || userQuery)
                : "<br/><br/>User: " + userQuery + "<br/>";
            chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight;
        }
        
        // Wait briefly to ensure interruption is processed
        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (!this.appConfig) {
            console.error('Configuration not loaded');
            return;
        }
        
        // Show assistant response header
        if (chatHistoryTextArea) {
            chatHistoryTextArea.innerHTML += imgUrlPath.trim() ? 'Assistant: ' : '<br/>Assistant: ';
        }
        
        // Stream chat response
        await this.sessionManager.streamChatResponse(
            this.messages,
            {
                enableOyd: this.appConfig.cognitiveSearch && this.appConfig.cognitiveSearch.enabled,
                systemPrompt: this.appConfig.systemPrompt,
                includeEngagement: true,
                cognitiveSearchEnabled: this.appConfig.cognitiveSearch && this.appConfig.cognitiveSearch.enabled,
                onChunk: (token, fullReply) => this.handleStreamChunk(token, fullReply),
                onComplete: (result) => this.handleStreamComplete(result),
                onError: (error) => this.handleStreamError(error)
            }
        );
    }
    
    /**
     * Handle streaming chunk.
     * 
     * @param {string} token - Current token
     * @param {string} fullReply - Full reply so far
     */
    handleStreamChunk(token, fullReply) {
        if (!this.sessionManager.shouldContinueStreaming()) {
            return;
        }
        
        // Update full reply
        this.assistantReply = fullReply;
        this.displaySentence += token;
        
        // Update UI display
        if (!this.enableDisplayTextAlignmentWithSpeech) {
            const chatHistoryTextArea = document.getElementById('chatHistory');
            if (chatHistoryTextArea) {
                chatHistoryTextArea.innerHTML += token.replace(/\n/g, '<br/>');
                chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight;
                this.displaySentence = '';
            }
        }
        
        // Queue for speech synthesis
        this.queueSpeechToken(token);
    }
    
    /**
     * Handle streaming completion.
     * 
     * @param {Object} result - Result object with assistantReply and toolContent
     */
    handleStreamComplete(result) {
        // Add tool message if present
        if (result.toolContent) {
            this.messages.push({ role: 'tool', content: result.toolContent });
        }
        
        // Add assistant message
        this.messages.push({ role: 'assistant', content: result.assistantReply });
        
        // Finalize UI display
        if (this.enableDisplayTextAlignmentWithSpeech && this.displaySentence) {
            const chatHistoryTextArea = document.getElementById('chatHistory');
            if (chatHistoryTextArea) {
                chatHistoryTextArea.innerHTML += this.displaySentence.replace(/\n/g, '<br/>');
                chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight;
                this.displaySentence = '';
            }
        }
        
        // Speak any remaining queued text
        this.finalizeSpeech();
    }
    
    /**
     * Handle streaming error.
     * 
     * @param {Error} error - Error object
     */
    handleStreamError(error) {
        console.error('Error in chat streaming:', error);
        const chatHistoryTextArea = document.getElementById('chatHistory');
        if (chatHistoryTextArea) {
            chatHistoryTextArea.innerHTML += '<br/>Error: ' + error.message;
            chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight;
        }
    }
    
    /**
     * Queue token for speech synthesis.
     * Always appends to buffer first, then speaks complete sentences.
     * Ensures no text is lost and multi-paragraph responses play fully.
     * 
     * @param {string} token - Token to queue
     */
    queueSpeechToken(token) {
        if (!this.avatarSynthesizer || (this.sessionManager && !this.sessionManager.shouldContinueStreaming())) {
            return;
        }
        
        // Always append token first to avoid losing any text
        this.speakingText += token;
        
        // Repeatedly extract and speak complete sentences, keep remainder
        while (this.speakingText.length > 0) {
            const match = this.findFirstSentenceEnd(this.speakingText);
            if (!match) {
                break;
            }
            const toSpeak = match[1];  // including the punctuation
            const rest = match[2];
            this.speakingText = rest;
            if (toSpeak.trim().length > 0) {
                this.speak(toSpeak.trim());
                // Only speak one sentence per call; rest will be handled on next token or loop
                break;
            }
        }
    }
    
    /**
     * Find the first sentence boundary in text.
     * @returns {null|Array} Null if no boundary, else [fullMatch, sentenceWithPunct, rest]
     */
    findFirstSentenceEnd(text) {
        if (!text || text.length === 0) return null;
        const puncts = this.sentenceLevelPunctuations;
        let earliest = -1;
        let which = '';
        for (const p of puncts) {
            const i = text.indexOf(p);
            if (i !== -1 && (earliest === -1 || i < earliest)) {
                earliest = i;
                which = p;
            }
        }
        // Also treat newline as sentence end
        const n = text.indexOf('\n');
        if (n !== -1 && (earliest === -1 || n < earliest)) {
            earliest = n;
            which = '\n';
        }
        if (earliest === -1) return null;
        const sentence = text.substring(0, earliest + which.length);
        const rest = text.substring(earliest + which.length);
        return [which, sentence, rest];
    }
    
    /**
     * Finalize speech synthesis (speak remaining text).
     */
    finalizeSpeech() {
        if (this.speakingText.trim() && !this.isInterrupted) {
            this.speak(this.speakingText);
            this.speakingText = '';
        }
    }
    
    /**
     * Speak text using avatar synthesizer.
     * 
     * @param {string} text - Text to speak
     * @param {number} endingSilenceMs - Optional ending silence in milliseconds
     */
    speak(text, endingSilenceMs = 0) {
        if (!this.avatarSynthesizer || !text.trim() || this.isInterrupted) {
            return;
        }
        
        // Check if already speaking
        if (this.isSpeaking) {
            this.spokenTextQueue.push(text);
            return;
        }
        
        // Reset interrupted flag
        this.isInterrupted = false;
        this.isSpeaking = true;
        this.speakingText = text;
        
        // HTML encode text
        const htmlEncode = (text) => {
            const entityMap = {
                '&': '&amp;', '<': '&lt;', '>': '&gt;',
                '"': '&quot;', "'": '&#39;', '/': '&#x2F;'
            };
            return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match]);
        };
        
        // Create SSML using TTS voice from config
        const ttsVoice = this.appConfig.sttTts.ttsVoice;
        let ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='${ttsVoice}'><mstts:leadingsilence-exact value='0'/>${htmlEncode(text)}</voice></speak>`;
        
        if (endingSilenceMs > 0) {
            ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'><voice name='${ttsVoice}'><mstts:leadingsilence-exact value='0'/>${htmlEncode(text)}<break time='${endingSilenceMs}ms' /></voice></speak>`;
        }
        
        // Update subtitles if enabled
        if (this.appConfig.avatar.showSubtitles) {
            const subtitles = document.getElementById('subtitles');
            if (subtitles) {
                subtitles.hidden = false;
                subtitles.innerHTML = text;
            }
        }
        
        // Speak and ensure we continue with the queue when done
        const advanceQueue = () => {
            this.speakingText = '';
            if (this.appConfig && this.appConfig.avatar && this.appConfig.avatar.showSubtitles) {
                const subtitles = document.getElementById('subtitles');
                if (subtitles) subtitles.hidden = true;
            }
            if (!this.isInterrupted && this.spokenTextQueue.length > 0) {
                this.speak(this.spokenTextQueue.shift());
            } else {
                this.isSpeaking = false;
            }
        };

        this.avatarSynthesizer.speakSsmlAsync(ssml).then((result) => {
            if (this.isInterrupted) {
                this.isInterrupted = false;
                this.isSpeaking = false;
                this.spokenTextQueue = [];
                this.speakingText = '';
                return;
            }
            
            if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
                advanceQueue();
            } else if (result.reason === SpeechSDK.ResultReason.Canceled) {
                this.isSpeaking = false;
                this.speakingText = '';
                if (this.appConfig && this.appConfig.avatar && this.appConfig.avatar.showSubtitles) {
                    const subtitles = document.getElementById('subtitles');
                    if (subtitles) subtitles.hidden = true;
                }
                if (!this.isInterrupted && this.spokenTextQueue.length > 0) {
                    this.speak(this.spokenTextQueue.shift());
                }
            } else {
                advanceQueue();
            }
        }).catch((error) => {
            if (this.isInterrupted) {
                this.isInterrupted = false;
                this.isSpeaking = false;
                return;
            }
            console.error('Error speaking:', error);
            this.isSpeaking = false;
            this.speakingText = '';
            if (this.appConfig && this.appConfig.avatar && this.appConfig.avatar.showSubtitles) {
                const subtitles = document.getElementById('subtitles');
                if (subtitles) subtitles.hidden = true;
            }
            if (!this.isInterrupted && this.spokenTextQueue.length > 0) {
                this.speak(this.spokenTextQueue.shift());
            }
        });
    }
    
    /**
     * Interrupt current response (streaming and speaking).
     */
    interruptCurrentResponse() {
        // Stop streaming
        if (this.sessionManager) {
            this.sessionManager.interruptStreaming();
        }
        
        // Stop speaking
        this.stopSpeaking();
    }
    
    /**
     * Stop speaking.
     */
    stopSpeaking() {
        this.isInterrupted = true;
        this.spokenTextQueue = [];
        this.speakingText = '';
        
        if (this.avatarSynthesizer && this.isSpeaking) {
            this.avatarSynthesizer.stopSpeakingAsync().then(() => {
                this.isSpeaking = false;
                this.isInterrupted = false;
            }).catch((error) => {
                console.debug('Error stopping speech:', error);
                this.isSpeaking = false;
                this.isInterrupted = false;
            });
        } else {
            this.isSpeaking = false;
            this.isInterrupted = false;
        }
    }
    
    /**
     * Initialize messages array with system prompt.
     */
    initMessages() {
        this.messages = [];
        
        if (!this.appConfig) {
            console.error('Configuration not loaded');
            return;
        }
        
        // Add system message
        const systemMessage = {
            role: 'system',
            content: this.appConfig.systemPrompt
        };
        this.messages.push(systemMessage);
    }
    
    /**
     * Clear chat history.
     */
    clearChatHistory() {
        this.messages = [];
        const chatHistoryTextArea = document.getElementById('chatHistory');
        if (chatHistoryTextArea) {
            chatHistoryTextArea.innerHTML = '';
        }
        this.initMessages();
    }
    
    /**
     * Get current messages.
     * 
     * @returns {Array} Messages array
     */
    getMessages() {
        return this.messages;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AvatarChatManager;
}
