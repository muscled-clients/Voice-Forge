// VoiceForge Advanced Playground - Real-time Implementation
class VoiceForgePlayground {
    constructor() {
        // Debug mode - set to false for production
        this.DEBUG_MODE = false;
        
        this.ws = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isStreaming = false;
        this.sessionId = null;
        this.recordingStartTime = null;
        this.currentMode = null;
        this.lastResults = {}; // Store results per mode
        
        // Configuration
        this.config = {
            wsUrl: 'ws://localhost:8000/ws/v1/transcribe',
            sampleRate: 16000,
            language: 'auto',
            speakerDiarization: true,
            languageDetection: true,
            noiseReduction: true,
            interimResults: true,
            vadEnabled: true
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkBrowserSupport();
    }
    
    // Debug logging helper
    debugLog(message, ...args) {
        if (this.DEBUG_MODE) {
            console.log(message, ...args);
        }
    }
    
    checkBrowserSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.');
            return false;
        }
        return true;
    }
    
    setupEventListeners() {
        // Mode switching
        const modeButtons = {
            'mode-upload': 'upload-area',
            'mode-youtube': 'youtube-area',
            'mode-record': 'record-area',
            'mode-stream': 'stream-area'
        };
        
        Object.keys(modeButtons).forEach(buttonId => {
            const btn = document.getElementById(buttonId);
            if (btn) {
                btn.addEventListener('click', () => this.switchMode(buttonId, modeButtons));
            }
        });
        
        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('advanced-file-input');
        
        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('border-cyan-400');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('border-cyan-400');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('border-cyan-400');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            });
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }
        
        // YouTube transcription
        const youtubeBtn = document.getElementById('youtube-transcribe-btn');
        if (youtubeBtn) {
            youtubeBtn.addEventListener('click', () => this.handleYouTubeTranscribe());
        }
        
        // Record button
        const recordBtn = document.getElementById('record-btn');
        if (recordBtn) {
            recordBtn.addEventListener('click', () => this.toggleRecording());
        }
        
        // Stream button
        const streamBtn = document.getElementById('stream-btn');
        if (streamBtn) {
            streamBtn.addEventListener('click', () => this.toggleStreaming());
        }
        
        // Feature toggles
        document.getElementById('enable-diarization')?.addEventListener('change', (e) => {
            this.config.speakerDiarization = e.target.checked;
        });
        document.getElementById('enable-language-detect')?.addEventListener('change', (e) => {
            this.config.languageDetection = e.target.checked;
        });
        document.getElementById('enable-noise-reduction')?.addEventListener('change', (e) => {
            this.config.noiseReduction = e.target.checked;
        });
    }
    
    switchMode(buttonId, modeButtons) {
        // Reset all buttons and areas
        Object.keys(modeButtons).forEach(id => {
            const btn = document.getElementById(id);
            const area = document.getElementById(modeButtons[id]);
            
            if (btn) {
                btn.classList.remove('bg-gradient-to-r', 'from-cyan-500', 'to-blue-500', 'text-white');
                btn.classList.add('text-gray-300');
            }
            if (area) {
                area.classList.add('hidden');
            }
        });
        
        // Store current results before switching
        if (this.currentMode) {
            this.storeCurrentResults();
        }
        
        // Update current mode
        const previousMode = this.currentMode;
        this.currentMode = buttonId;
        
        // Activate selected mode
        const selectedBtn = document.getElementById(buttonId);
        const selectedArea = document.getElementById(modeButtons[buttonId]);
        
        if (selectedBtn) {
            selectedBtn.classList.add('bg-gradient-to-r', 'from-cyan-500', 'to-blue-500', 'text-white');
            selectedBtn.classList.remove('text-gray-300');
        }
        if (selectedArea) {
            selectedArea.classList.remove('hidden');
        }
        
        // Stop any ongoing recording/streaming when switching modes
        if (this.isRecording) {
            this.stopRecording();
        }
        if (this.isStreaming) {
            this.stopStreaming();
        }
        
        // Try to restore previous results for this mode
        if (this.lastResults[buttonId]) {
            this.restoreResultsForMode(buttonId);
        } else {
            // Clear results only if no stored results for this mode
            this.clearResults();
            // Show mode-specific instructions
            this.showModeInstructions(buttonId);
        }
        
        // Reset input fields for the new mode
        this.resetInputsForMode(buttonId);
    }
    
    // FILE UPLOAD HANDLING
    async handleFileUpload(file) {
        if (!file) return;
        
        console.time('FileUpload');
        
        // Store the mode when starting processing
        const processingMode = 'mode-upload';
        
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB');
            return;
        }
        
        const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg', 'audio/flac', 'audio/webm'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|m4a|mp4|ogg|flac|webm)$/i)) {
            this.showError('Please upload a valid audio file (MP3, WAV, M4A, MP4, OGG, FLAC, WebM)');
            return;
        }
        
        // Clear previous results immediately
        this.clearResults();
        
        this.showLoading(`Processing: ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`);
        
        const formData = new FormData();
        formData.append('file', file);
        
        const startTime = Date.now();
        
        try {
            // // console.log('Sending file to server...');
            const response = await fetch('/api/v1/transcribe', {
                method: 'POST',
                body: formData
            });
            
            // console.log(`Server responded in ${Date.now() - startTime}ms`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            // // console.log('Result received:', result);
            
            // Hide loading immediately before processing results
            this.hideLoading();
            
            // Add file information to results
            result.filename = file.name;
            result.filesize = (file.size/1024/1024).toFixed(1) + ' MB';
            result.sourceMode = processingMode;
            
            // Only display results if we're still in the same mode
            if (this.currentMode === processingMode) {
                this.displayResults(result);
                this.showUploadInstructions();
            } else {
                // Store results for when user returns to this tab
                this.lastResults[processingMode] = {
                    html: this.generateResultsHTML(result),
                    timestamp: Date.now()
                };
            }
            
            console.timeEnd('FileUpload');
            
        } catch (error) {
            console.error('Upload error:', error);
            this.hideLoading();
            if (this.currentMode === processingMode) {
                this.showError('Failed to process audio file: ' + error.message);
            }
        }
    }
    
    // YOUTUBE TRANSCRIPTION
    async handleYouTubeTranscribe() {
        const urlInput = document.getElementById('youtube-url-input');
        const url = urlInput?.value.trim();
        
        if (!url) {
            this.showError('Please enter a YouTube URL');
            return;
        }
        
        // Store the mode when starting processing
        const processingMode = 'mode-youtube';
        
        // Clear previous results when new URL is processed
        this.clearResults();
        
        // Validate YouTube URL
        const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|embed\/)|youtu\.be\/)[\w-]+/;
        if (!youtubeRegex.test(url)) {
            this.showError('Please enter a valid YouTube URL');
            return;
        }
        
        this.showLoading(`Downloading and processing YouTube video...\n${url}`);
        
        try {
            const response = await fetch('/api/v1/transcribe/youtube', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            // Add URL information to results
            result.source_url = url;
            result.sourceMode = processingMode;
            
            // Only display results if we're still in the same mode
            if (this.currentMode === processingMode) {
                this.displayResults(result);
                this.showYouTubeInstructions();
            } else {
                // Store results for when user returns to this tab
                this.lastResults[processingMode] = {
                    html: this.generateResultsHTML(result),
                    timestamp: Date.now()
                };
            }
        } catch (error) {
            if (this.currentMode === processingMode) {
                this.showError('Failed to process YouTube video: ' + error.message);
            }
        } finally {
            this.hideLoading();
        }
    }
    
    // LIVE RECORDING
    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            await this.stopRecording();
        }
    }
    
    async startRecording() {
        if (!this.checkBrowserSupport()) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: this.config.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            this.recordingStartTime = Date.now();
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                await this.processRecordedAudio(audioBlob);
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            
            // Update UI
            const recordBtn = document.getElementById('record-btn');
            if (recordBtn) {
                recordBtn.textContent = 'Stop Recording';
                recordBtn.classList.remove('from-red-500', 'to-pink-500');
                recordBtn.classList.add('from-gray-500', 'to-gray-600', 'animate-pulse');
            }
            
            // Show recording indicator
            this.showRecordingIndicator();
            
        } catch (error) {
            this.showError('Failed to access microphone: ' + error.message);
        }
    }
    
    async stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            const recordBtn = document.getElementById('record-btn');
            if (recordBtn) {
                recordBtn.textContent = 'Start Recording';
                recordBtn.classList.add('from-red-500', 'to-pink-500');
                recordBtn.classList.remove('from-gray-500', 'to-gray-600', 'animate-pulse');
            }
            
            this.hideRecordingIndicator();
        }
    }
    
    async processRecordedAudio(audioBlob) {
        const processingMode = 'mode-record';
        const duration = (Date.now() - this.recordingStartTime) / 1000;
        this.showLoading(`Processing recorded audio... (${duration.toFixed(1)}s)`);
        
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');
        
        try {
            const response = await fetch('/api/v1/transcribe', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            result.duration = duration.toFixed(1) + 's';
            result.recorded_at = new Date().toLocaleTimeString();
            result.source_type = 'Live Recording';
            result.sourceMode = processingMode;
            
            // Only display results if we're still in the same mode
            if (this.currentMode === processingMode) {
                this.displayResults(result);
                this.showRecordInstructions();
            } else {
                // Store results for when user returns to this tab
                this.lastResults[processingMode] = {
                    html: this.generateResultsHTML(result),
                    timestamp: Date.now()
                };
            }
        } catch (error) {
            if (this.currentMode === processingMode) {
                this.showError('Failed to process recording: ' + error.message);
            }
        } finally {
            this.hideLoading();
        }
    }
    
    // WEBSOCKET STREAMING
    async toggleStreaming() {
        if (!this.isStreaming) {
            await this.startStreaming();
        } else {
            await this.stopStreaming();
        }
    }
    
    async startStreaming() {
        if (!this.checkBrowserSupport()) return;
        
        try {
            // Show connecting status immediately
            this.updateConnectionStatus('connecting', 'Connecting to WebSocket...');
            
            // Create WebSocket connection
            const wsUrl = `${this.config.wsUrl}?sample_rate=${this.config.sampleRate}&language=${this.config.language}`;
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = async () => {
                // // console.log('🟢 WebSocket connected successfully');
                this.updateConnectionStatus('connecting', 'Connected! Configuring session...');
                
                try {
                    // Send configuration
                    const config = {
                        type: 'configure',
                        config: {
                            encoding: 'linear16',
                            sample_rate: this.config.sampleRate,
                            language: this.config.language,
                            interim_results: this.config.interimResults,
                            vad_enabled: this.config.vadEnabled,
                            speaker_diarization: this.config.speakerDiarization,
                            language_detection: this.config.languageDetection,
                            noise_reduction: this.config.noiseReduction,
                            max_speakers: 5,
                            custom_vocabulary: ['VoiceForge', 'WebSocket', 'AI', 'transcription']
                        }
                    };
                    
                    // // console.log('📤 Sending configuration:', config);
                    this.ws.send(JSON.stringify(config));
                    this.updateConnectionStatus('configuring', 'Configuration sent, waiting for confirmation...');
                    
                    // // console.log('🎤 Starting audio capture...');
                    
                    // Start audio capture
                    await this.startAudioCapture();
                    
                    // Update UI
                    const streamBtn = document.getElementById('stream-btn');
                    if (streamBtn) {
                        streamBtn.textContent = 'Disconnect';
                        streamBtn.classList.remove('from-green-500', 'to-teal-500');
                        streamBtn.classList.add('from-gray-500', 'to-gray-600');
                    }
                    
                    this.isStreaming = true;
                    this.showStreamingIndicator();
                    
                    // Start keep-alive ping
                    this.startKeepAlive();
                    
                    // console.log('✅ WebSocket streaming fully initialized');
                    
                } catch (error) {
                    console.error('❌ Failed to initialize WebSocket streaming:', error);
                    this.showError('Failed to initialize streaming: ' + error.message);
                    this.stopStreaming();
                }
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    // console.log('📨 WebSocket message received:', message.type, message);
                    this.handleStreamingMessage(message);
                } catch (error) {
                    console.error('❌ Failed to parse WebSocket message:', error, event.data);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.updateConnectionStatus('error', 'Connection failed! Check server status.');
                this.showError('WebSocket connection error. Please try again.');
                this.stopStreaming();
            };
            
            this.ws.onclose = (event) => {
                // console.log('🔴 WebSocket disconnected - Code:', event.code, 'Reason:', event.reason);
                if (event.code !== 1000) { // 1000 is normal closure
                    console.warn('⚠️ WebSocket closed unexpectedly');
                }
                this.stopStreaming();
            };
            
        } catch (error) {
            this.showError('Failed to start streaming: ' + error.message);
        }
    }
    
    async startAudioCapture() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: this.config.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false
                } 
            });
            
            // console.log('Audio stream obtained successfully');
            
            // Create audio context for processing
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.config.sampleRate
            });
            
            // Resume audio context if suspended (Chrome requirement)
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            const source = audioContext.createMediaStreamSource(stream);
            
            // Use newer AudioWorklet if available, fallback to ScriptProcessor
            if (audioContext.audioWorklet) {
                // console.log('Using AudioWorklet for processing');
                // For now, still use ScriptProcessor for compatibility
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                let audioPacketsSent = 0;
                processor.onaudioprocess = (e) => {
                    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.isStreaming) {
                        if (audioPacketsSent > 0) {
                            // console.log(`⚠️ Audio processing stopped. WS state: ${this.ws?.readyState}, Streaming: ${this.isStreaming}`);
                        }
                        return;
                    }
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    
                    // Check if audio data has meaningful content
                    const hasAudio = inputData.some(sample => Math.abs(sample) > 0.01);
                    const maxVolume = Math.max(...inputData.map(Math.abs));
                    
                    if (hasAudio) {
                        const pcm16 = this.convertFloat32ToPCM16(inputData);
                        const base64 = this.arrayBufferToBase64(pcm16);
                        
                        try {
                            this.ws.send(JSON.stringify({
                                type: 'audio',
                                data: base64,
                                timestamp: Date.now()
                            }));
                            audioPacketsSent++;
                            
                            if (audioPacketsSent % 10 === 0) { // Log every 10 packets
                                // console.log(`🎵 Sent ${audioPacketsSent} audio packets. Volume: ${maxVolume.toFixed(3)}`);
                            }
                        } catch (error) {
                            console.error('❌ Failed to send audio data:', error);
                        }
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                this.processor = processor;
            } else {
                // console.log('Using ScriptProcessor for processing');
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isStreaming) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const pcm16 = this.convertFloat32ToPCM16(inputData);
                        const base64 = this.arrayBufferToBase64(pcm16);
                        
                        this.ws.send(JSON.stringify({
                            type: 'audio',
                            data: base64
                        }));
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                this.processor = processor;
            }
            
            // Store for cleanup
            this.audioContext = audioContext;
            this.audioStream = stream;
            
            // console.log('Audio capture setup completed');
            
        } catch (error) {
            console.error('Failed to start audio capture:', error);
            this.showError('Failed to access microphone: ' + error.message);
            throw error;
        }
    }
    
    startKeepAlive() {
        // Clear any existing keep-alive
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
        }
        
        // Send ping every 30 seconds
        this.keepAliveInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                // console.log('📡 Sending keep-alive ping');
                try {
                    this.ws.send(JSON.stringify({ type: 'ping' }));
                } catch (error) {
                    console.error('❌ Failed to send keep-alive ping:', error);
                }
            }
        }, 30000);
    }
    
    stopStreaming() {
        // console.log('🛑 Stopping WebSocket streaming...');
        
        this.isStreaming = false;
        
        // Clear keep-alive
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
            this.keepAliveInterval = null;
        }
        
        if (this.ws) {
            this.ws.close(1000, 'User disconnected');
            this.ws = null;
        }
        
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close().catch(console.error);
            this.audioContext = null;
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => {
                track.stop();
                // console.log('🎤 Audio track stopped:', track.label);
            });
            this.audioStream = null;
        }
        
        // Update UI
        const streamBtn = document.getElementById('stream-btn');
        if (streamBtn) {
            streamBtn.textContent = 'Connect WebSocket';
            streamBtn.classList.add('from-green-500', 'to-teal-500');
            streamBtn.classList.remove('from-gray-500', 'to-gray-600');
        }
        
        this.hideStreamingIndicator();
        
        // console.log('✅ WebSocket streaming stopped');
    }
    
    updateConnectionStatus(status, message) {
        const streamBtn = document.getElementById('stream-btn');
        if (!streamBtn) return;
        
        // Update button text and color based on status
        switch(status) {
            case 'connecting':
                streamBtn.textContent = 'Connecting...';
                streamBtn.classList.remove('from-green-500', 'to-teal-500', 'from-gray-500', 'to-gray-600', 'from-red-500', 'to-red-600');
                streamBtn.classList.add('from-yellow-500', 'to-orange-500');
                break;
            case 'configuring':
                streamBtn.textContent = 'Configuring...';
                streamBtn.classList.remove('from-green-500', 'to-teal-500', 'from-gray-500', 'to-gray-600', 'from-red-500', 'to-red-600');
                streamBtn.classList.add('from-blue-500', 'to-blue-600');
                break;
            case 'connected':
                streamBtn.textContent = 'Connected ✓';
                streamBtn.classList.remove('from-yellow-500', 'to-orange-500', 'from-blue-500', 'to-blue-600', 'from-red-500', 'to-red-600');
                streamBtn.classList.add('from-green-500', 'to-teal-500');
                break;
            case 'ready':
                streamBtn.textContent = 'Disconnect';
                streamBtn.classList.remove('from-green-500', 'to-teal-500', 'from-yellow-500', 'to-orange-500', 'from-blue-500', 'to-blue-600');
                streamBtn.classList.add('from-gray-500', 'to-gray-600');
                break;
            case 'error':
                streamBtn.textContent = 'Connection Failed - Retry';
                streamBtn.classList.remove('from-green-500', 'to-teal-500', 'from-gray-500', 'to-gray-600', 'from-yellow-500', 'to-orange-500');
                streamBtn.classList.add('from-red-500', 'to-red-600');
                break;
            default:
                streamBtn.textContent = 'Connect WebSocket';
                streamBtn.classList.remove('from-gray-500', 'to-gray-600', 'from-yellow-500', 'to-orange-500', 'from-blue-500', 'to-blue-600', 'from-red-500', 'to-red-600');
                streamBtn.classList.add('from-green-500', 'to-teal-500');
        }
        
        // Show status message with nice animation
        this.showStatusMessage(message);
    }
    
    showStatusMessage(message) {
        // Remove any existing status message
        const existingStatus = document.querySelector('.connection-status-message');
        if (existingStatus) {
            existingStatus.remove();
        }
        
        // Create status message element
        const statusEl = document.createElement('div');
        statusEl.className = 'connection-status-message bg-gray-800 text-white px-4 py-2 rounded-lg mb-4 text-center animate-pulse';
        statusEl.textContent = message;
        
        // Insert after the stream button
        const streamBtn = document.getElementById('stream-btn');
        if (streamBtn && streamBtn.parentNode) {
            streamBtn.parentNode.insertBefore(statusEl, streamBtn.nextSibling);
            
            // Auto-remove after 3 seconds for non-persistent messages
            if (!message.includes('Ready to listen') && !message.includes('Listening')) {
                setTimeout(() => {
                    if (statusEl.parentNode) {
                        statusEl.remove();
                    }
                }, 3000);
            }
        }
    }
    
    handleStreamingMessage(message) {
        // console.log('📨 Handling streaming message:', message.type, message);
        
        if (message.type === 'connected' || message.type === 'connection') {
            this.sessionId = message.session_id;
            // console.log('🔗 Session established:', message.session_id);
            this.updateConnectionStatus('connected', `Connected! Session: ${message.session_id.substring(0, 8)}...`);
            this.showSuccess(`WebSocket Connected! Session: ${message.session_id.substring(0, 8)}...`);
        } else if (message.type === 'configured' || message.type === 'config_updated') {
            // console.log('⚙️ Configuration updated successfully');
            this.updateConnectionStatus('ready', '🎤 Ready to listen! Start speaking...');
            this.showStreamingIndicator();
        } else if (message.type === 'interim') {
            // console.log('💬 Interim result:', message.transcript);
            this.updateConnectionStatus('ready', '🎤 Listening... (processing interim)');
            this.displayStreamingResult(message);
        } else if (message.type === 'final') {
            // console.log('✅ Final result:', message.transcript);
            this.updateConnectionStatus('ready', '🎤 Ready to listen! Start speaking...');
            this.displayStreamingResult(message);
        } else if (message.type === 'transcription') {
            // Handle generic transcription messages
            // console.log('📝 Transcription result:', message.text || message.transcript);
            this.displayStreamingResult(message);
        } else if (message.type === 'pong') {
            // console.log('🏓 Keep-alive pong received');
        } else if (message.type === 'error') {
            console.error('❌ Server error:', message.error);
            this.showError('Server error: ' + (message.error?.message || message.message));
        } else {
            console.warn('❓ Unknown message type:', message.type, message);
        }
    }
    
    generateResultsHTML(result) {
        return `
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-semibold">Analysis Results</h3>
                ${this.renderSourceInfo(result)}
            </div>
            
            ${result.speakers ? this.renderSpeakerTimeline(result.speakers) : ''}
            
            ${result.language ? this.renderLanguageDetection(result.language, result.language_confidence) : ''}
            
            <div class="mb-6">
                <h4 class="text-lg font-semibold mb-3">Transcription</h4>
                <div class="bg-gray-900 rounded-lg p-6 font-mono text-sm space-y-2 max-h-96 overflow-y-auto">
                    ${this.formatTranscription(result)}
                </div>
            </div>
            
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                ${result.confidence ? `
                    <div class="glass-effect rounded-lg p-3">
                        <div class="text-xl font-bold text-purple-400">${(result.confidence * 100).toFixed(1)}%</div>
                        <div class="text-xs text-gray-400">Confidence</div>
                    </div>
                ` : ''}
                ${result.duration ? `
                    <div class="glass-effect rounded-lg p-3">
                        <div class="text-xl font-bold text-cyan-400">${result.duration}</div>
                        <div class="text-xs text-gray-400">Duration</div>
                    </div>
                ` : ''}
                ${result.word_count ? `
                    <div class="glass-effect rounded-lg p-3">
                        <div class="text-xl font-bold text-pink-400">${result.word_count}</div>
                        <div class="text-xs text-gray-400">Words</div>
                    </div>
                ` : ''}
                ${result.processing_time ? `
                    <div class="glass-effect rounded-lg p-3">
                        <div class="text-xl font-bold text-green-400">${result.processing_time}ms</div>
                        <div class="text-xs text-gray-400">Processing</div>
                    </div>
                ` : ''}
            </div>
            
            <div class="mt-6 flex justify-between items-center text-sm text-gray-500">
                <span>Processed: ${new Date().toLocaleString()}</span>
                <div class="flex gap-2">
                    <button onclick="window.voiceForgePlayground.copyToClipboard()" class="bg-blue-600 hover:bg-blue-500 px-3 py-1 rounded transition-all text-xs">
                        📋 Copy
                    </button>
                    <button onclick="window.voiceForgePlayground.clearCurrentResults()" class="bg-red-600 hover:bg-red-500 px-3 py-1 rounded transition-all text-xs">
                        ✖ Clear
                    </button>
                </div>
            </div>
        `;
    }
    
    // DISPLAY RESULTS
    displayResults(result) {
        const resultsDiv = document.getElementById('advanced-results');
        if (!resultsDiv) {
            console.error('Results div not found');
            return;
        }
        
        // Force immediate update
        requestAnimationFrame(() => {
            resultsDiv.innerHTML = this.generateResultsHTML(result);
            resultsDiv.classList.remove('hidden');
            
            // Log display completion
            // console.log('Results displayed for:', result.filename || result.source_url || 'streaming');
        });
    }
    
    displayStreamingResult(message) {
        const resultsDiv = document.getElementById('advanced-results');
        if (!resultsDiv) return;
        
        // Create or update streaming results container
        let streamingDiv = document.getElementById('streaming-results');
        if (!streamingDiv) {
            resultsDiv.classList.remove('hidden');
            resultsDiv.innerHTML = `
                <h3 class="text-xl font-semibold mb-4">Live Transcription</h3>
                <div id="streaming-results" class="bg-gray-900 rounded-lg p-6 font-mono text-sm space-y-2 min-h-[200px]"></div>
                <div id="streaming-stats" class="grid grid-cols-3 gap-4 text-center mt-4"></div>
            `;
            streamingDiv = document.getElementById('streaming-results');
        }
        
        // Get transcript text from various message formats
        const transcript = message.transcript || message.text || '';
        
        // Update transcription
        if (message.type === 'final' || message.type === 'transcription') {
            if (transcript) {
                const transcriptLine = document.createElement('div');
                transcriptLine.className = 'mb-2 text-white';
                
                if (message.speaker_id) {
                    transcriptLine.innerHTML = `<span class="text-purple-400">[${message.speaker_id}]:</span> ${transcript}`;
                } else {
                    transcriptLine.textContent = transcript;
                }
                
                streamingDiv.appendChild(transcriptLine);
                streamingDiv.scrollTop = streamingDiv.scrollHeight;
            }
        } else if (message.type === 'interim') {
            if (transcript) {
                // Show interim results in a different style
                let interimDiv = document.getElementById('interim-result');
                if (!interimDiv) {
                    interimDiv = document.createElement('div');
                    interimDiv.id = 'interim-result';
                    interimDiv.className = 'text-gray-500 italic border-l-2 border-gray-600 pl-2';
                    streamingDiv.appendChild(interimDiv);
                }
                interimDiv.textContent = '... ' + transcript;
                streamingDiv.scrollTop = streamingDiv.scrollHeight;
            }
        }
        
        // Update stats
        const statsDiv = document.getElementById('streaming-stats');
        if (statsDiv && message.language) {
            statsDiv.innerHTML = `
                <div class="glass-effect rounded-lg p-3">
                    <div class="text-xl font-bold text-purple-400">${message.language}</div>
                    <div class="text-xs text-gray-400">Language</div>
                </div>
                <div class="glass-effect rounded-lg p-3">
                    <div class="text-xl font-bold text-cyan-400">${(message.confidence * 100).toFixed(0)}%</div>
                    <div class="text-xs text-gray-400">Confidence</div>
                </div>
                <div class="glass-effect rounded-lg p-3">
                    <div class="text-xl font-bold text-green-400">Live</div>
                    <div class="text-xs text-gray-400">Status</div>
                </div>
            `;
        }
    }
    
    renderSourceInfo(result) {
        if (result.filename) {
            return `<div class="text-sm text-gray-400 bg-gray-800 px-3 py-1 rounded">
                📁 ${result.filename} ${result.filesize ? `(${result.filesize})` : ''}
            </div>`;
        } else if (result.source_url) {
            return `<div class="text-sm text-gray-400 bg-gray-800 px-3 py-1 rounded">
                🎬 YouTube
            </div>`;
        } else if (result.source_type) {
            return `<div class="text-sm text-gray-400 bg-gray-800 px-3 py-1 rounded">
                🎤 ${result.source_type} ${result.recorded_at ? `at ${result.recorded_at}` : ''}
            </div>`;
        } else if (this.isStreaming) {
            return `<div class="text-sm text-green-400 bg-gray-800 px-3 py-1 rounded animate-pulse">
                🔴 Live Stream
            </div>`;
        }
        return '';
    }

    // HELPER METHODS
    renderSpeakerTimeline(speakers) {
        if (!speakers || speakers.length === 0) return '';
        
        const colors = ['purple', 'pink', 'cyan', 'green', 'yellow'];
        
        return `
            <div class="mb-6">
                <h4 class="text-lg font-semibold mb-3">Speaker Timeline</h4>
                <div class="bg-gray-900 rounded-lg p-4">
                    <div class="space-y-2">
                        ${speakers.map((speaker, idx) => `
                            <div class="flex items-center">
                                <span class="text-sm text-${colors[idx % colors.length]}-400 w-24">Speaker ${idx + 1}:</span>
                                <div class="flex-1 h-4 bg-${colors[idx % colors.length]}-500/30 rounded"></div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    renderLanguageDetection(language, confidence) {
        const languageNames = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian'
        };
        
        return `
            <div class="mb-6">
                <h4 class="text-lg font-semibold mb-3">Detected Language</h4>
                <div class="flex space-x-4">
                    <div class="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg">
                        <span class="font-bold">${languageNames[language] || language}</span> 
                        ${confidence ? `(${(confidence * 100).toFixed(0)}%)` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    formatTranscription(result) {
        // Handle different response formats efficiently
        let text = '';
        
        // Check for transcription object (API response) - most common case
        if (result.transcription) {
            const trans = result.transcription;
            text = trans.text;
            
            // Batch update fields for better performance
            Object.assign(result, {
                confidence: trans.confidence || result.confidence,
                duration: trans.duration ? trans.duration + 's' : result.duration,
                word_count: trans.word_count || result.word_count,
                processing_time: trans.processing_time ? Math.round(trans.processing_time * 1000) : result.processing_time,
                language: trans.language || result.language,
                segments: trans.segments || result.segments
            });
        } else if (result.text) {
            text = result.text;
        } else if (result.transcript) {
            text = result.transcript;
        }
        
        // Format with segments if available
        if (result.segments && Array.isArray(result.segments)) {
            // Use array join for better performance with large segments
            const formatted = result.segments.map(segment => 
                segment.speaker_id 
                    ? `<p><span class="text-purple-400">[${segment.speaker_id}]:</span> ${segment.text}</p>`
                    : `<p>${segment.text}</p>`
            );
            return formatted.join('');
        } else if (text) {
            // Escape HTML for security but keep it simple
            const escaped = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<p>${escaped}</p>`;
        }
        
        return '<p class="text-gray-500">No transcription available</p>';
    }
    
    convertFloat32ToPCM16(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        let offset = 0;
        for (let i = 0; i < float32Array.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        return buffer;
    }
    
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    // STATE MANAGEMENT
    clearResults() {
        const resultsDiv = document.getElementById('advanced-results');
        if (resultsDiv) {
            resultsDiv.classList.add('hidden');
            resultsDiv.innerHTML = '';
        }
        
        // Clear any streaming-specific elements
        const streamingDiv = document.getElementById('streaming-results');
        if (streamingDiv) {
            streamingDiv.remove();
        }
        
        const statsDiv = document.getElementById('streaming-stats');
        if (statsDiv) {
            statsDiv.remove();
        }
        
        // Clear interim results
        const interimDiv = document.getElementById('interim-result');
        if (interimDiv) {
            interimDiv.remove();
        }
    }
    
    storeCurrentResults() {
        const resultsDiv = document.getElementById('advanced-results');
        if (resultsDiv && !resultsDiv.classList.contains('hidden')) {
            this.lastResults[this.currentMode] = {
                html: resultsDiv.innerHTML,
                timestamp: Date.now()
            };
        }
    }
    
    restoreResultsForMode(mode) {
        if (this.lastResults[mode]) {
            const resultsDiv = document.getElementById('advanced-results');
            if (resultsDiv) {
                resultsDiv.innerHTML = this.lastResults[mode].html;
                resultsDiv.classList.remove('hidden');
            }
        }
    }
    
    resetInputsForMode(buttonId) {
        // Clean up all existing mode-specific buttons first
        this.cleanupModeButtons();
        
        switch(buttonId) {
            case 'mode-upload':
                const fileInput = document.getElementById('advanced-file-input');
                if (fileInput) {
                    fileInput.value = '';
                }
                // Only show instructions if we have results
                if (this.lastResults[buttonId]) {
                    this.showUploadInstructions();
                }
                break;
            case 'mode-youtube':
                const urlInput = document.getElementById('youtube-url-input');
                if (urlInput) {
                    urlInput.value = '';
                    urlInput.placeholder = 'Paste new YouTube URL here...';
                }
                // Only show instructions if we have results
                if (this.lastResults[buttonId]) {
                    this.showYouTubeInstructions();
                }
                break;
            case 'mode-record':
                // Only show instructions if we have results
                if (this.lastResults[buttonId]) {
                    this.showRecordInstructions();
                }
                break;
            case 'mode-stream':
                // Only show instructions if we have results or streaming
                if (this.lastResults[buttonId] || this.isStreaming) {
                    this.showStreamInstructions();
                }
                break;
        }
    }
    
    cleanupModeButtons() {
        // Remove all mode-specific buttons to prevent duplicates
        const buttons = document.querySelectorAll('.new-upload-btn, .new-youtube-btn, .clear-record, .clear-stream');
        buttons.forEach(btn => btn.remove());
    }
    
    showModeInstructions(buttonId) {
        const instructionsMap = {
            'mode-upload': 'Upload an audio file to transcribe with advanced AI features',
            'mode-youtube': 'Enter a YouTube URL to extract and transcribe audio',
            'mode-record': 'Record audio directly from your microphone',
            'mode-stream': 'Connect WebSocket for real-time streaming transcription'
        };
        
        // Show instruction in results area temporarily
        const resultsDiv = document.getElementById('advanced-results');
        if (resultsDiv && instructionsMap[buttonId]) {
            resultsDiv.classList.remove('hidden');
            resultsDiv.innerHTML = `
                <div class="text-center py-8">
                    <div class="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center">
                        <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                        </svg>
                    </div>
                    <h3 class="text-xl font-semibold mb-2">Ready to transcribe</h3>
                    <p class="text-gray-400">${instructionsMap[buttonId]}</p>
                    <div class="mt-4 text-sm text-gray-500">
                        Features: Speaker Diarization • Language Detection • Noise Reduction
                    </div>
                </div>
            `;
        }
    }
    
    showUploadInstructions() {
        // Add a "new upload" button
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea && uploadArea.parentNode) {
            // Check if button already exists in parent
            let existingBtn = uploadArea.parentNode.querySelector('.new-upload-btn');
            if (!existingBtn) {
                const newUploadBtn = document.createElement('div');
                newUploadBtn.className = 'new-upload-btn mt-2 text-center';
                newUploadBtn.innerHTML = `
                    <button onclick="window.voiceForgePlayground.newUpload()" class="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg text-sm transition-all">
                        📁 Upload New File
                    </button>
                `;
                uploadArea.parentNode.appendChild(newUploadBtn);
            }
        }
    }
    
    showYouTubeInstructions() {
        // Add a "new URL" button
        const container = document.getElementById('youtube-area');
        if (container) {
            // Check if button already exists
            let existingBtn = container.querySelector('.new-youtube-btn');
            if (!existingBtn) {
                const newBtn = document.createElement('div');
                newBtn.className = 'new-youtube-btn mt-2 text-center';
                newBtn.innerHTML = `
                    <button onclick="window.voiceForgePlayground.newYouTube()" class="bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg text-sm transition-all">
                        🎬 Process New URL
                    </button>
                `;
                container.appendChild(newBtn);
            }
        }
    }
    
    showRecordInstructions() {
        // Add a "clear results" button to record area
        const container = document.getElementById('record-area');
        if (container) {
            // Check if button already exists
            let existingClear = container.querySelector('.clear-record');
            if (!existingClear) {
                const clearBtn = document.createElement('button');
                clearBtn.className = 'clear-record mt-2 bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg text-sm transition-all';
                clearBtn.innerHTML = '🗑 Clear Results';
                clearBtn.onclick = () => this.clearRecordArea();
                container.appendChild(clearBtn);
            }
        }
    }
    
    showStreamInstructions() {
        // Add a "clear session" button to stream area
        const container = document.getElementById('stream-area');
        if (container) {
            // Check if button already exists
            let existingClear = container.querySelector('.clear-stream');
            if (!existingClear) {
                const clearBtn = document.createElement('button');
                clearBtn.className = 'clear-stream mt-2 bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg text-sm transition-all';
                clearBtn.innerHTML = '🗑 Clear Session';
                clearBtn.onclick = () => this.clearStreamArea();
                container.appendChild(clearBtn);
            }
        }
    }
    
    newUpload() {
        const fileInput = document.getElementById('advanced-file-input');
        if (fileInput) {
            fileInput.value = '';
            fileInput.click();
        }
    }
    
    newYouTube() {
        const urlInput = document.getElementById('youtube-url-input');
        if (urlInput) {
            urlInput.value = '';
            urlInput.focus();
            urlInput.placeholder = 'Paste new YouTube URL here...';
        }
    }
    
    clearCurrentResults() {
        this.clearResults();
        delete this.lastResults[this.currentMode];
        this.showModeInstructions(this.currentMode);
    }
    
    copyToClipboard() {
        const transcriptionDiv = document.querySelector('#advanced-results .bg-gray-900');
        if (transcriptionDiv) {
            const text = transcriptionDiv.innerText;
            navigator.clipboard.writeText(text).then(() => {
                // Show brief success message
                const btn = event.target;
                const original = btn.innerHTML;
                btn.innerHTML = '✅ Copied!';
                setTimeout(() => {
                    btn.innerHTML = original;
                }, 1500);
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        }
    }
    
    clearUploadArea() {
        const fileInput = document.getElementById('advanced-file-input');
        if (fileInput) {
            fileInput.value = '';
        }
        this.clearCurrentResults();
        const btn = document.querySelector('.new-upload-btn');
        if (btn) {
            btn.remove();
        }
    }
    
    clearYouTubeArea() {
        const urlInput = document.getElementById('youtube-url-input');
        if (urlInput) {
            urlInput.value = '';
            urlInput.placeholder = 'Paste new YouTube URL here...';
        }
        this.clearCurrentResults();
        const btn = document.querySelector('.new-youtube-btn');
        if (btn) {
            btn.remove();
        }
    }
    
    clearRecordArea() {
        this.clearResults();
        delete this.lastResults[this.currentMode];
    }
    
    clearStreamArea() {
        if (this.isStreaming) {
            this.stopStreaming();
        }
        this.clearResults();
        delete this.lastResults[this.currentMode];
    }
    
    // UI HELPERS
    showLoading(message = 'Processing...') {
        const resultsDiv = document.getElementById('advanced-results');
        if (resultsDiv) {
            resultsDiv.classList.remove('hidden');
            resultsDiv.innerHTML = `
                <div class="text-center py-8">
                    <div class="animate-spin w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                    <h3 class="text-xl font-semibold mb-2">${message}</h3>
                    <p class="text-gray-400">This may take a few moments...</p>
                </div>
            `;
        }
    }
    
    hideLoading() {
        // Clear loading state immediately
        const resultsDiv = document.getElementById('advanced-results');
        if (resultsDiv && resultsDiv.innerHTML.includes('animate-spin')) {
            // Only clear if it's showing loading animation
            resultsDiv.classList.add('hidden');
            resultsDiv.innerHTML = '';
        }
    }
    
    showError(message) {
        const resultsDiv = document.getElementById('advanced-results');
        if (resultsDiv) {
            resultsDiv.classList.remove('hidden');
            resultsDiv.innerHTML = `
                <div class="bg-red-500/20 border border-red-500 rounded-lg p-4">
                    <h3 class="text-red-400 font-semibold mb-2">Error</h3>
                    <p class="text-gray-300">${message}</p>
                </div>
            `;
        }
    }
    
    showSuccess(message) {
        // console.log('Success:', message);
    }
    
    showRecordingIndicator() {
        const recordArea = document.getElementById('record-area');
        if (recordArea) {
            const indicator = document.createElement('div');
            indicator.id = 'recording-indicator';
            indicator.className = 'mt-4';
            indicator.innerHTML = `
                <div class="flex items-center justify-center space-x-2">
                    <div class="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                    <span class="text-red-400">Recording...</span>
                </div>
            `;
            recordArea.appendChild(indicator);
        }
    }
    
    hideRecordingIndicator() {
        const indicator = document.getElementById('recording-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    showStreamingIndicator() {
        const streamArea = document.getElementById('stream-area');
        if (streamArea) {
            const indicator = document.createElement('div');
            indicator.id = 'streaming-indicator';
            indicator.className = 'mt-4';
            indicator.innerHTML = `
                <div class="flex items-center justify-center space-x-2">
                    <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <span class="text-green-400">Streaming live...</span>
                </div>
            `;
            streamArea.appendChild(indicator);
        }
    }
    
    hideStreamingIndicator() {
        const indicator = document.getElementById('streaming-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
}

// Initialize playground when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.voiceForgePlayground = new VoiceForgePlayground();
    });
} else {
    window.voiceForgePlayground = new VoiceForgePlayground();
}