// VoiceForge Interactive Playground
class VoiceForgePlayground {
    constructor() {
        this.apiBase = window.location.origin;
        this.initializeElements();
        this.setupEventListeners();
        this.loadStats();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.resultArea = document.getElementById('result-area');
        this.uploadBtn = document.getElementById('upload-btn');
        this.statsContainer = document.getElementById('stats-container');
    }

    setupEventListeners() {
        // File upload events
        if (this.uploadArea && this.fileInput) {
            this.uploadArea.addEventListener('click', () => this.fileInput.click());
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
            
            // Drag and drop
            this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
            this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        }
        
        // Upload button
        if (this.uploadBtn) {
            this.uploadBtn.addEventListener('click', () => this.uploadFile());
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.fileInput.files = files;
            this.displayFileInfo(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.displayFileInfo(file);
        }
    }

    displayFileInfo(file) {
        const fileSize = (file.size / 1024 / 1024).toFixed(2);
        const supportedFormats = ['mp3', 'wav', 'm4a', 'mp4', 'ogg', 'flac'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const isSupported = supportedFormats.includes(fileExtension);

        if (!this.resultArea) return;

        this.resultArea.innerHTML = `
            <div class="file-info">
                <h4>📁 Selected File</h4>
                <div class="file-details">
                    <div><strong>Name:</strong> ${file.name}</div>
                    <div><strong>Size:</strong> ${fileSize} MB</div>
                    <div><strong>Type:</strong> ${file.type || 'Unknown'}</div>
                    <div><strong>Status:</strong> 
                        <span class="${isSupported ? 'text-success' : 'text-error'}">
                            ${isSupported ? '✅ Supported' : '❌ Unsupported format'}
                        </span>
                    </div>
                </div>
                ${isSupported ? '<button class="btn btn-primary" id="transcribe-btn">🎤 Transcribe Audio</button>' : ''}
            </div>
        `;
        
        // Add event listener to the new button
        if (isSupported) {
            const transcribeBtn = document.getElementById('transcribe-btn');
            if (transcribeBtn) {
                transcribeBtn.addEventListener('click', () => this.uploadFile());
            }
        }
    }

    async uploadFile() {
        const file = this.fileInput.files[0];
        if (!file) {
            this.showError('Please select a file first');
            return;
        }

        this.showLoading();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${this.apiBase}/api/v1/transcribe`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.showResult(result);
            } else {
                this.showError(result.detail || 'Transcription failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }

    showLoading() {
        if (!this.resultArea) return;
        this.resultArea.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <h4>🎧 Processing Audio...</h4>
                <p>This may take a few moments depending on file size</p>
            </div>
        `;
    }

    showResult(result) {
        if (!this.resultArea) return;
        const transcription = result.transcription;
        const confidence = (transcription.confidence * 100).toFixed(1);
        
        this.resultArea.innerHTML = `
            <div class="transcription-result">
                <h4>✨ Transcription Complete</h4>
                <div class="result-card">
                    <div class="result-text">
                        <strong>Text:</strong>
                        <div class="transcript">${transcription.text}</div>
                    </div>
                    <div class="result-meta">
                        <div class="meta-grid">
                            <div><strong>Language:</strong> ${transcription.language}</div>
                            <div><strong>Confidence:</strong> ${confidence}%</div>
                            <div><strong>Duration:</strong> ${transcription.duration?.toFixed(2) || 'N/A'}s</div>
                            <div><strong>File Size:</strong> ${(transcription.file_size / 1024).toFixed(2)} KB</div>
                        </div>
                    </div>
                    <div class="result-actions">
                        <button class="btn btn-secondary" onclick="playground.copyText('${transcription.text}')">📋 Copy Text</button>
                        <button class="btn btn-secondary" onclick="playground.downloadResult()">💾 Download</button>
                    </div>
                </div>
            </div>
        `;
        
        this.loadStats(); // Refresh stats after transcription
    }

    showError(message) {
        if (!this.resultArea) return;
        this.resultArea.innerHTML = `
            <div class="error-message">
                <h4>❌ Error</h4>
                <p>${message}</p>
                <button class="btn btn-secondary" id="try-again-btn">Try Again</button>
            </div>
        `;
        const tryAgainBtn = document.getElementById('try-again-btn');
        if (tryAgainBtn) {
            tryAgainBtn.addEventListener('click', () => this.clearResult());
        }
    }

    copyText(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Text copied to clipboard!');
        });
    }

    downloadResult() {
        const transcript = document.querySelector('.transcript');
        if (transcript) {
            const blob = new Blob([transcript.textContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transcription.txt';
            a.click();
            URL.revokeObjectURL(url);
        }
    }

    clearResult() {
        if (this.resultArea) {
            this.resultArea.innerHTML = `
                <div class="placeholder">
                    <p>📄 Transcription results will appear here</p>
                </div>
            `;
        }
        if (this.fileInput) {
            this.fileInput.value = '';
        }
    }

    async loadStats() {
        if (!this.statsContainer) return;
        
        try {
            const response = await fetch(`${this.apiBase}/api/v1/stats`);
            const stats = await response.json();
            
            this.statsContainer.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_transcriptions}</div>
                    <div class="stat-label">Total Transcriptions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.active_sessions}</div>
                    <div class="stat-label">Active Sessions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.supported_languages}</div>
                    <div class="stat-label">Languages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.model_loaded ? '✅' : '⚠️'}</div>
                    <div class="stat-label">Model Status</div>
                </div>
            `;
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize playground when DOM is loaded
let playground = null;
document.addEventListener('DOMContentLoaded', () => {
    playground = new VoiceForgePlayground();
    window.playground = playground; // Make it globally accessible
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .file-info, .transcription-result, .error-message, .loading {
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .transcript {
        background: var(--background);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
        font-family: 'Fira Code', monospace;
        line-height: 1.6;
    }
    
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .result-actions {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .text-success { color: var(--success); }
    .text-error { color: var(--error); }
    
    .placeholder {
        text-align: center;
        color: var(--text-secondary);
        padding: 2rem;
    }
`;
document.head.appendChild(style);