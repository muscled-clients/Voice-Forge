// Simple upload handler without complex playground object
function simpleUploadFile() {
    const fileInput = document.getElementById('file-input');
    const resultArea = document.getElementById('result-area');
    
    if (!fileInput || !fileInput.files[0]) {
        alert('Please select a file first!');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Show loading
    if (resultArea) {
        resultArea.innerHTML = `
            <div class="loading" style="text-align: center; padding: 24px; background: var(--surface); border-radius: 12px; border: 1px solid var(--border);">
                <div class="spinner" style="margin-bottom: 16px;"></div>
                <h4 style="color: var(--primary-color); margin-bottom: 12px;">🎧 Processing Audio...</h4>
                <p style="color: var(--text-secondary); margin-bottom: 16px;">Uploading ${file.name}...</p>
                <div style="background: var(--background); padding: 12px; border-radius: 8px; font-size: 0.875rem; color: var(--text-secondary);">
                    <div>⚡ Using Whisper Tiny model for fast processing</div>
                    <div>📊 Estimated time: 30-60 seconds</div>
                    <div>🔄 Please wait while we transcribe your audio...</div>
                </div>
            </div>
        `;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Upload file
    fetch('/api/v1/transcribe', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // console.log('Response:', data);
        
        if (data.status === 'success' && resultArea) {
            const transcription = data.transcription;
            resultArea.innerHTML = `
                <div class="transcription-result" style="padding: 24px; background: var(--background); border: 1px solid var(--success); border-radius: 12px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);">
                    <h4 style="color: var(--success); margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                        ✨ Transcription Complete
                    </h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; padding: 16px; background: var(--surface); border-radius: 8px;">
                        <div style="color: var(--text-secondary);">
                            <strong style="color: var(--text-primary);">📁 File:</strong><br>
                            <span>${transcription.file_name}</span>
                        </div>
                        <div style="color: var(--text-secondary);">
                            <strong style="color: var(--text-primary);">📏 Size:</strong><br>
                            <span>${(transcription.file_size / 1024).toFixed(2)} KB</span>
                        </div>
                        <div style="color: var(--text-secondary);">
                            <strong style="color: var(--text-primary);">🌍 Language:</strong><br>
                            <span>${transcription.language.toUpperCase()}</span>
                        </div>
                        <div style="color: var(--text-secondary);">
                            <strong style="color: var(--text-primary);">🎯 Confidence:</strong><br>
                            <span style="color: var(--success);">${(transcription.confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <div style="margin-top: 20px; padding: 15px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;">
                        <h5 style="color: var(--primary-color); margin-bottom: 10px;">📝 Transcribed Text:</h5>
                        <p style="margin-top: 10px; color: var(--text-primary); font-size: 16px; line-height: 1.6; font-family: inherit;">${transcription.text}</p>
                    </div>
                    <div style="margin-top: 20px; text-align: center;">
                        <button onclick="location.reload()" class="btn btn-primary" style="display: inline-flex; align-items: center; gap: 8px;">
                            🔄 Try Another File
                        </button>
                    </div>
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        if (resultArea) {
            resultArea.innerHTML = `
                <div style="padding: 24px; background: var(--background); border: 1px solid var(--error); border-radius: 12px; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);">
                    <h4 style="color: var(--error); margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">❌ Upload Error</h4>
                    <p style="color: var(--text-secondary); margin-bottom: 20px;">${error.message}</p>
                    <div style="text-align: center;">
                        <button onclick="location.reload()" class="btn btn-primary" style="display: inline-flex; align-items: center; gap: 8px;">
                            🔄 Try Again
                        </button>
                    </div>
                </div>
            `;
        }
    });
}

// Make function globally available
window.simpleUploadFile = simpleUploadFile;