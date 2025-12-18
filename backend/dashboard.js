// Check if user is authenticated
if (!api.isAuthenticated()) {
    window.location.href = 'login.html';
}

// Display username
document.getElementById('usernameDisplay').textContent = api.getCurrentUser();

// Global variables
let selectedFile = null;
let currentAnalysisId = null;

// Initialize ECG chart on page load
window.addEventListener('load', () => {
    drawECG();
});

// Upload area click handler
document.getElementById('uploadArea').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

// File input change handler
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (e.target.files[0]) {
        selectedFile = e.target.files[0];
        document.getElementById('fileName').textContent = selectedFile.name;
        document.getElementById('fileInfo').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
    }
});

// Analyze button click handler
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Get patient data
    const patientData = {
        id: document.getElementById('patientId').value,
        age: document.getElementById('patientAge').value,
        gender: document.getElementById('patientGender').value
    };
    
    // Show processing steps
    document.getElementById('processingSteps').style.display = 'block';
    document.getElementById('analyzeBtn').disabled = true;
    
    try {
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 2;
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = progress;
            
            if (progress >= 90) {
                clearInterval(progressInterval);
            }
        }, 50);
        
        // Call API to analyze ECG
        const result = await api.analyzeECG(selectedFile, patientData);
        
        // Complete progress
        clearInterval(progressInterval);
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = '100';
        
        // Store analysis ID
        currentAnalysisId = result.analysis_id;
        
        // Display results
        setTimeout(() => {
            document.getElementById('processingSteps').style.display = 'none';
            displayResults(result);
            document.getElementById('analyzeBtn').disabled = false;
        }, 500);
        
    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed. Please try again.');
        document.getElementById('processingSteps').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
});

// Display results function
function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update diagnosis
    document.getElementById('diagnosisTitle').textContent = data.diagnosis || 'Ventricular Arrhythmia Detected';
    document.getElementById('confidenceValue').textContent = (data.confidence || 91) + '%';
    document.getElementById('confidenceFill').style.width = (data.confidence || 91) + '%';
    document.getElementById('diagnosisDescription').textContent = data.description || 'Irregular ventricular activity detected. Medical review recommended.';
    
    // Update statistics
    document.getElementById('normalBeats').textContent = (data.normal_beats || 78) + '%';
    document.getElementById('abnormalBeats').textContent = (data.abnormal_beats || 22) + '%';
    document.getElementById('heartRate').textContent = data.heart_rate || 82;
    
    // Display detected arrhythmias
    const eventList = document.getElementById('eventList');
    eventList.innerHTML = '';
    
    const events = data.events || [
        { name: 'Premature Ventricular Contractions (PVC)', count: 15, type: 'red' },
        { name: 'Ventricular Tachycardia', count: 3, type: 'orange' }
    ];
    
    events.forEach(event => {
        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';
        eventItem.innerHTML = `
            <span>${event.name}</span>
            <span class="event-badge ${event.type}">${event.count} events</span>
        `;
        eventList.appendChild(eventItem);
    });
    
    // Display timeline
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    const timelineEvents = data.timeline || [
        { time: '00:12.4s', type: 'PVC Detected', description: 'Premature ventricular contraction', color: 'linear-gradient(135deg, #ef4444, #ec4899)' },
        { time: '00:24.8s', type: 'PVC Detected', description: 'Premature ventricular contraction', color: 'linear-gradient(135deg, #ef4444, #ec4899)' },
        { time: '00:45.2s', type: 'V-Tach Episode', description: 'Ventricular tachycardia lasting 2.3s', color: 'linear-gradient(135deg, #f97316, #ef4444)' }
    ];
    
    timelineEvents.forEach((event, index) => {
        const timelineItem = document.createElement('div');
        timelineItem.className = 'timeline-item';
        timelineItem.innerHTML = `
            <div class="timeline-dot" style="background: ${event.color};"></div>
            <div>
                <p style="color: #cbd5e1; font-weight: 600;">${event.time} - ${event.type}</p>
                <p style="color: #64748b; font-size: 14px; margin-top: 5px;">${event.description}</p>
            </div>
        `;
        timeline.appendChild(timelineItem);
    });
    
    // Update ECG chart with results
    drawECGWithResults(data.ecg_data);
}

// Logout button handler
document.getElementById('logoutBtn').addEventListener('click', async () => {
    await api.logout();
    window.location.href = 'login.html';
});

// Download report button handler
document.getElementById('downloadBtn').addEventListener('click', async () => {
    if (!currentAnalysisId) {
        alert('No analysis available to download');
        return;
    }
    
    try {
        await api.downloadReport(currentAnalysisId);
    } catch (error) {
        alert('Failed to download report');
    }
});

// Save analysis button handler
document.getElementById('saveBtn').addEventListener('click', async () => {
    if (!currentAnalysisId) {
        alert('No analysis available to save');
        return;
    }
    
    const patientData = {
        id: document.getElementById('patientId').value,
        age: document.getElementById('patientAge').value,
        gender: document.getElementById('patientGender').value
    };
    
    try {
        await api.saveAnalysis(currentAnalysisId, patientData);
        alert('Analysis saved successfully!');
    } catch (error) {
        alert('Failed to save analysis');
    }
});