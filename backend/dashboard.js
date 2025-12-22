if (!api.isAuthenticated()) {
    window.location.href = 'login.html';
}

document.getElementById('usernameDisplay').textContent = api.getCurrentUser();

window.addEventListener('load', () => {
    drawECG();
});

document.getElementById('analyzeBtn').addEventListener('click', async () => {
    // 1. luam ID-ul din campul text
    const recordId = document.getElementById('recordId').value;
    
    // verif daca s a scris ceva
    if (!recordId) {
        alert('Please enter a Record ID (e.g., 100)');
        return;
    }
    
    // get data
    const patientData = {
        id: document.getElementById('patientId').value,
        age: document.getElementById('patientAge').value,
        gender: document.getElementById('patientGender').value
    };
    
    // processing steps
    document.getElementById('processingSteps').style.display = 'block';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('resultsSection').style.display = 'none'; // ascundem rezultatele vechi
    
    try {
        // simulate progress - animatie 
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress > 90) progress = 90; // nu depaseste 90% pana nu e gata
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = progress;
        }, 100);
        
        // 2. apelam API-ul cu ID-ul inregistrarii (nu cu fisier)
        console.log("Sending request for record:", recordId);
        const result = await api.analyzeECG(recordId, patientData);
        
        clearInterval(progressInterval);
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = '100';
        
        // ddisplay results
        setTimeout(() => {
            document.getElementById('processingSteps').style.display = 'none';
            displayResults(result);
            document.getElementById('analyzeBtn').disabled = false;
        }, 500);
        
    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed: ' + error.message);
        document.getElementById('processingSteps').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
});

// display results function
function displayResults(data) {
    document.getElementById('resultsSection').style.display = 'block';
    
    // --- START: LOGICA PENTRU CULORI SI STILURI DINAMICE ---
    const alertBox = document.querySelector('.alert-box');
    const alertIcon = document.querySelector('.alert-header span'); // Iconița (Emoji)
    const diagnosisTitle = document.getElementById('diagnosisTitle');
    const confidenceFill = document.getElementById('confidenceFill');

    // 1. reset clasele vechi ca sa nu se suprapuna
    alertBox.classList.remove('safe', 'warning');
    diagnosisTitle.classList.remove('text-safe', 'text-warning', 'text-danger');

    // 2. aplicam stilurile in funcție de ce a zis AI-ul
    if (data.diagnosis.includes("Normal") || data.abnormal_beats === 0) {
        // CAZUL VERDE (sanatos)
        alertBox.classList.add('safe');
        diagnosisTitle.classList.add('text-safe');
        if(alertIcon) alertIcon.textContent = '✅';
        confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)'; 
    } 
    else if (data.diagnosis.includes("Block") || data.diagnosis.includes("LBBB") || data.diagnosis.includes("RBBB")) {
        // CAZUL PORTOCALIU (Avertisment - LBBB, RBBB)
        alertBox.classList.add('warning');
        diagnosisTitle.classList.add('text-warning');
        if(alertIcon) alertIcon.textContent = '⚠️';
        confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)'; 
    } 
    else {
        // CAZUL ROSU (Pericol - Aritmii Ventriculare, PVC, APB)
        diagnosisTitle.classList.add('text-danger');
        if(alertIcon) alertIcon.textContent = '❌';
        confidenceFill.style.background = 'linear-gradient(90deg, #ef4444, #ec4899)'; 
    }
    // --- END: LOGICA CULORI ---

    // update diagnosis text and confidence
    diagnosisTitle.textContent = data.diagnosis;
    document.getElementById('confidenceValue').textContent = (data.confidence) + '%';
    confidenceFill.style.width = (data.confidence) + '%';
    document.getElementById('diagnosisDescription').textContent = data.description;
    
    // update statistics
    document.getElementById('normalBeats').textContent = (data.normal_beats) + '%';
    document.getElementById('abnormalBeats').textContent = (data.abnormal_beats) + '%';
    document.getElementById('heartRate').textContent = data.heart_rate;
    
    // display detected arrhythmias 
    const eventList = document.getElementById('eventList');
    eventList.innerHTML = '';
    
    if (data.events && data.events.length > 0) {
        data.events.forEach(event => {
            const eventItem = document.createElement('div');
            eventItem.className = 'event-item';
            eventItem.innerHTML = `
                <span>${event.name}</span>
                <span class="event-badge ${event.type}">${event.count} events</span>
            `;
            eventList.appendChild(eventItem);
        });
    } else {
        eventList.innerHTML = '<div style="padding:10px; color:#a78bfa;">No arrhythmias detected.</div>';
    }
    
    // display timeline
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    if (data.timeline && data.timeline.length > 0) {
        data.timeline.forEach(event => {
            const timelineItem = document.createElement('div');
            timelineItem.className = 'timeline-item';
            
            // setam culoarea bulinei din timeline in functie de tipul evenimentului
            let dotColor = event.color;
            if(!dotColor) {
                 // fallback logic daca backend nu trimite culoare
                 if(event.type.includes('N')) dotColor = '#10b981';
                 else if(event.type.includes('L') || event.type.includes('R')) dotColor = '#f59e0b';
                 else dotColor = 'linear-gradient(135deg, #ef4444, #ec4899)';
            }

            timelineItem.innerHTML = `
                <div class="timeline-dot" style="background: ${dotColor};"></div>
                <div>
                    <p style="color: #cbd5e1; font-weight: 600;">${event.time} - ${event.type}</p>
                    <p style="color: #64748b; font-size: 14px; margin-top: 5px;">${event.description}</p>
                </div>
            `;
            timeline.appendChild(timelineItem);
        });
    } else {
        timeline.innerHTML = '<p style="color: #64748b;">No abnormal events recorded on timeline.</p>';
    }
    
    // update ECG chart with results
    if (typeof drawECGWithResults === 'function') {
        // trimitem TOT obiectul 'data' ca sa am acces si la timeline
        drawECGWithResults(data); 
    }
}

// logout button handler
document.getElementById('logoutBtn').addEventListener('click', async () => {
    await api.logout();
    window.location.href = 'login.html';
});