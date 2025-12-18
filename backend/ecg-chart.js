// ECG Chart Drawing Functions

function drawECG() {
    const canvas = document.getElementById('ecgCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    drawGrid(ctx, canvas.width, canvas.height);
    
    // Draw ECG signal
    drawSignal(ctx, canvas.width, canvas.height);
}

function drawGrid(ctx, width, height) {
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
    ctx.lineWidth = 0.5;
    
    // Vertical lines
    for (let x = 0; x < width; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y < height; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
}

function drawSignal(ctx, width, height) {
    ctx.strokeStyle = '#7c3aed';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#7c3aed';
    
    ctx.beginPath();
    
    for (let x = 0; x < width; x++) {
        // Generate ECG-like waveform
        let y = height / 2;
        
        // Baseline variation
        y += Math.sin(x * 0.02) * 10;
        
        // P wave
        if (x % 150 === 20) {
            y -= Math.sin((x % 150 - 20) * 0.3) * 15;
        }
        
        // QRS complex (R-peak)
        if (x % 150 === 50) {
            y -= 80;
        } else if (x % 150 === 48) {
            y += 20;
        } else if (x % 150 === 52) {
            y += 20;
        }
        
        // T wave
        if (x % 150 > 60 && x % 150 < 90) {
            y -= Math.sin((x % 150 - 60) * 0.2) * 25;
        }
        
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    ctx.shadowBlur = 0;
}

function drawECGWithResults(data) {
    const canvas = document.getElementById('ecgCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    drawGrid(ctx, canvas.width, canvas.height);
    
    // If we have actual data, use it; otherwise use simulated data
    if (data && data.length > 0) {
        drawActualECGData(ctx, canvas.width, canvas.height, data);
    } else {
        drawSignal(ctx, canvas.width, canvas.height);
        // Highlight abnormal regions
        highlightAbnormalRegions(ctx, canvas.width, canvas.height);
    }
}

function drawActualECGData(ctx, width, height, data) {
    if (!data || data.length === 0) return;
    
    ctx.strokeStyle = '#7c3aed';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#7c3aed';
    
    ctx.beginPath();
    
    const scaleX = width / data.length;
    const scaleY = height / 2;
    const offsetY = height / 2;
    
    data.forEach((value, index) => {
        const x = index * scaleX;
        const y = offsetY - (value * scaleY);
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    ctx.shadowBlur = 0;
}

function highlightAbnormalRegions(ctx, width, height) {
    // Highlight abnormal regions with red overlay
    ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
    
    // First abnormal region
    ctx.fillRect(width * 0.3, 0, width * 0.1, height);
    
    // Second abnormal region
    ctx.fillRect(width * 0.6, 0, width * 0.1, height);
    
    // Add pulse animation to abnormal regions
    const animate = () => {
        ctx.globalAlpha = 0.5 + Math.sin(Date.now() / 500) * 0.3;
        requestAnimationFrame(animate);
    };
}

// Resize handler
window.addEventListener('resize', () => {
    drawECG();
});