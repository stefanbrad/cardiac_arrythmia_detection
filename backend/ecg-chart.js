// Get canvas elements
const ecgCanvas = document.getElementById('ecgCanvas');
const ctx = ecgCanvas.getContext('2d');

// Set canvas dimensions
function resizeCanvas() {
    const parent = ecgCanvas.parentElement;
    ecgCanvas.width = parent.offsetWidth;
    ecgCanvas.height = parent.offsetHeight;
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ECG plotting Configuration
const cfg = {
    gridColor: '#1e293b',
    gridThickColor: '#334155',
    signalColor: '#22d3ee', // Cyan strălucitor pentru normal
    anomalyColor: '#ef4444', // Roșu aprins pentru anomalii
    lineWidth: 2.5,          // Linie mai groasă
    anomalyLineWidth: 3.5,   // Linie și mai groasă pentru anomalii
    baseline: ecgCanvas.height / 2,
    amplitude: 150, // Mărim amplitudinea să fie mai vizibil
    glowAmount: 10  // Efect de strălucire
};

// Draw grid function (Rămâne la fel, dar cu culori mai bune)
function drawGrid() {
    ctx.clearRect(0, 0, ecgCanvas.width, ecgCanvas.height);
    ctx.beginPath();
    ctx.strokeStyle = cfg.gridColor;
    ctx.lineWidth = 0.5;

    const gridSize = 20;
    for (let x = 0; x < ecgCanvas.width; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, ecgCanvas.height);
    }
    for (let y = 0; y < ecgCanvas.height; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(ecgCanvas.width, y);
    }
    ctx.stroke();

    // Thicker lines every 5 boxes
    ctx.beginPath();
    ctx.strokeStyle = cfg.gridThickColor;
    ctx.lineWidth = 1;
    for (let x = 0; x < ecgCanvas.width; x += gridSize * 5) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, ecgCanvas.height);
    }
    for (let y = 0; y < ecgCanvas.height; y += gridSize * 5) {
        ctx.moveTo(0, y);
        ctx.lineTo(ecgCanvas.width, y);
    }
    ctx.stroke();
}

// --- FUNCȚIA NOUĂ DE DESENARE ---
function drawECGWithResults(fullData) {
    drawGrid();

    const signalData = fullData.ecg_data;
    const anomalies = fullData.timeline || [];

    if (!signalData || signalData.length === 0) return;

    // Calculăm scara orizontală
    const stepX = ecgCanvas.width / signalData.length;

    // Helper pentru a obține coordonata Y
    const getY = (val) => cfg.baseline - (val * cfg.amplitude);

    // 1. Desenăm SEMNALUL NORMAL (Baza albastră)
    ctx.beginPath();
    ctx.strokeStyle = cfg.signalColor;
    ctx.lineWidth = cfg.lineWidth;
    ctx.lineJoin = 'round';
    // Adăugăm strălucire (Glow effect)
    ctx.shadowColor = cfg.signalColor;
    ctx.shadowBlur = cfg.glowAmount;

    ctx.moveTo(0, getY(signalData[0]));
    for (let i = 1; i < signalData.length; i++) {
        ctx.lineTo(i * stepX, getY(signalData[i]));
    }
    ctx.stroke();

    // 2. Desenăm ANOMALIILE (Suprapunere Roșie)
    if (anomalies.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = cfg.anomalyColor;
        ctx.lineWidth = cfg.anomalyLineWidth;
        // Strălucire roșie mai intensă
        ctx.shadowColor = cfg.anomalyColor;
        ctx.shadowBlur = cfg.glowAmount + 5;

        anomalies.forEach(event => {
            // Verificăm dacă avem indecșii trimiși de backend
            if (event.start_index !== undefined && event.end_index !== undefined) {
                
                // Mutăm creionul la începutul anomaliei
                ctx.moveTo(event.start_index * stepX, getY(signalData[event.start_index]));
                
                // Desenăm segmentul roșu
                for (let i = event.start_index + 1; i <= event.end_index && i < signalData.length; i++) {
                    ctx.lineTo(i * stepX, getY(signalData[i]));
                }
            }
        });
        ctx.stroke();
    }

    // Reset shadow pentru alte desene
    ctx.shadowBlur = 0;
}

// Initial draw (placeholder)
drawGrid();