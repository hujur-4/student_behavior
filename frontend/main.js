// DOM Elements
const videoElement = document.getElementById('video-element');
const canvasOverlay = document.getElementById('canvas-overlay');
const ctx = canvasOverlay.getContext('2d');
const btnWebcam = document.getElementById('btn-webcam');
const fileUpload = document.getElementById('file-upload');
const videoPlaceholder = document.getElementById('video-placeholder');
const connectionDot = document.getElementById('connection-dot');
const connectionStatus = document.getElementById('connection-status');
const facesCounter = document.getElementById('faces-counter');
const idCardsCounter = document.getElementById('id-cards-counter');
const fpsCounter = document.getElementById('fps-counter');

// State
let isWebcamActive = false;
let stream = null;
let ws = null;
let isConnected = false;
let animationId = null;
let lastFrameTime = 0;
let fps = 0;
let isWaitingForResponse = false; // Backpressure lock to prevent frame queuing lag

// Connect to Backend WebSocket
function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8000/ws/detect');
    
    ws.onopen = () => {
        isConnected = true;
        connectionDot.classList.add('connected');
        connectionStatus.textContent = 'Connected to YOLOv8';
    };
    
    ws.onclose = () => {
        isConnected = false;
        connectionDot.classList.remove('connected');
        connectionStatus.textContent = 'Disconnected';
        isWaitingForResponse = false;
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
    };
    
    ws.onmessage = (event) => {
        isWaitingForResponse = false; // Reset lock when response is received
        const data = JSON.parse(event.data);
        if (data.boxes) {
            drawBoxes(data.boxes);
        }
        if (data.counts) {
            if (facesCounter) facesCounter.textContent = data.counts.faces;
            if (idCardsCounter) idCardsCounter.textContent = data.counts.id_cards;
        }
    };
}

// Initial connection
connectWebSocket();

// Handle Resize for Canvas
function resizeCanvas() {
    if (!videoElement.videoWidth || !videoElement.videoHeight) return;
    canvasOverlay.width = videoElement.clientWidth;
    canvasOverlay.height = videoElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
videoElement.addEventListener('loadedmetadata', resizeCanvas);

// Color mapping helper
function getColorForBox(box) {
    if (box.type === 'face' || box.label === 'FACE') {
        return '#3b82f6'; // Bright Blue for Faces
    }
    if (box.label === 'with_id_strap') {
        return '#10b981'; // Emerald Green for ID Card with strap
    }
    if (box.label === 'without_id_strap') {
        return '#f59e0b'; // Amber Orange for ID Card without strap
    }
    return '#8b5cf6'; // Violet Purple fallback
}

// Draw Bounding Boxes
function drawBoxes(boxes) {
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    
    if (!videoElement.videoWidth || !videoElement.videoHeight) return;

    // Scale coordinates from intrinsic video dimensions to canvas size
    const scaleX = canvasOverlay.width / videoElement.videoWidth;
    const scaleY = canvasOverlay.height / videoElement.videoHeight;
    
    boxes.forEach(box => {
        const x = box.x1 * scaleX;
        const y = box.y1 * scaleY;
        const w = (box.x2 - box.x1) * scaleX;
        const h = (box.y2 - box.y1) * scaleY;
        
        const boxColor = getColorForBox(box);

        // Draw bounding box
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        
        // Draw label background badge
        ctx.fillStyle = boxColor;
        const labelText = `${box.label || 'Detected'} ${(box.confidence * 100).toFixed(0)}%`;
        ctx.font = '600 13px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeHeight = 22;
        const badgeY = Math.max(0, y - badgeHeight);
        ctx.fillRect(x, badgeY, textWidth + 12, badgeHeight);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + 6, badgeY + 15);
    });
}

// Send current frame over WebSocket with backpressure control
function sendCurrentFrame() {
    if (!isConnected || ws.readyState !== WebSocket.OPEN) return;
    if (isWaitingForResponse) return; // Skip if backend is still processing previous frame
    if (!videoElement.videoWidth || !videoElement.videoHeight) return;

    try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoElement.videoWidth;
        tempCanvas.height = videoElement.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
        
        const base64Image = tempCanvas.toDataURL('image/jpeg', 0.6);
        isWaitingForResponse = true;
        ws.send(base64Image);
    } catch (e) {
        console.error("Error capturing video frame:", e);
    }
}

// Continuous video processing loop across frames
function processVideoFrame(timestamp) {
    if (!videoElement.paused && !videoElement.ended) {
        if (lastFrameTime > 0) {
            const delta = timestamp - lastFrameTime;
            fps = Math.round(1000 / delta);
            if (timestamp % 10 < 2 && fpsCounter) fpsCounter.textContent = fps;
        }
        lastFrameTime = timestamp;

        sendCurrentFrame();
    }
    
    animationId = requestAnimationFrame(processVideoFrame);
}

// Toggle Webcam
btnWebcam.addEventListener('click', async () => {
    if (isWebcamActive) {
        // Stop webcam
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        videoElement.srcObject = null;
        videoElement.removeAttribute('src');
        videoElement.controls = false;
        videoElement.style.opacity = '0';
        videoPlaceholder.style.display = 'flex';
        btnWebcam.innerHTML = '<span class="btn-icon">📷</span> Live Camera';
        btnWebcam.classList.remove('active');
        isWebcamActive = false;
        
        if (animationId) cancelAnimationFrame(animationId);
        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
        if (facesCounter) facesCounter.textContent = '0';
        if (idCardsCounter) idCardsCounter.textContent = '0';
        if (fpsCounter) fpsCounter.textContent = '0';
    } else {
        // Start webcam
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            videoElement.controls = false;
            videoElement.style.opacity = '1';
            videoPlaceholder.style.display = 'none';
            btnWebcam.innerHTML = '<span class="btn-icon">🛑</span> Stop Camera';
            btnWebcam.classList.add('active');
            isWebcamActive = true;
            fileUpload.value = '';
            
            videoElement.onloadedmetadata = () => {
                resizeCanvas();
                videoElement.play().catch(e => console.log(e));
            };
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam.");
        }
    }
});

// Handle Video Upload
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        if (isWebcamActive) {
            btnWebcam.click();
        }
        
        const url = URL.createObjectURL(file);
        videoElement.srcObject = null;
        videoElement.src = url;
        videoElement.controls = true; // Show video controls so user can play, pause, seek
        videoElement.loop = true;     // Loop video continuously
        videoElement.style.opacity = '1';
        videoPlaceholder.style.display = 'none';
        
        videoElement.onloadedmetadata = () => {
            resizeCanvas();
            videoElement.play().catch(err => console.log("Autoplay notification:", err));
        };
    }
});

// Event Listeners for Video Playback Events (Supports all timeline frame detection)
videoElement.addEventListener('play', () => {
    resizeCanvas();
    lastFrameTime = 0;
    isWaitingForResponse = false;
    if (animationId) cancelAnimationFrame(animationId);
    animationId = requestAnimationFrame(processVideoFrame);
});

videoElement.addEventListener('seeked', () => {
    resizeCanvas();
    isWaitingForResponse = false;
    sendCurrentFrame();
});

videoElement.addEventListener('pause', () => {
    isWaitingForResponse = false;
    sendCurrentFrame();
});
