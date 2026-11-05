// DOM Elements
const videoElement = document.getElementById('video-element');
const canvasOverlay = document.getElementById('canvas-overlay');
const ctx = canvasOverlay.getContext('2d');
const btnWebcam = document.getElementById('btn-webcam');
const fileUpload = document.getElementById('file-upload');
const videoPlaceholder = document.getElementById('video-placeholder');
const connectionDot = document.getElementById('connection-dot');
const connectionStatus = document.getElementById('connection-status');

// Counters
const fpsCounter = document.getElementById('fps-counter');
const facesCounter = document.getElementById('faces-counter');
const idCardsCounter = document.getElementById('id-cards-counter');
const violationsCounter = document.getElementById('violations-counter');

// Tab & Database Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const faissTotalCount = document.getElementById('faiss-total-count');
const violationsLogCount = document.getElementById('violations-log-count');
const studentsTableBody = document.getElementById('students-table-body');
const violationsTableBody = document.getElementById('violations-table-body');
const btnRefreshStudents = document.getElementById('btn-refresh-students');
const btnRefreshViolations = document.getElementById('btn-refresh-violations');
const btnResetPipeline = document.getElementById('btn-reset-pipeline');

// Modal Elements
const enrollModal = document.getElementById('enroll-modal');
const btnOpenEnroll = document.getElementById('btn-open-enroll');
const btnCancelEnroll = document.getElementById('btn-cancel-enroll');
const modalClose = document.getElementById('modal-close');
const enrollForm = document.getElementById('enroll-form');
const enrollStatusMsg = document.getElementById('enroll-status-msg');

const API_BASE = 'http://localhost:8000';

// State
let isWebcamActive = false;
let stream = null;
let ws = null;
let isConnected = false;
let animationId = null;
let lastFrameTime = 0;
let fps = 0;
let isWaitingForResponse = false; // Backpressure lock to prevent frame queuing lag

// ==================== WebSocket Real-time Detection ====================

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8000/ws/detect');
    
    ws.onopen = () => {
        isConnected = true;
        connectionDot.classList.add('connected');
        connectionStatus.textContent = 'Connected to VisionDetect AI';
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
        try {
            const data = JSON.parse(event.data);
            if (data.boxes) {
                drawBoxes(data.boxes);
            }
            if (data.counts) {
                if (facesCounter) facesCounter.textContent = data.counts.faces;
                if (idCardsCounter) idCardsCounter.textContent = data.counts.id_cards;
                if (violationsCounter) violationsCounter.textContent = data.counts.violations;
            }
        } catch (e) {
            console.error("Error parsing detection frame:", e);
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

// Color mapping for compliance and violation statuses
function getColorForBox(box) {
    if (box.status === 'compliant' || box.label === 'with_id_strap') {
        return '#10b981'; // Emerald Green: Compliant
    }
    if (box.status === 'identifying') {
        return '#f59e0b'; // Amber Orange: Identifying face in background
    }
    if (box.status === 'violation' || box.label === 'without_id_strap') {
        return '#ef4444'; // Bright Red: Violation
    }
    return '#3b82f6'; // Blue fallback
}

// Draw Bounding Boxes with Labels & Badges
function drawBoxes(boxes) {
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    
    if (!videoElement.videoWidth || !videoElement.videoHeight) return;

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
        
        // Draw label text & badge background
        const labelText = box.label || 'Detected';
        ctx.font = '600 13px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const badgeHeight = 24;
        const badgeY = Math.max(0, y - badgeHeight);

        ctx.fillStyle = boxColor;
        ctx.fillRect(x, badgeY, textWidth + 14, badgeHeight);
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + 7, badgeY + 16);
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

// ==================== Webcam & Video Playback Controls ====================

btnWebcam.addEventListener('click', async () => {
    if (isWebcamActive) {
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

fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        if (isWebcamActive) {
            btnWebcam.click();
        }
        
        const url = URL.createObjectURL(file);
        videoElement.srcObject = null;
        videoElement.src = url;
        videoElement.controls = true;
        videoElement.loop = true;
        videoElement.style.opacity = '1';
        videoPlaceholder.style.display = 'none';
        
        videoElement.onloadedmetadata = () => {
            resizeCanvas();
            videoElement.play().catch(err => console.log("Autoplay notification:", err));
        };
    }
});

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

// ==================== FAISS Database & Violations Log API ====================

// Fetch Enrolled Students from FAISS Database
async function fetchStudents() {
    try {
        const res = await fetch(`${API_BASE}/api/students`);
        const data = await res.json();
        
        if (faissTotalCount) faissTotalCount.textContent = data.count || 0;
        
        if (!data.students || data.students.length === 0) {
            studentsTableBody.innerHTML = `<tr><td colspan="5" class="empty-state">No students enrolled in FAISS database.</td></tr>`;
            return;
        }

        studentsTableBody.innerHTML = data.students.map(s => `
            <tr>
                <td><strong>#${s.student_id}</strong></td>
                <td>${s.name}</td>
                <td><span class="badge dept">${s.department || 'General'}</span></td>
                <td>${s.enrolled_at ? s.enrolled_at.split('T')[0] : 'Enrolled'}</td>
                <td>
                    <button class="btn delete-btn" onclick="deleteStudent('${s.student_id}')">Remove</button>
                </td>
            </tr>
        `).join('');
    } catch (e) {
        console.error("Error fetching students:", e);
        studentsTableBody.innerHTML = `<tr><td colspan="5" class="empty-state error">Failed to connect to FAISS database.</td></tr>`;
    }
}

// Delete student from FAISS database
window.deleteStudent = async function(studentId) {
    if (!confirm(`Are you sure you want to remove Student #${studentId} from the FAISS database?`)) return;
    try {
        const res = await fetch(`${API_BASE}/api/students/${studentId}`, { method: 'DELETE' });
        if (res.ok) {
            fetchStudents();
        } else {
            alert("Failed to delete student.");
        }
    } catch (e) {
        alert("Error deleting student: " + e.message);
    }
};

// Fetch Violation Logs from IncidentManager
async function fetchViolations() {
    try {
        const res = await fetch(`${API_BASE}/api/violations`);
        const data = await res.json();
        
        if (violationsLogCount) violationsLogCount.textContent = data.total_violations || 0;
        if (violationsCounter) violationsCounter.textContent = data.total_violations || 0;

        if (!data.logs || data.logs.length === 0) {
            violationsTableBody.innerHTML = `<tr><td colspan="5" class="empty-state">No violations recorded yet.</td></tr>`;
            return;
        }

        violationsTableBody.innerHTML = data.logs.map(log => `
            <tr>
                <td>${log.readable_time || 'Just now'}</td>
                <td><strong>${log.name}</strong></td>
                <td>#${log.student_id}</td>
                <td><span class="badge danger">${log.violation_type}</span></td>
                <td><span class="badge count">Offense #${log.incident_number}</span></td>
            </tr>
        `).join('');
    } catch (e) {
        console.error("Error fetching violations:", e);
    }
}

// Reset Active Pipeline
btnResetPipeline.addEventListener('click', async () => {
    if (!confirm("Reset active tracking and violation history?")) return;
    try {
        await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
        fetchViolations();
        if (violationsCounter) violationsCounter.textContent = '0';
        alert("Active tracker and violation logs reset.");
    } catch (e) {
        alert("Reset failed: " + e.message);
    }
});

// Refresh Buttons
btnRefreshStudents.addEventListener('click', fetchStudents);
btnRefreshViolations.addEventListener('click', fetchViolations);

// Tab Navigation
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        const targetTab = document.getElementById(btn.dataset.tab);
        if (targetTab) targetTab.classList.add('active');

        if (btn.dataset.tab === 'tab-faiss') fetchStudents();
        if (btn.dataset.tab === 'tab-violations') fetchViolations();
    });
});

// ==================== Student Enrollment Modal ====================

btnOpenEnroll.addEventListener('click', () => {
    enrollModal.classList.add('active');
    enrollStatusMsg.textContent = '';
});

function closeModal() {
    enrollModal.classList.remove('active');
    enrollForm.reset();
    enrollStatusMsg.textContent = '';
}

btnCancelEnroll.addEventListener('click', closeModal);
modalClose.addEventListener('click', closeModal);

enrollForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const studentId = document.getElementById('enroll-id').value.trim();
    const name = document.getElementById('enroll-name').value.trim();
    const department = document.getElementById('enroll-dept').value.trim();
    const photoFile = document.getElementById('enroll-photo').files[0];

    if (!photoFile) {
        alert("Please select a student face photo.");
        return;
    }

    const formData = new FormData();
    formData.append('student_id', studentId);
    formData.append('name', name);
    formData.append('department', department);
    formData.append('file', photoFile);

    enrollStatusMsg.textContent = 'Extracting 512-d InceptionResnetV1 embedding and adding to FAISS...';
    enrollStatusMsg.className = 'status-msg info';

    try {
        const res = await fetch(`${API_BASE}/api/students/enroll`, {
            method: 'POST',
            body: formData
        });

        const result = await res.json();
        if (res.ok) {
            enrollStatusMsg.textContent = `Success! Enrolled ${name} into FAISS database.`;
            enrollStatusMsg.className = 'status-msg success';
            setTimeout(() => {
                closeModal();
                fetchStudents();
            }, 1200);
        } else {
            enrollStatusMsg.textContent = `Error: ${result.detail || 'Enrollment failed.'}`;
            enrollStatusMsg.className = 'status-msg error';
        }
    } catch (err) {
        enrollStatusMsg.textContent = `Error: ${err.message}`;
        enrollStatusMsg.className = 'status-msg error';
    }
});

// Initial Data Load
fetchStudents();
fetchViolations();

// Periodic log update every 5 seconds
setInterval(() => {
    fetchViolations();
}, 5000);
