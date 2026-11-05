# VisionDetect AI: Real-Time Face & ID Card Compliance System

A production-ready computer vision pipeline for real-time ID card compliance, student face detection, and identity resolution using **YOLOv8**, **InceptionResnetV1 (FaceNet)**, and a **FAISS Vector Database**.

---

## 🚀 Key Features

* **Dual YOLOv8 Detection**:
  * Face detection (`models/model.pt`)
  * ID card & strap detection (`models/best.pt`) with `with_id_strap` and `without_id_strap` classification.
* **Option B (Conditional Trigger)**:
  * Compliant students (wearing ID straps) skip face recognition entirely $\rightarrow$ **0 ms embedding overhead**.
* **Option C (Decoupled Async Background Worker)**:
  * When a violation occurs (missing ID), InceptionResnetV1 computes 512-d embeddings in a background worker.
  * Live video streams at **30+ FPS (Zero Lag / No stutter)**.
* **FAISS Vector Database**:
  * Real-time identity resolution using cosine similarity (`IndexFlatIP`).
  * Live enrollment via web UI with photo upload and instant vector indexing.
* **Continuous Absence Tracking (No Double-Counting)**:
  * Uses an absence timer instead of a presence timer.
  * Standing continuously on screen never triggers duplicate FaceNet calls or double violations.
* **Incident Cooldown Window (30-Minute Debounce)**:
  * Repeat sightings within 30 minutes merge into the same incident.
  * Re-entering after 1 hour is accurately logged as **Offense #2**.

---

## 🛠️ Project Structure

```
student_behavior/
├── backend/
│   ├── data/                 # FAISS vector index & student metadata
│   ├── embedding.py          # InceptionResnetV1 512-d embedding extraction
│   ├── faiss_db.py           # FAISS Vector Index (IndexFlatIP) manager
│   ├── tracker.py            # ScreenTracker (Absence Timer) & IncidentManager
│   ├── main.py               # FastAPI backend & WebSocket endpoint
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html            # UI with video canvas, stats & FAISS registry
│   ├── main.js               # Canvas renderer & WebSocket client
│   ├── style.css             # Glassmorphism dark-mode UI
│   └── package.json          # Vite frontend dependencies
├── models/
│   ├── model.pt              # YOLO Face Detection weights
│   └── best.pt               # YOLO ID Strap Detection weights
├── test_pipeline.py          # Automated architecture & logic tests
└── README.md
```

---

## 📋 Prerequisites

* **Python 3.10 – 3.13**
* **Node.js 18+ & npm**
* Git

---

## 💻 Step-by-Step Installation & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd student_behavior
```

---

### 2. Backend Setup (Python Virtual Environment)

#### On Windows (PowerShell):
```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create a virtual environment named 'venv'
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all dependencies cleanly
pip install -r requirements.txt

# 6. Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### On Linux / macOS (Bash):
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create a virtual environment named 'venv'
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Note:** The backend is accessible at `http://localhost:8000`. You can test health by visiting `http://localhost:8000/`.

---

### 3. Frontend Setup (Vite Development Server)

Open a **new terminal window** in the project root:

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start development server
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 🧪 Running Automated Tests

To verify that the FAISS database, absence timer, and incident cooldown window are working properly:

```powershell
# From project root with backend venv activated:
python test_pipeline.py
```

Expected output:
```
=== 1. Testing FAISS Database ===
Total enrolled students in FAISS: 4
Search match result: Alex Carter with similarity score 1.0000

=== 2. Testing ScreenTracker Absence Timer ===
Frame 1 - Active tracks: 1, Has ID: True, Violation: False
Frame 23 (After standing continuously for 20s) - Track ID: 1, Active: 1 (Did not expire!)
After disappearing for 3s (> absence timeout) - Active tracks: 0 (Cleanly expired!)

=== 3. Testing IncidentManager Cooldown Window ===
Incident 1 at t=1000: New=True, Count=1
Incident 2 at t=1003 (within cooldown): New=False, Count=1 (No double count!)
Incident 3 at t=1016 (> cooldown / 1 hr later): New=True, Count=2 (Logged as Offense #2!)

ALL ARCHITECTURE TESTS PASSED SUCCESSFULLY!
```

---

## 📡 REST API & WebSocket Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Health check & model status summary |
| `/ws/detect` | `WebSocket` | Real-time continuous base64 frame stream |
| `/api/students` | `GET` | Lists all enrolled students in FAISS index |
| `/api/students/enroll` | `POST` | Enrolls student photo & vectorizes into FAISS |
| `/api/students/{id}` | `DELETE` | Removes student embedding from FAISS index |
| `/api/violations` | `GET` | Returns violation history & incident logs |
| `/api/config` | `POST` | Updates incident cooldown or absence timeout |
| `/api/reset` | `POST` | Resets active tracking and violation history |

---

## 🔧 Common Troubleshooting

1. **`ModuleNotFoundError: No module named 'backend'`**:
   * Fixed in `main.py` with dynamic path resolution. You can run either `cd backend && uvicorn main:app` or from root `uvicorn backend.main:app`.
2. **Windows Execution Policy**:
   * If `.\venv\Scripts\Activate` shows a script execution error, run once:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
3. **OpenCV GUI conflict**:
   * We use `opencv-python-headless` in `requirements.txt` to prevent headless Linux server conflicts.
