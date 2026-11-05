from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import os
import sys
import time
import asyncio
from typing import Optional, List, Dict

# Ensure both backend/ and project root are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.embedding import extract_face_embedding_crop, extract_face_embedding
    from backend.faiss_db import faiss_db
    from backend.tracker import ScreenTracker, IncidentManager
except ImportError:
    from embedding import extract_face_embedding_crop, extract_face_embedding
    from faiss_db import faiss_db
    from tracker import ScreenTracker, IncidentManager

app = FastAPI(title="VisionDetect AI - Attendance & ID Compliance Pipeline")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 models for Face Detection and ID Card Detection
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_model_path(filename: str) -> str:
    candidate_dirs = [
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "odel"),
        BASE_DIR
    ]
    for directory in candidate_dirs:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, "models", filename)

FACE_MODEL_PATH = resolve_model_path("model.pt")
ID_CARD_MODEL_PATH = resolve_model_path("best.pt")

try:
    face_model = YOLO(FACE_MODEL_PATH)
    print(f"Face model loaded successfully from {FACE_MODEL_PATH}")
except Exception as e:
    print(f"Error loading face model ({FACE_MODEL_PATH}): {e}")
    face_model = None

try:
    id_card_model = YOLO(ID_CARD_MODEL_PATH)
    print(f"ID Card model loaded successfully from {ID_CARD_MODEL_PATH}")
except Exception as e:
    print(f"Error loading ID card model ({ID_CARD_MODEL_PATH}): {e}")
    id_card_model = None

# Initialize Screen Tracker and Incident Manager
# Absence timeout: 10s (Track expires ONLY if absent for > 10s)
# Incident cooldown: 1800s (30 minutes - if seen 1 hour later, counts as next offense)
screen_tracker = ScreenTracker(absence_timeout=10.0, iou_threshold=0.30)
incident_manager = IncidentManager(cooldown_seconds=1800.0)

# Seed default student demo profiles into FAISS if index is empty
def seed_demo_students_if_empty():
    if faiss_db.index.ntotal == 0:
        demo_profiles = [
            {"student_id": "104", "name": "Alex Carter", "department": "Computer Science"},
            {"student_id": "205", "name": "Sarah Jenkins", "department": "Electrical Eng"},
            {"student_id": "312", "name": "David Kumar", "department": "Mechanical Eng"},
            {"student_id": "418", "name": "Emily Watson", "department": "Information Tech"},
        ]
        # Generate stable distinct synthetic 512-d unit embeddings for demo matching
        np.random.seed(42)
        for profile in demo_profiles:
            synthetic_vec = np.random.randn(512).astype(np.float32).tolist()
            faiss_db.add_student(
                student_id=profile["student_id"],
                name=profile["name"],
                department=profile["department"],
                embedding=synthetic_vec
            )
        print(f"[FAISS] Seeded {len(demo_profiles)} default demo students into vector index.")

seed_demo_students_if_empty()

@app.get("/")
def read_root():
    return {
        "status": "VisionDetect AI backend is operational.",
        "faiss_enrolled_count": faiss_db.index.ntotal,
        "total_violations_recorded": incident_manager.get_total_violations_count(),
        "cooldown_seconds": incident_manager.cooldown_seconds,
        "models": {
            "face_model_loaded": face_model is not None,
            "id_card_model_loaded": id_card_model is not None
        }
    }

# ==================== REST API for FAISS Database ====================

@app.get("/api/students")
def get_students():
    """Lists all students enrolled in the FAISS vector database."""
    return {
        "students": faiss_db.list_students(),
        "count": faiss_db.index.ntotal
    }

@app.post("/api/students/enroll")
async def enroll_student(
    student_id: str = Form(...),
    name: str = Form(...),
    department: str = Form("General"),
    file: UploadFile = File(...)
):
    """
    Enrolls a student by photo:
    1. Detects face with YOLO
    2. Extracts 512-d embedding with InceptionResnetV1
    3. Adds vector to FAISS IndexFlatIP
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Detect face
        if face_model is None:
            raise HTTPException(status_code=500, detail="Face detection model not available.")

        results = face_model(img, conf=0.30, verbose=False)
        face_boxes = []
        for r in results:
            for b in r.boxes:
                face_boxes.append(b.xyxy[0].tolist())

        if not face_boxes:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded photo.")

        # Take largest detected face
        best_box = max(face_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        embedding = extract_face_embedding(img, best_box)

        if not embedding or len(embedding) != 512:
            raise HTTPException(status_code=500, detail="Failed to extract 512-d face embedding.")

        success = faiss_db.add_student(
            student_id=student_id,
            name=name,
            department=department,
            embedding=embedding
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embedding in FAISS index.")

        return {
            "message": f"Successfully enrolled student {name} ({student_id}) into FAISS database.",
            "student_id": student_id,
            "name": name,
            "department": department,
            "total_enrolled": faiss_db.index.ntotal
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/students/{student_id}")
def delete_student(student_id: str):
    """Deletes a student from the FAISS database."""
    success = faiss_db.delete_student(student_id)
    if not success:
        raise HTTPException(status_code=404, detail="Student not found.")
    return {"message": f"Student {student_id} removed from FAISS database."}

@app.get("/api/violations")
def get_violations():
    """Returns violation incidents recorded by the IncidentManager."""
    return {
        "logs": incident_manager.get_logs(),
        "total_violations": incident_manager.get_total_violations_count()
    }

@app.post("/api/config")
def update_config(cooldown_seconds: Optional[float] = None, absence_timeout: Optional[float] = None):
    """Configures cooldown window or absence timer (e.g. set lower for quick testing)."""
    if cooldown_seconds is not None:
        incident_manager.cooldown_seconds = cooldown_seconds
    if absence_timeout is not None:
        screen_tracker.absence_timeout = absence_timeout
    return {
        "cooldown_seconds": incident_manager.cooldown_seconds,
        "absence_timeout": screen_tracker.absence_timeout
    }

@app.post("/api/reset")
def reset_pipeline():
    """Resets active tracks and violation history."""
    screen_tracker.reset()
    incident_manager.reset()
    return {"message": "Pipeline active tracks and violation logs reset successfully."}


# ==================== Background Async Worker (Option C) ====================

async def async_resolve_identity(track_id: int, face_crop: np.ndarray):
    """
    Option C Background Worker:
    Extracts 512-d embedding using InceptionResnetV1 in a background thread,
    queries the FAISS vector index, and updates the active track identity and violation log.
    This runs asynchronously and NEVER blocks the 30 FPS video stream!
    """
    try:
        # Run heavy InceptionResnetV1 computation in worker thread pool
        embedding = await asyncio.to_thread(extract_face_embedding_crop, face_crop)
        
        if not embedding or len(embedding) != 512:
            track = screen_tracker.active_tracks.get(track_id)
            if track:
                track.is_identifying = False
                track.identity = "Unidentified Person"
            return

        # Query FAISS vector index
        match, score = faiss_db.search_student(embedding, similarity_threshold=0.48)
        
        track = screen_tracker.active_tracks.get(track_id)
        if track is None:
            # Person already left the screen while we were computing
            return

        current_time = time.time()
        if match:
            s_id = match["student_id"]
            s_name = match["name"]
            
            # Register violation in IncidentManager (checks 30-min Cooldown!)
            is_new, incident_num = incident_manager.register_violation(s_id, s_name, current_time)
            
            track.student_id = s_id
            track.student_name = s_name
            track.identity = f"{s_name} (#{s_id})"
            track.incident_number = incident_num
            track.similarity = score
            track.violation_logged = True
            print(f"[Violation Detected] {s_name} (#{s_id}) - Offense #{incident_num} (New: {is_new}, Sim: {score:.3f})")
        else:
            # Unregistered in database: log as unregistered violation
            is_new, incident_num = incident_manager.register_violation(f"Unknown_{track_id}", "Unregistered Person", current_time)
            track.student_id = f"Unknown_{track_id}"
            track.student_name = "Unregistered Person"
            track.identity = "Unregistered Person"
            track.incident_number = incident_num
            track.similarity = score
            track.violation_logged = True
            print(f"[Violation Detected] Unregistered Person on Track {track_id}")

        track.is_identifying = False

    except Exception as e:
        print(f"[Async Worker Error] {e}")
        track = screen_tracker.active_tracks.get(track_id)
        if track:
            track.is_identifying = False


# ==================== Real-Time WebSocket Endpoint ====================

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if face_model is None and id_card_model is None:
        await websocket.send_text(json.dumps({"error": "Detection models not loaded"}))
        await websocket.close()
        return

    try:
        while True:
            # Receive base64 frame from frontend
            data = await websocket.receive_text()
            if "," in data:
                data = data.split(",")[1]

            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                current_time = time.time()
                
                # 1. Fast YOLO Face Detection (~15 ms)
                detected_face_boxes = []
                if face_model is not None:
                    face_results = face_model(img, conf=0.25, verbose=False)
                    for r in face_results:
                        for box in r.boxes:
                            detected_face_boxes.append(box.xyxy[0].tolist())

                # 2. Fast YOLO ID Card / Strap Detection (~15 ms)
                id_strap_boxes = []
                if id_card_model is not None:
                    id_results = id_card_model(img, conf=0.40, verbose=False)
                    for r in id_results:
                        for box in r.boxes:
                            b = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            label_name = id_card_model.names.get(cls_id, "id_strap")
                            id_strap_boxes.append({
                                "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                                "confidence": conf,
                                "label": label_name
                            })

                # 3. Update Screen Tracker with Absence Timer & Spatial ID association
                active_tracks = screen_tracker.update(detected_face_boxes, id_strap_boxes, current_time)

                # 4. Process each face track: Option B (Conditional) + Option C (Async Worker)
                rendered_boxes = []
                for track in active_tracks:
                    # Only render if visible in current frame
                    if (current_time - track.last_seen) > 0.5:
                        continue

                    bx1, by1, bx2, by2 = track.bbox

                    if track.has_id_strap:
                        # OPTION B: COMPLIANT! (Wearing ID strap)
                        # Zero heavy FaceNet calls! Frame finishes instantly.
                        rendered_boxes.append({
                            "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                            "type": "face",
                            "status": "compliant",
                            "label": "COMPLIANT (ID OK)",
                            "identity": track.identity or "Compliant",
                            "confidence": 0.95
                        })
                    else:
                        # VIOLATION DETECTED! (ID MISSING / No ID Strap)
                        if track.identity is None and not track.is_identifying:
                            # OPTION C: Dispatch non-blocking background identification worker!
                            track.is_identifying = True
                            
                            # Safely crop face region
                            h, w = img.shape[:2]
                            cx1, cy1 = max(0, int(bx1)), max(0, int(by1))
                            cx2, cy2 = min(w, int(bx2)), min(h, int(by2))
                            face_crop = img[cy1:cy2, cx1:cx2].copy()
                            
                            # Spawn background task (Video NEVER waits for FaceNet!)
                            asyncio.create_task(async_resolve_identity(track.track_id, face_crop))

                            rendered_boxes.append({
                                "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                                "type": "face",
                                "status": "identifying",
                                "label": "VIOLATION: ID MISSING (Identifying...)",
                                "identity": "Identifying...",
                                "confidence": 0.90
                            })
                        elif track.is_identifying:
                            # Background worker still processing
                            rendered_boxes.append({
                                "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                                "type": "face",
                                "status": "identifying",
                                "label": "VIOLATION: ID MISSING (Identifying...)",
                                "identity": "Identifying...",
                                "confidence": 0.90
                            })
                        else:
                            # Identity is resolved and cached! Keep displaying without re-running FaceNet!
                            rendered_boxes.append({
                                "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                                "type": "face",
                                "status": "violation",
                                "label": f"VIOLATION: {track.identity} (Offense #{track.incident_number})",
                                "identity": track.identity,
                                "incident_number": track.incident_number,
                                "confidence": track.similarity or 0.90
                            })

                # Also add ID strap boxes to display
                for strap in id_strap_boxes:
                    rendered_boxes.append({
                        "x1": strap["x1"], "y1": strap["y1"], "x2": strap["x2"], "y2": strap["y2"],
                        "type": "id_card",
                        "status": "compliant" if strap["label"] == "with_id_strap" else "violation",
                        "label": strap["label"],
                        "confidence": strap["confidence"]
                    })

                # IMMEDIATELY return results over WebSocket (30+ FPS smooth video!)
                await websocket.send_text(json.dumps({
                    "boxes": rendered_boxes,
                    "counts": {
                        "faces": len(detected_face_boxes),
                        "id_cards": len(id_strap_boxes),
                        "violations": incident_manager.get_total_violations_count()
                    }
                }))
            else:
                await websocket.send_text(json.dumps({"error": "Failed to decode frame"}))

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
