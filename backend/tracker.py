import time
import os
import json
from typing import Dict, List, Optional, Tuple

class TrackedFace:
    """Represents an active face tracked on screen."""
    def __init__(self, track_id: int, bbox: List[float], current_time: float):
        self.track_id: int = track_id
        self.bbox: List[float] = bbox  # [x1, y1, x2, y2]
        self.created_at: float = current_time
        self.last_seen: float = current_time  # Constantly refreshed every frame they are visible!
        self.has_id_strap: bool = False
        
        # Identification State (Option C - Async Worker)
        self.identity: Optional[str] = None         # e.g., "Alex #104"
        self.student_id: Optional[str] = None      # e.g., "104"
        self.student_name: Optional[str] = None    # e.g., "Alex Carter"
        self.is_identifying: bool = False          # True while InceptionResnetV1 is running
        self.similarity: float = 0.0
        
        # Violation state
        self.is_violation: bool = False
        self.incident_number: int = 1
        self.violation_logged: bool = False

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

class ScreenTracker:
    """
    Tracks active faces on screen using spatial IoU.
    Uses an ABSENCE TIMER:
    - While a person stands continuously on screen, last_seen is refreshed on every frame.
    - Track expires ONLY if the person is absent (completely gone) for > absence_timeout seconds.
    """
    def __init__(self, absence_timeout: float = 10.0, iou_threshold: float = 0.30):
        self.absence_timeout = absence_timeout
        self.iou_threshold = iou_threshold
        self.active_tracks: Dict[int, TrackedFace] = {}
        self.next_track_id: int = 1

    def update(self, detected_face_boxes: List[List[float]], 
               id_strap_boxes: List[Dict], 
               current_time: float) -> List[TrackedFace]:
        """
        Associates incoming YOLO face detections with active tracks and ID straps.
        """
        # 1. First, check which faces have an associated ID strap (Option B - Compliance check)
        face_strap_map = []
        for f_box in detected_face_boxes:
            has_valid_strap = False
            fx1, fy1, fx2, fy2 = f_box
            fcx = (fx1 + fx2) / 2.0
            f_height = fy2 - fy1

            for strap in id_strap_boxes:
                # Check if strap is "with_id_strap" and positioned below/near face (chest region)
                if strap.get("label") == "with_id_strap":
                    sx1, sy1, sx2, sy2 = strap["x1"], strap["y1"], strap["x2"], strap["y2"]
                    scx = (sx1 + sx2) / 2.0
                    
                    # Horizontal proximity check
                    horiz_ok = abs(fcx - scx) < (fx2 - fx1) * 1.8
                    # Vertical proximity: strap starts near or below face bottom
                    vert_ok = (sy1 >= fy1) and (sy1 <= fy2 + f_height * 3.5)
                    
                    if horiz_ok and vert_ok:
                        has_valid_strap = True
                        break
            
            face_strap_map.append(has_valid_strap)

        # 2. Match detected faces with existing active tracks
        unmatched_detections = list(range(len(detected_face_boxes)))
        matched_tracks = set()

        for track_id, track in self.active_tracks.items():
            best_iou = 0.0
            best_det_idx = None

            for det_idx in unmatched_detections:
                iou = calculate_iou(track.bbox, detected_face_boxes[det_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= self.iou_threshold and best_det_idx is not None:
                # Update existing track! Refresh last_seen so absence timer resets!
                track.bbox = detected_face_boxes[best_det_idx]
                track.last_seen = current_time
                track.has_id_strap = face_strap_map[best_det_idx]
                track.is_violation = not track.has_id_strap
                matched_tracks.add(track_id)
                unmatched_detections.remove(best_det_idx)

        # 3. Create new tracks for unmatched new faces
        for det_idx in unmatched_detections:
            new_track = TrackedFace(
                track_id=self.next_track_id,
                bbox=detected_face_boxes[det_idx],
                current_time=current_time
            )
            new_track.has_id_strap = face_strap_map[det_idx]
            new_track.is_violation = not new_track.has_id_strap
            self.active_tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # 4. Cleanup tracks that have been ABSENT (gone from screen) for > absence_timeout
        expired_ids = [
            t_id for t_id, track in self.active_tracks.items()
            if (current_time - track.last_seen) > self.absence_timeout
        ]
        for t_id in expired_ids:
            del self.active_tracks[t_id]

        return list(self.active_tracks.values())

    def reset(self):
        self.active_tracks.clear()
        self.next_track_id = 1


class IncidentManager:
    """
    Manages student violation history and Incident Cooldown Windows:
    - If a student is flagged multiple times within the cooldown window (e.g. 30 mins),
      it merges into the SAME incident and does NOT increment violation count.
    - If the student returns after the cooldown (e.g. 1 hour later), it logs an OFFENSE #2.
    """
    def __init__(self, cooldown_seconds: float = 1800.0):  # Default 30 minutes
        self.cooldown_seconds = cooldown_seconds
        # student_id -> { "last_incident_time": float, "incident_count": int, "name": str }
        self.student_history: Dict[str, Dict] = {}
        self.violations_log: List[Dict] = []
        
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "violations_log.json")
        self._load_logs()

    def _load_logs(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.violations_log = data.get("logs", [])
                    self.student_history = data.get("history", {})
            except Exception as e:
                print(f"[IncidentManager] Error loading logs: {e}")

    def _save_logs(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "history": self.student_history,
                    "logs": self.violations_log
                }, f, indent=2)
        except Exception as e:
            print(f"[IncidentManager] Error saving logs: {e}")

    def register_violation(self, student_id: str, student_name: str, 
                           current_time: Optional[float] = None) -> Tuple[bool, int]:
        """
        Records a violation for an identified student.
        Returns: (is_new_incident: bool, current_incident_number: int)
        """
        if current_time is None:
            current_time = time.time()

        history = self.student_history.get(student_id)

        if history is None:
            # First time this student has ever violated
            self.student_history[student_id] = {
                "name": student_name,
                "incident_count": 1,
                "last_incident_time": current_time
            }
            self._record_log_entry(student_id, student_name, 1, current_time, is_new=True)
            self._save_logs()
            return True, 1

        # Student has past history. Check cooldown window!
        time_since_last = current_time - history["last_incident_time"]

        if time_since_last > self.cooldown_seconds:
            # COOLDOWN EXPIRED! (e.g. 1 hour later) -> Count as a NEW offense!
            history["incident_count"] += 1
            history["last_incident_time"] = current_time
            self._record_log_entry(student_id, student_name, history["incident_count"], current_time, is_new=True)
            self._save_logs()
            return True, history["incident_count"]
        else:
            # WITHIN SAME COOLDOWN WINDOW! (e.g. 2 minutes later) -> Merge into same incident!
            return False, history["incident_count"]

    def _record_log_entry(self, student_id: str, name: str, incident_num: int, 
                          timestamp: float, is_new: bool):
        readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        entry = {
            "student_id": student_id,
            "name": name,
            "incident_number": incident_num,
            "timestamp": timestamp,
            "readable_time": readable_time,
            "violation_type": "ID_STRAP_MISSING",
            "is_new_incident": is_new
        }
        self.violations_log.append(entry)

    def get_logs(self) -> List[Dict]:
        return list(reversed(self.violations_log))

    def get_total_violations_count(self) -> int:
        return sum(h.get("incident_count", 0) for h in self.student_history.values())

    def reset(self):
        self.student_history.clear()
        self.violations_log.clear()
        self._save_logs()
