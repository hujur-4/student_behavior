import time
import numpy as np
from backend.faiss_db import faiss_db
from backend.tracker import ScreenTracker, IncidentManager

print("=== 1. Testing FAISS Database ===")
students = faiss_db.list_students()
print(f"Total enrolled students in FAISS: {len(students)}")
for s in students:
    print(f" - #{s['student_id']}: {s['name']} ({s.get('department')})")

# Test search with exact vector of first student
if faiss_db.index.ntotal > 0:
    first_vec = faiss_db.index.reconstruct(0).tolist()
    match, score = faiss_db.search_student(first_vec, similarity_threshold=0.50)
    print(f"Search match result: {match['name']} with similarity score {score:.4f}")
    assert match is not None, "FAISS search failed to match exact vector"

print("\n=== 2. Testing ScreenTracker Absence Timer ===")
tracker = ScreenTracker(absence_timeout=2.0)  # 2.0s for quick test

# Person appears at t=0
t0 = 100.0
face_box = [100, 100, 200, 200]
id_strap = [{"x1": 120, "y1": 220, "x2": 180, "y2": 320, "label": "with_id_strap"}]

# Frame 1 (Wearing ID)
tracks = tracker.update([face_box], id_strap, current_time=t0)
print(f"Frame 1 - Active tracks: {len(tracks)}, Has ID: {tracks[0].has_id_strap}, Violation: {tracks[0].is_violation}")
assert tracks[0].has_id_strap is True
assert tracks[0].is_violation is False

# Frame 2: 1 second later (Still wearing ID, stands continuously)
t1 = t0 + 1.0
tracks = tracker.update([face_box], id_strap, current_time=t1)
print(f"Frame 2 (1s later) - Track ID: {tracks[0].track_id}, Last seen: {tracks[0].last_seen}")
assert tracks[0].track_id == 1  # Same track!

# Frame 3: 20 seconds later continuously on screen without ID (Simulates standing for 20s)
# Each frame updates last_seen!
for step in range(1, 21):
    curr_t = t1 + step * 1.0
    tracks = tracker.update([face_box], [], current_time=curr_t)

print(f"Frame 23 (After standing continuously for 20s) - Track ID: {tracks[0].track_id}, Active: {len(tracks)}")
assert len(tracks) == 1, "Track should NOT expire while person is continuously on screen!"
assert tracks[0].track_id == 1, "Track ID should stay 1!"
assert tracks[0].is_violation is True, "Violation should be True since ID strap was missing!"

# Frame 24: Person disappears for 3 seconds (> absence_timeout of 2.0s)
t_gone = curr_t + 3.0
tracks = tracker.update([], [], current_time=t_gone)
print(f"After disappearing for 3s (> absence timeout) - Active tracks: {len(tracks)}")
assert len(tracks) == 0, "Track should expire after person is absent for > absence timeout!"

print("\n=== 3. Testing IncidentManager Cooldown Window ===")
# Cooldown = 10 seconds for testing
incident_mgr = IncidentManager(cooldown_seconds=10.0)
incident_mgr.reset()

# First violation at t=1000
is_new_1, count_1 = incident_mgr.register_violation("104", "Alex Carter", current_time=1000.0)
print(f"Incident 1 at t=1000: New={is_new_1}, Count={count_1}")
assert is_new_1 is True and count_1 == 1

# Same person seen 3 seconds later (within 10s cooldown)
is_new_2, count_2 = incident_mgr.register_violation("104", "Alex Carter", current_time=1003.0)
print(f"Incident 2 at t=1003 (within cooldown): New={is_new_2}, Count={count_2}")
assert is_new_2 is False and count_2 == 1, "Should NOT double count within cooldown window!"

# Same person returns 15 seconds later (e.g. 1 hour later in real life > 10s cooldown)
is_new_3, count_3 = incident_mgr.register_violation("104", "Alex Carter", current_time=1016.0)
print(f"Incident 3 at t=1016 (> cooldown): New={is_new_3}, Count={count_3}")
assert is_new_3 is True and count_3 == 2, "Should increment count to 2 after cooldown expires!"

print("\nALL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
