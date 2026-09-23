import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(filename):
    for folder in ["models", "odel", ""]:
        p = os.path.join(BASE_DIR, folder, filename) if folder else os.path.join(BASE_DIR, filename)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, "models", filename)

for filename in ["model.pt", "best.pt"]:
    path = get_model_path(filename)
    try:
        print(f"Loading {filename} from {path}...")
        m = YOLO(path)
        print(f"Loaded {filename} successfully! Classes: {m.names}")
    except Exception as e:
        print(f"Failed to load {filename}: {e}")

