import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1

# Initialize device and InceptionResnetV1 model for Deep Embedding (Layer 2 Box 3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            print("Loading InceptionResnetV1 (vggface2) model on device:", device)
            embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            print("InceptionResnetV1 loaded successfully.")
        except Exception as e:
            print(f"Error loading InceptionResnetV1 model: {e}")
            embedding_model = None
    return embedding_model

def extract_face_embedding_crop(face_crop: np.ndarray) -> list:
    """
    Extracts a 512-dimensional normalized deep face embedding vector
    from a cropped face image.
    """
    model = get_embedding_model()
    if model is None or face_crop is None or face_crop.size == 0:
        return []

    try:
        h, w = face_crop.shape[:2]
        if h < 10 or w < 10:
            return []

        # Convert BGR to RGB and resize to 160x160 (InceptionResnetV1 standard input size)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)

        # Standard normalization: (x - 127.5) / 128.0
        face_tensor = (torch.tensor(face_resized, dtype=torch.float32) - 127.5) / 128.0
        # Change layout from HWC to CHW and add batch dimension (1, 3, 160, 160)
        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(face_tensor)
            # L2 normalize the embedding vector
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding[0].cpu().numpy().tolist()

    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return []

def extract_face_embedding(img_bgr: np.ndarray, bbox: list) -> list:
    """
    Extracts a 512-dimensional normalized deep face embedding vector
    from an aligned/cropped face bounding box [x1, y1, x2, y2].
    """
    if img_bgr is None or not bbox:
        return []

    try:
        h, w, _ = img_bgr.shape
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Clamp bounding box coordinates to image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return []

        face_crop = img_bgr[y1:y2, x1:x2]
        return extract_face_embedding_crop(face_crop)

    except Exception as e:
        print(f"Error cropping face: {e}")
        return []
