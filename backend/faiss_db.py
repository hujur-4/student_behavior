import os
import json
import numpy as np
import faiss
from typing import Optional, Tuple, Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDEX_PATH = os.path.join(DATA_DIR, "students.index")
META_PATH = os.path.join(DATA_DIR, "students_meta.json")

EMBEDDING_DIM = 512

class FAISSDatabase:
    """
    Layer 2 Box 4: Identity Resolution using FAISS Vector Index.
    Stores and queries 512-d normalized face embeddings using Cosine Similarity (IndexFlatIP).
    """
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        os.makedirs(DATA_DIR, exist_ok=True)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)
        self.metadata: List[Dict] = []  # Maps index row id -> student dict
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"[FAISS] Loaded {self.index.ntotal} student embeddings from disk.")
            except Exception as e:
                print(f"[FAISS] Error loading index from disk: {e}. Creating new index.")
                self.index = faiss.IndexFlatIP(self.dim)
                self.metadata = []
        else:
            print("[FAISS] Initialized fresh FAISS IndexFlatIP.")

    def save_to_disk(self):
        try:
            faiss.write_index(self.index, INDEX_PATH)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            print(f"[FAISS] Saved {self.index.ntotal} records to {DATA_DIR}.")
        except Exception as e:
            print(f"[FAISS] Error saving index: {e}")

    def add_student(self, student_id: str, name: str, department: str = "General", 
                    embedding: Optional[List[float]] = None) -> bool:
        """
        Enrolls a student with their 512-d normalized embedding into the FAISS index.
        """
        if embedding is None or len(embedding) != self.dim:
            print(f"[FAISS] Invalid embedding dimension: {len(embedding) if embedding else None} vs {self.dim}")
            return False

        # Ensure embedding is float32 and L2-normalized
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        # Check if student already exists, update if found
        existing_idx = None
        for i, student in enumerate(self.metadata):
            if student["student_id"] == student_id:
                existing_idx = i
                break

        if existing_idx is not None:
            # Rebuild index without old entry to update
            vectors = []
            for i in range(self.index.ntotal):
                if i != existing_idx:
                    vectors.append(self.index.reconstruct(i))
            new_index = faiss.IndexFlatIP(self.dim)
            if vectors:
                new_index.add(np.array(vectors, dtype=np.float32))
            new_index.add(vec)
            self.index = new_index

            self.metadata[existing_idx] = {
                "student_id": student_id,
                "name": name,
                "department": department,
                "enrolled_at": str(np.datetime64('now'))
            }
        else:
            self.index.add(vec)
            self.metadata.append({
                "student_id": student_id,
                "name": name,
                "department": department,
                "enrolled_at": str(np.datetime64('now'))
            })

        self.save_to_disk()
        return True

    def search_student(self, query_embedding: List[float], 
                       similarity_threshold: float = 0.55) -> Tuple[Optional[Dict], float]:
        """
        Queries FAISS for the closest matching student.
        Returns (matched_student_dict, similarity_score).
        Returns (None, best_score) if score is below threshold.
        """
        if self.index.ntotal == 0 or not query_embedding:
            return None, 0.0

        q_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, 1)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_idx >= 0 and best_idx < len(self.metadata):
            if best_score >= similarity_threshold:
                match = dict(self.metadata[best_idx])
                match["similarity"] = round(best_score, 4)
                return match, best_score

        return None, best_score

    def list_students(self) -> List[Dict]:
        return list(self.metadata)

    def delete_student(self, student_id: str) -> bool:
        del_idx = None
        for i, s in enumerate(self.metadata):
            if s["student_id"] == student_id:
                del_idx = i
                break

        if del_idx is None:
            return False

        # Rebuild index without this student
        vectors = []
        for i in range(self.index.ntotal):
            if i != del_idx:
                vectors.append(self.index.reconstruct(i))

        new_index = faiss.IndexFlatIP(self.dim)
        if vectors:
            new_index.add(np.array(vectors, dtype=np.float32))
        self.index = new_index
        self.metadata.pop(del_idx)
        self.save_to_disk()
        return True

# Singleton instance
faiss_db = FAISSDatabase()
