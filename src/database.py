from pathlib import Path
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from utils import load_json

class RailwayDatabase:
    def __init__(self, store_dir: Path, embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.store_dir = store_dir
        self.index_path = store_dir / "faiss.index"
        self.chunks_path = store_dir / "chunks.json"
        self.embedder_name = embedder_name
        self.index = None
        self.chunks = None

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.chunks = load_json(self.chunks_path)

    def embed_query(self, query: str):
        emb_model = SentenceTransformer(self.embedder_name)
        vec = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec

    def search(self, query: str, top_k: int) -> List[Dict]:
        qvec = self.embed_query(query)
        D, I = self.index.search(qvec, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            c = self.chunks[idx]
            hits.append({
                "text": c["text"],
                "metadata": c["metadata"],
                "score": float(score)
            })
        return hits
