from typing import Any, List, Dict, cast
import faiss
from sentence_transformers import SentenceTransformer
from utils import load_json
from config import Config

class RailwayDatabase:
    def __init__(self, config: Config):
        self.config = config
        self.index: faiss.Index | None = None
        self.chunks: List[Dict] | None = None

    def load(self):
        self.index = faiss.read_index(str(self.config.database.index_path))
        self.chunks = load_json(self.config.database.chunks_path)

    def embed_query(self, query: str):
        emb_model = SentenceTransformer(self.config.database.embedder_model)
        vec = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec

    def search(self, query: str, top_k: int) -> List[Dict]:
        if self.index is None or self.chunks is None:
            raise RuntimeError("Database not loaded. Call load() before search().")
        qvec = self.embed_query(query)
        index = cast(Any, self.index)
        D, I = index.search(qvec, top_k)
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
