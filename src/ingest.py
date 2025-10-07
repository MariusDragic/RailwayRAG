from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import ensure_dir, dump_json
import os

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DATASET_DIR = Path("dataset")
STORE_DIR = Path("store")
INDEX_PATH = STORE_DIR / "faiss.index"
CHUNKS_PATH = STORE_DIR / "chunks.json"

class PDFIngestor:
    def __init__(self, dataset_dir: Path, store_dir: Path):
        self.dataset_dir = dataset_dir
        self.store_dir = store_dir
        self.index_path = store_dir / "faiss.index"
        self.chunks_path = store_dir / "chunks.json"

    def extract_pdf_text_with_pages(self, pdf_path: Path) -> List[Dict]:
        reader = PdfReader(str(pdf_path))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                docs.append({"page": i, "text": text, "source": str(pdf_path.name)})
        return docs

    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = []
        for d in docs:
            parts = splitter.split_text(d["text"])
            for idx, part in enumerate(parts):
                chunks.append({
                    "id": f"{d.get('source','')}_p{d['page']}_c{idx}",
                    "text": part,
                    "metadata": {"page": d["page"], "chunk_id": idx, "source": d.get("source", "")}
                })
        return chunks

    def build_faiss(self, chunks: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        embedder = SentenceTransformer(model_name)
        texts = [c["text"] for c in chunks]
        embs = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        return index, chunks

    def ingest(self):
        ensure_dir(self.store_dir)
        pdf_files = list(self.dataset_dir.glob("*.pdf"))
        assert pdf_files, f"Aucun PDF trouvé dans {self.dataset_dir}"
        all_docs = []
        for pdf in pdf_files:
            docs = self.extract_pdf_text_with_pages(pdf)
            all_docs.extend(docs)
        chunks = self.chunk_documents(all_docs)
        index, chunks = self.build_faiss(chunks)
        faiss.write_index(index, str(self.index_path))
        dump_json(chunks, self.chunks_path)

if __name__ == "__main__":
    ingestor = PDFIngestor(DATASET_DIR, STORE_DIR)
    ingestor.ingest()
