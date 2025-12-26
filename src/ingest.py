from pathlib import Path
from typing import Any, List, Dict, cast
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import ensure_dir, dump_json
from config import Config

class PDFIngestor:
    def __init__(self, config: Config):
        self.config = config

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
            chunk_size=self.config.database.chunk_size,
            chunk_overlap=self.config.database.chunk_overlap,
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

    def build_faiss(self, chunks: List[Dict]):
        embedder = SentenceTransformer(self.config.database.embedder_model)
        texts = [c["text"] for c in chunks]
        embs: NDArray[np.float32] = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32", copy=False)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        cast(Any, index).add(embs)
        return index, chunks

    def ingest(self):
        ensure_dir(self.config.database.store_dir)
        pdf_files = list(self.config.scraper.dataset_dir.glob("*.pdf"))
        assert pdf_files, f"Aucun PDF trouvé dans {self.config.scraper.dataset_dir}"
        all_docs = []
        for pdf in pdf_files:
            docs = self.extract_pdf_text_with_pages(pdf)
            all_docs.extend(docs)
        chunks = self.chunk_documents(all_docs)
        index, chunks = self.build_faiss(chunks)
        faiss.write_index(index, str(self.config.database.index_path))
        dump_json(chunks, self.config.database.chunks_path)

if __name__ == "__main__":
    from config import config
    ingestor = PDFIngestor(config)
    ingestor.ingest()
