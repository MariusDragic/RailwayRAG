# RAG - Railway Document Question Answering

## Overview
RAG is a Retrieval-Augmented Generation (RAG) backend designed to answer questions about railway-related documents (standards, sensors, autonomous trains, signaling, AI, perception, safety, infrastructure, etc.) using state-of-the-art language models and semantic search. It leverages a vector database (FAISS) and a local LLM (Ollama, e.g. mistral) to provide concise, structured, and sourced answers in French.

## Features
- Scrapes and downloads railway-related scientific papers from arXiv
- Extracts and chunks PDF content for semantic search
- Builds a FAISS vector store for fast retrieval
- Uses Sentence Transformers for embedding
- RAG pipeline: retrieves relevant chunks, builds a prompt, and queries a local LLM (Ollama)
- Interactive CLI: ask multiple questions in a single session
- All answers are concise, structured, and cite their sources ([1], [2], ...)

## Project Structure
```
5-RAG/
│
├── dataset/           # All ferro source PDFs
├── src/               # All backend source code 
│   ├── database.py    # RailwayDatabase: loads/searches the vector DB
│   ├── ingest.py      # PDFIngestor: builds the vector DB from PDFs
│   ├── main.py        # CLI entry point (interactive Q&A)
│   ├── ollama_client.py # OllamaClient: handles LLM API calls
│   ├── rag_prompt.py  # Prompt builder for RAG
│   ├── scraper.py     # ArxivScraper: downloads railway PDFs
│   └── utils.py       # Utility functions
├── store/             # Vector DB (faiss.index) and chunks (chunks.json)
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

## Dependencies
- Python 3.10+
- [Ollama](https://ollama.com/) (local LLM server, e.g. phi3:mini)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- [pypdf](https://pypdf.readthedocs.io/)
- [langchain-text-splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/)
- [rich](https://rich.readthedocs.io/)
- [feedparser](https://pythonhosted.org/feedparser/)
- [orjson](https://github.com/ijl/orjson)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Railway PDFs (optional)
You can use the arXiv scraper to download relevant PDFs:
```bash
python src/scraper.py
```
Move or copy the PDFs you want to use into the `dataset/` folder.

### 2. Build the Vector Store
Extracts, chunks, and indexes all PDFs in `dataset/`:
```bash
python src/ingest.py
```
This creates `store/faiss.index` and `store/chunks.json`.

### 3. Start the Ollama Server
Make sure Ollama is running and the `mistral` model is available:
```bash
ollama run mistral
```

### 4. Ask Questions (Interactive CLI)
Start the interactive backend:
```bash
python src/main.py
```
You can now ask questions in French about any aspect of the railway documents. Type `exit` to quit.

## Example

```
Votre question sur les documents ferroviaires (ou 'exit' pour quitter) : Quelles sont les normes safety relative au ferroviaire ?

Réponse :

Les normes safety relatives au ferroviaire mentionnées dans les extraits sont :

- IEC 62443 [1]
- IEC 62425 [5]
- SIL (Safety Integrity Level) [3], [4]

Sources utilisées :
[1] arxiv_A_Security_Architecture_for_Railway_Signalling.pdf, page 4
[2] arxiv_Towards_an_IT_Security_Risk_Assessment_Framework_for_Railway.pdf, page 3
[3] 2207.13412v2.pdf, page 2
[4] 2207.13412v2.pdf, page 7
[5] arxiv_Towards_an_IT_Security_Risk_Assessment_Framework_for_Railway.pdf, page 3
```

## License & Copyright
All code © 2024-2025 Marius Dragic. Scientific papers and PDFs remain the property of their respective authors and publishers. This project is for research and educational purposes only.

## Contact
For questions, suggestions, or contributions, contact :
**Marius Dragic**  
Email: marius.dragic@gmail.com
