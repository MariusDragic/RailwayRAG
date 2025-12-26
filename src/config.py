from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class DatabaseConfig(BaseModel):
    """Configuration for database and embeddings"""
    store_dir: Path = Field(default=Path("store"), description="Directory for storing index and chunks")
    embedder_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    chunk_size: int = Field(default=800, description="Size of text chunks")
    chunk_overlap: int = Field(default=150, description="Overlap between chunks")
    top_k: int = Field(default=5, description="Number of results to retrieve")
    index_path: Path = Field(default=Path("faiss.index"), description="Relative or absolute FAISS index path")
    chunks_path: Path = Field(default=Path("chunks.json"), description="Relative or absolute chunks JSON path")

    @model_validator(mode="after")
    def _set_paths(self):
        if not self.index_path.is_absolute():
            self.index_path = self.store_dir / self.index_path
        if not self.chunks_path.is_absolute():
            self.chunks_path = self.store_dir / self.chunks_path
        return self


class PromptConfig(BaseModel):
    """Configuration for prompts and LLM"""
    system_prompt: str = Field(
        default="Tu es un assistant spécialisé en analyse de documents ferroviaires. "
                "Réponds uniquement en te basant sur les documents fournis.",
        description="System prompt for the LLM"
    )
    ollama_endpoint: str = Field(default="http://localhost:11434", description="Ollama API endpoint")
    ollama_model: str = Field(default="mistral", description="Ollama model name")
    ollama_temperature: float = Field(default=0.2, description="Temperature for generation")


class ScraperConfig(BaseModel):
    """Configuration for web scraping"""
    dataset_dir: Path = Field(default=Path("dataset"), description="Directory for downloaded PDFs")
    headers: dict = Field(
        default_factory=lambda: {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/MariusDragic)"},
        description="HTTP headers for requests"
    )
    search_terms: list[str] = Field(
        default_factory=lambda: [
            "railway standards",
            "railway sensors",
            "autonomous train",
            "railway signaling",
            "railway AI",
            "railway perception camera lidar",
            "railway safety system",
            "railway infrastructure",
        ],
        description="Search terms for arXiv"
    )
    max_results_per_term: int = Field(default=10, description="Maximum results per search term")


class Config(BaseModel):
    """Main configuration class that bundles all configs"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)


config = Config()
