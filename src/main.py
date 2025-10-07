import os
from pathlib import Path
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from database import RailwayDatabase
from rag_prompt import build_prompt
from ollama_client import OllamaClient
from utils import print_sources

STORE_DIR = Path("store")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    db = RailwayDatabase(STORE_DIR, EMBEDDER_NAME)
    ollama = OllamaClient(model="mistral")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Chargement index & chunks...", total=None)
        db.load()
        progress.update(task, completed=1)
        progress.remove_task(task)

    while True:
        try:
            query = input("\nVotre question sur les documents ferroviaires (ou 'exit' pour quitter) : ")
        except (EOFError, KeyboardInterrupt):
            rprint("\n[bold red]Session terminée.[/bold red]")
            break
        if query.strip().lower() in {"exit", "quit", "q"}:
            rprint("[bold red]Session terminée.[/bold red]")
            break
        rprint(f"[bold cyan]Question:[/bold cyan] {query}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Recherche dans l'index...", total=None)
            hits = db.search(query, top_k=TOP_K)
            progress.update(task, completed=1)
            progress.remove_task(task)

            if not hits:
                rprint("[red]Aucun passage trouvé.[/red]")
                continue

            print_sources(hits)

            task = progress.add_task("Appel à Mistral (Ollama)...", total=None)
            prompt = build_prompt(query, hits)
            answer = ollama.chat_completion(prompt, temperature=0.1)
            progress.update(task, completed=1)
            progress.remove_task(task)

        rprint("\n[bold green]Réponse:[/bold green]\n")
        print(answer)

        rprint("\n[bold]Sources utilisées :[/bold]")
        for i, h in enumerate(hits, 1):
            meta = h["metadata"]
            src = meta.get("source", "?")
            page = meta.get("page", "?")
            rprint(f"[{i}] {src}, page {page}")

if __name__ == "__main__":
    main()
