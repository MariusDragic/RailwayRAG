import os
import time
import requests
import feedparser
from pathlib import Path

OUTPUT_DIR = Path("./dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/MariusDragic)"}

SEARCH_TERMS = [
    "railway standards",
    "railway sensors",
    "autonomous train",
    "railway signaling",
    "railway AI",
    "railway perception camera lidar",
    "railway safety system",
    "railway infrastructure",
]

class ArxivScraper:
    def __init__(self, output_dir: Path, headers: dict, search_terms: list):
        self.output_dir = output_dir
        self.headers = headers
        self.search_terms = search_terms

    def scrape_arxiv(self, term, max_results=10):
        print(f"[arXiv] Recherche : {term}")
        query = term.replace(" ", "+")
        url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        response = requests.get(url, headers=self.headers, timeout=30)
        if response.status_code != 200:
            print(f"Erreur {response.status_code} pour {url}")
            return
        feed = feedparser.parse(response.text)
        if not feed.entries:
            print(f"Aucun résultat trouvé pour '{term}' (flux vide ?)")
            print(response.text[:500])
            return
        for entry in feed.entries:
            pdf_link = None
            for link in entry.links:
                if link.get("type") == "application/pdf":
                    pdf_link = link.get("href")
                    break
            if not pdf_link:
                continue
            title = entry.title.replace(" ", "_").replace("/", "_")[:60]
            filename = self.output_dir / f"arxiv_{title}.pdf"
            try:
                r = requests.get(pdf_link, headers=self.headers, timeout=30)
                if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
                    with open(filename, "wb") as f:
                        f.write(r.content)
                    print(f"Téléchargé : {filename}")
                else:
                    print(f"PDF non disponible : {pdf_link}")
            except Exception as e:
                print(f"Erreur téléchargement {pdf_link}: {e}")
            time.sleep(1)

    def run(self):
        print("Démarrage du scraping ferroviaire...\n")
        for term in self.search_terms:
            self.scrape_arxiv(term)
            print("-" * 60)
        print("\n Scraping terminé ! Les PDFs sont dans ./ferro/")

if __name__ == "__main__":
    scraper = ArxivScraper(OUTPUT_DIR, HEADERS, SEARCH_TERMS)
    scraper.run()
