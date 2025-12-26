import time
import requests
import feedparser
from typing import Optional
from pathlib import Path
from config import Config

class ArxivScraper:
    def __init__(self, config: Config):
        self.config = config
        pdf_link: Optional[str]

    def scrape_arxiv(self, term, max_results=None):
        if max_results is None:
            max_results = self.config.scraper.max_results_per_term
        print(f"[arXiv] Recherche : {term}")
        query = term.replace(" ", "+")
        url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        response = requests.get(url, headers=self.config.scraper.headers, timeout=30)
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
            filename = self.config.scraper.dataset_dir / f"arxiv_{title}.pdf"
            try:
                r = requests.get(pdf_link, headers=self.config.scraper.headers, timeout=30)
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
        self.config.scraper.dataset_dir.mkdir(exist_ok=True)
        print("Démarrage du scraping ferroviaire...\n")
        for term in self.config.scraper.search_terms:
            self.scrape_arxiv(term)
            print("-" * 60)
        print(f"\n Scraping terminé ! Les PDFs sont dans {self.config.scraper.dataset_dir}")

if __name__ == "__main__":
    from config import config
    scraper = ArxivScraper(config)
    scraper.run()
