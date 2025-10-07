import os
import requests
from typing import List, Dict, Union
from dotenv import load_dotenv
import time

load_dotenv()


class OllamaClient:
    def __init__(self, endpoint: str = None, model: str = None):
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")

    def chat_completion(
        self,
        messages: Union[str, List[Dict[str, str]]],
        temperature: float = 0.2,
        keep_alive: str = "5m"
    ) -> str:
        """Appelle le modèle Ollama pour un chat complet."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        url = f"{self.endpoint}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature}
        }

        start = time.time()
        r = requests.post(url, json=payload, timeout=600)
        duration = time.time() - start

        if not r.ok:
            print("Requête Ollama invalide :", r.text)
            r.raise_for_status()

        data = r.json()

        if isinstance(data, dict):
            message = data.get("message", {})
            content = message.get("content", "")
            print(f"Réponse Ollama reçue en {duration:.1f}s")
            return content.strip()

        elif isinstance(data, list):
            content = ""
            for item in data:
                if "message" in item and "content" in item["message"]:
                    content += item["message"]["content"]
            print(f"Réponse Ollama reçue en {duration:.1f}s")
            return content.strip()

        else:
            print(f"Réponse inattendue d’Ollama : {type(data)}")
            return str(data)
