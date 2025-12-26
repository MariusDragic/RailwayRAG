import requests
from typing import List, Dict, Union
import time
from config import Config


class OllamaClient:
    def __init__(self, config: Config):
        self.config = config

    def chat_completion(
        self,
        messages: Union[str, List[Dict[str, str]]],
        temperature: float = None,
        keep_alive: str = "5m"
    ) -> str:
        """Appelle le modèle Ollama pour un chat complet."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        if temperature is None:
            temperature = self.config.prompt.ollama_temperature

        url = f"{self.config.prompt.ollama_endpoint}/api/chat"
        payload = {
            "model": self.config.prompt.ollama_model,
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
