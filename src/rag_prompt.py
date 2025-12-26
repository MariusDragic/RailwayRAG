from typing import List, Dict
from config import config

def build_prompt(query: str, contexts: List[Dict]) -> List[Dict]:
    ctx_block = "\n\n---\n\n".join(
        [f"[Source {i+1}] (fichier: {h['metadata'].get('source','?')}, page {h['metadata'].get('page','?')})\n{h['text']}" for i, h in enumerate(contexts)]
    )
    system = config.prompt.system_prompt
    user = (
        f"Contexte (extraits de la base ferroviaire) :\n{ctx_block}\n\n"
        f"Question : {query}\n\n"
        "Réponds de façon concise, structurée, et cite les sources comme [1], [2], etc."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
