from typing import List, Dict

def build_prompt(query: str, contexts: List[Dict]) -> List[Dict]:
    ctx_block = "\n\n---\n\n".join(
        [f"[Source {i+1}] (fichier: {h['metadata'].get('source','?')}, page {h['metadata'].get('page','?')})\n{h['text']}" for i, h in enumerate(contexts)]
    )
    system = (
        "Tu es un assistant expert des documents ferroviaires (standards, capteurs, trains autonomes, signalisation, IA, perception, sécurité, infrastructure, etc).\n"
        "Tu réponds STRICTEMENT à partir des extraits fournis.\n"
        "Réponds en français, de façon concise et structurée (phrases courtes, puces si pertinent).\n"
        "Pour chaque fait, cite la source utilisée sous la forme [n] (où n est le numéro de la source fournie).\n"
        "Si l'information n'est pas présente dans les extraits, dis-le explicitement.\n"
        "À la fin, liste les sources utilisées sous la forme : [n] fichier, page."
    )
    user = (
        f"Contexte (extraits de la base ferroviaire) :\n{ctx_block}\n\n"
        f"Question : {query}\n\n"
        "Réponds de façon concise, structurée, et cite les sources comme [1], [2], etc."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
