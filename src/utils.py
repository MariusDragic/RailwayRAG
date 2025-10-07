import json
import orjson
from pathlib import Path
from typing import Any, Dict, List
from rich import print as rprint

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def dump_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

def load_json(path: str | Path) -> Any:
    return orjson.loads(Path(path).read_bytes())

def preview(text: str, n: int = 240) -> str:
    s = " ".join(text.split())
    return (s[:n] + "…") if len(s) > n else s

def print_sources(hits: List[Dict]) -> None:
    rprint("\n[bold cyan]Sources (top-k):[/bold cyan]")
    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        rprint(f"[{i}] file={meta.get('source', '?')} page={meta.get('page', '?')} score={h['score']:.4f} → {preview(h['text'], 200)}")
