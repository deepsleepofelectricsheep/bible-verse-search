import json
import re
from pathlib import Path

_VERSES_PATH = Path(__file__).parent / "json" / "verses-1769.json"

# Matches refs like "Genesis 1:1", "1 Kings 3:5", "Song of Solomon 2:1"
_REF_RE = re.compile(r"^(.+?)\s+(\d+):(\d+)$")


def load_verses(path: Path = _VERSES_PATH) -> list[dict]:
    """Return a list of verse dicts with keys: book, chapter, verse, text, ref."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    verses = []
    for ref, text in raw.items():
        m = _REF_RE.match(ref)
        if not m:
            continue
        book, chapter, verse = m.group(1), int(m.group(2)), int(m.group(3))
        verses.append({"ref": ref, "book": book, "chapter": chapter, "verse": verse, "text": text})

    return verses
