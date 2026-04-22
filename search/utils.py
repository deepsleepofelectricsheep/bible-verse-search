import re

_TOKEN_RE = re.compile(r"[a-z]+")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "it", "its",
    "this", "that", "these", "those", "he", "she", "they", "we", "i", "you",
    "his", "her", "their", "our", "my", "your", "him", "them", "us", "me",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "as",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def build_vocab(tokenized: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for tokens in tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab
