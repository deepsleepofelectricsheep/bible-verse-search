import re
import math
import numpy as np
import torch

from .base import Searcher, SearchResult

_TOKEN_RE = re.compile(r"[a-z]+")

# Common English stop words to exclude from the vocabulary
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "it", "its",
    "this", "that", "these", "those", "he", "she", "they", "we", "i", "you",
    "his", "her", "their", "our", "my", "your", "him", "them", "us", "me",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "as",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


class TFIDFSearcher(Searcher):
    """
    TF-IDF with log-normalized TF and smoothed IDF.
    The document-term matrix is stored as a float32 torch tensor, enabling
    fast batched cosine similarity via matrix-vector multiplication.
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

    def index(self, verses: list[dict]) -> None:
        self._verses = verses
        n_docs = len(verses)

        # Build vocabulary from corpus
        tokenized = [_tokenize(v["text"]) for v in verses]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        vocab_size = len(vocab)

        # Term frequency matrix: TF[d, t] = log(1 + count(t, d))
        tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for doc_id, tokens in enumerate(tokenized):
            for token in tokens:
                tf[doc_id, vocab[token]] += 1.0
        np.log1p(tf, out=tf)

        # Inverse document frequency: IDF[t] = log((1 + N) / (1 + df(t))) + 1
        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self._idf = idf

        # Combine and L2-normalize each document row
        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        tfidf /= norms

        # Store as torch tensor for fast similarity computation
        self._matrix = torch.from_numpy(tfidf).to(self._device)  # (n_docs, vocab_size)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        vocab_size = len(self._vocab)
        tokens = _tokenize(query)

        # Build query TF-IDF vector using the same scheme
        tf_q = np.zeros(vocab_size, dtype=np.float32)
        for token in tokens:
            if token in self._vocab:
                tf_q[self._vocab[token]] += 1.0
        np.log1p(tf_q, out=tf_q)
        tf_q *= self._idf

        norm = np.linalg.norm(tf_q)
        if norm == 0:
            return []
        tf_q /= norm

        query_vec = torch.from_numpy(tf_q).to(self._device)  # (vocab_size,)

        # Cosine similarity: matrix @ query_vec  (n_docs,)
        scores = self._matrix @ query_vec

        k = min(top_k, len(self._verses))
        top_indices = torch.topk(scores, k).indices.cpu().numpy()

        results = []
        for idx in top_indices:
            v = self._verses[idx]
            results.append(SearchResult(
                book=v["book"],
                chapter=v["chapter"],
                verse=v["verse"],
                text=v["text"],
                score=float(scores[idx]),
            ))
        return results
