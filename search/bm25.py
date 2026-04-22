import re
import numpy as np
import torch

from .base import Searcher, SearchResult

_TOKEN_RE = re.compile(r"[a-z]+")

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "it", "its",
    "this", "that", "these", "those", "he", "she", "they", "we", "i", "you",
    "his", "her", "their", "our", "my", "your", "him", "them", "us", "me",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "as",
}

# BM25 hyperparameters (standard defaults)
_K1 = 1.5  # term frequency saturation
_B = 0.75  # document length normalization


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


class BM25Searcher(Searcher):
    """
    BM25 (Okapi BM25) ranking.
    Scores are pre-computed per term as a (n_docs, vocab_size) float32 torch
    tensor so that retrieval is a single matrix-vector multiply, matching the
    TFIDFSearcher interface.

    BM25 score for term t in document d:
        IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1 * (1 - b + b * dl/avgdl))

    IDF uses the Robertson-Sparck Jones formulation:
        log((N - df + 0.5) / (df + 0.5) + 1)
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

    def index(self, verses: list[dict]) -> None:
        self._verses = verses
        n_docs = len(verses)

        tokenized = [_tokenize(v["text"]) for v in verses]

        # Build vocabulary
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        vocab_size = len(vocab)

        # Raw term counts: tf[d, t]
        tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
        doc_lengths = np.zeros(n_docs, dtype=np.float32)
        for doc_id, tokens in enumerate(tokenized):
            doc_lengths[doc_id] = len(tokens)
            for token in tokens:
                tf[doc_id, vocab[token]] += 1.0

        avgdl = doc_lengths.mean()

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)  # (vocab_size,)

        # Length normalization factor per document: k1 * (1 - b + b * dl/avgdl)
        norm_factor = _K1 * (1.0 - _B + _B * (doc_lengths / avgdl))  # (n_docs,)

        # BM25 numerator per cell: tf * (k1 + 1)
        # BM25 denominator per cell: tf + norm_factor (broadcast)
        numerator = tf * (_K1 + 1.0)
        denominator = tf + norm_factor[:, np.newaxis]
        bm25 = idf[np.newaxis, :] * (numerator / denominator)  # (n_docs, vocab_size)

        self._matrix = torch.from_numpy(bm25.astype(np.float32)).to(self._device)  # (n_docs, vocab_size)
        self._idf = idf

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        vocab_size = len(self._vocab)
        tokens = _tokenize(query)

        # Query vector: 1 for each unique query term present in vocab, 0 otherwise.
        # BM25 doesn't apply TF or length norm to the query itself.
        query_vec = np.zeros(vocab_size, dtype=np.float32)
        for token in tokens:
            if token in self._vocab:
                query_vec[self._vocab[token]] = 1.0

        if query_vec.sum() == 0:
            return []

        q = torch.from_numpy(query_vec).to(self._device)

        # Score each doc: sum of BM25 weights for query terms
        scores = self._matrix @ q  # (n_docs,)

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
