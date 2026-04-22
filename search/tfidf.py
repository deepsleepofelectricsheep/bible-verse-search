import numpy as np
import torch

from .base import Searcher, SearchResult
from .utils import tokenize, build_vocab


class TFIDFSearcher(Searcher):
    """
    TF-IDF with log-normalized TF and smoothed IDF.
    The document-term matrix is stored as a float32 torch tensor, enabling
    fast batched cosine similarity via matrix-vector multiplication.
    """

    def index(self, verses: list[dict]) -> None:
        self._verses = verses
        n_docs = len(verses)

        tokenized = [tokenize(v["text"]) for v in verses]
        self._vocab = build_vocab(tokenized)
        vocab_size = len(self._vocab)

        # Term frequency matrix: TF[d, t] = log(1 + count(t, d))
        tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for doc_id, tokens in enumerate(tokenized):
            for token in tokens:
                tf[doc_id, self._vocab[token]] += 1.0
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

        self._matrix = torch.from_numpy(tfidf).to(self._device)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        vocab_size = len(self._vocab)
        tokens = tokenize(query)

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

        query_vec = torch.from_numpy(tf_q).to(self._device)
        scores = self._matrix @ query_vec

        k = min(top_k, len(self._verses))
        top_indices = torch.topk(scores, k).indices.cpu().numpy()
        return self._make_results(top_indices, scores)
