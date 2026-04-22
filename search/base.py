from abc import ABC, abstractmethod

import torch


class SearchResult:
    def __init__(self, book: str, chapter: int, verse: int, text: str, score: float):
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.text = text
        self.score = score

    def __repr__(self):
        return f"{self.book} {self.chapter}:{self.verse} (score={self.score:.4f})\n  {self.text}"


class Searcher(ABC):
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._verses: list[dict] = []

    @abstractmethod
    def index(self, verses: list[dict]) -> None:
        """Build the index from a list of verse dicts with keys: book, chapter, verse, text."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return the top_k most relevant SearchResults for the query."""

    def _make_results(self, indices, scores) -> list[SearchResult]:
        return [
            SearchResult(
                book=self._verses[idx]["book"],
                chapter=self._verses[idx]["chapter"],
                verse=self._verses[idx]["verse"],
                text=self._verses[idx]["text"],
                score=float(scores[idx]),
            )
            for idx in indices
        ]
