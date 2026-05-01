from abc import ABC, abstractmethod

import torch


class SearchResult:
    def __init__(self, book: str, chapter: int, verse: int, text: str, score: float, idx: int = 0):
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.text = text
        self.score = score
        self.idx = idx  # position in the full corpus, used for context navigation

    def __repr__(self):
        return f"{self.book} {self.chapter}:{self.verse} (score={self.score:.4f})\n  {self.text}"


class ChapterResult:
    def __init__(self, book: str, chapter: int, verses: list[dict], matched_verses: set[int], score: float):
        self.book = book
        self.chapter = chapter
        self.verses = verses                  # all verses in chapter, in order
        self.matched_verses = matched_verses  # verse numbers that matched the query
        self.score = score                    # best verse score in this chapter

    def __repr__(self):
        hits = ", ".join(str(v) for v in sorted(self.matched_verses))
        return f"{self.book} {self.chapter} (score={self.score:.4f}, matched: v{hits})"


def _build_chapter_map(verses: list[dict]) -> dict[tuple, list[dict]]:
    chapter_map: dict[tuple, list[dict]] = {}
    for v in verses:
        key = (v["book"], v["chapter"])
        chapter_map.setdefault(key, []).append(v)
    return chapter_map


def _group_by_chapter(
    verse_results: list[SearchResult],
    chapter_map: dict[tuple, list[dict]],
    top_k: int,
) -> list[ChapterResult]:
    seen: dict[tuple, ChapterResult] = {}
    for r in verse_results:
        key = (r.book, r.chapter)
        if key not in seen:
            seen[key] = ChapterResult(
                book=r.book,
                chapter=r.chapter,
                verses=chapter_map.get(key, []),
                matched_verses={r.verse},
                score=r.score,
            )
        else:
            seen[key].matched_verses.add(r.verse)
    return sorted(seen.values(), key=lambda x: -x.score)[:top_k]


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

    def search_chapters(self, query: str, top_k: int = 10) -> list[ChapterResult]:
        """Return top_k chapters ranked by best verse score, with matched verses annotated."""
        if not hasattr(self, "_chapter_map"):
            self._chapter_map = _build_chapter_map(self._verses)
        # Fetch enough verse candidates to reliably surface top_k unique chapters
        verse_results = self.search(query, top_k=top_k * 8)
        return _group_by_chapter(verse_results, self._chapter_map, top_k)

    def _make_results(self, indices, scores) -> list[SearchResult]:
        return [
            SearchResult(
                book=self._verses[idx]["book"],
                chapter=self._verses[idx]["chapter"],
                verse=self._verses[idx]["verse"],
                text=self._verses[idx]["text"],
                score=float(scores[idx]),
                idx=int(idx),
            )
            for idx in indices
        ]
