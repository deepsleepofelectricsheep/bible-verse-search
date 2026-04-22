"""
Offline evaluation: compares search methods using precision@k and MRR.

Run with:
    .venv/bin/python evaluate.py
"""
from search.base import Searcher

# Hand-labeled queries with known relevant verses.
# Each "relevant" list is the ground-truth set for that query.
LABELED_QUERIES = [
    {
        "query": "love your neighbor",
        "relevant": ["Matthew 22:39", "Mark 12:31", "Romans 13:9", "Leviticus 19:18", "Galatians 5:14"],
    },
    {
        "query": "do not be afraid",
        "relevant": ["Isaiah 41:10", "Joshua 1:9", "Deuteronomy 31:6", "Luke 12:7", "Matthew 10:31"],
    },
    {
        "query": "faith without works is dead",
        "relevant": ["James 2:26", "James 2:17", "James 2:20"],
    },
    {
        "query": "the lord is my shepherd",
        "relevant": ["Psalms 23:1"],
    },
    {
        "query": "in the beginning God created",
        "relevant": ["Genesis 1:1"],
    },
    {
        "query": "turn the other cheek",
        "relevant": ["Matthew 5:39", "Luke 6:29"],
    },
    {
        "query": "blessed are the poor in spirit",
        "relevant": ["Matthew 5:3"],
    },
    {
        "query": "forgive us our trespasses",
        "relevant": ["Matthew 6:12", "Luke 11:4"],
    },
    {
        "query": "wisdom and understanding",
        "relevant": ["Proverbs 4:7", "Proverbs 3:13", "James 1:5", "Proverbs 9:10", "Job 28:28"],
    },
    {
        "query": "resurrection from the dead",
        "relevant": ["John 11:25", "1 Corinthians 15:21", "Acts 4:2", "Romans 6:5"],
    },
]


def precision_at_k(results, relevant: set, k: int) -> float:
    hits = sum(1 for r in results[:k] if f"{r.book} {r.chapter}:{r.verse}" in relevant)
    return hits / k


def mean_reciprocal_rank(results, relevant: set) -> float:
    for rank, r in enumerate(results, start=1):
        if f"{r.book} {r.chapter}:{r.verse}" in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(searcher: Searcher, labeled_queries: list[dict], top_k: int = 10) -> dict:
    p_at_k_scores, mrr_scores = [], []
    for item in labeled_queries:
        relevant = set(item["relevant"])
        results = searcher.search(item["query"], top_k=top_k)
        p_at_k_scores.append(precision_at_k(results, relevant, top_k))
        mrr_scores.append(mean_reciprocal_rank(results, relevant))
    return {
        f"precision@{top_k}": sum(p_at_k_scores) / len(p_at_k_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
    }


def compare(searchers: dict[str, Searcher], top_k: int = 10) -> None:
    col_w = 20
    metric_w = 16
    metrics = [f"precision@{top_k}", "mrr"]

    header = f"{'method':<{col_w}}" + "".join(f"{m:>{metric_w}}" for m in metrics)
    print(header)
    print("-" * len(header))

    for name, searcher in searchers.items():
        results = evaluate(searcher, LABELED_QUERIES, top_k=top_k)
        row = f"{name:<{col_w}}" + "".join(f"{results[m]:>{metric_w}.4f}" for m in metrics)
        print(row)


if __name__ == "__main__":
    from data.loader import load_verses
    from search.tfidf import TFIDFSearcher
    from search.bm25 import BM25Searcher
    from search.embedding import EmbeddingSearcher

    print("Loading verses...")
    verses = load_verses()

    print("Indexing TF-IDF...")
    tfidf = TFIDFSearcher()
    tfidf.index(verses)

    print("Indexing BM25...")
    bm25 = BM25Searcher()
    bm25.index(verses)

    print("Indexing BGE embeddings (this takes a minute on CPU)...")
    emb = EmbeddingSearcher()
    emb.index(verses)

    print(f"\nEvaluating on {len(LABELED_QUERIES)} labeled queries (top_k=10)\n")
    compare({"TF-IDF": tfidf, "BM25": bm25, "BGE (bi-encoder)": emb})
