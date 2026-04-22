# Bible Verse Search Engine

Search across all 31,102 verses of the King James Bible using three retrieval methods:

| Method | Description |
|--------|-------------|
| **TF-IDF** | Log-normalized TF with smoothed IDF; cosine similarity via matrix-vector multiply |
| **BM25** | Okapi BM25 with Robertson-Sparck Jones IDF; pre-computed score matrix |
| **BGE (bi-encoder)** | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings; exact cosine similarity |

The lexical methods (TF-IDF, BM25) filter stop words and share a common tokenizer. BGE embeddings are computed once and cached to disk.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `verses-1769.json` in `data/json/` (source: [farskipper/kjv](https://github.com/farskipper/kjv)).

## Run

```bash
streamlit run app_ui.py
```

The BGE embeddings are generated on first run (~2 min) and cached to `data/bge_embeddings.pt`.

## License

MIT. Bible text (KJV 1769) is in the public domain.
