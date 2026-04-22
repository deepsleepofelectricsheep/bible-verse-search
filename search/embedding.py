from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .base import Searcher, SearchResult

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CACHE_PATH = Path(__file__).parent.parent / "data" / "bge_embeddings.pt"

# BGE retrieval models expect this prefix on queries (but not on passages)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_BATCH_SIZE = 256


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class EmbeddingSearcher(Searcher):
    """
    Bi-encoder retrieval using BAAI/bge-small-en-v1.5.
    Passages are encoded offline and stored as a normalized (n_docs, 384) float32
    tensor. Query encoding + exact cosine similarity via matmul runs at search time.
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModel.from_pretrained(_MODEL_NAME).to(self._device).eval()

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of texts into L2-normalized embeddings, (n, 384)."""
        all_embeddings = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            output = self._model(**encoded)
            embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def index(self, verses: list[dict]) -> None:
        self._verses = verses
        if _CACHE_PATH.exists():
            print(f"Loading cached embeddings from {_CACHE_PATH}...")
            self._matrix = torch.load(_CACHE_PATH, weights_only=True)
            print("Done.")
            return
        texts = [v["text"] for v in verses]
        print(f"Encoding {len(texts)} passages in batches of {_BATCH_SIZE}...")
        self._matrix = self._encode(texts)  # (n_docs, 384), normalized, on CPU
        torch.save(self._matrix, _CACHE_PATH)
        print(f"Embeddings cached to {_CACHE_PATH}.")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_emb = self._encode([_QUERY_PREFIX + query])  # (1, 384)
        scores = (self._matrix @ query_emb.T).squeeze(1)   # (n_docs,)

        k = min(top_k, len(self._verses))
        top_indices = torch.topk(scores, k).indices.numpy()

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
