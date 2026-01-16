from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional
import numpy as np


class EmbeddingBackend(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns (N, D) float32 array. Must be L2-normalized for cosine=dot usage."""
        ...


@dataclass
class SentenceTransformersEmbedder:
    model_name: str = "intfloat/multilingual-e5-small"
    device: Optional[str] = None  # e.g. "cpu", "mps", "cuda" (if available)

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        # E5 family works best with prefixes; we keep symmetry using "passage:"
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


@dataclass
class OllamaEmbedder:
    """
    Uses local Ollama embeddings endpoint.
    Requires Ollama running locally.

    Typical models:
    - nomic-embed-text
    - bge-m3
    - mxbai-embed-large
    """
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout_s: int = 60

    def embed(self, texts: List[str]) -> np.ndarray:
        import requests

        vecs = []
        for t in texts:
            payload = {"model": self.model, "prompt": t}
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            emb = r.json().get("embedding")
            if not emb:
                raise RuntimeError(f"Ollama embeddings returned empty vector for model={self.model}")
            v = np.asarray(emb, dtype=np.float32)
            # normalize to unit length for cosine=dot
            n = float(np.linalg.norm(v) + 1e-12)
            vecs.append(v / n)

        return np.stack(vecs, axis=0).astype(np.float32)
