"""Pluggable text embedders.

Ships with a dependency-free hashed n-gram embedder so the library works
fully offline out of the box. Swap in a real model (OpenAI, sentence-
transformers, etc.) by implementing the same `embed(text) -> list[float]`
interface.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Optional, Protocol


class Embedder(Protocol):
    dims: int

    def embed(self, text: str) -> list[float]: ...


def _tokens(text: str) -> Iterable[str]:
    text = text.lower()
    for word in re.findall(r"[a-z0-9]+", text):
        # char trigrams capture subword similarity (typos, plurals, stems)
        # without needing a tokenizer or vocabulary.
        if len(word) <= 3:
            yield word
        else:
            for i in range(len(word) - 2):
                yield word[i : i + 3]


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-trigrams embedder.

    Not as good as a real model, but requires no network, no API key, and
    no model download -- so the library is fully usable out of the box.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in _tokens(text):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dims
            sign = 1.0 if (h // self.dims) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VoyageEmbedder:
    """Voyage AI embeddings (Anthropic's recommended embedding provider --
    Anthropic does not offer a first-party embeddings endpoint).

    Requires the 'voyage' extra and a `VOYAGE_API_KEY` (or an explicit
    `api_key`). Model names change over time; pass one explicitly to pin it.
    """

    def __init__(self, model: str = "voyage-3.5", api_key: Optional[str] = None, dims: int = 1024):
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "VoyageEmbedder requires the 'voyage' extra: pip install 'effective-memory[voyage]'"
            ) from e
        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self.model, input_type="document")
        return list(result.embeddings[0])


class OpenAIEmbedder:
    """OpenAI embeddings. Requires the 'openai' extra and `OPENAI_API_KEY`
    (or an explicit `api_key`).
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' extra: pip install 'effective-memory[openai]'"
            ) from e
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dims = 1536 if model == "text-embedding-3-small" else 3072

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers -- no API key or network
    calls needed at inference time (a model is downloaded once). Requires
    the 'sentence-transformers' extra.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'sentence-transformers' extra: "
                "pip install 'effective-memory[sentence-transformers]'"
            ) from e
        self._model = SentenceTransformer(model)
        self.dims = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()


_EMBEDDER_FACTORIES = {
    "hashing": lambda model, api_key: HashingEmbedder(),
    "voyage": lambda model, api_key: VoyageEmbedder(model=model or "voyage-3.5", api_key=api_key),
    "openai": lambda model, api_key: OpenAIEmbedder(model=model or "text-embedding-3-small", api_key=api_key),
    "sentence-transformers": lambda model, api_key: SentenceTransformerEmbedder(model=model or "all-MiniLM-L6-v2"),
}


def get_embedder(name: str = "hashing", model: Optional[str] = None, api_key: Optional[str] = None) -> Embedder:
    """Build an embedder by name: 'hashing' (default, offline), 'voyage',
    'openai', or 'sentence-transformers'.
    """
    try:
        factory = _EMBEDDER_FACTORIES[name]
    except KeyError:
        raise ValueError(f"unknown embedder '{name}'; choose from {sorted(_EMBEDDER_FACTORIES)}") from None
    return factory(model, api_key)
