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
from typing import Iterable, Protocol


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
