"""Embeddings service.

MVP step: basic embeddings for Qdrant retrieval (no LlamaIndex).

Default provider is OpenAI embeddings (lightweight). Optionally supports
SentenceTransformers for multilingual-e5-* later.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List
import importlib
import hashlib
import math

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def _require_openai_key() -> None:
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
        raise RuntimeError("OPENAI_API_KEY is not configured")


@lru_cache(maxsize=1)
def _get_openai_client() -> AsyncOpenAI:
    _require_openai_key()
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache(maxsize=1)
def _get_sentence_transformer_model():
    try:
        st = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(st, "SentenceTransformer")
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it or switch EMBEDDINGS_PROVIDER=openai"
        ) from exc

    model_name = settings.EMBEDDING_MODEL_NAME
    # Allow using shorthand like "multilingual-e5-large" while developing.
    if "/" not in model_name:
        model_name = f"intfloat/{model_name}"

    logger.info("Loading SentenceTransformer model: %s", model_name)
    return SentenceTransformer(model_name)


E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


def _normalize_text_for_e5(text: str, *, is_query: bool) -> str:
    # E5 models expect a prefix for best performance.
    prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
    return f"{prefix}{text.strip()}"


def hash_embedding(text: str, dim: int) -> List[float]:
    """Feature-hashing embedding for EMBEDDINGS_PROVIDER=local: deterministic and offline.

    Not semantically strong, but good enough to test vector plumbing. Shared by
    this service and the LlamaIndex setup so both paths produce the same vectors.
    """
    if dim <= 0:
        raise ValueError("LOCAL_EMBEDDING_DIM must be > 0")
    vec = [0.0] * dim
    tokens = (text or "").lower().split()
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


async def embed_texts(texts: List[str], *, is_query: bool = False) -> List[List[float]]:
    provider = (settings.EMBEDDINGS_PROVIDER or "openai").strip().lower()

    if provider == "local":
        dim = int(settings.LOCAL_EMBEDDING_DIM)
        return [hash_embedding(t, dim) for t in texts]

    if provider == "openai":
        client = _get_openai_client()
        # OpenAI embeddings endpoint accepts a list of strings.
        resp = await client.embeddings.create(model=settings.OPENAI_EMBEDDING_MODEL, input=texts)
        # Keep order stable.
        return [item.embedding for item in resp.data]

    if provider == "sentence-transformers":
        model = _get_sentence_transformer_model()
        normalized = [_normalize_text_for_e5(t, is_query=is_query) for t in texts]
        vectors = model.encode(normalized, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {provider}")


async def embed_query(query: str) -> List[float]:
    vectors = await embed_texts([query], is_query=True)
    return vectors[0]
