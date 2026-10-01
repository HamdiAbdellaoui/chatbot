"""Global LlamaIndex configuration setup."""

from __future__ import annotations

import inspect
import logging
from typing import List

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import settings
from app.services.embeddings_service import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, hash_embedding

logger = logging.getLogger(__name__)


def _build_huggingface_embedding(model_name: str):
    # Optional dependency: only needed for EMBEDDINGS_PROVIDER=sentence-transformers.
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "EMBEDDINGS_PROVIDER=sentence-transformers requires the optional packages "
            "'llama-index-embeddings-huggingface' and 'sentence-transformers'. "
            "Install them (pip install llama-index-embeddings-huggingface sentence-transformers) "
            "or set EMBEDDINGS_PROVIDER=openai."
        ) from exc

    # e5 models expect "query: " / "passage: " prefixes (same as embeddings_service).
    params = inspect.signature(HuggingFaceEmbedding.__init__).parameters
    kwargs = {}
    if "query_instruction" in params and "text_instruction" in params:
        kwargs = {"query_instruction": E5_QUERY_PREFIX, "text_instruction": E5_PASSAGE_PREFIX}
    else:
        logger.warning("Installed HuggingFaceEmbedding does not support e5 instructions; prefixes not applied")
    return HuggingFaceEmbedding(model_name=model_name, **kwargs)


class LocalHashEmbedding(BaseEmbedding):
    """EMBEDDINGS_PROVIDER=local: same feature-hashing vectors as embeddings_service."""

    def _vector(self, text: str) -> List[float]:
        return hash_embedding(text, int(settings.LOCAL_EMBEDDING_DIM))

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._vector(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._vector(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]


def setup_llama_index() -> None:
    """Configures LlamaIndex embeddings provider based on settings."""
    provider = (settings.EMBEDDINGS_PROVIDER or "openai").strip().lower()

    if provider == "openai":
        LlamaIndexSettings.embed_model = OpenAIEmbedding(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    elif provider == "sentence-transformers":
        model_name = settings.EMBEDDING_MODEL_NAME
        if "/" not in model_name:
            model_name = f"intfloat/{model_name}"
        LlamaIndexSettings.embed_model = _build_huggingface_embedding(model_name)
    elif provider == "local":
        # Offline embedding, identical to embeddings_service (ingestion/demo scripts).
        LlamaIndexSettings.embed_model = LocalHashEmbedding()
    else:
        logger.warning("Unsupported EMBEDDINGS_PROVIDER: %s, defaulting to OpenAI", provider)
        LlamaIndexSettings.embed_model = OpenAIEmbedding(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
