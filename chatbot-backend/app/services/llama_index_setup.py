"""Global LlamaIndex configuration setup."""

from __future__ import annotations

import logging
from typing import List

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config import settings

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
    return HuggingFaceEmbedding(model_name=model_name)


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
        # Mock embedding for local testing setup
        from llama_index.core.base.embeddings.base import BaseEmbedding
        class MockEmbedding(BaseEmbedding):
            def _get_query_embedding(self, query: str) -> List[float]:
                return [0.0] * settings.LOCAL_EMBEDDING_DIM
            async def _aget_query_embedding(self, query: str) -> List[float]:
                return [0.0] * settings.LOCAL_EMBEDDING_DIM
            def _get_text_embedding(self, text: str) -> List[float]:
                return [0.0] * settings.LOCAL_EMBEDDING_DIM
            async def _aget_text_embedding(self, text: str) -> List[float]:
                return [0.0] * settings.LOCAL_EMBEDDING_DIM
            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return [[0.0] * settings.LOCAL_EMBEDDING_DIM for _ in texts]
        LlamaIndexSettings.embed_model = MockEmbedding()
    else:
        logger.warning("Unsupported EMBEDDINGS_PROVIDER: %s, defaulting to OpenAI", provider)
        LlamaIndexSettings.embed_model = OpenAIEmbedding(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
