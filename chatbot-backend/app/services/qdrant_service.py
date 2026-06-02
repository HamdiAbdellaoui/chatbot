"""Qdrant service for basic vector insert/search.

This implements the minimal building blocks you need for RAG retrieval:
- connect to Qdrant
- ensure a collection exists
- upsert documents with embeddings + payload
- search by query embedding

No LlamaIndex involved.
"""

from __future__ import annotations

import logging
import atexit
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.services.embeddings_service import embed_query, embed_texts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


def _normalize_point_id(raw_id: Any, *, collection: str) -> int | str:
    """Normalize an external document id into a Qdrant-compatible point id.

    Qdrant point ids must be either:
    - int
    - UUID (represented as uuid.UUID or a UUID string)

    We accept arbitrary strings in the app layer and convert them deterministically
    to UUIDs to keep ids stable across runs.
    """
    if isinstance(raw_id, int):
        return raw_id

    if raw_id is None:
        # Extremely defensive fallback; should not happen with our Document type.
        return str(uuid.uuid4())

    raw_str = str(raw_id).strip()
    if raw_str.isdigit():
        try:
            return int(raw_str)
        except Exception:
            pass

    try:
        return str(uuid.UUID(raw_str))
    except Exception:
        # Deterministic mapping for non-UUID strings.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant:{collection}:{raw_str}"))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Create a Qdrant client.

    Supports:
    - Remote server: QDRANT_URL="http://localhost:6333"
    - In-memory local mode: QDRANT_URL=":memory:"
    - Local persistent mode: QDRANT_URL="local:./qdrant_local"
    """
    target = (settings.QDRANT_URL or "").strip()
    api_key = settings.QDRANT_API_KEY or None

    if target in {":memory:", "memory"}:
        logger.warning("Using Qdrant local in-memory mode (data will NOT persist across processes)")
        client = QdrantClient(location=":memory:")
        atexit.register(lambda: client.close())
        return client

    if target.lower().startswith("local:"):
        path = target.split(":", 1)[1].strip() or "./qdrant_local"
        logger.warning("Using Qdrant local persistent mode at path=%s", path)
        client = QdrantClient(path=path)
        atexit.register(lambda: client.close())
        return client

    return QdrantClient(url=target, api_key=api_key)


def ensure_collection(*, client: QdrantClient, collection: str, vector_size: int) -> None:
    exists = client.collection_exists(collection_name=collection)
    if exists:
        return

    logger.info("Creating Qdrant collection=%s vector_size=%s", collection, vector_size)
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


async def upsert_documents(
    *,
    collection: str,
    documents: List[Document],
) -> None:
    if not documents:
        return

    client = get_client()

    texts = [d.text for d in documents]
    vectors = await embed_texts(texts, is_query=False)
    vector_size = len(vectors[0])

    ensure_collection(client=client, collection=collection, vector_size=vector_size)

    points: List[qm.PointStruct] = []
    for doc, vec in zip(documents, vectors, strict=False):
        payload = {"text": doc.text, **(doc.metadata or {})}
        payload.setdefault("doc_id", str(doc.id))
        point_id = _normalize_point_id(doc.id, collection=collection)
        points.append(qm.PointStruct(id=point_id, vector=vec, payload=payload))

    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %s documents into collection=%s", len(points), collection)


async def upsert_documents_llama_index(
    *,
    collection: str,
    documents: List[Document],
) -> None:
    """Upsert documents using LlamaIndex for ingestion, maintaining Qdrant compat.
    
    This converts our generic Document into LlamaIndex Document, and uses
    QdrantVectorStore and VectorStoreIndex to ingest them.
    This provides compatibility with LlamaIndex's ingestion pipeline features (like routers).
    """
    if not documents:
        return

    from app.services.llama_index_setup import setup_llama_index
    from llama_index.core import Document as LlamaDocument, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    
    # Ensure llama-index is configured
    setup_llama_index()
    
    client = get_client()

    vector_store = QdrantVectorStore(
        collection_name=collection,
        client=client
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    llama_docs = []
    for doc in documents:
        # Match our deterministic logic for Qdrant point IDs
        point_id = _normalize_point_id(doc.id, collection=collection)
        ldoc = LlamaDocument(
            text=doc.text,
            doc_id=str(point_id),
            metadata=(doc.metadata or {}).copy()
        )
        llama_docs.append(ldoc)
        
    # We use insertion vs. fresh index creation:
    _index = VectorStoreIndex.from_documents(
        llama_docs,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info("Upserted %s documents into collection=%s using LlamaIndex", len(llama_docs), collection)


async def search(
    *,
    collection: str,
    query: str,
    limit: int = 5,
    with_payload: bool = True,
) -> List[Dict[str, Any]]:
    client = get_client()

    query_vector = await embed_query(query)

    # qdrant-client API varies by version. Prefer the newest query API and
    # gracefully fall back when running against older clients.
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=with_payload,
            with_vectors=False,
        )
        results = response.points
    elif hasattr(client, "search_points"):
        # Older API
        results = client.search_points(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload,
            with_vectors=False,
        )
    else:
        # Oldest API
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload,
            with_vectors=False,
        )

    formatted: List[Dict[str, Any]] = []
    for r in results:
        formatted.append(
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
        )

    return formatted
