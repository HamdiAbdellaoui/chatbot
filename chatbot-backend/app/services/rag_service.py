"""LlamaIndex-based RAG retrieval service.

This step utilizes:
- QdrantVectorStore from llama-index-vector-stores-qdrant
- VectorStoreIndex from llama-index-core
- Formatting a context block to inject into the GPT prompt
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.config import settings
from app.services.qdrant_service import get_client
from app.services.llama_index_setup import setup_llama_index

logger = logging.getLogger(__name__)

# Initialize settings
setup_llama_index()

async def retrieve_context(
    *, 
    query: str, 
    collection: str | None = None, 
    limit: int | None = None,
    use_hierarchical: bool = False,
    docstore: SimpleDocumentStore | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve relevant context for a query from Qdrant using LlamaIndex.

    If use_hierarchical is True and docstore is provided, it uses AutoMergingRetriever
    to reconstruct parent chunk contexts. Otherwise, standard retrieval is used.

    Returns:
      (context_text, raw_hits)
    """
    collection_name = collection or settings.QDRANT_COLLECTION
    top_k = limit if limit is not None else settings.RAG_TOP_K
    max_chars = settings.RAG_MAX_CONTEXT_CHARS

    try:
        client = get_client()
        vector_store = QdrantVectorStore(
            collection_name=collection_name, 
            client=client, 
        )
        
        if use_hierarchical and docstore:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=docstore
            )
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            base_retriever = index.as_retriever(similarity_top_k=top_k)
            retriever = AutoMergingRetriever(
                vector_retriever=base_retriever,
                storage_context=storage_context,
                verbose=True
            )
        else:
            index = VectorStoreIndex.from_vector_store(vector_store)
            retriever = index.as_retriever(similarity_top_k=top_k)
        
        nodes = await retriever.aretrieve(query)
        
        hits: List[Dict[str, Any]] = []
        parts: List[str] = []
        used = 0

        for idx, node_with_score in enumerate(nodes, start=1):
            score = node_with_score.score or 0.0
            metadata = node_with_score.metadata or {}
            text = node_with_score.node.text or ""
            node_id = node_with_score.node.node_id
            
            source = metadata.get("source") or metadata.get("doc_id") or metadata.get("title") or "unknown"
            
            hits.append({
                "id": node_id,
                "score": score,
                "payload": {"text": text, **metadata}
            })
            
            snippet = text.strip().replace("\r\n", "\n")
            snippet = " ".join(snippet.split())
            
            block = f"[{idx}] source={source} score={score:.4f}\n{snippet}"
            
            if not block:
                continue

            if used + len(block) + 2 > max_chars:
                remaining = max(0, max_chars - used - 2)
                if remaining <= 0:
                    break
                block = block[:remaining]
            
            parts.append(block)
            used += len(block) + 2
            
            if used >= max_chars:
                break
                
        context = "\n\n".join(parts).strip()
        return context, hits
    except Exception:
        logger.exception("LlamaIndex RAG retrieval failed (collection=%s)", collection_name)
        return "", []
