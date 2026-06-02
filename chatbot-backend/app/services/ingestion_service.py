"""Advanced data ingestion pipeline using LlamaIndex.

This service handles taking raw documents, applying advanced chunking strategies 
(like Hierarchical Chunking), and indexing them into Qdrant. 

It preserves the flexibility to use custom pipelines or standard LlamaIndex ingestion.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from app.config import settings
from app.services.llama_index_setup import setup_llama_index
from app.services.qdrant_service import get_client, _normalize_point_id

from llama_index.core import Document as LlamaDocument, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

async def run_hierarchical_ingestion(
    *,
    collection: str,
    documents: List[Dict[str, Any]],  # Expects list of dicts: {"text": "...", "metadata": {}, "id": "..."}
    chunk_sizes: List[int] = [2048, 512, 128]
) -> None:
    """Ingest documents using Hierarchical Chunking.
    
    This splits large documents into a hierarchy of nodes (e.g. section -> paragraph -> sentence).
    The leaf nodes (smallest chunks) are what get embedded and indexed in Qdrant,
    while their relationships to larger parent nodes are retained for advanced retrieval 
    (like AutoMergingRetriever).
    """
    if not documents:
        return

    setup_llama_index()
    
    # 1. Convert input to LlamaIndex Documents
    llama_docs = []
    for doc in documents:
        doc_id = _normalize_point_id(doc.get("id"), collection=collection)
        ldoc = LlamaDocument(
            text=doc.get("text", ""),
            doc_id=str(doc_id),
            metadata=doc.get("metadata", {})
        )
        llama_docs.append(ldoc)

    # 2. Setup Hierarchical Node Parser
    # This creates a tree of nodes with varying chunk sizes
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes
    )
    
    nodes = node_parser.get_nodes_from_documents(llama_docs)
    leaf_nodes = get_leaf_nodes(nodes)
    
    logger.info(
        "Hierarchical chunking created %d total nodes, %d leaf nodes from %d documents.", 
        len(nodes), len(leaf_nodes), len(llama_docs)
    )

    # 3. Setup DocStore to hold parent-child relationships
    # In a full production setup, this docstore should be persistent (e.g. MongoDB/Redis).
    # For now we use the SimpleDocumentStore and will index the leaf nodes to Qdrant.
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # 4. Setup Qdrant Vector Store for the leaf nodes
    client = get_client()
    vector_store = QdrantVectorStore(
        collection_name=collection,
        client=client
    )
    
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store
    )
    
    # 5. Index only the leaf nodes into Qdrant
    _index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info("Hierarchical ingestion into Qdrant collection='%s' complete.", collection)
