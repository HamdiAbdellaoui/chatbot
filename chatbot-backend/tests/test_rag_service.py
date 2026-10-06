"""retrieve_context must use the sync retriever off the event loop.

get_client() returns a sync QdrantClient only; QdrantVectorStore.aquery()
raises "Async client is not initialized!" without an AsyncQdrantClient, so
aretrieve() can't be used. Failures must be logged, not swallowed silently.
"""

import asyncio
import logging
import threading
from types import SimpleNamespace

from app.services import rag_service


class _FakeRetriever:
    def __init__(self, nodes=None, error: Exception | None = None):
        self._nodes = nodes or []
        self._error = error
        self.retrieve_threads: list[int] = []

    def retrieve(self, query):
        self.retrieve_threads.append(threading.get_ident())
        if self._error:
            raise self._error
        return self._nodes

    async def aretrieve(self, query):
        raise ValueError("Async client is not initialized!")


def _node(text: str, score: float, source: str):
    return SimpleNamespace(score=score, metadata={"source": source}, node=SimpleNamespace(text=text, node_id=f"id-{source}"))


def _install(monkeypatch, retriever: _FakeRetriever):
    index = SimpleNamespace(as_retriever=lambda similarity_top_k: retriever)
    monkeypatch.setattr(rag_service, "get_client", lambda: object())
    monkeypatch.setattr(rag_service, "QdrantVectorStore", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(rag_service, "VectorStoreIndex", SimpleNamespace(from_vector_store=lambda store, **kw: index))


def test_retrieve_context_uses_sync_retrieve_in_a_worker_thread(monkeypatch):
    retriever = _FakeRetriever(nodes=[_node("Retour sous 14 jours.", 0.82, "retours.md")])
    _install(monkeypatch, retriever)

    async def scenario():
        result = await rag_service.retrieve_context(query="politique de retour", collection="c", limit=3)
        return result, threading.get_ident()

    (context, hits), loop_thread = asyncio.run(scenario())

    assert len(retriever.retrieve_threads) == 1
    assert retriever.retrieve_threads[0] != loop_thread
    assert hits == [{"id": "id-retours.md", "score": 0.82, "payload": {"text": "Retour sous 14 jours.", "source": "retours.md"}}]
    assert "Retour sous 14 jours." in context


def test_retrieve_context_logs_failures(monkeypatch, caplog):
    _install(monkeypatch, _FakeRetriever(error=RuntimeError("qdrant down")))

    with caplog.at_level(logging.ERROR, logger=rag_service.logger.name):
        context, hits = asyncio.run(rag_service.retrieve_context(query="q", collection="c"))

    assert (context, hits) == ("", [])
    records = [r for r in caplog.records if "RAG retrieval failed" in r.getMessage()]
    assert records and records[0].exc_info is not None
