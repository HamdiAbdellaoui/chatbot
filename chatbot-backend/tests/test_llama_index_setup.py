import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import settings
from app.services import llama_index_setup

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Makes `import llama_index.embeddings.huggingface` raise ImportError, as in an
# image built from requirements.txt (where the package is not installed).
_BLOCK_HF = "import sys; sys.modules['llama_index.embeddings.huggingface'] = None; "


def _run(code: str, provider: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "EMBEDDINGS_PROVIDER": provider}
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_HF + code],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_app_imports_without_huggingface_package():
    proc = _run("import app.main; print('IMPORT_OK')", provider="openai")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "IMPORT_OK" in proc.stdout


def test_sentence_transformers_without_package_gives_explicit_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.huggingface", None)
    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "sentence-transformers")
    with pytest.raises(RuntimeError, match="llama-index-embeddings-huggingface"):
        llama_index_setup.setup_llama_index()


def test_local_provider_same_vector_in_both_paths(monkeypatch):
    import asyncio

    from llama_index.core import Settings as LlamaIndexSettings

    from app.services.embeddings_service import embed_query, embed_texts

    monkeypatch.setattr(settings, "EMBEDDINGS_PROVIDER", "local")
    monkeypatch.setattr(settings, "LOCAL_EMBEDDING_DIM", 64)
    previous = LlamaIndexSettings._embed_model
    try:
        llama_index_setup.setup_llama_index()
        model = LlamaIndexSettings.embed_model
        text = "Quelle est la garantie de la perceuse ?"

        service_query = asyncio.run(embed_query(text))
        service_text = asyncio.run(embed_texts([text]))[0]

        assert model.get_query_embedding(text) == service_query
        assert model.get_text_embedding(text) == service_text
        assert asyncio.run(model.aget_query_embedding(text)) == service_query
        assert any(v != 0.0 for v in service_query)
        assert len(service_query) == 64
    finally:
        LlamaIndexSettings._embed_model = previous


def test_sentence_transformers_gets_e5_prefixes(monkeypatch):
    import types

    captured = {}

    class FakeHF:
        def __init__(self, model_name, query_instruction=None, text_instruction=None):
            captured.update(model_name=model_name, query_instruction=query_instruction, text_instruction=text_instruction)

    fake_module = types.ModuleType("llama_index.embeddings.huggingface")
    fake_module.HuggingFaceEmbedding = FakeHF
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.huggingface", fake_module)
    monkeypatch.setattr(settings, "EMBEDDING_MODEL_NAME", "multilingual-e5-large")

    llama_index_setup._build_huggingface_embedding("intfloat/multilingual-e5-large")

    assert captured == {
        "model_name": "intfloat/multilingual-e5-large",
        "query_instruction": "query: ",
        "text_instruction": "passage: ",
    }
