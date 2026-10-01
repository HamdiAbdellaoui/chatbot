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
