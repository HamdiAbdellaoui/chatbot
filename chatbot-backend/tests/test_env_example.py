"""Keep .env.example consistent with its own documented rules."""

import re
from pathlib import Path

from dotenv import dotenv_values

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"

_ASSIGNMENT_RE = re.compile(r"^(?:#\s*(?:Example[^:]*:\s*)?)?([A-Z][A-Z0-9_]*)=(.*)$")


def _assignments():
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT_RE.match(line.strip())
        if match:
            yield match.group(1), match.group(2)


def test_values_are_not_quoted_except_stores_json():
    for key, value in _assignments():
        if key == "STORES_JSON" and value:
            assert value.startswith("'") and value.endswith("'"), "STORES_JSON goes between single quotes"
        else:
            assert not value.startswith(('"', "'")), f"{key} must not be quoted"


def test_docker_compose_service_names_are_documented():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for hint in (
        "CHATWOOT_BASE_URL=http://chatwoot:3000",
        "QDRANT_URL=http://qdrant:6333",
        "REDIS_URL=redis://redis:6379/0",
        "@postgres:5432/",
    ):
        assert hint in text


def test_file_parses_with_dotenv():
    values = dotenv_values(ENV_EXAMPLE)
    assert values["QDRANT_URL"] == "http://localhost:6333"
    assert values["ESCALATION_LABEL"] == "human_handoff"
