from pathlib import Path

from training.dataset_utils import (
    FineTuningRecord,
    conversation_to_examples,
    evaluate_conversation_quality,
    example_to_review_row,
    extract_valid_records,
    load_jsonl_records,
    parse_review_row,
    split_records,
    validate_dataset_records,
)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_dataset_records_reports_distributions(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"id":"1","messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Bonjour"},{"role":"assistant","content":"Bonjour"}],"metadata":{"store":"root4pro","topic":"shipping","resolution":"resolved","language":"fr"}}',
            '{"id":"2","messages":[{"role":"user","content":"شنوة"},{"role":"assistant","content":"مرحبا"}],"metadata":{"store":"root4pro","topic":"returns","resolution":"resolved","language":"ar"}}',
            '{"id":"3","messages":[{"role":"user","content":"bad"}],"metadata":{"store":"root4pro","topic":"shipping","resolution":"resolved","language":"fr"}}',
        ],
    )

    report = validate_dataset_records(load_jsonl_records(dataset_path))
    assert report.total_examples == 3
    assert report.valid_examples == 2
    assert report.invalid_examples == 1
    assert report.language_distribution == {"ar": 1, "fr": 1}
    assert report.topic_distribution == {"returns": 1, "shipping": 1}
    assert report.average_conversation_length == 2.5


def test_split_records_is_reproducible():
    records = [
        FineTuningRecord(
            id=str(i),
            messages=[{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}],
            metadata={"store": "root4pro", "topic": "shipping", "resolution": "resolved", "language": "fr"},
        )
        for i in range(10)
    ]

    split_a = split_records(records, seed=123)
    split_b = split_records(records, seed=123)

    assert [item.id for item in split_a.train] == [item.id for item in split_b.train]
    assert [item.id for item in split_a.validation] == [item.id for item in split_b.validation]
    assert [item.id for item in split_a.test] == [item.id for item in split_b.test]
    assert len(split_a.train) == 8
    assert len(split_a.validation) == 1
    assert len(split_a.test) == 1


def test_review_row_round_trip():
    conversation = {
        "id": 9,
        "status": "resolved",
        "labels": [{"title": "returns"}],
        "messages": [
            {"id": 1, "sender_type": "contact", "content_type": "text", "content": "Bonjour, je veux retourner ma commande.", "created_at": "2026-06-02T10:00:00Z"},
            {"id": 2, "sender_type": "agent", "content_type": "text", "content": "Bien sûr, voici la procédure.", "created_at": "2026-06-02T10:01:00Z"},
        ],
    }

    quality = evaluate_conversation_quality(conversation)
    assert quality.accepted is True

    example = conversation_to_examples(conversation, quality_result=quality)[0]
    review_row = example_to_review_row(example)
    messages, metadata = parse_review_row({"messages_json": review_row.messages_json, "metadata_json": review_row.metadata_json})

    assert len(messages) == 3
    assert metadata["topic"] == "returns"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_extract_valid_records_filters_invalid_rows(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset_path,
        [
            '{"id":"ok","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}],"metadata":{"store":"root4pro","topic":"support","resolution":"resolved"}}',
            '{"id":"bad","messages":[],"metadata":{"store":"root4pro","topic":"support","resolution":"resolved"}}',
        ],
    )

    records = load_jsonl_records(dataset_path)
    valid_records = extract_valid_records(records)
    assert len(valid_records) == 1
    assert valid_records[0].id == "ok"
