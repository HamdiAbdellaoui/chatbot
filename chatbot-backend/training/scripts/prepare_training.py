"""Prepare fine-tuning data from a review CSV or a compiled JSONL dataset.

This command can:
- compile approved rows from a review CSV into a training JSONL
- validate the resulting JSONL
- split the dataset into train/validation/test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_utils import (
    extract_valid_records,
    load_jsonl_records,
    parse_review_row,
    records_to_jsonl_rows,
    split_records,
    validate_dataset_records,
    write_jsonl,
)

logger = logging.getLogger(__name__)


def _load_review_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _compile_review_csv(review_csv: Path, out_jsonl: Path, allowed_statuses: set[str]) -> dict[str, Any]:
    rows = _load_review_rows(review_csv)
    compiled_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("review_status") or row.get("status") or "pending").strip().lower()
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in allowed_statuses:
            continue
        messages, metadata = parse_review_row(row)
        if not messages:
            continue
        compiled_rows.append({"id": str(row.get("id") or row.get("example_id") or ""), "messages": messages, "metadata": metadata})

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, compiled_rows)
    return {"review_status_counts": status_counts, "compiled_examples": len(compiled_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a fine-tuning dataset locally.")
    parser.add_argument("--review-csv", default=None, help="Optional review CSV containing approved/rejected rows")
    parser.add_argument("--input-jsonl", default=None, help="Optional already compiled JSONL dataset")
    parser.add_argument("--output-dir", default=str(TRAINING_ROOT / "datasets"), help="Output directory for the prepared files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--allowed-statuses", nargs="*", default=["approved"], help="Review statuses included when compiling from CSV")
    args = parser.parse_args()

    if not args.review_csv and not args.input_jsonl:
        raise SystemExit("Provide either --review-csv or --input-jsonl.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_jsonl = output_dir / "training_dataset.jsonl"
    summary_json = output_dir / "training_preparation_summary.json"
    validation_json = output_dir / "dataset_validation.json"
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    prep_summary: dict[str, Any] = {}
    if args.review_csv:
        prep_summary = _compile_review_csv(Path(args.review_csv), compiled_jsonl, {status.strip().lower() for status in args.allowed_statuses if status.strip()})
    else:
        compiled_jsonl = Path(args.input_jsonl)

    records = load_jsonl_records(compiled_jsonl)
    report = validate_dataset_records(records)
    validation_payload = {
        "total_examples": report.total_examples,
        "valid_examples": report.valid_examples,
        "invalid_examples": report.invalid_examples,
        "language_distribution": report.language_distribution,
        "topic_distribution": report.topic_distribution,
        "average_conversation_length": report.average_conversation_length,
        "issues": [
            {"line_number": issue.line_number, "example_id": issue.example_id, "message": issue.message}
            for issue in report.issues
        ],
    }
    validation_json.write_text(json.dumps(validation_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if report.invalid_examples:
        raise SystemExit("Compiled dataset contains invalid rows. Fix the review CSV or input JSONL before continuing.")

    valid_records = extract_valid_records(records)
    split = split_records(
        valid_records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    write_jsonl(train_path, records_to_jsonl_rows(split.train))
    write_jsonl(validation_path, records_to_jsonl_rows(split.validation))
    write_jsonl(test_path, records_to_jsonl_rows(split.test))

    summary = {
        "source_review_csv": str(Path(args.review_csv)) if args.review_csv else None,
        "source_jsonl": str(compiled_jsonl),
        "preparation": prep_summary,
        "validation": validation_payload,
        "split_counts": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote compiled dataset: {compiled_jsonl}")
    print(f"Wrote validation report: {validation_json}")
    print(f"Wrote train split: {train_path}")
    print(f"Wrote validation split: {validation_path}")
    print(f"Wrote test split: {test_path}")
    print(f"Wrote summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
