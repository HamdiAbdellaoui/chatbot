"""Validate a fine-tuning JSONL dataset locally."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_utils import load_jsonl_records, validate_dataset_records

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a fine-tuning JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to the JSONL dataset")
    parser.add_argument("--report-json", default=str(TRAINING_ROOT / "datasets" / "dataset_validation.json"), help="Where to write the validation report")
    parser.add_argument("--issues-csv", default=None, help="Optional path for detailed validation issues")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any malformed rows are found")
    args = parser.parse_args()

    records = load_jsonl_records(args.input)
    report = validate_dataset_records(records)

    report_payload = {
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

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.issues_csv:
        issues_path = Path(args.issues_csv)
        issues_path.parent.mkdir(parents=True, exist_ok=True)
        with issues_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["line_number", "example_id", "message"])
            writer.writeheader()
            for issue in report.issues:
                writer.writerow({"line_number": issue.line_number, "example_id": issue.example_id or "", "message": issue.message})

    print(f"Validated examples: {report.valid_examples}/{report.total_examples}")
    print(f"Invalid examples: {report.invalid_examples}")
    print(f"Average conversation length: {report.average_conversation_length:.2f}")
    print(f"Language distribution: {report.language_distribution}")
    print(f"Topic distribution: {report.topic_distribution}")
    print(f"Report written to: {report_path}")

    if args.strict and report.invalid_examples:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
