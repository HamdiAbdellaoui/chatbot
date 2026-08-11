"""Build a training JSONL from a manual review CSV.

Workflow:
1) export_finetuning_dataset.py creates fine_tuning_review.csv with review_status=pending
2) a human updates review_status to approved/rejected/needs_edit
3) this script extracts approved rows and writes training_dataset.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from training.dataset_utils import parse_review_row


APPROVED_STATUSES = {"approved"}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a training JSONL from a manual review CSV.")
    parser.add_argument("--review-csv", required=True, help="Path to the review CSV produced by export_finetuning_dataset.py")
    parser.add_argument("--out-jsonl", default=str(BACKEND_ROOT / "artifacts" / "fine_tuning" / "training_dataset.jsonl"), help="Output JSONL path")
    parser.add_argument("--summary-json", default=str(BACKEND_ROOT / "artifacts" / "fine_tuning" / "training_dataset_summary.json"), help="Output summary JSON path")
    parser.add_argument("--allowed-statuses", nargs="*", default=["approved"], help="Review statuses to include in the output")
    args = parser.parse_args()

    review_path = Path(args.review_csv)
    rows = _load_rows(review_path)

    allowed = {str(status).strip().lower() for status in args.allowed_statuses if str(status).strip()}
    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for row in rows:
        status = str(row.get("review_status") or row.get("status") or "pending").strip().lower()
        status_counts[status] += 1
        if status not in allowed:
            skipped_counts[status] += 1
            continue

        messages, metadata = parse_review_row(row)
        if not messages:
            skipped_counts["invalid_messages"] += 1
            continue

        example_id = str(row.get("id") or row.get("example_id") or "")
        output_rows.append(
            {
                "id": example_id,
                "messages": messages,
                "metadata": metadata,
            }
        )

    if not output_rows:
        raise SystemExit("No approved review rows were found.")

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_review_csv": str(review_path),
        "training_jsonl": str(out_jsonl),
        "approved_examples": len(output_rows),
        "review_status_counts": dict(status_counts),
        "skipped_counts": dict(skipped_counts),
    }
    out_summary = Path(args.summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote training JSONL: {out_jsonl}")
    print(f"Wrote summary JSON: {out_summary}")
    print(f"Approved examples: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
