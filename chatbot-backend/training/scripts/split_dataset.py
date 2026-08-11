"""Split a validated fine-tuning dataset into train/validation/test sets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_utils import extract_valid_records, load_jsonl_records, records_to_jsonl_rows, split_records, validate_dataset_records, write_jsonl

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a fine-tuning dataset into train/validation/test sets.")
    parser.add_argument("--input", required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output-dir", default=str(TRAINING_ROOT / "datasets"), help="Directory to write the split files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--summary-json", default=None, help="Optional explicit path for the split summary JSON")
    parser.add_argument("--allow-invalid", action="store_true", help="Split only valid rows instead of failing on validation errors")
    args = parser.parse_args()

    records = load_jsonl_records(args.input)
    report = validate_dataset_records(records)
    if report.invalid_examples and not args.allow_invalid:
        print("Dataset contains invalid rows. Run validate_dataset.py first or use --allow-invalid.", file=sys.stderr)
        return 1

    valid_records = extract_valid_records(records)
    split = split_records(
        valid_records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"
    summary_path = Path(args.summary_json) if args.summary_json else (output_dir / "split_summary.json")

    write_jsonl(train_path, records_to_jsonl_rows(split.train))
    write_jsonl(validation_path, records_to_jsonl_rows(split.validation))
    write_jsonl(test_path, records_to_jsonl_rows(split.test))

    summary = {
        "source": str(Path(args.input)),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio},
        "input_examples": report.total_examples,
        "valid_examples": report.valid_examples,
        "invalid_examples": report.invalid_examples,
        "split_counts": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {train_path}")
    print(f"Wrote: {validation_path}")
    print(f"Wrote: {test_path}")
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
