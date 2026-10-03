"""Measure detect_language() accuracy against a small hand-labeled dataset.

Usage:
    python eval/run_language_eval.py
    python eval/run_language_eval.py --dataset eval/datasets/language_detection_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure imports work when running `python eval/run_language_eval.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.language_service import detect_language  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "language_detection_eval.jsonl"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return items


def run_eval(dataset_path: Path) -> None:
    examples = _load_jsonl(dataset_path)
    if not examples:
        print(f"No examples found in {dataset_path}")
        return

    total = 0
    correct = 0
    per_language_total: Dict[str, int] = defaultdict(int)
    per_language_correct: Dict[str, int] = defaultdict(int)
    mistakes: List[Dict[str, Any]] = []

    for example in examples:
        text = example["text"]
        expected = example["expected_language"]
        predicted = detect_language(text)

        total += 1
        per_language_total[expected] += 1

        if predicted == expected:
            correct += 1
            per_language_correct[expected] += 1
        else:
            mistakes.append({"text": text, "expected": expected, "predicted": predicted})

    accuracy = correct / total if total else 0.0

    print(f"Dataset: {dataset_path}")
    print(f"Examples: {total}")
    print(f"Overall accuracy: {accuracy:.1%} ({correct}/{total})")
    print()
    print("Per-language accuracy:")
    for language in sorted(per_language_total):
        lang_total = per_language_total[language]
        lang_correct = per_language_correct[language]
        lang_accuracy = lang_correct / lang_total if lang_total else 0.0
        print(f"  {language:8s} {lang_accuracy:6.1%} ({lang_correct}/{lang_total})")

    if mistakes:
        print()
        print(f"Misclassified ({len(mistakes)}):")
        for m in mistakes:
            print(f"  expected={m['expected']:8s} predicted={m['predicted']:8s} text={m['text']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detect_language() accuracy.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a JSONL file with {text, expected_language} per line.",
    )
    args = parser.parse_args()
    run_eval(args.dataset)


if __name__ == "__main__":
    main()
