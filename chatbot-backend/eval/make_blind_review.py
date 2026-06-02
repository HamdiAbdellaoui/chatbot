from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_pred_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "id" not in obj:
                raise SystemExit(f"Bad predictions JSONL at {path}:{line_no}")
            out[str(obj["id"])] = obj
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-a", required=True, help="Path to run A predictions.jsonl")
    p.add_argument("--run-b", required=True, help="Path to run B predictions.jsonl")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    random.seed(args.seed)

    a = _load_pred_jsonl(Path(args.run_a))
    b = _load_pred_jsonl(Path(args.run_b))

    ids = sorted(set(a.keys()) & set(b.keys()))
    if not ids:
        raise SystemExit("No overlapping ids between runs")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "answer_a",
        "answer_b",
        "winner",  # a|b|tie
        "score_a_helpfulness",  # 1-5
        "score_b_helpfulness",
        "score_a_groundedness",  # 1-5
        "score_b_groundedness",
        "score_a_clarity",  # 1-5
        "score_b_clarity",
        "notes",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for qid in ids:
            qa = a[qid]
            qb = b[qid]

            question = str(qa.get("question") or qb.get("question") or "")
            pred_a = str(qa.get("prediction") or "")
            pred_b = str(qb.get("prediction") or "")

            # Blind randomization per row.
            flip = random.choice([False, True])
            ans_a = pred_a if not flip else pred_b
            ans_b = pred_b if not flip else pred_a

            w.writerow(
                {
                    "id": qid,
                    "question": question,
                    "answer_a": ans_a,
                    "answer_b": ans_b,
                    "winner": "",
                    "score_a_helpfulness": "",
                    "score_b_helpfulness": "",
                    "score_a_groundedness": "",
                    "score_b_groundedness": "",
                    "score_a_clarity": "",
                    "score_b_clarity": "",
                    "notes": "",
                }
            )

    print(f"Wrote: {out_path}")
    print("Note: rows are randomized; do not try to infer which run is which from A/B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
