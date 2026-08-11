"""Benchmark base Qwen 2.5 versus a future fine-tuned model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.benchmark_utils import (
    BenchmarkRowResult,
    average_response_length,
    build_benchmark_prompt,
    language_consistency_score,
    policy_compliance_score,
)
from training.dataset_utils import extract_valid_records, load_jsonl_records, validate_dataset_records
from training.model_utils import generate_text, load_model_bundle, load_system_prompt, resolve_local_model_path


def _run_model(bundle, messages: list[dict[str, str]], *, max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, float]:
    start = time.perf_counter()
    response = generate_text(bundle, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return response, latency_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark base vs fine-tuned Qwen 2.5 models locally.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset (typically test.jsonl)")
    parser.add_argument("--base-model-path", required=True, help="Local path to the base Qwen model")
    parser.add_argument("--finetuned-model-path", default=None, help="Local path to a merged fine-tuned model")
    parser.add_argument("--finetuned-adapter-path", default=None, help="Optional LoRA adapter path for the fine-tuned model")
    parser.add_argument("--system-prompt-file", default=str(TRAINING_ROOT / "prompts" / "system_prompt.txt"), help="System prompt file")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark outputs")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of examples")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    records = load_jsonl_records(args.dataset)
    report = validate_dataset_records(records)
    if report.invalid_examples:
        raise SystemExit("Benchmark dataset contains malformed rows. Validate and split the dataset first.")

    valid_records = extract_valid_records(records)
    if args.max_samples is not None:
        valid_records = valid_records[: args.max_samples]
    if not valid_records:
        raise SystemExit("No valid benchmark examples found.")

    base_bundle = load_model_bundle(model_path=resolve_local_model_path(args.base_model_path, env_name="QWEN_BASE_MODEL_PATH"))
    if not args.finetuned_model_path:
        raise SystemExit("Provide --finetuned-model-path (or a merged fine-tuned model directory) before benchmarking.")
    fine_bundle = load_model_bundle(
        model_path=resolve_local_model_path(args.finetuned_model_path, env_name="QWEN_FINETUNED_MODEL_PATH"),
        adapter_path=args.finetuned_adapter_path,
    )

    system_prompt = load_system_prompt(args.system_prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "benchmark_rows.csv"
    jsonl_path = output_dir / "benchmark_rows.jsonl"
    summary_path = output_dir / "benchmark_summary.json"

    row_results: list[BenchmarkRowResult] = []
    jsonl_rows: list[dict[str, Any]] = []

    for record in valid_records:
        example = {"messages": record.messages, "metadata": record.metadata, "id": record.id}
        prompt_messages, prompt_text, reference = build_benchmark_prompt(example)
        if not prompt_messages:
            continue
        if prompt_messages[0]["role"] != "system":
            prompt_messages = [{"role": "system", "content": system_prompt}] + prompt_messages

        base_response, base_latency_ms = _run_model(base_bundle, prompt_messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        fine_response, fine_latency_ms = _run_model(fine_bundle, prompt_messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

        expected_refusal = record.metadata.get("expected_refusal")
        if not isinstance(expected_refusal, bool):
            expected_refusal = None

        row = BenchmarkRowResult(
            example_id=record.id,
            prompt=prompt_text,
            reference=reference,
            base_response=base_response,
            finetuned_response=fine_response,
            base_latency_ms=base_latency_ms,
            finetuned_latency_ms=fine_latency_ms,
            base_avg_length=average_response_length(base_response),
            finetuned_avg_length=average_response_length(fine_response),
            base_language_consistency=language_consistency_score(prompt_text, base_response),
            finetuned_language_consistency=language_consistency_score(prompt_text, fine_response),
            base_policy_compliance=policy_compliance_score(prompt=prompt_text, response=base_response, expected_refusal=expected_refusal),
            finetuned_policy_compliance=policy_compliance_score(prompt=prompt_text, response=fine_response, expected_refusal=expected_refusal),
        )
        row_results.append(row)
        jsonl_rows.append(asdict(row))

    if not row_results:
        raise SystemExit("Benchmark produced no rows.")

    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "prompt",
                "reference",
                "base_response",
                "finetuned_response",
                "base_latency_ms",
                "finetuned_latency_ms",
                "base_avg_length",
                "finetuned_avg_length",
                "base_language_consistency",
                "finetuned_language_consistency",
                "base_policy_compliance",
                "finetuned_policy_compliance",
                "human_winner",
                "human_notes",
            ],
        )
        writer.writeheader()
        for row in row_results:
            writer.writerow({**asdict(row), "human_winner": "", "human_notes": ""})

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_examples": len(row_results),
        "base_avg_response_length": mean(row.base_avg_length for row in row_results),
        "finetuned_avg_response_length": mean(row.finetuned_avg_length for row in row_results),
        "base_avg_latency_ms": mean(row.base_latency_ms for row in row_results),
        "finetuned_avg_latency_ms": mean(row.finetuned_latency_ms for row in row_results),
        "base_language_consistency": mean(row.base_language_consistency for row in row_results),
        "finetuned_language_consistency": mean(row.finetuned_language_consistency for row in row_results),
        "base_policy_compliance": mean(row.base_policy_compliance for row in row_results),
        "finetuned_policy_compliance": mean(row.finetuned_policy_compliance for row in row_results),
        "outputs": {"rows_csv": str(rows_path), "rows_jsonl": str(jsonl_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote benchmark CSV: {rows_path}")
    print(f"Wrote benchmark JSONL: {jsonl_path}")
    print(f"Wrote benchmark summary: {summary_path}")
    print(f"Examples benchmarked: {len(row_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
