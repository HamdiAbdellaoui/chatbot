# Evaluation Framework

This folder contains a lightweight evaluation toolkit to compare model versions and/or prompt changes.

## Dataset format (JSONL)
Each line is one JSON object.

Required:
- `id` (string)
- `question` (string)

Optional (recommended):
- `expected_answer` (string)
- `expected_refusal` (bool) — set true when the assistant *should refuse*
- `tags` (array of strings)

See `eval/datasets/qa_template.jsonl`.

## Run a batch eval
From `chatbot-backend/`:

- Chat-only (no RAG):
  - `python eval/run_batch_eval.py --dataset eval/datasets/qa_template.jsonl --mode chat`

- Override model:
  - `python eval/run_batch_eval.py --dataset eval/datasets/qa_template.jsonl --mode chat --model gpt-4o-mini`

- Override system prompt:
  - `python eval/run_batch_eval.py --dataset eval/datasets/qa_template.jsonl --mode chat --system-prompt-file eval/prompts/system_v1.txt`

Outputs are written under `eval/runs/<run_id>/` as:
- `predictions.jsonl`
- `predictions.csv`
- `metrics.json`

## Blind human review
Create a side-by-side comparison sheet for two runs:

- `python eval/make_blind_review.py --run-a eval/runs/<run_id_a>/predictions.jsonl --run-b eval/runs/<run_id_b>/predictions.jsonl --out eval/blind_review.csv`

Use `eval/human_rubric.md` as the scoring rubric.
