from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure imports work when running `python eval/run_batch_eval.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.llm_service import generate_reply, generate_grounded_reply  # noqa: E402
from app.services.rag_service import retrieve_context  # noqa: E402
from app.services.store_context_service import resolve_store  # noqa: E402
from eval.metrics import MetricsResult, token_jaccard_similarity  # noqa: E402


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {e}")
            if not isinstance(obj, dict):
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: expected object")
            items.append(obj)
    return items


def _read_text_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    return p.read_text(encoding="utf-8")


async def _openai_embed(texts: List[str], *, model: str) -> List[List[float]]:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    resp = await client.embeddings.create(model=model, input=texts)
    # Keep original order by index.
    data = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in data]


def _try_st_embed(texts: List[str]) -> Optional[List[List[float]]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    model_name = settings.EMBEDDING_MODEL_NAME or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=False)
    # `emb` is typically a numpy array; convert to lists.
    return [list(map(float, row)) for row in emb]


def _cosine(a: List[float], b: List[float]) -> float:
    # local cosine to avoid importing numpy
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = (na ** 0.5) * (nb ** 0.5)
    return 0.0 if denom == 0.0 else dot / denom


async def _semantic_similarity(
    *,
    preds: List[str],
    golds: List[str],
    backend: str,
) -> Tuple[List[Optional[float]], str]:
    """Compute similarity per row.

    backends:
      - "token": Jaccard over normalized tokens (free, heuristic)
      - "st": sentence-transformers cosine (local, requires package/model)
      - "openai": OpenAI embeddings cosine (paid, requires OPENAI_API_KEY)
      - "none": disabled

    Returns: (scores, backend_used)
    """

    if backend == "none":
        return [None for _ in preds], "none"

    if backend == "token":
        out: List[Optional[float]] = []
        for p, g in zip(preds, golds):
            out.append(token_jaccard_similarity(p, g))
        return out, "token"

    if backend == "st":
        emb = _try_st_embed([*preds, *golds])
        if emb is None:
            # fall back
            return await _semantic_similarity(preds=preds, golds=golds, backend="token")
        pred_emb = emb[: len(preds)]
        gold_emb = emb[len(preds) :]
        scores: List[Optional[float]] = []
        for i, g in enumerate(golds):
            if not (g or "").strip():
                scores.append(None)
                continue
            scores.append(_cosine(pred_emb[i], gold_emb[i]))
        return scores, "st"

    if backend == "openai":
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
            return await _semantic_similarity(preds=preds, golds=golds, backend="token")

        # Embed only rows that have a gold answer.
        idxs = [i for i, g in enumerate(golds) if (g or "").strip()]
        if not idxs:
            return [None for _ in preds], "openai"

        pred_sel = [preds[i] for i in idxs]
        gold_sel = [golds[i] for i in idxs]
        vectors = await _openai_embed([*pred_sel, *gold_sel], model=settings.OPENAI_EMBEDDING_MODEL)
        pred_vecs = vectors[: len(pred_sel)]
        gold_vecs = vectors[len(pred_sel) :]

        scores_all: List[Optional[float]] = [None for _ in preds]
        for j, i in enumerate(idxs):
            scores_all[i] = _cosine(pred_vecs[j], gold_vecs[j])
        return scores_all, "openai"

    # auto
    if backend == "auto":
        st = _try_st_embed(["test"])
        if st is not None:
            return await _semantic_similarity(preds=preds, golds=golds, backend="st")
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY not in {"your_openai_api_key", "sk-..."}:
            return await _semantic_similarity(preds=preds, golds=golds, backend="openai")
        return await _semantic_similarity(preds=preds, golds=golds, backend="token")

    raise SystemExit(f"Unknown semantic backend: {backend}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--mode", choices=["chat", "rag"], default="chat")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for this run")
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to eval/runs/<run_id>)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--semantic", choices=["auto", "token", "st", "openai", "none"], default="auto")
    parser.add_argument("--inbox-id", type=int, default=None, help="Optional: resolve store by Chatwoot inbox id")
    parser.add_argument("--inbox-name", default=None, help="Optional: resolve store by Chatwoot inbox name")
    parser.add_argument("--rag-top-k", type=int, default=None, help="Optional: override RAG_TOP_K for this run")
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_path = Path(args.dataset)
    items = _load_jsonl(dataset_path)
    if not items:
        raise SystemExit("Dataset is empty")

    system_prompt = _read_text_file(args.system_prompt_file)

    run_id = f"{_utc_ts()}_{random.randint(1000, 9999)}"
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "eval" / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow per-run model override without editing .env
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    predictions: List[Dict[str, Any]] = []

    store_context = resolve_store(inbox_id=args.inbox_id, inbox_name=args.inbox_name)

    for obj in items:
        qid = str(obj.get("id") or "")
        question = str(obj.get("question") or "").strip()
        expected_answer = obj.get("expected_answer")
        expected_answer = str(expected_answer) if expected_answer is not None else ""
        expected_refusal = obj.get("expected_refusal")
        if expected_refusal is not None:
            expected_refusal = bool(expected_refusal)

        if not qid or not question:
            raise SystemExit(f"Dataset row missing id/question: {obj}")

        if args.mode == "chat":
            pred = await generate_reply(user_message=question, system_prompt=system_prompt)
        else:
            context_text, _hits = await retrieve_context(
                query=question,
                collection=store_context.qdrant_collection,
                limit=args.rag_top_k,
            )
            pred = await generate_grounded_reply(
                user_message=question,
                context=context_text,
                system_prompt=system_prompt,
                store_context=store_context,
            )

        predictions.append(
            {
                "id": qid,
                "question": question,
                "prediction": pred,
                "expected_answer": expected_answer,
                "expected_refusal": expected_refusal,
                "tags": obj.get("tags") or [],
            }
        )

    preds = [p["prediction"] for p in predictions]
    golds = [p.get("expected_answer") or "" for p in predictions]

    semantic_scores, semantic_backend = await _semantic_similarity(preds=preds, golds=golds, backend=args.semantic)

    # Per-row metrics
    from eval.metrics import compute_metrics  # local import to avoid circular

    row_metrics: List[MetricsResult] = []
    for i, row in enumerate(predictions):
        m = compute_metrics(
            prediction=row["prediction"],
            expected_answer=(row.get("expected_answer") or ""),
            expected_refusal=row.get("expected_refusal"),
            semantic_similarity=semantic_scores[i],
        )
        row_metrics.append(m)
        row["metrics"] = asdict(m)

    # Aggregate metrics
    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in vals if v is not None]
        if not xs:
            return None
        return sum(xs) / len(xs)

    def _acc_bools(vals: List[Optional[bool]]) -> Optional[float]:
        xs = [v for v in vals if v is not None]
        if not xs:
            return None
        return sum(1 for v in xs if v) / len(xs)

    exacts = [m.exact_match for m in row_metrics]
    refusals = [m.refusal_correct for m in row_metrics]

    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "mode": args.mode,
        "model": os.environ.get("OPENAI_MODEL") or settings.OPENAI_MODEL,
        "semantic_backend": semantic_backend,
        "counts": {"total": len(predictions)},
        "metrics": {
            "exact_match_accuracy": _acc_bools(exacts),
            "refusal_accuracy": _acc_bools(refusals),
            "semantic_similarity_mean": _mean(semantic_scores),
        },
    }

    # Write outputs
    pred_jsonl = out_dir / "predictions.jsonl"
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "expected_answer",
                "expected_refusal",
                "prediction",
                "exact_match",
                "semantic_similarity",
                "refusal_correct",
                "tags",
            ],
        )
        w.writeheader()
        for row in predictions:
            m = row.get("metrics") or {}
            w.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected_answer": row.get("expected_answer") or "",
                    "expected_refusal": row.get("expected_refusal"),
                    "prediction": row.get("prediction") or "",
                    "exact_match": m.get("exact_match"),
                    "semantic_similarity": m.get("semantic_similarity"),
                    "refusal_correct": m.get("refusal_correct"),
                    "tags": json.dumps(row.get("tags") or [], ensure_ascii=False),
                }
            )

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {pred_jsonl}")
    print(f"Wrote: {pred_csv}")
    print(f"Wrote: {metrics_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
