from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> Optional[bool]:
    if gold is None or str(gold).strip() == "":
        return None
    return normalize_text(pred) == normalize_text(gold)


def token_jaccard_similarity(a: str, b: str) -> Optional[float]:
    if b is None or str(b).strip() == "":
        return None
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


# A refusal is a first-person statement that the assistant cannot/will not
# help. Apologies alone ("sorry", "désolé", "آسف") are NOT refusals: a correct
# policy answer often starts with "Sorry, used items can't be returned".
# Second-person forms ("you can't return...", "ma tnajamch", "ما تنجمش")
# describe a policy and must not match.
_REFUSAL_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(p, flags=re.IGNORECASE)
    for p in [
        # English
        r"\bi\s+(?:can\s*not|can't|can’t|won't|won’t|will not)\b",
        r"\bi(?:\s+am|'m|’m)\s+(?:unable to|not able to|not allowed to|not permitted to|not authorized to)\b",
        # French
        r"\bje ne (?:peux|pourrai|suis pas en mesure|suis pas autorisée?)\b",
        r"\bimpossible pour moi\b",
        # Tunisian Darija, Latin script: "ma nnejjemch", "manajamch", "ma n9adarch"
        r"\bma\s*n+[ae]?j+[ae]?m+ch\b",
        r"\bma\s*n[ae]?9[ae]?d[ae]?rch\b",
        r"\bma\s*n+جمش",  # mixed script, e.g. "ma nجمش"
        # Arabic script (Darija + MSA): "ما نجمش", "ما نقدرش", "لا أستطيع", "لا يمكنني"
        r"(?<![تي])ن+جمش",
        r"(?<![تي])نقدرش",
        r"لا\s*(?:أستطيع|استطيع|يمكنني|أقدر|نقدر)",
    ]
]


def is_refusal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _REFUSAL_PATTERNS)


def refusal_correctness(pred: str, expected_refusal: Optional[bool]) -> Optional[bool]:
    if expected_refusal is None:
        return None
    pred_is_refusal = is_refusal(pred)
    return bool(pred_is_refusal) == bool(expected_refusal)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector lengths differ")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(vec_a, vec_b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


@dataclass(frozen=True)
class MetricsResult:
    exact_match: Optional[bool]
    semantic_similarity: Optional[float]
    refusal_correct: Optional[bool]


def compute_metrics(
    *,
    prediction: str,
    expected_answer: Optional[str],
    expected_refusal: Optional[bool],
    semantic_similarity: Optional[float],
) -> MetricsResult:
    return MetricsResult(
        exact_match=exact_match(prediction, expected_answer or ""),
        semantic_similarity=semantic_similarity,
        refusal_correct=refusal_correctness(prediction, expected_refusal),
    )
