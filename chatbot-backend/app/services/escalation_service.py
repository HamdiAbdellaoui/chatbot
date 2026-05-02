"""Human escalation detection logic.

Design goals:
- Simple keyword-based detection (extensible)
- Optional low-confidence escalation based on retrieval score

This module is intentionally pure (no network calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: str | None = None


def _iter_keywords() -> list[str]:
    raw = settings.ESCALATION_KEYWORDS or ""
    parts = [p.strip().lower() for p in raw.split(",")]
    return [p for p in parts if p]


def _compile_pattern() -> re.Pattern[str]:
    # Match keyword as a substring, but require word boundaries when keyword is a single word.
    # For multi-word phrases, we just do a normalized substring match.
    keywords = _iter_keywords()
    if not keywords:
        return re.compile(r"$a")

    single_words = [re.escape(k) for k in keywords if " " not in k]
    chunks: list[str] = []
    if single_words:
        chunks.append(r"\b(?:" + "|".join(single_words) + r")\b")
    # phrases are handled separately with a simple substring check
    pattern = r"(?:" + "|".join(chunks) + r")" if chunks else r"$a"
    return re.compile(pattern, flags=re.IGNORECASE)


_WORD_RE = _compile_pattern()


def _norm_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def detect_escalation_request(user_message: str) -> EscalationDecision:
    """Keyword-based escalation detection."""
    msg = (user_message or "").strip()
    if not msg:
        return EscalationDecision(False)

    normalized = _norm_text(msg)

    # Fast substring check for phrases
    for phrase in _iter_keywords():
        if " " in phrase and phrase in normalized:
            return EscalationDecision(True, reason=f"keyword_phrase:{phrase}")

    if _WORD_RE.search(msg):
        return EscalationDecision(True, reason="keyword_word")

    return EscalationDecision(False)


def detect_low_confidence(*, top_score: float | None, hits_count: int) -> EscalationDecision:
    """Optional low-confidence escalation.

    This is OFF by default. When enabled, we escalate if:
    - no hits were retrieved
    - or top_score is below a threshold
    """

    if not settings.ESCALATION_LOW_CONFIDENCE_ENABLED:
        return EscalationDecision(False)

    if hits_count <= 0:
        return EscalationDecision(True, reason="low_confidence:no_hits")

    if top_score is None:
        return EscalationDecision(True, reason="low_confidence:no_score")

    if top_score < settings.ESCALATION_MIN_TOP_SCORE:
        return EscalationDecision(True, reason=f"low_confidence:top_score<{settings.ESCALATION_MIN_TOP_SCORE}")

    return EscalationDecision(False)
