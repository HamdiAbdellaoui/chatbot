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


def compile_keyword_matcher(keywords_csv: str):
    """Build a case-insensitive keyword matcher from a comma-separated list.

    Same matching model as detect_escalation_request: word-boundary match for
    single-word keywords, substring match for multi-word phrases. Shared so
    other rule-based classifiers (e.g. confidence_service.is_business_decision)
    don't have to reimplement this.
    """
    keywords = [p.strip().lower() for p in (keywords_csv or "").split(",") if p.strip()]
    single_words = [re.escape(k) for k in keywords if " " not in k]
    phrases = [k for k in keywords if " " in k]
    word_pattern = re.compile(r"\b(?:" + "|".join(single_words) + r")\b", re.IGNORECASE) if single_words else None

    def matches(text: str) -> bool:
        msg = (text or "").strip()
        if not msg:
            return False
        normalized = _norm_text(msg)
        if any(phrase in normalized for phrase in phrases):
            return True
        return bool(word_pattern and word_pattern.search(msg))

    return matches


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


def detect_low_confidence(*, confidence_score: float | None, hits_count: int) -> EscalationDecision:
    """Optional low-confidence escalation based on the combined 3-signal confidence score.

    See app/services/confidence_service.py for how confidence_score is computed
    (RAG retrieval score + LLM self-assessment + business-decision rule).

    This is OFF by default (ESCALATION_LOW_CONFIDENCE_ENABLED). When enabled,
    3 tiers apply (CONFIDENCE_HIGH_THRESHOLD / CONFIDENCE_LOW_THRESHOLD):
    - score >= HIGH: normal, no escalation.
    - LOW <= score < HIGH: uncertainty signal only (flagged for review via the
      returned reason, but should_escalate stays False).
    - score < LOW (or no hits / no score at all): full escalation.
    """

    if not settings.ESCALATION_LOW_CONFIDENCE_ENABLED:
        return EscalationDecision(False)

    if hits_count <= 0:
        return EscalationDecision(True, reason="low_confidence:escalate:no_hits")

    if confidence_score is None:
        return EscalationDecision(True, reason="low_confidence:escalate:no_score")

    if confidence_score < settings.CONFIDENCE_LOW_THRESHOLD:
        return EscalationDecision(True, reason=f"low_confidence:escalate:score<{settings.CONFIDENCE_LOW_THRESHOLD}")

    if confidence_score < settings.CONFIDENCE_HIGH_THRESHOLD:
        return EscalationDecision(False, reason=f"low_confidence:signal:score<{settings.CONFIDENCE_HIGH_THRESHOLD}")

    return EscalationDecision(False)
