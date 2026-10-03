"""Combine 3 confidence signals into a single escalation-ready score.

Signals:
1. rag_score      - Qdrant retrieval top score (already computed in chatbot_service.py).
2. llm_confidence  - the LLM's own self-assessment of how grounded its answer is in
                      the given context (see llm_service.estimate_context_grounding).
3. is_business_decision - rule-based flag: does the message look like a business
                      decision (refund, dispute, order cancellation, ...) that
                      deserves extra caution regardless of how confident the
                      other two signals are?

This module is intentionally pure (no network calls, no I/O) so it stays easy
to unit test; the network-dependent signal (llm_confidence) is computed
elsewhere and simply passed in.
"""

from __future__ import annotations

from typing import List, Tuple

from app.config import settings
from app.services.escalation_service import compile_keyword_matcher

# Equal weighting: neither the RAG retrieval score nor the LLM's own
# self-assessment is inherently more trustworthy than the other, so when both
# are available they contribute equally to the average. When only one signal
# is available, it is used alone (its weight becomes 1.0 after normalization).
RAG_SCORE_WEIGHT = 0.5
LLM_CONFIDENCE_WEIGHT = 0.5

# Flat penalty subtracted (never below 0.0) when the message looks like a
# business decision. Chosen to be large enough to push a borderline-high score
# (e.g. 0.80) down past the CONFIDENCE_HIGH_THRESHOLD default (0.75) into the
# "uncertainty signal" tier, without being so large that a single business
# keyword always forces a full escalation on its own.
BUSINESS_DECISION_PENALTY = 0.15

# Returned when none of the 3 signals are usable at all (e.g. RAG retrieval
# failed AND the LLM self-assessment call failed/timed out). Deliberately low
# ("uncertain" rather than "confident") so a total signal outage degrades
# towards caution rather than towards silently trusting the bot's answer.
NEUTRAL_CONFIDENCE_WHEN_NO_SIGNAL = 0.3


def is_business_decision(text: str) -> bool:
    """Rule-based check: does this message involve a business decision
    (refund, dispute, order cancellation, exception, ...) that should be
    handled with extra caution? Uses the same keyword-matching model as
    escalation_service.detect_escalation_request.
    """
    matcher = compile_keyword_matcher(settings.BUSINESS_DECISION_KEYWORDS)
    return matcher(text)


def should_call_llm_confidence_signal(*, rag_score: float | None, hits_count: int) -> bool:
    """Decide whether the (extra-latency) LLM self-assessment call is worth making.

    The LLM self-assessment is a second round-trip on top of the main reply
    call, so it should not be made unconditionally. When
    CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY is enabled (default), it is only
    called when the RAG score alone is genuinely ambiguous:
    - hits_count <= 0: no RAG hits at all -> detect_low_confidence already
      escalates regardless of the combined score, so the signal wouldn't
      change the outcome. Skip it.
    - rag_score is None (hits exist but no usable score): genuinely
      ambiguous, worth asking the LLM.
    - rag_score clearly high (>= CONFIDENCE_HIGH_THRESHOLD) or clearly low
      (< CONFIDENCE_LOW_THRESHOLD): the RAG signal alone is already decisive.
      Skip it.
    - Otherwise (CONFIDENCE_LOW_THRESHOLD <= rag_score < CONFIDENCE_HIGH_THRESHOLD):
      genuinely ambiguous, worth asking the LLM.
    """
    if not settings.CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY:
        return True

    if hits_count <= 0:
        return False

    if rag_score is None:
        return True

    return settings.CONFIDENCE_LOW_THRESHOLD <= rag_score < settings.CONFIDENCE_HIGH_THRESHOLD


def combine_confidence(
    *,
    rag_score: float | None,
    llm_confidence: float | None,
    is_business_decision: bool,
) -> float:
    """Combine the 3 signals into a single 0.0-1.0 confidence score.

    - Weighted average of whichever of (rag_score, llm_confidence) are
      available (RAG_SCORE_WEIGHT / LLM_CONFIDENCE_WEIGHT, both 0.5); a
      missing signal is dropped from the average rather than counted as 0.
    - If is_business_decision is True, subtract BUSINESS_DECISION_PENALTY
      (0.15), floored at 0.0.
    - If neither rag_score nor llm_confidence is available, returns
      NEUTRAL_CONFIDENCE_WHEN_NO_SIGNAL (0.3) instead of raising.
    - Result is always clamped to [0.0, 1.0].
    """
    weighted_signals: List[Tuple[float, float]] = []
    if rag_score is not None:
        weighted_signals.append((rag_score, RAG_SCORE_WEIGHT))
    if llm_confidence is not None:
        weighted_signals.append((llm_confidence, LLM_CONFIDENCE_WEIGHT))

    if not weighted_signals:
        score = NEUTRAL_CONFIDENCE_WHEN_NO_SIGNAL
    else:
        total_weight = sum(weight for _, weight in weighted_signals)
        score = sum(value * weight for value, weight in weighted_signals) / total_weight

    if is_business_decision:
        score = max(0.0, score - BUSINESS_DECISION_PENALTY)

    return max(0.0, min(1.0, score))
