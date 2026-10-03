import pytest

from app.config import settings
from app.services.confidence_service import (
    combine_confidence,
    is_business_decision,
    should_call_llm_confidence_signal,
)


def test_combine_confidence_with_all_signals_normal_case():
    score = combine_confidence(rag_score=0.9, llm_confidence=0.8, is_business_decision=False)
    # Equal-weight average of 0.9 and 0.8.
    assert score == pytest.approx(0.85)


def test_combine_confidence_penalizes_business_decision():
    baseline = combine_confidence(rag_score=0.9, llm_confidence=0.8, is_business_decision=False)
    penalized = combine_confidence(rag_score=0.9, llm_confidence=0.8, is_business_decision=True)

    assert penalized == pytest.approx(baseline - 0.15)
    assert penalized < baseline


def test_combine_confidence_penalty_is_floored_at_zero():
    score = combine_confidence(rag_score=0.05, llm_confidence=0.05, is_business_decision=True)
    assert score == 0.0


def test_combine_confidence_degrades_gracefully_with_one_missing_signal():
    only_rag = combine_confidence(rag_score=0.7, llm_confidence=None, is_business_decision=False)
    only_llm = combine_confidence(rag_score=None, llm_confidence=0.7, is_business_decision=False)

    assert only_rag == 0.7
    assert only_llm == 0.7


def test_combine_confidence_with_no_signal_returns_low_neutral_default():
    score = combine_confidence(rag_score=None, llm_confidence=None, is_business_decision=False)
    assert score == 0.3


def test_combine_confidence_result_is_always_clamped_to_0_1():
    score = combine_confidence(rag_score=1.0, llm_confidence=1.0, is_business_decision=False)
    assert 0.0 <= score <= 1.0


def test_is_business_decision_detects_refund_and_dispute_keywords():
    assert is_business_decision("Je veux un remboursement pour ma commande") is True
    assert is_business_decision("C'est un litige, je conteste cette facture") is True
    assert is_business_decision("I want to cancel my order please") is True


def test_is_business_decision_is_false_for_unrelated_message():
    assert is_business_decision("Bonjour, quels sont vos horaires d'ouverture ?") is False


def test_is_business_decision_handles_empty_text():
    assert is_business_decision("") is False


def test_should_call_llm_signal_skips_when_no_rag_hits(monkeypatch):
    monkeypatch.setattr(settings, "CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY", True)
    assert should_call_llm_confidence_signal(rag_score=None, hits_count=0) is False
    assert should_call_llm_confidence_signal(rag_score=0.9, hits_count=0) is False


def test_should_call_llm_signal_true_when_score_missing_but_hits_exist(monkeypatch):
    monkeypatch.setattr(settings, "CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY", True)
    assert should_call_llm_confidence_signal(rag_score=None, hits_count=3) is True


def test_should_call_llm_signal_skips_when_rag_score_clearly_high_or_low(monkeypatch):
    monkeypatch.setattr(settings, "CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY", True)
    monkeypatch.setattr(settings, "CONFIDENCE_HIGH_THRESHOLD", 0.75)
    monkeypatch.setattr(settings, "CONFIDENCE_LOW_THRESHOLD", 0.50)

    assert should_call_llm_confidence_signal(rag_score=0.9, hits_count=3) is False
    assert should_call_llm_confidence_signal(rag_score=0.2, hits_count=3) is False


def test_should_call_llm_signal_true_when_rag_score_ambiguous(monkeypatch):
    monkeypatch.setattr(settings, "CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY", True)
    monkeypatch.setattr(settings, "CONFIDENCE_HIGH_THRESHOLD", 0.75)
    monkeypatch.setattr(settings, "CONFIDENCE_LOW_THRESHOLD", 0.50)

    assert should_call_llm_confidence_signal(rag_score=0.6, hits_count=3) is True


def test_should_call_llm_signal_always_true_when_optimization_disabled(monkeypatch):
    monkeypatch.setattr(settings, "CONFIDENCE_LLM_SIGNAL_AMBIGUOUS_ONLY", False)
    assert should_call_llm_confidence_signal(rag_score=0.99, hits_count=0) is True
