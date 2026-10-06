"""Detect messages that are only a greeting ("bonjour", "aslema", "السلام عليكم"...).

Such messages carry no question: sending them through RAG yields a low score
and, with ESCALATION_LOW_CONFIDENCE_ENABLED, a needless escalation. The
pipeline answers them with a fixed localized greeting instead.

A message is a greeting only if it is short and *every* word is a greeting
word, so "bonjour, chneya siyessa mta3 el retour ?" is still a real question.
"""

from __future__ import annotations

import re
import unicodedata

MAX_GREETING_WORDS = 5

# Words that make a message a greeting on their own. Stored already normalized
# (lowercase, accents and Arabic diacritics/hamza folded, see _normalize).
_CORE_GREETINGS = {
    # French
    "bonjour", "bonsoir", "salut", "coucou", "bjr", "bsr", "slt", "cc", "allo",
    # English
    "hello", "hi", "hey", "hiya", "greetings", "morning", "evening", "afternoon",
    # Darija / Arabizi (Latin script)
    "aslema", "3aslema", "aslama", "3aslama", "salam", "salem", "slm", "salamou",
    "ahla", "ahlan", "ahlen", "marhba", "mar7ba", "marhaba", "mar7aba",
    "sbah", "sba7", "msa", "mse", "labes", "labess",
    # Arabic script (standard Arabic and Darija)
    "مرحبا", "السلام", "سلام", "اهلا", "هلا", "عسلامة", "صباح", "مساء", "تحية",
}

# Words that may accompany a greeting but are not a greeting by themselves.
_GREETING_COMPANIONS = {
    # French
    "tout", "le", "monde", "a", "tous", "madame", "monsieur", "mademoiselle", "messieurs", "mesdames",
    # English
    "good", "there", "everyone", "all", "guys",
    # Darija / Arabizi
    "3likom", "3alikom", "3likoum", "3alaykom", "alikom", "alaykom", "aleykoum", "alaikum", "alaykum",
    "w", "wa", "bik", "bikom", "sahla", "wsahla", "wasahla", "el", "lkhir", "l5ir", "khir", "5ir", "nour",
    "kol", "lkol", "jma3a",
    # Arabic script
    "عليكم", "ورحمة", "الله", "وبركاته", "وسهلا","بيك", "بكم", "الخير", "النور", "الجميع",
}

_TOKEN_RE = re.compile(r"\w+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_TATWEEL = "ـ"


def _normalize(text: str) -> str:
    # Lowercase, fold accents ("é" -> "e") and Arabic diacritics/hamza ("أ" -> "ا").
    folded = unicodedata.normalize("NFKD", text.lower().replace(_TATWEEL, ""))
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def _canonical_token(token: str) -> str:
    # "bonjourrr" -> "bonjour", "hiii" -> "hi".
    return _REPEATED_CHAR_RE.sub(r"\1", token)


def is_greeting_only(text: str) -> bool:
    """True if the message is a short greeting with no other content."""
    if not text or not text.strip():
        return False

    tokens = [_canonical_token(t) for t in _TOKEN_RE.findall(_normalize(text))]
    if not tokens or len(tokens) > MAX_GREETING_WORDS:
        return False

    if not all(t in _CORE_GREETINGS or t in _GREETING_COMPANIONS for t in tokens):
        return False
    return any(t in _CORE_GREETINGS for t in tokens)
