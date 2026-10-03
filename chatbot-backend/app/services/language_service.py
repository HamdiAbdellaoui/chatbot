"""Lightweight, offline language detector for French / Arabic / Tunisian Darija.

No network calls, no ML model: a fast heuristic good enough to pick the right
system prompt before calling the LLM. See eval/run_language_eval.py for a
measured accuracy report against a small hand-labeled dataset.
"""

from __future__ import annotations

from typing import Literal

Language = Literal["darija", "ar", "fr", "other"]

# Non-exhaustive list of Tunisian Darija lexical markers (chat-Arabic / Latin
# transliteration included, e.g. "3" for ع, "5"/"7" for خ/ح). Presence of any
# one of these (case-insensitive substring match) is treated as a strong
# signal that the text is Tunisian Darija rather than Modern Standard Arabic
# or French.
_DARIJA_MARKERS = [
    "bech", "besh",
    "mte3", "mte3i", "mte3ek", "mte3ha",
    "barcha",
    "shnowa", "chnowa", "chna7", "chna7where",
    "kifeh", "kifach",
    "wesh", "weh",
    "hnaya",
    "ahna",
    "lazem",
    "yezzi", "yezzik",
    "walla",
    "inchallah", "nchallah",
    "3andi", "3andek",
    "5ou", "5ouya",
    "3aslema", "aslema",
    "sahbi", "sahbti",
    "chwiya",
    "taw", "twa",
    "barka",
    "labes",
    "3ala5er",
    "n7eb", "nhab",
]

_ARABIC_RANGE = (0x0600, 0x06FF)


def _arabic_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic_count = sum(1 for ch in text if _ARABIC_RANGE[0] <= ord(ch) <= _ARABIC_RANGE[1])
    return arabic_count / len(text)


def _letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for ch in text if ch.isalpha())
    return letters / len(text)


def detect_language(text: str) -> Language:
    """Detect the language/register of a user message.

    Order of checks:
    1. Tunisian Darija lexical markers (case-insensitive substring match).
    2. Ratio of Arabic-script characters (U+0600-U+06FF) > 0.4 -> "ar".
    3. Default to "fr", unless the text is empty or has almost no
       latin/arabic letters at all -> "other".
    """
    if not text or not text.strip():
        return "other"

    stripped = text.strip()
    lower = stripped.lower()

    if any(marker in lower for marker in _DARIJA_MARKERS):
        return "darija"

    if _arabic_char_ratio(stripped) > 0.4:
        return "ar"

    if _letter_ratio(stripped) < 0.3:
        return "other"

    return "fr"
