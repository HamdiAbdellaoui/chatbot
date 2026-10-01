"""Lightweight, offline language detector for French / Arabic / Tunisian Darija.

No network calls, no ML model: a fast heuristic good enough to pick the right
system prompt before calling the LLM. See eval/run_language_eval.py for a
measured accuracy report against a small hand-labeled dataset.
"""

from __future__ import annotations

import re
from typing import Literal

Language = Literal["darija", "ar", "fr", "other"]

# Non-exhaustive list of Tunisian Darija lexical markers (chat-Arabic / Latin
# transliteration included, e.g. "3" for ع, "5"/"7" for خ/ح). Presence of any
# one of these as a whole word (case-insensitive) is treated as a strong
# signal that the text is Tunisian Darija rather than Modern Standard Arabic
# or French. Whole-word matching avoids false positives such as "Ottawa" -> "taw".
_DARIJA_MARKERS = {
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
    "nheb", "nahb",
    "nechri",
    "9adech", "9adeh",
    "famma", "fama",
    "ahla",
    "3aychek",
    "brabi",
    "yaatik",
    "mouch", "mech",
    "chnia",
    "kifech",
    "win",
    "wa9tech",
    "tawa",
}

# Tokens: runs of letters/digits, so arabizi digits stay inside words ("9adech").
_TOKEN_RE = re.compile(r"\w+")

# Arabizi: a Latin word mixing letters with the digits used for Arabic sounds
# (3=ع, 5=خ, 7=ح, 9=ق, 6=ط), e.g. "3andi", "n7eb", "ma3a".
_ARABIZI_TOKEN_RE = re.compile(r"[a-z]*[35679][a-z]+")

# Digit + letters tokens that are ordinals, units or times in French, not arabizi.
_NOT_ARABIZI_RE = re.compile(
    r"\d+(?:e|er|ere|eme|ieme|em|nd|nde|h|mn|min|s|kg|g|mg|cm|mm|m|km|l|cl|ml|v|w|kw|a|mah|x|go|mo|to|gb|mb|tb|k|p|dt|tnd|ans)"
)

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
    1. Tunisian Darija lexical markers (case-insensitive, whole words).
    2. Ratio of Arabic-script characters (U+0600-U+06FF) > 0.4 -> "ar".
    3. Latin arabizi word mixing letters and 3/5/6/7/9 (e.g. "9adeh") -> "darija".
    4. Default to "fr", unless the text is empty or has almost no
       latin/arabic letters at all -> "other".
    """
    if not text or not text.strip():
        return "other"

    stripped = text.strip()
    tokens = _TOKEN_RE.findall(stripped.lower())

    if any(token in _DARIJA_MARKERS for token in tokens):
        return "darija"

    if _arabic_char_ratio(stripped) > 0.4:
        return "ar"

    if any(_ARABIZI_TOKEN_RE.fullmatch(t) and not _NOT_ARABIZI_RE.fullmatch(t) for t in tokens):
        return "darija"

    if _letter_ratio(stripped) < 0.3:
        return "other"

    return "fr"
