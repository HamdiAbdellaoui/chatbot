"""PII detection and masking (regex + heuristics).

Production goals:
- Reduce false positives (especially numeric IDs vs phones)
- Support multilingual input (French / Arabic / English)
- Provide lightweight confidence scoring (0..1) per entity
- Allowlist safe terms (cities, known keywords) to avoid over-masking
- Overlap-safe masking and easy extensibility for future NER/ML upgrades

Privacy:
- This module never logs.
- `detect_pii()` returns raw values for internal use, but callers must never log them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence


PIIType = Literal["PHONE", "EMAIL", "NAME", "ADDRESS"]


@dataclass(frozen=True)
class PIIEntity:
    type: PIIType
    value: str
    start: int
    end: int
    confidence: float


# Conservative allowlist of known-safe terms. Keep this list short; extend via caller.
# Entries should be in *normalized* form (lowercase, trimmed, single spaces).
ALLOWED_TERMS: set[str] = {
    # Tunisia (FR/EN)
    "tunis",
    "tunisie",
    "sfax",
    "sousse",
    "bizerte",
    "nabeul",
    "ariana",
    "gabes",
    "kairouan",
    "mahdia",
    "monastir",
    "hammamet",
    "djerba",
    "jerba",
    "ben arous",
    # تونس (AR)
    "تونس",
    "صفاقس",
    "سوسة",
    "بنزرت",
    "نابل",
    "أريانة",
    "اريانة",
    "قابس",
    "القيروان",
    "المهدية",
    "المنستير",
    "حمامات",
    "جربة",
    "بن عروس",
}


_ARABIC_INDIC_DIGITS = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)


def normalize_text(text: str) -> str:
    """Normalize user text for more robust detection.

Normalization steps (heuristic):
- Lowercase
- Convert Arabic-Indic digits to Western digits
- Collapse whitespace to single spaces
- Trim leading/trailing whitespace

Important: This function is *not* used for masking indices directly.
"""
    normalized, _ = _normalize_with_mapping(text)
    return normalized


_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])"  # avoid matching inside larger tokens
    r"[a-z0-9._%+-]+"  # local part
    r"@"
    r"[a-z0-9.-]+"  # domain
    r"\."
    r"[a-z]{2,}"  # tld
    r"(?![\w.-])"  # avoid trailing token chars
)


# Candidate phone pattern; we validate digit count post-match.
# Matches common international and local variants:
# - +216 22 333 444
# - (212) 555-1212
# - 00216 22333444
# - 22-333-444
_PHONE_FORMATTED_RE = re.compile(
    r"(?<![\w@])"  # don't start in an email/word
    r"(?:\+|00)?\d"  # +<digit> or 00<digit> or <digit>
    r"[\d\s().\-]{6,}\d"  # separators + digits, end with digit
    r"(?![\w@])"  # don't end in an email/word
)

# Bare-digit local/international phone numbers (no separators).
# Example: 22333444, 2125551212, 0021622333444
_PHONE_BARE_RE = re.compile(r"(?<!\d)\d{8,15}(?!\d)")


# Address patterns are intentionally conservative. We focus on:
# - number + street keyword (EN/FR)
# - arabic street keywords + some following text
_ADDRESS_LATIN_RE = re.compile(
    r"(?i)\b"  # word boundary
    r"\d{1,5}\s+"  # street number
    r"(?:[\w\u00C0-\u024F'’\-]+\s+){0,6}"  # name tokens (unicode latin)
    r"(?:"
    r"street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|place|pl\.?|route|"
    r"rue|avenue|bd\.?|impasse|cite|cité|lotissement|residence|résidence"
    r")\b\.?(?:\s+\w+){0,6}"
)


_ADDRESS_AR_RE = re.compile(
    r"(?i)\b"
    r"(?:شارع|نهج|طريق|زنقة|حي|إقامة|اقامة|عمارة)"
    r"\s+"
    r"[^\n,]{3,60}"
)


# Optional name detection: only when preceded by explicit self-identification phrases.
# We mask only the captured name group.
_NAME_EN_FR_RE = re.compile(
    r"(?i)\b(?:my\s+name\s+is|i\s+am|i'm|je\s+m['’]?appelle|mon\s+nom\s+est|moi\s+c['’]?est|je\s+suis)\s+"
    r"(?P<name>[\w\u00C0-\u024F'’\-]{2,30}(?:\s+[\w\u00C0-\u024F'’\-]{2,30}){0,2})"
)

_NAME_AR_RE = re.compile(
    r"(?i)\b(?:اسمي|أنا\s+اسمي|انا\s+اسمي)\s+"
    r"(?P<name>[^\W\d_]{2,30}(?:\s+[^\W\d_]{2,30}){0,2})"
)

_NAME_AR_LOWER_CONF_RE = re.compile(
    r"(?i)\b(?:أنا|انا)\s+"
    r"(?P<name>[^\W\d_]{2,30}(?:\s+[^\W\d_]{2,30}){1,2})"
)


def _digits_count(s: str) -> int:
    return sum(1 for ch in s if ch.isdigit())


def _is_valid_formatted_phone(candidate: str) -> bool:
    if "@" in candidate:
        return False

    stripped = candidate.strip()
    if stripped.isdigit():
        return False

    digit_count = _digits_count(stripped)
    return 8 <= digit_count <= 15


def _is_valid_bare_phone(candidate: str) -> bool:
    stripped = candidate.strip()
    if not stripped.isdigit():
        return False
    digit_count = len(stripped)
    if digit_count < 8 or digit_count > 15:
        return False
    # Reject obvious junk.
    if len(set(stripped)) <= 2:
        return False
    return True


@dataclass(frozen=True)
class _PatternSpec:
    type: PIIType
    regex: re.Pattern[str]
    group: str | None = None
    validator: Callable[[str], bool] | None = None
    base_confidence: float = 0.6


_PATTERN_SPECS: tuple[_PatternSpec, ...] = (
    _PatternSpec(type="EMAIL", regex=_EMAIL_RE),
    _PatternSpec(type="PHONE", regex=_PHONE_FORMATTED_RE, validator=_is_valid_formatted_phone, base_confidence=0.78),
    _PatternSpec(type="PHONE", regex=_PHONE_BARE_RE, validator=_is_valid_bare_phone, base_confidence=0.48),
    _PatternSpec(type="ADDRESS", regex=_ADDRESS_LATIN_RE, base_confidence=0.72),
    _PatternSpec(type="ADDRESS", regex=_ADDRESS_AR_RE, base_confidence=0.70),
    _PatternSpec(type="NAME", regex=_NAME_EN_FR_RE, group="name", base_confidence=0.72),
    _PatternSpec(type="NAME", regex=_NAME_AR_RE, group="name", base_confidence=0.74),
    _PatternSpec(type="NAME", regex=_NAME_AR_LOWER_CONF_RE, group="name", base_confidence=0.55),
)


_PRIORITY: dict[PIIType, int] = {
    # Prefer "tighter" and higher-confidence types first.
    "EMAIL": 400,
    "PHONE": 300,
    "ADDRESS": 200,
    "NAME": 100,
}


PHONE_BLOCK_KEYWORDS: tuple[str, ...] = (
    # EN/FR
    "order",
    "commande",
    "cmd",
    "invoice",
    "facture",
    "tracking",
    "suivi",
    "reference",
    "référence",
    "ref",
    "réf",
    "id",
    "ticket",
    "case",
    "dossier",
    "sku",
    # AR
    "طلب",
    "رقم الطلب",
    "فاتورة",
    "تتبع",
    "رقم تتبع",
    "معرّف",
    "معرف",
)


PHONE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    # EN/FR
    "phone",
    "tel",
    "tél",
    "téléphone",
    "mobile",
    "whatsapp",
    "num",
    "numéro",
    "numero",
    "call",
    # AR
    "هاتف",
    "تلفون",
    "رقم",
    "موبايل",
    "واتساب",
)


DEFAULT_MIN_CONFIDENCE_BY_TYPE: dict[PIIType, float] = {
    "EMAIL": 0.80,
    "PHONE": 0.70,
    "NAME": 0.62,
    "ADDRESS": 0.68,
}


def _normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """Normalize text and produce a char-index mapping back to the original.

The mapping list has the same length as the normalized string; each normalized
character maps to an index in the original string.
"""
    if not text:
        return "", []

    normalized_chars: list[str] = []
    mapping: list[int] = []

    i = 0
    in_whitespace = False
    while i < len(text):
        ch = text[i]

        # Normalize common invisible separators into whitespace.
        if ch in {"\u200b", "\u200c", "\u200d", "\ufeff"}:
            ch = " "

        # Convert Arabic-Indic digits.
        ch = ch.translate(_ARABIC_INDIC_DIGITS)

        # Collapse all whitespace to a single space.
        if ch.isspace():
            if not in_whitespace:
                normalized_chars.append(" ")
                mapping.append(i)
                in_whitespace = True
            i += 1
            continue

        in_whitespace = False
        normalized_chars.append(ch.lower())
        mapping.append(i)
        i += 1

    normalized = "".join(normalized_chars)

    # Trim spaces while preserving mapping.
    left = 0
    while left < len(normalized) and normalized[left] == " ":
        left += 1
    right = len(normalized)
    while right > left and normalized[right - 1] == " ":
        right -= 1

    return normalized[left:right], mapping[left:right]


def _window(text: str, start: int, end: int, *, before: int, after: int) -> str:
    return text[max(0, start - before) : min(len(text), end + after)]


def _contains_any(haystack: str, needles: Sequence[str]) -> bool:
    h = haystack
    return any(n in h for n in needles)


def _score_email(value: str) -> float:
    if "@" not in value or "." not in value:
        return 0.0
    return 0.98


def _score_phone(*, value_norm: str, context_norm: str, whole_text_norm: str, is_bare: bool) -> float:
    if _contains_any(context_norm, PHONE_BLOCK_KEYWORDS):
        return 0.0

    digits = "".join(ch for ch in value_norm if ch.isdigit())
    if not (8 <= len(digits) <= 15):
        return 0.0

    if is_bare:
        # Bare digit numbers are risky (order IDs). Only trust them with context.
        if whole_text_norm.strip() == value_norm.strip():
            return 0.78
        if _contains_any(context_norm, PHONE_CONTEXT_KEYWORDS):
            return 0.78
        if digits.startswith("216") and len(digits) in {11, 12}:
            return 0.62
        if len(digits) == 8:
            return 0.55
        return 0.45

    # Formatted: generally higher confidence.
    score = 0.78
    if value_norm.strip().startswith("+") or value_norm.strip().startswith("00"):
        score = 0.86
    if _contains_any(context_norm, PHONE_CONTEXT_KEYWORDS):
        score = min(0.95, score + 0.08)
    return score


def _score_name(value: str, *, base: float, allowlist: set[str]) -> float:
    v_norm = normalize_text(value)
    if not v_norm or v_norm in allowlist:
        return 0.0
    if any(ch.isdigit() for ch in value):
        return 0.0
    tokens = [t for t in v_norm.split(" ") if t]
    if not tokens:
        return 0.0
    if len(tokens) == 1:
        # Single-token names are common but also noisy.
        return min(base, 0.62)
    if len(tokens) == 2:
        return min(0.90, base + 0.08)
    return min(0.90, base + 0.12)


def _score_address(value: str, *, base: float) -> float:
    v_norm = normalize_text(value)
    if len(v_norm) < 8:
        return 0.0
    if any(k in v_norm for k in ("rue", "avenue", "street", "road", "bd", "boulevard")):
        return min(0.92, base + 0.12)
    if any(k in v_norm for k in ("شارع", "نهج", "طريق", "زنقة", "حي", "إقامة", "اقامة")):
        return min(0.90, base + 0.10)
    return base


def detect_pii(text: str, *, allowed_terms: Iterable[str] | None = None) -> list[PIIEntity]:
    """Detect PII entities with confidence.

Returns entities with non-overlapping spans in ascending order.

Note: Returned `value` contains raw text; callers must never log it.
"""
    if not text:
        return []

    allowlist = set(ALLOWED_TERMS)
    if allowed_terms:
        allowlist.update(normalize_text(x) for x in allowed_terms if isinstance(x, str) and x.strip())

    normalized, mapping = _normalize_with_mapping(text)
    if not normalized:
        return []

    candidates: list[PIIEntity] = []

    for spec in _PATTERN_SPECS:
        for match in spec.regex.finditer(normalized):
            if spec.group:
                ns, ne = match.span(spec.group)
            else:
                ns, ne = match.span(0)

            if ns == ne:
                continue

            if ns < 0 or ne > len(mapping):
                continue

            # Map normalized indices back to original.
            os = mapping[ns]
            oe = mapping[ne - 1] + 1

            if os < 0 or oe > len(text) or os >= oe:
                continue

            value = text[os:oe]
            if spec.validator and not spec.validator(value):
                continue

            # Basic length guard.
            if oe - os > 140:
                continue

            ctx_norm = _window(normalized, ns, ne, before=28, after=12)
            raw_norm = normalized[ns:ne]

            confidence = spec.base_confidence
            if spec.type == "EMAIL":
                confidence = _score_email(value)
            elif spec.type == "PHONE":
                confidence = _score_phone(
                    value_norm=raw_norm,
                    context_norm=ctx_norm,
                    whole_text_norm=normalized,
                    is_bare=(spec.regex is _PHONE_BARE_RE),
                )
            elif spec.type == "NAME":
                confidence = _score_name(value, base=spec.base_confidence, allowlist=allowlist)
            elif spec.type == "ADDRESS":
                confidence = _score_address(value, base=spec.base_confidence)

            # Allowlist full-value matches (e.g., city names).
            if normalize_text(value) in allowlist:
                confidence = 0.0

            if confidence <= 0.0:
                continue

            candidates.append(
                PIIEntity(
                    type=spec.type,
                    value=value,
                    start=os,
                    end=oe,
                    confidence=float(max(0.0, min(1.0, confidence))),
                )
            )

    if not candidates:
        return []

    # Resolve overlaps using priority + confidence + length.
    candidates.sort(key=lambda e: (e.start, -(e.end - e.start)))
    selected: list[PIIEntity] = []

    def _rank(e: PIIEntity) -> tuple[int, float, int]:
        return (_PRIORITY[e.type], e.confidence, e.end - e.start)

    for ent in candidates:
        if not selected:
            selected.append(ent)
            continue

        last = selected[-1]
        if ent.start >= last.end:
            selected.append(ent)
            continue

        # Overlap: keep the higher-ranked one.
        if _rank(ent) > _rank(last):
            selected[-1] = ent

    # Ensure strictly non-overlapping.
    out: list[PIIEntity] = []
    for ent in sorted(selected, key=lambda e: e.start):
        if not out or ent.start >= out[-1].end:
            out.append(ent)
        else:
            prev = out[-1]
            if _rank(ent) > _rank(prev):
                out[-1] = ent

    return out


def select_entities_to_mask(
    entities: Sequence[PIIEntity],
    *,
    min_confidence: float | None = None,
    allowed_terms: Iterable[str] | None = None,
) -> list[PIIEntity]:
    """Select which detected entities should actually be masked."""
    if not entities:
        return []

    allowlist = set(ALLOWED_TERMS)
    if allowed_terms:
        allowlist.update(normalize_text(x) for x in allowed_terms if isinstance(x, str) and x.strip())

    out: list[PIIEntity] = []
    for ent in entities:
        if normalize_text(ent.value) in allowlist:
            continue

        threshold = float(min_confidence) if min_confidence is not None else DEFAULT_MIN_CONFIDENCE_BY_TYPE[ent.type]
        if ent.confidence < threshold:
            continue
        out.append(ent)

    out.sort(key=lambda e: e.start)
    return out


def mask_pii(
    text: str,
    *,
    entities: Sequence[PIIEntity] | None = None,
    min_confidence: float | None = None,
    allowed_terms: Iterable[str] | None = None,
) -> str:
    """Mask PII using placeholders.

Masking decisions are confidence-driven. Weak matches are ignored.

Placeholders:
- [PHONE]
- [EMAIL]
- [NAME]
- [ADDRESS]
"""
    if not text:
        return text

    detected = list(entities) if entities is not None else detect_pii(text, allowed_terms=allowed_terms)
    if not detected:
        return text

    spans = select_entities_to_mask(
        detected,
        min_confidence=min_confidence,
        allowed_terms=allowed_terms,
    )

    if not spans:
        return text

    spans.sort(key=lambda e: e.start)

    chunks: list[str] = []
    cursor = 0
    for ent in spans:
        if ent.start < cursor:
            continue
        chunks.append(text[cursor:ent.start])
        chunks.append(f"[{ent.type}]")
        cursor = ent.end
    chunks.append(text[cursor:])

    return "".join(chunks)
