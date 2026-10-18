"""Gujarati text normalisation shared by training and evaluation.

Every transcript (reference, hypothesis, pseudo-label) must pass through the
same `normalize()` call, otherwise WER numbers are not comparable across models.
"""

import re
import unicodedata

NUKTA = "઼"
CANDRABINDU = "ઁ"
ANUSVARA = "ં"
ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))

# Gujarati digits U+0AE6..U+0AEF -> ASCII
GU_DIGITS = {0x0AE6 + i: str(i) for i in range(10)}

# Vowel signs typed as two pieces instead of the single code point.
# NFC does not compose these for Gujarati, so do it by hand.
VOWEL_SIGN_FIXES = [
    ("ાે", "ો"),  # AA + E  -> O
    ("ેા", "ો"),  # E + AA  -> O
    ("ાૈ", "ૌ"),  # AA + AI -> AU
    ("ૈા", "ૌ"),  # AI + AA -> AU
]

_SPACE_RE = re.compile(r"\s+")


def _strip_punct(text: str) -> str:
    # Keep letters, combining marks (Gujarati vowel signs, virama) and digits;
    # everything else, including danda, becomes a space.
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "M", "N")) or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def normalize(
    text: str,
    *,
    remove_nukta: bool = True,
    candrabindu_to_anusvara: bool = True,
    ascii_digits: bool = True,
    strip_punct: bool = True,
    lowercase_latin: bool = True,
) -> str:
    """Return the canonical form of a Gujarati transcript.

    Defaults are the evaluation setting. Dialect spellings vary a lot in
    nukta and nasalisation marks, so those are folded by default.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(ZERO_WIDTH)
    for bad, good in VOWEL_SIGN_FIXES:
        text = text.replace(bad, good)
    if remove_nukta:
        text = text.replace(NUKTA, "")
    if candrabindu_to_anusvara:
        text = text.replace(CANDRABINDU, ANUSVARA)
    if ascii_digits:
        text = text.translate(GU_DIGITS)
    if strip_punct:
        text = _strip_punct(text)
    if lowercase_latin:
        text = text.lower()
    return _SPACE_RE.sub(" ", text).strip()


def is_mostly_gujarati(text: str, min_ratio: float = 0.6) -> bool:
    """True if at least `min_ratio` of the letters are in the Gujarati block."""
    letters = [c for c in text if unicodedata.category(c).startswith(("L", "M"))]
    if not letters:
        return False
    gu = sum(1 for c in letters if "઀" <= c <= "૿")
    return gu / len(letters) >= min_ratio
