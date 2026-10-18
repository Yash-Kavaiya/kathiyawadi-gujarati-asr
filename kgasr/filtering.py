"""Pseudo-label agreement filter.

Two independent models transcribe each segment. A segment is kept only when
their normalised outputs agree closely, which is a cheap proxy for "both are
probably right". The threshold is tuned on the gold dev set.
"""

from dataclasses import dataclass

from kgasr.metrics import edit_distance
from kgasr.normalize import is_mostly_gujarati, normalize


@dataclass
class FilterConfig:
    max_cer: float = 0.12          # agreement threshold between the two models
    min_chars_per_sec: float = 3.0  # below this: mostly silence/music, or truncated output
    max_chars_per_sec: float = 25.0 # above this: hallucinated / repeated text
    min_duration: float = 1.0
    max_duration: float = 30.0
    min_gujarati_ratio: float = 0.6


def pair_cer(a: str, b: str) -> float:
    """Symmetric character disagreement between two hypotheses (already normalised)."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    denom = max(len(a), len(b))
    return edit_distance(list(a), list(b)) / denom if denom else 1.0


def judge(hyp_a: str, hyp_b: str, duration: float, cfg: FilterConfig) -> tuple[bool, str, float]:
    """Return (keep, reason, agreement_cer). The kept label is hyp_a."""
    a, b = normalize(hyp_a), normalize(hyp_b)
    if not a or not b:
        return False, "empty", 1.0
    if not (cfg.min_duration <= duration <= cfg.max_duration):
        return False, "duration", 1.0
    if not is_mostly_gujarati(a, cfg.min_gujarati_ratio):
        return False, "not_gujarati", 1.0
    rate = len(a.replace(" ", "")) / duration
    if not (cfg.min_chars_per_sec <= rate <= cfg.max_chars_per_sec):
        return False, "char_rate", 1.0
    d = pair_cer(a, b)
    if d > cfg.max_cer:
        return False, "disagree", d
    return True, "ok", d
