"""WER/CER and bootstrap confidence intervals.

Corpus-level error rate = total edits / total reference tokens, which is what
papers report. Bootstrap resamples utterances, not tokens.
"""

import random
from dataclasses import dataclass
from typing import Sequence

from kgasr.normalize import normalize


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance between two token sequences."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def _tokens(text: str, unit: str) -> list:
    if unit == "word":
        return text.split()
    if unit == "char":
        return list(text.replace(" ", ""))
    raise ValueError(f"unknown unit {unit!r}")


def utterance_counts(refs: Sequence[str], hyps: Sequence[str], unit: str, norm: bool = True):
    """Per-utterance (edits, ref_len) pairs."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} refs vs {len(hyps)} hyps")
    out = []
    for r, h in zip(refs, hyps):
        if norm:
            r, h = normalize(r), normalize(h)
        rt, ht = _tokens(r, unit), _tokens(h, unit)
        out.append((edit_distance(rt, ht), len(rt)))
    return out


def error_rate(counts) -> float:
    edits = sum(e for e, _ in counts)
    total = sum(n for _, n in counts)
    return edits / total if total else 0.0


def wer(refs, hyps, norm: bool = True) -> float:
    return error_rate(utterance_counts(refs, hyps, "word", norm))


def cer(refs, hyps, norm: bool = True) -> float:
    return error_rate(utterance_counts(refs, hyps, "char", norm))


@dataclass
class ScoreWithCI:
    value: float
    low: float
    high: float

    def as_dict(self) -> dict:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high}


def bootstrap_ci(counts, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> ScoreWithCI:
    """Percentile bootstrap CI over utterances."""
    rng = random.Random(seed)
    n = len(counts)
    samples = []
    for _ in range(n_boot):
        draw = [counts[rng.randrange(n)] for _ in range(n)]
        samples.append(error_rate(draw))
    samples.sort()
    lo = samples[int(alpha / 2 * n_boot)]
    hi = samples[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return ScoreWithCI(error_rate(counts), lo, hi)


def paired_bootstrap(counts_a, counts_b, n_boot: int = 1000, seed: int = 0) -> float:
    """Fraction of resamples where system A is NOT better than B (a p-value)."""
    if len(counts_a) != len(counts_b):
        raise ValueError("systems must be scored on the same utterances")
    rng = random.Random(seed)
    n = len(counts_a)
    worse = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        if error_rate([counts_a[i] for i in idx]) >= error_rate([counts_b[i] for i in idx]):
            worse += 1
    return worse / n_boot
