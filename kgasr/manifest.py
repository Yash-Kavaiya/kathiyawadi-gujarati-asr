"""JSONL manifests (NeMo format) and leakage-free train/dev/test splits.

One line per segment:
  {"audio_filepath": ..., "duration": 4.2, "text": ..., "speaker": ..., "channel": ...,
   "source": "youtube|own|public", "label": "gold|subtitle|pseudo"}
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_manifest(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def group_key(row: dict) -> str:
    """Rows sharing a channel OR a speaker must stay in the same split.

    Channel is the coarser unit for YouTube (one channel often = one speaker
    set), so it wins when present.
    """
    return row.get("channel") or row.get("speaker") or row["audio_filepath"]


def split_by_group(rows: list[dict], fractions: dict[str, float], seed: int = 0) -> dict[str, list[dict]]:
    """Assign whole groups to splits so that split durations approximate `fractions`."""
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = sum(r.get("duration", 1.0) for r in rows)
    targets = {name: frac * total for name, frac in fractions.items()}
    filled = {name: 0.0 for name in fractions}
    out = {name: [] for name in fractions}
    # Fill the split that is furthest below its target first.
    for k in keys:
        dur = sum(r.get("duration", 1.0) for r in groups[k])
        name = max(fractions, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
        out[name].extend(groups[k])
        filled[name] += dur
    return out


def find_leaks(splits: dict[str, list[dict]], fields=("speaker", "channel")) -> list[str]:
    """Return human-readable descriptions of any speaker/channel shared by two splits."""
    leaks = []
    for field in fields:
        owner = {}
        for name, rows in splits.items():
            for r in rows:
                v = r.get(field)
                if not v:
                    continue
                if v in owner and owner[v] != name:
                    leaks.append(f"{field}={v!r} in both {owner[v]} and {name}")
                owner.setdefault(v, name)
    return sorted(set(leaks))


def total_hours(rows: Iterable[dict]) -> float:
    return sum(r.get("duration", 0.0) for r in rows) / 3600
