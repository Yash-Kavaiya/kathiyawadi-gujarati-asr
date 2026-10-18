"""Keep segments where two hypotheses agree; write a training manifest.

  python data_pipeline/filter.py --manifest data/segments/manifest.gu.jsonl \
      --primary hyp_conformer --secondary hyp_whisper --out data/manifests/pseudo.jsonl

If a segment has a YouTube subtitle, it is used as the secondary hypothesis
instead (human-written text is a stronger check than a second model), and the
subtitle becomes the label when it agrees.

--sweep prints the kept-hours / label-WER trade-off on a gold-labelled manifest
so the threshold can be chosen on dev data, not guessed.
"""

import argparse
import json
from collections import Counter

from kgasr.filtering import FilterConfig, judge
from kgasr.manifest import read_manifest, total_hours, write_manifest
from kgasr.metrics import wer


def run(rows, primary, secondary, cfg, use_subtitles=True):
    kept, reasons = [], Counter()
    for r in rows:
        a = r.get(primary, "")
        sub = r.get("subtitle", "") if use_subtitles else ""
        b = sub or r.get(secondary, "")
        ok, why, d = judge(a, b, r["duration"], cfg)
        reasons[why] += 1
        if ok:
            out = dict(r, text=sub if sub else a, label="subtitle" if sub else "pseudo",
                       agreement_cer=round(d, 4))
            kept.append(out)
    return kept, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max-cer", type=float, default=FilterConfig.max_cer)
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="manifest must have gold `text`")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)

    if args.sweep:
        print("max_cer  kept_h  kept%  label_WER_vs_gold")
        for t in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
            gold = {r["audio_filepath"]: r["text"] for r in rows}
            kept, _ = run(rows, args.primary, args.secondary, FilterConfig(max_cer=t), not args.no_subtitles)
            w = wer([gold[r["audio_filepath"]] for r in kept], [r["text"] for r in kept]) if kept else float("nan")
            print(f"{t:7.2f}  {total_hours(kept):6.2f}  {100 * len(kept) / len(rows):5.1f}  {w:.3f}")
        return

    kept, reasons = run(rows, args.primary, args.secondary, FilterConfig(max_cer=args.max_cer),
                        not args.no_subtitles)
    write_manifest(args.out, kept)
    report = {"in_segments": len(rows), "in_hours": round(total_hours(rows), 2),
              "kept_segments": len(kept), "kept_hours": round(total_hours(kept), 2),
              "reasons": dict(reasons)}
    print(json.dumps(report, indent=2))
    with open(args.out + ".report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
