"""Split gold-transcribed data into train/dev/test by channel/speaker, and check for leaks.

  python data_pipeline/make_splits.py --inputs data/manifests/gold.jsonl \
      --out data/manifests --dev 0.15 --test 0.25

Also checks that no pseudo-labelled training manifest (--train-extra) shares a
speaker or channel with dev/test. Exits non-zero on any leak.
"""

import argparse
import sys

from kgasr.manifest import find_leaks, read_manifest, split_by_group, total_hours, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--dev", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.25)
    ap.add_argument("--train-extra", nargs="*", default=[], help="pseudo/public train manifests to leak-check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [r for p in args.inputs for r in read_manifest(p)]
    splits = split_by_group(rows, {"train": 1 - args.dev - args.test, "dev": args.dev, "test": args.test},
                            seed=args.seed)
    check = dict(splits)
    check["train"] = splits["train"] + [r for p in args.train_extra for r in read_manifest(p)]
    leaks = find_leaks(check)
    if leaks:
        print("LEAKS FOUND:\n  " + "\n  ".join(leaks[:50]))
        sys.exit(1)

    for name, part in splits.items():
        write_manifest(f"{args.out}/kathiyawadi_{name}.jsonl", part)
        spk = len({r.get("speaker") or r.get("channel") for r in part})
        print(f"{name:5s} {len(part):6d} segs  {total_hours(part):6.2f} h  {spk} speakers/channels")


if __name__ == "__main__":
    main()
