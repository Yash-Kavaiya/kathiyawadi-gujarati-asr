"""Transcribe every segment with one model and store the output as `hyp_<name>`.

Run once per labelling model, e.g.:
  python data_pipeline/pseudolabel.py --manifest data/segments/manifest.gu.jsonl \
      --kind nemo --model ai4bharat/indicconformer_stt_gu_hybrid_ctc_rnnt_large --name conformer
  python data_pipeline/pseudolabel.py --manifest data/segments/manifest.gu.jsonl \
      --kind whisper --model /models/indicwhisper-gu --name whisper

The manifest is updated in place (written to a temp file first), so the two
runs accumulate into one file that filter.py consumes.
"""

import argparse
import os

from tqdm import tqdm

from kgasr.asr import load
from kgasr.manifest import read_manifest, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kind", required=True, choices=["whisper", "nemo", "indicconformer_multi", "mms"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="column suffix: hyp_<name>")
    ap.add_argument("--chunk", type=int, default=256, help="segments per transcribe() call")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    col = f"hyp_{args.name}"
    todo = [i for i, r in enumerate(rows) if col not in r]
    print(f"{len(todo)}/{len(rows)} segments to transcribe with {args.model}")
    asr = load(args.kind, args.model, device=args.device)

    tmp = args.manifest + ".tmp"
    for k in tqdm(range(0, len(todo), args.chunk)):
        idx = todo[k:k + args.chunk]
        for i, text in zip(idx, asr([rows[i]["audio_filepath"] for i in idx])):
            rows[i][col] = text
        write_manifest(tmp, rows)  # checkpoint so a crash does not lose hours of GPU time
        os.replace(tmp, args.manifest)


if __name__ == "__main__":
    main()
