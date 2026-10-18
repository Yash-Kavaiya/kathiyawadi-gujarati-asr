"""Drop segments that are not Gujarati speech (Hindi, English, music) via VoxLingua107 LID.

  python data_pipeline/lid_filter.py --manifest data/segments/manifest.jsonl \
      --out data/segments/manifest.gu.jsonl
"""

import argparse

from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm

from kgasr.manifest import read_manifest, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-prob", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    clf = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa", run_opts={"device": args.device})
    rows, kept = read_manifest(args.manifest), []
    for r in tqdm(rows):
        out_prob, score, _, label = clf.classify_file(r["audio_filepath"])
        lang = label[0].split(":")[0].strip()
        prob = float(score.exp()[0])
        r["lid"], r["lid_prob"] = lang, round(prob, 3)
        if lang == "gu" and prob >= args.min_prob:
            kept.append(r)
    write_manifest(args.out, kept)
    print(f"kept {len(kept)}/{len(rows)} segments as Gujarati")


if __name__ == "__main__":
    main()
