"""Export public Gujarati test sets from the HF Hub into local wav + manifest form.

  python eval/export_public.py --set fleurs --out data/manifests/eval

Column names differ per dataset and gated sets may change layout; the specs
below say where audio/text live. Check them once with `--peek` after access is granted.
"""

import argparse
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

from kgasr.manifest import write_manifest

SPECS = {
    "fleurs": dict(path="google/fleurs", name="gu_in", split="test", text="transcription",
                   out="fleurs_gu_test"),
    "kathbath": dict(path="ai4bharat/Kathbath", data_dir="gujarati", split="test_known", text="text",
                     out="kathbath_gu_test_known"),
    "kathbath_noisy": dict(path="ai4bharat/Kathbath", data_dir="gujarati", split="test_known_noisy",
                           text="text", out="kathbath_gu_test_noisy"),
    "indicvoices": dict(path="ai4bharat/IndicVoices", data_dir="gujarati", split="valid", text="text",
                        out="indicvoices_gu_valid"),
    "commonvoice": dict(path="mozilla-foundation/common_voice_17_0", name="gu", split="test", text="sentence",
                        out="commonvoice_gu_test"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=SPECS)
    ap.add_argument("--out", default="data/manifests/eval")
    ap.add_argument("--audio-dir", default="data/eval_audio")
    ap.add_argument("--peek", action="store_true", help="print the first row's columns and exit")
    args = ap.parse_args()
    s = SPECS[args.set]

    ds = load_dataset(s["path"], s.get("name"), data_dir=s.get("data_dir"), split=s["split"])
    if args.peek:
        print(ds)
        print({k: (v if k != "audio" else "...") for k, v in ds[0].items()})
        return
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    adir = Path(args.audio_dir) / s["out"]
    adir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        path = adir / f"{i:06d}.wav"
        arr = ex["audio"]["array"]
        sf.write(path, arr, 16000)
        rows.append({"audio_filepath": str(path), "duration": round(len(arr) / 16000, 3),
                     "text": ex[s["text"]], "speaker": str(ex.get("speaker_id", "")) or None})
    n = write_manifest(f"{args.out}/{s['out']}.jsonl", rows)
    print(f"{n} utterances -> {args.out}/{s['out']}.jsonl")


if __name__ == "__main__":
    main()
