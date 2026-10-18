"""Run every model on every test set with the same normaliser; write WER/CER with 95% CIs.

  python eval/run_benchmarks.py --config eval/baselines.yaml                 # everything
  python eval/run_benchmarks.py --models indicconformer_gu ours_conformer_r1 --testsets kathiyawadi_test

Hypotheses are cached in eval/results/hyps/<model>/<testset>.jsonl, so adding a
model only runs that model. eval/results/table.md is the paper table source.
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from kgasr.manifest import read_manifest, write_manifest
from kgasr.metrics import bootstrap_ci, paired_bootstrap, utterance_counts

RESULTS = Path("eval/results")


def transcribe_cached(name, spec, ts_name, rows, device):
    cache = RESULTS / "hyps" / name / f"{ts_name}.jsonl"
    if cache.exists():
        return [r["hyp"] for r in read_manifest(cache)], None
    from kgasr.asr import load
    asr = load(spec["kind"], spec["id"], device=device)
    t0 = time.time()
    hyps = asr([r["audio_filepath"] for r in rows])
    elapsed = time.time() - t0
    write_manifest(cache, [{"audio_filepath": r["audio_filepath"], "ref": r["text"], "hyp": h}
                           for r, h in zip(rows, hyps)])
    audio_s = sum(r.get("duration", 0) for r in rows)
    return hyps, elapsed / audio_s if audio_s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval/baselines.yaml")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--testsets", nargs="*")
    ap.add_argument("--reference-model", default="indicconformer_gu",
                    help="paired-bootstrap p-values are computed against this model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    models = {k: v for k, v in cfg["models"].items() if not args.models or k in args.models}
    testsets = {k: v for k, v in cfg["testsets"].items()
                if (not args.testsets or k in args.testsets) and Path(v).exists()}

    scores_path = RESULTS / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    counts = {}
    for ts_name, ts_path in testsets.items():
        rows = read_manifest(ts_path)
        for name, spec in models.items():
            hyps, rtf = transcribe_cached(name, spec, ts_name, rows, args.device)
            refs = [r["text"] for r in rows]
            wc = utterance_counts(refs, hyps, "word")
            cc = utterance_counts(refs, hyps, "char")
            counts[(name, ts_name)] = wc
            entry = scores.setdefault(name, {}).setdefault(ts_name, {})
            entry["wer"] = bootstrap_ci(wc, args.n_boot).as_dict()
            entry["cer"] = bootstrap_ci(cc, args.n_boot).as_dict()
            entry["n_utts"] = len(rows)
            if rtf is not None:
                entry["rtf"] = round(rtf, 4)
            print(f"{ts_name:20s} {name:26s} WER {entry['wer']['value']:.3f}  CER {entry['cer']['value']:.3f}")

        ref_key = (args.reference_model, ts_name)
        if ref_key in counts:
            for name in models:
                if name != args.reference_model:
                    p = paired_bootstrap(counts[(name, ts_name)], counts[ref_key], args.n_boot)
                    scores[name][ts_name][f"p_better_than_{args.reference_model}"] = p

    RESULTS.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2))
    write_table(scores)


def write_table(scores):
    tsets = sorted({t for m in scores.values() for t in m})
    lines = ["| model | " + " | ".join(tsets) + " |", "|---|" + "---|" * len(tsets)]
    for name in sorted(scores):
        cells = []
        for t in tsets:
            w = scores[name].get(t, {}).get("wer")
            cells.append(f"{100 * w['value']:.1f} ({100 * w['ci_low']:.1f}-{100 * w['ci_high']:.1f})" if w else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (RESULTS / "table.md").write_text("WER % (95% CI)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
