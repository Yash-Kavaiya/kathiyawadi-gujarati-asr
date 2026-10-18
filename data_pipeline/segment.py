"""Split long recordings into 2-20 s speech segments with Silero VAD.

  python data_pipeline/segment.py --raw data/raw --out data/segments

Writes data/segments/manifest.jsonl. If a matching .gu.vtt subtitle exists,
the subtitle text overlapping each segment is stored in `subtitle` so the
filter step can use it as an extra hypothesis.
"""

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tqdm import tqdm

SR = 16000
_TS = re.compile(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)")


def parse_vtt(path):
    cues, cur = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _TS.search(line)
        if m:
            g = list(map(int, m.groups()))
            cur = [g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                   g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, []]
            cues.append(cur)
        elif cur is not None and line.strip():
            cur[2].append(re.sub(r"<[^>]+>", "", line.strip()))
    return [(s, e, " ".join(t)) for s, e, t in cues]


def subtitle_for(cues, start, end, min_overlap=0.5):
    """Concatenate cues that overlap [start, end] by at least `min_overlap` of the cue."""
    parts = []
    for s, e, t in cues:
        ov = min(e, end) - max(s, start)
        if ov > 0 and ov / max(e - s, 1e-3) >= min_overlap:
            parts.append(t)
    return " ".join(parts)


def merge(stamps, max_len, max_gap):
    """Greedily join VAD chunks into segments no longer than max_len seconds."""
    segs = []
    for st in stamps:
        s, e = st["start"] / SR, st["end"] / SR
        if segs and e - segs[-1][0] <= max_len and s - segs[-1][1] <= max_gap:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    out = []
    for s, e in segs:  # hard-split anything VAD left too long
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/segments")
    ap.add_argument("--min-len", type=float, default=2.0)
    ap.add_argument("--max-len", type=float, default=20.0)
    ap.add_argument("--max-gap", type=float, default=0.6)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    meta = {}
    if (raw / "videos.jsonl").exists():
        for line in open(raw / "videos.jsonl", encoding="utf-8"):
            r = json.loads(line)
            meta[r["video_id"]] = r

    model = load_silero_vad()
    rows = []
    for wav in tqdm(sorted((raw / "audio").rglob("*.wav"))):
        vid = wav.stem
        info = meta.get(vid, {})
        audio = read_audio(str(wav), sampling_rate=SR)
        stamps = get_speech_timestamps(audio, model, sampling_rate=SR)
        vtt = next(wav.parent.glob(f"{vid}*.gu.vtt"), None)
        cues = parse_vtt(vtt) if vtt else []

        seg_dir = out / "audio" / wav.parent.name / vid
        seg_dir.mkdir(parents=True, exist_ok=True)
        for i, (s, e) in enumerate(merge(stamps, args.max_len, args.max_gap)):
            if e - s < args.min_len:
                continue
            path = seg_dir / f"{vid}_{i:05d}.wav"
            sf.write(path, audio[int(s * SR):int(e * SR)].numpy(), SR)
            row = {
                "audio_filepath": str(path),
                "duration": round(e - s, 3),
                "text": "",
                "video_id": vid,
                "start": round(s, 3),
                "end": round(e, 3),
                "channel": info.get("channel") or wav.parent.name,
                "speaker": info.get("speaker"),
                "dialect": info.get("dialect"),
                "source": "youtube" if info else "own",
            }
            if cues:
                row["subtitle"] = subtitle_for(cues, s, e)
            rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"{len(rows)} segments, {hours:.1f} h -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
