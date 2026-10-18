"""Download audio (+ Gujarati subtitles if any) from YouTube channels/playlists.

Only use channels you own or have written permission for (see README, Data rights).

  python data_pipeline/download.py --sources configs/sources.txt --out data/raw

`sources.txt`: one URL per line, optional `| speaker=..., dialect=...` suffix, `#` comments.
Re-running is safe: finished videos are recorded in a download archive and skipped.
"""

import argparse
import json
from pathlib import Path

import yt_dlp


def parse_sources(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url, _, meta = line.partition("|")
        tags = dict(kv.strip().split("=", 1) for kv in meta.split(",") if "=" in kv)
        yield url.strip(), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-videos", type=int, default=None, help="per source, for dry runs")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out / "videos.jsonl"

    for url, tags in parse_sources(args.sources):
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(out / "audio" / "%(channel_id)s" / "%(id)s.%(ext)s"),
            "download_archive": str(out / "archive.txt"),
            "ignoreerrors": True,
            "writesubtitles": True,
            "writeautomaticsub": False,  # auto-captions are too noisy to be useful
            "subtitleslangs": ["gu"],
            "subtitlesformat": "vtt",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": {"extractaudio": ["-ar", "16000", "-ac", "1"]},
            "playlistend": args.max_videos,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        entries = info.get("entries") or [info]
        with open(meta_path, "a", encoding="utf-8") as f:
            for e in entries:
                if not e:
                    continue
                row = {
                    "video_id": e["id"],
                    "channel": e.get("channel_id"),
                    "channel_name": e.get("channel"),
                    "title": e.get("title"),
                    "duration": e.get("duration"),
                    "source_url": url,
                    **tags,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"done: {url} ({len(entries)} videos)")


if __name__ == "__main__":
    main()
