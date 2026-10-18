from kgasr.manifest import find_leaks, read_manifest, split_by_group, write_manifest


def _rows():
    rows = []
    for ch in range(10):
        for i in range(5):
            rows.append({"audio_filepath": f"c{ch}_{i}.wav", "duration": 10.0,
                         "text": "x", "channel": f"ch{ch}", "speaker": f"spk{ch}"})
    return rows


def test_roundtrip(tmp_path):
    p = tmp_path / "m.jsonl"
    rows = [{"audio_filepath": "a.wav", "duration": 1.0, "text": "કેમ છો"}]
    assert write_manifest(p, rows) == 1
    assert read_manifest(p) == rows


def test_split_has_no_leaks_and_covers_everything():
    rows = _rows()
    splits = split_by_group(rows, {"train": 0.8, "dev": 0.1, "test": 0.1}, seed=1)
    assert find_leaks(splits) == []
    assert sum(len(v) for v in splits.values()) == len(rows)
    assert all(splits[s] for s in splits)


def test_find_leaks_detects_shared_speaker():
    splits = {"train": [{"speaker": "a"}], "test": [{"speaker": "a"}]}
    assert find_leaks(splits) == ["speaker='a' in both train and test"]
