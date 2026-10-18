import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))

from filter import run  # noqa: E402

from kgasr.filtering import FilterConfig  # noqa: E402

SENT = "આજે સવારે હું રાજકોટ ગયો હતો"


def test_subtitle_preferred_as_label_when_it_agrees():
    rows = [{"audio_filepath": "a.wav", "duration": 3.0, "hyp_a": SENT + " ને",
             "hyp_b": "કંઈ બીજું", "subtitle": SENT + " ને।"}]
    kept, reasons = run(rows, "hyp_a", "hyp_b", FilterConfig())
    assert len(kept) == 1
    assert kept[0]["label"] == "subtitle" and kept[0]["text"].endswith("।")


def test_falls_back_to_second_model_without_subtitle():
    rows = [{"audio_filepath": "a.wav", "duration": 3.0, "hyp_a": SENT, "hyp_b": "કંઈ બીજું જ બોલાયું"}]
    kept, reasons = run(rows, "hyp_a", "hyp_b", FilterConfig())
    assert kept == [] and reasons["disagree"] == 1
