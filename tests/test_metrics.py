import pytest

from kgasr.metrics import bootstrap_ci, cer, edit_distance, paired_bootstrap, utterance_counts, wer


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    assert edit_distance(["a", "b"], ["a", "b"]) == 0


def test_wer_is_corpus_level():
    refs = ["એક બે ત્રણ ચાર", "પાંચ"]
    hyps = ["એક બે ત્રણ ચાર", "છ"]
    # 1 error / 5 words, not mean of per-utterance rates (0.5)
    assert wer(refs, hyps) == pytest.approx(0.2)


def test_normalisation_applied_before_scoring():
    assert wer(["કેમ છો?"], ["કેમ છો"]) == 0.0


def test_cer_ignores_spaces():
    assert cer(["ab cd"], ["abcd"]) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        wer(["a"], [])


def test_bootstrap_ci_contains_point_estimate():
    refs = ["એક બે ત્રણ"] * 20 + ["ચાર પાંચ"] * 20
    hyps = ["એક બે ત્રણ"] * 20 + ["ચાર છ"] * 20
    ci = bootstrap_ci(utterance_counts(refs, hyps, "word"), n_boot=200)
    assert ci.low <= ci.value <= ci.high


def test_paired_bootstrap_prefers_better_system():
    refs = ["એક બે ત્રણ"] * 30
    good = utterance_counts(refs, ["એક બે ત્રણ"] * 30, "word")
    bad = utterance_counts(refs, ["એક બે"] * 30, "word")
    assert paired_bootstrap(good, bad, n_boot=200) == 0.0
