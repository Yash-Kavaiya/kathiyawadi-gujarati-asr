from kgasr.filtering import FilterConfig, judge, pair_cer

CFG = FilterConfig()
SENT = "આજે સવારે હું રાજકોટ ગયો હતો"  # ~22 Gujarati chars without spaces


def test_identical_hypotheses_kept():
    keep, reason, d = judge(SENT, SENT + "।", duration=3.0, cfg=CFG)
    assert keep and reason == "ok" and d == 0.0


def test_disagreement_rejected():
    keep, reason, _ = judge(SENT, "કંઈક બીજું જ બોલાયું", duration=3.0, cfg=CFG)
    assert not keep and reason == "disagree"


def test_empty_rejected():
    assert judge("", SENT, 3.0, CFG)[1] == "empty"


def test_non_gujarati_rejected():
    assert judge("hello how are you", "hello how are you", 2.0, CFG)[1] == "not_gujarati"


def test_char_rate_catches_hallucination():
    # 22 chars in 30 s of audio: model missed most of the speech
    assert judge(SENT, SENT, 30.0, CFG)[1] == "char_rate"


def test_pair_cer_symmetric():
    assert pair_cer("abcd", "abce") == pair_cer("abce", "abcd") == 0.25
