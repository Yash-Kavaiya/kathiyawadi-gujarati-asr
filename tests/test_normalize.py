from kgasr.normalize import is_mostly_gujarati, normalize


def test_punctuation_and_danda_removed():
    assert normalize("કેમ છો? હું મજામાં છું।") == "કેમ છો હું મજામાં છું"


def test_gujarati_digits_to_ascii():
    assert normalize("૨૦૨૬ માં") == "2026 માં"


def test_split_vowel_sign_composed():
    # ક + ા + ે typed separately must equal ક + ો
    assert normalize("કાે") == normalize("કો")
    assert normalize("કેા") == "કો"


def test_nukta_and_candrabindu_folded():
    assert normalize("જ઼ર") == "જર"
    assert normalize("હાઁ") == "હાં"


def test_flags_can_disable_folding():
    assert "઼" in normalize("જ઼ર", remove_nukta=False)


def test_zero_width_and_whitespace():
    assert normalize("  ક‍મ   \n છો ") == "કમ છો"


def test_code_mixed_latin_lowercased():
    assert normalize("મારો Phone નંબર") == "મારો phone નંબર"


def test_virama_kept():
    # ક્ષ uses virama (U+0ACD), a combining mark that must survive
    assert normalize("ક્ષત્રિય") == "ક્ષત્રિય"


def test_is_mostly_gujarati():
    assert is_mostly_gujarati("આ ગુજરાતી છે")
    assert not is_mostly_gujarati("this is english")
    assert not is_mostly_gujarati("")
