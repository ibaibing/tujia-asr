"""Tests for the inventory constraint on recognised syllables."""

import json

import pytest

from tujia_nlp import inventory as inventory_mod
from tujia_asr import postprocess


@pytest.fixture()
def inv(tmp_path):
    (tmp_path / "syllables-9.9.9.tsv").write_text(
        "syllable\tonset\trhyme\tstatus\tn_books\n"
        "liã\tl\tiã\tlegal\t3+\n"
        "piã\tp\tiã\tlegal\t3+\n"
        "ɣɤ\tɣ\tɤ\tlegal\t3+\n"
        "mɤ\tm\tɤ\tattested\t1\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"folds": {"ian": "iã", "e": "ɤ", "ye": "ɣɤ"}}), encoding="utf-8"
    )
    return inventory_mod.load(tmp_path / "syllables-9.9.9.tsv")


def test_check_counts_unknown(inv):
    toks = [{"shape": "liã"}, {"shape": "piã"}, {"shape": "zzz"}]
    r = postprocess.check(toks, inv=inv)
    assert (r.n_tokens, r.n_known, r.n_unknown) == (3, 2, 1)
    assert r.unknown == {"zzz": 1}
    assert r.unknown_rate == pytest.approx(1 / 3)


def test_fix_spelling_is_deterministic(inv):
    toks = [{"shape": "pian", "tone": 35, "score": 0.9}]
    out, r = postprocess.fix(toks, inv=inv)
    assert out[0]["shape"] == "piã"
    assert out[0]["shape_raw"] == "pian"
    assert out[0]["tone"] == 35 and out[0]["score"] == 0.9
    assert r.n_fixed_spelling == 1 and r.n_unfixed == 0


def test_fix_whole_syllable_rule(inv):
    out, r = postprocess.fix([{"shape": "ye"}], inv=inv)
    assert out[0]["shape"] == "ɣɤ"
    assert r.n_fixed_spelling == 1


def test_fix_does_not_guess_without_fuzzy(inv):
    out, r = postprocess.fix([{"shape": "kiɛ"}], inv=inv)
    assert out[0]["shape"] == "kiɛ"
    assert r.n_unfixed == 1 and r.n_fixed == 0


def test_fix_fuzzy_is_counted_separately(inv):
    out, r = postprocess.fix([{"shape": "kiɛ"}], inv=inv, fuzzy=True)
    assert r.n_fixed_fuzzy == 1
    assert r.n_fixed_spelling == 0
    assert out[0]["shape"] != "kiɛ"


def test_legal_only_treats_attested_as_unknown(inv):
    relaxed, _ = postprocess.fix([{"shape": "mɤ"}], inv=inv)
    strict, r = postprocess.fix([{"shape": "mɤ"}], inv=inv, legal_only=True)
    assert relaxed[0].get("shape_raw") is None
    assert r.n_unknown == 1 and r.n_unfixed == 1
    assert strict[0]["shape"] == "mɤ"


def test_input_is_not_mutated(inv):
    tok = {"shape": "pian", "tone": 21}
    postprocess.fix([tok], inv=inv)
    assert tok == {"shape": "pian", "tone": 21}


def test_real_inventory_normalises_training_spellings():
    """The v0.2 training labels write `e`/`ɛ` where the inventory has `ɤ`."""
    out, r = postprocess.fix([{"shape": "e", "tone": 35}, {"shape": "xɛ", "tone": 21}])
    assert [t["shape"] for t in out] == ["ɤ", "xɤ"]
    assert r.n_fixed_spelling == 2
    assert r.n_unfixed == 0
    assert r.version == inventory_mod.default().version
