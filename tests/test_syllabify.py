"""Tests for phone-to-syllable grouping, in particular stray aspiration."""

from tujia_asr import syllabify


def _spikes(*phones):
    return [
        {"phone": p, "start": i * 0.2, "end": i * 0.2 + 0.15, "score": 0.5}
        for i, p in enumerate(phones)
    ]


def test_bare_aspiration_is_not_a_syllable():
    toks = syllabify.group_acoustic(_spikes("ʰ", "u"))
    assert [t["shape"] for t in toks] == ["u"]


def test_aspiration_after_stop_stays_inside_the_syllable():
    toks = syllabify.group_acoustic(_spikes("t", "s", "ʰ", "u"))
    assert [t["shape"] for t in toks] == ["tsʰu"]


def test_aspiration_does_not_leak_into_previous_syllable():
    toks = syllabify.group_acoustic(_spikes("p", "a", "ʰ", "u"))
    assert [t["shape"] for t in toks] == ["pa", "u"]


def test_ordinary_grouping_is_unchanged():
    toks = syllabify.group_acoustic(_spikes("l", "a", "m", "ɤ"))
    assert [t["shape"] for t in toks] == ["la", "mɤ"]
