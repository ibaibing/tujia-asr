import json
import os
import subprocess
import sys
from pathlib import Path

from tujia_asr import decode, fuse, syllabify, tone_acoustic
from tujia_asr.config import Settings
from tujia_asr.lexicon import Lexicon


def _spike(phone, start, end, tone=None, conf=0.0):
    return {
        "phone": phone,
        "start": start,
        "end": end,
        "is_boundary": False,
        "score": 0.9,
        "tone": tone,
        "tone_conf": conf,
    }


def test_settings_defaults_without_dictionary():
    settings = Settings.from_env({})
    assert settings.frontend == "ctc"
    assert settings.dict_paths == ()
    assert settings.auth_headers() == {}


def test_settings_reads_dictionary_list():
    settings = Settings.from_env({"TUJIA_DICTS": "a:b"})
    assert settings.dict_paths == ("a", "b")


def test_empty_lexicon_and_empty_spikes():
    lexicon = Lexicon.load(None)
    assert lexicon.entries == []
    assert lexicon.shapes() == set()
    assert syllabify.segment([], lexicon) == ([], [])


def test_group_acoustic_splits_at_pause():
    spikes = [
        _spike("m", 0.00, 0.10),
        _spike("a", 0.10, 0.25, "55", 0.9),
        _spike("n", 0.55, 0.60),
        _spike("i", 0.60, 0.75, "21", 0.8),
    ]
    tokens = syllabify.group_acoustic(spikes)
    assert [t["shape"] for t in tokens] == ["ma", "ni"]
    assert tokens[0]["model_tone"] == "55"
    assert tokens[1]["t0"] == 0.55


def test_group_acoustic_one_vowel_per_syllable():
    spikes = [
        _spike("n", 0.00, 0.05),
        _spike("i", 0.05, 0.15),
        _spike("s", 0.15, 0.20),
        _spike("a", 0.20, 0.30),
    ]
    tokens = syllabify.group_acoustic(spikes)
    assert [t["shape"] for t in tokens] == ["ni", "sa"]


def test_group_acoustic_uses_nuclei_intervals():
    spikes = [_spike("a", 0.00, 0.10), _spike("ji", 0.20, 0.30)]
    tokens = syllabify.group_acoustic(spikes, intervals=[(0.0, 0.15), (0.15, 0.35)])
    assert [t["shape"] for t in tokens] == ["a", "ji"]


def test_tone_prototype_level_tone():
    feats = {"contour": [3, 3, 3, 3, 3], "dur": 0.2, "voiced_ratio": 0.9}
    tone, conf = tone_acoustic.classify(feats)
    assert tone == 55
    assert conf > 0


def test_decode_two_syllable_word(tmp_path):
    lexicon_path = tmp_path / "lexicon.jsonl"
    entries = [
        {"w": "a⁵⁵ji²¹", "g": "test", "pos": "n"},
        {"w": "ma⁵⁵", "g": "mother", "pos": "n"},
    ]
    lexicon_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )
    lexicon = Lexicon.load(lexicon_path)
    tokens = [
        {"shape": "a", "t0": 0.0, "t1": 0.1, "model_tone": "55"},
        {"shape": "ji", "t0": 0.1, "t1": 0.25, "model_tone": "21"},
    ]
    words = decode.segment_words(tokens, lexicon)
    assert len(words) == 1
    assert words[0]["ortho"] == "a⁵⁵ji²¹"
    assert words[0]["n_syllables"] == 2
    assert words[0]["tier"] == "A"


def test_word_does_not_bridge_silence():
    lexicon = Lexicon([{"w": "ma⁵⁵ni²¹", "g": "x"}])
    tokens = [
        {"shape": "ma", "t0": 0.0, "t1": 0.2, "model_tone": "55"},
        {"shape": "ni", "t0": 5.0, "t1": 5.2, "model_tone": "21"},
    ]
    assert decode.segment_words(tokens, lexicon) == []


def test_coverage_does_not_count_silence():
    lattice = {"n_frames": 100}
    words = [{"t0": 0.0, "t1": 0.5, "tier": "B"}]
    unknown = [{"t0": 0.5, "t1": 1.0, "reason": "gap"}]
    cov = fuse.coverage(lattice, words, unknown)
    assert cov["frames_explained"] == 50
    assert cov["ratio"] == 0.5


def test_builtin_lexicon_is_added_to_user_rows():
    from tujia_nlp.lexicon import pack_entries

    lex = Lexicon([{"w": "ma⁵⁵", "g": "x"}])
    lex.extend(pack_entries(variety="north"))
    assert lex.word_counts["ma⁵⁵"] == 1          # user row survives
    assert lex.syllable_count() > 100            # built-in syllables present
    assert len(pack_entries(variety="south")) < len(pack_entries())


def test_default_dict_can_be_turned_off():
    assert Settings.from_env({}).default_dict is True
    assert Settings.from_env({"TUJIA_NO_DEFAULT_DICT": "1"}).default_dict is False


def test_unmatched_spikes_are_kept_verbatim():
    """A syllable outside the inventory must survive with its recognised shape."""
    lex = Lexicon([{"w": "ma⁵⁵", "g": "x"}])
    spikes = [
        {"start": 0.00, "end": 0.05, "phone": "k", "score": 0.9},
        {"start": 0.05, "end": 0.20, "phone": "a", "score": 0.9},
    ]
    tokens, _ = syllabify.segment(spikes, lex)
    assert "".join(t["shape"] for t in tokens) == "ka"
    assert tokens[0]["dict_hit"] is False


def test_first_lexicon_wins_on_duplicate_headword():
    from tujia_nlp.lexicon import pack_entries

    head = pack_entries(variety="north")[0]["w"]
    lex = Lexicon([{"w": head, "g": "用户"}] + [{"w": head, "g": "内置"}])
    hits = [e["g"] for e in lex.entries if e["w"] == head]
    assert hits == ["用户"]
    assert lex.word_counts[head] == 1


def test_cli_help_runs():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    proc = subprocess.run(
        [sys.executable, "-m", "tujia_asr", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert "tujia-asr" in proc.stdout
