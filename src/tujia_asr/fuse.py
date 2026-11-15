"""L4: second-channel support, tiers, coverage, review list, token assembly."""

from . import tone as tone_mod
from . import tone_acoustic as ta
from tujia_nlp.zh_sound import class_edit, text_to_syllables

TIER_A_AVG = 1.5
TIER_A_SUPPORT = 0.5
TIER_C_SUPPORT = 0.6


def support(word, second):
    """Second-channel sound-class support for a word; (None, 'none', '') if absent."""
    if not second or second.get("status") != "ok":
        return None, "none", ""
    segs = [
        s for s in second.get("segments", [])
        if s.get("t1", 0) >= word["t0"] and s.get("t0", 0) <= word["t1"]
    ]
    if not segs:
        return None, "none", ""
    text = " ".join(s.get("text", "") for s in segs).strip()
    syls = text_to_syllables(text)
    if not syls:
        return None, "none", text
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        basis = "hanzi"
    elif second.get("mode") == "moss":
        basis = "pinyin"
    else:
        basis = "pinyin" if any(c.isalpha() for c in text) else "none"
    q = "".join(s["ipa"] for s in syls)
    z = word.get("q", "").replace(" ", "")
    if not z or not q:
        return None, basis, text
    dist = class_edit(z, q)
    val = max(0.0, min(1.0, 1.0 - dist / max(len(z), len(q))))
    return round(val, 3), basis, text


def assign_tier(avg, support_val, second_ok):
    if second_ok:
        if avg >= TIER_A_AVG and support_val is not None and support_val >= TIER_A_SUPPORT:
            return "A"
        if avg >= TIER_A_AVG:
            return "B"
        if support_val is not None and support_val >= TIER_C_SUPPORT:
            return "C"
    else:
        if avg >= TIER_A_AVG:
            return "B"
    return "D"


def _covers(word, tok):
    c = (tok["t0"] + tok["t1"]) / 2.0
    return word["t0"] <= c <= word["t1"]


def _hanzi_prior(tok, word, second):
    """Second-channel syllable tone -> a weak prior for this token."""
    if not second or second.get("status") != "ok" or not word:
        return None
    segs = [
        s for s in second.get("segments", [])
        if s.get("t1", 0) >= word["t0"] and s.get("t0", 0) <= word["t1"]
    ]
    if not segs:
        return None
    text = " ".join(s.get("text", "") for s in segs).strip()
    syls = text_to_syllables(text) or []
    if not syls:
        return None
    # Approximate in-word order; fall back to the nearest by midpoint.
    ordered = sorted(word.get("_tokens", []), key=lambda x: x["t0"])
    center = (tok["t0"] + tok["t1"]) / 2.0
    if len(syls) == len(ordered) and ordered:
        idx = min(
            range(len(ordered)),
            key=lambda k: abs((ordered[k]["t0"] + ordered[k]["t1"]) / 2.0 - center),
        )
    else:
        idx = 0
    syl = syls[min(idx, len(syls) - 1)]
    if syl.get("tone") is None:
        return None
    return {"value": syl["tone"], "strength": "weak", "char": syl.get("text", "")}


def _f0_window(tok, min_dur=0.12):
    """Spikes are short; widen the F0 window to 120 ms without moving the token."""
    dur = tok["t1"] - tok["t0"]
    if dur >= min_dur:
        return tok["t0"], tok["t1"]
    c = (tok["t0"] + tok["t1"]) / 2.0
    half = min_dur / 2.0
    return max(0.0, c - half), c + half


def build_tokens(tokens, words, second, track, lexicon, bank_templates=None,
                 tone_mode="prototype", tts_templates=None, speaker=None):
    """Attach tone evidence to syllable tokens; ``words`` supply dictionary priors."""
    for w in words:
        ordered = [t for t in tokens if _covers(w, t)]
        w["_tokens"] = ordered
        w["n_syllables"] = len(ordered)

    for tok in tokens:
        host = None
        for w in words:
            if _covers(w, tok):
                host = w
                break
        dict_tones = tok.get("tone_candidates") or []
        f0a, f0b = _f0_window(tok)
        feats = ta.syllable_features(track, f0a, f0b) if track else None
        bank = None
        if bank_templates:
            from .tone_bank import match as bank_match

            bank = bank_match(tok["shape"], feats, bank_templates, speaker=speaker)
        hanzi = _hanzi_prior(tok, host, second)
        model = None
        if tok.get("model_tone") is not None:
            # The front-end gives "35"; the engine works with integer 35.
            mt = tok["model_tone"]
            model = {"tone": int(mt) if str(mt).isdigit() else mt,
                     "conf": tok.get("model_tone_conf") or 0.0}
        final, ev = tone_mod.decide(
            tok["shape"], dict_tones, feats=feats, bank=bank, hanzi=hanzi,
            tone_mode=tone_mode, tts_templates=tts_templates, model=model,
        )
        tok["tone"] = final
        tok["tone_evidence"] = ev
        if final is None:
            tok["tone_candidates"] = tone_mod.rank_candidates(
                dict_tones, bank, feats, hanzi, model=model)
        else:
            tok["tone_candidates"] = [final] + [t for t in dict_tones if t != final]
        tok["tier"] = host.get("tier", "D") if host else "D"

    for w in words:
        w.pop("_tokens", None)
    return tokens


def coverage(lattice, words, unknown, tier_a_only=False):
    total = max(lattice.get("n_frames", 0), 1)
    frames_a = 0
    explained = 0
    for w in words:
        f = int(round((w["t1"] - w["t0"]) / 0.01))
        explained += f
        if w.get("tier") == "A":
            frames_a += f
    if not tier_a_only:
        for u in unknown:
            # Silence is not "explained" — counting it made the ratio 1.0 for
            # any file, however little was recognised.
            if u.get("reason") == "gap":
                continue
            explained += int(round((u["t1"] - u["t0"]) / 0.01))
    explained = min(explained, total)
    return {
        "frames_total": total,
        "frames_tier_a": frames_a,
        "frames_explained": explained,
        "ratio": round(explained / total, 3),
        "tier_a_ratio": round(frames_a / total, 3),
    }


def review_entries(words, tokens, unknown):
    out = []
    for w in words:
        if w.get("tier") == "C":
            out.append(
                {"t0": w["t0"], "t1": w["t1"], "why": "tier C: second-only",
                 "candidates": [w.get("ortho", "")]}
            )
    for u in unknown:
        out.append(
            {"t0": u["t0"], "t1": u["t1"], "why": u.get("reason", "unknown"),
             "candidates": []}
        )
    for t in tokens:
        ev = t.get("tone_evidence") or {}
        if ev.get("conflict"):
            out.append(
                {"t0": t["t0"], "t1": t["t1"], "why": "tone: dialect_or_error",
                 "candidates": [t["ipa"], str(t.get("tone"))]}
            )
    return out
