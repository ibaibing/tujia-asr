"""E0-E2 tone fusion: tone bank (primary) -> F0 prototype (fallback) -> null."""

from . import tone_acoustic as ta


def decide(shape, dict_tones, feats=None, bank=None, hanzi=None,
           tone_mode="prototype", tts_templates=None, model=None):
    """Return (tone, evidence) with a five-way evidence record.

    ``model`` is the tone evidence from the acoustic front-end (the CTC v0.2
    auxiliary head), shaped as {"tone", "conf"}.
    """
    bank_ev = {
        "shape": shape,
        "tone": None,
        "conf": 0.0,
        "dist": None,
        "speaker": None,
        "n_templates": 0,
    }
    if bank:
        bank_ev.update({k: bank.get(k) for k in ("tone", "conf", "dist", "speaker", "n_templates")})
        bank_ev["shape"] = shape

    ac_ev = {
        "value": None,
        "conf": 0.0,
        "method": None,
        "f0_median_hz": None,
        "slope_st_100ms": None,
        "range_st": None,
        "dur": None,
        "voiced_ratio": None,
    }
    if feats:
        tone, conf = ta.classify(feats)
        method = "f0-prototype"
        if tone_mode in ("tts", "both") and tts_templates:
            ttone, tconf = ta.template_match(feats, tts_templates)
            if tone_mode == "tts" or (ttone is not None and tconf > conf):
                tone, conf, method = ttone, tconf, "tts-template"
        ac_ev.update(
            {
                "value": tone,
                "conf": conf,
                "method": method,
                "f0_median_hz": feats.get("f0_median_hz"),
                "slope_st_100ms": feats.get("slope_st_100ms"),
                "range_st": feats.get("range_st"),
                "dur": feats.get("dur"),
                "voiced_ratio": feats.get("voiced_ratio"),
            }
        )
    elif bank:
        ac_ev["method"] = "f0-prototype"

    final, method, used_conf = None, None, 0.0
    model_ev = {
        "value": (model or {}).get("tone"),
        "conf": round(float((model or {}).get("conf") or 0.0), 3),
    }
    # The tone head always participates (ctc_front already skipped "none");
    # confidence gating was measured to hurt: fallback F0 is only ~0.30.
    if model_ev["value"] is not None:
        final, method, used_conf = model_ev["value"], "ctc-tone", model_ev["conf"]
    elif bank_ev["tone"] is not None and bank_ev["conf"] >= 0.5:
        final, method, used_conf = bank_ev["tone"], "tone-bank", bank_ev["conf"]
    elif ac_ev["value"] is not None and ac_ev["conf"] >= 0.5:
        final, method, used_conf = ac_ev["value"], ac_ev["method"], ac_ev["conf"]

    dict_ev = {
        "candidates": list(dict_tones or []),
        "agree": (final in dict_tones) if (final is not None and dict_tones) else None,
    }
    hanzi_ev = {
        "value": (hanzi or {}).get("value"),
        "strength": (hanzi or {}).get("strength", "none"),
        "char": (hanzi or {}).get("char", ""),
    }
    conflict = None
    thr = 0.8 if method == "ctc-tone" else 0.6
    if final is not None and dict_tones and final not in dict_tones and used_conf >= thr:
        conflict = "dialect_or_error"

    evidence = {
        "model": model_ev,
        "bank": bank_ev,
        "acoustic": ac_ev,
        "dict": dict_ev,
        "hanzi": hanzi_ev,
        "final": final,
        "method": method,
        "conflict": conflict,
    }
    return final, evidence


def rank_candidates(dict_tones, bank, feats, hanzi, model=None):
    """Reorder candidates when both stages are weak."""
    out = []
    for t in dict_tones or []:
        if t not in out:
            out.append(t)
    for t in ((model or {}).get("value"), (bank or {}).get("tone")):
        if t is not None and t not in out:
            out.append(t)
    if feats:
        tone, _ = ta.classify(feats)
        if tone is not None and tone not in out:
            out.append(tone)
    if hanzi and hanzi.get("value") is not None and hanzi["value"] not in out:
        out.append(hanzi["value"])
    return out
