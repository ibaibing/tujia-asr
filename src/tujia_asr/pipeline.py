"""Audio -> text: acoustic front-end, syllable decoding, optional word decoding."""

import os
from dataclasses import replace
from pathlib import Path

from tujia_nlp.lexicon import pack_entries

from . import decode, fuse, postprocess, second as second_front, syllabify, tone_acoustic
from .config import SUPPORTED_MODEL, Settings, duration_of
from .lexicon import Lexicon, read_entries, split_head


def _subtract(segments, words):
    """Remove word-covered time from the unknown spans."""
    out = []
    for u in segments:
        pieces = [(u["t0"], u["t1"])]
        for w in words:
            nxt = []
            for a, b in pieces:
                if w["t1"] <= a or w["t0"] >= b:
                    nxt.append((a, b))
                    continue
                if w["t0"] > a:
                    nxt.append((a, w["t0"]))
                if w["t1"] < b:
                    nxt.append((w["t1"], b))
            pieces = nxt
        for a, b in pieces:
            if b - a >= 0.08:
                out.append({**u, "t0": round(a, 3), "t1": round(b, 3)})
    return out


def _gaps(spans, dur, min_gap=0.2):
    covered = sorted((w["t0"], w["t1"]) for w in spans)
    out = []
    cur = 0.0
    for a, b in covered:
        if a - cur >= min_gap:
            out.append({"t0": round(cur, 3), "t1": round(a, 3), "phones": "", "reason": "gap"})
        cur = max(cur, b)
    if dur - cur >= min_gap:
        out.append({"t0": round(cur, 3), "t1": round(dur, 3), "phones": "", "reason": "gap"})
    return out


def _plausible_word(h, lat, max_gap=0.45):
    """Drop a word that bridges a long silence (cross-segment forced match)."""
    spikes = lat["spikes"]
    i, j = h.get("i", 0), h.get("j", 0)
    seg = spikes[i:j]
    if len(seg) < 2:
        return True
    gaps = [seg[k + 1]["start"] - seg[k]["end"] for k in range(len(seg) - 1)]
    return (max(gaps) if gaps else 0.0) <= max_gap


def _viterbi_zero(vit, lat):
    """Accept only exact (edit distance 0) Viterbi segments."""
    spikes = [s for s in lat["spikes"] if not s.get("is_boundary")]
    out = []
    for seg in vit.get("segments", []):
        if seg.get("unmatched") or not seg.get("candidates"):
            continue
        c = seg["candidates"][0]
        if c.get("dist") != 0.0:
            continue
        i, j = seg.get("start", 0), seg.get("end", 0)
        if not (0 <= i < j <= len(spikes)):
            continue
        out.append(
            {
                "t0": spikes[i]["start"], "t1": spikes[j - 1]["end"],
                "w": c["word"], "g": c.get("gloss", ""), "pos": c.get("pos", ""),
                "pg": c.get("page", ""), "q": " ".join(seg["phones"].split()),
                "score": 0.0, "avg": 1.5, "alt": 0, "floor": 0,
                "n_match": len(seg.get("phones", "").split()),
            }
        )
    return out


def _lattice(wav_path, settings, seg, search, device):
    if settings.frontend == "ctc":
        from . import ctc

        return ctc.lattice(
            str(wav_path), seg=seg, search=search, settings=settings,
            ckpt=settings.ckpt or None, device=device,
            encoder=settings.encoder or None,
        )
    if settings.frontend == "zipa":
        from . import zipa

        return zipa.lattice(str(wav_path), seg=seg, search=search, settings=settings)
    raise ValueError(f"未知前端：{settings.frontend}（可选 ctc / zipa）")


def _nuclei(wav_path, warnings):
    try:
        from . import nuclei as nuclei_mod

        return nuclei_mod.detect_file(wav_path)
    except Exception as e:  # librosa/scipy missing must not kill the pipeline
        warnings.append(f"nuclei: {type(e).__name__}")
        return None


def _uncovered(spikes, tokens):
    out = []
    for s in spikes:
        if s.get("is_boundary"):
            continue
        c = (s["start"] + s["end"]) / 2.0
        if any(t["t0"] <= c <= t["t1"] for t in tokens):
            continue
        out.append({"t0": s["start"], "t1": s["end"], "phones": s["phone"], "reason": "no_hit"})
    return out


def _words_from_chain(chain, lat, second):
    """ZIPA backend words (its lattice already did the lexical search)."""
    words = []
    for h in chain:
        if h.get("avg") is None:
            continue
        if not _plausible_word(h, lat):
            continue
        sup, basis, stext = fuse.support(h, second)
        tier = fuse.assign_tier(h["avg"], sup, second["status"] == "ok")
        words.append(
            {
                "t0": round(h["t0"], 3),
                "t1": round(h["t1"], 3),
                "ortho": h.get("w", ""),
                "ipa": h.get("w", ""),
                "han": h.get("g", ""),
                "pos": h.get("pos", ""),
                "pg": h.get("pg", ""),
                "zipa": {
                    "score": h.get("score"),
                    "avg": h.get("avg"),
                    "alt": h.get("alt", 0),
                    "floor": h.get("floor", 0),
                    "n_match": h.get("n_match", 0),
                },
                "second": {"support": sup, "basis": basis, "text": stext},
                "tier": tier,
                "q": h.get("q", ""),
                "n_syllables": len(split_head(h.get("w", ""))),
                "score": h.get("avg"),
            }
        )
    return words


def _tone_value(tok):
    tone = tok.get("tone")
    if tone is None:
        tone = tok.get("model_tone")
    if isinstance(tone, str) and tone.isdigit():
        return int(tone)
    return tone


def recognize(wav, *, dict_path=None, settings=None, mode=None, top=200,
              seg=10.0, search=2.0, tone_mode="prototype", tone=True,
              device="auto", second=True):
    """Recognise one audio file.

    Returns a JSON-serialisable dict; ``text`` is the transcript, ``words`` is
    empty when no lexicon was supplied and ``syllables`` is always populated.
    """
    settings = settings or Settings.from_env()
    if dict_path:
        paths = (dict_path,) if isinstance(dict_path, (str, os.PathLike)) else tuple(dict_path)
        settings = replace(
            settings,
            dict_paths=tuple(str(Path(p).expanduser()) for p in paths),
        )

    wav_path = Path(wav).expanduser()
    if not wav_path.is_file():
        raise FileNotFoundError(f"找不到音频：{wav_path}")
    mode = mode or settings.asr_mode
    dur = duration_of(str(wav_path))
    warnings = []

    lat = _lattice(wav_path, settings, seg, search, device)
    spikes = lat["spikes"]
    # User rows come first so that, when a headword is duplicated, the user's
    # row wins over the built-in one (see Lexicon.__init__).
    user_entries = []
    for p in settings.dict_paths:
        for e in read_entries(p):
            e.setdefault("source", Path(p).stem)
            user_entries.append(e)
    lexicon = Lexicon(user_entries)
    has_dict = bool(lexicon.entries)
    if settings.default_dict:
        # The open core vocabulary ships with tujia-nlp. Only the northern rows
        # are loaded: the recogniser is trained on northern Tujia, so southern
        # forms would only add words that cannot match.
        try:
            builtin = pack_entries(variety="north")
        except Exception as e:          # tujia-nlp too old, or a broken install
            builtin = []
            warnings.append(f"lexicon: built-in pack unavailable ({e})")
        if builtin:
            for e in builtin:
                e.setdefault("source", "tujia-open-core")
            lexicon.extend(builtin)
            has_dict = True
    if has_dict and lexicon.syllable_count() == 0:
        # Romanised packs (`BIFZIVSAR`) parse to nothing: `split_head` only reads
        # toned IPA (`sa⁵³zɿ²¹`). Without this the run silently returns no words.
        warnings.append(
            "lexicon: 0 syllables parsed from "
            + ", ".join(str(p) for p in settings.dict_paths)
            + " — needs toned IPA in `w` (e.g. sa⁵³zɿ²¹)"
        )
        has_dict = False   # fall back to acoustic grouping instead of emitting nothing

    if second:
        sec = second_front.run(str(wav_path), dur, mode=mode, settings=settings)
        if sec["status"] not in ("ok", "unavailable"):
            warnings.append(
                f"second: {sec['status']}" + (f" ({sec['note']})" if sec.get("note") else "")
            )
    else:
        sec = {
            "status": "unavailable", "mode": "none", "model": "", "text": "",
            "segments": [], "weight": 0.0, "note": "disabled",
        }

    chain = []
    if settings.frontend == "zipa" and has_dict:
        from . import zipa

        lex = zipa.lexdec(lat, top=top, settings=settings)
        chain = lex.get("chain", [])
        if not chain:
            vit = zipa.viterbi_fallback(spikes, settings=settings)
            chain = _viterbi_zero(vit, lat)
            warnings.append(f"zipa: lexdec empty, viterbi fallback ({len(chain)} exact)")

    if has_dict:
        tokens, unknown_raw = syllabify.segment(spikes, lexicon)
    else:
        tokens = syllabify.group_acoustic(spikes)
        unknown_raw = syllabify._merge_unknown(_uncovered(spikes, tokens))

    # The training transcriptions write some phones in a different shape than
    # the frozen syllable inventory (`e` for `ɤ`, `ɛ` for `ɤ`). Normalise to
    # the inventory spelling here, before word decoding, so `to_ye` and the
    # lexicon both see one form. Fuzzy snapping is deliberately off: it can
    # rewrite a real syllable.
    tokens, inv_report = postprocess.fix(tokens)
    if inv_report.n_fixed_spelling:
        warnings.append(
            f"inventory: {inv_report.n_fixed_spelling} syllable spelling(s) normalised"
        )
    if inv_report.n_unfixed:
        warnings.append(
            f"inventory: {inv_report.n_unfixed} syllable(s) not in the inventory "
            f"({', '.join(sorted(inv_report.unfixed)[:5])})"
        )
    inventory = {
        "version": inv_report.version,
        "n_tokens": inv_report.n_tokens,
        "n_normalised": inv_report.n_fixed_spelling,
        "n_unknown": inv_report.n_unfixed,
        "unknown": dict(sorted(inv_report.unfixed.items(), key=lambda kv: -kv[1])[:20]),
    }

    if has_dict and settings.frontend == "ctc":
        words = decode.segment_words(tokens, lexicon)
    elif has_dict and settings.frontend == "zipa":
        words = _words_from_chain(chain, lat, sec)
    else:
        words = []

    dropped = [w for w in words if w.get("tier") == "D"]
    if dropped:
        warnings.append(f"dropped {len(dropped)} word(s) below tier D")
    words = [w for w in words if w.get("tier") != "D"]

    track = None
    if tone:
        try:
            import librosa

            y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            track = tone_acoustic.f0_track(y, sr)
        except ImportError:
            warnings.append("librosa missing: F0 tone evidence disabled")

    bank_templates, bank_note = [], ""
    if tone and settings.tone_bank:
        try:
            from .tone_bank import load_bank

            bank_templates, bank_note = load_bank(settings)
            if bank_templates:
                warnings.append(bank_note)
            elif bank_note != "tone bank absent":
                warnings.append(bank_note)
        except Exception as e:
            warnings.append(f"tone bank: {type(e).__name__}")

    fuse.build_tokens(
        tokens, words, sec, track, lexicon,
        bank_templates=bank_templates or None, tone_mode=tone_mode,
    )

    syllables = []
    for t in tokens:
        shape = t.get("shape") or t.get("ipa") or ""
        value = _tone_value(t)
        ev = t.get("tone_evidence") or {}
        conf = (ev.get("model") or {}).get("conf")
        if conf is None:
            conf = t.get("model_tone_conf") or 0.0
        syllables.append(
            {
                "t0": round(t["t0"], 3),
                "t1": round(t["t1"], 3),
                "shape": shape,
                "ipa": f"{shape}{value}" if value is not None else shape,
                "tone": value,
                "tone_conf": round(float(conf), 3),
                "tier": t.get("tier", "B"),
                "in_lexicon": bool(t.get("dict_hit", True)),
            }
        )
    for t in tokens:
        t.pop("i", None)
        t.pop("j", None)

    # Spikes outside the inventory are kept as raw syllables (in_lexicon=False),
    # so the same span must not be reported as unknown audio as well.
    oov = [t for t in tokens if not t.get("dict_hit", True)]
    if oov:
        unknown_raw = [
            u for u in unknown_raw
            if not any(u["t0"] < t["t1"] and t["t0"] < u["t1"] for t in oov)
        ]

    unknown = unknown_raw + [
        {"t0": w["t0"], "t1": w["t1"], "phones": "", "reason": "tier_d"}
        for w in dropped
    ]
    unknown = _subtract(unknown, words)
    unknown += _gaps(words if words else tokens, dur)
    unknown.sort(key=lambda u: u["t0"])

    cov = fuse.coverage(lat, words if words else tokens, unknown)
    review = fuse.review_entries(words, tokens, unknown)

    # Assemble in time order: spans that became words use the dictionary
    # spelling, spans that did not fall back to the recognised syllables.
    # Joining only ``words`` drops unmatched syllables; joining only
    # ``syllables`` loses the words and their glosses.
    parts, k = [], 0
    for w in words:
        while k < w["token_i"]:
            parts.append(syllables[k]["ipa"])
            k += 1
        parts.append(w["ortho"])
        k = w["token_j"]
    parts.extend(s["ipa"] for s in syllables[k:])
    text = " ".join(p for p in parts if p)

    return {
        "schema": "tujia.asr/1",
        "text": text,
        "audio": {
            "path": str(wav_path),
            "dur": round(dur, 3),
            "sr": 16000,
            "segmented": lat["n_windows"] > 1,
            "windows": lat["n_windows"],
        },
        "model": {
            "name": SUPPORTED_MODEL if settings.frontend == "ctc" else str(lat.get("model", "")),
            "frontend": settings.frontend,
            "ckpt": str(settings.ckpt or ""),
            "encoder": str(settings.encoder or os.environ.get("TUJIA_XLSR", "")),
        },
        "frontends": {
            "lattice": {
                "backend": settings.frontend,
                "model": lat["model"],
                "n_spikes": sum(1 for s in spikes if not s.get("is_boundary")),
                "n_windows": lat["n_windows"],
                "elapsed_s": lat["elapsed_s"],
            },
            "second": {k: sec[k] for k in
                       ("status", "mode", "model", "text", "segments", "weight", "note")},
        },
        "syllables": syllables,
        "words": words,
        "tokens": tokens,
        "inventory": inventory,
        "unknown": unknown,
        "coverage": cov,
        "review": review,
        "warnings": warnings,
    }
