"""Lexicon-constrained word decoding over recognised syllables.

The released CTC model already emits syllables, so word decoding is a
syllable-sequence segmentation problem: find the dictionary word sequence that
best explains the syllable tokens.  This replaces the external ZIPA lattice
decoder, keeping the package self-contained.

Scoring per candidate word (``k`` syllables):

    sum(1 - normalized_shape_distance)   per syllable
  + TONE_BONUS / TONE_MISMATCH           per syllable with a known tone
  + WORD_COST                            once per word (keeps tokens together)
  + 0.1 * log1p(word frequency)          tie-breaker

Unmatched syllables take a fixed ``UNKNOWN_COST`` and are reported separately.
"""

import math
from collections import defaultdict

from .syllabify import _dist

MAX_WORD_SYL = 6
SHAPE_MAX_NORM = 0.40
MAX_SYL_GAP = 0.45         # a word cannot bridge a longer silence
UNKNOWN_COST = 0.55
TONE_BONUS = 0.20
TONE_MISMATCH = -0.10
WORD_COST = -0.10          # per word; avoids splitting a real word into pieces

_ND_CACHE = {}


def _nd(a, b):
    """Normalized character distance between two syllable shapes."""
    key = (a, b)
    value = _ND_CACHE.get(key)
    if value is None:
        value = 1.0 if not a or not b else _dist(a, b) / max(len(a), len(b), 1)
        _ND_CACHE[key] = value
    return value


def _entries(lexicon):
    """Dictionary entries paired with their syllable sequence."""
    from .lexicon import split_head

    out = []
    for e in lexicon.entries:
        syls = split_head(e.get("w", ""))
        if not syls or len(syls) > MAX_WORD_SYL:
            continue
        if not all(shape for shape, _ in syls):
            continue
        out.append((e, syls))
    return out


def build_index(lexicon):
    """Lexicon -> {first syllable shape: {length: [(entry, syllables)]}}."""
    index = defaultdict(lambda: defaultdict(list))
    for e, syls in _entries(lexicon):
        index[syls[0][0]][len(syls)].append((e, syls))
    return {shape: dict(by_len) for shape, by_len in index.items()}


def _index(lexicon):
    cached = getattr(lexicon, "_asr_word_index", None)
    if cached is None:
        cached = build_index(lexicon)
        lexicon._asr_word_index = cached
    return cached


def _tone_of(tok):
    tone = tok.get("model_tone")
    return tone if tone is not None else tok.get("tone")


def _match(tokens, i, syls):
    """Score a word covering ``syls`` at token ``i``; None when it does not fit."""
    sim = 0.0
    tone_term = 0.0
    shape_exact = True
    tone_ok = True
    for m, (shape, tone) in enumerate(syls):
        tok = tokens[i + m]
        if tok.get("dict_hit") is False:
            # A syllable the inventory could not explain is kept in the output
            # verbatim; it must not be absorbed into a neighbouring word either.
            return None
        if m and tok["t0"] - tokens[i + m - 1]["t1"] > MAX_SYL_GAP:
            return None
        nd = _nd(tok.get("shape") or tok.get("ipa") or "", shape)
        if nd > SHAPE_MAX_NORM:
            return None
        if nd > 0.0:
            shape_exact = False
        sim += 1.0 - nd
        observed = _tone_of(tok)
        if tone is not None and observed is not None:
            if str(tone) == str(observed):
                tone_term += TONE_BONUS
            else:
                tone_term += TONE_MISMATCH
                tone_ok = False
        elif tone is not None:
            tone_ok = False
    return sim, tone_term, shape_exact, tone_ok


def segment_words(tokens, lexicon, max_syl=MAX_WORD_SYL):
    """Viterbi word segmentation over syllable tokens.

    Returns word dicts with ``t0``, ``t1``, ``ortho``, ``ipa``, ``han``,
    ``pos``, ``pg``, ``score``, ``n_syllables``, ``tier``, ``token_i`` and
    ``token_j``.
    """
    tokens = list(tokens)
    n = len(tokens)
    if n == 0 or not lexicon.entries:
        return []
    index = _index(lexicon)
    neg = float("-inf")
    best = [neg] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0.0

    for i in range(n):
        if best[i] == neg:
            continue
        # Unknown: consume one syllable.
        cand = best[i] - UNKNOWN_COST
        if cand > best[i + 1]:
            best[i + 1] = cand
            back[i + 1] = (i, None, None, None, cand)

        shape0 = tokens[i].get("shape") or tokens[i].get("ipa") or ""
        firsts = []
        for shape, by_len in index.items():
            nd = _nd(shape0, shape)
            if nd <= SHAPE_MAX_NORM:
                firsts.append(by_len)
        if not firsts:
            continue

        kmax = min(max_syl, n - i)
        for by_len in firsts:
            for k in range(1, kmax + 1):
                for entry, syls in by_len.get(k, ()):  # len(syls) == k
                    scored = _match(tokens, i, syls)
                    if scored is None:
                        continue
                    sim, tone_term, shape_exact, tone_ok = scored
                    freq = 0.1 * math.log1p(lexicon.word_counts.get(entry.get("w", ""), 0))
                    total = best[i] + sim + tone_term + WORD_COST + freq
                    j = i + k
                    if total > best[j]:
                        best[j] = total
                        back[j] = (i, entry, syls, (shape_exact, tone_ok), total)

    words = []
    j = n
    while j > 0:
        item = back[j]
        if item is None:
            j -= 1
            continue
        i, entry, syls, meta, total = item
        if entry is None:
            j = i
            continue
        shape_exact, tone_ok = meta
        # ``ortho`` is the dictionary spelling; ``ipa`` is what was actually
        # recognised. Keeping them apart means a tone mismatch stays visible
        # instead of being papered over by the dictionary form.
        heard = []
        for m in range(len(syls)):
            tok = tokens[i + m]
            shape = tok.get("shape") or tok.get("ipa") or ""
            tone = _tone_of(tok)
            heard.append(f"{shape}{tone}" if tone is not None else shape)
        words.append(
            {
                "t0": tokens[i]["t0"],
                "t1": tokens[i + len(syls) - 1]["t1"],
                "ortho": entry.get("w", ""),
                "ipa": " ".join(heard),
                "tone_model": [_tone_of(tokens[i + m]) for m in range(len(syls))],
                "tone_dict": [tone for _, tone in syls],
                "han": entry.get("g", ""),
                "pos": entry.get("pos", ""),
                "pg": entry.get("pg", ""),
                "source": entry.get("source", ""),
                "score": round(total - best[i], 3),
                "n_syllables": len(syls),
                "tier": "A" if (shape_exact and tone_ok) else "B",
                "token_i": i,
                "token_j": i + len(syls),
            }
        )
        j = i
    words.reverse()
    return words
