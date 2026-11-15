"""L1/L2: spike -> syllable lattice (dictionary inventory DP with an unknown escape)."""

from tujia_nlp.syllable import phone_seq

MAX_LEN = 6
MAX_NORM = 0.45
UNKNOWN_COST = 0.5
TOKEN_COST = 0.6
# A syllable cannot span a long pause, otherwise distant spikes get merged.
MAX_GAP = 0.22
MAX_DUR = 0.80

_VOWELS = set("aiueoyɤɛɨɯɑãũẽĩɒæøœ")

_EQUIV = {}
for _pair in (
    ("a", "ɑ"), ("a", "ɛ"), ("o", "u"), ("o", "ɤ"), ("ɤ", "ə"), ("e", "ɛ"),
    ("i", "ɨ"), ("i", "y"), ("u", "y"), ("u", "ɯ"), ("s", "ʂ"), ("s", "ɕ"),
    ("ɕ", "ʃ"), ("z", "ʒ"), ("n", "ŋ"), ("m", "n"), ("l", "n"), ("j", "i"),
    ("w", "u"), ("x", "k"), ("x", "ɕ"), ("f", "p"), ("ts", "tʂ"), ("tɕ", "tʂ"),
):
    _EQUIV[_pair] = 0.4
    _EQUIV[(_pair[1], _pair[0])] = 0.4

_CHEAP = {"ʰ": 0.3, "ʔ": 0.3, "t": 0.6, "s": 0.6}


def _sub(a, b):
    if a == b:
        return 0.0
    return _EQUIV.get((a, b), 1.0)


def _dist(a, b):
    """Character-level weighted Levenshtein (``a``=recognised, ``b``=syllable)."""
    la, lb = len(a), len(b)
    if la == 0:
        return sum(_CHEAP.get(p, 1.0) for p in b)
    if lb == 0:
        return sum(_CHEAP.get(p, 1.0) for p in a)
    prev = [0.0] * (lb + 1)
    for j in range(1, lb + 1):
        prev[j] = prev[j - 1] + _CHEAP.get(b[j - 1], 1.0)
    for i in range(1, la + 1):
        cur = [prev[0] + _CHEAP.get(a[i - 1], 1.0)] + [0.0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(
                prev[j] + _CHEAP.get(a[i - 1], 1.0),
                cur[j - 1] + _CHEAP.get(b[j - 1], 1.0),
                prev[j - 1] + _sub(a[i - 1], b[j - 1]),
            )
        prev = cur
    return prev[lb]


def build(shapes):
    """Syllable inventory -> [(shape, phones)]; keep only vowel-bearing shapes.

    Sorted on purpose: ``segment`` keeps the first candidate that reaches a tie,
    so iterating a raw set would make the transcript depend on PYTHONHASHSEED and
    change between runs of the same audio.
    """
    out = []
    for shape in sorted(shapes):
        phones = phone_seq(shape)
        if not (1 <= len(phones) <= MAX_LEN):
            continue
        if not any(ch in _VOWELS for ch in phones):
            continue
        out.append((shape, phones))
    return out


def segment(spikes, lexicon, max_len=MAX_LEN, max_norm=MAX_NORM):
    """Return (tokens, unknown). Tokens carry shape, span and tone candidates."""
    items = [s for s in spikes if not s.get("is_boundary")]
    n = len(items)
    if n == 0:
        return [], []
    table = build(lexicon.shapes())

    best = [float("-inf")] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        if best[i] == float("-inf"):
            continue
        # Unknown escape: skip one spike.
        if best[i] - UNKNOWN_COST > best[i + 1]:
            best[i + 1] = best[i] - UNKNOWN_COST
            back[i + 1] = (i, "unknown", None, 0.0)
        for L in range(1, max_len + 1):
            j = i + L
            if j > n:
                break
            if L > 1 and (
                items[j - 1]["end"] - items[i]["start"] > MAX_DUR
                or items[j - 1]["start"] - items[j - 2]["end"] > MAX_GAP
            ):
                break
            seg_phones = [s["phone"] for s in items[i:j]]
            for shape, phones in table:
                if abs(len(phones) - L) > 2:
                    continue
                d = _dist(seg_phones, phones)
                norm = d / max(len(phones), L)
                if norm > max_norm:
                    continue
                score = (L - d) - TOKEN_COST
                if best[i] + score > best[j]:
                    best[j] = best[i] + score
                    back[j] = (i, shape, phones, score)

    tokens, unknown = [], []
    j = n
    while j > 0:
        entry = back[j]
        if entry is None:
            j -= 1
            continue
        i, shape, phones, score = entry
        if shape == "unknown":
            unknown.append(
                {
                    "t0": items[i]["start"],
                    "t1": items[i]["end"],
                    "phones": items[i]["phone"],
                    "reason": "no_hit",
                }
            )
        else:
            tok = {
                "t0": items[i]["start"],
                "t1": items[j - 1]["end"],
                "shape": shape,
                "ipa": shape,
                "score": round(score, 3),
                "tone_candidates": lexicon.tones_for(shape),
                "i": i,
                "j": j,
            }
            # The acoustic front-end may already provide a tone (CTC v0.2):
            # keep the most confident spike inside the syllable.
            toned = [(s.get("tone"), s.get("tone_conf") or 0.0)
                     for s in items[i:j] if s.get("tone")]
            if toned:
                tok["model_tone"], tok["model_tone_conf"] = max(toned, key=lambda x: x[1])
            tokens.append(tok)
        j = i
    tokens.reverse()
    unknown.reverse()

    # Spikes the inventory DP could not explain are still real audio. Emit them
    # as raw tokens with the shape left as recognised, instead of dropping them:
    # a word outside the lexicon used to vanish from the transcript entirely.
    covered = set()
    for tok in tokens:
        covered.update(range(tok["i"], tok["j"]))
    rest_idx = [k for k in range(n) if k not in covered]
    if rest_idx:
        raw = group_acoustic([items[k] for k in rest_idx])
        for tok in raw:
            tok["i"] = rest_idx[tok["i"]]
            tok["j"] = rest_idx[tok["j"] - 1] + 1
            tok["dict_hit"] = False
            tok["score"] = 0.0
        tokens.extend(raw)
        tokens.sort(key=lambda t: t["t0"])
    for tok in tokens:
        tok.setdefault("dict_hit", True)

    return tokens, _merge_unknown(unknown)


# A phone-level front-end emits aspiration as its own spike. No syllable in
# the inventory starts with it: aspiration belongs to a preceding stop, and
# when that stop is there the two land in the same group anyway. Left alone,
# a stray spike becomes a pseudo-syllable (`ʰu`, `ʰai`), which is what the
# frozen inventory caught on the v0.2 held-out set.
_BARE_ONSET = {"ʰ"}


def _drop_bare_onset(groups):
    """Remove an aspiration spike that opens a group with no stop in front."""
    out = []
    for g in groups:
        if g and g[0][1]["phone"] in _BARE_ONSET:
            g = g[1:]
        if g:
            out.append(g)
    return out


def group_acoustic(spikes, intervals=None, max_gap=MAX_GAP):
    """Group spikes into syllables without a lexicon.

    ``intervals`` are acoustic syllable nuclei (see :mod:`tujia_asr.nuclei`).
    Without them, phones are grouped by one vowel nucleus per syllable.
    Tokens are shaped like :func:`segment` output so downstream code is unchanged.
    """
    items = [s for s in spikes if not s.get("is_boundary")]
    if not items:
        return []

    if intervals:
        buckets = [[] for _ in intervals]
        for idx, s in enumerate(items):
            c = (s["start"] + s["end"]) / 2.0
            hit = None
            for k, (t0, t1) in enumerate(intervals):
                if t0 <= c <= t1:
                    hit = k
                    break
            if hit is None:
                hit = min(
                    range(len(intervals)),
                    key=lambda k: min(abs(c - intervals[k][0]), abs(c - intervals[k][1])),
                )
            buckets[hit].append((idx, s))
        raw_groups = [b for b in buckets if b]
    else:
        raw_groups = [[(idx, s) for idx, s in enumerate(items)]]

    # One vowel nucleus per syllable; consonants before a vowel join that
    # syllable (onset), not the previous one.  Pauses and long spans also split.
    def _split(g):
        vowels = [k for k, (_, s) in enumerate(g) if s["phone"] in _VOWELS]
        bounds = []
        for prev_v, nxt_v in zip(vowels, vowels[1:]):
            k = nxt_v
            while k - 1 > prev_v and g[k - 1][1]["phone"] not in _VOWELS:
                k -= 1
            bounds.append(k)
        pieces, start = [], 0
        for b in bounds + [len(g)]:
            if b > start:
                pieces.append(g[start:b])
            start = b

        out = []
        for piece in pieces:
            cur = []
            for item in piece:
                s = item[1]
                if cur and (s["start"] - cur[-1][1]["end"] > max_gap
                            or len(cur) >= MAX_LEN):
                    out.append(cur)
                    cur = []
                cur.append(item)
            if cur:
                out.append(cur)
        return out

    groups = []
    for g in raw_groups:
        groups.extend(_split(g))

    groups = _drop_bare_onset(groups)

    tokens = []
    for g in groups:
        shape = "".join(s["phone"] for _, s in g)
        if not shape:
            continue
        toned = [(s.get("tone"), s.get("tone_conf") or 0.0)
                 for _, s in g if s.get("tone")]
        tone, conf = max(toned, key=lambda x: x[1]) if toned else (None, 0.0)
        scores = [s.get("score") or 0.0 for _, s in g]
        tokens.append(
            {
                "t0": round(g[0][1]["start"], 3),
                "t1": round(g[-1][1]["end"], 3),
                "shape": shape,
                "ipa": shape,
                "score": round(sum(scores) / len(scores), 3),
                "tone_candidates": [],
                "model_tone": tone,
                "model_tone_conf": round(float(conf), 3),
                "i": g[0][0],
                "j": g[-1][0] + 1,
            }
        )
    return tokens


def _merge_unknown(rows):
    out = []
    for r in rows:
        if out and abs(r["t0"] - out[-1]["t1"]) <= 0.05:
            out[-1]["t1"] = r["t1"]
            out[-1]["phones"] = (out[-1]["phones"] + " " + r["phones"]).strip()
        else:
            out.append(dict(r))
    return out
