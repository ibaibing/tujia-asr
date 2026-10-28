# Evaluation protocol

Every number quoted in the README and in the archives is produced by one of the two
protocols below. They measure different things and are **not interchangeable**: the
same decode scores 89.90% under recall and 0.4634 SER under the strict protocol.
Always name the protocol when quoting a number.

## Corpus and splits

| Item | Value |
| --- | --- |
| Audio | 685 clips, 16 kHz mono, 2.571 hours, single third-party collection, Northern dialect |
| Reference labels | toned IPA syllables taken from the collection's own annotation, one per syllable |
| Total reference syllables | 20,568 (17,295 train + 3,273 held out) |
| Held-out groups | `wav21`, `wav32`, `wav34`, `wav51`, `wav53` — 108 clips |
| Training groups | the remaining 577 clips |
| Dev / test (v0.2 report only) | dev = 43 clips from `wav21`; test = the other 65 held-out clips |

The corpus is not distributed. To apply the protocol to your own recordings, produce
the manifest described in [data-format.md](data-format.md).

## Protocol A — strict SER

The unit is the full model label: a toned syllable such as `tsʰeŋ21`.

1. Greedy-decode the clip into a label sequence.
2. Align reference and hypothesis with Needleman–Wunsch, substitution cost 1,
   insertion and deletion cost 1, on the **raw** labels (no normalization).
3. `SER = edit distance / reference length`.

Tone error is the same computation restricted to the tone substring of each label
(`21`, `35`, `53`, `55`).

Reported for v0.2 only:

| Configuration | dev SER | test SER |
| --- | --- | --- |
| Fine-tuned + greedy | 0.6646 | 0.4767 |
| Fine-tuned + syllable bigram (α = 1.5) | 0.6555 | 0.4634 |

`val SER` in the model card is a slightly different subset: sequence-level SER over
the 108 held-out clips, best epoch 18, value 0.5527.

## Protocol B — lenient syllable recall

Used for v0.1, for the λ_tone ablation, and for any cross-version comparison.

1. Split each reference label into shape and tone; keep shapes only
   (`la35` → `la`).
2. Normalize shapes (table below).
3. Align with Needleman–Wunsch where the substitution cost of two shapes is the
   normalized Levenshtein distance of their character strings:
   `cost(a, b) = lev(a, b) / max(len(a), len(b))`.
4. A pair is a hit when `1 - cost ≥ 0.5`, i.e. edit similarity ≥ 0.5.
5. `recall = hits / reference syllables`, `precision = hits / predicted syllables`,
   `F1 = 2PR / (P + R)`.

Shape normalization, applied in this order:

| From | To | From | To |
| --- | --- | --- | --- |
| `ʨ` `ʧ` `tʃ` | `tɕ` | `ɨ` `z̩` | `ɿ` |
| `ʦ` | `ts` | `ʁ` | `ɣ` |
| `ç` `ʃ` | `ɕ` | `ã` `ĩ` `ũ` `ẽ` `õ` | `a` `i` `u` `e` `o` |
| — | — | `aŋ` `eŋ` `iŋ` `oŋ` `uŋ` | `a` `e` `i` `o` `u` |

Results:

| Checkpoint | all 685 | held-out 108 | train 577 |
| --- | --- | --- | --- |
| v0.1 | 89.82 / 92.89 / 91.33 | 80.26 / 84.33 / 82.25 | 91.63 / 94.48 / 93.03 |
| v0.2 | 89.90 / 90.52 / 90.21 | 83.93 / 84.37 / 84.15 | 91.03 / 91.68 / 91.36 |

Cells are `recall / precision / F1` in percent. Tone accuracy is only defined for
v0.2: 80.64% over all 685 clips, 65.59% on the held-out 108.

## Reference implementation

Protocol B, stdlib only:

```python
import re

ORTH = [("ʨ", "tɕ"), ("ʧ", "tɕ"), ("tʃ", "tɕ"), ("ʦ", "ts"), ("ç", "ɕ"), ("ʃ", "ɕ"),
        ("ɨ", "ɿ"), ("z̩", "ɿ"), ("ʁ", "ɣ"), ("ã", "a"), ("ĩ", "i"), ("ũ", "u"),
        ("ẽ", "e"), ("õ", "o"), ("aŋ", "a"), ("eŋ", "e"), ("iŋ", "i"),
        ("oŋ", "o"), ("uŋ", "u")]

def norm(shape):
    for a, b in ORTH:
        shape = shape.replace(a, b)
    return shape

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]

def align(exp, got):
    """exp / got: normalized shape lists. Returns (hits, ref_len, got_len)."""
    n, m = len(exp), len(got)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    P = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0], P[i][0] = float(i), "up"
    for j in range(1, m + 1):
        D[0][j], P[0][j] = float(j), "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            denom = max(len(exp[i - 1]), len(got[j - 1]), 1)
            sub = D[i - 1][j - 1] + lev(exp[i - 1], got[j - 1]) / denom
            D[i][j], P[i][j] = min((sub, "diag"), (D[i - 1][j] + 1, "up"),
                                   (D[i][j - 1] + 1, "left"))
    hits, i, j = 0, n, m
    while i or j:
        op = P[i][j]
        if op == "diag":
            denom = max(len(exp[i - 1]), len(got[j - 1]), 1)
            if 1 - lev(exp[i - 1], got[j - 1]) / denom >= 0.5:
                hits += 1
            i, j = i - 1, j - 1
        elif op == "up":
            i -= 1
        else:
            j -= 1
    return hits, n, m
```

Pass `[norm(x) for x in ref_shapes]` and `[norm(x) for x in hyp_shapes]`; the tone
suffix must be stripped first (`re.sub(r"\d+$", "", label)`).

## What cannot be verified from the released artifacts

The corpus, the exact train/held-out clip lists and the training code are not
published. A third party can therefore re-implement the metric and check the
arithmetic, but cannot resample the same split. Numbers in this repository should be
read as reported results, not as independently reproduced ones.
