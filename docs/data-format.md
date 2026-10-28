# Data format

The models were trained and evaluated from a JSONL manifest, one object per clip. The
manifest is not distributed, but this is the schema any re-implementation needs.

## Manifest

```json
{"wav": "/path/to/t21-01.wav", "group": "wav21", "dur": 16.41, "ids": [143, 83, 455, 11, 220, 97]}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `wav` | string | path to a 16 kHz mono audio file; absolute or relative to your own root |
| `group` | string | recording group / page identifier, used to define splits |
| `dur` | float | duration in seconds |
| `ids` | int list | reference labels as indices into `vocab.txt` |

Concatenate frames with `<sep>` (index 11 in both releases) between segments; index 0
is the CTC blank and never appears in `ids`.

The reference label sequence is recovered with:

```python
vocab = [line.rstrip("\n") for line in open("model/vocab.txt")]
refs = [vocab[i] for i in ids]
```

## Vocabularies

- **v0.1** — 264 entries: blank, `<sep>`, and 262 toneless syllable shapes.
- **v0.2** — 709 entries: blank, `<sep>`, and 707 toned syllables of the form
  `shape + tone`, tone ∈ `{21, 35, 53, 55}` (e.g. `la35`, `tsʰi53`).

The label set and its ordering are packaged with each release; do not substitute a
vocabulary from the other version.

## Audio

- 16 kHz, mono; anything else is resampled by `infer.py`.
- One utterance per clip. Keep clips under ~60 s; `infer.py --start/--duration`
  decodes a span of a longer file.
- No loudness normalization is applied at inference. The training corpus was recorded
  from a single source, so level differences between your recordings and that source
  are the most common cause of degraded output.

## Splits

| Split | Definition | Clips |
| --- | --- | --- |
| held-out | groups `wav21`, `wav32`, `wav34`, `wav51`, `wav53` | 108 |
| train | all other groups | 577 |
| dev (report only) | 43 clips inside `wav21` | 43 |
| test (report only) | remaining held-out clips | 65 |

Group names are opaque internal identifiers kept only for split bookkeeping; they
carry no meaning for users of the released weights.
