# Pipeline

What happens between an audio file and a toned syllable string. Every section
names the module that implements it, so the description can be checked against
the code rather than trusted.

```
audio file
   │  read_mono            resample to 16 kHz mono, cut span
   │  plan_windows         split long audio at silence
   ▼
waveform windows
   │  CtcFront.features    XLSR-53 frame features (50 Hz)
   │  HeadTone             CTC head: toned syllable + tone logits
   ▼
frame-level spikes
   │  postprocess.check/fix   inventory constraint (spelling level only)
   │  tone.py               pick one tone per spike from five evidence sources
   │  syllabify             group spikes into syllables
   │  lexicon + decode      Viterbi word segmentation (optional)
   │  fuse                  A/B/C/D reliability tier
   ▼
text + tokens + warnings
```

## 1. Preprocessing

`tujia_asr.audio.read_mono` is the only place audio is read. It does four
things and nothing else:

| step | behaviour |
| --- | --- |
| decode | `soundfile` (wav / flac / ogg); mp4 and m4a must be demuxed first |
| mono | multi-channel input is averaged, not channel-0 picked |
| resample | anything not 16 kHz goes through `torchaudio.functional.resample` |
| span | `offset` / `duration` select a slice before any other work |

Deliberately **not** done by default: loudness normalisation, noise
suppression, VAD. The released encoder was fine-tuned on field recordings as
they were, and a denoiser in front of it shifts the spectral tilt the model
learned. A denoiser exists for the optional ZIPA front-end
(`zipa.denoise`), and it is off unless that backend is selected.

Long audio is handled by `zipa.plan_windows`, which is front-end independent:

| parameter | default | meaning |
| --- | --- | --- |
| `target` | 10.0 s | nominal window length |
| `search` | ±2.0 s | how far a cut may move to land on silence |
| `min_sil` | 0.12 s | shortest silence run accepted as a cut |
| `max_dur` | 14.0 s | hard ceiling per window |

Silence is detected as consecutive 30 ms frames more than 45 dB below the 95th
percentile of the file's RMS. A cut is placed at the centre of the longest
silence run inside the search radius; if there is none, the nominal position is
used and the window may reach `max_dur`.

## 2. Acoustic front-end

`tujia_asr.ctc.CtcFront` runs the released `tujia-ctc-v0.2` checkpoint on
`tujia-xlsr53-ft` features. Output is one spike per non-blank, non-repeating
frame run, with a syllable label, a frame span, and two confidences (syllable
and tone). A spike's span is the midpoint to its neighbours, clamped to ±0.30 s
around its own peak: without the clamp a lone syllable inside a silent stretch
claims the whole silence, which then blocks syllable grouping and inflates
coverage. Frame spacing is 20 ms (`XLSR_FRAME_SEC`); the coverage statistics
are reported on a 10 ms grid.

## 3. Post-processing

Post-processing is split into stages that either *report* or *rewrite*, never
both. That separation is the point: a model upgrade should be judged with the
same reporting code, not compared against output that has already been
corrected.

### 3.1 Tone

`tone.py` collects five independent opinions per syllable and records all of
them in `tone_evidence`:

| source | key | note |
| --- | --- | --- |
| CTC tone head | `model` | trained jointly with the syllable head |
| tone bank templates | `bank` | off unless `TUJIA_TONE_BANK` is set |
| F0 prototype | `acoustic` | median F0, slope and range over the voiced part |
| dictionary | `dict` | candidates that agree with the recognised shape |
| hanzi | `hanzi` | only when a gloss is available |

Conflicts are kept in `tone_evidence.conflict` rather than resolved silently.
With only the CTC head available — the default — `method` reads `ctc-tone`.

### 3.2 Inventory constraint

`postprocess.check(tokens)` counts what is off the frozen inventory and changes
nothing. `postprocess.fix(tokens)` rewrites two tiers and reports them apart:

- **spelling** — the inventory's own equivalence table (`ian` → `iã`,
  `ye` → `ɣɤ`, `khe` → `kʰɤ`). Same phones, different writing. Deterministic.
- **fuzzy** — nearest attested syllable by edit distance. Only with
  `fuzzy=True`, never on by default, because it can turn `kiɛ` into `iɛ`.

Both keep the original shape in `shape_raw`, so a caller can log the change.

### 3.3 Syllables, words, tiers

`syllabify.segment` assigns spikes to syllables using one vowel nucleus each.
`lexicon` + `decode.segment_words` then run a Viterbi over recognised syllables:
dictionary words match syllable by syllable with a character-distance ceiling of
0.40 (`SHAPE_MAX_NORM`), words longer than six syllables are not attempted
(`MAX_WORD_SYL`), a word may not bridge a silence longer than 0.45 s
(`MAX_SYL_GAP`), and a matching tone adds to the score. With no dictionary the
`words` list stays empty and `syllables` is always populated.

`text` is assembled in time order: spans that became words use the dictionary
spelling (`words[].ortho`), everything else falls back to the recognised toned
syllables. A partially matched sentence therefore keeps its unmatched syllables
instead of dropping them. `words[].ipa` holds what was actually heard, with
`tone_model` / `tone_dict` side by side, so a tone mismatch stays visible rather
than being overwritten by the dictionary form.

`fuse.assign_tier` labels each syllable A/B/C/D from average acoustic score and
second-channel support:

| tier | condition |
| --- | --- |
| A | second channel agrees, avg ≥ 1.5, support ≥ 0.5 |
| B | second channel agrees but support < 0.5, or no second channel and avg ≥ 1.5 |
| C | second channel agrees, support ≥ 0.6 |
| D | everything else |

With the default `ctc` backend and no second channel configured, output is
A-less by construction. That is why the Web demo highlights by confidence
rather than by tier.

### 3.4 Accounting

Nothing is dropped silently. Unmatched spans, syllables outside the inventory
and windows whose coverage is low all land in `unknown`, `inventory.unknown`
and `warnings`. `coverage` reports frames explained and frames in tier A
separately, so a model that explains more frames *badly* cannot look like an
improvement. Silence (`reason == "gap"`) is not counted as explained; counting
it made the ratio 1.0 for every file regardless of how little was recognised.

A lexicon that yields no syllables at all — romanised packs, where `w` is
`BIFZIVSAR` instead of `sa⁵³zɿ²¹` — is reported in `warnings` and ignored, so
the run degrades to syllable output instead of returning nothing.

## 4. Worked example

`space/examples/sample_field.wav` — 6.6 s of field interview audio used with the
speaker's permission, 16 kHz mono, short enough not to be segmented:

```
ka35 i35 ta53 tɕi35 a35 u35 xa21 tʰa55 tʰi55 ni21 ɤ21 ka55 ni55 ɤ55 tʰi55
tu21 o21 mi21 ŋti55 tsʰu21 o21 tʰa35 i21 nɤ21 ɕi55 li35 pu35 la55
```

28 syllables. The stages above left these marks on it:

| field | value |
| --- | --- |
| `audio.dur` | 6.6 |
| `audio.segmented` | false (1 window) |
| `syllables` | 28 |
| `words` | 0 — no lexicon supplied |
| `inventory.n_normalised` | 3 (spelling tier only) |
| `inventory.n_unknown` | 1 — `ŋti`, which is not an attested syllable |
| `coverage.ratio` | 0.98 (647 / 660 frames explained) |
| `coverage.tier_a_ratio` | 0.0 — no second channel, so no A |

A single token, showing the shape the tone evidence takes:

```json
{"t0": 0.0, "t1": 0.147, "shape": "ka", "ipa": "ka", "score": 0.754,
 "model_tone": "35", "model_tone_conf": 0.566, "tone": 35,
 "tone_evidence": {"model": {"value": 35, "conf": 0.566},
                   "acoustic": {"value": null, "conf": 0.09,
                                "f0_median_hz": 178.3, "slope_st_100ms": 0.68},
                   "bank": {"n_templates": 0}, "dict": {"candidates": []}},
 "tier": "D"}
```

Low acoustic confidence and a `D` tier are consistent here: this is a 147 ms
syllable at the start of the file, and the model is not sure of the tone.

With a lexicon loaded, each syllable also carries `in_lexicon`. Rows outside the
inventory are kept verbatim with `in_lexicon: false` rather than dropped, and a
word never absorbs them — so a word the vocabulary does not know still appears
in the transcript as its recognised syllables.

## 5. Reproduce

```bash
python -m tujia_asr space/examples/sample_field.wav \
  --ckpt   path/to/tujia-ctc-v0.2.pt \
  --encoder path/to/tujia-xlsr53-ft \
  --device cpu --out result.json
```

`--out` writes the full result shown above; without it the CLI prints one line
per syllable plus the text. `--format ipa|ascii|ye` changes only the rendering of
labels, never the recognition.

## 6. What this pipeline does not do

- No language model rescoring. Word choice is dictionary match plus tone bonus.
- No forced alignment against a known transcript; this is recognition only.
- No streaming. Each call decodes a file; long audio is split on silence.
- No denoising by default, so a noisy recording is recognised as recorded.
