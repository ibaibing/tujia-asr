# tujia-asr

Speech recognition for **Tujia**: audio in, toned syllables and — with the
bundled open vocabulary, or one you add — words out. The v0.4 inventory covers
both northern and southern survey varieties in one syllable table.

The package ships the inference pipeline only. It bundles the released
syllable-level CTC models (v0.1 toneless, v0.2 toned, v0.3 toned, v0.4
toneless-plus-tone), a lexicon-constrained word decoder, and optional
tone-evidence fusion. Training code, training logs and the corpus are not
published; the private `tujia-asr-lib` repository is used for training and
architecture comparison.

```
audio.wav  →  CTC acoustic model  →  syllable lattice  →  word decoder  →  text
                                              ↘ optional: F0 tone fusion
```

New to the project? [docs/note.md](docs/note.md) explains what the pipeline does,
what the numbers mean, and where it fails — written for both technical and
non-technical readers.

## Install

Python 3.10–3.12, 64-bit. Audio is read with `soundfile`; other sample rates are
resampled on the fly.

```bash
pip install tujia-asr
```

From a checkout instead:

```bash
git clone https://github.com/ibaibing/tujia-asr
cd tujia-asr
pip install -e .
```

Model weights are distributed separately (~1.26 GB encoder + 19 MB head):

| What | Where |
| --- | --- |
| `tujia-ctc-v0.4.pt` (current head, 932 syllables + tone head) | [GitHub Releases](https://github.com/ibaibing/tujia-asr/releases) → `tujia-ctc-0.4.tar.gz` → `model/` |
| `tujia-xlsr53-v03f` (encoder for v0.4 and v0.3) | Hugging Face / ModelScope `TujiaNLP/tujia-xlsr53-v03f` |
| `tujia-ctc-v0.3.pt` (previous head, 1199 toned classes) | `tujia-ctc-0.3.tar.gz` → `model/` |
| `tujia-ctc-v0.2.pt` (older head, 709 classes) | `tujia-ctc-0.2.tar.gz` → `model/` |
| `tujia-xlsr53-ft` (encoder for v0.2) | Hugging Face / ModelScope `TujiaNLP/tujia-xlsr53-ft` |
| `tujia-ctc-v0.1` (toneless head) | `tujia-ctc-0.1.tar.gz` |

Point the package at them with environment variables or flags:

```bash
export TUJIA_CTC_CKPT=/path/to/tujia-ctc-v0.4.pt
export TUJIA_XLSR=/path/to/tujia-xlsr53-v03f    # directory containing config.json
```

The fine-tuned encoder **cannot** be replaced by the original
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` weights: the features differ and
recognition degrades substantially. v0.4 and v0.3 share the same encoder; v0.2
uses the older `tujia-xlsr53-ft`.

## Quick start

```bash
# syllables only (built-in vocabulary disabled)
tujia-asr speech.wav --no-default-dict

# words: the built-in open vocabulary is used automatically
tujia-asr speech.wav

# add your own lexicon on top of it (JSONL)
tujia-asr speech.wav --dict lexicon.jsonl

# machine-readable result
tujia-asr speech.wav --dict lexicon.jsonl --json out.json
```

Human-readable output:

```
# 01_nif_caf.wav  4.6s  ctc  13 spikes  tier-A 43%

=== words
[   0.00→   0.43] ni³⁵ 「代你，人称代词称对方」  A
[   0.43→   1.19] sa³⁵ 「名木排系汉语借词」        A
...

text: ni³⁵ sa³⁵ sɿ³⁵ xa²¹ tʰũ²¹ sɿ³⁵
```

Syllable mode, with `--no-default-dict` and no `--dict`:

```
=== syllables
[   0.00→   0.31] tɕiɛ³⁵    tone=35  conf=0.81
...
```

Useful flags: `--device cpu|mps|cuda`, `--ckpt`, `--encoder`, `--format ipa|ascii|ye`,
`--no-tone`, `--seg` / `--search` (long-audio windowing), `--frontend ctc|zipa`.
Run `tujia-asr --help` for the full list.

## Python API

```python
from tujia_asr import recognize

res = recognize("speech.wav", dict_path="lexicon.jsonl")
print(res["text"])          # transcript
print(res["words"])         # word spans + glosses
print(res["syllables"])     # always present
```

`recognize` accepts `settings=Settings.from_env()`, `dict_path=` (str or list),
`mode=` (second channel), `seg`/`search` for long audio, `tone=False` to skip F0
evidence, and `device=`.

The result is a JSON-serialisable dict with `schema: "tujia.asr/1"`:

| Key | Meaning |
| --- | --- |
| `text` | transcript in time order: dictionary spelling for matched words, toned syllables elsewhere |
| `syllables` | every recognised syllable with time span, tone, confidence |
| `words` | word spans with `ortho` (dictionary spelling), `ipa` (what was heard), `tone_model` / `tone_dict`, `han`, `pos`, `pg`, `tier` |
| `tokens` | syllable lattice with tone evidence per token |
| `inventory` | inventory version, syllables normalised, syllables still unknown |
| `unknown` | unexplained spans (no syllable match, silence, dropped words) |
| `coverage` | fraction of speech explained (silence excluded), and tier-A fraction |
| `frontends` | acoustic backend, spike count, windows, second-channel status |
| `model` | model name, checkpoint and encoder actually used |
| `warnings` | non-fatal notes (missing tone bank, disabled channel, …) |

## Syllable inventory check

`syllables` are emitted in the spelling the model was trained on, which is not
always the spelling the frozen syllable inventory uses (`e` where the table has
`ɤ`, `ɛ` for `ɤ`). `postprocess` checks every token against that inventory and
can rewrite the ones that are the same phones written differently:

```python
from tujia_asr import postprocess

report = postprocess.check(res["tokens"])      # count only, output untouched
tokens, report = postprocess.fix(res["tokens"])  # spelling-level corrections
print(report.n_unknown, report.n_fixed_spelling, report.fixes)
```

`fix` is conservative: it only applies the inventory's own equivalence table
and never guesses. `fuzzy=True` additionally snaps anything still unknown to
the nearest attested syllable and reports those separately
(`report.n_fixed_fuzzy`), because edit distance can land on the wrong word —
on the v0.2 held-out set it turns `kiɛ` into `iɛ` and `ʰu` into `au`.

Both functions leave the input list alone and keep the original spelling in
`shape_raw`, so a caller can log what changed.

## Lexicon

A lexicon ships with the dependency: `tujia-nlp` bundles the open core vocabulary
(`tujia-open-core`, 1149 northern entries, CC-BY-4.0, derived from Zhou (2020) via
[`luggroo/bizic-dictionary`](https://github.com/luggroo/bizic-dictionary)). It is
loaded by default and only the northern rows are used, because the recogniser is
trained on northern Tujia. Pass `--dict` to **add** your own rows — the built-in
ones stay — or set `TUJIA_NO_DEFAULT_DICT=1` to drop them.

Your own lexicon is one JSON object per line, UTF-8, toned IPA in `w`
(superscript tone digits, e.g. `ni³⁵`):

```json
{"w": "ni³⁵", "g": "你", "pos": "代"}
{"w": "a⁵⁵ji²¹", "g": "…", "pos": "名", "pg": "123"}
```

Only `w` is required. `g` (gloss), `pos` and `pg` are passed through to the
output. Nothing the vocabulary does not cover is thrown away: unmatched
syllables stay in `syllables` with `in_lexicon: false`, and words never absorb
them, so out-of-vocabulary speech still shows up in the transcript. Turn the
built-in pack off and supply nothing and you still get syllables, grouping
phone spikes by one vowel nucleus per syllable.

Word decoding is a Viterbi segmentation over recognised syllables: dictionary
words are matched syllable by syllable (fuzzy character distance ≤ 0.40), with a
bonus when the model's tone agrees with the dictionary tone. Words are never
emitted for spans that do not match the dictionary; those become `unknown`.

What the lexicon does **not** do is improve recognition. The acoustic model
output is unchanged (same syllables, same tone confidence with or without it);
the lexicon only adds word boundaries, glosses and a tone prior. It is a
post-processor, not a corrector: a misheard syllable is snapped to the nearest
dictionary word rather than fixed.

**Several lexicons.** `--dict` is repeatable and rows are tried in order, so a
headword appears once and the first file that defines it wins:

```bash
tujia-asr speech.wav --dict project.jsonl --dict local.jsonl
```

The built-in pack is appended last, which means your rows override it on
collision; `--no-default-dict` drops it entirely. Every word in the output
carries `source` (`tujia-open-core`, or the file stem) so you can tell where it
came from.

**Same syllables, different tone.** These are distinct words in Tujia, so both
rows are kept. The Viterbi decoder picks between them with the tone evidence:
the model's tone gives +0.20 when it agrees and −0.10 when it does not. Each
word reports `tone_dict` and `tone_model` side by side, so a disagreement stays
visible instead of being silently resolved.

## Optional backends

| Feature | How to enable | Default |
| --- | --- | --- |
| ZIPA-NS lattice front-end | `--frontend zipa --zipa-root … --zipa-python …` | off |
| Second channel (qwen3 / whisper / moss) | `--mode …` + OpenAI-compatible `ASR_ENDPOINT` | off |
| Tone bank templates | `TUJIA_TONE_BANK=/path/to/bank` | off |

The default `ctc` backend is self-contained and needs no external tools.

A JSON config file is also read from `TUJIA_ASR_CONFIG`; environment variables
take precedence. See `src/tujia_asr/config.py` for the keys.

## Models

| | v0.1 | v0.2 |
| --- | --- | --- |
| Labels | 264 toneless syllables | 708 toned syllables (tones `21 / 35 / 53 / 55`) |
| Encoder | `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, frozen | same encoder, fine-tuned (top 4 of 12 layers) |
| Head | LayerNorm + 2× BiLSTM(256) + Linear(512→264) | same + Linear(512→709), tone auxiliary CTC head during training |
| Trainable parameters | 4,339,976 (17 MB) | 4,570,826 (18 MB) |
| Best epoch | 35 | 18 |
| Held-out recall (108 clips) | 80.26% | 83.93% |
| Tone accuracy (held-out) | not predicted | 65.59% |
| Archive | `tujia-ctc-0.1.tar.gz` | `tujia-ctc-0.2.tar.gz` |

Both expect **16 kHz mono** audio and emit one syllable per frame run. v0.2
attaches the tone to the label, e.g. `la35`.

### v0.4 (shipped with `tujia-asr` 1.1.0 and later)

| | |
| --- | --- |
| Labels | 932 toneless syllables + a separate 4-class tone head (`21 / 35 / 53 / 55`) |
| Encoder | the same fine-tuned `v0.3f` encoder as v0.3 |
| Training | 3,635 clips from an earlier corpus, repeated twice, plus 9,864 field-survey clips (17,134 rows / 16.2 h) |
| Best epoch | 34 |
| Test SER | **0.2132** syllable / **0.2054** tone / **0.3543** joint toned |
| Old-data validation SER | **0.5571** |

The v0.4 test set has 840 field-survey clips (0.25 h); v0.3 scores 0.7652
syllable / 0.5616 tone on the same clips. The split is by lexical item, so every
test speaker also appears in training — this is **not** a speaker-held-out
number. See [`releases/tujia-ctc-v0.4.md`](releases/tujia-ctc-v0.4.md).

### v0.3 (shipped with `tujia-asr` 1.0.0 and later)

| | |
| --- | --- |
| Labels | 1,199 toned syllables (tones `21 / 35 / 53 / 55`) |
| Encoder | fine-tuned from the `v0.3f` encoder |
| Training | 5,243 clips, tone-auxiliary loss weight `lambda_tone = 0.4` |
| Best epoch | 8 — validation SER 0.5126 over a 141-clip split |
| Held-out SER | **0.7330** on 353 leak-free clips (see [Evaluation](#evaluation)) |

The 0.5126 is the validation split used for early stopping, so it is not an
independent number. The held-out figure comes from 353 clips that appear in
neither v0.3's training nor its validation split; the older 856-clip test split
cannot be used for v0.3 because 851 of its clips are in v0.3's training set.
Under the same 353-clip protocol v0.2 scores SER 0.8627, so v0.3 is 13 points
lower — but it deletes more (0.3901 vs 0.1941), i.e. it is more likely to
under-transcribe than to mishear. When proofreading matters more than the SER
number, `--blank-bias 1.0` turns a third of those deletions into visible
(if wrong) syllables at a 0.6-point SER cost; see
[`docs/usage.md`](docs/usage.md#fewer-dropped-syllables---blank-bias). The demo
Space runs at 1.0.

## Where the weights live

| Channel | Contents |
| --- | --- |
| GitHub Releases (this repo) | `tujia-ctc-0.4.tar.gz` (17 MB), `tujia-ctc-0.3.tar.gz` (19 MB), `tujia-ctc-0.2.tar.gz`, `tujia-ctc-0.1.tar.gz`, `SHA256SUMS` |
| Hugging Face | `TujiaNLP/tujia-xlsr53-v03f`, `TujiaNLP/tujia-xlsr53-ft`, `TujiaNLP/tujia-ctc-v0.4`, `TujiaNLP/tujia-ctc-v0.3`, `TujiaNLP/tujia-ctc-v0.2`, `TujiaNLP/tujia-ctc-v0.1` |
| ModelScope | same repositories under `TujiaNLP` (mainland mirror) |

The 1.26 GB encoder is required for v0.4 and v0.3 (and the older 1.2 GB one for
v0.2); v0.1 can use the original Hugging Face XLSR-53 weights. Set
`TUJIA_ENCODER_SOURCE=modelscope` when fetching, or
`HF_ENDPOINT=https://hf-mirror.com` for a mirror.

Each archive contains the checkpoint, `vocab.txt`, `model_card.json`, a
standalone `infer.py`, `fetch_encoder.sh`, `LICENSE`, `NOTICE`, `SHA256SUMS`
and `MANIFEST.json`. The archive scripts remain usable on their own; this
package is the recommended interface.

## Evaluation

Two protocols exist and they are **not interchangeable**. Always name the
protocol when quoting a number.

- **SER** — strict syllable error rate: token sequences aligned with
  Needleman–Wunsch, edit distance over reference length. Used for v0.2–v0.4.
  **Lower is better**: 73.30% means roughly a quarter of the syllables are
  right, and `1 − SER` is a fair approximation of syllable accuracy.
- **Recall** — lenient syllable recall: normalized shapes, alignment hit at
  edit similarity ≥ 0.5. Used for v0.1 and cross-version comparisons. Again
  lower SER / higher recall is better; do not read a recall number as an error
  rate or vice versa.

Definitions and splits are in [docs/protocol.md](docs/protocol.md); the manifest
schema is in [docs/data-format.md](docs/data-format.md).

| Checkpoint | Recall, all 685 | Recall, held-out 108 | Precision, held-out 108 |
| --- | --- | --- | --- |
| v0.1 | 89.82% | 80.26% | 84.33% |
| v0.2 | 89.90% | 83.93% | 84.37% |

### SER, v0.4

v0.4 is scored on an 840-clip field-survey test set (0.25 h, split by lexical
item). v0.3 is shown on the same clips; it was not trained on this set.

| Checkpoint | Syllable SER | Tone SER | Joint toned SER |
| --- | --- | --- | --- |
| v0.3 | 76.52% | 56.16% | — |
| v0.4 | **21.32%** | **20.54%** | **35.43%** |

By area: north 20.86% syllable / 20.19% tone; south 22.80% / 21.65%. Old-data
validation SER is 55.71% (v0.3: 64.90%). The test speakers also appear in
training, so these are seen-speaker numbers, not a speaker-held-out score.

### SER, v0.2 vs v0.3

The 108-clip split above belongs to v0.1/v0.2; 74 of those clips are inside
v0.3's training set, and 851 of the 856 clips in the larger test split are too.
The comparable set is the 353 clips that neither model's training or validation
split contains (0.40 h, 8 speakers, 300 of them one male teacher):

| Checkpoint | SER | SER, no tone | Tone SER | Sub | Del | Ins |
| --- | --- | --- | --- | --- | --- | --- |
| v0.2 | 86.27% | 70.34% | 62.28% | 56.56% | 19.41% | 10.30% |
| v0.3 | **73.30%** | 64.57% | 58.32% | 30.57% | 39.01% | 3.72% |

v0.2 emits 708 labels, so its predictions are folded to the 0.6.3 inventory
before scoring; reference labels are read from the 0.6.3 vocabulary for both
runs. v0.3 trades substitutions for deletions — it is right more often but
emits fewer syllables, so faint or fast speech can come out short.

Word-level accuracy is **not** covered by these numbers: the released models
emit syllables, and word output depends on the lexicon you supply and on the
decoder.

## Data summary

Only aggregate properties of the training data are disclosed. The recordings,
their transcripts, the speakers and the collection are not published.

| Property | Value |
| --- | --- |
| Audio | 17,134 training rows, 16.2 h, 16 kHz mono (3,635 earlier-corpus clips used twice + 9,864 field-survey clips) |
| Annotation unit | one toned IPA syllable per segment |
| Variety | northern and southern survey areas, unified syllable inventory |
| Speakers | multiple, identities withheld |
| Label inventory | 932 toneless syllables + 4 tones (v0.4); 1,198 toned (v0.3); 708 / 264 (v0.2 / v0.1) |
| Evaluation | 840 field test clips (0.25 h), 414 old-data + 439 field validation clips |
| Recording conditions | read speech and field interviews, mixed quality and levels |

## Known limitations

1. **Uneven coverage.** The field data span several survey areas but not evenly;
   some speakers and recording conditions are underrepresented. Southern
   varieties are measured but not separately tuned.
2. **Length behaviour differs by version.** v0.2 over-segmented, v0.3 deleted
   more, and v0.4 sits between them (syllable SER 21.3%, tone SER 20.5%).
3. **Tone is the residual error.** About one tone in five is wrong on the field
   test set, mostly `55 ↔ 21` and `55 ↔ 53`.
4. **Narrowband audio degrades sharply.** On a field recording with 85% of its
   energy below 390 Hz, v0.1 recall drops to 48.85%.
5. **Small splits.** Dev/test is 43/65 clips; treat differences under ~1 point
   as noise.
6. **Word output is lexicon-bound.** Unknown words cannot be recovered; spans
   that do not match the lexicon are reported as `unknown`. The syllable-only
   mode is the honest floor.
7. **No streaming.** Each call decodes a file; long audio is split on silence.

## Alternatives considered

A fully fine-tuned `openai/whisper-small` was also trained on the same 577 clips
and is **not** released. It memorises the training set within a few epochs, it
emits text rather than frame-level labels (so syllable boundaries would need a
separate alignment pass), and it is roughly 50× the parameter count of the
released head with much slower decoding. Its held-out score sits within noise of
v0.2, which does not justify shipping a second, harder-to-run pipeline.

## Data and license

- Code in this repository and in every archive: MIT (see [LICENSE](LICENSE)).
- Model weights and the fine-tuned encoder: **research use only**. No rights to
  the training recordings or transcripts are granted, and none of that material
  is distributed.
- Upstream encoder `facebook/wav2vec2-xlsr-53-espeak-cv-ft` is Apache-2.0 and
  must retain its attribution.
- The parent project [TujiaNLP](https://github.com/ibaibing/TujiaNLP) maintains
  the field collection, annotation standards and orthography conversion.
- See [NOTICE](NOTICE) for the full boundary.

`tujia-asr` and the private `tujia-asr-lib` share the import name `tujia_asr`;
install only one of them in an environment.

## Citation

See [CITATION.cff](CITATION.cff). If you use a specific version, cite that
version number.
