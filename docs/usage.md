# Usage

Two ways to run the released models:

1. **the `tujia-asr` package** (recommended) — syllables and words, one command;
2. **a release archive** — the standalone `infer.py`, syllables only.

Both use the same checkpoints. Commands are given for macOS / Linux and for
Windows where they differ.

---

## 1. What you need

| Item | Requirement | Notes |
| --- | --- | --- |
| Python | 3.10 – 3.12, **64-bit** | 32-bit Python has no PyTorch wheels |
| Disk | ~3.5 GB free | encoder 1.26 GB + dependencies ~2 GB |
| RAM | 4 GB free | the encoder is loaded in fp32 |
| Network | to GitHub, and to Hugging Face or ModelScope | the encoder is a separate download |
| ffmpeg | optional | only needed to convert mp4/m4a into wav |
| GPU | not required | CPU is enough, `--device cpu` |

Check the Python build first:

```bash
python3 -c "import sys, platform; print(sys.version); print(platform.architecture())"
```

---

## 2. Install the package

```bash
pip install tujia-asr
```

From a checkout instead:

```bash
git clone https://github.com/ibaibing/tujia-asr
cd tujia-asr
pip install -e .
```

Verify:

```bash
tujia-asr --help
python -c "import tujia_asr; print(tujia_asr.__version__)"
```

---

## 3. Get the weights

| File | Source |
| --- | --- |
| `tujia-ctc-v0.2.pt` | GitHub Releases → `tujia-ctc-0.2.tar.gz` → `model/` |
| `tujia-xlsr53-ft/` | Hugging Face or ModelScope `TujiaNLP/tujia-xlsr53-ft` |
| `tujia-ctc-v0.1.pt` | `tujia-ctc-0.1.tar.gz` → `model/` (toneless) |

The fine-tuned encoder directory must contain `config.json`:

```
tujia-xlsr53-ft/
├── config.json
├── preprocessor_config.json
├── model-00001-of-00005.safetensors
├── ...
└── model.safetensors.index.json
```

Point the package at both:

```bash
export TUJIA_CTC_CKPT=/path/to/tujia-ctc-v0.2.pt
export TUJIA_XLSR=/path/to/tujia-xlsr53-ft
```

Or pass `--ckpt` / `--encoder` per run. The v0.2 encoder **cannot** be replaced
by the original `facebook/wav2vec2-xlsr-53-espeak-cv-ft`; v0.1 can use the
original.

**Mainland China.** Use mirrors:

```bash
HF_ENDPOINT=https://hf-mirror.com tujia-asr speech.wav ...
```

The archive's `fetch_encoder.sh` also accepts `TUJIA_ENCODER_SOURCE=modelscope`.

---

## 4. Prepare the audio

`soundfile` reads wav, flac, ogg. Container formats such as mp4 or m4a must be
demuxed first:

```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 audio.wav
```

16 kHz mono is ideal. Other sample rates are resampled on the fly (requires
`torchaudio`). Keep each clip under ~60 seconds; longer files are split on
silence automatically (`--seg` / `--search`).

### Fewer dropped syllables (`--blank-bias`)

v0.3 is conservative: on the 353-clip held-out set it drops 41% of the reference
syllables and emits only 62% of the expected length, against 24% drops and 0.86
length for v0.2. When a human proofreads the output, a visible wrong syllable is
more useful than a gap, so `--blank-bias 1.0` lowers the CTC blank logit and
converts many drops into substitutions:

- `1.0` — deletions 41% → 31%, length 0.62 → 0.75, SER +0.6 point
- `1.5` — deletions 41% → 27%, length 0.62 → 0.80, SER +1.6 points

`TUJIA_BLANK_BIAS` sets the same value from the environment; the demo Space runs
at 1.0. The sweep reads reference labels through the v0.3i vocabulary, so its
own baseline SER (0.759) sits 2.6 points above the 73.30% in the README; the
deltas above are all from that single sweep.

---

## 5. Run

```bash
# syllables only (no built-in vocabulary)
tujia-asr audio.wav --device cpu --no-default-dict

# words: the open vocabulary from tujia-nlp is used by default
tujia-asr audio.wav --device cpu

# add your own rows on top of the built-in vocabulary
tujia-asr audio.wav --dict lexicon.jsonl --device cpu

# write JSON
tujia-asr audio.wav --dict lexicon.jsonl --json out.json
```

Windows (PowerShell):

```powershell
$env:TUJIA_CTC_CKPT = "C:\models\tujia-ctc-v0.2.pt"
$env:TUJIA_XLSR = "C:\models\tujia-xlsr53-ft"
tujia-asr audio.wav --dict lexicon.jsonl
```

Output, word mode:

```
# audio.wav  4.6s  ctc  13 spikes  tier-A 43%

=== words
[   0.00→   0.43] ni³⁵ 「代你，人称代词称对方」  A
[   0.43→   1.19] sa³⁵ 「名木排系汉语借词」        A

text: ni³⁵ sa³⁵ ...
```

Columns: start, end, word (toned IPA), gloss, tier. With `--no-default-dict` and
no `--dict`, the same layout lists syllables with tone confidence.

Tiers: `A` exact syllable match with agreeing tone, `B` fuzzy match or tone
disagreement, `C` supported only by the second channel, `D` dropped (only with
the ZIPA backend).

### Options

| Flag | Effect |
| --- | --- |
| `--dict FILE` | extra lexicon JSONL (repeatable), added to the built-in pack |
| `--no-default-dict` | do not load the `tujia-nlp` open core vocabulary |
| `--json [FILE]` | JSON output; no value means stdout |
| `--out FILE` | alias for `--json FILE` |
| `--format ipa\|ascii\|ye` | syllable display format |
| `--no-tone` | skip F0 evidence (the CTC tone head still runs) |
| `--frontend ctc\|zipa` | acoustic backend (default `ctc`) |
| `--ckpt FILE` | different head checkpoint |
| `--encoder DIR` | different XLSR encoder |
| `--device auto\|mps\|cpu\|cuda` | compute device |
| `--mode none\|qwen3\|moss\|whisper` | optional second channel |
| `--seg` / `--search` | long-audio window target / silence search radius |
| `--blank-bias FLOAT` | lower the CTC blank logit: `1.0` trades a few substitutions for far fewer dropped syllables (see below) |
| `--quiet` | suppress the human-readable report |

Environment variables: `TUJIA_CTC_CKPT`, `TUJIA_XLSR`, `TUJIA_DICTS`
(path-separator list), `TUJIA_NO_DEFAULT_DICT`, `TUJIA_FRONTEND`,
`TUJIA_TONE_BANK`, `TUJIA_AUDIO_ROOT`, `TUJIA_AUDIO_PY`, `TUJIA_BLANK_BIAS`,
`ASR_ENDPOINT`,
`ASR_MODEL`, `ASR_CHAT_MODEL`, `ASR_API_KEY`, `ASR_MODE`. A JSON file at
`TUJIA_ASR_CONFIG` is read first; env wins.

---

## 6. Lexicon format

The built-in vocabulary (`tujia-open-core`, 1149 northern entries, CC-BY-4.0)
comes from `tujia-nlp` and is loaded automatically. Rows you pass with `--dict`
are read first, so your spellings win on collision; the built-in rows are
appended after them. Use `--no-default-dict` to run on your lexicon alone.

The dictionary is a **post-processor, not a corrector**: it adds word
boundaries, glosses and a tone prior, but does not change what the acoustic
model heard. With and without it the syllables and their tone confidence are the
same. `--dict` is repeatable; the first file to define a headword wins, and the
built-in pack comes last. Words that differ only in tone are kept as separate
entries and resolved by tone evidence (`tone_dict` vs `tone_model` in the
output). Format, one JSON object
per line, UTF-8:

```json
{"w": "ni³⁵", "g": "你", "pos": "代"}
{"w": "a⁵⁵ji²¹", "g": "…", "pos": "名", "pg": "123"}
```

- `w` (required): toned IPA, superscript tone digits.
- `g`: gloss, `pos`: part of speech, `pg`: page or other reference. Optional.

Multiple files are merged at load time. Entries whose head has no tone cannot
participate in word decoding but still contribute syllable shapes.

---

## 7. Python API

```python
from tujia_asr import Settings, recognize

settings = Settings.from_env()
res = recognize(
    "audio.wav",
    dict_path="lexicon.jsonl",   # rows added to the built-in pack; None = built-in only
    settings=settings,
    device="cpu",
)
print(res["text"])
print(res["words"][0]["ortho"], res["words"][0]["tier"])
```

`res["schema"]` is `tujia.asr/1`. See the README for the full key list.

---

## 8. Optional backends

- **ZIPA**: `--frontend zipa --zipa-root ~/workspace/tools/zipa --zipa-python …`
  runs the external lattice decoder; needs a lexicon. Off by default.
- **Second channel**: `--mode qwen3` (or `whisper` / `moss`) plus
  `ASR_ENDPOINT=http://host:port/v1`. Used only as weak evidence.
- **Tone bank**: `TUJIA_TONE_BANK=/path/to/bank` with a `manifest.csv`;
  supplies speaker-matched tone templates.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Could not find a version that satisfies the requirement torch` | interpreter mismatch or 32-bit Python | see step 1 |
| `未找到 CTC 权重` | no checkpoint configured | set `TUJIA_CTC_CKPT` / `--ckpt` |
| `编码器目录不存在` | `TUJIA_XLSR` unset or one level too high | point at the folder containing `config.json` |
| `metadata.get("format")` / safetensors error | an old copy of the encoder | delete the folder and re-download |
| `soundfile` error on `.mp4` | unsupported container | convert to wav first |
| `KeyError` on load | `vocab.txt` does not match the checkpoint | do not mix v0.1 and v0.2 files |
| No words, only syllables | no `--dict`, or lexicon not matching | check the `unknown` spans and the lexicon |
| Very slow on CPU | expected | ~1× real time per clip; use `--seg` |

---

## 10. Manual usage of a release archive

The archives remain standalone. This is the syllables-only `infer.py` path.

```bash
# verify
shasum -a 256 tujia-ctc-0.2.tar.gz

# unpack
tar -xzf tujia-ctc-0.2.tar.gz && cd tujia-ctc-0.2

# environment
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# encoder (v0.2 only)
./fetch_encoder.sh ./models

# decode
TUJIA_XLSR=./models/tujia-xlsr53-ft python infer.py audio.wav --device cpu
```

Windows: install `torch torchaudio` with the CPU index, then run

```powershell
$env:TUJIA_XLSR = "models\tujia-xlsr53-ft"
python infer.py audio.wav --device cpu
```

The archive's `infer.py` now imports `tujia_nlp` for tone / Ye formatting.
Install it from source (`pip install -e /path/to/TujiaNLP`) if it is not on
PyPI. Verify the archive with `shasum -a 256 -c SHA256SUMS`; all nine lines must
print `OK`.

Archive `infer.py` flags: `--json`, `--start`, `--duration`, `--device`,
`--ckpt`, `--format ipa|ascii|ye`.

---

## 11. What the labels mean

v0.2 emits 708 toned syllables plus blank. The tone (`21 / 35 / 53 / 55`) is part
of the label, so `la35` is one unit, not `la` plus a tone. There is no word
segmentation, translation or punctuation inside the model itself; the package
adds word segmentation when a lexicon is supplied.

v0.1 emits toneless syllable shapes only.

---

## 12. License boundaries

- Scripts and documentation: MIT.
- Model weights: research use only. No rights to the training recordings or
  transcripts are granted, and none of that material is distributed.
- The upstream encoder `facebook/wav2vec2-xlsr-53-espeak-cv-ft` is Apache-2.0
  and must retain its attribution.

See `NOTICE` for the full boundary.
