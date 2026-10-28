# Manual usage (GitHub)

Step-by-step instructions for running a release archive on your own machine.
Everything here is done by hand — no agent, no automation. Commands are given for
macOS / Linux and for Windows where they differ.

This assumes the GitHub Release channel. Hugging Face and ModelScope hold the same
files; only the download URLs differ.

---

## 1. What you need

| Item | Requirement | Notes |
| --- | --- | --- |
| Python | 3.10 – 3.12, **64-bit** | 32-bit Python has no PyTorch wheels |
| Disk | ~3.5 GB free | archive 17 MB + encoder 1.26 GB + dependencies ~2 GB |
| RAM | 4 GB free | the encoder is loaded in fp32 |
| Network | to GitHub, and to Hugging Face or ModelScope | the encoder is a separate 1.26 GB download |
| ffmpeg | optional | only needed to convert mp4/m4a into wav |
| GPU | not required | CPU is enough, `--device cpu` |

Check the Python build first:

```bash
python3 -c "import sys, platform; print(sys.version); print(platform.architecture())"
```

You want a `3.x` version in the 3.10–3.12 range and `64bit`.

---

## 2. Download the archive

Open the Releases page:

```
https://github.com/ibaibing/tujia-asr/releases
```

Pick a version. Each release has one asset:

| Release | Asset | Bytes | SHA256 |
| --- | --- | --- | --- |
| `tujia-ctc-v0.2` | `tujia-ctc-0.2.tar.gz` | 16,976,694 | `974cf2a629b12203398a8f9f842a6974c5332ec27d79d2b0d8a1c9b4f703478b` |
| `tujia-ctc-v0.1` | `tujia-ctc-0.1.tar.gz` | 16,080,784 | `b74415578fbdc696b64386b70d4bc34b1582cf26a12ecfa6ca777523e46a7ab0` |

Do **not** download "Source code (zip)" or "Source code (tar.gz)". Those are GitHub's
automatic snapshots of the repository text: they contain the documentation only, with
no model weights and no `infer.py`.

Direct link pattern:

```
https://github.com/ibaibing/tujia-asr/releases/download/tujia-ctc-v0.2/tujia-ctc-0.2.tar.gz
```

## 3. Verify the download

```bash
# macOS / Linux
shasum -a 256 tujia-ctc-0.2.tar.gz
```

```bat
:: Windows (PowerShell)
Get-FileHash tujia-ctc-0.2.tar.gz -Algorithm SHA256
```

The value must match the table above. If it does not, the download is corrupt — delete
and fetch again.

## 4. Unpack

```bash
tar -xzf tujia-ctc-0.2.tar.gz
cd tujia-ctc-0.2
```

```bat
:: Windows 10+ ships tar as well
tar -xzf tujia-ctc-0.2.tar.gz
cd tujia-ctc-0.2
```

You should see:

```
model/tujia-ctc-v0.2.pt      head weights
model/vocab.txt              709 labels
model/model_card.json        architecture, metrics, license
infer.py                     inference script
fetch_encoder.sh             encoder downloader (POSIX shell)
requirements.txt
SHA256SUMS  MANIFEST.json  README.md  NOTICE  LICENSE
```

## 5. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
```

```bat
python -m venv .venv
.venv\Scripts\activate
```

## 6. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, install the CPU build explicitly. The default PyPI index may pull a 2.5 GB
CUDA package you do not need:

```bat
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers soundfile numpy huggingface_hub
```

### If `pip` cannot find torch

`ERROR: Could not find a version that satisfies the requirement torch>=2.0` means no
wheel matches your interpreter. Check the three usual causes in this order:

1. Python is 3.13+ or 3.9- — install 3.10–3.12.
2. Python is 32-bit — reinstall the 64-bit build.
3. `pip` is outdated — `python -m pip install --upgrade pip`.

## 7. Get the encoder (v0.2 only)

v0.2 cannot use the original `facebook/wav2vec2-xlsr-53-espeak-cv-ft` weights: the
encoder was fine-tuned, and mixing the two degrades recognition badly.

**Option A — download script (macOS / Linux)**

```bash
./fetch_encoder.sh ./models
```

This is a POSIX shell script. It does **not** run on Windows.

**Option B — manual download (any OS)**

Download the repository `TujiaNLP/tujia-xlsr53-ft` from Hugging Face or ModelScope and
place the 5 safetensors shards plus the config files in one folder:

```
models/tujia-xlsr53-ft/
├── config.json
├── preprocessor_config.json
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
└── model.safetensors.index.json
```

**Mainland China.** Hugging Face is usually unreachable. Two workarounds:

```bash
# ModelScope mirror
TUJIA_ENCODER_SOURCE=modelscope ./fetch_encoder.sh ./models

# Hugging Face through a mirror
HF_ENDPOINT=https://hf-mirror.com ./fetch_encoder.sh ./models
```

The folder name carries no meaning — `--encoder` and `TUJIA_XLSR` accept any path, as
long as it contains `config.json`.

v0.1 uses the original encoder instead. Either let `transformers` fetch it from Hugging
Face on first run, or download it once with `./fetch_encoder.sh` (that script points at
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` in the v0.1 archive).

## 8. Prepare the audio

`infer.py` reads audio with `soundfile`: wav, flac, ogg. Container formats such as mp4
or m4a must be demuxed first:

```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 audio.wav
```

16 kHz mono is ideal. Other sample rates are resampled on the fly, which requires
`torchaudio`. Keep each clip under ~60 seconds; for longer files decode a window at a
time with `--start` / `--duration`.

## 9. Run

macOS / Linux:

```bash
TUJIA_XLSR=./models/tujia-xlsr53-ft python infer.py audio.wav --device cpu
```

Windows (PowerShell):

```powershell
$env:TUJIA_XLSR = "models\tujia-xlsr53-ft"
python infer.py audio.wav --device cpu
```

Windows (CMD):

```bat
set TUJIA_XLSR=models\tujia-xlsr53-ft
python infer.py audio.wav --device cpu
```

Or pass the path explicitly, which avoids environment variables entirely:

```bash
python infer.py audio.wav --device cpu --encoder ./models/tujia-xlsr53-ft
```

Output, one syllable per line:

```
    4.42     4.58  kɤ53        0.718
    4.56     4.70  tsʰeŋ21     0.320
    4.70     4.86  xu21        0.479
```

Columns: start seconds, end seconds, toned syllable label, confidence.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--json out.json` | write structured output with timestamps |
| `--start 12.0 --duration 8.0` | decode one window of a long file |
| `--device mps` / `cuda` | use Apple GPU or an NVIDIA GPU |
| `--ckpt path.pt` | use a different head checkpoint |

## 10. Verify the package against its manifest

Every archive carries `SHA256SUMS` and `MANIFEST.json`:

```bash
shasum -a 256 -c SHA256SUMS
```

All nine lines must print `OK`. This checks the files inside the archive against the
hashes recorded at build time, independent of the outer `.tar.gz`.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Could not find a version that satisfies the requirement torch` | interpreter mismatch or 32-bit Python | see step 6 |
| `Fine-tuned encoder directory not found` | `TUJIA_XLSR` unset or pointing one level too high | point at the folder containing `config.json` |
| `metadata.get("format")` / safetensors error | an old copy of the encoder downloaded before 2026-10-28 | delete the folder and re-download |
| `soundfile` error on `.mp4` | unsupported container | convert to wav first |
| Everything downloads, then `KeyError` on load | `vocab.txt` does not match the checkpoint | do not mix v0.1 and v0.2 files |
| Very slow on CPU | expected | ~1× real time per clip on a laptop CPU; use `--start/--duration` |

## What the labels mean

v0.2 emits 708 toned syllables plus blank. The tone (`21 / 35 / 53 / 55`) is part of the
label, so `la35` is one unit, not `la` plus a tone. There is no word segmentation, no
translation and no punctuation — the model is a syllable recogniser.

v0.1 emits toneless syllable shapes only; tones must be restored downstream from a
lexicon.

## License boundaries

- Scripts and documentation: MIT.
- Model weights: research use only. No rights to the training recordings or transcripts
  are granted, and none of that material is distributed.
- The upstream encoder `facebook/wav2vec2-xlsr-53-espeak-cv-ft` is Apache-2.0 and must
  retain its attribution.

See `NOTICE` inside the archive for the full boundary.
