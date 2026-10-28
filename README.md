# tujia-asr

Released acoustic models and the evaluation protocol for **syllable-level Tujia (Northern dialect) speech recognition**.

This repository is the public release hub. It distributes:

- model weights as versioned archives (GitHub Releases, mirrored on Hugging Face and ModelScope),
- the standalone inference script that ships with each archive,
- the evaluation protocol and data format used for every number quoted here.

Training code, training logs and the corpus itself are **not** published; the private `tujia-asr-lib` repository is used for training and architecture comparison. See [docs/protocol.md](docs/protocol.md) for what can and cannot be verified from the released artifacts.

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

Both models expect **16 kHz mono** audio and emit one syllable per line with a time span and a score. v0.2 additionally attaches the tone to the label, e.g. `la35`.

## Quick start

```bash
# 1. download and unpack a release archive
tar xzf tujia-ctc-0.2.tar.gz && cd tujia-ctc-0.2

# 2. dependencies
pip install -r requirements.txt

# 3. encoder (~1.2 GB, downloaded once)
./fetch_encoder.sh ./models

# 4. decode
TUJIA_XLSR=./models/tujia-xlsr53-ft python infer.py your_audio.wav
```

Output:

```
    4.42     4.58  kɤ53        0.718
    4.56     4.70  tsʰeŋ21     0.320
```

`--json out.json` writes structured output, `--start` / `--duration` decode a single span. Keep each input under ~60 s; split longer audio with `--start` / `--duration`.

The v0.2 encoder **cannot** be swapped for the original Hugging Face XLSR-53 weights: the features differ and recognition degrades substantially. It is fetched by `fetch_encoder.sh`.

## Where the weights live

| Channel | Contents |
| --- | --- |
| GitHub Releases (this repo) | `tujia-ctc-0.1.tar.gz`, `tujia-ctc-0.2.tar.gz` (~17 MB each), `SHA256SUMS` |
| Hugging Face | `xbaibing/tujia-xlsr53-ft` (the 1.2 GB fine-tuned encoder), `xbaibing/tujia-ctc-0.1`, `xbaibing/tujia-ctc-0.2` |
| ModelScope | `bbingx/tujia-xlsr53-ft`, `bbingx/tujia-ctc-0.1`, `bbingx/tujia-ctc-0.2` (mainland mirror) |

Large weights are hosted on the model hubs, not in GitHub Releases. The 1.2 GB encoder is
only needed for v0.2 and is downloaded by `./fetch_encoder.sh`; set
`TUJIA_ENCODER_SOURCE=modelscope` to fetch it from the mainland mirror, or
`HF_ENDPOINT=https://hf-mirror.com` to reach Hugging Face through a mirror.

Each archive contains the checkpoint, `vocab.txt`, `model_card.json`, `infer.py`, `fetch_encoder.sh`, `LICENSE`, `NOTICE`, `SHA256SUMS` and `MANIFEST.json`. Verify a download with `shasum -a 256 -c SHA256SUMS`.

## Evaluation

Two protocols exist and they are **not interchangeable**. Quoting a number without naming its protocol is ambiguous:

- **SER** — strict syllable error rate: token sequences aligned with Needleman–Wunsch, edit distance divided by reference length. Used for v0.2.
- **Recall** — lenient syllable recall: syllable shapes normalized, aligned with NW, a reference syllable counts as a hit at edit similarity ≥ 0.5. Used for v0.1 and for cross-version comparisons.

Definitions, normalization rules and the exact splits are in [docs/protocol.md](docs/protocol.md). The manifest schema is in [docs/data-format.md](docs/data-format.md).

Headline numbers, all on the same 685-clip corpus (577 train / 108 held out):

| Checkpoint | Recall, all 685 | Recall, held-out 108 | Precision, held-out 108 |
| --- | --- | --- | --- |
| v0.1 | 89.82% | 80.26% | 84.33% |
| v0.2 | 89.90% | 83.93% | 84.37% |

## Data summary

Only aggregate properties of the training data are disclosed. The recordings, their
transcripts, the speakers and the collection they come from are not published, and the
annotation source is not named. Nothing in this repository or in the release archives
identifies them.

| Property | Value |
| --- | --- |
| Audio | 685 clips, 16 kHz mono, 2.571 hours |
| Annotation unit | one toned IPA syllable per segment, with boundaries |
| Variety | Northern Tujia; one collection, no dialect mixing |
| Speakers | a small number of speakers, identities withheld |
| Label inventory | 264 toneless syllables (v0.1); 708 toned syllables over the tones `21 / 35 / 53 / 55` (v0.2) |
| Reference syllables | 20,568 in total (17,295 train / 3,273 held out) |
| Split | 577 train / 108 held out, grouped by recording session |
| Recording conditions | close-mic read speech of mixed quality across sessions |

The label inventory is an IPA syllable list, not a lexical resource: the model emits
syllables, and no word list, gloss or dictionary entry is included in either archive.
The mapping from the collection's annotation to this inventory is part of the private
training pipeline and is not distributed.

## Known limitations

1. **One source, one variety.** About 2.6 hours from a single third-party collection, few speakers, Northern dialect. Other dialect points are likely to degrade; fall back to a phone-level front-end plus lexicon.
2. **Over-segmentation.** Predicted syllable counts exceed reference counts on slow or noisy speech; deletions are comparatively rare.
3. **Tone is the residual error for v0.2.** Roughly one tone in three is wrong on the test split, spread over `21 ↔ 35 ↔ 55`.
4. **Narrowband audio degrades sharply.** On a field recording with 85% of its energy below 390 Hz, v0.1 recall drops to 48.85%.
5. **Small splits.** Dev/test is 43/65 clips; treat differences under ~1 point as noise.

## Alternatives considered

A fully fine-tuned `openai/whisper-small` was also trained on the same 577 clips and is
**not** released. It memorises the training set within a few epochs (training loss well
below 0.01 after 15 epochs while validation loss rises from epoch 3 onward), it emits
text rather than frame-level labels, so syllable boundaries would need a separate
alignment pass, and it is roughly 50× the parameter count of the released head with much
slower decoding. Its held-out score sits within noise of v0.2, which does not justify
shipping a second, harder-to-run pipeline. No numbers from that run are comparable with
the tables above.

## Data and license

- Code in this repository and in every archive: MIT (see [LICENSE](LICENSE)).
- Model weights and the fine-tuned encoder: **research use only**. No rights to the training recordings or transcripts are granted, and none of that material is distributed.
- Upstream encoder `facebook/wav2vec2-xlsr-53-espeak-cv-ft` is Apache-2.0 and must retain its attribution.
- The parent project [TujiaNLP](https://github.com/ibaibing/TujiaNLP) maintains the field collection, annotation standards and orthography conversion; corpus release and its terms live there, not in this repository.
- See [NOTICE](NOTICE) for the full boundary.

## Citation

See [CITATION.cff](CITATION.cff). If you use a specific version, cite that version number.
