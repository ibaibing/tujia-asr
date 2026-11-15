# tujia-asr 1.0.0

First release that installs as a package. `pip install tujia-asr` gives you the CLI and a
Python API; the acoustic weights still download separately.

| | |
| --- | --- |
| Package | `tujia-asr` 1.0.0 |
| Model | `tujia-ctc-v0.3` (training run `v03j`) |
| Runtime deps | `tujia-nlp>=0.4.0`, `torch`, `transformers` |

## What changed since the model-only releases

v0.1 and v0.2 shipped weights with no code entry point. This release adds the two halves
around the acoustic model, so a recording goes in and words come out.

### Pre-processing

- Any input format ffmpeg can decode, resampled to 16 kHz mono.
- Long audio is split into windows on silence (10 s target) and stitched back on a single
  timeline, so hour-long recordings no longer need manual cutting.

### Post-processing

- **Lexicon decoding.** The northern open core vocabulary (1,149 forms) ships inside
  `tujia-nlp` 0.4.0 and is used by default. `--dict` appends user lexicons; earlier files
  win on duplicate headwords, so a user entry overrides the built-in one.
- **Words, not just syllables.** A Viterbi pass over the syllable lattice produces words
  with glosses, POS, the source file, and a tier (A/B/C/D) that records how much
  independent evidence supports each one.
- **Nothing is dropped.** Syllables the lexicon cannot explain are emitted as-is with
  `in_lexicon: false` instead of being discarded, and they act as barriers so no word can
  span across them.
- **Review list.** The JSON carries `review`: tier-C words, unexplained audio spans, and
  tone conflicts, each with time offsets.
- `--format ipa | ye | ascii` applies to the per-syllable lines and the summary line alike.

### Model

Ships `tujia-ctc` v0.3 (training run `v03j`): 1,199 toned syllables, held-out SER 0.7330
on a 353-clip leak-free split, against 0.8627 for v0.2 on the same clips. The model has
its own notes — [tujia-ctc-v0.3.md](tujia-ctc-v0.3.md) — because the checkpoint and the
package now version independently.

## Migration

There is nothing to migrate from — v0.1/v0.2 had no runnable interface. If you were
driving the weights through a private script, the equivalent is now:

```bash
pip install tujia-asr
tujia-asr recording.wav --json out.json
```

## Known limitations

- v0.3 under-emits: deletion rate 0.39 against v0.2's 0.19 on the held-out set. Quiet
  or fast passages lose syllables rather than produce wrong ones; the `review` list
  marks the audio spans that nothing was decoded for.
- The lexicon is a post-processor, not a corrector. A misheard syllable is mapped to the
  nearest dictionary form; recognition itself is unchanged (tone confidence 0.753 with
  the lexicon vs 0.759 without, on the p11 page).
- About 56% of the open core vocabulary is monosyllabic, so on teaching recordings most
  matches stay single syllables. Multi-syllable segmentation only pays off with a denser
  vocabulary.
- A few syllables (`nɤ⁵⁵`, `ɕi⁵⁵`) have no Ye spelling and fall back to IPA in `--format ye`.
- The ZIPA frontend and the second-pass `qwen3` channel need assets or an API key that the
  package does not bundle.
