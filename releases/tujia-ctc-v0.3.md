## tujia-ctc v0.3 — toned 1199-class model

Syllable-level speech recognition for Tujia (Northern dialect) with tone marks. Same
output shape as v0.2 (`la35`, `tsʰi53`, `ɕiɛ35`), with the inventory widened to every
toned syllable that the training data supports.

- fine-tuned `v0.3f` encoder + LayerNorm + 2× BiLSTM(256) + Linear(512→1199); a 4-class
  tone auxiliary CTC head is used during training only (`lambda_tone = 0.4`)
- 1,199 labels: 1,198 toned syllables + blank; tones `21 / 35 / 53 / 55`
- 5,243 training clips, 141-clip validation split
- best epoch 8, validation SER 0.5126 (early-stopping split, not an independent test)

### Held-out evaluation

Both v0.2 and v0.3 scored on the same 353 clips — the only set that is in neither
v0.3's training nor its validation split (0.40 h, 8 speakers, 300 of them one male
teacher). v0.2's 708-label predictions are folded to the 0.6.3 inventory before
scoring; reference labels come from the 0.6.3 vocabulary for both runs.

| Metric | v0.2 | v0.3 |
| --- | --- | --- |
| SER | 0.8627 | **0.7330** |
| SER, no tone | 0.7034 | 0.6457 |
| Tone SER | 0.6228 | 0.5832 |
| Substitution rate | 0.5656 | 0.3057 |
| Deletion rate | 0.1941 | 0.3901 |
| Insertion rate | 0.1030 | 0.0372 |

The 856-clip test split used for v0.2 cannot be reused here: 851 of its clips are in
v0.3's training set.

v0.3 is 13 points better overall, but the gain comes from fewer substitutions rather
than fewer deletions — it emits fewer syllables than the reference, so faint or fast
speech can come out short.

### Fewer dropped syllables (`--blank-bias`)

`tujia-asr` 1.0.1 lowers the CTC blank logit on request. Re-scoring the same 353
clips with that knob (`TUJIA_BLANK_BIAS` or `--blank-bias`):

| Bias | SER | Sub | Del | Ins | Hyp/ref length |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.759 | 0.314 | 0.410 | 0.034 | 0.62 |
| 1.0 | 0.764 | 0.400 | 0.310 | 0.055 | 0.75 |
| 1.5 | 0.775 | 0.429 | 0.273 | 0.073 | 0.80 |

The sweep reads reference labels through the v0.3i vocabulary, which is why its
bias-0 row sits 2.6 points above the 0.7330 in the table above; the three rows are
directly comparable to each other. The published demo Space runs at 1.0 — a
visible wrong syllable is easier to correct than a gap, at a 0.6-point SER cost.

### Assets

- GitHub tag `tujia-ctc-v0.3` marks this model's notes; the v0.3 branch of the code
  lives in the same repository.
- The checkpoint (`best.pt`, run `v03j`) is published as
  `TujiaNLP/tujia-ctc-v0.3` on ModelScope and Hugging Face, next to the v0.3f
  encoder in `TujiaNLP/tujia-xlsr53-v03f`. GitHub Releases is still missing it
  (`gh` is not authenticated on this machine).
