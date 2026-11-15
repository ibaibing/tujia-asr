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

### Assets

- GitHub tag `tujia-ctc-v0.3` marks this model's notes; the v0.3 branch of the code
  lives in the same repository.
- The checkpoint (`best.pt`, run `v03j`) is **not yet published** to GitHub Releases,
  Hugging Face or ModelScope — that is what the `tujia-asr` 1.0.0 Space still points at
  v0.2 for. Uploading it is pending.
