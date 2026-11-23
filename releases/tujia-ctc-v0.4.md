## tujia-ctc v0.4 — 932-syllable head + separate tone head

Syllable-level speech recognition for Tujia with tone marks. v0.4 replaces the
single 1,199-class toned head with a 932-class **toneless** syllable head plus a
4-class tone head, and adds field-survey recordings to training.

- fine-tuned `v0.3f` encoder + LayerNorm + 2× BiLSTM(256); 932-class main head
  and a 5-way tone head (`none / 55 / 35 / 21 / 53`)
- tone is assigned at each syllable's main-head peak frame
- 17,134 training rows / 16.2 h: 3,635 earlier-corpus clips used twice, plus
  9,864 field-survey clips
- best epoch 34; encoder unchanged from v0.3

### Evaluation

The v0.4 test set is 840 field-survey clips (0.25 h), split by lexical
item. v0.3 is scored on the same clips for reference.

| Checkpoint | Syllable SER | Tone SER | Joint toned SER |
| --- | --- | --- | --- |
| v0.3 | 0.7652 | 0.5616 | — |
| v0.4 | **0.2132** | **0.2054** | **0.3543** |

By area: north 0.2086 syllable / 0.2019 tone; south 0.2280 / 0.2165. Old-data
validation SER: 0.5571 (v0.3: 0.6490).

Error composition on the test set: substitution 19.2%, deletion 0.6%,
insertion 1.5%. Unlike v0.3, v0.4 no longer under-generates: its output length is
about 1.20× the reference, so the failure mode moved from missing syllables to
substitutions. `--blank-bias` is therefore not needed by default.

**Caveat.** The split is by lexical item; every test speaker also appears in
training. These are seen-speaker numbers, not a speaker-held-out score.

### Assets

- `tujia-ctc-0.4.tar.gz` — 17 MB, checkpoint + vocab + standalone `infer.py` +
  encoder fetch script + model card
- Encoder: `TujiaNLP/tujia-xlsr53-v03f` (same as v0.3, ~1.26 GB)
- Checkpoint repo: `TujiaNLP/tujia-ctc-v0.4` on Hugging Face and ModelScope
- GitHub tag: `tujia-ctc-v0.4`
