# tujia-asr 1.1.0

Adds the v0.4 acoustic model and updates the default decoding density.

| | |
| --- | --- |
| Package | `tujia-asr` 1.1.0 |
| Model | `tujia-ctc-v0.4` (932 toneless syllables + 4-class tone head) |
| Demo | <https://modelscope.cn/studios/TujiaNLP/tujia-asr-demo> |

## What changed

- **v0.4 model support.** The loader already understood the two-head
  architecture; v0.4 checkpoints are detected from `arch`/`state_dict` and the
  tone head is read as `55 / 35 / 21 / 53`. No API change.
- **Default `blank_bias` is now 0.0.** v0.3 under-generated and needed 1.0 to
  keep syllables visible. v0.4 deletes only 0.6% of reference syllables and runs
  slightly long, so raising the blank bias would add noise. The Space and the
  library default are aligned at 0.0.
- **Docs and model table** point at `tujia-ctc-0.4.tar.gz`; v0.3/v0.2/v0.1 remain
  available and unchanged.
- **Encoder unchanged.** v0.4 uses `TujiaNLP/tujia-xlsr53-v03f`, the same
  fine-tuned encoder as v0.3.

## Numbers

On the 840-clip field-survey test set (split by lexical item; speakers are seen
in training):

| | Syllable SER | Tone SER | Joint toned SER |
| --- | --- | --- | --- |
| v0.3 | 0.7652 | 0.5616 | — |
| v0.4 | **0.2132** | **0.2054** | **0.3543** |

Old-data validation SER is 0.5571 (v0.3: 0.6490).

## Upgrade

```bash
pip install -U tujia-asr
export TUJIA_CTC_CKPT=/path/to/tujia-ctc-v0.4.pt
export TUJIA_XLSR=/path/to/tujia-xlsr53-v03f
```

Existing scripts keep working; pass `--blank-bias 1.0` only if you prefer more
visible (possibly spurious) syllables.
